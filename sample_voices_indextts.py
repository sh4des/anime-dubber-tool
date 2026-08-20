#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# sample_voices_indextts.py  (Phase A.5: listen-before-commit voice QC)
# -----------------------------------------------------------------------------
# After Phase A builds the cast (anime-dub-profile.json) and BEFORE the multi-hour
# Phase B render + remux, audition every character's cloned voice so the two
# recurring failure modes can be caught early:
#
#   * SILENCED voices - a speaker whose reference clip(s) are all missing on disk
#     renders NOTHING in the full run (srt_to_speech_indextts.py timbre_for() ->
#     None -> every cue for that speaker is skipped). This is detected STATICALLY
#     here (no render needed) and flagged NO_TIMBRE. A clone that renders but comes
#     out abnormally quiet is flagged LOW_LEVEL.
#   * ECHOED voices - reverb/room bleed in the speaker's source material. Each
#     sample is rendered through the SAME timbre + emotion + emo_alpha path the
#     full engine uses, so an echoey clone sounds echoey here too.
#
# For each speaker we render ONE fixed English line (same text for all, so voices
# are directly comparable), using:
#   * timbre    = the speaker's first reference clip that exists on disk
#                 (identical rule to the engine's timbre_for()),
#   * emotion   = a DIFFERENT existing ref clip of the same speaker when available
#                 (else the same clip, or none), driving delivery like emo_audio_prompt,
#   * emo_alpha / temperature / seed / max_text_tokens = the SAME voice-tuning knobs.
# The reference clips are Demucs-vocal-derived, i.e. the clean-stem path the full
# run uses with -EmotionStemsDir, so the audition matches production output.
#
# Outputs (to --out-dir):
#   spkNN_<bucket>.wav        one audition per speaker (peak-normalized for fair A/B)
#   _contact_sheet.wav        all auditions back-to-back with gaps (scrub the cast)
#   voice_samples_report.txt  human-readable table, PROBLEMS FIRST + contact-sheet timeline
#   voice_samples.csv         machine-readable manifest
#
# Runs under the IndexTTS2 venv (same as srt_to_speech_indextts.py). Needs only
# indextts + numpy + soundfile (NOT the demucs/speechbrain profiler stack).
#
# Exit codes: 0 ok; 3 a dependency is missing; 4 bad profile / checkpoints.
# -----------------------------------------------------------------------------
import argparse
import csv
import json
import math
import os
import sys
import time

# Reuse the engine's profile loader (resolves each speaker's _refs_abs) so the
# timbre resolution here is byte-identical to the full render.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srt_to_speech_indextts import load_profile  # noqa: E402

OUT_SR = 24000

# A line with statements, a dash-break and a question so the audition exercises
# varied prosody (flat delivery hides both dropouts and echo).
DEFAULT_TEXT = ("I never asked for any of this. But if we don't move now, "
                "we lose everyone - do you understand me?")

LOW_LEVEL_DBFS = -34.0   # intrinsic clone RMS below this -> flag LOW_LEVEL
HOT_DBFS       = -12.0    # intrinsic clone RMS above this -> flag HOT (may distort)
CLIP_FRAC_THRESH = 0.001 # >=0.1% of samples flat-topped at full scale -> real CLIPPED
NORM_PEAK      = 0.89     # per-file + contact-sheet peak (~-1 dBFS) for fair A/B


def eprint(*a):
    print(*a, file=sys.stderr)


def dbfs(rms):
    return 20.0 * math.log10(rms + 1e-12)


def existing_refs(sp):
    return [r for r in sp.get("_refs_abs", []) if os.path.isfile(r)]


def main():
    ap = argparse.ArgumentParser(description="Audition every profiled voice before Phase B.")
    ap.add_argument("--profile", required=True, help="anime-dub-profile.json from Phase A")
    ap.add_argument("--checkpoints-dir", required=True, help="IndexTTS2 checkpoints (config.yaml + weights)")
    ap.add_argument("--out-dir", required=True, help="where the sample WAVs + report go")
    ap.add_argument("--voice-tuning", default=None, help="same voice_tuning.json Phase B uses")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="line every speaker says")
    ap.add_argument("--emo-alpha", type=float, default=0.45, help="default emotion strength (per-speaker tuning overrides)")
    ap.add_argument("--no-emotion", action="store_true", help="timbre-only clone, no emotion reference")
    ap.add_argument("--speakers", default=None, help="comma-separated speaker ids to limit to (default: all)")
    ap.add_argument("--list", action="store_true", help="only run the static NO_TIMBRE pre-check; render nothing")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    try:
        import numpy as np
        import soundfile as sf
    except ModuleNotFoundError as e:
        eprint(f"[sample] FATAL: missing dependency ({e}). uv pip install soundfile")
        return 3

    cfg = os.path.join(args.checkpoints_dir, "config.yaml")
    if not os.path.isfile(cfg):
        eprint(f"[sample] FATAL: no config.yaml in {args.checkpoints_dir}")
        return 4

    _, speakers = load_profile(args.profile)
    if not speakers:
        eprint("[sample] FATAL: profile has no speakers.")
        return 4

    if args.speakers:
        want = {int(x) for x in args.speakers.split(",") if x.strip()}
        speakers = [s for s in speakers if s["id"] in want]

    # --- voice tuning (same parsing as the engine) --------------------------
    tuning = {}
    if args.voice_tuning and os.path.isfile(args.voice_tuning):
        with open(args.voice_tuning, encoding="utf-8") as f:
            tuning = json.load(f)
        eprint(f"[sample] loaded voice tuning from {args.voice_tuning}")
    g = tuning.get("global", {})
    G_EMO    = float(g.get("emo_alpha", args.emo_alpha))
    G_TEMP   = float(g.get("temperature", 0.8))
    G_MAXSEG = int(g.get("max_text_tokens_per_segment", 120))
    G_SEED   = int(g.get("seed", 0))
    spk_tune = {int(k): v for k, v in (tuning.get("speakers", {}) or {}).items()}
    def emo_for(sid):
        return float(spk_tune.get(sid, {}).get("emo_alpha", G_EMO))

    _torch = None
    if G_SEED:
        try:
            import torch
            _torch = torch
        except Exception:
            _torch = None

    os.makedirs(args.out_dir, exist_ok=True)

    # --- static pre-check: which speakers would be SILENT in the full run ----
    rows = []
    for sp in speakers:
        refs = existing_refs(sp)
        rows.append({
            # `or "?"` not .get(default): the profile can carry an explicit null
            # gender/bucket (pitch-based detection fails for some speakers), and a
            # null reaches the report's "{...:<8}" width format as None ->
            # TypeError, killing the report AFTER every voice has been rendered.
            "id": sp["id"], "bucket": sp.get("bucket") or "?", "gender": sp.get("gender") or "?",
            "total_speech_sec": float(sp.get("total_speech_sec") or 0.0),
            "n_refs_profile": len(sp.get("references", [])), "n_refs_on_disk": len(refs),
            "emo_alpha": round(emo_for(sp["id"]), 3),
            "timbre": refs[0] if refs else "", "has_timbre": bool(refs),
            "raw_rms_dbfs": None, "raw_peak": None, "out_file": "", "flags": "",
        })
    missing = [r for r in rows if not r["has_timbre"]]
    if missing:
        eprint(f"[sample] STATIC: {len(missing)} speaker(s) have NO reference clip on "
               f"disk and WILL RENDER SILENT in Phase B: "
               f"{', '.join('spk%02d' % r['id'] for r in missing)}")

    if args.list:
        _write_report(args, rows, [], sr=OUT_SR)
        return 0

    # --- load IndexTTS2 once -------------------------------------------------
    try:
        from indextts.infer_v2 import IndexTTS2
    except ModuleNotFoundError as e:
        eprint(f"[sample] FATAL: cannot import IndexTTS2 ({e}). Run under the IndexTTS2 venv.")
        return 3
    eprint(f"[sample] loading IndexTTS2 from {args.checkpoints_dir} (fp16={args.fp16}) ...")
    tts = IndexTTS2(cfg_path=cfg, model_dir=args.checkpoints_dir,
                    use_fp16=args.fp16, use_deepspeed=False, use_cuda_kernel=False)

    import soundfile as sf
    import numpy as np

    contact = []          # (id, bucket, audio) for the contact sheet
    gap = np.zeros(int(0.5 * OUT_SR), dtype=np.float32)
    t0 = time.time()
    order = sorted(rows, key=lambda r: -r["total_speech_sec"])
    for k, r in enumerate(order):
        sid = r["id"]
        if not r["has_timbre"]:
            r["flags"] = "NO_TIMBRE(silent)"
            eprint(f"[sample] spk{sid:02d} {r['bucket']}: NO_TIMBRE - skipping render")
            continue
        refs = existing_refs(next(s for s in speakers if s["id"] == sid))
        timbre = refs[0]
        emo = None if args.no_emotion else (refs[1] if len(refs) > 1 else refs[0])
        ce = emo_for(sid)
        if args.no_emotion:
            ce = 0.0
        out = os.path.join(args.out_dir, f"spk{sid:02d}_{r['bucket']}.wav")
        if _torch is not None:
            _torch.manual_seed(G_SEED + sid)
        try:
            tts.infer(spk_audio_prompt=timbre, text=args.text, output_path=out,
                      emo_audio_prompt=emo, emo_alpha=ce, verbose=False,
                      max_text_tokens_per_segment=G_MAXSEG, temperature=G_TEMP)
        except Exception as e:
            r["flags"] = f"RENDER_FAILED:{e}"
            eprint(f"[sample] spk{sid:02d}: render failed: {e}")
            continue
        y, _ = sf.read(out, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        raw_rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
        raw_peak = float(np.max(np.abs(y))) if y.size else 0.0
        # fraction of samples pinned at ~full scale. IndexTTS2 routinely *touches*
        # 1.0 without distorting (and Phase B caps peaks at target_peak anyway), so a
        # bare peak>=1.0 is NOT clipping - only sustained flat-topping is. This
        # measures real distortion.
        clip_frac = float(np.mean(np.abs(y) >= 0.9995)) if y.size else 0.0
        r["raw_rms_dbfs"] = round(dbfs(raw_rms), 1)
        r["raw_peak"] = round(raw_peak, 3)
        r["clip_pct"] = round(clip_frac * 100, 3)
        flags = []
        if raw_rms <= 1e-5:
            flags.append("SILENT_CLONE")
        elif dbfs(raw_rms) < LOW_LEVEL_DBFS:
            flags.append("LOW_LEVEL")
        if clip_frac >= CLIP_FRAC_THRESH:
            flags.append("CLIPPED")          # genuinely flat-topped -> audible distortion
        elif dbfs(raw_rms) >= HOT_DBFS:
            flags.append("HOT")              # very hot delivery; may sound harsh, lower emo_alpha
        r["flags"] = ",".join(flags) if flags else "ok"
        # peak-normalize for fair A/B listening (raw stats retained above)
        yn = y * (NORM_PEAK / raw_peak) if raw_peak > 1e-6 else y
        sf.write(out, yn, OUT_SR)
        r["out_file"] = os.path.basename(out)
        contact.append((sid, r["bucket"], yn))
        el = time.time() - t0
        eprint(f"[sample] {k+1}/{len(order)} spk{sid:02d} {r['bucket']:<14} "
               f"rms={r['raw_rms_dbfs']}dBFS peak={r['raw_peak']} [{r['flags']}] "
               f"({el/60:.1f}m)")

    # --- contact sheet -------------------------------------------------------
    sheet_timeline = []
    if contact:
        parts, t = [], 0.0
        for sid, bucket, y in contact:
            sheet_timeline.append((t, sid, bucket))
            parts.append(y)
            parts.append(gap)
            t += (len(y) + len(gap)) / OUT_SR
        sheet = np.concatenate(parts).astype(np.float32)
        pk = float(np.max(np.abs(sheet))) or 1.0
        if pk > NORM_PEAK:
            sheet *= (NORM_PEAK / pk)
        sheet_path = os.path.join(args.out_dir, "_contact_sheet.wav")
        sf.write(sheet_path, sheet, OUT_SR)
        eprint(f"[sample] wrote contact sheet {sheet_path} ({t/60:.1f} min, {len(contact)} voices)")

    _write_report(args, rows, sheet_timeline, sr=OUT_SR)
    return 0


def _write_report(args, rows, sheet_timeline, sr):
    # CSV manifest
    csv_path = os.path.join(args.out_dir, "voice_samples.csv")
    cols = ["id", "bucket", "gender", "total_speech_sec", "n_refs_profile",
            "n_refs_on_disk", "has_timbre", "emo_alpha", "raw_rms_dbfs",
            "raw_peak", "clip_pct", "out_file", "flags", "timbre"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # human report: problems first
    def rank(r):
        f = r.get("flags", "") or ""
        sev = 0 if "NO_TIMBRE" in f else 1 if ("SILENT" in f or "LOW_LEVEL" in f or "FAILED" in f) \
            else 2 if "CLIPPED" in f else 3 if "HOT" in f else 4
        return (sev, -r["total_speech_sec"])
    ordered = sorted(rows, key=rank)
    txt_path = os.path.join(args.out_dir, "voice_samples_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Voice audition report (listen BEFORE Phase B)\n")
        f.write(f"profile : {os.path.abspath(args.profile)}\n")
        f.write(f"samples : {os.path.abspath(args.out_dir)}\n")
        f.write(f"line    : {args.text!r}\n\n")
        probs = [r for r in ordered if (r.get("flags") or "") not in ("ok", "")]
        if probs:
            f.write(f"** {len(probs)} SPEAKER(S) NEED ATTENTION **\n")
            for r in probs:
                f.write(f"  spk{r['id']:02d} {r['bucket']:<14} {r['flags']}"
                        f"  (refs on disk {r['n_refs_on_disk']}/{r['n_refs_profile']}, "
                        f"rms {r['raw_rms_dbfs']} dBFS)\n")
            f.write("\n  NO_TIMBRE  -> would render SILENT in Phase B; restore/repoint this "
                    "speaker's reference clip(s) or map cues to another speaker.\n")
            f.write("  LOW_LEVEL/SILENT_CLONE -> clone came out too quiet; check the ref clip, "
                    "or raise gain_db for this speaker in voice_tuning.\n")
            f.write("  CLIPPED    -> genuinely flat-topped (clip_pct>=0.1%); lower emo_alpha/gain_db.\n")
            f.write("  HOT        -> very hot delivery (RMS>=-12 dBFS) but not flat-topped; may sound "
                    "harsh. Phase B caps peaks at target_peak, but consider emo_alpha 0.30-0.35.\n\n")
        else:
            f.write("No problems detected.\n\n")
        f.write("Full cast (id / bucket / gender / speech / emo_alpha / rms dBFS / peak / flags):\n")
        for r in sorted(rows, key=lambda x: -x["total_speech_sec"]):
            f.write(f"  spk{r['id']:02d} {r['bucket']:<14} {r['gender']:<8} "
                    f"{r['total_speech_sec']:7.0f}s  a={r['emo_alpha']:<5} "
                    f"rms={str(r['raw_rms_dbfs']):>6} pk={str(r['raw_peak']):>5} "
                    f"clip={str(r.get('clip_pct','')):>6}%  {r.get('flags','')}\n")
        if sheet_timeline:
            f.write("\n_contact_sheet.wav timeline (mm:ss -> speaker):\n")
            for t, sid, bucket in sheet_timeline:
                f.write(f"  {int(t//60):d}:{int(t%60):02d}  spk{sid:02d} {bucket}\n")
    eprint(f"[sample] report -> {txt_path}")
    eprint(f"[sample] manifest -> {csv_path}")


if __name__ == "__main__":
    sys.exit(main())
