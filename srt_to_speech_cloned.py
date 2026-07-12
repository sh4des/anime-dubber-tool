#!/usr/bin/env python3
# srt_to_speech_cloned.py
# -----------------------------------------------------------------------------
# PHASE B of the "cloned + performance" dub pipeline.
#
# Same job as srt_to_speech_multivoice.py (SRT -> one timed English dub WAV), but
# instead of assigning each character a voice from a fixed pool, it CLONES the
# character's actual (Japanese) voice using the show profile built by
# profile_show.py (Phase A). The English line inherits the source timbre.
#
# Cue -> speaker matching (two paths)
# -----------------------------------
#   * If this episode was profiled (its name has stored diarization "turns" in
#     the profile), each cue is matched to the speaker whose turn overlaps it
#     most. Fast, accurate, needs no models.
#   * Otherwise each cue's ORIGINAL audio is Demucs-cleaned, embedded with the
#     same encoder the profile used (ecapa / resemblyzer), and matched to the
#     nearest speaker centroid by cosine similarity. Works on any episode.
#
# Synthesis + quality gate
# ------------------------
#   Each matched speaker is voiced by CLONING its reference clips (XTTS
#   conditioning latents computed once per speaker and cached). If a cue does not
#   confidently match any speaker, or the matched speaker has no usable reference
#   clips, the line falls back to that speaker's pool voice (or a global default)
#   - so a weak match never ships a broken clone.
#
# HONEST STATUS
#   * This is cloning + duration-fit. Full prosody/performance transfer (pitch &
#     energy contour) and VC polish (seed-vc) are planned follow-ups, not here.
#   * Per-cue embedding matching on short/noisy cues can misfire; the confidence
#     gate + pool fallback bound the damage. Prefer profiling every episode you
#     dub (so the reliable time-overlap path is used).
#   * XTTS cross-lingual cloning can carry a slight Japanese accent; VC polish
#     (later) mitigates it.
#   * UNTESTED end-to-end on GPU as written - iterate on a test episode first.
# -----------------------------------------------------------------------------

import argparse
import json
import os
import sys
import tempfile

os.environ.setdefault("COQUI_TOS_AGREED", "1")

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Reuse Phase A audio helpers and the multi-voice text helpers (all cheap imports
# whose heavy deps live inside functions).
try:
    from profile_show import (demucs_vocals, load_audio, TARGET_SR,
                              make_ecapa_embed, make_resemblyzer_embed)
    from srt_to_speech_multivoice import (clean_text, split_long, cosine,
                                          make_encoder, CHILD_BUCKETS,
                                          ELDERLY_BUCKETS, pitch_shift_semitones)
except Exception as e:   # pragma: no cover
    print(f"[clone] FATAL: cannot import sibling helpers ({e})", file=sys.stderr)
    sys.exit(3)


def load_profile(path):
    """Load the Phase A profile; resolve reference clip paths to absolute."""
    import numpy as np
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    speakers = []
    for s in data.get("speakers", []):
        refs = [os.path.normpath(os.path.join(base, r)) for r in s.get("references", [])]
        refs = [r for r in refs if os.path.exists(r)]
        speakers.append({
            "id": s["id"],
            "bucket": s.get("bucket"),
            "gender": s.get("gender"),
            "voice": s.get("fallback_voice"),
            "centroid": np.asarray(s["centroid"], dtype=np.float64) if s.get("centroid") else None,
            "refs": refs,
            "turns": s.get("turns", {}),
        })
    return data, speakers


def backend_from_profile(data):
    """Which encoder produced the centroids (for the embedding match path)."""
    d = (data.get("diarizer") or "").lower()
    if "ecapa" in d:
        return "ecapa"
    if "resemblyzer" in d:
        return "resemblyzer"
    if "pyannote" in d:
        return "pyannote"
    return "resemblyzer"


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def match_by_turns(cues, speakers, episode_name):
    """Assign each cue to the speaker whose turns overlap it most. Returns True if
    this episode has any turns (i.e. the time-overlap path is usable)."""
    # speaker_id -> list of (start, end)
    turns = {}
    for sp in speakers:
        t = sp["turns"].get(episode_name)
        if t:
            turns[sp["id"]] = [(x["start"], x["end"]) for x in t]
    if not turns:
        return False
    by_id = {sp["id"]: sp for sp in speakers}
    for c in cues:
        best_id, best_ov = None, 0.0
        for sid, segs in turns.items():
            ov = sum(overlap(c["start"], c["end"], s0, s1) for s0, s1 in segs)
            if ov > best_ov:
                best_ov, best_id = ov, sid
        c["speaker"] = by_id.get(best_id) if best_id is not None else None
    return True


def match_by_embedding(cues, speakers, ref_clean, sr, embed_one, min_sim):
    """Assign each cue by embedding its cleaned audio vs speaker centroids."""
    cand = [sp for sp in speakers if sp["centroid"] is not None]
    for c in cues:
        a = int(c["start"] * sr)
        b = min(int(c["end"] * sr), ref_clean.size)
        c["speaker"] = None
        if b - a < int(0.4 * sr):
            continue
        v = embed_one(ref_clean[a:b])
        if v is None:
            continue
        best, best_sim = None, -1.0
        for sp in cand:
            sim = cosine(v, sp["centroid"])
            if sim > best_sim:
                best_sim, best = sim, sp
        if best is not None and best_sim >= min_sim:
            c["speaker"] = best


def main() -> int:
    ap = argparse.ArgumentParser(description="SRT -> cloned-voice English dub WAV")
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", required=True, help="Phase A profile JSON")
    ap.add_argument("--ref-audio", required=True,
                    help="mono WAV of the ORIGINAL speech track, aligned to subs")
    ap.add_argument("--episode-name", default=None,
                    help="profiled episode name (enables the time-overlap match)")
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--language", default="en")
    ap.add_argument("--match-min-sim", type=float, default=None,
                    help="min cosine similarity to accept an embedding match "
                         "(per-backend default)")
    ap.add_argument("--no-demucs-ref", action="store_true",
                    help="don't Demucs-clean the ref before embedding (faster, "
                         "worse matching); only relevant to the embedding path")
    ap.add_argument("--child-pitch-shift", type=float, default=0.0,
                    help="semitones to raise POOL-fallback child voices (cloned "
                         "lines are never shifted)")
    ap.add_argument("--elder-pitch-shift", type=float, default=0.0)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--max-speed", type=float, default=1.6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        import numpy as np
        import srt as srtlib
        from pydub import AudioSegment
        from pydub.effects import speedup
        import torch
        import torchaudio
        from TTS.api import TTS
    except ModuleNotFoundError as e:
        print(f"[clone] missing dependency: {e.name}", file=sys.stderr)
        return 3

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clone] device={device}")

    data, speakers = load_profile(args.profile)
    if not speakers:
        print("[clone] profile has no speakers.", file=sys.stderr)
        return 4
    backend = backend_from_profile(data)
    print(f"[clone] profile: {len(speakers)} speaker(s), encoder={backend}, "
          f"{sum(1 for s in speakers if s['refs'])} with reference clips")

    # --- subtitles ------------------------------------------------------------
    with open(args.srt, encoding="utf-8-sig") as f:
        raw = list(srtlib.parse(f.read()))
    cues, dropped = [], 0
    for s in raw:
        text = clean_text(s.content)
        if not text:
            continue
        start, end = s.start.total_seconds(), s.end.total_seconds()
        if any(c["text"] == text and start < c["end"] + 0.5 for c in cues[-8:]):
            dropped += 1
            continue
        cues.append({"text": text, "start": start, "end": end, "speaker": None})
    if not cues:
        print("[clone] no subtitle cues.", file=sys.stderr)
        return 2
    if dropped:
        print(f"[clone] dropped {dropped} duplicate cue(s)")

    # --- cue -> speaker -------------------------------------------------------
    used_turns = False
    if args.episode_name:
        used_turns = match_by_turns(cues, speakers, args.episode_name)
    if used_turns:
        print(f"[clone] matched cues by TIME-OVERLAP (episode is profiled)")
    else:
        # embedding path: load ref, optionally Demucs-clean, embed each cue
        print("[clone] episode not in profile turns -> matching by EMBEDDING")
        if backend == "pyannote":
            print("[clone] FATAL: profile used pyannote embeddings, which this "
                  "script can't recompute. Re-profile with --diarizer ecapa, or "
                  "dub only profiled episodes (with --episode-name).",
                  file=sys.stderr)
            return 4
        min_sim = args.match_min_sim
        if min_sim is None:
            min_sim = 0.45 if backend == "ecapa" else 0.70
        src = args.ref_audio
        if not args.no_demucs_ref:
            cleaned = os.path.join(tempfile.gettempdir(), "clone_ref_vocals.wav")
            got = demucs_vocals(args.ref_audio, cleaned, device, args.verbose)
            if got:
                src = got
        ref_clean = load_audio(src, TARGET_SR)
        if backend == "ecapa":
            embed_one = make_ecapa_embed(device, TARGET_SR)
        else:
            embed_one = make_resemblyzer_embed(make_encoder(device), TARGET_SR)
        if embed_one is None:
            print(f"[clone] FATAL: could not load the {backend} encoder.",
                  file=sys.stderr)
            return 3
        match_by_embedding(cues, speakers, ref_clean, TARGET_SR, embed_one, min_sim)

    matched = sum(1 for c in cues if c["speaker"] is not None)
    print(f"[clone] {matched}/{len(cues)} cues matched to a profiled speaker "
          f"({len(cues) - matched} -> default voice)")

    # --- XTTS + per-speaker conditioning --------------------------------------
    print(f"[clone] loading {XTTS_MODEL} ...")
    tts = TTS(XTTS_MODEL).to(device)
    model = tts.synthesizer.tts_model
    default_voice = next((s["voice"] for s in speakers if s["voice"]), "Damien Black")

    cond_cache = {}   # speaker id -> (gpt_cond, spk_emb) or None (=use pool)

    def get_cond(sp):
        if sp["id"] in cond_cache:
            return cond_cache[sp["id"]]
        val = None
        if sp["refs"]:
            try:
                gpt_cond, spk_emb = model.get_conditioning_latents(audio_path=sp["refs"])
                val = (gpt_cond, spk_emb)
            except Exception as e:
                print(f"[clone] WARNING: conditioning failed for spk{sp['id']} "
                      f"({e}); using pool voice.", file=sys.stderr)
        cond_cache[sp["id"]] = val
        return val

    xtts_kw = dict(temperature=0.7, length_penalty=1.0, repetition_penalty=5.0,
                   top_k=50, top_p=0.85, enable_text_splitting=True)

    def synth_clone(text, path, cond):
        gpt_cond, spk_emb = cond
        out = model.inference(text, args.language, gpt_cond, spk_emb, **xtts_kw)
        wav = out["wav"]
        t = torch.as_tensor(wav).float().reshape(1, -1)
        torchaudio.save(path, t, 24000)

    def synth_pool(text, path, voice):
        try:
            tts.tts_to_file(text=text, language=args.language, file_path=path,
                            speaker=voice, **xtts_kw)
        except TypeError:
            tts.tts_to_file(text=text, language=args.language, file_path=path,
                            speaker=voice)

    # --- render ---------------------------------------------------------------
    sr = 24000
    last_end_ms = max(c["end"] for c in cues) * 1000.0
    total_ms = int(max(args.duration * 1000.0, last_end_ms)) + 2000
    canvas = AudioSegment.silent(duration=total_ms, frame_rate=sr)
    tmp = tempfile.mkdtemp(prefix="clone_")
    n = len(cues)
    clones = pools = 0

    for i, c in enumerate(cues):
        sp = c["speaker"]
        cond = get_cond(sp) if sp else None
        clip = AudioSegment.empty()
        for j, piece in enumerate(split_long(c["text"])):
            wp = os.path.join(tmp, f"{i}_{j}.wav")
            if cond is not None:
                try:
                    synth_clone(piece, wp, cond)
                except Exception as e:
                    print(f"[clone] WARNING: clone synth failed ({e}); pool voice.",
                          file=sys.stderr)
                    cond = None
                    cond_cache[sp["id"]] = None
            if cond is None:
                synth_pool(piece, wp, (sp["voice"] if sp and sp["voice"] else default_voice))
            clip += AudioSegment.from_file(wp)

        if cond is not None:
            clones += 1
        else:
            pools += 1
            # only pool-voice fallbacks get age pitch-shifted (clones carry the
            # source's own pitch already)
            b = sp["bucket"] if sp else None
            if b in CHILD_BUCKETS and args.child_pitch_shift:
                clip = pitch_shift_semitones(clip, args.child_pitch_shift)
            elif b in ELDERLY_BUCKETS and args.elder_pitch_shift:
                clip = pitch_shift_semitones(clip, -abs(args.elder_pitch_shift))

        start_ms = int(c["start"] * 1000)
        fitted = ""
        if args.fit and i + 1 < n:
            gap = int(cues[i + 1]["start"] * 1000) - start_ms
            if gap > 0 and len(clip) > gap:
                factor = min(len(clip) / gap, args.max_speed)
                if factor > 1.01:
                    clip = speedup(clip, playback_speed=factor)
                    fitted = f" x{factor:.2f}"
        canvas = canvas.overlay(clip, position=start_ms)

        if args.verbose:
            who = f"spk{sp['id']}" if sp else "default"
            how = "clone" if cond is not None else "pool"
            t = c["start"]
            ts = f"{int(t//3600):02d}:{int(t//60%60):02d}:{int(t%60):02d}"
            prev = c["text"] if len(c["text"]) <= 40 else c["text"][:37] + "..."
            print(f"[clone] {i+1:>4}/{n} [{ts}] {who:<8} {how:<5} "
                  f"{len(clip)/1000:4.1f}s {prev}{fitted}", flush=True)
        elif (i + 1) % 25 == 0 or i + 1 == n:
            print(f"[clone] {i+1}/{n} lines", flush=True)

    canvas.export(args.out, format="wav")
    print(f"[clone] wrote {args.out}  (cloned {clones}, pool-fallback {pools})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
