#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# srt_to_speech_indextts.py  (Phase B engine: IndexTTS2 expressive cloned dub)
# -----------------------------------------------------------------------------
# Render a timed English dub where each line is spoken in the CHARACTER's own
# cloned voice AND carries the emotion/tone of that character's ORIGINAL Japanese
# line. For each subtitle cue we:
#   * match the cue to a profiled speaker by TIME-OVERLAP against the Phase A
#     profile's per-episode `turns` (every episode is profiled, so this covers
#     the whole show; unmatched cues fall back to the temporally-nearest speaker),
#   * clone that speaker's timbre from their Phase A reference clip
#     (spk_audio_prompt),
#   * transfer emotion from the ORIGINAL Japanese audio under the cue
#     (emo_audio_prompt sliced from --ref-audio), and
#   * synthesize the English subtitle text with IndexTTS2, placing it at the cue
#     timestamp and (optionally) time-compressing overruns to fit the next cue.
#
# Deliberately self-contained: needs only indextts + srt + numpy + librosa +
# soundfile (NOT the demucs/speechbrain profiler stack), because matching is done
# from the profile JSON's turns, not by re-embedding audio. Runs under the
# IndexTTS2 venv (Python 3.10/3.11), separate from the profiler's .venv-dub.
#
# Exit codes: 0 ok; 2 no subtitle cues; 3 a dependency is missing;
#             4 bad profile / ref-audio / checkpoints.
# -----------------------------------------------------------------------------
import argparse
import json
import os
import re
import sys
import tempfile
import time

OUT_SR = 24000          # output track sample rate (mixed to 48k later by ffmpeg)


def eprint(*a):
    print(*a, file=sys.stderr)


# --- subtitle parsing ---------------------------------------------------------

_TAG = re.compile(r"\{[^}]*\}")          # ASS override tags {\an8} etc.
_HTML = re.compile(r"<[^>]*>")           # <i> </i>


def clean_text(s):
    s = _TAG.sub("", s)
    s = _HTML.sub("", s)
    s = s.replace("\\N", " ").replace("\\n", " ")
    return " ".join(s.split()).strip()


def load_cues(srt_path):
    import srt
    with open(srt_path, encoding="utf-8", errors="replace") as f:
        subs = list(srt.parse(f.read()))
    cues = []
    for s in subs:
        text = clean_text(s.content)
        if not text or not re.search(r"[A-Za-z0-9]", text):
            continue
        cues.append({"start": s.start.total_seconds(),
                     "end": s.end.total_seconds(), "text": text})
    cues.sort(key=lambda c: c["start"])
    # drop stacked duplicates (same text overlapping the previous cue)
    deduped = []
    for c in cues:
        if deduped and c["text"] == deduped[-1]["text"] \
                and c["start"] < deduped[-1]["end"] + 0.05:
            continue
        deduped.append(c)
    return deduped


def split_long(text, limit=220):
    """Split an over-long cue on sentence boundaries so IndexTTS2 stays stable."""
    if len(text) <= limit:
        return [text]
    parts, buf = [], ""
    for chunk in re.split(r"(?<=[.!?])\s+", text):
        if len(buf) + len(chunk) + 1 > limit and buf:
            parts.append(buf.strip())
            buf = chunk
        else:
            buf = f"{buf} {chunk}".strip()
    if buf:
        parts.append(buf.strip())
    return parts or [text]


# --- profile + cue->speaker matching -----------------------------------------

def load_profile(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    speakers = data.get("speakers", [])
    base = os.path.dirname(os.path.abspath(path))
    for sp in speakers:
        sp["_refs_abs"] = [os.path.normpath(os.path.join(base, r))
                           for r in sp.get("references", [])]
    return data, speakers


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def build_turns(speakers, episode_name):
    """{speaker_id: [(start,end), ...]} for this episode."""
    turns = {}
    for sp in speakers:
        segs = (sp.get("turns") or {}).get(episode_name)
        if segs:
            turns[sp["id"]] = [(x["start"], x["end"]) for x in segs]
    return turns


def match_cue(cue, turns):
    """Speaker id by max time-overlap, else the temporally-nearest turn."""
    best_id, best_ov = None, 0.0
    for sid, segs in turns.items():
        ov = sum(overlap(cue["start"], cue["end"], s0, s1) for s0, s1 in segs)
        if ov > best_ov:
            best_ov, best_id = ov, sid
    if best_id is not None:
        return best_id
    mid = 0.5 * (cue["start"] + cue["end"])
    best_id, best_gap = None, 1e18
    for sid, segs in turns.items():
        for s0, s1 in segs:
            gap = 0.0 if s0 <= mid <= s1 else min(abs(mid - s0), abs(mid - s1))
            if gap < best_gap:
                best_gap, best_id = gap, sid
    return best_id


# --- emotion reference slicing ------------------------------------------------

def load_ref_audio(path):
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def emotion_clip(ref, sr, start, end, work, idx):
    """Write the original-audio slice under a cue to a temp wav, or None if the
    slice is too short/quiet to carry useful emotion."""
    import numpy as np
    import soundfile as sf
    a = max(0, int(start * sr))
    b = min(len(ref), int(end * sr))
    if b - a < int(0.4 * sr):
        return None
    seg = ref[a:b]
    if float(np.sqrt(np.mean(seg ** 2))) < 1e-3:      # near silence
        return None
    p = os.path.join(work, f"emo_{idx:04d}.wav")
    sf.write(p, seg, sr)
    return p


# --- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--ref-audio", required=True,
                    help="mono WAV of the ORIGINAL speech, aligned to the subs; "
                         "sliced per cue as the emotion reference")
    ap.add_argument("--episode-name", default=None,
                    help="profiled episode base name (keys the profile turns)")
    ap.add_argument("--checkpoints-dir", required=True,
                    help="IndexTTS2 checkpoints dir (config.yaml + weights)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="video duration (s); pads the output track")
    ap.add_argument("--emo-alpha", type=float, default=0.7,
                    help="emotion transfer strength 0..1 (0 = ignore original "
                         "delivery, pure neutral clone)")
    ap.add_argument("--no-emotion", action="store_true",
                    help="disable emotion transfer entirely (timbre clone only)")
    ap.add_argument("--fallback-speaker-id", type=int, default=None,
                    help="speaker id whose voice covers un-matched cues "
                         "(default: the speaker with the most speech)")
    ap.add_argument("--fit", action="store_true",
                    help="time-compress a line that overruns the next cue start")
    ap.add_argument("--max-speed", type=float, default=1.6)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        import numpy as np
        import librosa
        import soundfile as sf
    except ModuleNotFoundError as e:
        eprint(f"[indextts] FATAL: missing dependency ({e}). "
               f"Install: uv pip install srt soundfile  (librosa comes with indextts)")
        return 3

    cfg = os.path.join(args.checkpoints_dir, "config.yaml")
    if not os.path.isfile(cfg):
        eprint(f"[indextts] FATAL: no config.yaml in {args.checkpoints_dir}")
        return 4

    cues = load_cues(args.srt)
    if not cues:
        eprint("[indextts] no subtitle cues after cleaning; nothing to render.")
        return 2

    data, speakers = load_profile(args.profile)
    if not speakers:
        eprint("[indextts] FATAL: profile has no speakers.")
        return 4
    by_id = {sp["id"]: sp for sp in speakers}

    # fallback speaker = most speech (speakers are rank-ordered) unless overridden
    fb_id = args.fallback_speaker_id
    if fb_id is None or fb_id not in by_id:
        fb_id = max(speakers, key=lambda s: s.get("total_speech_sec", 0))["id"]

    turns = build_turns(speakers, args.episode_name) if args.episode_name else {}
    if not turns:
        eprint(f"[indextts] WARNING: no profile turns for episode "
               f"'{args.episode_name}'; every cue uses the fallback voice "
               f"(spk{fb_id}). Re-profile this episode for per-character voices.")

    ref, ref_sr = load_ref_audio(args.ref_audio)

    # assign a speaker to every cue
    for c in cues:
        c["speaker"] = match_cue(c, turns) if turns else fb_id
        if c["speaker"] is None or c["speaker"] not in by_id:
            c["speaker"] = fb_id

    if args.verbose:
        from collections import Counter
        tally = Counter(c["speaker"] for c in cues)
        eprint(f"[indextts] {len(cues)} cues, {len(tally)} distinct speakers")
        for sid, n in tally.most_common():
            sp = by_id[sid]
            eprint(f"[indextts]   spk{sid:02d} {sp.get('bucket','?'):<14} "
                   f"{n:4d} cue(s)  refs={len(sp['_refs_abs'])}")

    # timbre prompt per speaker: the (longest) reference clip, cached
    def timbre_for(sid):
        sp = by_id.get(sid) or by_id[fb_id]
        refs = sp["_refs_abs"] or by_id[fb_id]["_refs_abs"]
        return refs[0] if refs else None

    # --- load IndexTTS2 -------------------------------------------------------
    try:
        from indextts.infer_v2 import IndexTTS2
    except ModuleNotFoundError as e:
        eprint(f"[indextts] FATAL: cannot import IndexTTS2 ({e}). "
               f"Run this under the IndexTTS2 venv.")
        return 3
    eprint(f"[indextts] loading IndexTTS2 from {args.checkpoints_dir} "
           f"(fp16={args.fp16}) ...")
    tts = IndexTTS2(cfg_path=cfg, model_dir=args.checkpoints_dir,
                    use_fp16=args.fp16, use_deepspeed=False,
                    use_cuda_kernel=False)

    # --- render ---------------------------------------------------------------
    total_sec = max(args.duration, max(c["end"] for c in cues)) + 2.0
    canvas = np.zeros(int(total_sec * OUT_SR) + 1, dtype=np.float32)
    n = len(cues)
    work = tempfile.mkdtemp(prefix="indextts_")
    rendered = 0
    t_start = time.time()
    for i, c in enumerate(cues):
        timbre = timbre_for(c["speaker"])
        if not timbre or not os.path.isfile(timbre):
            eprint(f"[indextts]   cue {i}: no timbre clip for spk{c['speaker']}, "
                   f"skipping")
            continue
        emo = None
        if not args.no_emotion and args.emo_alpha > 0:
            emo = emotion_clip(ref, ref_sr, c["start"], c["end"], work, i)

        pieces = []
        for j, part in enumerate(split_long(c["text"])):
            tmp = os.path.join(work, f"cue_{i:04d}_{j}.wav")
            try:
                tts.infer(spk_audio_prompt=timbre, text=part, output_path=tmp,
                          emo_audio_prompt=emo, emo_alpha=args.emo_alpha,
                          verbose=False)
            except Exception as e:
                eprint(f"[indextts]   cue {i} piece {j} failed: {e}")
                continue
            y, _ = librosa.load(tmp, sr=OUT_SR, mono=True)
            pieces.append(y)
            try:
                os.remove(tmp)
            except OSError:
                pass
        if emo:
            try:
                os.remove(emo)
            except OSError:
                pass
        if not pieces:
            continue
        clip = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

        # fit: compress if the line overruns the gap to the next cue
        if args.fit and i + 1 < n:
            gap = cues[i + 1]["start"] - c["start"]
            clip_sec = len(clip) / OUT_SR
            if gap > 0 and clip_sec > gap:
                rate = min(clip_sec / gap, args.max_speed)
                if rate > 1.01:
                    clip = librosa.effects.time_stretch(clip, rate=rate)

        pos = int(c["start"] * OUT_SR)
        end = min(pos + len(clip), canvas.size)
        canvas[pos:end] += clip[:end - pos]
        rendered += 1
        if args.verbose and (rendered % 10 == 0 or i == n - 1):
            el = time.time() - t_start
            rate = rendered / el if el > 0 else 0.0
            eta_m = ((n - (i + 1)) / rate / 60.0) if rate > 0 else 0.0
            eprint(f"[indextts]   {rendered}/{n} lines  "
                   f"({el / 60:.1f}m elapsed, {rate * 60:.0f}/min, ~{eta_m:.0f}m left)")

    peak = float(np.max(np.abs(canvas))) if canvas.size else 0.0
    if peak > 1.0:
        canvas /= peak
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, canvas, OUT_SR)
    eprint(f"[indextts] wrote {args.out}  ({rendered}/{n} lines, "
           f"{total_sec:.0f}s @ {OUT_SR}Hz)")

    try:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
