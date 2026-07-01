#!/usr/bin/env python3
# srt_to_speech_multivoice.py
# -----------------------------------------------------------------------------
# Multi-voice variant of srt_to_speech.py.
#
# Same job (SRT -> one timed English dub WAV, lines placed at their cue times),
# but instead of ONE voice for everything, each subtitle line is spoken by one
# of four voices chosen from the ORIGINAL (Japanese) audio under that cue:
#
#     adult male | adult female | child male | child female
#
# How the "who's speaking" call is made
# -------------------------------------
# We do NOT do face/scene analysis - that's slow and unreliable. Instead we look
# at the original speech that already lines up with each subtitle: for every cue
# we slice the reference audio to [start,end], estimate the median voiced pitch
# (F0) with torchaudio's pitch detector, and bucket it:
#
#     F0 <  --male-max          -> adult male     (default 155 Hz)
#     F0 <  --adult-female-max  -> adult female   (default 250 Hz)
#     F0 <  --child-split       -> child male     (default 300 Hz)
#     else                      -> child female
#
# Each bucket maps to a distinct XTTS built-in speaker (see --voice-* args).
# Child buckets are optionally pitch-shifted up (--child-pitch-shift) so the
# adult XTTS voices read younger.
#
# HONEST LIMITATIONS
#   * Child male vs child female is barely separable by pitch (pre-pubescent
#     voices overlap heavily). The --child-split boundary is a best-effort knob,
#     not a reliable classifier. If you only care about "a kid voice", set both
#     child voices to the same speaker.
#   * A cue with two people talking, or only music/SFX, classifies as whoever/
#     whatever is loudest, or falls back to --voice-default when no clear voiced
#     pitch is found.
#   * Requires the reference audio to line up with the subtitle timing (true for
#     embedded subs and same-release sidecars; a mismatched sidecar will
#     mis-time the classification).
#
# TTS engine + install: identical to srt_to_speech.py (Coqui XTTS-v2). No new
# dependencies - torchaudio (already required) provides both pitch detection and
# the resample used for the child pitch-shift.
# -----------------------------------------------------------------------------

import argparse
import os
import re
import sys
import tempfile

os.environ.setdefault("COQUI_TOS_AGREED", "1")

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
MAX_CHARS = 240

# Distinct XTTS-v2 built-in speakers per bucket. All swappable via --voice-*.
# "Damien Black" matches the single-voice script's default so the adult-male
# dub sounds the same as the basic variant.
DEFAULT_VOICES = {
    "adult_male":    "Damien Black",
    "adult_female":  "Alison Dietlinde",
    "child_male":    "Andrew Chipper",
    "child_female":  "Tammie Ema",
}
BUCKET_LABELS = {
    "adult_male":   "adult male",
    "adult_female": "adult female",
    "child_male":   "child male",
    "child_female": "child female",
    "default":      "default/unclear",
}


def clean_text(raw: str) -> str:
    """Strip subtitle markup so only spoken words reach the TTS engine."""
    t = raw
    t = re.sub(r"\{[^}]*\}", " ", t)        # ASS/SSA override tags {\an8} etc.
    t = re.sub(r"<[^>]+>", " ", t)          # HTML-ish <i>, <b>, <font ...>
    t = re.sub(r"\\N|\\n", " ", t)          # ASS hard line breaks
    t = t.replace("\n", " ")                # SRT multi-line cues
    t = re.sub(r"^\s*[-–—]\s*", "", t)      # leading dialogue dash
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_long(text: str, limit: int = MAX_CHARS):
    """Split overly long lines on sentence boundaries to keep XTTS stable."""
    if len(text) <= limit:
        return [text]
    parts, buf = [], ""
    for chunk in re.split(r"(?<=[.!?…])\s+", text):
        if len(buf) + len(chunk) + 1 > limit and buf:
            parts.append(buf.strip())
            buf = chunk
        else:
            buf = f"{buf} {chunk}".strip()
    if buf:
        parts.append(buf.strip())
    return parts


def voiced_median_pitch(seg, sr, fmin=70.0, fmax=450.0):
    """Median voiced F0 (Hz) of a mono audio segment, or None if no clear voice.

    Uses torchaudio's autocorrelation pitch detector, then keeps only frames
    that are both loud (> 15% of the segment's peak RMS) and inside the plausible
    speech range - this rejects silence, music and SFX so the median reflects the
    person actually talking.
    """
    import torch
    import torchaudio.functional as taf

    if seg.dim() > 1:
        seg = seg.mean(dim=0)
    if seg.numel() < int(0.15 * sr):     # too short to trust
        return None

    try:
        pitch = taf.detect_pitch_frequency(seg.unsqueeze(0), sr).squeeze(0)
    except Exception:
        return None
    if pitch.numel() == 0:
        return None

    # Per-frame RMS aligned to the pitch frames, to gate out non-speech frames.
    hop = max(1, seg.numel() // pitch.numel())
    usable = hop * pitch.numel()
    frames = seg[:usable].reshape(pitch.numel(), hop)
    rms = frames.pow(2).mean(dim=1).sqrt()
    peak = float(rms.max())
    if peak <= 0:
        return None

    mask = (rms > 0.15 * peak) & (pitch >= fmin) & (pitch <= fmax)
    vals = pitch[mask]
    if vals.numel() < 3:
        return None
    return float(torch.median(vals))


def classify(f0, male_max, adult_female_max, child_split):
    """Map an F0 (Hz) to one of the four buckets, or 'default' if unknown."""
    if f0 is None:
        return "default"
    if f0 < male_max:
        return "adult_male"
    if f0 < adult_female_max:
        return "adult_female"
    if f0 < child_split:
        return "child_male"
    return "child_female"


def pitch_shift_semitones(seg, semitones):
    """Shift a pydub segment up by N semitones (tape-speed style, no new deps).

    Also speeds the clip up a little; kept small (a couple semitones) so child
    lines read younger without sounding like chipmunks. Runs before --fit so the
    slightly shorter clip just helps it land inside its cue.
    """
    if not semitones:
        return seg
    new_rate = int(seg.frame_rate * (2.0 ** (semitones / 12.0)))
    shifted = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate})
    return shifted.set_frame_rate(seg.frame_rate)


def main() -> int:
    ap = argparse.ArgumentParser(description="SRT -> timed multi-voice English TTS WAV")
    ap.add_argument("--srt", required=True, help="input .srt path")
    ap.add_argument("--out", required=True, help="output .wav path")
    ap.add_argument("--ref-audio", required=True,
                    help="mono WAV of the ORIGINAL speech track, aligned to the "
                         "subtitle timing, used to classify each line's speaker")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="video duration in seconds (pads the track to length)")
    ap.add_argument("--language", default="en")

    # Per-bucket voices (XTTS built-in speaker names).
    ap.add_argument("--voice-adult-male",   default=DEFAULT_VOICES["adult_male"])
    ap.add_argument("--voice-adult-female", default=DEFAULT_VOICES["adult_female"])
    ap.add_argument("--voice-child-male",   default=DEFAULT_VOICES["child_male"])
    ap.add_argument("--voice-child-female", default=DEFAULT_VOICES["child_female"])
    ap.add_argument("--voice-default", default=None,
                    help="voice for cues with no clear pitch (default: adult male)")

    # Classification thresholds (Hz) - tune per show.
    ap.add_argument("--male-max", type=float, default=155.0)
    ap.add_argument("--adult-female-max", type=float, default=250.0)
    ap.add_argument("--child-split", type=float, default=300.0,
                    help="F0 boundary between child male and child female "
                         "(approximate - pre-pubescent voices overlap heavily)")
    ap.add_argument("--child-pitch-shift", type=float, default=2.0,
                    help="semitones to raise child-voice clips (0 to disable)")

    ap.add_argument("--fit", action="store_true",
                    help="speed up lines that overrun into the next cue")
    ap.add_argument("--max-speed", type=float, default=1.6,
                    help="cap for --fit time-compression")
    ap.add_argument("--verbose", action="store_true",
                    help="log every line, its detected pitch and chosen voice")
    args = ap.parse_args()

    voices = {
        "adult_male":   args.voice_adult_male,
        "adult_female": args.voice_adult_female,
        "child_male":   args.voice_child_male,
        "child_female": args.voice_child_female,
    }
    voices["default"] = args.voice_default or args.voice_adult_male
    child_buckets = {"child_male", "child_female"}

    # Heavy imports after arg parsing so --help stays instant.
    try:
        import srt as srtlib
        from pydub import AudioSegment
        from pydub.effects import speedup
        import torch
        import torchaudio
        from TTS.api import TTS
    except ModuleNotFoundError as e:
        print(f"[tts] missing Python dependency: {e.name}", file=sys.stderr)
        print("[tts] install the GPU stack (RTX 50-series needs CUDA 12.8):",
              file=sys.stderr)
        print("      pip install coqui-tts pydub srt", file=sys.stderr)
        print("      pip install torch torchaudio "
              "--index-url https://download.pytorch.org/whl/cu128", file=sys.stderr)
        return 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[tts] device=cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("[tts] WARNING: CUDA not available, running on CPU (very slow).",
              file=sys.stderr)
    print(f"[tts] loading model {XTTS_MODEL} ...")
    tts = TTS(XTTS_MODEL).to(device)
    print("[tts] voices: " + ", ".join(
        f"{BUCKET_LABELS[b]}={voices[b]}"
        for b in ("adult_male", "adult_female", "child_male", "child_female")))

    # --- load + classify from the reference audio -----------------------------
    print(f"[tts] loading reference audio {args.ref_audio} ...")
    try:
        ref_wave, ref_sr = torchaudio.load(args.ref_audio)
    except Exception as e:
        print(f"[tts] could not load reference audio: {e}", file=sys.stderr)
        return 4
    ref = ref_wave.mean(dim=0)                      # mono
    if ref_sr != 16000:                             # 16k is plenty for pitch
        ref = torchaudio.functional.resample(ref, ref_sr, 16000)
        ref_sr = 16000

    with open(args.srt, encoding="utf-8-sig") as f:
        raw_subs = list(srtlib.parse(f.read()))
    if not raw_subs:
        print("[tts] no subtitle cues found, nothing to do.", file=sys.stderr)
        return 2

    # Deduplicate cues (anime ASS subs often stack the same line on layers).
    cues = []
    dropped = 0
    for s in raw_subs:
        text = clean_text(s.content)
        if not text:
            continue
        start, end = s.start.total_seconds(), s.end.total_seconds()
        if any(c["text"] == text and start < c["end"] + 0.5 for c in cues[-8:]):
            dropped += 1
            continue
        cues.append({"text": text, "start": start, "end": end})
    if dropped:
        print(f"[tts] dropped {dropped} duplicate/overlapping cue(s)")

    # Classify each surviving cue against the reference audio.
    counts = {b: 0 for b in BUCKET_LABELS}
    for c in cues:
        a = int(c["start"] * ref_sr)
        b = min(int(c["end"] * ref_sr), ref.numel())
        f0 = voiced_median_pitch(ref[a:b], ref_sr) if b > a else None
        bucket = classify(f0, args.male_max, args.adult_female_max, args.child_split)
        c["bucket"] = bucket
        c["f0"] = f0
        counts[bucket] += 1
    print("[tts] classification: " + ", ".join(
        f"{BUCKET_LABELS[b]}={counts[b]}" for b in BUCKET_LABELS if counts[b]))

    sr = 24000  # XTTS native sample rate
    last_end_ms = max(c["end"] for c in cues) * 1000.0
    total_ms = int(max(args.duration * 1000.0, last_end_ms)) + 2000  # tail pad
    print(f"[tts] {len(cues)} cues, building a {total_ms / 1000:.0f}s track")
    canvas = AudioSegment.silent(duration=total_ms, frame_rate=sr)

    # XTTS inference params tuned to stop the model looping/repeating a line.
    xtts_kw = dict(
        temperature=0.7,
        length_penalty=1.0,
        repetition_penalty=5.0,
        top_k=50,
        top_p=0.85,
        enable_text_splitting=True,
    )

    def synth(text, path, speaker):
        try:
            tts.tts_to_file(text=text, language=args.language, file_path=path,
                            speaker=speaker, **xtts_kw)
        except TypeError:
            # Older coqui-tts that rejects the extra inference kwargs.
            tts.tts_to_file(text=text, language=args.language, file_path=path,
                            speaker=speaker)

    tmp = tempfile.mkdtemp(prefix="dub_")
    n = len(cues)
    for i, c in enumerate(cues):
        text = c["text"]
        bucket = c["bucket"]
        speaker = voices[bucket]

        # Synthesize the (possibly chunked) line into one clip.
        clip = AudioSegment.empty()
        for j, piece in enumerate(split_long(text)):
            wav_path = os.path.join(tmp, f"{i}_{j}.wav")
            synth(piece, wav_path, speaker)
            clip += AudioSegment.from_file(wav_path)

        # Raise child voices so the adult XTTS speakers read younger.
        if bucket in child_buckets and args.child_pitch_shift:
            clip = pitch_shift_semitones(clip, args.child_pitch_shift)

        start_ms = int(c["start"] * 1000)
        fitted = ""

        # Optionally compress a line so it doesn't bleed over the next one.
        if args.fit and i + 1 < n:
            gap = int(cues[i + 1]["start"] * 1000) - start_ms
            if gap > 0 and len(clip) > gap:
                factor = min(len(clip) / gap, args.max_speed)
                if factor > 1.01:
                    clip = speedup(clip, playback_speed=factor)
                    fitted = f" (sped up {factor:.2f}x to fit)"

        canvas = canvas.overlay(clip, position=start_ms)

        if args.verbose:
            t = c["start"]
            ts = f"{int(t//3600):02d}:{int(t//60%60):02d}:{int(t%60):02d}"
            preview = text if len(text) <= 48 else text[:45] + "..."
            f0s = f"{c['f0']:5.0f}Hz" if c["f0"] else "  --  "
            print(f"[tts] {i + 1:>4}/{n} [{ts}] {f0s} {BUCKET_LABELS[bucket]:<13} "
                  f"{len(clip) / 1000:4.1f}s {preview}{fitted}", flush=True)
        elif (i + 1) % 25 == 0 or i + 1 == n:
            print(f"[tts] {i + 1}/{n} lines", flush=True)

    canvas.export(args.out, format="wav")
    print(f"[tts] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
