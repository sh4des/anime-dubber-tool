#!/usr/bin/env python3
# srt_to_speech.py
# -----------------------------------------------------------------------------
# Reads an .srt subtitle file and produces a single WAV that is the spoken
# English dub, with every line placed at its subtitle timestamp so the result
# stays roughly in sync with the video.
#
# TTS engine: Coqui XTTS-v2 (neural, GPU accelerated on CUDA / your RTX 5080).
# Output: 24kHz mono WAV at --out, total length >= video duration.
#
# Install (Windows, in a venv):
#   # PyTorch with CUDA matching a 50-series (Blackwell) card. Use cu128+:
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
#   pip install coqui-tts pydub srt
#   # ffmpeg must be on PATH (pydub uses it to load/export audio).
# -----------------------------------------------------------------------------

import argparse
import os
import re
import sys
import tempfile

# XTTS-v2 ships under a non-commercial license that the lib prompts for on first
# run; agreeing here keeps the script non-interactive.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# Default XTTS-v2 built-in speaker, so the script works without a reference clip.
# Swap with --speaker (built-in name) or --speaker-wav (path to a voice sample).
DEFAULT_SPEAKER = "Damien Black"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# XTTS chokes on very long inputs; keep each synth call to a sane size.
MAX_CHARS = 240


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


def main() -> int:
    ap = argparse.ArgumentParser(description="SRT -> timed English TTS WAV")
    ap.add_argument("--srt", required=True, help="input .srt path")
    ap.add_argument("--out", required=True, help="output .wav path")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="video duration in seconds (pads the track to length)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--speaker", default=DEFAULT_SPEAKER,
                    help="XTTS built-in speaker name")
    ap.add_argument("--speaker-wav", default=None,
                    help="path to a voice sample to clone instead of --speaker")
    ap.add_argument("--fit", action="store_true",
                    help="speed up lines that overrun into the next cue")
    ap.add_argument("--max-speed", type=float, default=1.6,
                    help="cap for --fit time-compression")
    ap.add_argument("--verbose", action="store_true",
                    help="log every line as it is synthesized")
    args = ap.parse_args()

    # Heavy imports after arg parsing so --help stays instant.
    try:
        import srt as srtlib
        from pydub import AudioSegment
        from pydub.effects import speedup
        import torch
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
        name = torch.cuda.get_device_name(0)
        print(f"[tts] device=cuda ({name})")
    else:
        print("[tts] WARNING: CUDA not available, running on CPU (very slow).",
              file=sys.stderr)
    print(f"[tts] loading model {XTTS_MODEL} ...")
    tts = TTS(XTTS_MODEL).to(device)
    voice_desc = args.speaker_wav if args.speaker_wav else args.speaker
    print(f"[tts] voice={voice_desc}  language={args.language}  fit={args.fit}")

    with open(args.srt, encoding="utf-8-sig") as f:
        raw_subs = list(srtlib.parse(f.read()))
    if not raw_subs:
        print("[tts] no subtitle cues found, nothing to do.", file=sys.stderr)
        return 2

    # Deduplicate cues. Anime ASS subtitles often render the same dialogue on
    # multiple layers/styles, so the ass->srt conversion yields the SAME text at
    # overlapping times. Speaking each copy is one source of the "echo".
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

    sr = 24000  # XTTS native sample rate
    last_end_ms = max(c["end"] for c in cues) * 1000.0
    total_ms = int(max(args.duration * 1000.0, last_end_ms)) + 2000  # tail pad
    print(f"[tts] {len(cues)} cues, building a {total_ms / 1000:.0f}s track")
    canvas = AudioSegment.silent(duration=total_ms, frame_rate=sr)

    voice = {"speaker_wav": args.speaker_wav} if args.speaker_wav \
        else {"speaker": args.speaker}

    # XTTS inference params tuned to stop the model looping/repeating a line
    # (its most common failure mode) - the other source of the "echo".
    xtts_kw = dict(
        temperature=0.7,
        length_penalty=1.0,
        repetition_penalty=5.0,
        top_k=50,
        top_p=0.85,
        enable_text_splitting=True,
    )

    def synth(text, path):
        # Fall back if a coqui-tts version rejects the extra inference kwargs.
        try:
            tts.tts_to_file(text=text, language=args.language,
                            file_path=path, **voice, **xtts_kw)
        except TypeError:
            tts.tts_to_file(text=text, language=args.language,
                            file_path=path, **voice)

    tmp = tempfile.mkdtemp(prefix="dub_")
    n = len(cues)
    for i, c in enumerate(cues):
        text = c["text"]

        # Synthesize the (possibly chunked) line into one clip.
        clip = AudioSegment.empty()
        for j, piece in enumerate(split_long(text)):
            wav_path = os.path.join(tmp, f"{i}_{j}.wav")
            synth(piece, wav_path)
            clip += AudioSegment.from_file(wav_path)

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
            preview = text if len(text) <= 60 else text[:57] + "..."
            print(f"[tts] {i + 1:>4}/{n} [{ts}] {len(clip) / 1000:4.1f}s "
                  f"{preview}{fitted}", flush=True)
        elif (i + 1) % 25 == 0 or i + 1 == n:
            print(f"[tts] {i + 1}/{n} lines", flush=True)

    canvas.export(args.out, format="wav")
    print(f"[tts] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
