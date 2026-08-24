#!/usr/bin/env python3
"""Whisper-translate fallback: generate English .srt for films subliminal missed.

For each film, extract the original-language audio, run faster-whisper in
`translate` mode (source -> English) and write <base>.en.srt beside the source.
Fully local on the RVA venv (has the Blackwell cuBLAS fix). Resumable: skips a
film whose .en.srt already exists.

Run with the RVA venv:
  PYTHONUTF8=1 .../ring-video-analyzer/.venv/Scripts/python.exe whisper_subs.py
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# (title, source-language hint, file path). Language hint sharpens translation.
# Forward-slash UNC is used throughout (reliable on Windows Python, sidesteps
# backslash-escaping); the T-34 folder name starts with Cyrillic capital Te.
_ROOT = "//NAS/media/movies"
FILMS = [
    ("Aniara (2018)", "sv", f"{_ROOT}/15-18/Aniara (2019)/Aniara (2019) Bluray-1080p {{edition-DTS 5.1}}.mkv"),
    ("Godzilla vs. Destoroyah (1995)", "ja", f"{_ROOT}/15-18/Godzilla vs. Destoroyah (1995)/Godzilla vs. Destoroyah (1995) Bluray-1080p {{edition-AC3 2.0}}.mkv"),
    ("The Wandering Earth (2019)", "zh", f"{_ROOT}/15-18/The Wandering Earth (2019)/The Wandering Earth (2019) Bluray-1080p {{edition-AC3 5.1}}.mkv"),
    ("Hell (2011)", "de", f"{_ROOT}/15-18/Hell (2011)/Hell (2011) WEBDL-1080p {{edition-DTS 5.1}}.mkv"),
    ("The Zone of Interest (2023)", "de", f"{_ROOT}/18+/The Zone of Interest (2023)/The Zone of Interest (2023) Bluray-2160p {{edition-AAC 5.1}}.mkv"),
    ("T-34 (2018)", "ru", f"{_ROOT}/18+/{chr(0x0422)}-34 (2018)/T-34 (2018) Bluray-1080p {{edition-AC3 5.1}}.mkv"),
]
TMP = r"G:\Transcode\whisper-subs-tmp"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ensure_cuda_dlls() -> None:
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:
        pass
    if hasattr(os, "add_dll_directory"):
        try:
            import nvidia
            base = Path(nvidia.__file__).resolve().parent
            for b in base.glob("*/bin"):
                try:
                    os.add_dll_directory(str(b))
                except OSError:
                    pass
        except Exception:
            pass


def srt_ts(sec: float) -> str:
    if sec < 0:
        sec = 0
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extract_audio(video: str, dst: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-v", "error", "-i", video,
         "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", dst],
        check=True,
    )


def write_srt(segments, path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            n += 1
            f.write(f"{n}\n{srt_ts(seg.start)} --> {srt_ts(seg.end)}\n{text}\n\n")
    return n


def load_list(path: str):
    """TSV: title <tab> src-lang <tab> video-path (forward-slash UNC). Blank
    lines and #-comments ignored. Lets a caller pass its own batch instead of
    editing FILMS."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                log(f"WARN bad list line (need 3 tab-separated fields): {line}")
                continue
            out.append((parts[0], parts[1], parts[2]))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="", help="TSV of title/lang/path to use instead of FILMS")
    args = ap.parse_args()
    films = load_list(args.list) if args.list else FILMS

    os.makedirs(TMP, exist_ok=True)
    ensure_cuda_dlls()
    from faster_whisper import WhisperModel
    log("loading faster-whisper large-v3 (cuda/float16) ...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    log("model ready")

    for title, lang, video in films:
        base, _ = os.path.splitext(video)
        out_srt = base + ".en.srt"
        if os.path.exists(out_srt):
            log(f"SKIP {title}: {out_srt} exists")
            continue
        if not os.path.exists(video):
            log(f"ERROR {title}: source missing {video}")
            continue
        log(f"=== {title}  (src={lang}) ===")
        wav = os.path.join(TMP, f"{abs(hash(video))}.wav")
        try:
            log("  extracting audio ...")
            extract_audio(video, wav)
            log("  transcribe+translate ...")
            segments, info = model.transcribe(
                wav, task="translate", language=lang,
                vad_filter=True, beam_size=5,
                condition_on_previous_text=False,
            )
            tmp_srt = wav + ".en.srt"
            n = write_srt(segments, tmp_srt)
            # move to NAS beside source
            import shutil
            shutil.move(tmp_srt, out_srt)
            log(f"  DONE {title}: {n} cues -> {out_srt}")
        except Exception as e:
            log(f"  FAIL {title}: {type(e).__name__}: {e}")
        finally:
            try:
                if os.path.exists(wav):
                    os.remove(wav)
            except OSError:
                pass
    log("all films processed")


if __name__ == "__main__":
    main()
