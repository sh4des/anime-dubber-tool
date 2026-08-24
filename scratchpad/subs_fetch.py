#!/usr/bin/env python3
"""Fetch English subtitles for the 9 dub-blocked films via subliminal.

For each film (paths resolved from Plex by find_sub_targets), run subliminal's
CLI to download the best English subtitle from free/anonymous providers
(hash-matched where possible) and save it beside the source as <base>.en.srt.
Reports HIT / MISS so misses can go to the Whisper-translate fallback.

Run with the subs venv:
  PYTHONUTF8=1 G:/Transcode/.venv-subs/Scripts/python.exe subs_fetch.py
"""
from __future__ import annotations
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_sub_targets import TARGETS, movie_section, find, parts  # noqa: E402

VENV_PY = r"G:\Transcode\.venv-subs\Scripts\python.exe"
# Anonymous providers only (no OpenSubtitles account, per user's "different source")
PROVIDERS = ["podnapisi", "bsplayer", "tvsubtitles"]
CACHE = r"G:\Transcode\.venv-subs\cache"


def en_srt_path(video: str) -> str:
    base, _ = os.path.splitext(video)
    return base + ".en.srt"


def already(video: str) -> str | None:
    base, _ = os.path.splitext(video)
    for cand in (base + ".en.srt", base + ".eng.srt"):
        if os.path.exists(cand):
            return cand
    return None


def fetch(video: str) -> tuple[str, str]:
    """Return (status, detail). status in HIT/MISS/ERROR/EXISTS."""
    if not os.path.exists(video):
        return "ERROR", "source missing"
    pre = already(video)
    if pre:
        return "EXISTS", pre
    cmd = [VENV_PY, "-m", "subliminal", "--cache-dir", CACHE, "download",
           "-l", "en"]
    for p in PROVIDERS:
        cmd += ["-p", p]
    cmd += ["--", video]
    env = dict(os.environ, PYTHONUTF8="1")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=300, env=env)
    except subprocess.TimeoutExpired:
        return "ERROR", "subliminal timeout"
    out = (r.stdout or "") + (r.stderr or "")
    got = en_srt_path(video)
    if os.path.exists(got):
        return "HIT", got
    # subliminal prints "Downloaded N subtitle(s)" — surface last meaningful line
    tail = " | ".join([l.strip() for l in out.splitlines() if l.strip()][-2:])
    return "MISS", tail[:200]


def main():
    section = movie_section()
    print(f"providers: {PROVIDERS}\n")
    hits, misses = [], []
    for title, year in TARGETS:
        v = find(section, title, year)
        files = parts(v) if v is not None else []
        if not files:
            print(f"[ERROR] {title} ({year}): no Plex file")
            misses.append((title, year, None))
            continue
        video = files[0]
        status, detail = fetch(video)
        print(f"[{status:6}] {title} ({year})")
        if status in ("HIT", "EXISTS"):
            print(f"          -> {detail}")
            hits.append((title, year, detail))
        else:
            print(f"          {detail}")
            misses.append((title, year, video))
    print("\n=== SUMMARY ===")
    print(f"got subs : {len(hits)}")
    for t, y, d in hits:
        print(f"  HIT  {t} ({y})")
    print(f"need Whisper fallback : {len(misses)}")
    for t, y, vid in misses:
        print(f"  MISS {t} ({y})  {vid or ''}")


if __name__ == "__main__":
    main()
