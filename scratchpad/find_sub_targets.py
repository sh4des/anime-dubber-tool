#!/usr/bin/env python3
"""Resolve the 9 dub-blocked films to their on-disk file paths + existing subs.

Queries Plex (same host/token as plex-audio-language-scan.py) for each title,
prints the movie folder, the video file, and whether an English .srt already
sits beside it. Output feeds the subliminal/Whisper subtitle fetch.

Point it at your own server before running:

    PLEX_URL=http://plex.lan:32400 PLEX_TOKEN=xxxx python find_sub_targets.py

(Get the token from any Plex web URL's X-Plex-Token parameter.)
"""
from __future__ import annotations
import json
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"{name} is not set - export it before running (see the module docstring).")
    return val


BASE = _require("PLEX_URL").rstrip("/")
TOKEN = _require("PLEX_TOKEN")

# (title, year) — year disambiguates remakes (Hell, T-34, etc.)
TARGETS = [
    ("Aniara", 2018),
    ("Godzilla vs. Destoroyah", 1995),
    ("The Wandering Earth", 2019),
    ("Shin Godzilla", 2016),
    ("Hell", 2011),
    ("The Zone of Interest", 2023),
    ("Shaolin Soccer", 2001),
    ("T-34", 2018),
    ("Cinema Paradiso", 1988),
]


def _get(path: str, **params) -> ET.Element:
    params["X-Plex-Token"] = TOKEN
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/xml"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return ET.fromstring(resp.read())


def movie_section() -> str:
    root = _get("/library/sections")
    for d in root.findall("Directory"):
        if d.get("type") == "movie":
            return d.get("key")
    return "1"


def find(section: str, title: str, year: int):
    # Plex title search within the movies section
    root = _get(f"/library/sections/{section}/all", title=title)
    best = None
    for v in root.findall("Video"):
        vt = v.get("title") or ""
        vy = v.get("year")
        if vt.lower() == title.lower() and (not vy or int(vy) == year):
            best = v
            break
        if title.lower() in vt.lower() and best is None:
            best = v
    return best


def parts(video: ET.Element):
    out = []
    for media in video.findall("Media"):
        for part in media.findall("Part"):
            out.append(part.get("file"))
    return out


def main():
    section = movie_section()
    results = []
    for title, year in TARGETS:
        v = find(section, title, year)
        if v is None:
            results.append({"title": title, "year": year, "found": False, "files": []})
            continue
        files = parts(v)
        entry = {
            "title": title,
            "year": year,
            "found": True,
            "plex_title": v.get("title"),
            "plex_year": v.get("year"),
            "files": files,
        }
        # existing subs beside each file
        subs = []
        for f in files:
            folder = os.path.dirname(f)
            try:
                for name in os.listdir(folder):
                    if name.lower().endswith((".srt", ".ass", ".ssa")):
                        subs.append(os.path.join(folder, name))
            except OSError:
                pass
        entry["existing_subs"] = subs
        results.append(entry)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
