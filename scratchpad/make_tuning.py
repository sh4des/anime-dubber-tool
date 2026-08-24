#!/usr/bin/env python3
"""Turn each film's voice-audition report into a PER-MOVIE voice_tuning.json.

The audition flags two recoverable faults per speaker:
  CLIPPED -> genuinely flat-topped, audible distortion  -> emo_alpha 0.30
  HOT     -> very hot delivery, may sound harsh         -> emo_alpha 0.35
Everything else keeps the global default (0.45).

Why per-movie: `movie_voice_tuning.json` is SHARED by every film and its
"speakers" map is keyed by speaker id, but spk01 in Colony is a different person
from spk01 in Parasite. A shared per-speaker map would cross-apply the wrong
correction to every other film, so each movie gets its own file, seeded from the
shared globals.

  python make_tuning.py            # write tuning/<key>_voice_tuning.json for all
  python make_tuning.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re

AUD = r"G:\Transcode\voice-auditions"
HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.join(HERE, "movie_voice_tuning.json")
OUTDIR = os.path.join(HERE, "tuning")

ALPHA = {"CLIPPED": 0.30, "HOT": 0.35}
# a speaker whose clone came out too quiet needs level, not less emotion
GAIN = {"LOW_LEVEL": 3.0, "SILENT_CLONE": 6.0}

ROW = re.compile(r"^\s+spk(\d+)\s+(\S+)\s+([A-Z_,]+)\s+\(refs on disk (\d+)/(\d+)", re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(SHARED, encoding="utf-8") as f:
        shared = json.load(f)
    os.makedirs(OUTDIR, exist_ok=True)

    total_c = total_h = 0
    for d in sorted(os.listdir(AUD)):
        rep = os.path.join(AUD, d, "voice_samples_report.txt")
        if not os.path.exists(rep):
            continue
        text = open(rep, encoding="utf-8", errors="replace").read()
        head = text.split("Full cast")[0]          # problem block only
        speakers = {}
        nc = nh = 0
        for sid, bucket, flags, on_disk, in_prof in ROW.findall(head):
            ent = {}
            for fl in flags.split(","):
                if fl in ALPHA:
                    ent["emo_alpha"] = min(ent.get("emo_alpha", 9), ALPHA[fl])
                    nc += fl == "CLIPPED"
                    nh += fl == "HOT"
                if fl in GAIN:
                    ent["gain_db"] = max(ent.get("gain_db", -99), GAIN[fl])
            if ent:
                speakers[str(int(sid))] = ent
        total_c += nc
        total_h += nh

        out = {
            "_doc": (f"Per-movie IndexTTS2 tuning for {d}, generated from its voice "
                     f"audition report. CLIPPED -> emo_alpha 0.30, HOT -> 0.35; the rest "
                     f"inherit the global 0.45. Regenerate with make_tuning.py after "
                     f"re-auditioning."),
            "global": shared.get("global", {}),
            "speakers": speakers,
        }
        path = os.path.join(OUTDIR, f"{d}_voice_tuning.json")
        print(f"  {d:<40} {len(speakers):>3} speaker override(s)  "
              f"(CLIPPED {nc}, HOT {nh})")
        if not args.dry_run:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=1)
    print(f"\ntotal overrides: CLIPPED {total_c}, HOT {total_h}")
    print(f"-> {OUTDIR}" + ("  (dry run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
