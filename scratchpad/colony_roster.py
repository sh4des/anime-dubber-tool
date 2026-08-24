#!/usr/bin/env python3
"""Roster report for a profiled title: speakers ordered by how many SUBTITLE
LINES they will actually be given in Phase B.

Replicates srt_to_speech_indextts.py's cue->speaker assignment exactly
(load_cues dedup + build_turns + match_cue max-overlap-then-nearest) so the
counts here are the counts the render will use - not a diarization-time
approximation.

Usage:
  python colony_roster.py --profile <anime-dub-profile.json> --srt <file.srt>
                          [--episode "Colony (2026)"] [--samples 2]
                          [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

_TAG = re.compile(r"\{[^}]*\}")
_HTML = re.compile(r"<[^>]*>")


def clean_text(s: str) -> str:
    s = _TAG.sub("", s)
    s = _HTML.sub("", s)
    s = s.replace("\\N", " ").replace("\\n", " ")
    return " ".join(s.split()).strip()


def load_cues(srt_path: str):
    """Same filtering + stacked-duplicate dedup as the render engine."""
    import srt as srtlib
    with open(srt_path, encoding="utf-8", errors="replace") as f:
        subs = list(srtlib.parse(f.read()))
    cues = []
    for s in subs:
        text = clean_text(s.content)
        if not text or not re.search(r"[A-Za-z0-9]", text):
            continue
        cues.append({"start": s.start.total_seconds(),
                     "end": s.end.total_seconds(), "text": text})
    cues.sort(key=lambda c: c["start"])
    deduped = []
    for c in cues:
        if deduped and c["text"] == deduped[-1]["text"] \
                and c["start"] < deduped[-1]["end"] + 0.05:
            continue
        deduped.append(c)
    return deduped


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def build_turns(speakers, episode_name):
    turns = {}
    for sp in speakers:
        segs = (sp.get("turns") or {}).get(episode_name)
        if segs:
            turns[sp["id"]] = [(x["start"], x["end"]) for x in segs]
    return turns


def match_cue(cue, turns):
    best_id, best_ov = None, 0.0
    for sid, segs in turns.items():
        ov = sum(overlap(cue["start"], cue["end"], s0, s1) for s0, s1 in segs)
        if ov > best_ov:
            best_ov, best_id = ov, sid
    if best_id is not None:
        return best_id, True          # matched by real overlap
    mid = 0.5 * (cue["start"] + cue["end"])
    best_id, best_gap = None, 1e18
    for sid, segs in turns.items():
        for s0, s1 in segs:
            gap = 0.0 if s0 <= mid <= s1 else min(abs(mid - s0), abs(mid - s1))
            if gap < best_gap:
                best_gap, best_id = gap, sid
    return best_id, False             # fell back to nearest-turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--episode", default="")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    with open(args.profile, encoding="utf-8") as f:
        prof = json.load(f)
    speakers = prof.get("speakers", [])
    base = os.path.dirname(os.path.abspath(args.profile))

    # Episode key: the profile's turns are keyed by source base name.
    keys = set()
    for sp in speakers:
        keys.update((sp.get("turns") or {}).keys())
    episode = args.episode
    if not episode:
        if len(keys) == 1:
            episode = next(iter(keys))
        else:
            sys.exit(f"--episode required; profile has turn keys: {sorted(keys)}")
    if episode not in keys:
        sys.exit(f"episode {episode!r} not in profile turn keys: {sorted(keys)}")

    cues = load_cues(args.srt)
    turns = build_turns(speakers, episode)
    if not turns:
        sys.exit(f"no turns for episode {episode!r} - Phase B would use ONE fallback voice")

    by_id = {sp["id"]: sp for sp in speakers}
    stats = {}
    for c in cues:
        sid, real = match_cue(c, turns)
        st = stats.setdefault(sid, {"lines": 0, "nearest": 0, "chars": 0,
                                    "first": c["start"], "last": c["end"], "texts": []})
        st["lines"] += 1
        st["chars"] += len(c["text"])
        st["nearest"] += 0 if real else 1
        st["first"] = min(st["first"], c["start"])
        st["last"] = max(st["last"], c["end"])
        st["texts"].append(c["text"])

    rows = []
    for sid, st in stats.items():
        sp = by_id.get(sid, {})
        segs = turns.get(sid, [])
        speech = sum(s1 - s0 for s0, s1 in segs)
        refs = [os.path.normpath(os.path.join(base, r)) for r in sp.get("references", [])]
        refs_ok = [r for r in refs if os.path.exists(r)]
        rows.append({
            "speaker": sid,
            "lines": st["lines"],
            "pct": 100.0 * st["lines"] / len(cues),
            "nearest_fallback": st["nearest"],
            "chars": st["chars"],
            "speech_s": speech,
            "turns": len(segs),
            "gender": sp.get("gender", "?"),
            "bucket": sp.get("bucket", "?"),
            "refs_total": len(refs),
            "refs_on_disk": len(refs_ok),
            "first_s": st["first"],
            "last_s": st["last"],
            "samples": st["texts"],
        })
    rows.sort(key=lambda r: -r["lines"])

    def ts(x):
        return f"{int(x)//3600:d}:{(int(x)%3600)//60:02d}:{int(x)%60:02d}"

    print(f"\n=== ROSTER: {episode} ===")
    print(f"{len(cues)} dialogue cues -> {len(rows)} speaking roles "
          f"(profile cast {len(speakers)})\n")
    hdr = (f"{'#':>3}  {'speaker':<8} {'lines':>6} {'%':>6} {'speech':>8} "
           f"{'turns':>6} {'clips':>7} {'gender':<7} {'first':>8} {'last':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows, 1):
        clips = f"{r['refs_on_disk']}/{r['refs_total']}"
        warn = "  <-- NO TIMBRE" if r["refs_on_disk"] == 0 else ""
        print(f"{i:>3}  {r['speaker']:<8} {r['lines']:>6} {r['pct']:>5.1f}% "
              f"{ts(r['speech_s']):>8} {r['turns']:>6} {clips:>7} "
              f"{r['gender']:<7} {ts(r['first_s']):>8} {ts(r['last_s']):>8}{warn}")

    silent = [sp['id'] for sp in speakers if sp['id'] not in stats]
    if silent:
        print(f"\nprofiled but get 0 lines ({len(silent)}): {', '.join(silent)}")
    tot_near = sum(r["nearest_fallback"] for r in rows)
    print(f"\ncues assigned by real overlap: {len(cues)-tot_near}  "
          f"by nearest-turn fallback: {tot_near} ({100.0*tot_near/len(cues):.1f}%)")

    if args.samples:
        print("\n--- sample lines per role ---")
        for i, r in enumerate(rows, 1):
            picks = r["samples"][:args.samples]
            print(f"{i:>3}  {r['speaker']:<8} " + " | ".join(f'"{p}"' for p in picks))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "speaker", "lines", "pct", "nearest_fallback", "chars",
                        "speech_s", "turns", "gender", "bucket", "refs_on_disk",
                        "refs_total", "first_s", "last_s"])
            for i, r in enumerate(rows, 1):
                w.writerow([i, r["speaker"], r["lines"], f"{r['pct']:.2f}", r["nearest_fallback"],
                            r["chars"], f"{r['speech_s']:.1f}", r["turns"], r["gender"],
                            r["bucket"], r["refs_on_disk"], r["refs_total"],
                            f"{r['first_s']:.1f}", f"{r['last_s']:.1f}"])
        print(f"\ncsv -> {args.csv}")


if __name__ == "__main__":
    main()
