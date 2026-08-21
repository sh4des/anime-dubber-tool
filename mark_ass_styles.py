#!/usr/bin/env python3
"""Mark ASS cues that must NOT be spoken, so their ORIGINAL audio survives.

Why this exists
---------------
srt_to_speech_indextts.py and build_music_bed.py already agree on a contract:
a cue carrying a music note or a bare sound tag is *not dialogue*, so the dub
stays silent there and the music bed keeps the original vocal stem underneath.
That contract is what preserves the real singing over an OP/ED.

But the marker has to be IN the text, and plenty of releases carry no marker at
all - they distinguish song lyrics from dialogue by ASS **style** instead. After
War Gundam X is one: its OP/ED lyrics are plain prose in styles `OP` / `OP2`,
with no music note anywhere in the file. ffmpeg's `-c:s srt` conversion then
throws the style away, so by the time the dub engine sees the cue it looks like
ordinary dialogue. The result is a TTS voice reading the theme song aloud while
the bed mutes the actual singing underneath - both halves of the failure at once.

This bridges that gap: read the styles from the raw ASS, and prefix the matching
cues in the converted SRT with a music note. Nothing downstream changes - both
consumers already honour the marker.

Matching is on (start, end) plus normalised text, so cues that share a timing
(a two-line title card, say) are still told apart.

Usage:
    mark_ass_styles.py --ass in.ass --srt in.srt [--out out.srt]
                       --non-spoken-styles "OP,OP2,title1,title2"

Exit codes: 0 ok (even if nothing matched - see the printed counts), 2 bad input.
"""
from __future__ import annotations

import argparse
import re
import sys

MARK = "♪"                                  # the marker both consumers honour
_ASS_TAG = re.compile(r"\{[^}]*\}")
_DRAW = re.compile(r"\\p[1-9]")                  # vector drawing blocks, never speech

# Styles treated as non-spoken when --non-spoken-styles is not given. Covers the
# usual naming for opening/ending themes, insert songs and karaoke layers.
DEFAULT_NON_SPOKEN = r"^(op|ed|opening|ending|song|insert|lyric|lyrics|karaoke|kara)\d*$"


def merge_spans(spans, gap=0.5):
    """Sort and coalesce (start,end) pairs closer together than `gap` seconds."""
    out = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [[round(a, 3), round(b, 3)] for a, b in out]


def norm(s: str) -> str:
    """Comparable form of a cue's text: tags gone, whitespace collapsed."""
    s = _ASS_TAG.sub("", s or "")
    s = s.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    s = re.sub(r"<[^>]*>", "", s)
    return " ".join(s.split()).strip().lower()


def ass_time(t: str) -> int:
    """ASS 'H:MM:SS.cc' -> milliseconds."""
    h, m, rest = t.split(":")
    s, cs = rest.split(".")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(cs.ljust(2, "0")) * 10


def srt_time(t: str) -> int:
    """SRT 'HH:MM:SS,mmm' -> milliseconds."""
    hms, ms = re.split(r"[,.]", t)
    h, m, s = hms.split(":")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms.ljust(3, "0"))


def read_ass(path):
    """[(start_ms, end_ms, style, normalised_text)] for every Dialogue line."""
    fmt = None
    out = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            low = line.lower()
            if low.startswith("format:") and fmt is None and "text" in low:
                fmt = [c.strip().lower() for c in line.split(":", 1)[1].split(",")]
            elif low.startswith("dialogue:"):
                if not fmt:
                    fmt = ["layer", "start", "end", "style", "name", "marginl",
                           "marginr", "marginv", "effect", "text"]
                body = line.split(":", 1)[1]
                parts = body.split(",", len(fmt) - 1)
                if len(parts) < len(fmt):
                    continue
                rec = dict(zip(fmt, parts))
                text = rec.get("text", "")
                if _DRAW.search(text):
                    continue
                try:
                    out.append((ass_time(rec["start"].strip()),
                                ass_time(rec["end"].strip()),
                                rec.get("style", "").strip(),
                                norm(text)))
                except (ValueError, KeyError):
                    continue
    return out


def read_srt(path):
    """[(index_text, timing_line, [text lines])] preserving the original blocks."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        raw = f.read()
    blocks = []
    for chunk in re.split(r"\r?\n\s*\r?\n", raw.strip()):
        lines = chunk.splitlines()
        ti = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ti is None:
            continue
        blocks.append((lines[:ti], lines[ti], lines[ti + 1:]))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ass", required=True)
    ap.add_argument("--srt", default=None,
                    help="srt to mark; omit when only --emit-spans is wanted")
    ap.add_argument("--out", default=None, help="default: rewrite --srt in place")
    ap.add_argument("--emit-spans", default=None,
                    help="also write the non-spoken (start,end) spans as JSON. "
                         "Phase A uses these to keep OP/ED singing out of the "
                         "cast - see profile_show.py --exclude-spans.")
    ap.add_argument("--span-pad", type=float, default=0.25,
                    help="seconds added each side of an emitted span. Song cues "
                         "are timed to the lyric, not to the note, so a little "
                         "sung audio sits outside them; without the pad that "
                         "residue survives as a thin 'speaker'. Measured on "
                         "Gundam X: 0.25 removed 2 more phantom singers and cost "
                         "no real character, and larger pads gained nothing.")
    ap.add_argument("--non-spoken-styles", default="",
                    help="comma-separated ASS style names; empty = built-in "
                         "song/lyric pattern")
    ap.add_argument("--tolerance-ms", type=int, default=60,
                    help="timing slack when pairing an SRT cue to its ASS line")
    args = ap.parse_args()

    if args.non_spoken_styles.strip():
        wanted = {s.strip().lower() for s in args.non_spoken_styles.split(",")
                  if s.strip()}
        def is_non_spoken(style):
            return style.strip().lower() in wanted
        shown = ",".join(sorted(wanted))
    else:
        pat = re.compile(DEFAULT_NON_SPOKEN, re.I)
        def is_non_spoken(style):
            return bool(pat.match(style.strip()))
        shown = f"/{DEFAULT_NON_SPOKEN}/"

    try:
        ass = read_ass(args.ass)
        blocks = read_srt(args.srt) if args.srt else []
    except OSError as e:
        print(f"[styles] FATAL: {e}", file=sys.stderr)
        return 2

    styles_seen = sorted({a[2] for a in ass})
    targets = [a for a in ass if is_non_spoken(a[2])]
    hit_styles = sorted({a[2] for a in targets})
    print(f"[styles] ASS styles present: {', '.join(styles_seen) or '(none)'}")
    print(f"[styles] non-spoken rule: {shown} -> matches {', '.join(hit_styles) or '(none)'}")

    if args.emit_spans:
        import json
        p = max(0.0, args.span_pad)
        spans = merge_spans([(max(0.0, a[0] / 1000.0 - p), a[1] / 1000.0 + p)
                             for a in targets])
        with open(args.emit_spans, "w", encoding="utf-8") as f:
            json.dump(spans, f)
        total = sum(b - a for a, b in spans)
        print(f"[styles] {len(spans)} non-spoken span(s), {total:.1f}s "
              f"-> {args.emit_spans}")

    if not args.srt:
        return 0

    if not targets:
        print("[styles] nothing to mark; SRT left unchanged")
        if args.out and args.out != args.srt:
            with open(args.srt, encoding="utf-8-sig", errors="replace") as f:
                open(args.out, "w", encoding="utf-8").write(f.read())
        return 0

    marked = 0
    already = 0
    unmatched = list(targets)
    out_lines = []
    for pre, timing, text in blocks:
        m = re.search(r"([\d:,.]+)\s*-->\s*([\d:,.]+)", timing)
        body = "\n".join(text)
        if m:
            s_ms, e_ms = srt_time(m.group(1)), srt_time(m.group(2))
            n = norm(body)
            best = None
            for cand in unmatched:
                if abs(cand[0] - s_ms) <= args.tolerance_ms and \
                        abs(cand[1] - e_ms) <= args.tolerance_ms:
                    # prefer an exact text match when several share a timing
                    if cand[3] == n:
                        best = cand
                        break
                    if best is None:
                        best = cand
            if best is not None and body.strip():
                unmatched.remove(best)
                if MARK in body:
                    already += 1          # re-run over an already-marked srt
                else:
                    text = [MARK + " " + text[0]] + text[1:] if text else [MARK]
                    marked += 1
        out_lines.append("\n".join(pre + [timing] + text))

    dest = args.out or args.srt
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\n".join(out_lines) + "\n")

    extra = f", {already} already marked" if already else ""
    print(f"[styles] marked {marked} cue(s) as non-spoken{extra} -> {dest}")
    if unmatched:
        print(f"[styles] WARNING: {len(unmatched)} styled cue(s) had no matching SRT "
              f"cue within {args.tolerance_ms}ms and were NOT marked "
              f"(first: {unmatched[0][0]/1000:.2f}s style={unmatched[0][2]!r})",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
