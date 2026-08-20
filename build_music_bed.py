#!/usr/bin/env python3
"""Build the Phase B bed as  no_vocals + (vocals outside dialogue).

Why this exists
---------------
The bed used to be Demucs' `no_vocals.wav` alone, which deletes the ENTIRE vocal
stem for the whole runtime. The dub only puts sound back where a subtitle cue
exists, so everything vocal-but-not-dialogue was silently lost:

  * laughing, screaming, grunts, effort sounds - never subtitled, so never re-dubbed
  * SFX with breath/formant content - htdemucs is a MUSIC model with no SFX class,
    so screams, crowd noise and some whooshes get classified as "vocals"
  * sung vocals in an OP/ED - the instrumental survived in no_vocals, the singing
    did not, leaving a karaoke-sounding bed

Fix: keep the vocal stem everywhere EXCEPT inside dialogue cues. Inside a cue the
original voice is still fully removed (that is the whole point of the music bed -
no original-language bleed under the English), but outside cues the original vocal
track passes through untouched, in the original actor's voice.

Lyric cues (music notes) and bare sound tags like "[laughs]" are NOT treated as
dialogue: they are not dubbed, so their audio must survive.

Streams in blocks - a 2h stereo 44.1k stem is ~2.6 GB as float32, and holding two
of them at once is what produced "numpy: Unable to allocate ..." in Phase A.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import soundfile as sf

BLOCK = 1 << 20                      # ~23.8 s per block at 44.1 kHz

_MUSIC = re.compile(r"[♪♫♬♩]")        # music notes
_TAG_ONLY = re.compile(r"^\s*[\[(][^\])]*[\])]\s*$")       # "[laughs]" / "(screams)"
_TS = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{1,3})")


def _sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def dialogue_spans(srt_path):
    """(start,end) seconds for cues that WILL be spoken by the dub.

    Skips music/lyric cues and bare sound tags so their original audio is kept
    rather than muted - dubbing a song or speaking "[laughs]" is not wanted.
    """
    with open(srt_path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    spans, skipped = [], 0
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = block.splitlines()
        m = None
        for i, ln in enumerate(lines):
            m = _TS.search(ln)
            if m:
                text = " ".join(lines[i + 1:]).strip()
                break
        if not m:
            continue
        text_clean = re.sub(r"<[^>]*>", "", text or "")
        if not text_clean.strip():
            continue
        if _MUSIC.search(text_clean) or _TAG_ONLY.match(text_clean):
            skipped += 1
            continue
        if not re.search(r"[A-Za-z0-9]", text_clean):
            continue
        a = _sec(*m.group(1, 2, 3, 4))
        b = _sec(*m.group(5, 6, 7, 8))
        if b > a:
            spans.append((a, b))
    spans.sort()
    return spans, skipped


def merge(spans, pad, fade):
    """Pad each span, then merge any that would overlap once fades are applied."""
    out = []
    for a, b in spans:
        a, b = a - pad, b + pad
        if out and a - out[-1][1] < 2 * fade:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([max(0.0, a), b])
    return [(a, b) for a, b in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vocals", required=True)
    ap.add_argument("--vocals", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=0.12,
                    help="seconds of extra mute each side of a cue")
    ap.add_argument("--fade", type=float, default=0.03,
                    help="seconds of ramp into/out of each muted region")
    ap.add_argument("--residual-gain", type=float, default=1.0,
                    help="level of the kept non-dialogue vocal audio (1.0 = as-is)")
    args = ap.parse_args()

    nv = sf.SoundFile(args.no_vocals)
    vo = sf.SoundFile(args.vocals)
    if nv.samplerate != vo.samplerate:
        print(f"[bed] FATAL: sample-rate mismatch {nv.samplerate} vs {vo.samplerate}", file=sys.stderr)
        return 2
    sr = nv.samplerate
    ch = nv.channels
    n = min(len(nv), len(vo))

    spans, skipped = dialogue_spans(args.srt)
    if not spans:
        print("[bed] WARNING: no dialogue cues parsed - bed would keep ALL original "
              "vocals, which means original-language dialogue under the dub. "
              "Falling back to no_vocals only.", file=sys.stderr)
    merged = merge(spans, args.pad, args.fade)
    fade_n = max(1, int(args.fade * sr))

    iv = [(int(a * sr), int(b * sr)) for a, b in merged]
    muted = sum(b - a for a, b in iv)
    print(f"[bed] {len(spans)} dialogue cue(s) -> {len(iv)} muted region(s); "
          f"{skipped} lyric/sound-tag cue(s) kept as original audio")
    print(f"[bed] muting {muted / sr / 60:.1f} min of {n / sr / 60:.1f} min "
          f"({100.0 * muted / max(n, 1):.1f}%); the rest of the vocal stem is retained")

    out = sf.SoundFile(args.out, mode="w", samplerate=sr, channels=ch, subtype="PCM_16")
    peak = 0.0
    pos = 0
    k = 0
    with nv, vo, out:
        while pos < n:
            cnt = min(BLOCK, n - pos)
            a = nv.read(cnt, dtype="float32", always_2d=True)
            b = vo.read(cnt, dtype="float32", always_2d=True)
            if a.shape[0] == 0 or b.shape[0] == 0:
                break
            cnt = min(a.shape[0], b.shape[0])
            a, b = a[:cnt], b[:cnt]

            # gain envelope for the vocal stem: 1 = keep, 0 = dialogue (muted)
            g = np.ones(cnt, dtype=np.float32)
            if spans:
                while k > 0 and iv[k - 1][1] > pos:
                    k -= 1
                j = k
                while j < len(iv) and iv[j][0] < pos + cnt:
                    s, e = iv[j]
                    lo, hi = max(s, pos), min(e, pos + cnt)
                    if hi > lo:
                        g[lo - pos:hi - pos] = 0.0
                    # ramp down into the region, ramp up out of it
                    fs, fe = s - fade_n, s
                    lo, hi = max(fs, pos), min(fe, pos + cnt)
                    if hi > lo:
                        t = (np.arange(lo, hi, dtype=np.float32) - fs) / fade_n
                        g[lo - pos:hi - pos] = np.minimum(g[lo - pos:hi - pos], 1.0 - t)
                    fs, fe = e, e + fade_n
                    lo, hi = max(fs, pos), min(fe, pos + cnt)
                    if hi > lo:
                        t = (np.arange(lo, hi, dtype=np.float32) - fs) / fade_n
                        g[lo - pos:hi - pos] = np.minimum(g[lo - pos:hi - pos], t)
                    if e <= pos + cnt:
                        j += 1
                        k = j
                    else:
                        break

            mix = a + b * g[:, None] * args.residual_gain
            peak = max(peak, float(np.max(np.abs(mix))) if mix.size else 0.0)
            out.write(mix)
            pos += cnt

    print(f"[bed] wrote {args.out}  peak={peak:.3f}")
    if peak > 0.999:
        print("[bed] NOTE: bed peaks at full scale; the mux stage still applies its "
              "own level control, but consider --residual-gain < 1.0 if it distorts",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
