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
_MUSIC_CUE = re.compile(r"[♪♫♬♩]")     # music notes = sung lyric
_TAG_ONLY_CUE = re.compile(r"^\s*[\[(][^\])]*[\])]\s*$")   # "[laughs]" / "(screams)"


# Cyrillic characters that are visually identical to a Latin one. OCR'd subs
# for foreign films are riddled with them: Solaris (1972)'s English track has
# "I соuIdn't see the sun" (Cyrillic s and o) and "FOR К. KELVIN" (Cyrillic Ka).
# They look right on screen but reach the TTS as Russian letters, so the line is
# either mispronounced or mangled by the phonemiser.
_CONFUSABLES = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I",
    "Ј": "J", "Ԁ": "D", "Ԍ": "G",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d", "ѵ": "v", "ѡ": "w",
})
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")
# The same OCR pass also swaps lowercase l for capital I ("I couIdn't see the
# sun"). Only fold when BOTH neighbours are lowercase, which leaves the word
# "I", "I'm" and "AI" alone but catches it mid-word.
_CAP_I_FOR_L = re.compile(r"(?<=[a-z])I(?=[a-z])")


def deconfuse(s):
    """Fold Cyrillic look-alikes back to Latin in otherwise-Latin text.

    Only fires when the line already contains Latin letters AND every Cyrillic
    character in it is a known look-alike - the signature of OCR noise. A
    genuinely Russian line (or one with any non-confusable Cyrillic, e.g. "П")
    is left completely alone, so real foreign text is never corrupted.
    """
    s = _CAP_I_FOR_L.sub("l", s)
    if not _CYRILLIC.search(s) or not _LATIN.search(s):
        return s
    if any(ord(ch) not in _CONFUSABLES for ch in s if _CYRILLIC.match(ch)):
        return s
    return s.translate(_CONFUSABLES)


def clean_text(s):
    s = _TAG.sub("", s)
    s = _HTML.sub("", s)
    s = s.replace("\\N", " ").replace("\\n", " ")
    return deconfuse(" ".join(s.split()).strip())


def load_cues(srt_path):
    import srt
    with open(srt_path, encoding="utf-8", errors="replace") as f:
        subs = list(srt.parse(f.read()))
    cues = []
    skipped_music = 0
    for s in subs:
        text = clean_text(s.content)
        if not text or not re.search(r"[A-Za-z0-9]", text):
            continue
        # Never SPEAK a song lyric or a bare sound tag. Lyric cues would be read
        # out flatly over the OP/ED, and "[laughs]" would be read literally.
        # build_music_bed.py applies the same rule, so these regions keep their
        # ORIGINAL audio (the real singing / the real laugh) instead.
        if _MUSIC_CUE.search(text) or _TAG_ONLY_CUE.match(text):
            skipped_music += 1
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

class LazyRefAudio:
    """The emotion reference, read from disk one cue-slice at a time.

    Only `len()` and `[a:b]` are ever used on it, so this is a drop-in for the
    numpy array this used to be - and it keeps the footprint independent of
    runtime. Reading it whole costs 1.77 GB for a 2h47m mono 44.1 kHz stem as
    float32 (the isolated vocal stem from Phase A), on top of the IndexTTS2
    weights. On Solaris (1972) that was enough to exhaust the Windows commit
    limit and abort the render on a *920 KiB* numpy allocation at cue ~10 -
    a failure that looks nothing like "out of memory" in the log.
    """

    def __init__(self, path):
        import soundfile as sf
        self._sf = sf
        self.path = path
        info = sf.info(path)
        self.samplerate = info.samplerate
        self._frames = int(info.frames)

    def __len__(self):
        return self._frames

    def __getitem__(self, key):
        import numpy as np
        if not isinstance(key, slice):
            raise TypeError("LazyRefAudio supports slicing only")
        a, b, step = key.indices(self._frames)
        if step != 1:
            raise ValueError("LazyRefAudio supports contiguous slices only")
        if b <= a:
            return np.zeros(0, dtype="float32")
        y, _ = self._sf.read(self.path, start=a, stop=b, dtype="float32",
                             always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y


def load_ref_audio(path):
    ref = LazyRefAudio(path)
    return ref, ref.samplerate


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
    ap.add_argument("--voice-tuning", default=None,
                    help="JSON of per-speaker + global knobs (emo_alpha, gain_db, "
                         "temperature, max_text_tokens_per_segment, seed, no_split_under, "
                         "target_peak). Overrides --emo-alpha per speaker.")
    ap.add_argument("--fallback-speaker-id", type=int, default=None,
                    help="speaker id whose voice covers un-matched cues "
                         "(default: the speaker with the most speech)")
    ap.add_argument("--fit", action="store_true",
                    help="time-compress a line that overruns the next cue start")
    ap.add_argument("--max-speed", type=float, default=1.6)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cue-cache", default=None,
                    help="keep each cue's raw TTS output here, keyed by everything that "
                         "shapes it, so a killed render (game-gate releasing the GPU) "
                         "resumes from the last finished cue instead of from cue 0. "
                         "Emptied after the dub WAV is written.")
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

    # --- tuning knobs (global + per-speaker) --------------------------------
    tuning = {}
    if args.voice_tuning and os.path.isfile(args.voice_tuning):
        with open(args.voice_tuning, encoding="utf-8") as f:
            tuning = json.load(f)
        eprint(f"[indextts] loaded voice tuning from {args.voice_tuning}")
    g = tuning.get("global", {})
    G_EMO     = float(g.get("emo_alpha", args.emo_alpha))
    G_TEMP    = float(g.get("temperature", 0.8))
    G_MAXSEG  = int(g.get("max_text_tokens_per_segment", 120))
    G_SEED    = int(g.get("seed", 0))
    G_NOSPLIT = int(g.get("no_split_under", 220))
    G_PEAK    = float(g.get("target_peak", 0.0))     # 0 = disabled; else true-peak ceiling
    # Match each dubbed line's loudness to the ORIGINAL line it replaces (measured
    # from the isolated-vocal stem), preserving the source mix's dynamics/intent
    # instead of leveling characters against each other.
    G_MATCH   = bool(g.get("match_source", False))
    M_FLOOR   = 10.0 ** (float(g.get("match_floor_db",      -38.0)) / 20.0)  # below = treat as no dialogue
    M_DEFAULT = 10.0 ** (float(g.get("match_default_rms_db",-20.0)) / 20.0)  # fallback level when original is silent
    M_GMIN    = 10.0 ** (float(g.get("match_min_gain_db",   -12.0)) / 20.0)  # clamp match gain
    M_GMAX    = 10.0 ** (float(g.get("match_max_gain_db",    12.0)) / 20.0)
    spk_tune = {int(k): v for k, v in (tuning.get("speakers", {}) or {}).items()}
    def emo_for(sid):  return float(spk_tune.get(sid, {}).get("emo_alpha", G_EMO))
    def gain_for(sid): return 10.0 ** (float(spk_tune.get(sid, {}).get("gain_db", 0.0)) / 20.0)
    if G_SEED:
        try:
            import torch
            _torch = torch
        except Exception:
            _torch = None
    else:
        _torch = None

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

    # timbre prompt per speaker: the first reference clip that exists on disk,
    # falling back to the fallback speaker. The profile's refs[0] is not always
    # present (some _0.wav clips were pruned), so a blind refs[0] silently
    # dropped every cue for that speaker (e.g. spk1/Judau). Pick the first file
    # that is actually on disk instead.
    def timbre_for(sid):
        sp = by_id.get(sid) or by_id[fb_id]
        for refs in (sp["_refs_abs"], by_id[fb_id]["_refs_abs"]):
            for r in refs:
                if os.path.isfile(r):
                    return r
        return None

    # --- load IndexTTS2 -------------------------------------------------------
    try:
        from indextts.infer_v2 import IndexTTS2
    except ModuleNotFoundError as e:
        eprint(f"[indextts] FATAL: cannot import IndexTTS2 ({e}). "
               f"Run this under the IndexTTS2 venv.")
        return 3
    # The model is loaded on the first cue the cache cannot supply, so a resumed
    # render whose cues are all cached never touches the GPU.
    tts = None

    def get_tts():
        nonlocal tts
        if tts is None:
            eprint(f"[indextts] loading IndexTTS2 from {args.checkpoints_dir} "
                   f"(fp16={args.fp16}) ...")
            tts = IndexTTS2(cfg_path=cfg, model_dir=args.checkpoints_dir,
                            use_fp16=args.fp16, use_deepspeed=False,
                            use_cuda_kernel=False)
        return tts

    # --- per-cue resume cache ---------------------------------------------------
    # Key = every input that shapes a cue's RAW TTS output. Gain / fit / placement
    # are applied after the cache, so tuning those never invalidates it.
    cache = args.cue_cache
    if cache:
        import hashlib
        os.makedirs(cache, exist_ok=True)
        try:
            ck_sig = sorted((f, os.path.getsize(os.path.join(args.checkpoints_dir, f)))
                            for f in os.listdir(args.checkpoints_dir))
        except OSError:
            ck_sig = []
        reused = 0

    def cue_key(i, c, timbre, ce, parts):
        try:
            ts = (os.path.getsize(timbre), int(os.path.getmtime(timbre)))
        except OSError:
            ts = None
        blob = json.dumps({
            "i": i, "parts": parts, "timbre": os.path.abspath(timbre), "ts": ts,
            "emo": None if (args.no_emotion or ce <= 0) else
                   [os.path.basename(args.ref_audio), round(c["start"], 3), round(c["end"], 3), ce],
            "seed": (G_SEED + i) if _torch is not None else None, "temp": G_TEMP,
            "maxseg": G_MAXSEG, "fp16": bool(args.fp16), "ck": ck_sig,
        }, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

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
        ce = emo_for(c["speaker"])
        parts = split_long(c["text"], limit=G_NOSPLIT)
        cpath = None
        if cache:
            cpath = os.path.join(cache, f"cue_{i:04d}_{cue_key(i, c, timbre, ce, parts)}.npy")
        if cpath and os.path.isfile(cpath):
            try:
                clip = np.load(cpath)
                reused += 1
            except (OSError, ValueError):
                clip = None
        else:
            clip = None
        emo = None
        if clip is None and not args.no_emotion and ce > 0:
            emo = emotion_clip(ref, ref_sr, c["start"], c["end"], work, i)

        # loudness of the ORIGINAL line under this cue (from the isolated vocal
        # stem) -> used to scale the dub so it sits where the original sat.
        orig_rms = 0.0
        if G_MATCH:
            oa = max(0, int(c["start"] * ref_sr)); ob = min(len(ref), int(c["end"] * ref_sr))
            if ob > oa:
                seg = ref[oa:ob]
                orig_rms = float(np.sqrt(np.mean(seg ** 2)))

        if clip is None:
            # fixed seed per cue: reproducible, and keeps any split pieces consistent
            # (prevents the voice drifting mid-line)
            if _torch is not None:
                _torch.manual_seed(G_SEED + i)

            pieces = []
            failed = False
            for j, part in enumerate(parts):
                tmp = os.path.join(work, f"cue_{i:04d}_{j}.wav")
                try:
                    get_tts().infer(spk_audio_prompt=timbre, text=part, output_path=tmp,
                                    emo_audio_prompt=emo, emo_alpha=ce,
                                    verbose=False,
                                    max_text_tokens_per_segment=G_MAXSEG,
                                    temperature=G_TEMP)
                except Exception as e:
                    eprint(f"[indextts]   cue {i} piece {j} failed: {e}")
                    failed = True
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
            # only a cue whose every piece rendered is cached; written to a temp
            # name and renamed, so a kill mid-write never leaves a torn cue behind
            if cpath and not failed:
                tmpc = cpath + ".part.npy"
                np.save(tmpc, clip.astype(np.float32, copy=False))
                os.replace(tmpc, cpath)

        # match this line's loudness to the ORIGINAL line it replaces (retains the
        # source mix's dynamics/intent), rather than leveling characters equally.
        if G_MATCH and clip.size:
            cr = float(np.sqrt(np.mean(clip ** 2)))
            if cr > 1e-6:
                tgt = orig_rms if orig_rms >= M_FLOOR else M_DEFAULT
                gm = min(max(tgt / cr, M_GMIN), M_GMAX)
                clip = clip * gm
        # optional per-character trim ON TOP of the match (default 0 dB)
        gl = gain_for(c["speaker"])
        if gl != 1.0:
            clip = clip * gl
        # true-peak ceiling (anti-clip; never flattens dynamics, only caps overs)
        if G_PEAK > 0.0 and clip.size:
            pk = float(np.max(np.abs(clip)))
            if pk > G_PEAK:
                clip = clip * (G_PEAK / pk)

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
    if cache:
        eprint(f"[indextts] cue cache: {reused}/{n} cues resumed from {cache}")

    try:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    except Exception:
        pass
    # The render is complete and on disk, so the resume cache has done its job:
    # remove only the cue files this script writes, then the folder if now empty.
    if cache:
        import glob
        for f in glob.glob(os.path.join(cache, "cue_[0-9][0-9][0-9][0-9]_*.npy")):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(cache)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
