#!/usr/bin/env python3
# srt_to_speech_multivoice.py
# -----------------------------------------------------------------------------
# Multi-voice, character-tracking variant of srt_to_speech.py.
#
# Same job (SRT -> one timed English dub WAV, lines placed at their cue times),
# but each subtitle line is spoken by the voice of the CHARACTER who says it,
# and characters are tracked ACROSS EPISODES so a recurring character keeps the
# same voice through the whole show.
#
# How character tracking works
# ----------------------------
# The target folder is one show, so this script keeps a persistent profile of
# the show's characters at --profile (a JSON file the PowerShell wrapper points
# at the show folder). It is invoked once PER EPISODE; each run:
#
#   1. loads the show profile (list of known characters, or empty on episode 1),
#   2. for every subtitle cue, slices the ORIGINAL (Japanese) audio under it and
#      computes a voice fingerprint (resemblyzer d-vector) + median pitch,
#   3. matches each cue to the nearest known character by cosine similarity
#      (>= --match-threshold), or mints a NEW character if it matches no one,
#   4. after seeing the whole episode, assigns each NEW character a pitch bucket
#      (adult/child x male/female) from its aggregated pitch and a DISTINCT voice
#      from that bucket's pool - known characters keep their locked-in voice,
#   5. synthesizes every line in its character's voice,
#   6. updates the characters' fingerprints/counts and saves the profile back.
#
# Because the profile is shared and episodes are processed in name order,
# "character #4" is the same person in episode 1 and episode 25, so they get the
# same voice every time. Distinct same-gender characters get DIFFERENT voices
# from the pool (cycling if the show has more characters than pool voices).
#
# Fingerprints come from resemblyzer (a small pretrained speaker encoder). If it
# is not installed, the script degrades gracefully to stateless per-line pitch
# bucketing (4 voices, no cross-episode identity) and says so.
#
# HONEST LIMITATIONS
#   * Online nearest-centroid matching is order-dependent and imperfect: two
#     similar voices can merge, or one actor doing two roles can split/merge.
#     Tune --match-threshold (higher = more distinct characters, more splits).
#   * A cue with two people talking, or only music/SFX, gets a blended or absent
#     fingerprint - those fall back to the pitch bucket's primary voice and are
#     not tracked as characters.
#   * Child male vs child female is only approximate (pre-pubescent pitch
#     overlaps); it only affects which child voice/pool a NEW child character is
#     assigned. Requires the reference audio to line up with the subtitle timing.
#
# TTS engine + base install: identical to srt_to_speech.py (Coqui XTTS-v2).
# Extra dependency for character tracking:  pip install resemblyzer
# -----------------------------------------------------------------------------

import argparse
import json
import os
import re
import sys
import tempfile

os.environ.setdefault("COQUI_TOS_AGREED", "1")

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
MAX_CHARS = 240

# Distinct XTTS-v2 built-in speakers per bucket. New characters are handed the
# first unused voice from their bucket's pool (then cycled if exhausted).
# "Damien Black" is first in the adult-male pool so it matches the basic script.
VOICE_POOLS = {
    "adult_male":   ["Damien Black", "Viktor Eka", "Baldur Sanjin",
                     "Craig Gutsy", "Aaron Dreschner", "Marcos Rudaski"],
    "adult_female": ["Alison Dietlinde", "Sofia Hellen", "Ana Florence",
                     "Gracie Wise", "Daisy Studious", "Brenda Stern"],
    "child_male":   ["Andrew Chipper", "Craig Gutsy"],
    "child_female": ["Tammie Ema", "Gracie Wise"],
}
BUCKET_LABELS = {
    "adult_male":   "adult male",
    "adult_female": "adult female",
    "child_male":   "child male",
    "child_female": "child female",
    "default":      "default/unclear",
}
CHILD_BUCKETS = {"child_male", "child_female"}
PROFILE_VERSION = 1


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
    """Median voiced F0 (Hz) of a mono torch segment, or None if no clear voice.

    Loud frames (> 15% of peak RMS) inside the plausible speech range only, so
    silence/music/SFX don't drag the estimate.
    """
    import torch
    import torchaudio.functional as taf

    if seg.dim() > 1:
        seg = seg.mean(dim=0)
    if seg.numel() < int(0.15 * sr):
        return None
    try:
        pitch = taf.detect_pitch_frequency(seg.unsqueeze(0), sr).squeeze(0)
    except Exception:
        return None
    if pitch.numel() == 0:
        return None

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


def classify_bucket(f0, male_max, adult_female_max, child_split):
    """Map an F0 (Hz) to a pitch bucket. Defaults to adult male if unknown."""
    if f0 is None:
        return "adult_male"
    if f0 < male_max:
        return "adult_male"
    if f0 < adult_female_max:
        return "adult_female"
    if f0 < child_split:
        return "child_male"
    return "child_female"


def pitch_shift_semitones(seg, semitones):
    """Shift a pydub segment up by N semitones (tape-speed style, no new deps)."""
    if not semitones:
        return seg
    new_rate = int(seg.frame_rate * (2.0 ** (semitones / 12.0)))
    shifted = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate})
    return shifted.set_frame_rate(seg.frame_rate)


# --- persistent character profile --------------------------------------------

def load_profile(path):
    """Return (characters, next_id). characters carry numpy centroids."""
    import numpy as np
    if not path or not os.path.exists(path):
        return [], 1
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[tts] WARNING: could not read profile {path}: {e}", file=sys.stderr)
        return [], 1
    chars = []
    for c in data.get("characters", []):
        chars.append({
            "id": int(c["id"]),
            "bucket": c["bucket"],
            "voice": c["voice"],
            "centroid": np.asarray(c["centroid"], dtype=np.float64),
            "count": int(c.get("count", 1)),
            "pitch": c.get("pitch"),
            "pitch_samples": [],   # transient: this episode's samples
            "is_new": False,
            "lines": 0,            # transient: lines in this episode
        })
    next_id = max((c["id"] for c in chars), default=0) + 1
    return chars, next_id


def save_profile(path, chars):
    if not path:
        return
    payload = {
        "version": PROFILE_VERSION,
        "characters": [{
            "id": c["id"],
            "bucket": c["bucket"],
            "voice": c["voice"],
            "centroid": [round(float(x), 6) for x in c["centroid"]],
            "count": int(c["count"]),
            "pitch": (round(float(c["pitch"]), 1) if c["pitch"] else None),
        } for c in chars if c["bucket"] and c["voice"]],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def cosine(a, b):
    import numpy as np
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def _shim_scipy_morphology():
    """resemblyzer 0.1.4 imports scipy.ndimage.morphology, which SciPy >=1.14
    removed. Re-expose it from scipy.ndimage so the import keeps working."""
    import sys
    import types
    if "scipy.ndimage.morphology" in sys.modules:
        return
    try:
        import scipy.ndimage as ndi
    except ModuleNotFoundError:
        return
    if not hasattr(ndi, "binary_dilation"):
        return
    mod = types.ModuleType("scipy.ndimage.morphology")
    mod.binary_dilation = ndi.binary_dilation
    sys.modules["scipy.ndimage.morphology"] = mod


def make_encoder(device):
    """Load resemblyzer's speaker encoder, or None if unavailable."""
    try:
        _shim_scipy_morphology()
        from resemblyzer import VoiceEncoder
    except ModuleNotFoundError:
        print("[tts] WARNING: resemblyzer not installed - no cross-episode "
              "character tracking. Falling back to per-line pitch buckets.",
              file=sys.stderr)
        print("[tts]          enable it with:  pip install resemblyzer",
              file=sys.stderr)
        return None
    try:
        return VoiceEncoder(device=device, verbose=False)
    except Exception as e:
        print(f"[tts] WARNING: could not init speaker encoder ({e}); "
              "falling back to per-line pitch buckets.", file=sys.stderr)
        return None


def embed_segment(encoder, seg_np, sr):
    """d-vector for a mono numpy segment, or None if too short / silent."""
    if encoder is None or seg_np.size < int(0.4 * sr):
        return None
    try:
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(seg_np, source_sr=sr)
        if wav.size < int(0.3 * sr):     # VAD trimmed it to ~nothing
            return None
        return encoder.embed_utterance(wav)
    except Exception:
        return None


def parse_pool(cli_value, bucket):
    """User override ('A;B;C') or the built-in pool for a bucket."""
    if cli_value:
        names = [n.strip() for n in cli_value.split(";") if n.strip()]
        if names:
            return names
    return VOICE_POOLS[bucket]


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser(description="SRT -> timed character-voiced English TTS WAV")
    ap.add_argument("--srt", required=True, help="input .srt path")
    ap.add_argument("--out", required=True, help="output .wav path")
    ap.add_argument("--ref-audio", required=True,
                    help="mono WAV of the ORIGINAL speech track, aligned to the "
                         "subtitle timing, used to identify each line's speaker")
    ap.add_argument("--profile", default=None,
                    help="show-level JSON of tracked characters (persists across "
                         "episodes); omit to disable cross-episode tracking")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="video duration in seconds (pads the track to length)")
    ap.add_argument("--language", default="en")

    # Per-bucket voice pools (';'-separated XTTS built-in speaker names).
    ap.add_argument("--voices-adult-male", default=None)
    ap.add_argument("--voices-adult-female", default=None)
    ap.add_argument("--voices-child-male", default=None)
    ap.add_argument("--voices-child-female", default=None)
    ap.add_argument("--voice-default", default=None,
                    help="voice for cues with no clear speaker (default: first "
                         "adult-male pool voice)")

    # Character matching + classification.
    ap.add_argument("--match-threshold", type=float, default=0.75,
                    help="cosine similarity to treat a cue as an existing "
                         "character (higher = more distinct characters)")
    ap.add_argument("--male-max", type=float, default=155.0)
    ap.add_argument("--adult-female-max", type=float, default=250.0)
    ap.add_argument("--child-split", type=float, default=300.0)
    ap.add_argument("--child-pitch-shift", type=float, default=2.0,
                    help="semitones to raise child-voice clips (0 to disable)")
    ap.add_argument("--embed-device", default=None, help="cuda|cpu (auto)")

    ap.add_argument("--fit", action="store_true",
                    help="speed up lines that overrun into the next cue")
    ap.add_argument("--max-speed", type=float, default=1.6)
    ap.add_argument("--verbose", action="store_true",
                    help="log every line, its character and chosen voice")
    args = ap.parse_args()

    pools = {
        "adult_male":   parse_pool(args.voices_adult_male, "adult_male"),
        "adult_female": parse_pool(args.voices_adult_female, "adult_female"),
        "child_male":   parse_pool(args.voices_child_male, "child_male"),
        "child_female": parse_pool(args.voices_child_female, "child_female"),
    }
    default_voice = args.voice_default or pools["adult_male"][0]

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
        print("      pip install coqui-tts pydub srt resemblyzer", file=sys.stderr)
        print("      pip install torch torchaudio "
              "--index-url https://download.pytorch.org/whl/cu128", file=sys.stderr)
        return 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[tts] device=cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("[tts] WARNING: CUDA not available, running on CPU (very slow).",
              file=sys.stderr)

    encoder = make_encoder(args.embed_device or device)
    tracking = encoder is not None and bool(args.profile)

    print(f"[tts] loading model {XTTS_MODEL} ...")
    tts = TTS(XTTS_MODEL).to(device)

    # --- load reference audio + profile ---------------------------------------
    print(f"[tts] loading reference audio {args.ref_audio} ...")
    try:
        ref_wave, ref_sr = torchaudio.load(args.ref_audio)
    except Exception as e:
        print(f"[tts] could not load reference audio: {e}", file=sys.stderr)
        return 4
    ref = ref_wave.mean(dim=0)
    if ref_sr != 16000:
        ref = torchaudio.functional.resample(ref, ref_sr, 16000)
        ref_sr = 16000
    ref_np = ref.detach().cpu().numpy().astype("float32")

    chars, next_id = load_profile(args.profile) if tracking else ([], 1)
    if tracking:
        print(f"[tts] character tracking ON  profile={args.profile}  "
              f"known={len(chars)}  match>={args.match_threshold}")
    else:
        print("[tts] character tracking OFF (per-line pitch buckets only)")

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

    # --- pass 1: identify the speaker of every cue ----------------------------
    print(f"[tts] pass 1/2: identifying speakers across {len(cues)} cue(s)...")
    fallback_lines = 0
    for c in cues:
        a = int(c["start"] * ref_sr)
        b = min(int(c["end"] * ref_sr), ref_np.size)
        seg_t = ref[a:b] if b > a else ref[0:0]
        f0 = voiced_median_pitch(seg_t, ref_sr) if b > a else None
        emb = embed_segment(encoder, ref_np[a:b], ref_sr) if b > a else None
        c["f0"] = f0

        if emb is None:
            # No usable fingerprint: not a tracked character, pitch bucket only.
            c["char"] = None
            c["bucket"] = classify_bucket(f0, args.male_max,
                                          args.adult_female_max, args.child_split)
            fallback_lines += 1
            continue

        # Nearest known/created character by cosine similarity.
        best, best_sim = None, -1.0
        for ch in chars:
            sim = cosine(emb, ch["centroid"])
            if sim > best_sim:
                best_sim, best = sim, ch

        if best is not None and best_sim >= args.match_threshold:
            # Online-update the centroid so later cues match more tightly.
            best["centroid"] = (best["centroid"] * best["count"] + emb) / (best["count"] + 1)
            best["count"] += 1
            best["lines"] += 1
            if f0:
                best["pitch_samples"].append(f0)
            c["char"] = best
        else:
            new = {"id": next_id, "bucket": None, "voice": None,
                   "centroid": np.asarray(emb, dtype=np.float64), "count": 1,
                   "pitch": None, "pitch_samples": ([f0] if f0 else []),
                   "is_new": True, "lines": 1}
            next_id += 1
            chars.append(new)
            c["char"] = new

    # Assign a bucket + distinct voice to each NEW character (known ones keep
    # their locked-in voice). Bucket comes from the character's aggregate pitch,
    # not any single noisy line.
    used = {b: set() for b in pools}
    for ch in chars:
        if not ch["is_new"] and ch["voice"]:
            used.setdefault(ch["bucket"], set()).add(ch["voice"])
    for ch in chars:
        if not ch["is_new"]:
            # Refine stored pitch lightly for reporting/future tuning.
            if ch["pitch_samples"]:
                med = float(np.median(ch["pitch_samples"]))
                ch["pitch"] = med if ch["pitch"] is None else 0.7 * ch["pitch"] + 0.3 * med
            continue
        med = float(np.median(ch["pitch_samples"])) if ch["pitch_samples"] else None
        bucket = classify_bucket(med, args.male_max, args.adult_female_max, args.child_split)
        pool = pools[bucket]
        chosen = next((v for v in pool if v not in used[bucket]), None)
        if chosen is None:                      # pool exhausted - cycle
            chosen = pool[len(used[bucket]) % len(pool)]
            print(f"[tts]   note: reusing voice '{chosen}' for a new "
                  f"{BUCKET_LABELS[bucket]} character (pool exhausted)")
        used[bucket].add(chosen)
        ch["bucket"], ch["voice"], ch["pitch"] = bucket, chosen, med

    # Resolve each cue's actual voice now that characters are assigned.
    for c in cues:
        if c["char"] is not None:
            c["voice"] = c["char"]["voice"]
            c["bucket"] = c["char"]["bucket"]
        else:
            c["voice"] = default_voice   # untracked fallback line
    active = [c for c in cues]
    seen_chars = [ch for ch in chars if ch["lines"] > 0]
    new_here = sum(1 for ch in seen_chars if ch["is_new"])
    print(f"[tts] speakers this episode: {len(seen_chars)} "
          f"({new_here} new) + {fallback_lines} untracked line(s)")
    if args.verbose:
        for ch in sorted(seen_chars, key=lambda x: -x["lines"]):
            tag = "NEW" if ch["is_new"] else "   "
            pit = f"{ch['pitch']:5.0f}Hz" if ch["pitch"] else "  --  "
            print(f"[tts]   {tag} char#{ch['id']:<3} {BUCKET_LABELS[ch['bucket']]:<13} "
                  f"{pit}  {ch['voice']:<18} {ch['lines']} line(s)")

    # --- pass 2: synthesize ---------------------------------------------------
    sr = 24000
    last_end_ms = max(c["end"] for c in active) * 1000.0
    total_ms = int(max(args.duration * 1000.0, last_end_ms)) + 2000
    print(f"[tts] pass 2/2: synthesizing {len(active)} line(s), "
          f"building a {total_ms / 1000:.0f}s track")
    canvas = AudioSegment.silent(duration=total_ms, frame_rate=sr)

    xtts_kw = dict(temperature=0.7, length_penalty=1.0, repetition_penalty=5.0,
                   top_k=50, top_p=0.85, enable_text_splitting=True)

    def synth(text, path, speaker):
        try:
            tts.tts_to_file(text=text, language=args.language, file_path=path,
                            speaker=speaker, **xtts_kw)
        except TypeError:
            tts.tts_to_file(text=text, language=args.language, file_path=path,
                            speaker=speaker)

    tmp = tempfile.mkdtemp(prefix="dub_")
    n = len(active)
    for i, c in enumerate(active):
        text, bucket, voice = c["text"], c["bucket"], c["voice"]

        clip = AudioSegment.empty()
        for j, piece in enumerate(split_long(text)):
            wav_path = os.path.join(tmp, f"{i}_{j}.wav")
            synth(piece, wav_path, voice)
            clip += AudioSegment.from_file(wav_path)

        if bucket in CHILD_BUCKETS and args.child_pitch_shift:
            clip = pitch_shift_semitones(clip, args.child_pitch_shift)

        start_ms = int(c["start"] * 1000)
        fitted = ""
        if args.fit and i + 1 < n:
            gap = int(active[i + 1]["start"] * 1000) - start_ms
            if gap > 0 and len(clip) > gap:
                factor = min(len(clip) / gap, args.max_speed)
                if factor > 1.01:
                    clip = speedup(clip, playback_speed=factor)
                    fitted = f" (sped up {factor:.2f}x to fit)"

        canvas = canvas.overlay(clip, position=start_ms)

        if args.verbose:
            t = c["start"]
            ts = f"{int(t//3600):02d}:{int(t//60%60):02d}:{int(t%60):02d}"
            who = f"char#{c['char']['id']}" if c["char"] else "untracked"
            preview = text if len(text) <= 40 else text[:37] + "..."
            print(f"[tts] {i + 1:>4}/{n} [{ts}] {who:<10} {voice:<16} "
                  f"{len(clip) / 1000:4.1f}s {preview}{fitted}", flush=True)
        elif (i + 1) % 25 == 0 or i + 1 == n:
            print(f"[tts] {i + 1}/{n} lines", flush=True)

    canvas.export(args.out, format="wav")
    print(f"[tts] wrote {args.out}")

    if tracking:
        save_profile(args.profile, chars)
        print(f"[tts] saved profile: {len([c for c in chars if c['voice']])} "
              f"character(s) -> {args.profile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
