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
#      computes a voice fingerprint (resemblyzer d-vector),
#   3. matches each cue to the nearest known character by cosine similarity
#      (>= --match-threshold), or mints a NEW character if it matches no one,
#   4. after seeing the whole episode, classifies each NEW character's GENDER +
#      AGE from its aggregated original speech and picks a DISTINCT voice from
#      that gender/age bucket's pool - known characters keep their locked voice,
#   5. synthesizes every line in its character's voice,
#   6. updates the characters' fingerprints/counts and saves the profile back.
#
# Because the profile is shared and episodes are processed in name order,
# "character #4" is the same person in episode 1 and episode 25, so they get the
# same voice every time. Distinct same-bucket characters get DIFFERENT voices
# from the pool (cycling if the show has more characters than pool voices).
#
# Gender/age matching (the important bit)
# ---------------------------------------
# GENDER and AGE are what decide which English voice a character gets, so they
# must be right. Earlier versions guessed both from median pitch (F0) alone with
# fixed Hz thresholds - unreliable, because timbre (not pitch) carries most of
# the gender/age percept, anime delivery swings F0 wildly, and pitch has no age
# axis at all (it can't tell an old man from a young one).
#
# This version runs a pretrained speech age+gender model on each new character's
# aggregated original audio:
#     audeering/wav2vec2-large-robust-24-ft-age-gender
# It outputs a continuous AGE (years) and a GENDER distribution {female, male,
# child}, from which we pick one of six buckets:
#     child_male / child_female / adult_male / adult_female
#     elderly_male / elderly_female                (age >= --elder-age)
# Child sub-gender (the model only has one "child" class) is split by the model's
# male-vs-female logits. The model needs no extra pip package (it runs on the
# transformers + torch stack already installed); it downloads ~1GB of weights to
# the HuggingFace cache on first use. Set HF_HOME to relocate that cache.
#
# Median pitch is kept only as a FALLBACK, used when the age-gender model is
# disabled (--no-age-gender), can't load (offline first run), or a character has
# too little clean speech to classify.
#
# Identity clustering still comes from resemblyzer (a small pretrained speaker
# encoder). If it is not installed, the script degrades to stateless per-cue
# classification (no cross-episode identity) and says so.
#
# NOTE: a character's voice is LOCKED when first minted. To re-cast an existing
# show with this improved classifier, DELETE the profile JSON and re-run.
#
# HONEST LIMITATIONS
#   * Online nearest-centroid matching is order-dependent and imperfect: two
#     similar voices can merge, or one actor doing two roles can split/merge.
#     Tune --match-threshold (higher = more distinct characters, more splits).
#   * The age-gender model was trained on real speech. Anime convention (adult
#     women voicing boys, stylised delivery) can still fool it; it is far better
#     than pitch but not infallible. Tune --elder-age per show if elders read
#     too young/old, and override any bucket's pool with --voices-*.
#   * A cue with two people talking, or only music/SFX, gets a blended or absent
#     fingerprint - those fall back to the default voice and are not tracked.
#   * Requires the reference audio to line up with the subtitle timing.
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

# Pretrained speech age+gender model (wav2vec2). Outputs continuous age (0-1 ->
# *100 years) and a 3-way gender distribution [female, male, child]. Runs on the
# transformers+torch stack already installed; weights download to the HF cache.
AGE_GENDER_MODEL = "audeering/wav2vec2-large-robust-24-ft-age-gender"

# Distinct XTTS-v2 built-in speakers per bucket. New characters are handed the
# first unused voice from their bucket's pool (then cycled if exhausted).
# "Damien Black" is first in the adult-male pool so it matches the basic script.
# Elderly pools reuse the maturest/gravest built-ins; a small downward pitch
# shift (--elder-pitch-shift) ages them further, since XTTS has no explicit
# "old" voices.
VOICE_POOLS = {
    "adult_male":     ["Damien Black", "Viktor Eka", "Baldur Sanjin",
                       "Craig Gutsy", "Aaron Dreschner", "Marcos Rudaski"],
    "adult_female":   ["Alison Dietlinde", "Sofia Hellen", "Ana Florence",
                       "Gracie Wise", "Daisy Studious", "Brenda Stern"],
    "elderly_male":   ["Baldur Sanjin", "Marcos Rudaski", "Damien Black"],
    "elderly_female": ["Brenda Stern", "Daisy Studious", "Ana Florence"],
    "child_male":     ["Andrew Chipper", "Craig Gutsy"],
    "child_female":   ["Tammie Ema", "Gracie Wise"],
}
BUCKET_LABELS = {
    "adult_male":     "adult male",
    "adult_female":   "adult female",
    "elderly_male":   "elderly male",
    "elderly_female": "elderly female",
    "child_male":     "child male",
    "child_female":   "child female",
    "default":        "default/unclear",
}
CHILD_BUCKETS = {"child_male", "child_female"}
ELDERLY_BUCKETS = {"elderly_male", "elderly_female"}
PROFILE_VERSION = 2

# Cap how much original audio per new character we feed the age-gender model.
# ~20s of clean speech is plenty for a stable estimate and keeps memory small.
AG_MAX_SECONDS = 20.0
CHILD_AGE_MAX = 13.0    # predicted age (years) below which we treat as a child


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
    silence/music/SFX don't drag the estimate. Used only as a FALLBACK when the
    age-gender model is unavailable.
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


def classify_bucket_pitch(f0, male_max, adult_female_max, child_split):
    """FALLBACK: map an F0 (Hz) to a bucket. No age axis, so never 'elderly'.
    Defaults to adult male if unknown."""
    if f0 is None:
        return "adult_male"
    if f0 < male_max:
        return "adult_male"
    if f0 < adult_female_max:
        return "adult_female"
    if f0 < child_split:
        return "child_male"
    return "child_female"


def classify_bucket_model(age, gender_probs, elder_age):
    """Map an age-gender prediction to a bucket.

    age          : predicted age in years (float) or None
    gender_probs : [p_female, p_male, p_child] or None
    """
    if gender_probs is None:
        return None
    p_female, p_male, p_child = (float(gender_probs[0]),
                                 float(gender_probs[1]),
                                 float(gender_probs[2]))
    masc = p_male >= p_female                       # child class carries no sex
    is_child = (p_child >= p_male and p_child >= p_female) or \
               (age is not None and age < CHILD_AGE_MAX)
    if is_child:
        return "child_male" if masc else "child_female"
    if age is not None and age >= elder_age:
        return "elderly_male" if masc else "elderly_female"
    return "adult_male" if masc else "adult_female"


def gender_label(gender_probs):
    """Human-readable dominant gender for reporting."""
    if gender_probs is None:
        return "?"
    return ["female", "male", "child"][int(max(range(3),
            key=lambda i: gender_probs[i]))]


def pitch_shift_semitones(seg, semitones):
    """Shift a pydub segment by N semitones (tape-speed style, no new deps).
    Positive raises pitch (younger), negative lowers it (older)."""
    if not semitones:
        return seg
    new_rate = int(seg.frame_rate * (2.0 ** (semitones / 12.0)))
    shifted = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate})
    return shifted.set_frame_rate(seg.frame_rate)


# --- age + gender classifier --------------------------------------------------

class AgeGenderPredictor:
    """Lazy wrapper around the audeering wav2vec2 age+gender model.

    predict(mono_16k_float_np) -> (age_years, gender_probs[3]) or (None, None).
    Returns (None, None) rather than raising if anything goes wrong, so the
    caller can fall back to pitch. Weights download to the HF cache on first use.
    """

    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.ok = None       # tri-state: None=untried, True/False after load

    def _build(self):
        import torch
        import torch.nn as nn
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model, Wav2Vec2PreTrainedModel)

        # Model definition per the audeering model card. Embedding it here means
        # we never execute remote code - transformers only fetches the weights.
        class ModelHead(nn.Module):
            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, x):
                x = self.dropout(x)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                return self.out_proj(x)

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.init_weights()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return (hidden, self.age(hidden),
                        torch.softmax(self.gender(hidden), dim=1))

        self._AgeGenderModel = AgeGenderModel

    def _load(self):
        if self.ok is not None:
            return self.ok
        try:
            self._build()
            model = self._AgeGenderModel.from_pretrained(self.model_name)
            model = model.to(self.device).eval()
            self.model = model
            self.ok = True
            print(f"[tts] age+gender model loaded: {self.model_name} "
                  f"(device={self.device})")
        except Exception as e:
            print(f"[tts] WARNING: could not load age+gender model "
                  f"({type(e).__name__}: {e}); falling back to pitch buckets "
                  f"for gender/age.", file=sys.stderr)
            self.ok = False
        return self.ok

    def predict(self, seg_np, sr):
        """(age_years, gender_probs) for a mono numpy segment, or (None, None)."""
        import numpy as np
        import torch
        if seg_np is None or seg_np.size < int(0.4 * sr):
            return None, None
        if not self._load():
            return None, None
        try:
            x = np.asarray(seg_np, dtype=np.float32)
            # Zero-mean, unit-variance normalisation (what the wav2vec2 feature
            # extractor does with do_normalize=True); avoids a processor dep.
            x = (x - x.mean()) / (np.sqrt(x.var()) + 1e-7)
            iv = torch.from_numpy(x).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, logits_age, probs_gender = self.model(iv)
            age = float(logits_age[0, 0]) * 100.0
            gender = probs_gender[0].detach().cpu().numpy()
            return age, gender
        except Exception as e:
            print(f"[tts] WARNING: age+gender inference failed ({e})",
                  file=sys.stderr)
            return None, None


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
            "age": c.get("age"),               # persisted model age (years)
            "gender": c.get("gender"),          # persisted dominant gender label
            "pitch_samples": [],   # transient: this episode's samples
            "ag_clips": [],        # transient: this episode's audio for AG model
            "ag_len": 0,           # transient: accumulated AG sample count
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
            "age": (round(float(c["age"]), 1) if c.get("age") else None),
            "gender": c.get("gender"),
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
              "character tracking. Falling back to per-cue classification.",
              file=sys.stderr)
        print("[tts]          enable it with:  pip install resemblyzer",
              file=sys.stderr)
        return None
    try:
        return VoiceEncoder(device=device, verbose=False)
    except Exception as e:
        print(f"[tts] WARNING: could not init speaker encoder ({e}); "
              "falling back to per-cue classification.", file=sys.stderr)
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
    ap.add_argument("--voices-elderly-male", default=None)
    ap.add_argument("--voices-elderly-female", default=None)
    ap.add_argument("--voices-child-male", default=None)
    ap.add_argument("--voices-child-female", default=None)
    ap.add_argument("--voice-default", default=None,
                    help="voice for cues with no clear speaker (default: first "
                         "adult-male pool voice)")

    # Character matching.
    ap.add_argument("--match-threshold", type=float, default=0.75,
                    help="cosine similarity to treat a cue as an existing "
                         "character (higher = more distinct characters)")

    # Age + gender classification (primary path).
    ap.add_argument("--age-gender-model", default=AGE_GENDER_MODEL,
                    help="HuggingFace id of the speech age+gender model")
    ap.add_argument("--no-age-gender", action="store_true",
                    help="disable the age+gender model; use pitch buckets only")
    ap.add_argument("--age-gender-device", default=None, help="cuda|cpu (auto)")
    ap.add_argument("--elder-age", type=float, default=58.0,
                    help="predicted age (years) at/above which a character is "
                         "bucketed elderly")
    ap.add_argument("--elder-pitch-shift", type=float, default=1.5,
                    help="semitones to LOWER elderly-voice clips to age them "
                         "(0 to disable)")

    # Pitch fallback thresholds (only used when the model is off/unavailable).
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
        "adult_male":     parse_pool(args.voices_adult_male, "adult_male"),
        "adult_female":   parse_pool(args.voices_adult_female, "adult_female"),
        "elderly_male":   parse_pool(args.voices_elderly_male, "elderly_male"),
        "elderly_female": parse_pool(args.voices_elderly_female, "elderly_female"),
        "child_male":     parse_pool(args.voices_child_male, "child_male"),
        "child_female":   parse_pool(args.voices_child_female, "child_female"),
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

    predictor = None
    if not args.no_age_gender:
        predictor = AgeGenderPredictor(args.age_gender_model,
                                       args.age_gender_device or device)

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
        print("[tts] character tracking OFF (per-cue classification only)")
    if predictor is not None:
        print(f"[tts] gender/age: age+gender model (elder>={args.elder_age:.0f}y); "
              f"pitch is fallback")
    else:
        print("[tts] gender/age: pitch buckets only (--no-age-gender)")

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

    def accumulate_ag_clip(ch, seg_np):
        """Stash a NEW character's speech for the age-gender model, capped."""
        cap = int(AG_MAX_SECONDS * ref_sr)
        if ch["ag_len"] >= cap:
            return
        ch["ag_clips"].append(seg_np)
        ch["ag_len"] += seg_np.size

    # --- pass 1: identify the speaker of every cue ----------------------------
    print(f"[tts] pass 1/2: identifying speakers across {len(cues)} cue(s)...")
    fallback_lines = 0
    for c in cues:
        a = int(c["start"] * ref_sr)
        b = min(int(c["end"] * ref_sr), ref_np.size)
        seg_t = ref[a:b] if b > a else ref[0:0]
        seg_np = ref_np[a:b] if b > a else ref_np[0:0]
        f0 = voiced_median_pitch(seg_t, ref_sr) if b > a else None
        emb = embed_segment(encoder, seg_np, ref_sr) if b > a else None
        c["f0"] = f0

        if emb is None:
            # No usable fingerprint: not a tracked character. Bucket it (pitch
            # fallback - it only drives the pitch-shift on the default voice).
            c["char"] = None
            c["bucket"] = classify_bucket_pitch(f0, args.male_max,
                                                args.adult_female_max,
                                                args.child_split)
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
            if best["is_new"] and seg_np.size:
                accumulate_ag_clip(best, seg_np)
            c["char"] = best
        else:
            new = {"id": next_id, "bucket": None, "voice": None,
                   "centroid": np.asarray(emb, dtype=np.float64), "count": 1,
                   "pitch": None, "age": None, "gender": None,
                   "pitch_samples": ([f0] if f0 else []),
                   "ag_clips": [], "ag_len": 0,
                   "is_new": True, "lines": 1}
            if seg_np.size:
                accumulate_ag_clip(new, seg_np)
            next_id += 1
            chars.append(new)
            c["char"] = new

    # Classify each NEW character (gender + age -> bucket) and give it a distinct
    # voice. Known characters keep their locked-in voice. Gender/age come from the
    # age-gender model over the character's aggregated speech, with pitch as the
    # fallback when the model is off or the character has too little clean audio.
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
        ch["pitch"] = med

        bucket = None
        if predictor is not None and ch["ag_clips"]:
            clip = np.concatenate(ch["ag_clips"]).astype("float32")
            age, gender = predictor.predict(clip, ref_sr)
            if gender is not None:
                ch["age"] = age
                ch["gender"] = gender_label(gender)
                bucket = classify_bucket_model(age, gender, args.elder_age)
        if bucket is None:      # model off/unavailable/too little audio
            bucket = classify_bucket_pitch(med, args.male_max,
                                           args.adult_female_max,
                                           args.child_split)

        pool = pools[bucket]
        chosen = next((v for v in pool if v not in used[bucket]), None)
        if chosen is None:                      # pool exhausted - cycle
            chosen = pool[len(used[bucket]) % len(pool)]
            print(f"[tts]   note: reusing voice '{chosen}' for a new "
                  f"{BUCKET_LABELS[bucket]} character (pool exhausted)")
        used[bucket].add(chosen)
        ch["bucket"], ch["voice"] = bucket, chosen

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
            age = f"{ch['age']:4.0f}y" if ch.get("age") else "  -- "
            gen = (ch.get("gender") or "?")[:6]
            pit = f"{ch['pitch']:5.0f}Hz" if ch["pitch"] else "  --  "
            print(f"[tts]   {tag} char#{ch['id']:<3} {BUCKET_LABELS[ch['bucket']]:<14} "
                  f"{gen:<6} {age} {pit}  {ch['voice']:<18} {ch['lines']} line(s)")

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

        # Age the voice: raise children, lower elders (XTTS has no explicit
        # child/old voices, so a small tape-style shift approximates it).
        if bucket in CHILD_BUCKETS and args.child_pitch_shift:
            clip = pitch_shift_semitones(clip, args.child_pitch_shift)
        elif bucket in ELDERLY_BUCKETS and args.elder_pitch_shift:
            clip = pitch_shift_semitones(clip, -abs(args.elder_pitch_shift))

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
