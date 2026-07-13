#!/usr/bin/env python3
# profile_show.py
# -----------------------------------------------------------------------------
# PHASE A of the "cloned + performance" dub pipeline: build a show-level VOICE
# PROFILE from the original (Japanese) audio, ONCE per show, so Phase B can clone
# each character's actual voice instead of picking an English voice from a pool.
#
# This is the "listen to the originals, identify the voices, profile them" step.
# It is deliberately separate from synthesis: it is heavy (source separation +
# diarization over every episode) but only needs to run once, and its output is
# inspectable before you synthesize anything.
#
# Pipeline
# --------
#   1. DIALOGUE ISOLATION  - Demucs (htdemucs) vocal stem per episode, to strip
#      background music / SFX. Cleaner input for both identification and for the
#      reference clips we later clone from. (--no-demucs to skip.)
#   2. DIARIZATION         - "who spoke when" per episode. Default backend is
#      pyannote.audio 3.1 (VAD + overlap-aware); if pyannote is unavailable or
#      its gated model is not authorized, falls back to energy-VAD + resemblyzer
#      window embeddings clustered per episode. Either way each episode yields a
#      set of LOCAL speakers, each with a mean embedding + speech segments.
#   3. GLOBAL CLUSTERING   - agglomerative clustering (cosine) over every
#      per-episode local speaker across the WHOLE show, so the same character in
#      episode 1 and episode 25 becomes one GLOBAL speaker. Seeing all episodes
#      at once avoids the order-dependent merge/split of online matching.
#   4. PROFILING           - per global speaker: gender + age (the shared
#      age+gender model), total speaking time (ranks main vs minor cast), a
#      cluster centroid, and the top-K longest/cleanest solo REFERENCE CLIPS
#      (saved as WAVs) that Phase B clones from. A distinct fallback pool voice
#      is also assigned globally, used when a speaker has no clean reference.
#   5. OUTPUT              - a profile JSON (speakers + per-episode turns) plus a
#      plain-text QC report so you can spot-check the cast without a manual step.
#
# The wrapper (subtitle-anime-profile.ps1) extracts one 44.1kHz audio WAV per
# episode and hands the list here; all ML lives in this file.
#
# HONEST LIMITATIONS
#   * Demucs is trained on music, not anime; singing/heavy SFX still leak.
#   * Fully automatic clustering can merge two similar voices or split one across
#     shouting/whispering. Tune --cluster-threshold. There is no manual naming
#     step by design (see the AskUserQuestion answer that drove this).
#   * pyannote's gated model needs a one-time HF login+accept; without it the
#     resemblyzer fallback is used (no overlap handling, slightly weaker).
#   * UNTESTED end-to-end on GPU as written - iterate on a test episode first.
# -----------------------------------------------------------------------------

import argparse
import json
import os
import sys

os.environ.setdefault("COQUI_TOS_AGREED", "1")

# Reuse the shared age+gender model + helpers from the multi-voice script (its
# heavy imports live inside methods, so importing the module is cheap).
try:
    from srt_to_speech_multivoice import (
        AgeGenderPredictor, gender_label, classify_bucket_model,
        classify_bucket_pitch, voiced_median_pitch, AGE_GENDER_MODEL, VOICE_POOLS)
except Exception as e:   # pragma: no cover - only if the sibling file moved
    print(f"[profile] FATAL: cannot import srt_to_speech_multivoice ({e})",
          file=sys.stderr)
    sys.exit(3)

PROFILE_VERSION = 1
TARGET_SR = 16000          # sample rate for diarization / embeddings / age-gender
REF_SR = 24000             # sample rate reference clips are saved at (XTTS-friendly)
DEMUCS_MODEL = "htdemucs"


# --- small audio helpers ------------------------------------------------------

def load_audio(path, sr):
    """Load an audio file as (mono float32 numpy at sr, sr)."""
    import torchaudio
    wav, in_sr = torchaudio.load(path)
    wav = wav.mean(dim=0)                      # mono
    if in_sr != sr:
        wav = torchaudio.functional.resample(wav, in_sr, sr)
    return wav.detach().cpu().numpy().astype("float32")


def save_clip(np_mono, sr_in, out_path, sr_out=REF_SR):
    """Save a mono numpy clip to WAV at sr_out."""
    import torch
    import torchaudio
    t = torch.from_numpy(np_mono).unsqueeze(0)
    if sr_in != sr_out:
        t = torchaudio.functional.resample(t, sr_in, sr_out)
    # light peak-normalise so reference clips are a consistent level
    peak = float(t.abs().max())
    if peak > 0:
        t = t / peak * 0.97
    torchaudio.save(out_path, t, sr_out)


def energy_vad(mono, sr, frame_ms=30, thresh_ratio=0.06, min_speech_ms=250,
               pad_ms=150):
    """Very small energy-based VAD. Returns [(start_sample, end_sample), ...].

    Works well on a Demucs VOCALS stem (near-silent between speech). Frames above
    thresh_ratio * the loud-frame reference RMS are speech; short gaps are bridged
    and regions padded slightly so we don't clip word onsets.
    """
    import numpy as np
    n = mono.size
    fl = max(1, int(sr * frame_ms / 1000))
    nf = n // fl
    if nf == 0:
        return []
    frames = mono[:nf * fl].reshape(nf, fl)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    ref = np.percentile(rms, 95) or float(rms.max())
    if ref <= 0:
        return []
    speech = rms > (thresh_ratio * ref)

    # bridge short gaps (<= 1 frame) then collect runs
    regions = []
    i = 0
    gap_bridge = 1
    while i < nf:
        if not speech[i]:
            i += 1
            continue
        j = i
        gap = 0
        while j + 1 < nf and (speech[j + 1] or gap < gap_bridge):
            if speech[j + 1]:
                gap = 0
            else:
                gap += 1
            j += 1
        a, b = i * fl, min(n, (j + 1) * fl)
        regions.append([a, b])
        i = j + 1

    pad = int(sr * pad_ms / 1000)
    min_len = int(sr * min_speech_ms / 1000)
    out = []
    for a, b in regions:
        a = max(0, a - pad)
        b = min(n, b + pad)
        if b - a >= min_len:
            out.append((a, b))
    return out


def normalize_rows(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def agglomerative_cosine(embs, threshold, method="average"):
    """Cluster row-embeddings by cosine distance. Returns integer labels (1..k).

    Default 'average' matches the per-episode window diarizer's original tuning
    (complete linkage is far too strict at the window level and over-splits into
    hundreds of one-window "speakers"). The GLOBAL speaker clustering passes
    method='complete' explicitly (via --linkage) to resist the chaining that made
    average linkage collapse most of the cast into one blob - complete only merges
    two clusters when ALL members are within threshold, so clusters stay tight.
    'ward' is also available: it feeds Euclidean distances of the L2-normalized
    rows (== sqrt(2*(1-cos)), monotonic in cosine), since ward needs Euclidean
    geometry.
    """
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    n = len(embs)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([1], dtype=int)
    X = normalize_rows(embs)
    d = pdist(X, metric="euclidean" if method == "ward" else "cosine")
    Z = linkage(d, method=method)
    return fcluster(Z, t=threshold, criterion="distance")


def merge_close_speakers(speakers, thr):
    """Recombine global speakers whose centroids are within cosine distance `thr`.

    Fixes the main cause of scene-to-scene voice drift: one character split
    across several clusters (loud vs quiet delivery drifts the per-episode
    embedding). Iteratively merges the closest pair under the threshold until
    none remain. Each speaker is {"members","dur","centroid"}.
    """
    import numpy as np
    if thr is None or thr <= 0:
        return speakers
    speakers = list(speakers)
    while len(speakers) > 1:
        cents = normalize_rows(np.stack([s["centroid"] for s in speakers]))
        best = None
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                dist = 1.0 - float(np.dot(cents[i], cents[j]))
                if dist < thr and (best is None or dist < best[0]):
                    best = (dist, i, j)
        if best is None:
            break
        _, i, j = best
        members = speakers[i]["members"] + speakers[j]["members"]
        merged = {
            "members": members,
            "dur": speakers[i]["dur"] + speakers[j]["dur"],
            "centroid": np.mean(np.stack([m["emb"] for m in members]), axis=0),
        }
        speakers = [s for k, s in enumerate(speakers) if k not in (i, j)]
        speakers.append(merged)
    return speakers


def cluster_to_k(embs, k, method="ward"):
    """Cut the dendrogram to exactly k clusters (maxclust), not a distance.

    The per-episode diarizer over-segments into thousands of window-level
    fragments; no single cosine-distance cut cleanly recovers the cast from them
    (too small -> one chained blob, too large -> thousands of singletons). Asking
    for a fixed cluster COUNT is robust to that. 'ward' (on Euclidean distances of
    L2-normalized rows) gives the most balanced clusters.
    """
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    n = len(embs)
    if n == 0:
        return np.array([], dtype=int)
    if n <= k:
        return np.arange(1, n + 1, dtype=int)
    X = normalize_rows(embs)
    d = pdist(X, metric="euclidean" if method == "ward" else "cosine")
    Z = linkage(d, method=method)
    return fcluster(Z, t=k, criterion="maxclust")


# --- stage 1: dialogue isolation (Demucs) -------------------------------------

def demucs_vocals(mono_or_path, out_wav, device, verbose=False):
    """Separate the vocal stem with Demucs, written to out_wav (44.1k stereo->mono).
    Returns out_wav on success, or None to signal 'use the original audio'."""
    try:
        import torch
        import torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except Exception as e:
        print(f"[profile] Demucs unavailable ({e}); using un-separated audio.",
              file=sys.stderr)
        return None
    try:
        model = get_model(DEMUCS_MODEL)
        model.to(device).eval()
        sr = model.samplerate                    # 44100
        wav, in_sr = torchaudio.load(mono_or_path)
        if wav.shape[0] == 1:                     # demucs wants stereo
            wav = wav.repeat(2, 1)
        if in_sr != sr:
            wav = torchaudio.functional.resample(wav, in_sr, sr)
        with torch.no_grad():
            est = apply_model(model, wav[None].to(device), device=device,
                              split=True, overlap=0.1)[0]
        vocals = est[model.sources.index("vocals")].mean(dim=0, keepdim=True).cpu()
        torchaudio.save(out_wav, vocals, sr)
        if verbose:
            print(f"[profile]   demucs vocals -> {out_wav}")
        return out_wav
    except Exception as e:
        print(f"[profile] Demucs failed ({e}); using un-separated audio.",
              file=sys.stderr)
        return None


# --- stage 2: diarization backends --------------------------------------------
# Each returns a list of LOCAL speakers for one episode:
#   [{"emb": np[d], "segments": [(a,b), ...], "dur": float_seconds}, ...]
# with sample indices a,b into the TARGET_SR mono array it was given.

def _speech_windows(mono, sr, win_s=1.5, hop_s=0.75):
    """VAD -> sliding windows over speech regions. Returns [(a, b), ...]."""
    regions = energy_vad(mono, sr)
    win, hop = int(win_s * sr), int(hop_s * sr)
    out = []
    for a, b in regions:
        if b - a < int(0.6 * sr):
            out.append((a, b))
            continue
        t = a
        while t + win <= b:
            out.append((t, t + win))
            t += hop
        if out[-1][1] < b - int(0.3 * sr):
            out.append((max(a, b - win), b))
    return out


def diarize_windows(mono, sr, embed_one, local_threshold, label="", verbose=False):
    """Generic window-embed + cluster diarizer. embed_one(seg_np)->vec|None.
    Returns local speakers: [{"emb", "segments", "dur"}, ...]."""
    import numpy as np
    windows = _speech_windows(mono, sr)
    if not windows:
        return []
    embs, keep = [], []
    for a, b in windows:
        v = embed_one(mono[a:b])
        if v is not None:
            embs.append(np.asarray(v, dtype=np.float64).reshape(-1))
            keep.append((a, b))
    if not embs:
        return []
    labels = agglomerative_cosine(embs, local_threshold)
    spk = {}
    for (a, b), lab, emb in zip(keep, labels, embs):
        s = spk.setdefault(int(lab), {"emb": [], "segments": []})
        s["emb"].append(emb)
        s["segments"].append((a, b))
    out = []
    for s in spk.values():
        segs = _merge_segments(s["segments"])
        dur = sum((b - a) for a, b in segs) / sr
        out.append({"emb": np.mean(s["emb"], axis=0), "segments": segs, "dur": dur})
    if verbose:
        print(f"[profile]   {label} diarizer: {len(out)} local speaker(s)")
    return out


def make_resemblyzer_embed(encoder, sr):
    """embed_one closure for the resemblyzer backend (already-installed)."""
    if encoder is None:
        return None
    from resemblyzer import preprocess_wav

    def embed_one(seg):
        try:
            w = preprocess_wav(seg, source_sr=sr)
            if w.size < int(0.3 * sr):
                return None
            return encoder.embed_utterance(w)
        except Exception:
            return None
    return embed_one


def make_ecapa_embed(device, sr):
    """embed_one closure for SpeechBrain's ECAPA-TDNN (non-gated, local, more
    discriminative than resemblyzer). Returns None if speechbrain is missing."""
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as e:
        print(f"[profile] ECAPA unavailable ({e}); install:  pip install speechbrain",
              file=sys.stderr)
        return None
    savedir = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache")),
                           "speechbrain-ecapa")
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}, savedir=savedir)

    def embed_one(seg):
        try:
            if seg.size < int(0.4 * sr):
                return None
            t = torch.from_numpy(seg).float().unsqueeze(0).to(device)
            e = clf.encode_batch(t)          # (1, 1, 192)
            return e.squeeze().detach().cpu().numpy()
        except Exception:
            return None
    print(f"[profile] ECAPA speaker encoder loaded (device={device})")
    return embed_one


def diarize_pyannote(wav_path, sr, token, device, verbose=False):
    """pyannote 3.1 diarization + per-turn embeddings. Returns local speakers or
    [] to signal the caller should fall back to resemblyzer."""
    import numpy as np
    try:
        import torch
        import torchaudio
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.speaker_verification import (
            PretrainedSpeakerEmbedding)
    except Exception as e:
        if verbose:
            print(f"[profile]   pyannote unavailable ({e}); falling back.",
                  file=sys.stderr)
        return None
    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token=token or True)
        pipe.to(torch.device(device))
        embedder = PretrainedSpeakerEmbedding(
            "pyannote/wespeaker-voxceleb-resnet34-LM", device=torch.device(device))
        diar = pipe(wav_path)

        wav, in_sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0, keepdim=True)
        if in_sr != sr:
            wav = torchaudio.functional.resample(wav, in_sr, sr)
        arr = wav.squeeze(0).numpy().astype("float32")

        spk = {}
        for turn, _, label in diar.itertracks(yield_label=True):
            a, b = int(turn.start * sr), int(turn.end * sr)
            if b - a < int(0.4 * sr):
                continue
            seg = arr[a:b]
            try:
                emb = embedder(torch.from_numpy(seg)[None, None, :])
                emb = np.asarray(emb).reshape(-1)
            except Exception:
                continue
            s = spk.setdefault(str(label), {"emb": [], "segments": []})
            s["emb"].append(emb)
            s["segments"].append((a, b))
        out = []
        for s in spk.values():
            if not s["emb"]:
                continue
            segs = _merge_segments(s["segments"])
            dur = sum((b - a) for a, b in segs) / sr
            out.append({"emb": np.mean(s["emb"], axis=0), "segments": segs,
                        "dur": dur})
        if verbose:
            print(f"[profile]   pyannote diarizer: {len(out)} local speaker(s)")
        return out
    except Exception as e:
        print(f"[profile]   pyannote failed ({e}); falling back to resemblyzer.",
              file=sys.stderr)
        return None


def _merge_segments(segs, gap=0):
    segs = sorted(segs)
    out = []
    for a, b in segs:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


# --- reference-clip selection -------------------------------------------------

def pick_reference_segments(segments, sr, top_k, min_sec, max_sec):
    """Longest solo segments >= min_sec, each capped to max_sec, up to top_k."""
    cands = []
    for a, b in segments:
        if (b - a) / sr >= min_sec:
            cands.append((a, min(b, a + int(max_sec * sr))))
    cands.sort(key=lambda ab: (ab[1] - ab[0]), reverse=True)
    return cands[:top_k]


# --- main ---------------------------------------------------------------------

def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser(
        description="Build a show voice profile (Phase A) from original audio")
    ap.add_argument("--audio-list", required=True,
                    help="text file, one line per episode: <wav_path>\\t<name>")
    ap.add_argument("--out", required=True, help="output profile JSON path")
    ap.add_argument("--clip-dir", required=True,
                    help="directory to write reference clip WAVs into")
    ap.add_argument("--scratch", default=None,
                    help="scratch dir for Demucs stems (default: alongside clips)")

    ap.add_argument("--diarizer",
                    choices=["auto", "ecapa", "pyannote", "resemblyzer"],
                    default="auto",
                    help="ecapa=SpeechBrain (no signup, best local quality); "
                         "resemblyzer=already installed (crude); pyannote=gated. "
                         "auto prefers ecapa, then resemblyzer.")
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token for the gated pyannote model")
    ap.add_argument("--no-demucs", action="store_true",
                    help="skip Demucs source separation")
    ap.add_argument("--reuse-stems", action="store_true",
                    help="reuse Demucs vocal stems already in --scratch instead of "
                         "re-separating (fast when tuning thresholds)")
    ap.add_argument("--max-speakers", type=int, default=40,
                    help="keep at most this many global speakers (by speech time)")

    # NOTE: these are cosine DISTANCES (0=identical, ~1=unrelated). resemblyzer
    # embeddings are compressed - even different speakers are only ~0.2-0.4 apart
    # - so these thresholds must be SMALL or everyone merges into one blob. Tuned
    # for the resemblyzer backend; pyannote (wespeaker) embeddings are more spread
    # and tolerate larger values.
    ap.add_argument("--cluster-threshold", type=float, default=None,
                    help="cosine distance to merge speakers GLOBALLY (higher = "
                         "fewer, broader speakers). Default is per-backend.")
    ap.add_argument("--local-threshold", type=float, default=None,
                    help="cosine distance for per-episode window clustering. "
                         "Default is per-backend.")
    ap.add_argument("--linkage", choices=["complete", "average", "ward"],
                    default="ward",
                    help="global clustering linkage for the maxclust cut. 'ward' "
                         "(default) gives the most balanced clusters.")
    ap.add_argument("--global-clusters", type=int, default=0,
                    help="target number of global clusters to cut to before "
                         "pruning (0 = auto = 3x --max-speakers). Deterministic "
                         "maxclust cut, robust to the over-segmented locals.")
    ap.add_argument("--merge-threshold", type=float, default=0.25,
                    help="after clustering, merge two global speakers whose "
                         "centroids are within this cosine distance (recombines a "
                         "character split across clusters; 0 disables).")
    ap.add_argument("--reuse-locals", action="store_true",
                    help="skip Demucs+diarization and reuse cached per-episode "
                         "embeddings (locals_cache.pkl in the stem dir) - lets you "
                         "re-tune the global clustering in seconds. Requires a "
                         "prior full run that wrote the cache.")

    ap.add_argument("--top-k-refs", type=int, default=4,
                    help="reference clips saved per speaker")
    ap.add_argument("--min-ref-sec", type=float, default=2.5)
    ap.add_argument("--max-ref-sec", type=float, default=12.0)
    ap.add_argument("--min-speaker-sec", type=float, default=8.0,
                    help="drop global speakers with less total speech than this "
                         "(likely noise/one-off crowd lines)")

    # age/gender (fallback pool assignment); same knobs as the dub script.
    ap.add_argument("--age-gender-model", default=AGE_GENDER_MODEL)
    ap.add_argument("--no-age-gender", action="store_true")
    ap.add_argument("--require-age-gender", action="store_true",
                    help="fail fast if the age+gender model cannot load, instead "
                         "of silently degrading to the pitch fallback.")
    ap.add_argument("--elder-age", type=float, default=58.0)
    ap.add_argument("--male-max", type=float, default=155.0)
    ap.add_argument("--adult-female-max", type=float, default=250.0)
    ap.add_argument("--child-split", type=float, default=300.0)

    ap.add_argument("--device", default=None, help="cuda|cpu (auto)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("[profile] missing torch; install the GPU stack.", file=sys.stderr)
        return 3
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[profile] device={device}")

    # Fail loudly on missing requested backends, so we don't silently degrade to
    # un-separated audio / a weaker diarizer and waste a full run.
    import importlib.util
    def _have(m):
        return importlib.util.find_spec(m) is not None
    if not args.no_demucs and not _have("demucs"):
        print("[profile] FATAL: Demucs requested but 'demucs' is not installed.\n"
              "         install:  pip install demucs\n"
              "         or pass --no-demucs to run on un-separated audio (much worse).",
              file=sys.stderr)
        return 3
    if args.diarizer == "pyannote" and not _have("pyannote.audio"):
        print("[profile] FATAL: --diarizer pyannote but 'pyannote.audio' is not "
              "installed.\n         install:  pip install pyannote.audio\n"
              "         then accept terms for pyannote/speaker-diarization-3.1 AND\n"
              "         pyannote/wespeaker-voxceleb-resnet34-LM on HuggingFace, and\n"
              "         set HF_TOKEN (or pass --hf-token).", file=sys.stderr)
        return 3

    # Resolve 'auto' to a concrete backend: prefer pyannote (best, gated), then
    # ecapa (SpeechBrain, no signup), then resemblyzer (installed, crude).
    backend = args.diarizer
    if backend == "auto":
        backend = ("pyannote" if _have("pyannote.audio")
                   else "ecapa" if _have("speechbrain")
                   else "resemblyzer")
        print(f"[profile] diarizer=auto -> {backend}")
    if backend == "ecapa" and not _have("speechbrain"):
        print("[profile] FATAL: --diarizer ecapa but 'speechbrain' is not "
              "installed.\n         install:  pip install speechbrain",
              file=sys.stderr)
        return 3

    # Per-backend default thresholds (cosine distance). These differ because each
    # encoder's embedding space is scaled differently: resemblyzer is compressed,
    # ecapa/wespeaker are more spread. Overridable via the CLI.
    if args.cluster_threshold is None:
        args.cluster_threshold = {"pyannote": 0.50, "ecapa": 0.45}.get(backend, 0.35)
    if args.local_threshold is None:
        args.local_threshold = 0.55 if backend == "ecapa" else 0.45
    print(f"[profile] thresholds: local<={args.local_threshold} "
          f"global<={args.cluster_threshold}")

    os.makedirs(args.clip_dir, exist_ok=True)
    scratch = args.scratch or os.path.join(args.clip_dir, "_stems")
    os.makedirs(scratch, exist_ok=True)

    # episode list
    episodes = []
    with open(args.audio_list, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            wav = parts[0]
            name = parts[1] if len(parts) > 1 else os.path.splitext(
                os.path.basename(wav))[0]
            episodes.append((wav, name))
    if not episodes:
        print("[profile] audio list is empty.", file=sys.stderr)
        return 2
    print(f"[profile] {len(episodes)} episode(s) to profile")

    # Build the window-embedding function for the chosen local backend. pyannote
    # is self-contained (no embed_one needed).
    embed_one = None
    if backend == "resemblyzer":
        from srt_to_speech_multivoice import make_encoder
        enc = make_encoder(device)
        embed_one = make_resemblyzer_embed(enc, TARGET_SR)
        if embed_one is None:
            print("[profile] FATAL: resemblyzer unavailable.", file=sys.stderr)
            return 3
    elif backend == "ecapa":
        embed_one = make_ecapa_embed(device, TARGET_SR)
        if embed_one is None:
            return 3
    print(f"[profile] diarizer backend: {backend}")

    # --- stage 1+2: per-episode isolation + diarization -----------------------
    # We keep each episode's TARGET_SR mono array on disk (as the stem wav) so we
    # can slice reference clips AFTER global clustering, without holding every
    # episode's audio in RAM at once.
    locals_cache = os.path.join(scratch, "locals_cache.pkl")
    local_speakers = []     # flat list across episodes
    ep_stub = []            # (name, stem_wav_path, sr)
    backends_used = set()   # which diarizer actually produced speakers
    reused_locals = False
    if args.reuse_locals and os.path.exists(locals_cache):
        import pickle
        with open(locals_cache, "rb") as f:
            cached = pickle.load(f)
        local_speakers = cached["local_speakers"]
        ep_stub = cached["ep_stub"]
        backends_used = set(cached.get("backends_used") or [backend])
        reused_locals = True
        print(f"[profile] --reuse-locals: loaded {len(local_speakers)} local "
              f"speaker(s) from {len(ep_stub)} episode(s); skipping diarization")
    for wav_path, name in episodes:
        if reused_locals:
            break
        print(f"[profile] === {name} ===")
        if not os.path.exists(wav_path):
            print(f"[profile]   missing audio {wav_path}, skipping.",
                  file=sys.stderr)
            continue

        # dialogue isolation -> a working wav (vocals stem, or the original).
        # Reuse a cached stem when tuning so we don't re-run Demucs every time.
        work_wav = os.path.join(scratch, f"{name}.vocals.wav")
        if args.no_demucs:
            used = None
        elif args.reuse_stems and os.path.exists(work_wav):
            print(f"[profile]   reusing cached vocal stem")
            used = work_wav
        else:
            used = demucs_vocals(wav_path, work_wav, device, args.verbose)
        src_wav = used or wav_path

        mono = load_audio(src_wav, TARGET_SR)
        # persist the (mono, TARGET_SR) we diarized on, for later clip slicing
        stem_wav = os.path.join(scratch, f"{name}.mono16k.wav")
        save_clip(mono, TARGET_SR, stem_wav, sr_out=TARGET_SR)
        ep_idx = len(ep_stub)
        ep_stub.append((name, stem_wav, TARGET_SR))

        if backend == "pyannote":
            locals_ = diarize_pyannote(src_wav, TARGET_SR, args.hf_token,
                                       device, args.verbose)
            if locals_ is None:
                print("[profile] FATAL: pyannote diarization failed (see above). "
                      "Fix install/auth, or use --diarizer ecapa|resemblyzer.",
                      file=sys.stderr)
                return 4
            backends_used.add("pyannote")
        else:
            locals_ = diarize_windows(mono, TARGET_SR, embed_one,
                                      args.local_threshold, label=backend,
                                      verbose=args.verbose)
            if locals_:
                backends_used.add(backend)
        for ls in locals_:
            ls["episode"] = ep_idx
            local_speakers.append(ls)
        # NOTE: vocal stems are kept in --scratch for --reuse-stems on re-tuning.

    if not reused_locals and local_speakers:
        import pickle
        with open(locals_cache, "wb") as f:
            pickle.dump({"local_speakers": local_speakers, "ep_stub": ep_stub,
                         "backends_used": list(backends_used)}, f)
        print(f"[profile] cached {len(local_speakers)} local embedding(s) -> "
              f"{locals_cache}")

    if not local_speakers:
        print("[profile] no speakers found in any episode.", file=sys.stderr)
        return 2

    # --- stage 3: GLOBAL clustering across the whole show ---------------------
    # Cut to a fixed NUMBER of clusters (maxclust), not a distance: robust to the
    # over-segmented locals. Ask for ~3x the final cap so noise clusters can be
    # pruned by speech time afterwards.
    embs = np.stack([ls["emb"] for ls in local_speakers])
    target_k = args.global_clusters or min(3 * args.max_speakers, len(local_speakers))
    glabels = cluster_to_k(embs, target_k, method=args.linkage)
    groups = {}
    for ls, g in zip(local_speakers, glabels):
        groups.setdefault(int(g), []).append(ls)
    print(f"[profile] global clusters ({args.linkage} linkage, "
          f"maxclust={target_k}): {len(groups)}")

    # aggregate per global speaker
    speakers = []
    for g, members in groups.items():
        total = sum(m["dur"] for m in members)
        centroid = np.mean(np.stack([m["emb"] for m in members]), axis=0)
        speakers.append({"members": members, "dur": total, "centroid": centroid})

    # refinement: recombine a character split across clusters (drift fix). Bounded
    # - merge is O(n^2) per merge, so only run it on the already-capped cluster set.
    if len(speakers) <= 400:
        before_merge = len(speakers)
        speakers = merge_close_speakers(speakers, args.merge_threshold)
        if len(speakers) < before_merge:
            print(f"[profile] merged {before_merge} -> {len(speakers)} speakers "
                  f"(centroids within {args.merge_threshold} cosine)")

    speakers = [s for s in speakers if s["dur"] >= args.min_speaker_sec]
    speakers.sort(key=lambda s: -s["dur"])

    # Prune speakers that cannot yield a clean reference clip (no solo segment
    # >= min-ref-sec) - they can't be cloned, only pollute the cast. Then cap to
    # the top --max-speakers by speech time (the over-split tail is noise).
    def longest_solo_sec(s):
        best = 0.0
        for m in s["members"]:
            sr = ep_stub[m["episode"]][2]
            for a, b in m["segments"]:
                best = max(best, (b - a) / sr)
        return best
    before = len(speakers)
    speakers = [s for s in speakers if longest_solo_sec(s) >= args.min_ref_sec]
    dropped_noref = before - len(speakers)
    if len(speakers) > args.max_speakers:
        print(f"[profile] capping {len(speakers)} -> {args.max_speakers} "
              f"speakers (by speech time)")
        speakers = speakers[:args.max_speakers]
    print(f"[profile] global speakers kept: {len(speakers)} "
          f"(>= {args.min_speaker_sec}s, with a clean reference; "
          f"dropped {dropped_noref} un-cloneable)")

    # --- stage 4: gender/age + references + fallback pool voice ---------------
    predictor = None
    if not args.no_age_gender:
        predictor = AgeGenderPredictor(args.age_gender_model, device)
        if args.require_age_gender and not predictor._load():
            print("[profile] FATAL: --require-age-gender set but the age+gender "
                  "model failed to load (see the traceback above).",
                  file=sys.stderr)
            return 4

    used_pool = {b: set() for b in VOICE_POOLS}

    out_speakers = []
    for rank, s in enumerate(speakers, start=1):
        sid = rank
        # gather reference segments (longest solo) from this speaker's episodes
        refs, ref_audio = [], []
        for m in s["members"]:
            name, stem_wav, sr = ep_stub[m["episode"]]
            picks = pick_reference_segments(m["segments"], sr, args.top_k_refs,
                                            args.min_ref_sec, args.max_ref_sec)
            if not picks:
                continue
            mono = load_audio(stem_wav, sr)
            for k, (a, b) in enumerate(picks):
                clip = mono[a:b]
                fn = os.path.join(args.clip_dir, f"spk{sid:02d}_{name}_{k}.wav")
                save_clip(clip, sr, fn)
                refs.append(os.path.relpath(fn, os.path.dirname(args.out)))
                if len(ref_audio) < 6:
                    ref_audio.append(clip)      # for age/gender
        # cap total refs to top_k by keeping the first (already longest-first)
        refs = refs[:args.top_k_refs]

        # gender/age
        age, gender, bucket = None, None, None
        concat = (np.concatenate(ref_audio).astype("float32")
                  if ref_audio else None)
        if predictor is not None and concat is not None:
            age, g = predictor.predict(concat, TARGET_SR)
            if g is not None:
                gender = gender_label(g)
                bucket = classify_bucket_model(age, g, args.elder_age)
        if bucket is None:
            # Model off/failed: measure a REAL median F0 from this speaker's
            # clean speech and use the pitch fallback, so we never blindly
            # bucket everyone adult_male (the old bug passed a literal None).
            f0 = None
            if concat is not None:
                import torch
                f0 = voiced_median_pitch(torch.from_numpy(concat), TARGET_SR)
            bucket = classify_bucket_pitch(f0, args.male_max,
                                           args.adult_female_max,
                                           args.child_split)
            if gender is None and f0 is not None:
                gender = "female" if "female" in bucket else "male"

        # distinct fallback pool voice (global dedup - better than online)
        pool = VOICE_POOLS[bucket]
        voice = next((v for v in pool if v not in used_pool[bucket]), None)
        if voice is None:
            voice = pool[len(used_pool[bucket]) % len(pool)]
        used_pool[bucket].add(voice)

        # per-episode turns for this speaker (Phase B maps cues by time-overlap)
        turns = {}
        for m in s["members"]:
            name = ep_stub[m["episode"]][0]
            sr = ep_stub[m["episode"]][2]
            turns.setdefault(name, []).extend(
                [{"start": round(a / sr, 3), "end": round(b / sr, 3)}
                 for a, b in m["segments"]])

        out_speakers.append({
            "id": sid,
            "rank": rank,
            "gender": gender,
            "age": (round(age, 1) if age else None),
            "bucket": bucket,
            "fallback_voice": voice,
            "total_speech_sec": round(s["dur"], 1),
            "n_references": len(refs),
            "references": refs,
            "centroid": [round(float(x), 6) for x in s["centroid"]],
            "turns": turns,
        })
        print(f"[profile]   spk{sid:02d} {bucket:<14} "
              f"{(gender or '?'):<6} {('%.0fy' % age) if age else '  -':<5} "
              f"{s['dur']:6.0f}s  {len(refs)} ref(s)  fallback={voice}")

    # --- stage 5: write profile + QC report -----------------------------------
    payload = {
        "version": PROFILE_VERSION,
        "diarizer": ("+".join(sorted(backends_used)) if backends_used else "none"),
        "demucs": (not args.no_demucs),
        "clip_dir": os.path.relpath(args.clip_dir, os.path.dirname(args.out)),
        "speakers": out_speakers,
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, args.out)
    print(f"[profile] wrote {args.out}  ({len(out_speakers)} speakers)")

    # nearest-neighbour centroid distance per speaker: a small value means two
    # clusters look like the same character (over-split / a drift risk that the
    # merge pass did not catch). A useful sanity signal in automatic mode.
    nearest = {}
    if len(out_speakers) > 1:
        C = normalize_rows(np.array([sp["centroid"] for sp in out_speakers]))
        sims = C @ C.T
        np.fill_diagonal(sims, -1.0)
        for idx, sp in enumerate(out_speakers):
            nearest[sp["id"]] = 1.0 - float(sims[idx].max())

    qc = os.path.splitext(args.out)[0] + ".qc.txt"
    with open(qc, "w", encoding="utf-8") as f:
        f.write(f"Voice profile QC - {len(out_speakers)} speakers\n")
        f.write(f"diarizer={payload['diarizer']} demucs={payload['demucs']} "
                f"linkage={args.linkage} cluster_thr={args.cluster_threshold} "
                f"merge_thr={args.merge_threshold}\n")
        f.write("('near' = cosine distance to the most similar other speaker; "
                "small = possible over-split)\n\n")
        f.write(f"{'spk':>4} {'bucket':<14} {'gender':<7} {'age':>4} "
                f"{'speech':>8} {'refs':>4} {'near':>5}  fallback_voice\n")
        for sp in out_speakers:
            f.write(f"{sp['id']:>4} {sp['bucket']:<14} "
                    f"{(sp['gender'] or '?'):<7} "
                    f"{(('%.0f' % sp['age']) if sp['age'] else '-'):>4} "
                    f"{sp['total_speech_sec']:>7.0f}s {sp['n_references']:>4} "
                    f"{nearest.get(sp['id'], float('nan')):>5.2f}  "
                    f"{sp['fallback_voice']}\n")
    print(f"[profile] wrote QC report {qc}")
    print("[profile] NOTE: listen to a few clips in the clip dir to sanity-check "
          "the cast before running Phase B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
