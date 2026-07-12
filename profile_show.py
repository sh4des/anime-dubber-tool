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
        classify_bucket_pitch, AGE_GENDER_MODEL, VOICE_POOLS)
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


def agglomerative_cosine(embs, threshold):
    """Cluster row-embeddings by cosine distance. Returns integer labels (1..k)."""
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    n = len(embs)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([1], dtype=int)
    X = normalize_rows(embs)
    d = pdist(X, metric="cosine")
    Z = linkage(d, method="average")
    return fcluster(Z, t=threshold, criterion="distance")


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

def diarize_resemblyzer(mono16k, sr, encoder, local_threshold, verbose=False):
    import numpy as np
    if encoder is None:
        return []
    regions = energy_vad(mono16k, sr)
    win = int(1.5 * sr)
    hop = int(0.75 * sr)
    windows = []          # (a, b)
    for a, b in regions:
        if b - a < int(0.6 * sr):
            windows.append((a, b))
            continue
        t = a
        while t + win <= b:
            windows.append((t, t + win))
            t += hop
        if not windows or windows[-1][1] < b - int(0.3 * sr):
            windows.append((max(a, b - win), b))
    if not windows:
        return []

    from resemblyzer import preprocess_wav
    embs, keep = [], []
    for a, b in windows:
        seg = mono16k[a:b]
        try:
            w = preprocess_wav(seg, source_sr=sr)
            if w.size < int(0.3 * sr):
                continue
            embs.append(encoder.embed_utterance(w))
            keep.append((a, b))
        except Exception:
            continue
    if not embs:
        return []
    labels = agglomerative_cosine(embs, local_threshold)

    # merge windows sharing a label into per-local-speaker segments
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
        print(f"[profile]   resemblyzer diarizer: {len(out)} local speaker(s)")
    return out


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

    ap.add_argument("--diarizer", choices=["auto", "pyannote", "resemblyzer"],
                    default="auto")
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token for the gated pyannote model")
    ap.add_argument("--no-demucs", action="store_true",
                    help="skip Demucs source separation")

    ap.add_argument("--cluster-threshold", type=float, default=0.70,
                    help="cosine distance to merge speakers GLOBALLY (higher = "
                         "fewer, broader speakers)")
    ap.add_argument("--local-threshold", type=float, default=0.60,
                    help="cosine distance for per-episode clustering (resemblyzer "
                         "backend only)")

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

    # optional resemblyzer encoder (fallback diarizer / needed if pyannote off)
    encoder = None
    if args.diarizer in ("auto", "resemblyzer"):
        from srt_to_speech_multivoice import make_encoder
        encoder = make_encoder(device)

    use_pyannote = args.diarizer in ("auto", "pyannote")

    # --- stage 1+2: per-episode isolation + diarization -----------------------
    # We keep each episode's TARGET_SR mono array on disk (as the stem wav) so we
    # can slice reference clips AFTER global clustering, without holding every
    # episode's audio in RAM at once.
    local_speakers = []     # flat list across episodes
    ep_stub = []            # (name, stem_wav_path, sr)
    backends_used = set()   # which diarizer actually produced speakers
    for wav_path, name in episodes:
        print(f"[profile] === {name} ===")
        if not os.path.exists(wav_path):
            print(f"[profile]   missing audio {wav_path}, skipping.",
                  file=sys.stderr)
            continue

        # dialogue isolation -> a working wav (vocals stem, or the original)
        work_wav = os.path.join(scratch, f"{name}.vocals.wav")
        used = None if args.no_demucs else demucs_vocals(
            wav_path, work_wav, device, args.verbose)
        src_wav = used or wav_path

        mono = load_audio(src_wav, TARGET_SR)
        # persist the (mono, TARGET_SR) we diarized on, for later clip slicing
        stem_wav = os.path.join(scratch, f"{name}.mono16k.wav")
        save_clip(mono, TARGET_SR, stem_wav, sr_out=TARGET_SR)
        ep_idx = len(ep_stub)
        ep_stub.append((name, stem_wav, TARGET_SR))

        locals_ = None
        if use_pyannote:
            locals_ = diarize_pyannote(src_wav, TARGET_SR, args.hf_token,
                                       device, args.verbose)
            if locals_ is None and args.diarizer == "pyannote":
                print("[profile]   pyannote requested but unavailable; "
                      "install/authorize it or use --diarizer auto.",
                      file=sys.stderr)
            elif locals_ is not None:
                backends_used.add("pyannote")
        if locals_ is None:
            locals_ = diarize_resemblyzer(mono, TARGET_SR, encoder,
                                          args.local_threshold, args.verbose)
            if locals_:
                backends_used.add("resemblyzer")
        for ls in locals_:
            ls["episode"] = ep_idx
            local_speakers.append(ls)

        # drop the demucs stem now; keep the mono16k for slicing
        if used and os.path.exists(work_wav):
            os.remove(work_wav)

    if not local_speakers:
        print("[profile] no speakers found in any episode.", file=sys.stderr)
        return 2

    # --- stage 3: GLOBAL clustering across the whole show ---------------------
    embs = np.stack([ls["emb"] for ls in local_speakers])
    glabels = agglomerative_cosine(embs, args.cluster_threshold)
    groups = {}
    for ls, g in zip(local_speakers, glabels):
        groups.setdefault(int(g), []).append(ls)
    print(f"[profile] global speakers before pruning: {len(groups)}")

    # aggregate per global speaker
    speakers = []
    for g, members in groups.items():
        total = sum(m["dur"] for m in members)
        centroid = np.mean(np.stack([m["emb"] for m in members]), axis=0)
        speakers.append({"members": members, "dur": total, "centroid": centroid})
    speakers = [s for s in speakers if s["dur"] >= args.min_speaker_sec]
    speakers.sort(key=lambda s: -s["dur"])
    print(f"[profile] global speakers kept (>= {args.min_speaker_sec}s): "
          f"{len(speakers)}")

    # --- stage 4: gender/age + references + fallback pool voice ---------------
    predictor = None
    if not args.no_age_gender:
        predictor = AgeGenderPredictor(args.age_gender_model, device)

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
        if predictor is not None and ref_audio:
            concat = np.concatenate(ref_audio).astype("float32")
            age, g = predictor.predict(concat, TARGET_SR)
            if g is not None:
                gender = gender_label(g)
                bucket = classify_bucket_model(age, g, args.elder_age)
        if bucket is None:
            bucket = classify_bucket_pitch(None, args.male_max,
                                           args.adult_female_max,
                                           args.child_split)

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

    qc = os.path.splitext(args.out)[0] + ".qc.txt"
    with open(qc, "w", encoding="utf-8") as f:
        f.write(f"Voice profile QC - {len(out_speakers)} speakers\n")
        f.write(f"diarizer={payload['diarizer']} demucs={payload['demucs']}\n\n")
        f.write(f"{'spk':>4} {'bucket':<14} {'gender':<7} {'age':>4} "
                f"{'speech':>8} {'refs':>4}  fallback_voice\n")
        for sp in out_speakers:
            f.write(f"{sp['id']:>4} {sp['bucket']:<14} "
                    f"{(sp['gender'] or '?'):<7} "
                    f"{(('%.0f' % sp['age']) if sp['age'] else '-'):>4} "
                    f"{sp['total_speech_sec']:>7.0f}s {sp['n_references']:>4}  "
                    f"{sp['fallback_voice']}\n")
    print(f"[profile] wrote QC report {qc}")
    print("[profile] NOTE: listen to a few clips in the clip dir to sanity-check "
          "the cast before running Phase B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
