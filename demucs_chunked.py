"""Memory-bounded Demucs separation for feature-length inputs.

`apply_model` returns EVERY source at FULL length in one tensor allocated on
`mix.device`. Even with the mix left on the CPU so only local computation runs
on the GPU, that return value is host RAM:

    4 stems x 2 ch x 4 bytes x (seconds x 44100)

which is ~0.68 GB for a 22-minute episode but **14.1 GB for a 2h47m film**. That
is a single contiguous allocation, so it fails outright rather than swapping -
measured on Solaris (1972), 2:46:55, on a 64 GB box with 11 GB free:

    Demucs failed ([enforce fail at alloc_cpu.cpp:121] DefaultCPUAllocator:
    not enough memory: you tried to allocate 14132183488 bytes.)

Both callers swallow that and silently degrade (the profiler falls back to
un-separated audio, the music bed falls back to ducking the original), so a long
film quietly loses the vocal isolation the whole pipeline is built on.

The fix here is to feed Demucs bounded windows and stream the requested stems
straight to disk with a linear crossfade across the window seams. An earlier
attempt at hand-rolled chunking left audible seams; the difference is that this
crossfades (`xfade_sec`, 2 s by default) instead of butt-joining, and that
Demucs still does its own overlap-add *within* each window.

Sample-exact and drift-free: a window's output is the same length as its input,
so the concatenated result is exactly as long as the source. That matters for
the music bed, which has to stay aligned with the video. It requires the input
to already be at the model's rate (both callers extract at 44.1 kHz); anything
else returns None so the caller can keep its existing fallback.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "htdemucs"
# 8 minutes -> ~0.68 GB for the 4-stem return, the same peak a normal episode
# already had. Lower it if something else is holding a lot of RAM.
DEFAULT_CHUNK_SEC = 480.0
DEFAULT_XFADE_SEC = 2.0


def separate_to_files(src, outputs, device="cuda", model_name=DEFAULT_MODEL,
                      chunk_sec=DEFAULT_CHUNK_SEC, xfade_sec=DEFAULT_XFADE_SEC,
                      mono=False, progress=False, log=print):
    """Separate `src`, streaming each requested stem to its own file.

    outputs: {stem_key: path}. `stem_key` is one of the model's own sources
             ("vocals", "drums", "bass", "other") or the synthetic "no_vocals"
             (= every source except vocals, i.e. what `demucs --two-stems=vocals`
             writes alongside vocals.wav).
    mono:    average the stem to one channel before writing.

    Returns `outputs` on success, or None if the caller should fall back (input
    sample rate != model rate).
    """
    import torch
    import torchaudio
    import soundfile as sf
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    info = torchaudio.info(src)
    model = get_model(model_name)
    model.to(device).eval()
    sr = model.samplerate
    if info.sample_rate != sr:
        log(f"[demucs-chunked] {src} is {info.sample_rate} Hz but {model_name} "
            f"wants {sr} Hz; chunking would drift against the video - bailing out.")
        return None

    total = int(info.num_frames)
    if total <= 0:
        log(f"[demucs-chunked] {src} has no frames.")
        return None

    chunk = max(int(chunk_sec * sr), 8 * sr)
    xfade = max(min(int(xfade_sec * sr), chunk // 4), 1)
    hop = chunk - xfade
    names = list(model.sources)
    for key in outputs:
        if key != "no_vocals" and key not in names:
            raise ValueError(f"unknown stem {key!r}; {model_name} has {names}")

    ramp_up = np.linspace(0.0, 1.0, xfade, dtype=np.float32)[:, None]
    ramp_down = 1.0 - ramp_up
    channels = 1 if mono else 2

    writers = {}
    tails = {key: None for key in outputs}
    try:
        for key, path in outputs.items():
            writers[key] = sf.SoundFile(path, "w", samplerate=sr,
                                        channels=channels, subtype="PCM_16")
        log(f"[demucs-chunked] {total/sr/60:.1f} min at {sr} Hz -> "
            f"{chunk/sr/60:.1f} min windows, {xfade/sr:.1f}s crossfade")
        pos = 0
        while True:
            n = min(chunk, total - pos)
            # Absorb a runt tail rather than emit a window too short to crossfade.
            if 0 < total - (pos + n) < 2 * xfade:
                n = total - pos
            wav, _ = torchaudio.load(src, frame_offset=pos, num_frames=n)
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)          # demucs wants stereo
            elif wav.shape[0] > 2:
                wav = wav[:2]
            with torch.no_grad():
                est = apply_model(model, wav[None], device=device, split=True,
                                  overlap=0.1, progress=progress)[0]
            blocks = {}
            for key in outputs:
                if key == "no_vocals":
                    stem = est.sum(dim=0) - est[names.index("vocals")]
                else:
                    stem = est[names.index(key)]
                if mono:
                    stem = stem.mean(dim=0, keepdim=True)
                blocks[key] = stem.cpu().numpy().T.astype(np.float32)  # (L, ch)
            del est

            last = (pos + n) >= total
            for key, arr in blocks.items():
                if tails[key] is None:
                    head = 0
                else:
                    writers[key].write(tails[key] * ramp_down + arr[:xfade] * ramp_up)
                    head = xfade
                body_end = arr.shape[0] if last else arr.shape[0] - xfade
                writers[key].write(arr[head:body_end])
                tails[key] = None if last else arr[arr.shape[0] - xfade:].copy()

            done = pos + n if last else pos + hop
            log(f"[demucs-chunked] {done/sr/60:.1f}/{total/sr/60:.1f} min")
            if last:
                break
            pos += hop
    finally:
        for w in writers.values():
            w.close()
    return outputs


def _main(argv=None):
    """CLI used by the music-bed step: emit vocals + no_vocals for one wav."""
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-vocals", default="")
    ap.add_argument("--out-no-vocals", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--chunk-sec", type=float, default=DEFAULT_CHUNK_SEC)
    ap.add_argument("--xfade-sec", type=float, default=DEFAULT_XFADE_SEC)
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    outputs = {}
    if args.out_vocals:
        outputs["vocals"] = args.out_vocals
    if args.out_no_vocals:
        outputs["no_vocals"] = args.out_no_vocals
    if not outputs:
        print("nothing to do: pass --out-vocals and/or --out-no-vocals",
              file=sys.stderr)
        return 2

    def log(msg):
        if not args.quiet:
            print(msg, flush=True)

    try:
        import torch
        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            log("[demucs-chunked] CUDA unavailable; falling back to CPU")
            device = "cpu"
        ok = separate_to_files(args.input, outputs, device=device,
                               model_name=args.model, chunk_sec=args.chunk_sec,
                               xfade_sec=args.xfade_sec, mono=args.mono, log=log)
    except Exception as e:                                  # noqa: BLE001
        print(f"[demucs-chunked] failed: {e}", file=sys.stderr)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
