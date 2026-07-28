# Foreign-language movie dub — run report (2026-07-17 → 07-23)

Batch-dubbed every foreign-language film in the Plex **Movies** library into English
using the IndexTTS2 expressive clone+emotion engine (variant 4), tagged each into a
pinned Plex collection **`zDUBBED`**, and ran it unattended overnight under a
15-minute crash-monitor.

**Result: 10/10 target foreign films dubbed + The Beast = 11 real dubs in `zDUBBED`.**
(Roma + Apocalypto initially tagged-but-never-produced — caught 2026-07-23, re-rendered
overnight 2026-07-24 and verified on disk: Roma 2,509 MB, Apocalypto 10,595 MB
`-DUBBED.mkv` in their source folders. Verified by checking the actual output FILE,
not the tag/log.)

---

## Discovery (how the shortlist was found)

`plex/plex-audio-language-scan.py` scanned all movies for non-English audio tracks:

| Metric | Count |
|---|---|
| Movies in library (section 33) | 2,745 |
| Movies with **any** non-English audio track | 461 |
| Movies with a Japanese track (raw heuristic) | 129 |
| Movies with **NO English audio at all** (true foreign films) | 35 |

The 461 is mostly Hollywood films bundling dub tracks — **not** foreign films. The
real signal is "no English audio track." From the 35, the user's dub shortlist was
20; **10 had a usable full English *text* subtitle** and were dubbed (the rest were
forced/image-sub-only or not requested).

---

## The 10 dubbed films

Dubs are written **non-destructively** to `<movie>\dubbed\<name>.dubbed.mkv` beside
each source (originals untouched); the English AI track is set default, original
audio + subs preserved.

| # | Film | Orig lang | Cues | Notes |
|---|------|-----------|-----:|-------|
| 1 | Fantastic Planet (1973) | fr | ~sparse | first done; sparse-dialogue art film → fast Phase A |
| 2 | Final Account (2021) | de/en | 949 | dialogue-dense documentary |
| 3 | The Secret Agent (2025) | pt | 1935 | 2160p; longest render (~2.6 h) |
| 4 | Ip Man (2008) | zh | 938 | |
| 5 | Stalker (1979) | ru | 1031 | 1080p FLAC remux (~40 GB dubbed copy) |
| 6 | Come and See (1985) | ru | 734 | failed night 1 (SMB), succeeded night 2 after fix |
| 7 | Kalashnikov AK-47 (2020) | ru | 857 | failed night 1 (SMB), succeeded night 2 via cached embeds |
| 8 | Roma (2018) | es | 886 | re-rendered 2026-07-24 (first tag was empty); 2,509 MB, verified |
| 9 | Apocalypto (2006) | myn (Mayan) | 447 | re-rendered 2026-07-24 (first tag was empty); 10,595 MB, verified |
| 10 | Godzilla Minus One (2023) | ja | 1060 | 2160p TrueHD Atmos remux; the outlier — ~3 h Phase A, 67 GB source-copy-local + 71 GB result-copy-to-NAS |
| + | The Beast (2024) | fr | 847 | done in an earlier session (validation title) |

Deferred/skipped (not dubbed): 3 forced-sub-only titles (Aniara, Godzilla vs.
Destoroyah, The Wandering Earth) and 6 with no usable English text sub (Shin
Godzilla, Hell, The Zone of Interest, Shaolin Soccer, T-34, Cinema Paradiso) — all
need a full-dialogue `<base>.en.srt` sidecar before they can be dubbed.

**Subtitle sourcing (2026-07-24):** to unblock those 9, English subs are being
fetched via `scratchpad/subs_fetch.py` (subliminal, anonymous providers
podnapisi/bsplayer/tvsubtitles, hash-matched) with a local Whisper-translate
fallback (`scratchpad/whisper_subs.py`, faster-whisper large-v3 on the RVA venv).
subliminal landed 3 (Shin Godzilla 1,791 cues, Shaolin Soccer 995, Cinema Paradiso
858 — all validated full-dialogue English). The other 6 went to Whisper
`task=translate` overnight 2026-07-25, all verified full-dialogue English:
Aniara 495 cues, The Wandering Earth 1,050, Hell 213 (sparse film, spans 81m),
The Zone of Interest 568, T-34 748, Godzilla vs. Destoroyah 510 — all written
beside their sources. Godzilla's folder initially denied SMB writes (was
`nobody:users` mode 755, no group write); normalized to 775 (owner-authorized)
and its srt placed + verified. **Net: all 9 previously-blocked films now have a
full-dialogue English `.en.srt` in place, so all 9 are dubbable.** Subs only —
no dubbing kicked off.

---

## Findings (what we learned)

### 1. Phase A cost is driven by dialogue density, not runtime
The slow step is diarization: `_speech_windows` slices 1.5 s windows every 0.75 s,
then embeds each with **one un-batched ECAPA call per window**
(`profile_show.py` `diarize_windows`). Dialogue-dense films → thousands of windows →
Phase A of **2–3 h** (GMO ~3 h; sparse films like Fantastic Planet finish in minutes).
This, not the render, dominates wall-clock. **Un-applied speed lever:** batch the
ECAPA embeds (or widen `hop_s`) to cut Phase A substantially.

### 2. The SMB clip-write bug (the big one)
Phase A originally wrote reference clips to the NAS
(`<movie>\anime-dub-clips\spkNN_*.wav`), which **intermittently failed at the very
end** with `soundfile.LibsndfileError: ... System error` — throwing away a completed
2–3 h Phase A (cost Kalashnikov and Come and See a night each). **Fix:** write the
profile JSON + clips to **local disk** (`G:\Transcode\dubprofile-local_<key>\`) and
have Phase B clone from the local profile; only the final mux still touches the NAS.
Clip paths are stored relative to the profile dir, so co-locating locally keeps them
resolvable. **After the fix: 0 SMB errors across nights 2 and 3.**

### 3. Diagnose "stuck vs slow" by power draw + scoped per-PID CPU/IO — not util% or log mtime
GMO looked "wedged" (100 % GPU util, logs frozen 100+ min) but was really doing
un-batched CUDA work. The reliable tells:
- **GPU power draw** (near-TDP = real compute; ~idle 65 W with 100 % util = pinned
  context / tiny ops), and **temp**.
- **Per-PID** CPU-time advancing + disk-IO deltas — scoped to the worker PID
  (summing all `python.exe` is useless: the RVA doorbell services mask it).
- `[indextts] N/M lines … ~Xm left` in the err-log is the authoritative render ETA;
  the main log is block-buffered and looks frozen mid-phase.

### 4. Other gotchas
- **`-Redub` re-does listed titles** — the runner does NOT skip already-dubbed movies
  when they're passed explicitly, so the nightly title list must be pruned to
  not-done titles each night or it wastes hours re-dubbing completed ones.
- **MusicBed Demucs OOMs on 2 h+ films** (`enforce fail at alloc_cpu.cpp`) → the
  pipeline gracefully falls back to "ducked original" (dub @100 % over original
  @60 %). Acceptable degradation, not a failure.
- **Start-Process arg-quoting** splits titles with spaces → launch via a wrapper
  `.ps1` with `-OnlyTitles @(...)` instead of passing the array through the CLI.
- **Cached embeddings** (`locals_cache.pkl` in the stem dir) make a failed title's
  retry fast — Kalashnikov/Come and See re-ran Phase A in minutes on night 2.

### 5. Throughput
Render ≈ **10–13 cues/min**. A dense/long film ≈ 2–3 h Phase A + 1–2.5 h render +
(for remuxes) large local/NAS copies. ~1–4 titles complete per 9 h overnight window.

---

## Scheduling & safety
- Ran **only overnight 23:00–08:00 AU** (machine is AUS Eastern), never during
  business hours — enforced by the 15-min monitor (launch at 23:00, `stop_dub.ps1`
  at 08:00). Resume-safe across nights.
- Non-destructive throughout; originals never modified.

## Scripts (all in `scratchpad/`, except the two Plex tools)
| Script | Role |
|---|---|
| `scratchpad/dub_foreign_movies.ps1` | batch runner (tiers, `-Stage`, `-OnlyTitles`, local-clip fix, auto-tag) |
| `scratchpad/run_night.ps1` | overnight wrapper (prune to not-done titles) |
| `scratchpad/stop_dub.ps1` | 08:00 kill of launcher tree + workers |
| `scratchpad/movie_voice_tuning.json` | globals-only IndexTTS2 knobs for movies |
| `plex/plex-audio-language-scan.py` | scan library for non-English audio |
| `plex/plex-dub-collection.py` | create/pin `zDUBBED`, tag dubbed movies |
