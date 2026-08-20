# anime-dubber-tool

Adds an **English AI dub track** to anime / foreign-language video files, driven by their
English subtitles. In the recommended mode each character is voiced by a clone of their
*own* original (Japanese) voice, and the emotion of the original line is transferred onto
the English one. The original audio and subtitles stay in the file — the dub is added as a
new track and set default, so Plex picks it up automatically and keeps its library entry.

Windows + PowerShell 7 + an NVIDIA GPU (built and run on an RTX 5080). Everything runs
locally: no accounts, no uploads, no audio ever leaves the machine. Model weights download
once from HuggingFace, then all inference is local.

---

## 3 examples

All three assume the [one-time setup](#one-time-setup) is done. `G:\Transcode` is just the
fast local scratch disk this repo was built around — point it anywhere with room for a
couple of episodes.

```powershell
$Py = "G:\Transcode\.venv-dub\Scripts\python.exe"     # venv A, see setup
```

### 1. Smoke test — one episode, one voice

```powershell
pwsh ./subtitle-anime.ps1 -Folder "\\10.0.23.105\media\tv\My Show" `
     -Scratch "G:\Transcode" -Python $Py
```

Dubs the **first** episode only, then asks whether to continue with the rest. Every line is
spoken by one XTTS voice — no analysis of the original audio, nothing to profile. This is
the fastest way to confirm ffmpeg, the venv, the GPU and the subtitle detection all work.
Add `-All` to do the whole folder without the prompt.

### 2. The real thing — a whole show, cloned voices + emotion

```powershell
pwsh ./subtitle-anime-dub-show.ps1 -Folder "\\10.0.23.105\media\tv\My Show" `
     -Scratch "G:\Transcode" -Python $Py -MusicBed -BackupOriginal
```

One command, two phases:

* **Phase A** profiles the *entire* show — Demucs isolates the dialogue, ECAPA embeddings
  diarize and cluster speakers across all episodes, each speaker gets gender/age plus
  reference clips. Writes `anime-dub-profile.json`, a `.qc.txt` report, and
  `anime-dub-clips\` next to the episodes.
* **Phase B** dubs every episode with IndexTTS2: clone each character's timbre from those
  clips, transfer the emotion of the line being replaced, mix, remux.

`-MusicBed` lays the dub over a Demucs music+SFX bed (original dialogue removed) instead of
just ducking the original — no Japanese-voice bleed, and laughs / screams / OP-ED singing
are kept. `-BackupOriginal` leaves a `<name>.pre-dub.mkv` rollback copy.

Budget hours per show. Ctrl-C is safe: Phase B is resume-safe, so re-running the same
command picks up where it stopped. For a noticeably cleaner clone, also pass the Phase A
stem cache as the emotion reference:

```powershell
     -EmotionStemsDir "G:\Transcode\dubprofile-stems_My_Show"
```

That directory is `<Scratch>\dubprofile-stems_<folder name>`, with anything outside
`[A-Za-z0-9_.-]` replaced by `_`. It makes the emotion reference the clean vocal stem
rather than the raw mix, which removes the reverb/echo bleed.

### 3. One movie

```powershell
pwsh ./subtitle-anime-dub-show.ps1 -Folder "\\10.0.23.105\media\movies\Akira (1988)" `
     -Scratch "G:\Transcode" -Python $Py -MusicBed -UseDubbedFolder
```

A movie is just a folder that happens to hold one video file — same command, same two
phases, cast built from that single file. `-UseDubbedFolder` writes to `<Folder>\dubbed\`
instead of replacing the original in place.

The film needs a full English **text** subtitle: an embedded SRT/ASS track, or a
`<base>.en.srt` sidecar next to the video. Image subtitles (PGS/VOBSUB) and forced/partial
subs are skipped — supply a sidecar for those.

---

## What happens to my files

* **In place by default.** Each source is copied to `-Scratch`, all work happens on local
  disk, then the finished file replaces the original under the *same filename* so Plex keeps
  its entry. The original audio and all subtitles are preserved inside; only a new audio
  track is added and marked default.
* `-UseDubbedFolder` writes `<Folder>\dubbed\<name>.dubbed.mkv` and never touches the source.
* `-BackupOriginal` copies the original to `<name>.pre-dub.<ext>` once before replacing it.
* The new track is titled `English Dub (AI, ...)` — that marker is how re-runs detect an
  already-dubbed episode and skip it, and how `-Redub` finds the old track to strip.
* Video is always stream-copied, never re-encoded.
* Scratch needs roughly **2x one episode** free, plus ~150 MB/episode for Demucs stems in
  Phase A.

---

## One-time setup

### 1. External tools

| Tool | Why | Get it |
|---|---|---|
| `ffmpeg` + `ffprobe` | extracts, mixes and remuxes everything | <https://www.gyan.dev/ffmpeg/builds/> |
| `mkvmerge` (MKVToolNix) | each source is remuxed into a clean container first — ffmpeg's demuxer aborts partway through some playable-but-malformed MKVs and silently truncates the dub | <https://mkvtoolnix.download/> |
| PowerShell 7 (`pwsh`) | all orchestrators | `winget install Microsoft.PowerShell` |

`ffmpeg` / `ffprobe` must be on `PATH`. `mkvmerge` is found on `PATH`, or automatically at
`%ProgramFiles%\MKVToolNix\mkvmerge.exe`, or pass `-Mkvmerge <path>`.

### 2. venv A — profiler + XTTS (Python 3.12)

This is the `-Python` you pass to every script. It runs Phase A (Demucs + ECAPA), the
single-voice and pool engines, and the music-bed builder.

```powershell
New-Item -ItemType Directory -Force "G:\Transcode\tmp","G:\Transcode\pip-cache" | Out-Null
$env:TMP = "G:\Transcode\tmp"; $env:TEMP = "G:\Transcode\tmp"
$env:PIP_CACHE_DIR = "G:\Transcode\pip-cache"

py -3.12 -m venv "G:\Transcode\.venv-dub"
$Py = "G:\Transcode\.venv-dub\Scripts\python.exe"

& $Py -m pip install --upgrade pip
& $Py -m pip install coqui-tts pydub srt "transformers<5"
& $Py -m pip install "torch<2.9" "torchaudio<2.9" --index-url https://download.pytorch.org/whl/cu128
& $Py -m pip install demucs speechbrain          # Phase A: separation + speaker embeddings

# Optional — only for the older pool engine's cross-episode tracking.
# resemblyzer needs webrtcvad, which has no py3.12 wheel and tries to compile; use the
# prebuilt drop-in, then install resemblyzer with --no-deps (numpy/scipy/librosa/torch
# already came from the lines above).
& $Py -m pip install webrtcvad-wheels
& $Py -m pip install --no-deps resemblyzer

# Verify the GPU is visible and the stack imports
& $Py -c "import torch, demucs, speechbrain; from TTS.api import TTS; print('OK', torch.__version__, 'cuda', torch.cuda.is_available())"
```

### 3. venv B — IndexTTS2 (Python 3.10, the default dub engine)

IndexTTS2 needs its **own** environment: it pins Python 3.10/3.11 + transformers 4.52 +
numpy 1.26, which conflict with the stack above. `uv` builds it and fetches a matching
Python — no admin, no system 3.10 needed. Fully local; the Bilibili licence permits personal
use and the weights download anonymously.

```powershell
py -3.12 -m pip install uv                        # or: winget install astral-sh.uv
git clone https://github.com/index-tts/index-tts.git "G:\Transcode\index-tts"
Push-Location "G:\Transcode\index-tts"
uv sync                                           # builds .venv (py3.10, torch cu128)
uv pip install srt                                # extra dep the dub engine needs
hf download IndexTeam/IndexTTS-2 --local-dir "G:\Transcode\index-tts\checkpoints"
Pop-Location

& "G:\Transcode\index-tts\.venv\Scripts\python.exe" -c "from indextts.infer_v2 import IndexTTS2; print('IndexTTS2 OK')"
```

The scripts default to exactly these two paths:

* `-IndexTtsPython G:\Transcode\index-tts\.venv\Scripts\python.exe`
* `-CheckpointsDir G:\Transcode\index-tts\checkpoints`

Install elsewhere and you must pass both on every call.

### 4. Keep the caches off C:

Set these once per shell (or in your profile) so ~10 GB of model weights and scratch don't
land on the system drive:

```powershell
$env:TTS_HOME = "G:\Transcode\tts-cache"    # XTTS models
$env:HF_HOME  = "G:\Transcode\hf-cache"     # Demucs, ECAPA, age+gender (~1 GB)
```

`subtitle-anime-sample-voices.ps1` sets these itself from `-Scratch`.

---

## Recommended workflow for a new show

Example 2 above is the one-shot version. When you care about the result, run the phases by
hand so you can audition and tune the cast **before** committing to the multi-hour render.

```powershell
$Show  = "\\10.0.23.105\media\tv\My Show"
$Py    = "G:\Transcode\.venv-dub\Scripts\python.exe"
$Idx   = "G:\Transcode\index-tts\.venv\Scripts\python.exe"
$Ck    = "G:\Transcode\index-tts\checkpoints"
$Stems = "G:\Transcode\dubprofile-stems_My_Show"
$Tune  = "G:\Transcode\my-show-voice-tuning.json"      # see "Voice tuning" below

# PHASE A — profile the whole show (once). Heavy; minutes per episode.
pwsh ./subtitle-anime-profile.ps1 -Folder $Show -Scratch "G:\Transcode" -Python $Py -Diarizer ecapa

# Read the QC report at <Show>\anime-dub-profile.qc.txt: cast size, buckets, reference-clip
# counts per speaker. Over-split or over-merged? Re-cluster without re-running Demucs:
#   pwsh ./subtitle-anime-profile.ps1 -Folder $Show -Scratch "G:\Transcode" -Python $Py `
#        -Diarizer ecapa -ReuseStems -ClusterThreshold 0.5

# PHASE A.5 — audition every voice (one line each, through the same path Phase B uses).
pwsh ./subtitle-anime-sample-voices.ps1 -Folder $Show -Scratch "G:\Transcode" `
     -IndexTtsPython $Idx -CheckpointsDir $Ck -VoiceTuning $Tune
# -> <Show>\voice-samples\: per-speaker WAVs, _contact_sheet.wav, voice_samples_report.txt
#    Add -ListOnly for an instant static check (which voices would be SILENT), no GPU.

# PHASE B — render the show.
pwsh ./subtitle-anime-unique-voices.ps1 -Folder $Show -Scratch "G:\Transcode" -Python $Py `
     -Clone -Engine indextts -All -Redub -FitToCues -MusicBed `
     -EmoAlpha 0.45 -EmotionStemsDir $Stems -VoiceTuning $Tune `
     -IndexTtsPython $Idx -CheckpointsDir $Ck
```

Fix a bad voice by editing the tuning JSON or replacing its reference clip in
`anime-dub-clips\`, re-auditioning, and only then rendering. A silent or echoey lead found at
audition costs seconds; found after the render it costs the whole show.

---

## Targeting: shows, seasons, movies, subsets

The orchestrators take a `-Folder` and process every video file in it — **top level only,
non-recursive**. Filenames need no `SxxExx` pattern; the episode filters are plain substring
matches.

| Target | How |
|---|---|
| Whole show, flat folder | `-Folder <show>` |
| One season in a subfolder | `-Folder "<show>\Season 1"` — profile, clips and dubs all live there |
| One show-wide cast across season subfolders | profile the show once, then dub each season with `-CloneProfile "<show>\anime-dub-profile.json"` |
| A movie | `-Folder` = the folder holding the one video file |
| One episode | `-OnlyEpisodes "S01E07"` |
| A handful | `-OnlyEpisodes "S01E07","S01E11"` |
| Skip known-good ones | `-ExcludeEpisodes "S01E20","S01E21"` |
| Start mid-season, wrap around | `-StartFrom "S01E22"` — E22..end, then E01..E21 |

Reference-clip paths inside a profile resolve **relative to the profile's own directory**, so
`-CloneProfile` can point at a completely different folder than `-Folder`. That is what makes
the shared-cast and read-only-folder workarounds below possible.

---

## Voice tuning

`-VoiceTuning <json>` feeds per-character and global knobs to IndexTTS2. Every field is
optional. The values below are the ones that worked in production (Gundam ZZ, 47 episodes):

```json
{
  "global": {
    "emo_alpha": 0.45,
    "temperature": 0.5,
    "max_text_tokens_per_segment": 400,
    "seed": 12345,
    "match_source": true,
    "loudnorm": false
  },
  "speakers": {
    "3":  { "emo_alpha": 0.30, "gain_db": -2.0 },
    "11": { "emo_alpha": 0.55, "gain_db":  1.5 }
  }
}
```

* `emo_alpha` — emotion strength, 0..1. Above ~0.35 IndexTTS2 starts bleeding the prompt's
  **acoustics** (room reverb) into the clone, not just its prosody; loud or shouty leads
  distort. Turn it *down* per speaker for those.
* `match_source: true` — scale each dubbed line to the RMS of the original line it replaces
  (+/-12 dB clamp). Keeps the mix's dynamics.
* `loudnorm` — whole-track EBU pass. Leave **off**: it flattens intent. This key also gates
  the loudness filter in the mux step, not just the render.
* `speakers` keys are the numeric speaker ids from the Phase A QC report.

Other recognised globals: `no_split_under`, `target_peak`, `match_floor_db`,
`match_default_rms_db`, `match_min_gain_db`, `match_max_gain_db`.

---

## Re-dubbing and rollback

An episode dubbed in place already carries an `English Dub (AI...)` track, so plain re-runs
skip it. To rebuild:

```powershell
# cloned pipeline: re-profile + re-dub everything
pwsh ./subtitle-anime-dub-show.ps1 -Folder $Show -Scratch "G:\Transcode" -Python $Py

# keep the existing cast, just re-render
pwsh ./subtitle-anime-dub-show.ps1 -Folder $Show -Scratch "G:\Transcode" -Python $Py -SkipProfile

# one episode, forcing past the resume-skip (e.g. testing mix settings)
pwsh ./subtitle-anime-unique-voices.ps1 -Folder $Show -Clone -All -Redub -Force `
     -OnlyEpisodes "S01E07" -Scratch "G:\Transcode" -Python $Py
```

`-Redub` strips the previous AI track and rebuilds it. This is lossless — the original audio
is still inside the dubbed file and the video is copied, not re-encoded. `-Force` bypasses
the resume-skip; don't use it for a full run, since the skip is what lets an interrupted run
continue instead of redoing finished episodes.

To roll back, restore the `<name>.pre-dub.<ext>` copy `-BackupOriginal` left behind. Without
it there is no undo for an in-place run — use `-UseDubbedFolder` if that matters.

For the older **pool** engine, a character's voice is locked when first minted. Delete
`<Folder>\anime-dub-voices.json` to re-cast, then `-Redub`.

---

## Troubleshooting

**An episode is voiced entirely by one speaker.** It wasn't in the Phase A profile.
IndexTTS2 assigns characters from the profile's per-episode `turns`, keyed by filename; an
episode with no turns falls back to a single voice for every cue. Re-profiling the whole show
fixes it but renumbers speakers, which invalidates your tuning JSON and clip references —
instead profile just the new episodes to a temp profile, map each local speaker to the
nearest existing centroid by cosine distance, and inject those `turns` into the main profile
under the mapped global id.

**`Access to the path ... is denied`.** The target folder is read-only (common for season
subfolders on a share). Copy the sources into a writable sibling and dub there — the
profile's clip paths anchor to the profile directory, so the cast still resolves.

**`remux recovered only N of M s - re-acquire this file`.** The length guard is refusing a
truncated source. To dub the recoverable part, stream-copy it first so the container's
declared duration matches the intact content, then dub the result:

```powershell
ffmpeg -nostdin -y -err_detect ignore_err -i bad.mkv -map 0:v:0 -map 0:a:0 -map 0:s:0 -c copy repaired.mkv
```

**Subtitles not found.** Embedded **text** subs (SRT / ASS / SSA / mov_text / WebVTT) are
extracted automatically. Image subs (PGS / VOBSUB) can't be converted to text and are
skipped — drop a `<base>.en.srt` sidecar next to the video.

**Clones sound echoey.** Pass `-EmotionStemsDir` so the emotion reference is the clean Demucs
vocal stem rather than the raw mix, and lower `emo_alpha` for that speaker.

**Intermittent `LibsndfileError` writing reference clips to a NAS.** Write the profile and
clips to local disk (`-Out` / `-ClipDir`) and point Phase B at them with `-CloneProfile`.

---

## Script reference

PowerShell orchestrators are what you run; each calls a Python helper once per episode. The
Python scripts are usable directly if you want to debug a single `.srt` — every one has a
full argument list in its own header comment and `--help`.

| Script | Role |
|---|---|
| `subtitle-anime-dub-show.ps1` | **Start here.** Phase A then Phase B for a whole show, one command |
| `subtitle-anime-profile.ps1` | Phase A: build the show voice profile, via `profile_show.py` |
| `subtitle-anime-sample-voices.ps1` | Phase A.5: audition the cast, via `sample_voices_indextts.py` |
| `subtitle-anime-unique-voices.ps1` | Phase B (with `-Clone`), via `srt_to_speech_indextts.py` or `srt_to_speech_cloned.py`; also the older pool engine (without `-Clone`), via `srt_to_speech_multivoice.py` |
| `subtitle-anime.ps1` | Single voice for every line, via `srt_to_speech.py` |
| `build_music_bed.py` | Builds the music+SFX bed; called by Phase B when `-MusicBed` is set |

### The engines

| Engine | Invocation | What it does |
|---|---|---|
| **IndexTTS2 clone + emotion** (default, recommended) | `-Clone -Engine indextts` | Clones each character's timbre from their Phase A clips *and* transfers the emotion of the original line. Needs venv B and a Phase A profile |
| **XTTS clone** | `-Clone -Engine xtts` | Timbre-only clone from the same profile. Runs in venv A |
| **Pool** | `subtitle-anime-unique-voices.ps1` without `-Clone` | No cloning: fingerprints each cue, tracks the character across episodes, hands it a distinct built-in XTTS voice by gender + age. State in `anime-dub-voices.json`. Needs resemblyzer |
| **Single** | `subtitle-anime.ps1` | One XTTS voice for everything. No analysis, fewest dependencies, fastest |

### `subtitle-anime-dub-show.ps1`

| Parameter | Default | Meaning |
|---|---|---|
| `-Folder` | *required* | Show / season / movie folder |
| `-Scratch` | `%TEMP%` | Fast local disk for the working copy + stems |
| `-Python` | `python` | venv A interpreter (auto-uses `.\.venv\Scripts\python.exe` if present) |
| `-Mkvmerge` | `mkvmerge` | mkvmerge binary, passed to Phase B |
| `-Diarizer` | `ecapa` | `auto` / `ecapa` / `pyannote` / `resemblyzer`. `ecapa` needs no signup |
| `-SkipProfile` | off | Reuse the existing `anime-dub-profile.json` (skip Phase A) |
| `-FreshStems` | off | Force fresh Demucs separation (otherwise cached stems are reused) |
| `-ProfileEpisodes` | `0` | Cap Phase A to the first N episodes. `0` = all, which gives the best cast *and* lets every episode use the fast time-overlap match |
| `-Engine` | `indextts` | `indextts` or `xtts` |
| `-IndexTtsPython` | `G:\Transcode\index-tts\.venv\Scripts\python.exe` | venv B interpreter |
| `-CheckpointsDir` | `G:\Transcode\index-tts\checkpoints` | IndexTTS2 weights |
| `-EmoAlpha` | `0.45` | Emotion strength 0..1 |
| `-EmotionStemsDir` | `""` | Phase A stem cache; makes the emotion reference the clean vocal stem |
| `-MusicBed` / `-BedVolume` | off / `0.9` | Dub over a Demucs music+SFX bed instead of ducking the original |
| `-BackupOriginal` | off | Keep `<name>.pre-dub.<ext>` |
| `-UseDubbedFolder` | off | Write to `<Folder>\dubbed\` |
| `-NoFit` | off | Disable cue duration-fit, which is **on** by default in this wrapper |
| `-StartFrom` | `""` | Phase B render order: this episode..end, then wrap |

Phase B is always invoked with `-Clone -Redub -All`. Note this wrapper does **not** pass
`-VoiceTuning` — for tuned runs call the two phases directly, as in the workflow above.

### `subtitle-anime-profile.ps1` (Phase A)

Extracts each episode's original audio, then runs `profile_show.py` once over the whole set:
Demucs vocal isolation, diarization, global speaker clustering across all episodes,
gender/age + top-K reference clips. Writes `anime-dub-profile.json`,
`anime-dub-profile.qc.txt` and `anime-dub-clips\*.wav`.

| Parameter | Default | Meaning |
|---|---|---|
| `-Folder` | *required* | Show folder |
| `-Out` / `-ClipDir` | beside the episodes | Profile JSON / reference-clip dir |
| `-Scratch` | `%TEMP%` | Extracted audio + Demucs stems (~150 MB/episode) |
| `-Diarizer` | `auto` | `ecapa` (SpeechBrain, no signup, best local quality), `resemblyzer` (cruder), `pyannote` (gated, needs a HF account + `-HfToken`) |
| `-MaxEpisodes` | `0` | Profile only the first N. `0` = all |
| `-ReuseStems` | off | Reuse cached stems — makes threshold re-tuning fast, since Demucs is the slow part |
| `-FreshStems` | off | Clear the cached stem dir first |
| `-NoDemucs` | off | Skip source separation entirely |
| `-MaxSpeakers` | `40` | Keep at most N speakers by speech time; the tail is usually noise |
| `-ClusterThreshold` / `-LocalThreshold` | per-backend | Cosine **distance** — smaller = more distinct speakers. Lower it if the QC report over-merges, raise it if it over-splits |
| `-RefAudioIndex` | `0` | Which original audio stream to analyse (0 = first, usually Japanese) |
| `-HfToken` | `$env:HF_TOKEN` | Only for `pyannote` |
| `-Python` | `python` | venv A interpreter |

Stems cache at `<Scratch>\dubprofile-stems_<folder name>` — that's the path to hand to
`-EmotionStemsDir` in Phase B.

### `subtitle-anime-unique-voices.ps1` (Phase B, and the pool engine)

| Parameter | Default | Meaning |
|---|---|---|
| `-Folder` | *required* | Show folder |
| `-All` | off | Process every episode; skip the single-episode test + confirm prompt |
| `-Clone` | off | Use the cloned engine. Requires a Phase A profile |
| `-Engine` | `indextts` | `indextts` (timbre + emotion, venv B) or `xtts` (timbre only) |
| `-CloneProfile` | `<Folder>\anime-dub-profile.json` | Profile to clone from. Point elsewhere to reuse a show-wide cast |
| `-EmoAlpha` | `0.45` | Emotion strength 0..1 |
| `-EmotionStemsDir` | `""` | Phase A stem cache, for a clean emotion reference |
| `-VoiceTuning` | `""` | Tuning JSON (see above) |
| `-IndexTtsPython` / `-CheckpointsDir` | `G:\Transcode\index-tts\...` | venv B interpreter / weights |
| `-FitToCues` | off | Time-compress lines that overrun the next cue (cap 1.6x) |
| `-MusicBed` | off | Dub over a Demucs music+SFX bed. Adds ~1-2 min/episode |
| `-BedVolume` | `0.9` | Bed level under the dub (no competing dialogue, so it can sit high) |
| `-BedResidualGain` | `1.0` | Level of the **original vocal stem kept outside dialogue cues** — laughs, screams, grunts, breath SFX, OP/ED singing. `1.0` = original level |
| `-BedDialoguePad` | `0.12` | Extra seconds muted each side of a cue, for loose subtitle timing |
| `-BedFade` | `0.03` | Ramp in/out of each muted region, seconds |
| `-OriginalVolume` | `0.6` | Original-audio level under the dub when *not* using `-MusicBed` |
| `-Redub` | off | Strip the existing AI dub track and rebuild it (lossless) |
| `-Force` | off | Bypass the resume-skip for episodes already dubbed by this engine |
| `-ReuseDub` | off | Reuse a cached dub WAV from `<Scratch>\dubwav-cache` and skip the TTS render — lets you re-run just the mux/install step |
| `-OnlyEpisodes` / `-ExcludeEpisodes` | `@()` | Filename substring filters |
| `-StartFrom` | `""` | Rotate order: this episode..end, then wrap |
| `-UseDubbedFolder` / `-BackupOriginal` | off | Output location / rollback copy |
| `-Scratch` / `-Python` / `-Mkvmerge` | `%TEMP%` / `python` / `mkvmerge` | Tool locations |
| `-RefAudioIndex` | `0` | Which original audio stream to analyse |

Pool-engine-only parameters (unused with `-Clone`): `-ProfilePath`
(`<Folder>\anime-dub-voices.json`), `-MatchThreshold` `0.75`, the XTTS speaker-name pools
`-VoicesAdultMale` / `-VoicesAdultFemale` / `-VoicesElderlyMale` / `-VoicesElderlyFemale` /
`-VoicesChildMale` / `-VoicesChildFemale` / `-VoiceDefault`, `-DisableAgeGender`,
`-ElderAge` `58`, `-ElderPitchShift` `1.5`, `-AgeGenderModel`, and the pitch fallback
`-MaleMax` `155` / `-AdultFemaleMax` `250` / `-ChildSplit` `300` / `-ChildPitchShift` `2.0`.

### `subtitle-anime.ps1` (single voice)

`-Folder` (required), `-All`, `-Speaker` (`"Damien Black"`), `-SpeakerWav` (clone from a
sample instead), `-FitToCues`, `-OriginalVolume` `0.6`, `-UseDubbedFolder`,
`-BackupOriginal`, `-Scratch`, `-Python`.

### `subtitle-anime-sample-voices.ps1` (audition)

`-Folder` (required), `-CloneProfile` (pass this if the profile isn't at
`<Folder>\anime-dub-profile.json`), `-OutDir` (`<Folder>\voice-samples`), `-VoiceTuning`,
`-EmoAlpha` `0.45`, `-Text` (the line everyone says), `-Speakers` (e.g. `"1,2,5"`),
`-ListOnly` (static silent-voice check, no GPU), `-Fp16`, `-Scratch`, `-IndexTtsPython`,
`-CheckpointsDir`.

### Python helpers

All take `--srt` and `--out`, plus `--duration` to pad the track to the video length,
`--fit` / `--max-speed 1.6` for duration-fitting, and `--verbose`. Exit codes: `0` ok,
`2` no subtitle cues, `3` a Python dependency is missing, `4` bad reference audio or profile.

| Script | Extra required args |
|---|---|
| `srt_to_speech.py` | none (`--speaker` / `--speaker-wav` optional) |
| `srt_to_speech_multivoice.py` | `--ref-audio`; optional `--profile` for cross-episode tracking |
| `srt_to_speech_cloned.py` | `--profile`, `--ref-audio`; optional `--episode-name` |
| `srt_to_speech_indextts.py` | `--profile`, `--ref-audio`, `--checkpoints-dir` (run with venv B) |
| `sample_voices_indextts.py` | `--profile`, `--checkpoints-dir`, `--out-dir` (venv B) |
| `build_music_bed.py` | `--no-vocals`, `--vocals`, `--srt`, `--out` |
| `profile_show.py` | driven by `subtitle-anime-profile.ps1`; see its header |

Lyric cues (music-note lines) and bare sound tags (`[laughs]`, `(screams)`) are never spoken
— the music bed's retained original audio carries the real singing and the real laugh
instead.

---

## Also in this repo

* [FOREIGN-MOVIES-DUB-REPORT.md](FOREIGN-MOVIES-DUB-REPORT.md) — write-up of the batch run
  that dubbed every foreign-language film in a 2,745-title Plex library: per-title cues and
  timings, throughput, the SMB clip-write failure and its fix, and the diagnosis method. The
  batch runner scripts it describes live in a scratchpad and are not part of this repo.

The three fixes in the most recent commit — non-verbal audio retained in the music bed,
lyric and sound-tag cues skipped, and the Demucs VRAM fix that made films over ~2h viable —
post-date the last full-show run and have not yet been through one end to end.
