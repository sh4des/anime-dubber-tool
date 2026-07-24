# =============================================================================
# FOREIGN-LANGUAGE MOVIE BATCH (2026-07) — 10/10 Plex foreign films dubbed
# =============================================================================
# Batch-dubbed every foreign-language film in the Plex Movies library and tagged
# each into a pinned Plex collection "zDUBBED". Full write-up + stats + findings:
#   FOREIGN-MOVIES-DUB-REPORT.md   (per-title cues/timings, the SMB clip-write fix,
#                                   diagnosis method, throughput, gotchas)
# Scripts (scratchpad/, run overnight 23:00-08:00 AU under a 15-min crash monitor):
#   dub_foreign_movies.ps1  batch runner: tiers, -Stage profile|sample|render|all,
#                           -OnlyTitles, -IncludeForced/-IncludeSidecar; writes the
#                           profile+clips to LOCAL disk (dubprofile-local_<key>) to
#                           dodge the intermittent NAS clip-write LibsndfileError;
#                           non-destructive -> dubbed\; auto-tags each into zDUBBED.
#   run_night.ps1           overnight wrapper (list ONLY not-done titles: the runner
#                           passes -Redub so listed titles are re-dubbed, not skipped)
#   stop_dub.ps1            kills the launcher tree + workers (the 08:00 cutoff)
#   movie_voice_tuning.json globals-only IndexTTS2 knobs shared by all movies
# Plex tools (../../../plex/): plex-audio-language-scan.py (find non-English audio),
#   plex-dub-collection.py (create/pin zDUBBED at library top, tag dubbed movies).
# A MOVIE is just a one-file folder (see "TARGETING" below); needs a full English
# TEXT sub (image/forced subs are skipped -> supply a <base>.en.srt sidecar).


# install dependencies
New-Item -ItemType Directory -Force "G:\Transcode\tmp","G:\Transcode\pip-cache" | Out-Null
$env:TMP = "G:\Transcode\tmp"; $env:TEMP = "G:\Transcode\tmp"
$env:PIP_CACHE_DIR = "G:\Transcode\pip-cache"

py -3.12 -m venv "G:\Transcode\.venv-dub"
$V = "G:\Transcode\.venv-dub\Scripts\python.exe"

& $V -m pip install --upgrade pip
& $V -m pip install coqui-tts pydub srt "transformers<5"
& $V -m pip install "torch<2.9" "torchaudio<2.9" --index-url https://download.pytorch.org/whl/cu128
# variant 2 only: cross-episode character tracking (speaker fingerprints).
# Omit and the multi-voice script falls back to per-line pitch buckets.
# resemblyzer needs webrtcvad, which has NO prebuilt wheel for py3.12 and tries
# to compile (fails without MS C++ Build Tools). Use the prebuilt drop-in
# webrtcvad-wheels, then install resemblyzer with --no-deps (its other deps -
# numpy/scipy/librosa/torch - already came from coqui-tts + torch above).
& $V -m pip install webrtcvad-wheels
& $V -m pip install --no-deps resemblyzer
& $V -c "from resemblyzer import VoiceEncoder, preprocess_wav; print('resemblyzer OK')"

# variant 3 (RECOMMENDED cloned pipeline): source separation + speaker embeddings.
#   demucs      - isolates the dialogue (vocal) stem from music/SFX
#   speechbrain - ECAPA speaker embeddings for diarization/clustering (no signup;
#                 downloads weights anonymously, all inference local)
& $V -m pip install demucs speechbrain
& $V -c "import demucs, speechbrain; print('demucs + speechbrain OK')"

# confirm 5080 is visible
& $V -c "import torch, torchaudio, transformers, numpy; from TTS.api import TTS; print('ALL OK', torch.__version__, 'cuda', torch.cuda.is_available(), '| tf', transformers.__version__)"


# =============================================================================
# variant 4 engine: IndexTTS2  (RECOMMENDED expressive cloned dub)
# =============================================================================
# IndexTTS2 clones each character's timbre AND transfers the emotion/tone of their
# ORIGINAL Japanese line (emo_audio_prompt), guided by the subtitle text. It needs
# its OWN venv: it pins Python 3.10/3.11 + transformers 4.52 + numpy 1.26, which
# conflict with the coqui/demucs stack in .venv-dub above. Fully local, no signup
# (Bilibili license permits personal use; weights download anonymously).
#
# uv manages the env + a matching Python (no admin, no system 3.11 needed):
py -3.12 -m pip install uv                              # or: winget install astral-sh.uv
git clone https://github.com/index-tts/index-tts.git "G:\Transcode\index-tts"
Push-Location "G:\Transcode\index-tts"
uv sync                                                 # builds .venv (py3.10, torch cu128, indextts)
uv pip install srt                                      # extra dep for the dub engine (numpy/librosa/soundfile come with indextts)
hf download IndexTeam/IndexTTS-2 --local-dir "G:\Transcode\index-tts\checkpoints"
Pop-Location
# smoke check:
& "G:\Transcode\index-tts\.venv\Scripts\python.exe" -c "from indextts.infer_v2 import IndexTTS2; print('IndexTTS2 import OK')"
# The dub scripts default to this engine and expect:
#   IndexTtsPython = G:\Transcode\index-tts\.venv\Scripts\python.exe
#   CheckpointsDir = G:\Transcode\index-tts\checkpoints
# (override with -IndexTtsPython / -CheckpointsDir if you install elsewhere).


# =============================================================================
# run script  (env + shared args - used by every variant below)
# =============================================================================
$env:TTS_HOME = "G:\Transcode\tts-cache"   # XTTS model cache off C:
$env:HF_HOME  = "G:\Transcode\hf-cache"    # HuggingFace cache: age+gender model
                                           # (~1GB), Demucs, ECAPA - all download
                                           # weights ONCE, then run locally.

$Show = "\\10.0.23.105\media\tv\15-18 Animated\Mobile Suit Gundam ZZ"
$DubArgs = @{
    Folder  = $Show
    Scratch = "G:\Transcode"
    Python  = "G:\Transcode\.venv-dub\Scripts\python.exe"
}

# Output is in-place by default: each episode is built on local scratch, then
# copied back and swapped over the original (same filename, so Plex keeps its
# entry). The original audio + subs are preserved inside; only an English dub
# track is added and set default. Common options (all variants):
#   -All                 process every episode (skip the single-episode prompt)
#   -BackupOriginal      keep <name>.pre-dub.<ext> before replacing (rollback)
#   -UseDubbedFolder     write to <Folder>\dubbed\ instead of replacing in place


# =============================================================================
# variant 3 (RECOMMENDED): CLONED voices, whole show, one command
# =============================================================================
# Clone each character's ACTUAL (Japanese) voice instead of assigning one from a
# pool - the English line inherits the source timbre. Fully local, no signups
# (Demucs + SpeechBrain ECAPA + XTTS download weights anonymously; no audio ever
# leaves the machine). Two phases, run by one orchestrator:
#   PHASE A  profile the WHOLE show  -> <Folder>\anime-dub-profile.json,
#            a QC report (.qc.txt), and reference clips in <Folder>\anime-dub-clips\
#   PHASE B  dub every episode by cloning each character from that profile;
#            cues are matched to a character by time-overlap (profiled episodes)
#            or by embedding, with a pool-voice fallback for weak matches.
# Safe to re-run: profiling overwrites the profile; dubbing is -Redub and
# resume-safe (skips episodes already rebuilt with the cloned track), so an
# interrupted run just continues.

# whole series, end to end (profile all -> clone-dub all):
pwsh ./subtitle-anime-dub-show.ps1 @DubArgs

# with a .pre-dub rollback copy of each original:
# pwsh ./subtitle-anime-dub-show.ps1 @DubArgs -BackupOriginal

# reuse an existing profile (skip Phase A), just (re)dub:
# pwsh ./subtitle-anime-dub-show.ps1 @DubArgs -SkipProfile

# same thing, phases run by hand (equivalent to the orchestrator):
#   pwsh ./subtitle-anime-profile.ps1        @DubArgs -Diarizer ecapa          # Phase A (all eps)
#   pwsh ./subtitle-anime-unique-voices.ps1  @DubArgs -Clone -Redub -All -FitToCues  # Phase B

# tune the cast if the QC report looks over/under-split, then re-dub:
#   pwsh ./subtitle-anime-profile.ps1 @DubArgs -Diarizer ecapa -ReuseStems -ClusterThreshold 0.5
#   pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs -Clone -Redub -All


# =============================================================================
# variant 4 (RECOMMENDED): EXPRESSIVE cloned dub (IndexTTS2)
# =============================================================================
# Same two-phase model as variant 3, but Phase B uses IndexTTS2 (-Engine indextts,
# the default): it clones each character's timbre AND transfers the EMOTION/tone of
# their original Japanese line (per-cue emo_audio_prompt), guided by the subtitle
# text. Needs the separate IndexTTS2 venv (install block above) + a Phase A profile.
# Phase A is identical to variant 3 (Demucs + ECAPA diarization -> global cast +
# per-episode `turns` in anime-dub-profile.json). Fully local, no audio leaves the
# box. Quality knobs learned in production (Gundam ZZ, 47 eps):
#   -EmotionStemsDir   point at the Phase A stem cache
#                      (<Scratch>\dubprofile-stems_<show>); the emotion reference is
#                      then the CLEAN Demucs vocal stem, not the raw mix - removes
#                      the music/reverb bleed that makes clones sound echoey.
#   -EmoAlpha 0.45     emotion strength; higher bleeds the prompt's ACOUSTICS into
#                      the clone. Loud/shouty leads distort at >0.35.
#   -VoiceTuning <json>  per-character + global knobs (see scratchpad/voice_tuning
#                      .json): per-speaker emo_alpha/gain_db, plus global temperature
#                      0.5, max_text_tokens_per_segment 400, fixed seed, and
#                      match_source:true (scale each dubbed line to the RMS of the
#                      ORIGINAL line it replaces, +/-12 dB clamp - keeps the mix's
#                      dynamics; whole-track loudnorm stays OFF or it flattens intent).
#
# Full show, expressive clone, in place:
$Idx = "G:\Transcode\index-tts\.venv\Scripts\python.exe"
$Ck  = "G:\Transcode\index-tts\checkpoints"
$Stems = "G:\Transcode\dubprofile-stems_Mobile_Suit_Gundam_ZZ"
$Tune  = "$PSScriptRoot\scratchpad\voice_tuning.json"   # or your own knobs json
# pwsh ./subtitle-anime-profile.ps1 @DubArgs -Diarizer ecapa       # Phase A (once)
# pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs `
#      -Clone -Engine indextts -Redub -Force -All -FitToCues -MusicBed `
#      -EmoAlpha 0.45 -EmotionStemsDir $Stems -VoiceTuning $Tune `
#      -IndexTtsPython $Idx -CheckpointsDir $Ck


# =============================================================================
# TARGETING: shows, seasons, movies, and subsets
# =============================================================================
# The orchestrators operate on a -Folder and process every VIDEO FILE in it (top
# level only, non-recursive). Filenames need no SxxExx pattern; -OnlyEpisodes /
# -ExcludeEpisodes / -StartFrom just substring-match the filename. The Phase A
# profile (cast + reference clips) is anchored to the folder that holds
# anime-dub-profile.json - reference-clip paths resolve relative to the PROFILE's
# own directory, so -CloneProfile can point elsewhere while -Folder differs.
#
# --- a whole SHOW (all episodes in one flat folder) --------------------------
#   pwsh ./subtitle-anime-dub-show.ps1 @DubArgs            # profile all + dub all
#
# --- a single SEASON in its own subfolder ------------------------------------
# Point -Folder at the season dir; the profile/clips/dubs live there too:
#   $S1 = Join-Path $Show "Season 1"
#   pwsh ./subtitle-anime-dub-show.ps1 -Folder $S1 -Scratch "G:\Transcode" -Python $V
# To share ONE show-wide cast across season subfolders instead, profile the whole
# show once, then dub each season with -CloneProfile pointing at the show profile:
#   pwsh ./subtitle-anime-unique-voices.ps1 -Folder $S1 -Clone -Engine indextts `
#        -CloneProfile (Join-Path $Show "anime-dub-profile.json") `
#        -Redub -Force -All -FitToCues -MusicBed -EmotionStemsDir $Stems -VoiceTuning $Tune `
#        -IndexTtsPython $Idx -CheckpointsDir $Ck
#
# --- a MOVIE (single file) ---------------------------------------------------
# A movie is just a one-file folder. Profile that one file (its own cast), then dub:
#   $Movie = "\\10.0.23.105\media\movies\Akira (1988)"     # folder containing the mkv
#   pwsh ./subtitle-anime-dub-show.ps1 -Folder $Movie -Scratch "G:\Transcode" -Python $V
# The profile's `turns` are keyed by the movie's filename; multi-voice works from a
# single file. For a franchise, reuse one profile across movies via -CloneProfile.
#
# --- specific episodes / ranges ----------------------------------------------
#   -OnlyEpisodes "S01E07"                 just one (pilot / re-do a single episode)
#   -OnlyEpisodes "S01E07","S01E11"        a handful
#   -ExcludeEpisodes "S01E20","S01E21"     skip already-good ones during a -Force redo
#   -StartFrom "S01E22"                    process E22..end, then wrap E01..E21
#   (all of the above take -All; combine with -Redub -Force to rebuild in place.)


# =============================================================================
# RECIPES / gotchas from real runs
# =============================================================================
# * UNPROFILED / late-arriving episodes voice as ONE speaker. The IndexTTS2 engine
#   assigns characters from the profile's per-episode `turns` (keyed by filename);
#   an episode not present during Phase A has no turns -> every cue uses the
#   fallback voice. Fix WITHOUT re-clustering the whole cast (which would renumber
#   speakers and break voice_tuning + clip refs): profile just the new episodes to
#   a TEMP profile, map each local speaker to the nearest existing-cast centroid
#   (cosine), and inject their `turns` into the main profile under the mapped
#   global id. See scratchpad/profile_e03_e12.ps1 + inject_turns.py.
# * READ-ONLY target folder (e.g. a season subfolder the share won't let you write):
#   in-place install fails with "Access to the path ... is denied". Copy each source
#   into a writable sibling (the show's parent folder) and dub there; the profile's
#   relative clip paths still resolve because they anchor to the profile dir.
# * DAMAGED source ("remux recovered only N of M s - re-acquire this file"): the
#   length guard refuses to dub a truncated source. To dub the RECOVERABLE part,
#   stream-copy remux it first so the container's declared duration matches the
#   intact content, then dub that (drops the unreadable tail):
#     ffmpeg -nostdin -y -err_detect ignore_err -i bad.mkv `
#            -map 0:v:0 -map 0:a:0 -map 0:s:0 -c copy repaired.mkv
# * ASS/embedded TEXT subtitles are extracted automatically; only IMAGE subs
#   (PGS/VOBSUB) are skipped (can't be converted to text) - supply a sidecar .srt.


# =============================================================================
# variant 2: unique voice per CHARACTER from a POOL (no cloning)
# =============================================================================
# Older approach: fingerprint each line's original speech, track the character
# across episodes, and give them a distinct XTTS pool voice matched by GENDER +
# AGE (audeering age+gender model; median pitch is only a fallback). State lives
# in <Folder>\anime-dub-voices.json (delete to reset the cast). Use this if you
# want clean built-in voices rather than clones.
# NOTE: a voice is locked when a character is first minted - delete the JSON to
# re-cast with the current classifier.
pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs -All

# tune matching/thresholds per show, e.g.:
#   pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs -All `
#        -MatchThreshold 0.78 -ChildSplit 310 -ChildPitchShift 3


# =============================================================================
# variant 1: basic, single voice for every line
# =============================================================================
# No analysis of the original audio - every line uses one XTTS speaker. Fastest.
pwsh ./subtitle-anime.ps1 @DubArgs -All

# --- RE-ANALYZE / RE-DUB a show you have ALREADY dubbed ----------------------
# Episodes dubbed in-place already carry an "English Dub (AI)" track. -Redub
# strips the old AI track and rebuilds it (lossless: the original audio stays in
# the file, video is copied not re-encoded).
#
# Cloned pipeline (variant 3): just re-run the orchestrator. Phase A overwrites
# the profile; Phase B is -Redub + resume-safe, so it rebuilds every episode:
#   pwsh ./subtitle-anime-dub-show.ps1 @DubArgs                 # re-profile + re-dub all
#   pwsh ./subtitle-anime-dub-show.ps1 @DubArgs -SkipProfile    # keep profile, re-dub only
#
# Pool pipeline (variant 2): voices are locked in the profile JSON, so to re-cast
# from scratch delete it first, then -Redub. Process in name order:
#   Remove-Item -LiteralPath (Join-Path $Show "anime-dub-voices.json") -ErrorAction Ignore
#   pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs -All -Redub
# Omit the Remove-Item to keep the cast and only re-render. -UseDubbedFolder +
# -Redub rebuilds into <Folder>\dubbed\ instead of replacing in place.


# =============================================================================
# CLI Reference
# =============================================================================
#
# PowerShell orchestrators you run per show folder; each calls a Python helper
# once per episode. You normally only invoke the PowerShell wrappers - the Python
# scripts are documented so you can run/debug a single .srt directly.
#
#   subtitle-anime-dub-show.ps1       -> runs Phase A then Phase B (cloned)   [variant 3, RECOMMENDED]
#   subtitle-anime-profile.ps1        -> calls profile_show.py            (Phase A: build the show voice profile)
#   subtitle-anime-unique-voices.ps1  -> calls srt_to_speech_cloned.py    (Phase B, with -Clone)
#                                     -> calls srt_to_speech_multivoice.py (pool engine, without -Clone)  [variant 2]
#   subtitle-anime.ps1                -> calls srt_to_speech.py            (one voice for all lines)       [variant 1]
#
# The three engines
# -----------------
#   * CLONED (variant 3 - subtitle-anime-dub-show.ps1 / profile_show.py +
#     srt_to_speech_cloned.py): Phase A profiles the whole show - Demucs isolates
#     dialogue, ECAPA embeddings diarize + cluster speakers across ALL episodes,
#     and each speaker gets gender/age + reference clips (anime-dub-profile.json).
#     Phase B voices each character by CLONING their own reference clips with XTTS
#     (matching cues by time-overlap or embedding), so the English inherits the
#     source timbre. Falls back to a pool voice for weak matches. All local, no
#     signups; needs demucs + speechbrain + mkvmerge (weights download once).
#   * POOL (variant 2 - subtitle-anime-unique-voices.ps1 / srt_to_speech_multivoice.py):
#     fingerprints each cue's ORIGINAL speech, tracks the character across
#     episodes, and assigns a distinct XTTS pool voice by GENDER + AGE (audeering
#     age+gender model; median pitch is only a fallback). Needs resemblyzer +
#     mkvmerge; the age+gender model runs on the transformers+torch stack.
#   * SINGLE (variant 1 - subtitle-anime.ps1 / srt_to_speech.py): EVERY line is
#     one XTTS speaker ("Damien Black" by default). No original-audio analysis.
#     Fastest, simplest, no extra dependencies.
#
#
# -----------------------------------------------------------------------------
# 1) subtitle-anime.ps1  (single-voice batch orchestrator)
# -----------------------------------------------------------------------------
# Purpose: for a folder of anime episodes, locate English subtitles (sidecar .srt
#   or an embedded text subtitle), synthesize a single-voice English dub with
#   srt_to_speech.py, mix it over the original audio (original ducked to 60%),
#   and remux so Plex auto-selects the new "English Dub (AI)" track while the
#   original audio + subs are preserved. Copies each source to local scratch,
#   works locally, then replaces the file in place (default). Re-runs skip
#   episodes that already carry an "English Dub (AI)" audio track.
# Requires: ffmpeg + ffprobe on PATH; a Python with srt_to_speech.py's deps.
# Invocation:
#   pwsh ./subtitle-anime.ps1 -Folder <path> [options]
#
# Parameters (name  type  default  meaning):
#   -Folder           string  (required)   Show folder of episodes to process.
#   -All              switch  off          Process every episode; skip the single-
#                                          episode test + [y/N] confirm prompt.
#   -UseDubbedFolder  switch  off          Keep originals; write results to
#                                          <Folder>\dubbed\<name>.dubbed.mkv
#                                          instead of replacing in place.
#   -BackupOriginal   switch  off          Before an in-place replace, copy the
#                                          original once to <name>.pre-dub.<ext>.
#   -OriginalVolume   double  0.6          Volume the original audio is mixed to
#                                          under the dub (0.6 = 60%).
#   -Speaker          string  "Damien Black"  XTTS built-in speaker for all lines.
#   -SpeakerWav       string  ""           Path to a voice sample to CLONE instead
#                                          of -Speaker (overrides -Speaker).
#   -FitToCues        switch  off          Speed up dub lines that overrun the next
#                                          cue (passes --fit; cap 1.6x).
#   -Scratch          string  %TEMP%       Local fast disk for the working copy
#                                          (needs ~2x one episode free).
#   -Python           string  "python"     Python interpreter for the TTS helper.
#                                          If not passed, auto-uses .\.venv\
#                                          Scripts\python.exe when present.
#   -Verbose          switch  ON           On by default; run -Verbose:$false to
#                                          quiet the per-step logging.
#
# Examples:
#   # Test on the first episode, then confirm before doing the rest:
#   pwsh ./subtitle-anime.ps1 -Folder "\\10.0.23.105\media\tv\15-18 Animated\Mobile Suit Gundam ZZ"
#   # Whole folder, local scratch on a fast disk, keep rollback copies:
#   pwsh ./subtitle-anime.ps1 -Folder "\\10.0.23.105\media\tv\15-18 Animated\Mobile Suit Gundam ZZ" `
#        -All -Scratch "G:\Transcode" -BackupOriginal `
#        -Python "G:\Transcode\.venv-dub\Scripts\python.exe"
#
#
# -----------------------------------------------------------------------------
# 2) subtitle-anime-unique-voices.ps1  (multi-voice / per-character orchestrator)
# -----------------------------------------------------------------------------
# Purpose: same pipeline as subtitle-anime.ps1, but each line is spoken by the
#   voice of the CHARACTER who says it, tracked across episodes via a persistent
#   profile JSON. Extracts one original audio stream as a reference for speaker
#   classification, and first remuxes each source into a clean Matroska container
#   with mkvmerge (ffmpeg's demuxer aborts on some playable-but-malformed MKVs
#   and silently truncates the dub). Writes an "English Dub (AI, multi-voice)"
#   track. Same in-place/scratch/backup behaviour as the basic script.
# Requires: ffmpeg + ffprobe on PATH; mkvmerge (MKVToolNix); a Python with the
#   base deps plus `pip install resemblyzer` (webrtcvad-wheels) for cross-episode
#   tracking. Without resemblyzer it degrades to per-line pitch buckets.
# Invocation:
#   pwsh ./subtitle-anime-unique-voices.ps1 -Folder <path> [options]
#
# Parameters (name  type  default  meaning):
#   -Folder            string    (required)   Show folder of episodes.
#   -All               switch    off          Skip the single-episode test prompt.
#   -UseDubbedFolder   switch    off          Write to <Folder>\dubbed\ instead of
#                                             replacing in place.
#   -BackupOriginal    switch    off          Copy original to <name>.pre-dub.<ext>
#                                             once before an in-place replace.
#   -Redub             switch    off          Re-process episodes that ALREADY
#                                             have an AI dub track: strip the old
#                                             AI track and rebuild it (lossless -
#                                             the original audio stays in the file).
#                                             Also delete the profile JSON first to
#                                             RE-CAST voices, else the locked-in
#                                             voices are simply re-rendered.
#   -OriginalVolume    double    0.6          Original-audio volume under the dub.
#   -ProfilePath       string    <Folder>\anime-dub-voices.json
#                                             Show-level character profile JSON;
#                                             recurring characters keep their voice
#                                             across episodes. Delete to reset cast.
#   -MatchThreshold    double    0.75         Cosine similarity to treat a line as
#                                             an EXISTING character. Higher = more
#                                             distinct characters (more splits);
#                                             lower = more merging.
#   -VoicesAdultMale   string[]  @()          Override the adult-male voice pool
#                                             (XTTS speaker names). Empty = built-in
#                                             pool (Damien Black, Viktor Eka,
#                                             Baldur Sanjin, Craig Gutsy,
#                                             Aaron Dreschner, Marcos Rudaski).
#   -VoicesAdultFemale string[]  @()          Override adult-female pool. Built-in:
#                                             Alison Dietlinde, Sofia Hellen,
#                                             Ana Florence, Gracie Wise,
#                                             Daisy Studious, Brenda Stern.
#   -VoicesElderlyMale   string[] @()         Override elderly-male pool. Built-in:
#                                             Baldur Sanjin, Marcos Rudaski,
#                                             Damien Black.
#   -VoicesElderlyFemale string[] @()         Override elderly-female pool. Built-in:
#                                             Brenda Stern, Daisy Studious,
#                                             Ana Florence.
#   -VoicesChildMale   string[]  @()          Override child-male pool. Built-in:
#                                             Andrew Chipper, Craig Gutsy.
#   -VoicesChildFemale string[]  @()          Override child-female pool. Built-in:
#                                             Tammie Ema, Gracie Wise.
#   -VoiceDefault      string    ""           Voice for lines with no clear speaker
#                                             (music/SFX/overlap). Default = first
#                                             adult-male pool voice.
#   --- gender/age classification (primary) ---
#   -DisableAgeGender  switch    off          Turn the age+gender model OFF and use
#                                             the pitch fallback below for gender/age.
#   -ElderAge          double    58           Predicted age (yrs) at/above which a
#                                             new character is bucketed elderly.
#   -ElderPitchShift   double    1.5          Semitones to LOWER elderly clips to
#                                             age them (0 = disable).
#   -AgeGenderModel    string    ""           Override the HF age+gender model id.
#   --- pitch fallback (only when the model is off/unavailable) ---
#   -MaleMax           double    155          Pitch (Hz) below which a NEW character
#                                             is bucketed adult male.
#   -AdultFemaleMax    double    250          Below this (and >=MaleMax) -> adult
#                                             female.
#   -ChildSplit        double    300          Below this (and >=AdultFemaleMax) ->
#                                             child male; else child female.
#   -ChildPitchShift   double    2.0          Semitones to raise child clips so
#                                             adult XTTS voices read younger
#                                             (0 = disable).
#   -RefAudioIndex     int       0            Which ORIGINAL audio stream to analyse
#                                             (0 = first, usually Japanese).
#   -FitToCues         switch    off          Speed up dub lines that overrun.
#   -Scratch           string    %TEMP%       Local scratch disk (~2x one episode).
#   -Python            string    "python"     TTS-helper interpreter; auto-uses
#                                             .\.venv\Scripts\python.exe if present.
#   -Mkvmerge          string    "mkvmerge"   mkvmerge binary; if not on PATH, auto-
#                                             uses %ProgramFiles%\MKVToolNix\
#                                             mkvmerge.exe when present.
#   -Verbose           switch    ON           On by default; -Verbose:$false quiets.
#   --- Phase B cloned engine (variants 3 & 4; requires a Phase A profile) ------
#   -Clone             switch    off          Use the cloned engine (clone each
#                                             character's own voice) instead of the
#                                             pool engine. Needs anime-dub-profile.json.
#   -Engine            enum      indextts      indextts = IndexTTS2 (timbre + emotion,
#                                             own venv); xtts = Coqui clone (timbre only).
#   -CloneProfile      string    <Folder>\anime-dub-profile.json
#                                             Phase A profile to clone from. Point
#                                             elsewhere to reuse a show-wide cast when
#                                             -Folder is a season/subset (clip paths
#                                             resolve relative to the profile's dir).
#   -EmoAlpha          double    0.45         IndexTTS2 emotion strength 0..1 (>0.35
#                                             distorts loud leads; high bleeds acoustics).
#   -EmotionStemsDir   string    ""           Phase A stem cache; emotion ref = clean
#                                             Demucs vocal stem, not the raw mix (de-reverb).
#   -VoiceTuning       string    ""           JSON of per-character + global knobs
#                                             (emo_alpha/gain_db/temperature/seed/
#                                             max_text_tokens_per_segment/match_source/
#                                             loudnorm). See scratchpad/voice_tuning.json.
#   -MusicBed          switch    off          Lay the dub over a Demucs music+SFX bed
#                                             (JP dialogue removed) instead of ducking.
#   -BedVolume         double    0.9          Music+SFX bed level under the dub.
#   -Redub             switch    off          Strip the existing AI dub track + rebuild.
#   -Force             switch    off          Bypass the resume-skip (redo this pass).
#   -OnlyEpisodes      string[]  @()          Process only filenames containing these
#                                             substrings (e.g. "S01E07").
#   -ExcludeEpisodes   string[]  @()          Skip filenames containing these.
#   -StartFrom         string    ""           Rotate order: this episode..end, then wrap.
#   -IndexTtsPython    string    G:\Transcode\index-tts\.venv\Scripts\python.exe
#                                             IndexTTS2 venv interpreter (variant 4).
#   -CheckpointsDir    string    G:\Transcode\index-tts\checkpoints   IndexTTS2 weights.
#
# Examples:
#   # Whole show, defaults, in-place; profile auto-created beside the episodes:
#   pwsh ./subtitle-anime-unique-voices.ps1 -Folder "\\10.0.23.105\media\tv\15-18 Animated\Mobile Suit Gundam ZZ" -All
#   # Tune matching + child handling for a show, with an explicit venv python:
#   pwsh ./subtitle-anime-unique-voices.ps1 -Folder "\\10.0.23.105\media\tv\15-18 Animated\Mobile Suit Gundam ZZ" `
#        -All -MatchThreshold 0.78 -ChildSplit 310 -ChildPitchShift 3 `
#        -Python "G:\Transcode\.venv-dub\Scripts\python.exe"
#
#
# -----------------------------------------------------------------------------
# 3) srt_to_speech.py  (single-voice TTS helper - called by subtitle-anime.ps1)
# -----------------------------------------------------------------------------
# Purpose: read an .srt and render ONE timed English dub WAV (24kHz mono), every
#   line spoken by the same XTTS voice and placed at its subtitle timestamp.
#   Engine: Coqui XTTS-v2 (tts_models/multilingual/multi-dataset/xtts_v2), GPU
#   accelerated on CUDA. Deduplicates stacked/overlapping ASS cues to avoid echo.
# Requires: coqui-tts, pydub (ffmpeg on PATH), srt, torch/torchaudio (CUDA 12.8+
#   for RTX 50-series).
# Env vars: COQUI_TOS_AGREED - set to "1" automatically so the XTTS non-commercial
#   license prompt stays non-interactive. TTS_HOME - honoured by the TTS lib to
#   relocate the model cache (see the install block above).
# Exit codes: 0 ok; 2 no subtitle cues; 3 a Python dependency is missing.
# Invocation:
#   python srt_to_speech.py --srt <in.srt> --out <out.wav> [options]
#
# Arguments (flag  type  default  meaning):
#   --srt          str    (required)      Input .srt path.
#   --out          str    (required)      Output .wav path.
#   --duration     float  0.0             Video duration (s); pads the track so it
#                                         is at least this long.
#   --language     str    en              XTTS synthesis language.
#   --speaker      str    "Damien Black"  XTTS built-in speaker name.
#   --speaker-wav  str    None            Path to a voice sample to CLONE instead
#                                         of --speaker (takes precedence).
#   --fit          flag   off             Time-compress a line that overruns the
#                                         next cue's start.
#   --max-speed    float  1.6             Cap for --fit speed-up factor.
#   --verbose      flag   off             Log every synthesized line.
#
# Examples:
#   python srt_to_speech.py --srt ep01.srt --out ep01.dub.wav --duration 1440 --verbose
#   python srt_to_speech.py --srt ep01.srt --out ep01.dub.wav --speaker-wav mynarrator.wav --fit
#
#
# -----------------------------------------------------------------------------
# 4) srt_to_speech_multivoice.py  (multi-voice TTS helper - called by
#    subtitle-anime-unique-voices.ps1)
# -----------------------------------------------------------------------------
# Purpose: same job as srt_to_speech.py, but each cue is voiced by the CHARACTER
#   who speaks it. Pass 1 slices the --ref-audio under each cue, computes a
#   resemblyzer d-vector, and matches it to the nearest known character
#   (>= --match-threshold) or mints a new one. Each new character's GENDER + AGE
#   is then read from a pretrained speech age+gender model (audeering wav2vec2)
#   over its aggregated original speech, which places it in a bucket
#   (child / adult / elderly x male / female) and hands it a DISTINCT voice from
#   that bucket's pool. Median pitch is only a fallback (--no-age-gender, model
#   load failure, or too little clean speech). Pass 2 synthesizes each line in its
#   character's voice, aging children up and elders down by a small pitch shift.
#   With --profile, character identities persist across episodes (invoke once per
#   episode, in name order); a voice is locked when a character is first minted,
#   so delete the profile to re-cast. Without resemblyzer it degrades to stateless
#   per-cue classification and says so.
# Requires: everything srt_to_speech.py needs, plus resemblyzer (and torchaudio,
#   used for the pitch fallback). The age+gender model needs no extra pip package
#   (runs on transformers+torch) but downloads ~1GB of weights on first use.
#   Env vars: same as srt_to_speech.py, plus HF_HOME to relocate the model cache.
# Config file: --profile JSON stores each character's id, bucket, voice, averaged
#   voice centroid, line count and pitch; rewritten after every run.
# Exit codes: 0 ok; 2 no subtitle cues; 3 a Python dependency is missing;
#   4 the reference audio could not be loaded.
# Invocation:
#   python srt_to_speech_multivoice.py --srt <in.srt> --out <out.wav> --ref-audio <ref.wav> [options]
#
# Arguments (flag  type  default  meaning):
#   --srt                  str    (required)   Input .srt path.
#   --out                  str    (required)   Output .wav path.
#   --ref-audio            str    (required)   Mono WAV of the ORIGINAL speech
#                                              track, aligned to the subtitle
#                                              timing; used to identify speakers.
#   --profile              str    None         Show-level tracked-characters JSON.
#                                              Omit to disable cross-episode
#                                              tracking (per-episode buckets only).
#   --duration             float  0.0          Video duration (s); pads the track.
#   --language             str    en           XTTS synthesis language.
#   --voices-adult-male    str    None         ';'-separated XTTS speaker names to
#                                              override the adult-male pool.
#   --voices-adult-female  str    None         Override the adult-female pool.
#   --voices-elderly-male   str   None         Override the elderly-male pool.
#   --voices-elderly-female str   None         Override the elderly-female pool.
#   --voices-child-male    str    None         Override the child-male pool.
#   --voices-child-female  str    None         Override the child-female pool.
#   --voice-default        str    None         Voice for cues with no clear speaker
#                                              (default: first adult-male voice).
#   --match-threshold      float  0.75         Cosine similarity to treat a cue as
#                                              an existing character.
#   --- gender/age (primary) ---
#   --age-gender-model     str    audeering/wav2vec2-large-robust-24-ft-age-gender
#                                              HF id of the speech age+gender model.
#   --no-age-gender        flag   off          Disable the model; pitch buckets only.
#   --age-gender-device    str    None         cuda|cpu for the age+gender model.
#   --elder-age            float  58.0         Age (yrs) >= this -> elderly bucket.
#   --elder-pitch-shift    float  1.5          Semitones to LOWER elderly clips
#                                              (0 = disable).
#   --- pitch fallback (only when the model is off/unavailable) ---
#   --male-max             float  155.0        Pitch (Hz) upper bound for adult male.
#   --adult-female-max     float  250.0        Upper bound for adult female.
#   --child-split          float  300.0        Below -> child male; above -> child
#                                              female.
#   --child-pitch-shift    float  2.0          Semitones to raise child clips
#                                              (0 = disable).
#   --embed-device         str    None         cuda|cpu for the speaker encoder
#                                              (auto-detected when omitted).
#   --fit                  flag   off           Time-compress overrunning lines.
#   --max-speed            float  1.6          Cap for --fit speed-up factor.
#   --verbose              flag   off          Per-line + per-character logging
#                                              (shows each character's gender/age).
#
# Built-in voice pools (used when a --voices-* override is not given):
#   adult_male:     Damien Black, Viktor Eka, Baldur Sanjin, Craig Gutsy,
#                   Aaron Dreschner, Marcos Rudaski
#   adult_female:   Alison Dietlinde, Sofia Hellen, Ana Florence, Gracie Wise,
#                   Daisy Studious, Brenda Stern
#   elderly_male:   Baldur Sanjin, Marcos Rudaski, Damien Black
#   elderly_female: Brenda Stern, Daisy Studious, Ana Florence
#   child_male:     Andrew Chipper, Craig Gutsy
#   child_female:   Tammie Ema, Gracie Wise
#
# Examples:
#   # Track characters across a show (profile persists between episode runs):
#   python srt_to_speech_multivoice.py --srt ep01.srt --out ep01.dub.wav `
#       --ref-audio ep01.ref.wav --profile "\\10.0.23.105\media\tv\15-18 Animated\Mobile Suit Gundam ZZ\anime-dub-voices.json" `
#       --duration 1440 --verbose
#   # Stateless per-cue voices (no cross-episode identity), custom female pool:
#   python srt_to_speech_multivoice.py --srt ep01.srt --out ep01.dub.wav `
#       --ref-audio ep01.ref.wav --voices-adult-female "Sofia Hellen;Ana Florence"


# -----------------------------------------------------------------------------
# 5) subtitle-anime-dub-show.ps1  (RECOMMENDED end-to-end orchestrator)
# -----------------------------------------------------------------------------
# Purpose: one command for a whole show - runs Phase A (subtitle-anime-profile.ps1)
#   then Phase B (subtitle-anime-unique-voices.ps1 -Clone -Redub -All). This is the
#   standard entry point for a new series. Fully local, no signups.
# Requires: ffmpeg/ffprobe, mkvmerge, and the venv with demucs + speechbrain.
# Invocation:
#   pwsh ./subtitle-anime-dub-show.ps1 -Folder <path> [options]
# Parameters (name  type  default  meaning):
#   -Folder          string  (required)  Show folder.
#   -Scratch         string  %TEMP%      Local fast disk for extraction/stems.
#   -Python          string  "python"    Interpreter (auto .\.venv if present).
#   -Mkvmerge        string  "mkvmerge"  mkvmerge binary (passed to Phase B).
#   -Diarizer        string  "ecapa"     Phase A backend (ecapa=no signup).
#   -SkipProfile     switch  off         Reuse existing anime-dub-profile.json.
#   -FreshStems      switch  off         Force fresh Demucs separation in Phase A.
#   -BackupOriginal  switch  off         Keep <name>.pre-dub.<ext> before replacing.
#   -UseDubbedFolder switch  off         Write to <Folder>\dubbed\ instead of in place.
#   -NoFit           switch  off         Disable cue duration-fit (on by default here).
#   -ProfileEpisodes int     0           Cap Phase A to first N episodes (0 = all).
# Example:
#   pwsh ./subtitle-anime-dub-show.ps1 `
#       -Folder "\\10.0.23.105\media\tv\15-18 Animated\Mobile Suit Gundam ZZ" `
#       -Scratch "G:\Transcode" -Python "G:\Transcode\.venv-dub\Scripts\python.exe"
#
#
# -----------------------------------------------------------------------------
# 6) subtitle-anime-profile.ps1 / profile_show.py  (Phase A: build the profile)
# -----------------------------------------------------------------------------
# Purpose: profile a show ONCE from the original audio. Extracts each episode's
#   audio, then profile_show.py: Demucs vocal isolation -> diarization -> GLOBAL
#   speaker clustering across all episodes -> per-speaker gender/age + top-K
#   reference clips + a fallback pool voice. Writes:
#     <Folder>\anime-dub-profile.json   speakers (+ centroid, refs, per-ep turns)
#     <Folder>\anime-dub-profile.qc.txt review report (cast, buckets, ref counts)
#     <Folder>\anime-dub-clips\*.wav    reference clips Phase B clones from
# Requires: ffmpeg/ffprobe; venv with demucs; a diarizer backend - ecapa
#   (speechbrain, no signup) or resemblyzer (installed, cruder) or pyannote
#   (gated, needs a HF account - avoid if you don't want signups).
# Key ps1 parameters:
#   -Folder          (required)   Show folder.
#   -Diarizer        "auto"       auto|ecapa|resemblyzer|pyannote (prefer ecapa).
#   -MaxEpisodes     0            Profile only first N (0 = all; all is best).
#   -ReuseStems      off          Reuse cached Demucs stems (fast re-tuning).
#   -FreshStems      off          Clear the cached stems first.
#   -MaxSpeakers     40           Keep at most N speakers (by speech time).
#   -ClusterThreshold / -LocalThreshold   cosine-DISTANCE knobs (smaller = more
#                                 distinct). Leave unset for per-backend defaults;
#                                 lower if the QC report over-merges, raise if it
#                                 over-splits. Stems cache in
#                                 <Scratch>\dubprofile-stems_<show> for -ReuseStems.
# Example (whole show, no signup):
#   pwsh ./subtitle-anime-profile.ps1 `
#       -Folder "\\10.0.23.105\media\tv\15-18 Animated\Mobile Suit Gundam ZZ" `
#       -Scratch "G:\Transcode" -Python "G:\Transcode\.venv-dub\Scripts\python.exe" `
#       -Diarizer ecapa
#
#
# -----------------------------------------------------------------------------
# 7) srt_to_speech_cloned.py  (Phase B TTS helper - called with -Clone)
# -----------------------------------------------------------------------------
# Purpose: render the timed English dub by CLONING each character's own voice
#   from the Phase A profile. Matches each cue to a profiled speaker by
#   time-overlap (if the episode has stored turns) or by embedding its
#   Demucs-cleaned audio vs speaker centroids, then clones that speaker's
#   reference clips with XTTS (conditioning cached per speaker). Weak matches or
#   speakers without clean clips fall back to a pool voice.
# Requires: coqui-tts, pydub, srt, torch; profile_show.py's helpers (demucs +
#   the profile's encoder) for the embedding path on un-profiled episodes.
# Exit codes: 0 ok; 2 no cues; 3 a dependency is missing; 4 bad profile/ref/backend.
# Key arguments:
#   --srt --out --profile --ref-audio   (required)
#   --episode-name   name used in the profile turns (enables time-overlap match)
#   --match-min-sim  min cosine similarity for the embedding match (per-backend)
#   --no-demucs-ref  skip Demucs on the ref before embedding (faster, worse)
#   --fit / --max-speed   duration-fit overrunning lines
# Note: v1 is cloning + duration-fit. Pitch/energy prosody transfer and seed-vc
#   accent polish are planned follow-ups.