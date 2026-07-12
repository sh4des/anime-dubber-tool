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

# confirm 5080 is visible
& $V -c "import torch, torchaudio, transformers, numpy; from TTS.api import TTS; print('ALL OK', torch.__version__, 'cuda', torch.cuda.is_available(), '| tf', transformers.__version__)"


# run script
$env:TTS_HOME = "G:\Transcode\tts-cache"
# The multi-voice script also downloads a ~1GB speech age+gender model
# (audeering/wav2vec2-large-robust-24-ft-age-gender) on first run - it needs
# internet once, then caches. Relocate that cache off C: with:
$env:HF_HOME  = "G:\Transcode\hf-cache"

$Show = "\\10.0.23.105\media\TV\15-18 Animated\Mobile Suit Gundam ZZ"
$DubArgs = @{
    Folder  = $Show
    Scratch = "G:\Transcode"
    Python  = "G:\Transcode\.venv-dub\Scripts\python.exe"
}

# Default output: in-place replace. Each episode is built on local scratch, then
# copied back and swapped over the original (same filename for Plex). Re-runs
# skip episodes that already have an "English Dub (AI)" audio track. Use -All to
# skip the single-episode test prompt and process the whole folder.
#
# Options:
#   -All                 process every episode (no test/confirm step)
#   -BackupOriginal      keep <name>.pre-dub.<ext> before replacing (rollback copy)
#   -UseDubbedFolder     old behaviour: write to <Folder>\dubbed\ instead of replacing

# --- variant 1: basic, single voice for every line ---------------------------
pwsh ./subtitle-anime.ps1 @DubArgs -All

# with a rollback copy of each original before replace:
# pwsh ./subtitle-anime.ps1 @DubArgs -All -BackupOriginal

# keep originals untouched; write dubbed copies to <Folder>\dubbed\ instead:
# pwsh ./subtitle-anime.ps1 @DubArgs -All -UseDubbedFolder

# --- variant 3 (RECOMMENDED): cloned voices, whole show, one command ---------
# Clone each character's ACTUAL (Japanese) voice instead of picking one from a
# pool. Two phases, both local / no signups (Demucs + SpeechBrain ECAPA + XTTS
# download weights anonymously; no audio leaves the machine):
#   PHASE A  profile the whole show  -> anime-dub-profile.json + reference clips
#   PHASE B  dub every episode by cloning each character from that profile
# The orchestrator runs both. Safe to re-run (profiling overwrites; dubbing is
# -Redub + resume-safe: it skips episodes already rebuilt with the cloned engine,
# so an interrupted run just continues). Needs: pip install demucs speechbrain
#   pwsh ./subtitle-anime-dub-show.ps1 @DubArgs
#   pwsh ./subtitle-anime-dub-show.ps1 @DubArgs -BackupOriginal   # rollback copies
#   pwsh ./subtitle-anime-dub-show.ps1 @DubArgs -SkipProfile      # reuse profile
# Or run the phases yourself:
#   pwsh ./subtitle-anime-profile.ps1 @DubArgs -Diarizer ecapa            # Phase A
#   pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs -Clone -Redub -All   # Phase B
# Profiling the FULL show (no -MaxEpisodes) is best: it discovers the whole cast
# and lets every episode dub via the fast, accurate time-overlap path.


# --- variant 2: unique voice per CHARACTER, tracked across episodes ----------
# Fingerprints the original Japanese speech under each line, matches it to the
# show's recurring characters, and gives each character their own consistent
# voice for the whole series. State lives in <Folder>\anime-dub-voices.json
# (delete it to reset the cast). Each new character's GENDER + AGE is read from a
# pretrained speech age+gender model over its original audio, which picks a
# bucket (child / adult / elderly x male / female) and a distinct voice from that
# pool. (Median pitch is only a fallback if the model is off/unavailable.)
# NOTE: a voice is locked when a character is first minted - delete the JSON to
# re-cast an existing show with the current classifier.
pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs -All

# with rollback copies:
# pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs -All -BackupOriginal

# tune matching/thresholds per show, e.g.:
#   pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs -All `
#        -MatchThreshold 0.78 -ChildSplit 310 -ChildPitchShift 3
# NOTE: process episodes in order (default sort) on the first pass so characters
# are discovered consistently; the profile is updated after every episode.

# --- RE-ANALYZE a show you have ALREADY dubbed -------------------------------
# Episodes dubbed in-place already carry an "English Dub (AI)" track, so a plain
# re-run skips them; and each character's voice is locked in the profile JSON.
# To re-analyze from scratch with the current classifier: delete the profile so
# the cast is re-assigned, then -Redub to reprocess (it strips the previous AI
# dub track and rebuilds it - lossless, since the original audio is still inside
# the file and the video is copied, not re-encoded). Process in name order.
#
#   # 1) re-cast: drop the character profile (voices are otherwise locked)
#   Remove-Item -LiteralPath (Join-Path $Show "anime-dub-voices.json") -ErrorAction Ignore
#   # 2) rebuild every episode with the age+gender voice matching
#   pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs -All -Redub
#
# Omit the Remove-Item to keep the existing cast and only re-render (e.g. after
# changing -OriginalVolume or -ElderPitchShift). Use -UseDubbedFolder + -Redub to
# rebuild into <Folder>\dubbed\ instead of replacing in place.


# =============================================================================
# CLI Reference
# =============================================================================
#
# Four command-line entry points. Two are PowerShell batch orchestrators you run
# per show folder; each calls a Python TTS helper once per episode. You normally
# only invoke the PowerShell wrappers - the Python scripts are documented so you
# can run/debug the TTS step directly on a single .srt.
#
#   subtitle-anime.ps1               -> calls srt_to_speech.py            (one voice for all lines)
#   subtitle-anime-unique-voices.ps1 -> calls srt_to_speech_multivoice.py (one voice per character, tracked across episodes)
#
# Single-voice vs multi-voice / unique-voices
# -------------------------------------------
#   * Single voice (subtitle-anime.ps1 / srt_to_speech.py): EVERY subtitle line
#     is spoken by ONE XTTS speaker ("Damien Black" by default). No analysis of
#     the original audio. Fastest, simplest, no extra dependencies.
#   * Unique voices (subtitle-anime-unique-voices.ps1 / srt_to_speech_multivoice.py):
#     fingerprints the ORIGINAL (Japanese) speech under each cue, matches it to a
#     tracked character, and gives each character their own consistent voice for
#     the whole series. Each new character's GENDER + AGE is read from a
#     pretrained speech age+gender model over its original audio (audeering
#     wav2vec2), which picks a bucket - child / adult / elderly x male / female -
#     and a distinct voice from that pool; median pitch is only a fallback. Needs
#     a reference-audio track, a persistent profile JSON, resemblyzer, and
#     mkvmerge. The age+gender model needs no extra pip package (runs on the
#     transformers+torch stack) but downloads ~1GB of weights on first use.
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
#   pwsh ./subtitle-anime.ps1 -Folder "\\10.0.23.105\media\TV\15-18 Animated\Mobile Suit Gundam ZZ"
#   # Whole folder, local scratch on a fast disk, keep rollback copies:
#   pwsh ./subtitle-anime.ps1 -Folder "\\10.0.23.105\media\TV\...\MyShow" `
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
#
# Examples:
#   # Whole show, defaults, in-place; profile auto-created beside the episodes:
#   pwsh ./subtitle-anime-unique-voices.ps1 -Folder "\\10.0.23.105\media\TV\...\MyShow" -All
#   # Tune matching + child handling for a show, with an explicit venv python:
#   pwsh ./subtitle-anime-unique-voices.ps1 -Folder "\\10.0.23.105\media\TV\...\MyShow" `
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
#       --ref-audio ep01.ref.wav --profile "\\10.0.23.105\...\MyShow\anime-dub-voices.json" `
#       --duration 1440 --verbose
#   # Stateless per-cue voices (no cross-episode identity), custom female pool:
#   python srt_to_speech_multivoice.py --srt ep01.srt --out ep01.dub.wav `
#       --ref-audio ep01.ref.wav --voices-adult-female "Sofia Hellen;Ana Florence"