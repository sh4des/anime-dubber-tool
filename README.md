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

# --- variant 2: unique voice per CHARACTER, tracked across episodes ----------
# Fingerprints the original Japanese speech under each line, matches it to the
# show's recurring characters, and gives each character their own consistent
# voice for the whole series. State lives in <Folder>\anime-dub-voices.json
# (delete it to reset the cast). New characters are bucketed adult/child x
# male/female by pitch and assigned a distinct voice from that pool.
pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs -All

# with rollback copies:
# pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs -All -BackupOriginal

# tune matching/thresholds per show, e.g.:
#   pwsh ./subtitle-anime-unique-voices.ps1 @DubArgs -All `
#        -MatchThreshold 0.78 -ChildSplit 310 -ChildPitchShift 3
# NOTE: process episodes in order (default sort) on the first pass so characters
# are discovered consistently; the profile is updated after every episode.