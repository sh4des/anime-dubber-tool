# run_whisper_night.ps1 (2026-07-24) - overnight (23:00-08:00 AU) launcher for the
# Whisper-translate subtitle fallback. Generates English <base>.en.srt for the 6
# foreign films subliminal couldn't find. Uses the RVA venv (Blackwell cuBLAS fix).
# Resumable: whisper_subs.py skips any film whose .en.srt already exists.
$env:PYTHONUTF8 = '1'
$py = if ($env:WHISPER_PY) { $env:WHISPER_PY } else { 'python' }
$script = Join-Path $PSScriptRoot 'whisper_subs.py'
& $py $script
