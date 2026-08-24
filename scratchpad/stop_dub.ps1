# stop_dub.ps1 - kill the foreign-movie dub run and all its workers.
# Used by the 08:00 business-hours guard (Windows scheduled task DubNightStop)
# and callable manually. No GPU needed; safe from any context.
$ErrorActionPreference = 'SilentlyContinue'

# 1) kill the launcher pwsh(es) running the night/deferred/short wrappers or the
#    orchestrator, WITH their child trees (/T catches ffmpeg + python children).
$launchers = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'pwsh.exe' -and
    $_.CommandLine -match 'run_night|run_deferred|run_short|dub_foreign_movies'
}
foreach ($p in $launchers) { & taskkill /PID $p.ProcessId /T /F 2>$null }

# 2) belt-and-suspenders: kill any lingering dub workers (profiler / IndexTTS2
#    render / Demucs) that may have been orphaned from a prior tree.
$workers = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and
    $_.CommandLine -match 'profile_show|srt_to_speech_indextts|\\.venv-dub\\|index-tts\\.venv|demucs'
}
foreach ($p in $workers) { & taskkill /PID $p.ProcessId /T /F 2>$null }

Write-Output ("stop_dub: killed {0} launcher(s), {1} worker(s) at {2}" -f `
    $launchers.Count, $workers.Count, (Get-Date -Format 'HH:mm:ss'))
