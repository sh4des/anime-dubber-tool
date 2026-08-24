<#
  run_cast_queue.ps1 - overnight "build every cast so Dale can pick" run.

  For each pending foreign film: [whisper .en.srt if missing] -> Phase A profile
  -> voice audition. The user reviews the auditions afterwards and chooses a cast
  per movie; NO Phase B render happens here.

  Hard requirement: ONE FILM MUST NEVER BLOCK THE REST. Every film is wrapped in
  try/catch AND given a wall-clock timeout - Colony (2026) burned 3h45m in Phase A
  before dying, which would otherwise have eaten a whole night on its own.

  Resumable: a film whose profile JSON already exists skips Phase A; one whose
  voice-samples report already exists skips the audition. Safe to re-run.

    pwsh ./run_cast_queue.ps1                 # dry-run plan
    pwsh ./run_cast_queue.ps1 -Go
    pwsh ./run_cast_queue.ps1 -Go -SkipSubs
#>
[CmdletBinding()]
param(
    [switch]$Go,
    [switch]$SkipSubs,
    [int]$ProfileTimeoutMin = 120,
    [int]$AuditionTimeoutMin = 45,
    [string[]]$OnlyTitles = @()
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()

$Here    = $PSScriptRoot
$AppDir  = Split-Path $Here -Parent                                    # repo root (scratchpad/ is one level down)
$Scratch = if ($env:DUB_SCRATCH) { $env:DUB_SCRATCH } else { 'G:\Transcode' }
$V       = Join-Path $Scratch '.venv-dub\Scripts\python.exe'
$Idx     = Join-Path $Scratch 'index-tts\.venv\Scripts\python.exe'
$Ck      = Join-Path $Scratch 'index-tts\checkpoints'
$Tune    = Join-Path $Here 'movie_voice_tuning.json'
# Whisper subtitle fallback runs under a CUDA-capable venv that lives outside this
# repo (any venv with faster-whisper + the Blackwell cuBLAS fix will do).
$RvaPy   = if ($env:WHISPER_PY) { $env:WHISPER_PY } else { 'python' }
$SubsPy  = Join-Path $Here 'whisper_subs.py'
$SubsTsv = Join-Path $Here 'cast_queue_subs.tsv'
$OrderTx = Join-Path $Here 'cast_queue_order.txt'
$AuditRoot = Join-Path $Scratch 'voice-auditions'

New-Item -ItemType Directory -Force $Scratch, $AuditRoot | Out-Null
$env:TMP = "$Scratch\tmp"; $env:TEMP = $env:TMP
$env:TTS_HOME = "$Scratch\tts-cache"; $env:HF_HOME = "$Scratch\hf-cache"
$env:PYTHONUNBUFFERED = '1'
# T-34's folder starts with Cyrillic Te (U+0422). Python defaults stdout to cp1252
# here, so merely PRINTING the title raised UnicodeEncodeError and killed Phase A.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
# huggingface_hub tries to symlink into HF_HOME; on Windows without Developer Mode
# that is WinError 1314 "A required privilege is not held", which killed the whole
# whisper subtitle stage before it loaded. Copy instead of symlinking.
$env:HF_HUB_DISABLE_SYMLINKS = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

function Log($m, $c = 'Gray') { Write-Host ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $m) -ForegroundColor $c }

# --- resolve the film list from the shared shortlist -------------------------
$ps1 = Join-Path $Here 'dub_foreign_movies.ps1'
$txt = Get-Content -Raw -LiteralPath $ps1
$done = @('Fantastic Planet (1973)','Stalker (1979)','Come and See (1985)','Apocalypto (2006)',
          'Ip Man (2008)','Roma (2018)','Kalashnikov AK-47 (2020)','Final Account (2021)',
          'Godzilla Minus One (2023)','The Beast (2024)','The Secret Agent (2025)')
$all = @()
foreach ($m in [regex]::Matches($txt, "Title='([^']+)';\s*Folder='([^']+)';\s*Tier='([^']+)'")) {
    $t = $m.Groups[1].Value; $f = $m.Groups[2].Value; $tier = $m.Groups[3].Value
    if ($tier -eq 'inprogress' -or $done -contains $t) { continue }
    $all += [pscustomobject]@{ Title = $t; Folder = $f; Tier = $tier }
}
# apply the curated order (Colony first, then this session's finds, then the rest)
$order = @(); if (Test-Path -LiteralPath $OrderTx) { $order = Get-Content -LiteralPath $OrderTx }
$all = $all | Sort-Object @{ Expression = { $i = $order.IndexOf($_.Title); if ($i -lt 0) { 999 } else { $i } } }
if ($OnlyTitles.Count) { $all = @($all | Where-Object { $OnlyTitles -contains $_.Title }) }

function Get-Key([string]$Folder) { (Split-Path $Folder -Leaf) -replace '[^\w.-]','_' }
function Get-ProfileJson([string]$Folder) { Join-Path $Scratch ("dubprofile-local_" + (Get-Key $Folder) + "\anime-dub-profile.json") }
function Get-ClipDir([string]$Folder)     { Join-Path $Scratch ("dubprofile-local_" + (Get-Key $Folder) + "\anime-dub-clips") }
function Get-AuditDir([string]$Folder)    { Join-Path $AuditRoot (Get-Key $Folder) }

Write-Host "`n=== cast queue :: $($all.Count) film(s) ===" -ForegroundColor Cyan
foreach ($m in $all) {
    $pj = Get-ProfileJson $m.Folder; $ad = Get-AuditDir $m.Folder
    $st = @()
    if (Test-Path -LiteralPath $pj) { $st += 'profile OK' }
    if (Test-Path -LiteralPath (Join-Path $ad 'voice_samples_report.txt')) { $st += 'audition OK' }
    Write-Host ("  [{0,-7}] {1} {2}" -f $m.Tier, $m.Title, ($(if ($st) { "($($st -join ', '))" } else { '' }))) -ForegroundColor DarkGray
}
if (-not $Go) { Write-Host "`n(dry-run - re-run with -Go)`n" -ForegroundColor Yellow; return }

# --- run a child process with a wall-clock cap -------------------------------
# Run a child pwsh under a wall-clock cap.
#
# Arguments are baked into a GENERATED WRAPPER SCRIPT rather than passed on the
# command line. Measured on real movie paths, both command-line routes are broken:
#   raw -ArgumentList array  -> "...\Illang The Wolf Brigade (2018)" is TRUNCATED
#                               at the first space
#   manually quoted          -> survives spaces, but a UNC's leading "\\" is
#                               eaten down to "\", so every NAS path is invalid
# A wrapper has no command line to mangle: the values are PowerShell string
# literals, so spaces, parentheses and "\\" all survive verbatim.
#
# $p.Handle must be touched before WaitForExit(ms) or it never observes the exit
# and every film would falsely "time out" and be killed.
function Invoke-Capped {
    param([string]$ScriptPath, [hashtable]$Params,
          [string]$Exe, [string[]]$ExeArgs,
          [int]$TimeoutMin, [string]$Log)
    $lines = @('$ErrorActionPreference = ''Stop''')
    if ($ScriptPath) {
        $lines += '$p = @{'
        foreach ($k in $Params.Keys) {
            $val = $Params[$k]
            if ($val -is [bool]) {
                $lines += "    '$k' = `$$($val.ToString().ToLower())"   # switches
            } else {
                $lines += "    '$k' = '" + ([string]$val -replace "'", "''") + "'"
            }
        }
        $lines += '}'
        $lines += "& '" + ($ScriptPath -replace "'", "''") + "' @p"
    } else {
        $a = ($ExeArgs | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ', '
        $lines += "& '" + ($Exe -replace "'", "''") + "' @($a)"
    }
    $lines += 'exit $LASTEXITCODE'
    $wrap = [IO.Path]::Combine($env:TMP, "castwrap_" + [Guid]::NewGuid().ToString('N') + ".ps1")
    Set-Content -LiteralPath $wrap -Value $lines -Encoding UTF8

    try {
        $p = Start-Process -FilePath 'pwsh' `
                -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $wrap) `
                -PassThru -NoNewWindow -RedirectStandardOutput $Log -RedirectStandardError "$Log.err"
        $null = $p.Handle
        if (-not $p.WaitForExit($TimeoutMin * 60 * 1000)) {
            Log "    TIMEOUT after ${TimeoutMin}m - killing tree (PID $($p.Id))" 'Red'
            & taskkill /PID $p.Id /T /F 2>&1 | Out-Null
            Start-Sleep 5
            throw "timed out after ${TimeoutMin} min"
        }
        if ($p.ExitCode -ne 0) { throw "exited $($p.ExitCode) (see $Log.err)" }
    } finally {
        Remove-Item -LiteralPath $wrap -ErrorAction SilentlyContinue
    }
}

# Free VRAM right now, in MiB (-1 if nvidia-smi is unavailable).
function Get-FreeVram {
    try {
        $v = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        if ($v) { return [int](($v | Select-Object -First 1).Trim()) }
    } catch { }
    return -1
}

# Wait for GPU headroom before a VRAM-hungry step. Something else on this box
# (ComfyUI-restyle, ollama serving LocalAI) can be holding many GB; starting
# IndexTTS2 into a nearly-full card just OOMs and wastes the slot. Waits up to
# $MaxWaitMin, then proceeds anyway so a permanently-busy GPU cannot stall the
# whole queue - the per-film try/catch still contains any failure.
function Wait-Vram {
    param([int]$NeedMiB, [int]$MaxWaitMin = 20, [string]$What = "step")
    $free = Get-FreeVram
    if ($free -lt 0) { Log "    (nvidia-smi unavailable - skipping VRAM check)"; return }
    if ($free -ge $NeedMiB) { Log "    VRAM free ${free} MiB - ok for $What"; return }
    $deadline = (Get-Date).AddMinutes($MaxWaitMin)
    Log "    only ${free} MiB free, need ${NeedMiB} for $What - waiting up to ${MaxWaitMin}m" 'Yellow'
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 30
        $free = Get-FreeVram
        if ($free -ge $NeedMiB) { Log "    VRAM freed (${free} MiB) - proceeding" 'Green'; return }
    }
    Log "    still only ${free} MiB free after ${MaxWaitMin}m - trying $What anyway" 'Yellow'
}

# --- stage 1: per-film profile -> audition (this is the actual deliverable) --
# Subtitles are deliberately NOT done first: Phase A profiles audio only and the
# audition renders a fixed line, so neither needs a sub. Subs are only required
# for the Phase B render later, so they run last - that way a slow/wedged
# whisper pass can never eat the night before a single cast is produced.
Log "STAGE 1: profile + audition, per film" 'Cyan'
$okP = 0; $okA = 0; $fail = @()
foreach ($m in $all) {
    $title = $m.Title
    $pj = Get-ProfileJson $m.Folder
    $ad = Get-AuditDir $m.Folder
    Log "=== $title ===" 'Cyan'

    # Phase A
    try {
        if (Test-Path -LiteralPath $pj) {
            Log "  profile exists - skip Phase A"; $okP++
        } else {
            Log "  Phase A profile (cap ${ProfileTimeoutMin}m) ..."
            # Demucs now streams segments (measured ~0.6 GB peak), so this is modest.
            Wait-Vram -NeedMiB 2500 -MaxWaitMin 20 -What 'Phase A'
            Invoke-Capped -ScriptPath (Join-Path $AppDir 'subtitle-anime-profile.ps1') -Params @{
                              Folder = $m.Folder; Out = $pj; ClipDir = (Get-ClipDir $m.Folder)
                              Scratch = $Scratch; Python = $V; Diarizer = 'ecapa'; ReuseStems = $true
                          } -TimeoutMin $ProfileTimeoutMin `
                          -Log (Join-Path $Scratch ("cast_profile_" + (Get-Key $m.Folder) + ".log"))
            if (-not (Test-Path -LiteralPath $pj)) { throw "no profile produced" }
            Log "  profile OK" 'Green'; $okP++
        }
    } catch {
        Log "  PHASE A FAILED: $($_.Exception.Message)" 'Red'
        $fail += "$title (profile: $($_.Exception.Message))"
        continue        # no profile -> audition is impossible; next film
    }

    # Phase A.5 audition
    try {
        if (Test-Path -LiteralPath (Join-Path $ad 'voice_samples_report.txt')) {
            Log "  audition exists - skip"; $okA++
        } else {
            Log "  audition (cap ${AuditionTimeoutMin}m) ..."
            # IndexTTS2 is the heavy one - give it real headroom or it just OOMs.
            Wait-Vram -NeedMiB 6000 -MaxWaitMin 20 -What 'IndexTTS2 audition'
            New-Item -ItemType Directory -Force $ad | Out-Null
            Invoke-Capped -ScriptPath (Join-Path $AppDir 'subtitle-anime-sample-voices.ps1') -Params @{
                              Folder = $m.Folder; CloneProfile = $pj; OutDir = $ad
                              Scratch = $Scratch; IndexTtsPython = $Idx; CheckpointsDir = $Ck
                              VoiceTuning = $Tune
                          } -TimeoutMin $AuditionTimeoutMin `
                          -Log (Join-Path $Scratch ("cast_audition_" + (Get-Key $m.Folder) + ".log"))
            Log "  audition OK -> $ad" 'Green'; $okA++
        }
    } catch {
        Log "  AUDITION FAILED: $($_.Exception.Message)" 'Red'
        $fail += "$title (audition: $($_.Exception.Message))"
    }
}

Write-Host "`n=== casts complete ===" -ForegroundColor Green
Log "profiles: $okP/$($all.Count)   auditions: $okA/$($all.Count)"
if ($fail.Count) {
    Write-Host "failed ($($fail.Count)):" -ForegroundColor Yellow
    $fail | ForEach-Object { Write-Host "   - $_" -ForegroundColor Yellow }
}
Log "auditions are under $AuditRoot\<film>\ - listen to _contact_sheet.wav per film"

# --- stage 2: subtitles (prep for the LATER Phase B render, not needed above) -
if (-not $SkipSubs -and (Test-Path -LiteralPath $SubsTsv)) {
    Log "STAGE 2: whisper .en.srt for films with no English text sub" 'Cyan'
    try {
        Invoke-Capped -Exe $RvaPy -ExeArgs @($SubsPy, '--list', $SubsTsv) -TimeoutMin 240 `
                      -Log (Join-Path $Scratch 'cast_queue_subs.log')
        Log "  subtitles stage done" 'Green'
    } catch {
        Log "  subtitles stage FAILED: $($_.Exception.Message)" 'Yellow'
    }
} else {
    Log "STAGE 2 skipped (no $SubsTsv or -SkipSubs)"
}
Log "ALL DONE"
