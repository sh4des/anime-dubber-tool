# subtitle-anime-profile.ps1
# -----------------------------------------------------------------------------
# PHASE A wrapper: build a show-level VOICE PROFILE from the original (Japanese)
# audio, ONCE per show, so the Phase B "cloned" dub can clone each character's
# actual voice instead of assigning one from a pool.
#
# For each episode it extracts the original audio stream to a 44.1kHz WAV, then
# calls profile_show.py ONCE over the whole set. profile_show.py does the heavy
# ML (Demucs source separation -> diarization -> global clustering across all
# episodes -> gender/age + reference-clip selection) and writes:
#     <Out>                     the profile JSON (speakers + per-episode turns)
#     <Out without .json>.qc.txt a QC report you can eyeball
#     <ClipDir>\spkNN_*.wav     the reference clips Phase B clones from
#
# This is heavy but runs ONCE. Extraction reads each source once (over the share
# if the folder is a UNC path); pass -Scratch on a fast local disk if you prefer.
#
# Requires: ffmpeg/ffprobe on PATH; a Python with the dub stack plus:
#     pip install demucs
#     pip install pyannote.audio           # optional; falls back to resemblyzer
# For pyannote you must accept the model terms on HuggingFace once and pass
# -HfToken (or set HF_TOKEN); without it the resemblyzer fallback is used.
#
# Usage:
#   pwsh ./subtitle-anime-profile.ps1 -Folder "\\10.0.23.105\media\tv\...\MyShow"
#   pwsh ./subtitle-anime-profile.ps1 -Folder "..." -Diarizer resemblyzer -NoDemucs
#   pwsh ./subtitle-anime-profile.ps1 -Folder "..." -MaxEpisodes 12   # sample
# -----------------------------------------------------------------------------

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Folder,

    # Output profile JSON + reference-clip dir (default beside the episodes).
    [string]$Out     = "",
    [string]$ClipDir = "",

    # Local fast disk for extracted audio + Demucs stems (~150MB/episode).
    [string]$Scratch = ([IO.Path]::GetTempPath()),

    # Which ORIGINAL audio stream to analyse (0 = first = usually Japanese).
    [int]$RefAudioIndex = 0,

    # Profile only the first N episodes (0 = all). Recurring cast appears early,
    # so a sample of ~10-15 often captures the main characters far faster.
    [int]$MaxEpisodes = 0,

    # Diarization backend + knobs (see profile_show.py).
    [ValidateSet("auto", "pyannote", "resemblyzer")]
    [string]$Diarizer = "auto",
    [string]$HfToken  = "",
    [switch]$NoDemucs,
    [double]$ClusterThreshold = 0.70,

    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
if (-not $PSBoundParameters.ContainsKey('Verbose')) { $VerbosePreference = 'Continue' }

$FFMPEG  = "ffmpeg"
$FFPROBE = "ffprobe"
if (-not $PSBoundParameters.ContainsKey('Python')) {
    $venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPy) { $Python = $venvPy }
}
$PROFILE_SCRIPT = Join-Path $PSScriptRoot "profile_show.py"
$FF_LOGLEVEL = "info"

if (-not $Out)     { $Out     = Join-Path $Folder "anime-dub-profile.json" }
if (-not $ClipDir) { $ClipDir = Join-Path $Folder "anime-dub-clips" }
if (-not $HfToken) { $HfToken = $env:HF_TOKEN }

$VideoExtensions = @(".mkv", ".mp4", ".m4v", ".avi", ".ts")

function Write-Cmd {
    param([string]$Exe, [string[]]$Arguments)
    $rendered = ($Arguments | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
    Write-Host "  > $Exe $rendered" -ForegroundColor DarkGray
}

function Get-MediaDuration {
    param([string]$Path)
    $d = & $FFPROBE -v error -show_entries format=duration -of csv=p=0 -- "$Path"
    if ($LASTEXITCODE -ne 0 -or -not $d) { return 0.0 }
    return [double]$d
}

# --- checks ------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $Folder)) { throw "Folder not found: $Folder" }
if (-not (Test-Path -LiteralPath $PROFILE_SCRIPT)) { throw "Missing $PROFILE_SCRIPT" }
if (-not (Test-Path -LiteralPath $Scratch)) { throw "Scratch not found: $Scratch" }

$episodes = @(
    Get-ChildItem -LiteralPath $Folder -File |
        Where-Object { $VideoExtensions -contains $_.Extension.ToLower() } |
        Where-Object { $_.Name -notlike "*.pre-dub.*" } |
        Where-Object { $_.Name -notlike "*.replacing.*" } |
        Where-Object { $_.Name -notlike "*.part" } |
        Sort-Object Name
)
if ($episodes.Count -eq 0) { throw "No video files found in $Folder" }
if ($MaxEpisodes -gt 0 -and $episodes.Count -gt $MaxEpisodes) {
    Write-Host "Sampling first $MaxEpisodes of $($episodes.Count) episode(s)."
    $episodes = $episodes | Select-Object -First $MaxEpisodes
}
Write-Host "Profiling $($episodes.Count) episode(s) from $Folder"
Write-Host "Profile out: $Out"
Write-Host "Clip dir:    $ClipDir"
Write-Host "Diarizer:    $Diarizer  (demucs=$(-not $NoDemucs), cluster>=$ClusterThreshold)"
if ($Diarizer -ne 'resemblyzer' -and -not $HfToken) {
    Write-Warning "No -HfToken/HF_TOKEN set; pyannote (gated) will fall back to resemblyzer."
}

$work = Join-Path $Scratch ("profile_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $work | Out-Null
New-Item -ItemType Directory -Force $ClipDir | Out-Null
$listFile = Join-Path $work "audio-list.tsv"

try {
    # --- extract one original-audio WAV per episode --------------------------
    $lines = New-Object System.Collections.Generic.List[string]
    $n = 0
    foreach ($ep in $episodes) {
        $n++
        $base = [IO.Path]::GetFileNameWithoutExtension($ep.Name)
        $wav  = Join-Path $work "$base.orig.wav"
        Write-Host "`n[$n/$($episodes.Count)] extracting audio: $($ep.Name)"
        # Tolerant decode (some anime MKVs abort a strict demux). 44.1k stereo so
        # Demucs separates well; profile_show.py downsamples internally.
        $ffArgs = @("-y", "-v", $FF_LOGLEVEL, "-stats",
                    "-fflags", "+genpts+discardcorrupt", "-err_detect", "ignore_err",
                    "-i", $ep.FullName, "-map", "0:a:$RefAudioIndex",
                    "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", $wav)
        Write-Cmd $FFMPEG $ffArgs
        & $FFMPEG @ffArgs
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $wav)) {
            Write-Warning "  extraction failed for $($ep.Name); skipping."
            continue
        }
        $srcDur = Get-MediaDuration $ep.FullName
        $wavDur = Get-MediaDuration $wav
        if ($srcDur -gt 0 -and $wavDur -lt ($srcDur * 0.95)) {
            Write-Warning ("  audio truncated ({0:n0}s of {1:n0}s) - if this show hit the EBML demux bug, remux it first (see subtitle-anime-unique-voices.ps1 Repair-Container)." -f $wavDur, $srcDur)
        }
        $lines.Add("$wav`t$base")
    }
    if ($lines.Count -eq 0) { throw "No audio extracted from any episode." }
    Set-Content -LiteralPath $listFile -Value $lines -Encoding UTF8

    # --- run the profiler ----------------------------------------------------
    $pyArgs = @(
        $PROFILE_SCRIPT,
        "--audio-list", $listFile,
        "--out", $Out,
        "--clip-dir", $ClipDir,
        "--scratch", (Join-Path $work "stems"),
        "--diarizer", $Diarizer,
        "--cluster-threshold", $ClusterThreshold,
        "--verbose"
    )
    if ($NoDemucs) { $pyArgs += "--no-demucs" }
    if ($HfToken)  { $pyArgs += @("--hf-token", $HfToken) }

    Write-Host "`nBuilding profile (Demucs + diarization + clustering) - this is the slow part..."
    Write-Cmd $Python $pyArgs
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $Python @pyArgs
    $sw.Stop()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Out)) {
        throw "profile_show.py failed (exit $LASTEXITCODE)"
    }
    Write-Host ("`nDone in {0:n1}s. Profile: {1}" -f $sw.Elapsed.TotalSeconds, $Out) -ForegroundColor Green
    Write-Host "Review the .qc.txt report and listen to a few clips in $ClipDir before Phase B."
}
finally {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}
