# subtitle-anime-sample-voices.ps1
# -----------------------------------------------------------------------------
# Phase A.5: audition every profiled voice BEFORE the multi-hour Phase B render.
# Renders one comparable English line per character through the SAME timbre +
# emotion + tuning path the full engine uses, so silenced / echoed voices are
# caught while they're still cheap to fix.
#
# Run it after Phase A (subtitle-anime-profile.ps1 / -dub-show.ps1 without
# -SkipProfile) and before Phase B:
#
#   pwsh ./subtitle-anime-sample-voices.ps1 -Folder "\\...\Mobile Suit Gundam ZZ" `
#        -VoiceTuning .\scratchpad\voice_tuning.json
#
#   # instant static check only (no GPU / no render): which voices would be SILENT
#   pwsh ./subtitle-anime-sample-voices.ps1 -Folder "\\...\MyShow" -ListOnly
#
# Output goes to <Folder>\voice-samples\ (override with -OutDir): one WAV per
# speaker, a _contact_sheet.wav of the whole cast, and voice_samples_report.txt
# (problems first). Listen, tune voice_tuning.json / fix ref clips, then run
# Phase B (subtitle-anime-unique-voices.ps1 / -dub-show.ps1 -SkipProfile).
# -----------------------------------------------------------------------------
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Folder,

    [string]$Scratch = "G:\Transcode",
    [string]$IndexTtsPython = "G:\Transcode\index-tts\.venv\Scripts\python.exe",
    [string]$CheckpointsDir = "G:\Transcode\index-tts\checkpoints",
    [string]$VoiceTuning = "",
    [string]$OutDir = "",
    [string]$Text = "",
    [double]$EmoAlpha = 0.45,
    [string]$Speakers = "",       # e.g. "1,2,5" to limit; default all
    [switch]$ListOnly,            # static NO_TIMBRE check only, no render
    [switch]$Fp16
)

$ErrorActionPreference = "Stop"
if (-not $PSBoundParameters.ContainsKey('Verbose')) { $VerbosePreference = 'Continue' }

$py = Join-Path $PSScriptRoot "sample_voices_indextts.py"
if (-not (Test-Path -LiteralPath $py)) { throw "Missing $py" }
if (-not (Test-Path -LiteralPath $Folder)) { throw "Folder not found: $Folder" }
$profileJson = Join-Path $Folder "anime-dub-profile.json"
if (-not (Test-Path -LiteralPath $profileJson)) {
    throw "No Phase A profile at $profileJson. Run subtitle-anime-profile.ps1 first."
}
if (-not $ListOnly -and -not (Test-Path -LiteralPath $IndexTtsPython)) {
    throw "IndexTTS2 venv python not found: $IndexTtsPython"
}
if (-not $OutDir) { $OutDir = Join-Path $Folder "voice-samples" }

# caches off C:, same as Phase B
New-Item -ItemType Directory -Force (Join-Path $Scratch "tmp") | Out-Null
$env:TMP = (Join-Path $Scratch "tmp"); $env:TEMP = $env:TMP
$env:TTS_HOME = (Join-Path $Scratch "tts-cache")
$env:HF_HOME  = (Join-Path $Scratch "hf-cache")

$a = @("--profile", $profileJson, "--checkpoints-dir", $CheckpointsDir,
       "--out-dir", $OutDir, "--emo-alpha", $EmoAlpha)
if ($VoiceTuning -and (Test-Path -LiteralPath $VoiceTuning)) { $a += @("--voice-tuning", $VoiceTuning) }
if ($Text)     { $a += @("--text", $Text) }
if ($Speakers) { $a += @("--speakers", $Speakers) }
if ($ListOnly) { $a += "--list" }
if ($Fp16)     { $a += "--fp16" }

Write-Host "`n=== Voice audition: $(Split-Path $Folder -Leaf) -> $OutDir ===" -ForegroundColor Cyan
& $IndexTtsPython $py @a
$code = $LASTEXITCODE
if ($code -ne 0) { throw "sampler exited $code" }

if (-not $ListOnly) {
    Write-Host "`nListen to $OutDir\_contact_sheet.wav and per-speaker WAVs." -ForegroundColor Green
    Write-Host "Read $OutDir\voice_samples_report.txt (problems first), tune voice_tuning.json / fix ref clips," -ForegroundColor Green
    Write-Host "then run Phase B with -SkipProfile." -ForegroundColor Green
}
