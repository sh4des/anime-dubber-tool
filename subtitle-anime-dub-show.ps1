# subtitle-anime-dub-show.ps1
# -----------------------------------------------------------------------------
# One-command, end-to-end cloned dub for a whole show. This is the standard entry
# point for a new series:
#
#   PHASE A  profile the entire show (subtitle-anime-profile.ps1):
#            Demucs-isolate dialogue -> diarize -> cluster speakers across ALL
#            episodes -> gender/age + reference clips -> anime-dub-profile.json
#   PHASE B  dub every episode (subtitle-anime-unique-voices.ps1 -Clone):
#            clone each character's own (Japanese) voice from the profile, mix
#            over the original, remux so Plex auto-picks the English track.
#
# All local, no signups (ECAPA + Demucs + XTTS download weights anonymously; no
# audio ever leaves the machine). Safe to re-run: profiling overwrites the
# profile; dubbing is -Redub + resume-safe (skips episodes already rebuilt with
# the cloned engine this pass), so an interrupted run just continues.
#
# Usage:
#   pwsh ./subtitle-anime-dub-show.ps1 -Folder "\\10.0.23.105\media\tv\...\MyShow" `
#       -Scratch "G:\Transcode" -Python "G:\Transcode\.venv-dub\Scripts\python.exe"
#
#   # reuse an existing profile (skip Phase A), just (re)dub:
#   pwsh ./subtitle-anime-dub-show.ps1 -Folder "..." -SkipProfile ...
#
#   # keep a .pre-dub rollback copy of each original:
#   pwsh ./subtitle-anime-dub-show.ps1 -Folder "..." -BackupOriginal ...
# -----------------------------------------------------------------------------

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Folder,

    [string]$Scratch = ([IO.Path]::GetTempPath()),
    [string]$Python  = "python",
    [string]$Mkvmerge = "mkvmerge",

    # Diarizer for Phase A. 'ecapa' = SpeechBrain (no signup, best local quality).
    [ValidateSet("auto", "ecapa", "pyannote", "resemblyzer")]
    [string]$Diarizer = "ecapa",

    # Skip Phase A and reuse the existing <Folder>\anime-dub-profile.json.
    [switch]$SkipProfile,
    # Force fresh Demucs separation in Phase A (else cached stems are reused).
    [switch]$FreshStems,

    # Phase B voice engine. 'indextts' (default, RECOMMENDED) = IndexTTS2 clone +
    # emotion transfer from the original Japanese line; 'xtts' = older timbre-only
    # clone. See subtitle-anime-unique-voices.ps1 for details.
    [ValidateSet("indextts", "xtts")]
    [string]$Engine = "indextts",
    [string]$IndexTtsPython = "G:\Transcode\index-tts\.venv\Scripts\python.exe",
    [string]$CheckpointsDir = "G:\Transcode\index-tts\checkpoints",
    [double]$EmoAlpha = 0.45,
    # Demucs-isolated vocal stems dir (<base>.vocals.wav from Phase A). When set,
    # IndexTTS2 takes its emotion reference from the clean vocal stem instead of
    # the raw mix, removing the reverb/echo bleed. Passed through to Phase B.
    [string]$EmotionStemsDir = "",

    # Lay the English dub over a Demucs music+SFX bed (original dialogue removed)
    # instead of ducking the full original. Passed through to Phase B.
    [switch]$MusicBed,
    [double]$BedVolume = 0.9,

    # Passed through to Phase B.
    [switch]$BackupOriginal,     # keep <name>.pre-dub.<ext> before replacing
    [switch]$UseDubbedFolder,    # write to <Folder>\dubbed\ instead of in place
    [switch]$NoFit,              # disable cue duration-fit (on by default here)

    # Advanced: cap Phase A to the first N episodes (0 = all). The full show gives
    # the best cast + lets every episode use the fast time-overlap dub path.
    [int]$ProfileEpisodes = 0,

    # Render order for Phase B: start at the first episode matching this substring
    # and wrap earlier episodes to the end (e.g. "S01E20" => E20..end, then E01..E19).
    [string]$StartFrom = ""
)

$ErrorActionPreference = "Stop"
if (-not $PSBoundParameters.ContainsKey('Verbose')) { $VerbosePreference = 'Continue' }

if (-not $PSBoundParameters.ContainsKey('Python')) {
    $venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPy) { $Python = $venvPy }
}

$profileScript = Join-Path $PSScriptRoot "subtitle-anime-profile.ps1"
$dubScript     = Join-Path $PSScriptRoot "subtitle-anime-unique-voices.ps1"
$profileJson   = Join-Path $Folder "anime-dub-profile.json"
foreach ($s in @($profileScript, $dubScript)) {
    if (-not (Test-Path -LiteralPath $s)) { throw "Missing sibling script: $s" }
}
if (-not (Test-Path -LiteralPath $Folder)) { throw "Folder not found: $Folder" }

$overall = [Diagnostics.Stopwatch]::StartNew()

# --- PHASE A: profile the whole show -----------------------------------------
if ($SkipProfile) {
    if (-not (Test-Path -LiteralPath $profileJson)) {
        throw "-SkipProfile set but no profile at $profileJson. Run without -SkipProfile first."
    }
    Write-Host "`n=== PHASE A: skipped (reusing $profileJson) ===" -ForegroundColor Cyan
}
else {
    Write-Host "`n=== PHASE A: profiling the whole show ===" -ForegroundColor Cyan
    $pa = @{
        Folder   = $Folder
        Scratch  = $Scratch
        Python   = $Python
        Diarizer = $Diarizer
    }
    if ($ProfileEpisodes -gt 0) { $pa.MaxEpisodes = $ProfileEpisodes }
    if ($FreshStems)            { $pa.FreshStems  = $true } else { $pa.ReuseStems = $true }
    & $profileScript @pa
    if (-not (Test-Path -LiteralPath $profileJson)) {
        throw "Phase A did not produce $profileJson - stopping before dub."
    }
    Write-Host "Review $($profileJson -replace '\.json$', '.qc.txt') if you want; proceeding to dub." -ForegroundColor DarkGray
}

# --- PHASE B: dub every episode with cloned voices ---------------------------
Write-Host "`n=== PHASE B: dubbing every episode (cloned voices) ===" -ForegroundColor Cyan
$pb = @{
    Folder       = $Folder
    Clone        = $true
    Engine       = $Engine
    CloneProfile = $profileJson
    Redub        = $true       # rebuild existing AI tracks; resume-safe per engine
    All          = $true       # whole folder, no per-episode prompt
    Scratch      = $Scratch
    Python       = $Python
    Mkvmerge     = $Mkvmerge
}
if ($Engine -eq "indextts") {
    $pb.IndexTtsPython = $IndexTtsPython
    $pb.CheckpointsDir = $CheckpointsDir
    $pb.EmoAlpha       = $EmoAlpha
    if ($EmotionStemsDir) { $pb.EmotionStemsDir = $EmotionStemsDir }
}
if (-not $NoFit)        { $pb.FitToCues      = $true }
if ($BackupOriginal)    { $pb.BackupOriginal = $true }
if ($UseDubbedFolder)   { $pb.UseDubbedFolder = $true }
if ($StartFrom)         { $pb.StartFrom      = $StartFrom }
if ($MusicBed)          { $pb.MusicBed = $true; $pb.BedVolume = $BedVolume }
& $dubScript @pb

$overall.Stop()
Write-Host ("`n=== DONE: {0:n1} min total for {1} ===" -f ($overall.Elapsed.TotalMinutes), (Split-Path $Folder -Leaf)) -ForegroundColor Green
