<#
  dub_foreign_movies.ps1 - batch expressive clone-dub of the foreign-language
  movie shortlist (2026-07-17). Mirrors the per-movie Beast flow (beast_env +
  beast_phaseA + beast_phaseB) but loops over the shortlist. Non-destructive:
  every dub is written to <Movie>\dubbed\, the original remux is never touched.

  PER-MOVIE FLOW: Phase A (profile the cast) -> [audition gate] -> Phase B
  (IndexTTS2 clone+emotion render -> mux -> dubbed\) -> tag into the Plex
  "zDUBBED" collection.

  STAGES (-Stage):
    profile   Phase A only for the selected tier (default; stops before render)
    sample    audition every profiled voice (subtitle-anime-sample-voices.ps1)
    render    Phase B render + mux -> dubbed\, then Plex-tag (run after audition)
    all       profile -> render -> tag, no audition pause (unattended)

  TIER SELECTION (which movies are eligible; see $Movies notes):
    (default)         'ready' only - movies with a full embedded English TEXT sub
    -IncludeForced    also the 3 whose only/first English sub is FORCED (signage-
                      only risk); supply a full-dialogue sidecar .srt first
    -IncludeSidecar   also the 6 with NO usable English text sub (image-only or
                      none); a sidecar <base>.en.srt is REQUIRED next to the mkv
    -OnlyTitles "Roma (2018)","Stalker (1979)"   restrict to explicit titles

  NOTHING RUNS UNLESS YOU PASS -Go (dry-run by default: prints the plan).

  Examples:
    pwsh ./dub_foreign_movies.ps1                       # dry-run plan (ready tier)
    pwsh ./dub_foreign_movies.ps1 -Stage profile -Go    # profile the 10 ready movies
    pwsh ./dub_foreign_movies.ps1 -Stage sample  -Go    # audition their voices
    pwsh ./dub_foreign_movies.ps1 -Stage render  -Go    # render+mux+tag the ready movies
    pwsh ./dub_foreign_movies.ps1 -OnlyTitles "Roma (2018)" -Stage all -Go
#>
[CmdletBinding()]
param(
    [ValidateSet('profile','sample','render','all')]
    [string]$Stage = 'profile',
    [switch]$IncludeForced,
    [switch]$IncludeSidecar,
    [string[]]$OnlyTitles = @(),
    [switch]$SkipPlexTag,        # do the dub but don't touch the Plex collection
    [switch]$Go                  # actually execute; omit for a dry-run plan
)

$ErrorActionPreference = 'Stop'
$VerbosePreference     = 'Continue'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()

# --- shared paths / env (same as beast_env.ps1) ------------------------------
New-Item -ItemType Directory -Force 'G:\Transcode\tmp' | Out-Null
$env:TMP = 'G:\Transcode\tmp'; $env:TEMP = 'G:\Transcode\tmp'
$env:TTS_HOME = 'G:\Transcode\tts-cache'
$env:HF_HOME  = 'G:\Transcode\hf-cache'

# This script lives in scratchpad/, so the repo root is one level up.
$AppDir  = Split-Path $PSScriptRoot -Parent
$Scratch = if ($env:DUB_SCRATCH) { $env:DUB_SCRATCH } else { 'G:\Transcode' }
$V       = Join-Path $Scratch '.venv-dub\Scripts\python.exe'           # Phase A / Demucs / profiler
$Idx     = Join-Path $Scratch 'index-tts\.venv\Scripts\python.exe'     # IndexTTS2 render
$Ck      = Join-Path $Scratch 'index-tts\checkpoints'
$Tune    = Join-Path $PSScriptRoot 'movie_voice_tuning.json'          # globals-only knobs
# Optional Plex tagging step; lives outside this repo. Set both to enable it,
# else the zDUBBED tagging is skipped (see Invoke-PlexTag below).
$PlexTagPy = $env:PLEX_TAG_PY                                          # e.g. ...\plex\plex-dub-collection.py
$PlexPy    = if ($env:PLEX_PY) { $env:PLEX_PY } else { 'python' }

# --- the shortlist -----------------------------------------------------------
# tier: ready   = full embedded English TEXT sub, safe first-pick
#       forced  = only/first English sub is FORCED (signage-only risk) -> sidecar
#       sidecar = no usable English text sub (image-only/none) -> sidecar REQUIRED
#       inprogress = already running elsewhere (skipped)
$Movies = @(
    @{ Title='The Beast (2024)';               Folder='\\NAS\media\movies\15-18\The Beast (2024)';                                                                                                          Tier='inprogress'; Note='DONE 2026-07-18 (dubbed\...dubbed.mkv on NAS). Excluded from this batch; kept for reference.' }
    @{ Title='Godzilla Minus One (2023)';       Folder='\\NAS\media\movies4k\REMUX\Godzilla Minus One 2023 Japanese 2160p Blu-Ray REMUX DD+5 1 Atmos+TrueHD 7 1 Atmos DV-HDR HEVC-TheBiscuitMan';           Tier='ready';   Note='2 en/srt; 2160p REMUX (dubbed\ copy is large)' }
    @{ Title='Fantastic Planet (1973)';         Folder='\\NAS\media\movies\18+\Fantastic Planet (1973)';                                                                                                     Tier='ready';   Note='en/srt full (FR original)' }
    @{ Title='Final Account (2021)';            Folder='\\NAS\media\movies\Documentaries\Final Account (2021)';                                                                                               Tier='ready';   Note='en/srt full (mostly EN doc w/ DE interviews)' }
    @{ Title='Come and See (1985)';             Folder='\\NAS\media\movies\18+\Come and See (1985)';                                                                                                          Tier='ready';   Note='en/srt full (RU original; en/pgs ignored)' }
    @{ Title='Stalker (1979)';                  Folder='\\NAS\media\movies\REMUX 1080P\Stalker (1979)';                                                                                                       Tier='ready';   Note='en/srt full (RU original; pgs ignored)' }
    @{ Title='Kalashnikov AK-47 (2020)';        Folder='\\NAS\media\movies\18+\Kalashnikov AK 47 (2020)';                                                                                                    Tier='ready';   Note='en/srt full (RU original)' }
    @{ Title='Roma (2018)';                     Folder='\\NAS\media\movies\Sam\Roma (2018)';                                                                                                                  Tier='ready';   Note='en/srt full (ES original)' }
    @{ Title='Apocalypto (2006)';               Folder='\\NAS\media\movies\15-18\Apocalypto (2006)';                                                                                                          Tier='ready';   Note='en/srt full (Mayan original)' }
    @{ Title='The Secret Agent (2025)';         Folder='\\NAS\media\movies\15-18\The Secret Agent (2025)';                                                                                                    Tier='ready';   Note='en-CA/srt (PT original); confirm lang tag = eng' }
    @{ Title='Ip Man (2008)';                   Folder='\\NAS\media\movies\15-18\Ip Man (2008)';                                                                                                              Tier='ready';   Note='en/srt full (ZH original)' }
    @{ Title='Colony (2026)';                   Folder='G:\Transcode\colony-work\Colony (2026)';                                                                                                                      Tier='ready';   Note='en/srt full 1282 cues (KO original; audio MIStagged eng). 2h02m -> run off the LOCAL mkvmerge remux per the >2h Phase-A wedge recipe; copy the dub back to \\NAS\media\movies\15-18\Colony (2026) when done.' }

    @{ Title='Aniara (2019)';                   Folder='\\NAS\media\movies\15-18\Aniara (2019)';                                                                                                              Tier='forced';  Note='FIRST en sub is FORCED; full dialogue is the SDH/plain en/srt - drop a sidecar or the dub will be signage-only' }
    @{ Title='Godzilla vs. Destoroyah (1995)';  Folder='\\NAS\media\movies\15-18\Godzilla vs. Destoroyah (1995)';                                                                                             Tier='forced';  Note='only en/srt is FORCED; verify it is full dialogue else sidecar' }
    @{ Title='The Wandering Earth (2019)';      Folder='\\NAS\media\movies\15-18\The Wandering Earth (2019)';                                                                                                  Tier='forced';  Note='only en/srt is FORCED; verify full dialogue else sidecar' }

    @{ Title='Shin Godzilla (2016)';            Folder='\\NAS\media\movies\8-14\Shin Godzilla (2016)';                                                                                                        Tier='sidecar'; Note='only ja/pgs (image) - needs <base>.en.srt' }
    @{ Title='Hell (2011)';                     Folder='\\NAS\media\movies\15-18\Hell (2011)';                                                                                                                 Tier='sidecar'; Note='no embedded subs - needs <base>.en.srt' }
    @{ Title='The Zone of Interest (2023)';     Folder='\\NAS\media\movies\18+\The Zone of Interest (2023)';                                                                                                   Tier='sidecar'; Note='en sub is VOBSUB (image) - needs <base>.en.srt (or OCR the vobsub)' }
    @{ Title='Shaolin Soccer (2001)';           Folder='\\NAS\media\movies\8-14\Shaolin Soccer (2001)';                                                                                                       Tier='sidecar'; Note='no embedded subs - needs <base>.en.srt' }
    @{ Title='T-34 (2018)';                     Folder='\\NAS\media\movies\18+\Т-34 (2018)';                                                                                                                  Tier='sidecar'; Note='no embedded subs (Cyrillic folder) - needs <base>.en.srt' }
    @{ Title='Cinema Paradiso (1988)';          Folder='\\NAS\media\movies\15-18\Cinema Paradiso (1988)';                                                                                                     Tier='sidecar'; Note='no embedded subs - needs <base>.en.srt' }

    # --- Korean tier, added 2026-08-14 -----------------------------------------
    # These were INVISIBLE to plex-audio-language-scan.py: their audio is tagged
    # 'eng' (or untagged) but is actually Korean, so the non-English-audio filter
    # never returned them. Verified by faster-whisper langid, not by tag.
    # All three are live-action; none have ANY usable English text sub, so each
    # needs a whisper_subs.py translate pass -> <base>.en.srt BEFORE dubbing.
    # Excluded on purpose: Train to Busan (2016) already carries a real English
    # dub track (langid en p=0.94), and The Man from Nowhere (2010) IS an English
    # dub already - neither needs the pipeline.
    @{ Title='Parasite (2019)';                 Folder='\\NAS\media\movies\15-18\Parasite (2019)';                                                                                                            Tier='sidecar'; Note='KO original (langid ko p=1.00), audio MIStagged eng; NO subtitle track at all - needs <base>.en.srt' }
    @{ Title='Peninsula (2020)';                Folder='\\NAS\media\movies\15-18\Peninsula (2020)';                                                                                                           Tier='sidecar'; Note='KO original (langid ko p=0.79), audio MIStagged eng; NO subtitle track at all - needs <base>.en.srt' }
    @{ Title='Illang: The Wolf Brigade (2018)'; Folder='\\NAS\media\movies\15-18\Illang The Wolf Brigade (2018)';                                                                                             Tier='sidecar'; Note='KO original (langid ko p=0.97), audio UNtagged; .mp4 container; NO subtitle track at all - needs <base>.en.srt' }

    # --- mistagged-audio audit finds, 2026-08-14 (user-approved queue) ---------
    # Same blind spot as the Korean tier: audio tagged eng/und but NOT English.
    # Found by a full-library sweep (2763 movies -> 521 foreign-origin screened
    # -> 5-sample/all-track verification). Every one below was confirmed by
    # reading the transcribed dialogue, and confirmed to have NO English track
    # anywhere, so nothing here replaces a real English audio track.
    # All need a whisper_subs.py pass -> <base>.en.srt before Phase A.
    @{ Title='Timecrimes (2007)';               Folder='\\NAS\media\movies\15-18\Timecrimes (2007)';                                                                                                          Tier='sidecar'; Note='ES original (Los Cronocrimenes), 5/5 es; screen called it en p=0.64 - a FALSE NEGATIVE caught by re-test. No subs.' }
    @{ Title='The Angel (2018)';                Folder='\\NAS\media\movies\15-18\The Angel (2018)';                                                                                                           Tier='sidecar'; Note='ES original (El Angel, Argentine), 5/5 es. No subs.' }
    @{ Title='Gagarin: First in Space (2013)';  Folder='\\NAS\media\movies\15-18\Gagarin - First in Space (2013)';                                                                                            Tier='sidecar'; Note='RU original, 5/5 ru. No subs. 720p WEBDL.' }
    @{ Title='Mothra vs. Godzilla (1964)';      Folder='\\NAS\media\movies\8-14\Mothra vs. Godzilla (1964)';                                                                                                  Tier='sidecar'; Note='JA original (live-action tokusatsu), 4/5 ja. No subs.' }
    @{ Title='Godzilla Raids Again (1955)';     Folder='\\NAS\media\movies\15-18\Godzilla Raids Again (1955)';                                                                                                Tier='sidecar'; Note='JA original (live-action), ja. No subs. SDTV source - lowest quality of the batch.' }
    @{ Title='Godzilla Against MechaGodzilla (2002)'; Folder='\\NAS\media\movies\15-18\Godzilla Against MechaGodzilla (2002)';                                                                                Tier='sidecar'; Note='JA original (live-action), ja. No subs. .mp4 container.' }
    @{ Title='Legend of the Eight Samurai (1983)';    Folder='\\NAS\media\movies\18+\Legend of the Eight Samurai (1983)';                                                                                     Tier='sidecar'; Note='JA original (live-action), ja; the en/nn screen hits were SILENCE (empty transcripts). No subs. .mp4 container.' }
)

# --- selection ---------------------------------------------------------------
$allowed = @('ready')
if ($IncludeForced)  { $allowed += 'forced' }
if ($IncludeSidecar) { $allowed += 'sidecar' }

$selected = $Movies | Where-Object {
    ($_.Tier -ne 'inprogress') -and
    (($OnlyTitles.Count -gt 0 -and $OnlyTitles -contains $_.Title) -or
     ($OnlyTitles.Count -eq 0 -and $allowed -contains $_.Tier))
}

Write-Host "`n=== dub_foreign_movies :: stage=$Stage :: $(($selected).Count) movie(s) ===" -ForegroundColor Cyan
foreach ($m in $selected) { Write-Host ("  [{0,-7}] {1}" -f $m.Tier, $m.Title) -ForegroundColor DarkGray; Write-Host "            $($m.Note)" -ForegroundColor DarkGray }
if (-not $Go) { Write-Host "`n(dry-run - re-run with -Go to execute)`n" -ForegroundColor Yellow; return }

# --- helpers -----------------------------------------------------------------
function Get-StemDir([string]$Folder) {
    $key = (Split-Path $Folder -Leaf) -replace '[^\w.-]','_'
    Join-Path $Scratch "dubprofile-stems_$key"
}
# Per-movie tuning if make_tuning.py produced one, else the shared globals file.
# MUST be per-movie for anything under "speakers": that map is keyed by speaker id
# and spk01 in Colony is a different person from spk01 in Parasite, so a shared
# per-speaker map would cross-apply one film's corrections to every other film.
function Get-Tuning([string]$Folder) {
    $key = (Split-Path $Folder -Leaf) -replace '[^\w.-]','_'
    $per = Join-Path $PSScriptRoot "tuning\${key}_voice_tuning.json"
    if (Test-Path -LiteralPath $per) { return $per }
    return $Tune
}
# Profile JSON + reference clips written to LOCAL disk (co-located so the
# profile's relative clip paths still resolve in Phase B). Fixes the intermittent
# SMB clip-write failures ('LibsndfileError: System error') that skipped
# Kalashnikov + Come and See after hours of Phase A. Only the final dubbed mkv
# still copies to the NAS (that mux step is reliable).
function Get-LocalProfile([string]$Folder) {
    $key = (Split-Path $Folder -Leaf) -replace '[^\w.-]','_'
    $dir = Join-Path $Scratch "dubprofile-local_$key"
    [pscustomobject]@{
        Json  = (Join-Path $dir 'anime-dub-profile.json')
        Clips = (Join-Path $dir 'anime-dub-clips')
    }
}
function Invoke-ProfileStage($m) {
    Write-Host "`n--- PHASE A profile: $($m.Title) ---" -ForegroundColor Cyan
    $lp = Get-LocalProfile $m.Folder
    $pa = @{ Folder=$m.Folder; Out=$lp.Json; ClipDir=$lp.Clips; Scratch=$Scratch; Python=$V; Diarizer='ecapa'; ReuseStems=$true }
    & (Join-Path $AppDir 'subtitle-anime-profile.ps1') @pa
    if (-not (Test-Path -LiteralPath $lp.Json)) { throw "no profile produced for $($m.Title)" }
    Write-Host "    profile (local) -> $($lp.Json)" -ForegroundColor Green
}
function Invoke-SampleStage($m) {
    Write-Host "`n--- AUDITION voices: $($m.Title) ---" -ForegroundColor Cyan
    $sa = @{ Folder=$m.Folder; Scratch=$Scratch; IndexTtsPython=$Idx; CheckpointsDir=$Ck; VoiceTuning=(Get-Tuning $m.Folder) }
    & (Join-Path $AppDir 'subtitle-anime-sample-voices.ps1') @sa
    Write-Host "    listen: $($m.Folder)\voice-samples\_contact_sheet.wav" -ForegroundColor Green
}
function Invoke-RenderStage($m) {
    $pj = (Get-LocalProfile $m.Folder).Json
    if (-not (Test-Path -LiteralPath $pj)) { throw "$($m.Title): no local profile - run -Stage profile first" }
    Write-Host "`n--- PHASE B render+mux -> dubbed\ : $($m.Title) ---" -ForegroundColor Cyan
    $pb = @{
        Folder          = $m.Folder
        Clone           = $true
        Engine          = 'indextts'
        CloneProfile    = $pj
        Redub           = $true
        All             = $true
        Scratch         = $Scratch
        Python          = $V
        IndexTtsPython  = $Idx
        CheckpointsDir  = $Ck
        EmoAlpha        = 0.45
        EmotionStemsDir = (Get-StemDir $m.Folder)
        VoiceTuning     = (Get-Tuning $m.Folder)
        FitToCues       = $true
        MusicBed        = $true
        UseDubbedFolder = $true
    }
    & (Join-Path $AppDir 'subtitle-anime-unique-voices.ps1') @pb
    Write-Host "    dub -> $($m.Folder)\dubbed\" -ForegroundColor Green
    if (-not $SkipPlexTag) {
        if ($PlexTagPy -and (Test-Path $PlexTagPy)) {
            Write-Host "    tagging into Plex zDUBBED collection..." -ForegroundColor Cyan
            & $PlexPy $PlexTagPy $m.Title
        } else {
            Write-Host "    Plex tagging skipped (set PLEX_TAG_PY to enable)" -ForegroundColor DarkGray
        }
    }
}

# --- drive -------------------------------------------------------------------
foreach ($m in $selected) {
    try {
        switch ($Stage) {
            'profile' { Invoke-ProfileStage $m }
            'sample'  { Invoke-SampleStage  $m }
            'render'  { Invoke-RenderStage  $m }
            'all'     { Invoke-ProfileStage $m; Invoke-RenderStage $m }
        }
    } catch {
        Write-Warning "SKIP $($m.Title): $($_.Exception.Message)"
    }
}
Write-Host "`n=== stage '$Stage' complete ===" -ForegroundColor Green
