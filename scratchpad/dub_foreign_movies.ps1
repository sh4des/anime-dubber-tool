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
    @{ Title='Fantastic Planet (1973)';         Folder='\\10.0.23.70\media\movies\18+ Animated\Fantastic Planet (1973)';                                                                                     Tier='inprogress'; Note='DONE 2026-10-23 02:59 (...-DUBBED.mkv beside the original, run_2026-10-22b). FR confirmed 10-pt listen 2026-10-23.' }
    @{ Title='Final Account (2021)';            Folder='\\NAS\media\movies\Documentaries\Final Account (2021)';                                                                                               Tier='ready';   Note='en/srt full (mostly EN doc w/ DE interviews)' }
    @{ Title='Come and See (1985)';             Folder='\\10.0.23.70\media\movies\18+\Come and See (1985)';                                                                                                   Tier='inprogress'; Note='DONE 2026-10-23 04:38 (.dubbed.mkv, run_2026-10-22b). RU confirmed 10-pt listen 2026-10-23.' }
    @{ Title='Stalker (1979)';                  Folder='\\10.0.23.70\media\movies\REMUX 1080P\Stalker (1979)';                                                                                                Tier='ready';   Note='en/srt full s7 (RU original; pgs ignored). 2026-10-23 10-pt listen: original = a2 rus; a3 is TAGGED eng but is RUSSIAN - do not treat a3 as an English track. 2h41 (>2h wedge recipe).' }
    @{ Title='Kalashnikov AK-47 (2020)';        Folder='\\NAS\media\movies\18+\Kalashnikov AK 47 (2020)';                                                                                                    Tier='ready';   Note='en/srt full (RU original)' }
    @{ Title='Roma (2018)';                     Folder='\\NAS\media\movies\Sam\Roma (2018)';                                                                                                                  Tier='ready';   Note='en/srt full (ES original)' }
    @{ Title='Apocalypto (2006)';               Folder='\\NAS\media\movies\15-18\Apocalypto (2006)';                                                                                                          Tier='ready';   Note='en/srt full (Mayan original)' }
    @{ Title='The Secret Agent (2025)';         Folder='\\NAS\media\movies\15-18\The Secret Agent (2025)';                                                                                                    Tier='ready';   Note='en-CA/srt (PT original); confirm lang tag = eng' }
    @{ Title='Ip Man (2008)';                   Folder='\\10.0.23.70\media\movies\15-18\Ip Man (2008)';                                                                                                       Tier='sidecar'; Note='2026-10-23: file replaced since July - NO English sub now (1 non-eng sub) -> needs <base>.en.srt. ZH confirmed 10-pt listen; original = a2 chi; a4 is TAGGED eng but is CHINESE.' }
    @{ Title='Colony (2026)';                   Folder='G:\Transcode\colony-work\Colony (2026)';                                                                                                                      Tier='ready';   Note='en/srt full 1282 cues (KO original; audio MIStagged eng). 2h02m -> run off the LOCAL mkvmerge remux per the >2h Phase-A wedge recipe; copy the dub back to \\NAS\media\movies\15-18\Colony (2026) when done.' }

    @{ Title='Aniara (2019)';                   Folder='\\10.0.23.70\media\movies\15-18\Aniara (2019)';                                                                                                       Tier='sidecar'; Note='2026-10-23: only en subs now are PGS (image: s3 "English for Swedish audio", s4 signs) -> needs <base>.en.srt (OCR s3). SV confirmed 10-pt listen; original = a1 swe; a2 is an English COMMENTARY tagged swe.' }
    @{ Title='Godzilla vs. Destoroyah (1995)';  Folder='\\NAS\media\movies\15-18\Godzilla vs. Destoroyah (1995)';                                                                                             Tier='forced';  Note='only en/srt is FORCED; verify it is full dialogue else sidecar' }
    @{ Title='The Wandering Earth (2019)';      Folder='\\10.0.23.70\media\movies\15-18\The Wandering Earth (2019)';                                                                                          Tier='ready';   Note='2026-10-23: file replaced - now s2 en/srt NOT forced (s1 forced pgs ignored); not checked that s2 covers all dialogue. ZH confirmed 10-pt listen; original = a32 chi truehd. 2h05 (>2h wedge recipe).' }

    @{ Title='Shin Godzilla (2016)';            Folder='\\NAS\media\movies\8-14\Shin Godzilla (2016)';                                                                                                        Tier='sidecar'; Note='only ja/pgs (image) - needs <base>.en.srt' }
    @{ Title='Hell (2011)';                     Folder='\\10.0.23.70\media\movies\15-18\Hell (2011)';                                                                                                         Tier='sidecar'; Note='no embedded subs - needs <base>.en.srt. 2026-10-23 10-pt listen: BOTH tracks untagged; a1 is an ITALIAN dub, a2 is the GERMAN original - dub from a2, never a1.' }
    @{ Title='The Zone of Interest (2023)';     Folder='\\10.0.23.70\media\movies\18+\The Zone of Interest (2023)';                                                                                           Tier='ready';   Note='2026-10-23: file replaced - now s3 en/srt (text). DE confirmed 10-pt listen (de 4/5; some Polish/Yiddish scenes); original = a5 ger.' }
    @{ Title='Shaolin Soccer (2001)';           Folder='\\NAS\media\movies\8-14\Shaolin Soccer (2001)';                                                                                                       Tier='sidecar'; Note='no embedded subs - needs <base>.en.srt' }
    @{ Title='T-34 (2018)';                     Folder='\\10.0.23.70\media\movies\18+\Т-34 (2018)';                                                                                                           Tier='ready';   Note='2026-10-23: sidecar <base>.en.srt now present (no embedded subs; Cyrillic folder). RU confirmed 10-pt listen; original = a1 rus. 2h19 (>2h wedge recipe).' }
    @{ Title='Cinema Paradiso (1988)';          Folder='\\10.0.23.70\media\movies\15-18\Cinema Paradiso (1988)';                                                                                              Tier='sidecar'; Note='no embedded subs - needs <base>.en.srt. IT confirmed 10-pt listen 2026-10-23; original = a1 ita. 2h04 (>2h wedge recipe).' }

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
    @{ Title='Peninsula (2020)';                Folder='\\10.0.23.70\media\movies\15-18\Peninsula (2020)';                                                                                                    Tier='sidecar'; Note='KO original, audio MIStagged eng (a1). 2026-10-23: only en sub is FORCED ass (s2) -> needs <base>.en.srt. KO confirmed 10-pt listen.' }
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
    @{ Title='Mothra vs. Godzilla (1964)';      Folder='\\10.0.23.70\media\movies\15-18\Mothra vs. Godzilla (1964)';                                                                                          Tier='inprogress'; Note='DONE 2026-10-23 02:11 (...-DUBBED.mkv beside the original, run_2026-10-22b). JA confirmed 10-pt listen.' }
    @{ Title='Godzilla Raids Again (1955)';     Folder='\\10.0.23.70\media\movies\15-18\Godzilla Raids Again (1955)';                                                                                         Tier='inprogress'; Note='DONE 2026-10-22 20:26 (...-DUBBED.mkv beside the original, run_2026-10-22b). JA confirmed 10-pt listen.' }
    @{ Title='Godzilla Against MechaGodzilla (2002)'; Folder='\\NAS\media\movies\15-18\Godzilla Against MechaGodzilla (2002)';                                                                                Tier='sidecar'; Note='JA original (live-action), ja. No subs. .mp4 container.' }
    @{ Title='Legend of the Eight Samurai (1983)';    Folder='\\NAS\media\movies\18+\Legend of the Eight Samurai (1983)';                                                                                     Tier='sidecar'; Note='JA original (live-action), ja; the en/nn screen hits were SILENCE (empty transcripts). No subs. .mp4 container.' }

    # --- Dale's list, added 2026-10-23 -----------------------------------------
    # From the dw-projects plex/foreign-index listen (Whisper on EVERY audio track),
    # then a 10-point deep listen of each one before queueing ("be sure"). Every
    # film below is confirmed non-English with NO English dialogue track.
    # 'orig = aN' is the container stream index of the original-language track;
    # several are mis-tagged, so never pick the source track by language tag.
    # Paths are \\10.0.23.70 (samba-ingest): \\NAS no longer resolves on this PC.
    @{ Title='Alienoid (2022)';                 Folder='\\10.0.23.70\media\movies\15-18\Alienoid (2022)';                                                                                                     Tier='sidecar'; Note='KO, orig = a1 kor (8/8 samples). Only en sub is FORCED ass -> needs <base>.en.srt. 2h22 (>2h wedge recipe).' }
    @{ Title='Amelie (2001)';                   Folder='\\10.0.23.70\media\movies\15-18\Amélie (2001)';                                                                                                       Tier='sidecar'; Note='FR, orig = a7 fre truehd (a8 fre DD; a9 commentary). Only en sub is FORCED pgs -> needs <base>.en.srt. 2h01 (>2h wedge recipe).' }
    @{ Title='Downfall (2004)';                 Folder='\\10.0.23.70\media\movies4k\REMUX\Downfall (2004)';                                                                                                   Tier='sidecar'; Note='DE, orig = a5 ger (a6 = German commentary). en sub is PGS (image) -> needs <base>.en.srt. REMUX, 2h35 (>2h wedge recipe).' }
    @{ Title='Dreams (1990)';                   Folder='\\10.0.23.70\media\movies\18+\Dreams (1990)';                                                                                                         Tier='ready';   Note='JA (Kurosawa), orig = a1 jpn; a2 = English commentary. en/srt s3.' }
    @{ Title='Hero (2002)';                     Folder='\\10.0.23.70\media\movies\15-18\Hero (2002)';                                                                                                         Tier='sidecar'; Note='ZH, orig = a1 chi. en sub is PGS (image) -> needs <base>.en.srt.' }
    @{ Title='Letters from Iwo Jima (2006)';    Folder='\\10.0.23.70\media\movies\18+\Letters from Iwo Jima (2006)';                                                                                          Tier='sidecar'; Note='JA, orig = a1 - TAGGED fin, IS Japanese. A few genuine English lines (the captured US soldier) - keep them out of the sidecar so they stay in the original voice. No subs. 2h20 (>2h wedge recipe).' }
    @{ Title='Memories (1995)';                 Folder='\\10.0.23.70\media\movies4k\REMUX\Memories (1995)';                                                                                                   Tier='sidecar'; Note='JA anime anthology, orig = a4 jpn truehd (a5 jpn). en sub is PGS -> needs <base>.en.srt. REMUX.' }
    @{ Title='Mirror (1975)';                   Folder='\\10.0.23.70\media\movies\15-18\Mirror (1975)';                                                                                                       Tier='sidecar'; Note='RU (Tarkovsky), orig = a1; a1 AND a2 are TAGGED eng but are RUSSIAN. No subs.' }
    @{ Title='Pans Labyrinth (2006)';           Folder="\\10.0.23.70\media\movies\15-18\Pan's Labyrinth (2006)";                                                                                              Tier='ready';   Note='ES, orig = a1 spa. en/srt s3.' }
    @{ Title='Ran (1985)';                      Folder='\\10.0.23.70\media\movies4k\4K NOT REMUX OR WEB\Ran (1985)';                                                                                          Tier='sidecar'; Note='JA (Kurosawa), orig = a13 jpn. Only en sub is FORCED pgs -> needs <base>.en.srt. 4K, 2h40 (>2h wedge recipe).' }
    @{ Title='The City of Lost Children (1995)'; Folder='\\10.0.23.70\media\movies\18+\The City of Lost Children (1995)';                                                                                     Tier='sidecar'; Note='FR, orig = a1 fre. Only en sub is FORCED srt -> needs <base>.en.srt.' }
    @{ Title='The Hidden Fortress (1958)';      Folder='\\10.0.23.70\media\movies4k\REMUX\The Hidden Fortress (1958)';                                                                                        Tier='sidecar'; Note='JA (Kurosawa), orig = a1 jpn; a2 = English commentary. en sub is PGS -> needs <base>.en.srt. REMUX, 2h18 (>2h wedge recipe).' }
    @{ Title='TPB AFK (2013)';                  Folder='\\10.0.23.70\media\movies\Documentaries\The Pirate Bay - Away From Keyboard (2013)';                                                                  Tier='ready';   Note='SV documentary, orig = a1 - TAGGED eng, IS Swedish (10/10 sv). en/srt s2.' }

    # HOLD - on Dale's list but not a straight dub; needs his call (asked 2026-10-23).
    @{ Title='The Longest Day (1962)';          Folder='\\10.0.23.70\media\movies\18+\The Longest Day (1962)';                                                                                                Tier='hold';    Note='ENGLISH film with German/French scenes (10-pt: en 6, de 3). Dub ONLY the DE/FR lines: needs a sidecar holding just those cues - the music bed keeps the original voice outside cues, so the English cast stays untouched. A full en/srt would re-voice John Wayne et al. 2h58.' }
    @{ Title='For a Few Dollars More (1965)';   Folder='\\10.0.23.70\media\movies4k\REMUX\For a Few Dollars More (1965)';                                                                                     Tier='hold';    Note='ALL 5 tracks are ENGLISH (10-pt listen; a4/a5 are untitled commentaries). There is no Italian in this file, so an English dub would replace English with English.' }
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
