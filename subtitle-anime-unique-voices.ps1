# 2026-07-01 - Multi-voice variant of subtitle-anime.ps1.
#
# Same pipeline as the "basic" single-voice script (subtitle-anime.ps1): find
# subtitles, TTS them to an English track, mix over the original at reduced
# volume, remux so Plex auto-picks the English dub and the original audio stays.
#
# The difference from the basic script: each subtitle line is spoken by the
# voice of the CHARACTER who says it, and characters are tracked ACROSS EPISODES
# so a recurring character keeps the same voice through the whole show.
#
# The target folder is one show, so a persistent character profile is kept at
# -ProfilePath (defaults to <Folder>\anime-dub-voices.json). For each cue the
# ORIGINAL (Japanese) audio under it is fingerprinted (resemblyzer d-vector) and
# matched to the nearest known character; unknown voices become new characters,
# each assigned a pitch bucket (adult/child x male/female) and a DISTINCT voice
# from that bucket's pool. See srt_to_speech_multivoice.py for the full method
# and its (honest) limitations. Without resemblyzer installed it degrades to
# stateless per-line pitch buckets and says so.
#
# -----------------------------------------------------------------------------
# Requires (on PATH or set the *_EXE vars below):
#   ffmpeg, ffprobe   (https://www.gyan.dev/ffmpeg/builds/)
#   python            (basic deps + `pip install resemblyzer` for cross-episode
#                      character tracking; pitch detection reuses torchaudio)
# Usage:
#   pwsh ./subtitle-anime-unique-voices.ps1 -Folder "\\10.0.23.105\media\tv\...\MyShow"
#   pwsh ./subtitle-anime-unique-voices.ps1 -Folder "..." -All
#   pwsh ./subtitle-anime-unique-voices.ps1 -Folder "..." -Scratch "D:\dub-scratch"
#   # tune matching / thresholds for a show:
#   pwsh ./subtitle-anime-unique-voices.ps1 -Folder "..." `
#        -MatchThreshold 0.78 -ChildSplit 310 -ChildPitchShift 3
#
# Same network-share behaviour as the basic script: source is copied to
# -Scratch (local disk), all work happens locally, result is copied back to
# <Folder>\dubbed. Budget ~2x one episode of scratch space.
# -----------------------------------------------------------------------------

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Folder,

    # Process every episode without the single-episode test/confirm step.
    [switch]$All,

    # Volume the original audio is mixed down to under the English dub (0.6 = 60%).
    [double]$OriginalVolume = 0.6,

    # --- persistent character profile (show-level) ---------------------------
    # One JSON per show; recurring characters keep their voice across episodes.
    # Delete it to reset the cast; back it up before re-running to reuse it.
    [string]$ProfilePath = "",
    # Cosine similarity to treat a line as an existing character. Higher = more
    # distinct characters (and more accidental splits); lower = more merging.
    [double]$MatchThreshold = 0.75,

    # --- voice pools per bucket (XTTS built-in speaker names) -----------------
    # New characters get the first unused pool voice for their bucket, then the
    # pool cycles. Empty = use the script's built-in pools (see the .py header).
    # "Damien Black" leads the adult-male pool, matching the basic script.
    [string[]]$VoicesAdultMale   = @(),
    [string[]]$VoicesAdultFemale = @(),
    [string[]]$VoicesChildMale   = @(),
    [string[]]$VoicesChildFemale = @(),
    # Voice for lines with no clear speaker (music/SFX/overlap); default = first
    # adult-male pool voice.
    [string]$VoiceDefault     = "",

    # --- pitch buckets (Hz) - decide a NEW character's gender/age + voice pool -
    [double]$MaleMax        = 155,   # below this  -> adult male
    [double]$AdultFemaleMax = 250,   # below this  -> adult female
    [double]$ChildSplit     = 300,   # below this  -> child male, else child female
    # Semitones to raise child clips so adult XTTS voices read younger (0 = off).
    [double]$ChildPitchShift = 2.0,

    # Which ORIGINAL audio stream to analyse (0 = first = usually JP).
    [int]$RefAudioIndex = 0,

    # Speed up dub lines that overrun the next subtitle cue (keeps lip-ish sync).
    [switch]$FitToCues,

    # Local fast disk used as scratch (source copied here to avoid slow network
    # reads/writes; result copied back after).
    [string]$Scratch = ([IO.Path]::GetTempPath()),

    # Python interpreter to run the TTS helper. Point at a venv to avoid global
    # dependency conflicts, e.g.
    #   -Python "G:\Transcode\.venv-dub\Scripts\python.exe"
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
# -Verbose is on by default so you can see every step; run with -Verbose:$false to quiet it.
if (-not $PSBoundParameters.ContainsKey('Verbose')) { $VerbosePreference = 'Continue' }

# --- tool locations ----------------------------------------------------------
$FFMPEG  = "ffmpeg"
$FFPROBE = "ffprobe"
# If -Python wasn't given explicitly, prefer a local .venv over the global python.
if (-not $PSBoundParameters.ContainsKey('Python')) {
    $venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPy) { $Python = $venvPy }
}
$PYTHON  = $Python
$TTS_SCRIPT = Join-Path $PSScriptRoot "srt_to_speech_multivoice.py"

# ffmpeg log level: 'info' shows what it's doing without the per-frame firehose.
$FF_LOGLEVEL = "info"

# Echo an external command before running it, so the exact invocation is visible.
function Write-Cmd {
    param([string]$Exe, [string[]]$Arguments)
    $rendered = ($Arguments | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
    Write-Host "  > $Exe $rendered" -ForegroundColor DarkGray
}

$VideoExtensions = @(".mkv", ".mp4", ".m4v", ".avi", ".ts")
$TextSubCodecs   = @("subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text")

# Where finished files go (originals are never modified in place).
$OutputDir = Join-Path $Folder "dubbed"

# Show-level character profile shared by every episode in this folder.
if (-not $ProfilePath) { $ProfilePath = Join-Path $Folder "anime-dub-voices.json" }


# --- helpers -----------------------------------------------------------------

function Invoke-FFProbeJson {
    param([string]$Path)
    Write-Verbose "Probing streams: $Path"
    # JSON output must stay clean, so probe errors are still suppressed here.
    $json = & $FFPROBE -v error -print_format json -show_format -show_streams -- "$Path"
    if ($LASTEXITCODE -ne 0) { throw "ffprobe failed on $Path" }
    $data = $json | ConvertFrom-Json

    # Summarize what we found so the user can see the source layout.
    foreach ($s in $data.streams) {
        $lang = if ($s.tags.language) { $s.tags.language } else { "und" }
        $extra = switch ($s.codec_type) {
            "video" { "$($s.width)x$($s.height)" }
            "audio" { "$($s.channels)ch $($s.sample_rate)Hz" }
            default { "" }
        }
        Write-Verbose ("  stream #{0,-2} {1,-9} {2,-10} lang={3,-3} {4}" -f `
            $s.index, $s.codec_type, $s.codec_name, $lang, $extra)
    }
    return $data
}

function Get-VideoDuration {
    param($Probe)
    [double]$Probe.format.duration
}

# Decide where this episode's English subtitles will come from, WITHOUT reading
# the big video file. Prefers a sidecar .srt, then an embedded *text* subtitle.
function Get-SubtitlePlan {
    param([string]$Video, $Probe)

    $base = [IO.Path]::GetFileNameWithoutExtension($Video)
    $dir  = [IO.Path]::GetDirectoryName($Video)

    # 1) sidecar files next to the video
    foreach ($cand in @("$base.en.srt", "$base.eng.srt", "$base.srt")) {
        $p = Join-Path $dir $cand
        if (Test-Path -LiteralPath $p) {
            return [pscustomobject]@{ Type = "sidecar"; Path = $p; Desc = "sidecar $cand" }
        }
    }

    # 2) embedded subtitle streams (text only - image subs like PGS can't be read)
    $subs = @($Probe.streams | Where-Object { $_.codec_type -eq "subtitle" })
    if ($subs.Count -eq 0) { return $null }

    $textSubs = @($subs | Where-Object { $TextSubCodecs -contains $_.codec_name })
    if ($textSubs.Count -eq 0) {
        Write-Warning "  only image-based subtitles found (PGS/VOBSUB) - cannot convert to text, skipping."
        return $null
    }

    # prefer English; otherwise first text sub
    $pick = $textSubs | Where-Object { $_.tags.language -eq "eng" } | Select-Object -First 1
    if (-not $pick) { $pick = $textSubs[0] }

    # ffmpeg -map 0:s:N uses the subtitle-relative index, so find it among subs.
    $relIndex = [Array]::IndexOf(($subs | ForEach-Object { $_.index }), $pick.index)

    return [pscustomobject]@{
        Type     = "embedded"
        RelIndex = $relIndex
        Desc     = "embedded stream #$($pick.index) ($($pick.codec_name))"
    }
}

# Produce the actual .srt from a plan. For embedded subs this reads the video,
# so it is pointed at the LOCAL copy, not the network source.
function Export-Subtitle {
    param($Plan, [string]$ExtractFrom, [string]$WorkDir, [string]$Base)

    if ($Plan.Type -eq "sidecar") {
        Write-Host "  subtitle: $($Plan.Desc)"
        return $Plan.Path
    }

    $outSrt = Join-Path $WorkDir "$Base.extracted.srt"
    Write-Host "  subtitle: $($Plan.Desc) -> srt"
    $exArgs = @("-y", "-v", $FF_LOGLEVEL, "-i", $ExtractFrom, "-map", "0:s:$($Plan.RelIndex)", "-c:s", "srt", $outSrt)
    Write-Cmd $FFMPEG $exArgs
    & $FFMPEG @exArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $outSrt)) {
        Write-Warning "  subtitle extraction failed."
        return $null
    }
    $cueCount = (Select-String -LiteralPath $outSrt -Pattern '^\d+\s*$').Count
    Write-Verbose "  extracted $cueCount subtitle cue(s) to $outSrt"
    return $outSrt
}

# Extract one ORIGINAL audio stream to a mono 16kHz WAV, used ONLY to classify
# each line's speaker by pitch. Reads the LOCAL copy, not the network source.
function Export-ReferenceAudio {
    param([string]$ExtractFrom, [string]$WorkDir, [string]$Base)

    $outWav = Join-Path $WorkDir "$Base.ref.wav"
    Write-Host "  reference audio: original stream 0:a:$RefAudioIndex -> mono 16kHz wav"
    $rfArgs = @("-y", "-v", $FF_LOGLEVEL, "-i", $ExtractFrom,
                "-map", "0:a:$RefAudioIndex", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", $outWav)
    Write-Cmd $FFMPEG $rfArgs
    & $FFMPEG @rfArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $outWav)) {
        Write-Warning "  reference audio extraction failed."
        return $null
    }
    return $outWav
}

# Copy a file with progress/throughput logging. Returns the destination path.
function Copy-WithProgress {
    param([string]$Source, [string]$Destination, [string]$Label)
    $srcInfo = Get-Item -LiteralPath $Source
    $sizeMB = [math]::Round($srcInfo.Length / 1MB, 1)
    Write-Host "  $Label ($sizeMB MB)..."
    Write-Verbose "    $Source -> $Destination"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    # Use the .NET API, not Copy-Item: these filenames contain [brackets], which
    # Copy-Item's -Destination treats as wildcards (there is no -LiteralPath for
    # the destination). Over a UNC share that glob fails to resolve and throws
    # "Could not find a part of the path". File.Copy is fully literal + UNC-safe.
    [System.IO.File]::Copy($Source, $Destination, $true)
    $sw.Stop()
    $secs = [math]::Max($sw.Elapsed.TotalSeconds, 0.001)
    Write-Verbose ("    copied in {0:n1}s ({1:n1} MB/s)" -f $secs, ($sizeMB / $secs))
    return $Destination
}

# Calls the multi-voice Python TTS helper to render the timed English WAV.
function New-DubTrack {
    param([string]$Srt, [string]$RefAudio, [double]$Duration, [string]$OutWav)

    $pyArgs = @(
        $TTS_SCRIPT,
        "--srt", $Srt,
        "--out", $OutWav,
        "--ref-audio", $RefAudio,
        "--profile", $ProfilePath,
        "--duration", $Duration,
        "--language", "en",
        "--match-threshold",  $MatchThreshold,
        "--male-max",         $MaleMax,
        "--adult-female-max", $AdultFemaleMax,
        "--child-split",      $ChildSplit,
        "--child-pitch-shift", $ChildPitchShift
    )
    # Voice pools are ';'-separated; only send overrides, else the .py defaults win.
    if ($VoicesAdultMale)   { $pyArgs += @("--voices-adult-male",   ($VoicesAdultMale   -join ';')) }
    if ($VoicesAdultFemale) { $pyArgs += @("--voices-adult-female", ($VoicesAdultFemale -join ';')) }
    if ($VoicesChildMale)   { $pyArgs += @("--voices-child-male",   ($VoicesChildMale   -join ';')) }
    if ($VoicesChildFemale) { $pyArgs += @("--voices-child-female", ($VoicesChildFemale -join ';')) }
    if ($VoiceDefault) { $pyArgs += @("--voice-default", $VoiceDefault) }
    if ($FitToCues)    { $pyArgs += "--fit" }
    $pyArgs += "--verbose"

    Write-Host "  synthesizing multi-voice dub on GPU..."
    Write-Cmd $PYTHON $pyArgs
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $PYTHON @pyArgs
    $sw.Stop()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $OutWav)) {
        throw "TTS generation failed for $Srt"
    }
    Write-Verbose ("  TTS finished in {0:n1}s -> {1}" -f $sw.Elapsed.TotalSeconds, $OutWav)
}

# Muxes: video copy + all original audio (untouched) + new English track
# (dub at full volume mixed with original at $OriginalVolume) + original subs.
# Two passes on purpose (see the basic script for the full rationale): render
# the mixed English track to a clean AAC first, then pure copy-mux everything.
function Merge-DubIntoVideo {
    param([string]$Video, [string]$DubWav, $Probe, [string]$OutVideo, [string]$WorkDir)

    $audioStreams = @($Probe.streams | Where-Object { $_.codec_type -eq "audio" })
    $origAudioCount = $audioStreams.Count
    $newIdx = $origAudioCount   # output audio index of the mixed English track
    $mixedAac = Join-Path $WorkDir "mixed_english.m4a"

    # English track = original(0:a:0) at $OriginalVolume + dub(1:a) at full.
    $filter = "[0:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=$OriginalVolume[orig];" +
              "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=1.0[dub];" +
              "[orig][dub]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95[mix]"

    Write-Verbose "  [pass 1] rendering mixed English track: dub@100% + original@$([int]($OriginalVolume*100))% (stereo, 48kHz AAC)"
    Write-Verbose "  filter: $filter"
    $p1 = @(
        "-y", "-v", $FF_LOGLEVEL, "-stats",
        "-i", $Video,
        "-i", $DubWav,
        "-filter_complex", $filter,
        "-map", "[mix]",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
        $mixedAac
    )
    Write-Cmd $FFMPEG $p1
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $FFMPEG @p1
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $mixedAac)) {
        throw "ffmpeg mix (pass 1) failed for $Video"
    }

    # Pass 2: pure copy-mux. Video + all original audio + new English track
    # (copied from the clean AAC) + subs + attachments. No re-encode.
    Write-Verbose "  [pass 2] copy-muxing: video + $origAudioCount original audio + English track a:$newIdx (default)"
    $p2 = @(
        "-y", "-v", $FF_LOGLEVEL, "-stats",
        "-i", $Video,
        "-i", $mixedAac,
        "-map", "0:v",
        "-map", "0:a",
        "-map", "1:a",
        "-map", "0:s?",
        "-map", "0:t?",
        "-c", "copy",
        "-metadata:s:a:$newIdx", "language=eng",
        "-metadata:s:a:$newIdx", "title=English Dub (AI, multi-voice)",
        "-disposition:a:$newIdx", "default",
        # clean up interleaving/timestamps so no track hitches the video
        "-max_interleave_delta", "0",
        "-avoid_negative_ts", "make_zero"
    )
    # Clear the default flag on the original audio tracks so Plex auto-picks English.
    for ($i = 0; $i -lt $origAudioCount; $i++) {
        $p2 += @("-disposition:a:$i", "0")
    }
    $p2 += @($OutVideo)

    Write-Cmd $FFMPEG $p2
    & $FFMPEG @p2
    $sw.Stop()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $OutVideo)) {
        throw "ffmpeg mux (pass 2) failed for $Video"
    }
    Remove-Item -LiteralPath $mixedAac -Force -ErrorAction SilentlyContinue
    $sizeMB = [math]::Round((Get-Item -LiteralPath $OutVideo).Length / 1MB, 1)
    Write-Verbose ("  mux finished in {0:n1}s -> {1} ({2} MB)" -f $sw.Elapsed.TotalSeconds, $OutVideo, $sizeMB)
}

# Full pipeline for one episode. Returns $true on success, $false if skipped.
function Invoke-Episode {
    param([string]$Video)

    Write-Host "`n=== $([IO.Path]::GetFileName($Video)) ===" -ForegroundColor Cyan
    $epSw = [Diagnostics.Stopwatch]::StartNew()
    $base    = [IO.Path]::GetFileNameWithoutExtension($Video)
    $outFile = Join-Path $OutputDir "$base.dubbed.mkv"
    if (Test-Path -LiteralPath $outFile) {
        Write-Host "  already done, skipping (output in $OutputDir)."
        return $true
    }
    # Also skip if a finished dub was moved back next to the source, so re-runs
    # over the show folder don't redo episodes you've already filed away.
    $srcDub = Join-Path ([IO.Path]::GetDirectoryName($Video)) "$base.dubbed.mkv"
    if (Test-Path -LiteralPath $srcDub) {
        Write-Host "  already dubbed in source, skipping ($([IO.Path]::GetFileName($srcDub)))."
        return $true
    }

    Write-Host "  [1/7] probing source"
    $probe = Invoke-FFProbeJson -Path $Video
    $dur = Get-VideoDuration $probe
    Write-Verbose ("  duration: {0:n0}s ({1:hh\:mm\:ss})" -f $dur, [TimeSpan]::FromSeconds($dur))

    # Decide subtitle source over the network first - if there's nothing usable,
    # bail before copying gigabytes we'd only throw away.
    Write-Host "  [2/7] locating subtitles"
    $plan = Get-SubtitlePlan -Video $Video -Probe $probe
    if (-not $plan) {
        Write-Warning "  no usable subtitles, skipping (no copy made)."
        return $false
    }
    Write-Verbose "  subtitle plan: $($plan.Desc)"

    $work = Join-Path $Scratch ("dub_" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $work | Out-Null
    Write-Verbose "  local scratch: $work"
    try {
        # Copy the source to local disk so extraction + mux don't stream it over
        # the network repeatedly.
        Write-Host "  [3/7] copying source to local scratch"
        $localVideo = Join-Path $work ([IO.Path]::GetFileName($Video))
        Copy-WithProgress -Source $Video -Destination $localVideo -Label "copy source local" | Out-Null

        Write-Host "  [4/7] extracting subtitles"
        $srt = Export-Subtitle -Plan $plan -ExtractFrom $localVideo -WorkDir $work -Base $base
        if (-not $srt) { return $false }

        Write-Host "  [5/7] extracting reference audio (for voice classification)"
        $refWav = Export-ReferenceAudio -ExtractFrom $localVideo -WorkDir $work -Base $base
        if (-not $refWav) { return $false }

        Write-Host "  [6/7] generating multi-voice English dub (TTS)"
        $dubWav = Join-Path $work "$base.dub.wav"
        New-DubTrack -Srt $srt -RefAudio $refWav -Duration $dur -OutWav $dubWav

        Write-Host "  [7/7] muxing tracks (local)"
        $localOut = Join-Path $work "$base.dubbed.mkv"
        Merge-DubIntoVideo -Video $localVideo -DubWav $dubWav -Probe $probe -OutVideo $localOut -WorkDir $work

        # Only now, after a fully successful local build, push the result back.
        # Copy to a .part file and rename on success, so an interrupted transfer
        # never leaves a truncated file that later looks "already done".
        Write-Host "  copying result back to $OutputDir"
        $partFile = "$outFile.part"
        if (Test-Path -LiteralPath $partFile) { Remove-Item -LiteralPath $partFile -Force }
        Copy-WithProgress -Source $localOut -Destination $partFile -Label "copy result back" | Out-Null
        # .NET Move, not Move-Item: bracketed $outFile would be glob-parsed by
        # Move-Item's -Destination (same bug as the copy above).
        [System.IO.File]::Move($partFile, $outFile, $true)

        $epSw.Stop()
        Write-Host ("  done in {0:n1}s -> {1}" -f $epSw.Elapsed.TotalSeconds, $outFile) -ForegroundColor Green
        return $true
    }
    finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}


# --- main --------------------------------------------------------------------

if (-not (Test-Path -LiteralPath $Folder)) { throw "Folder not found: $Folder" }
if (-not (Test-Path -LiteralPath $TTS_SCRIPT)) { throw "Missing TTS helper: $TTS_SCRIPT" }
if (-not (Test-Path -LiteralPath $Scratch)) { throw "Scratch folder not found: $Scratch" }
[System.IO.Directory]::CreateDirectory($OutputDir) | Out-Null

# Report scratch location + free space so a full/wrong drive is obvious up front.
$scratchRoot = [IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $Scratch).Path)
$freeGB = try {
    [math]::Round((Get-PSDrive -Name $scratchRoot.TrimEnd(':\')).Free / 1GB, 1)
} catch { "?" }
Write-Host "Local scratch: $Scratch  (drive $scratchRoot, $freeGB GB free)"
Write-Host "Python: $PYTHON"
$profileState = if (Test-Path -LiteralPath $ProfilePath) { "existing" } else { "new" }
Write-Host "Character profile: $ProfilePath ($profileState, match>=$MatchThreshold)"
Write-Host ("Pitch buckets for new characters (Hz): male<$MaleMax, adultF<$AdultFemaleMax, " +
            "childM<$ChildSplit, else childF  | child +$ChildPitchShift semitones")

$episodes = @(
    Get-ChildItem -LiteralPath $Folder -File |
        Where-Object { $VideoExtensions -contains $_.Extension.ToLower() } |
        # Skip our own outputs if they were moved back into the source folder,
        # so we never try to dub an already-dubbed file.
        Where-Object { $_.Name -notlike "*.dubbed.mkv" } |
        Sort-Object Name
)
if ($episodes.Count -eq 0) { throw "No video files found in $Folder" }
Write-Host "Found $($episodes.Count) episode(s) in $Folder"

# 1) test on a single episode first
if (-not $All) {
    Write-Host "`n--- TEST RUN: first episode only ---"
    $ok = Invoke-Episode -Video $episodes[0].FullName
    if (-not $ok) {
        Write-Warning "Test episode produced no output. Fix the issue above before running the full folder."
        return
    }
    $answer = Read-Host "`nTest looks good? Process the remaining $($episodes.Count - 1) episode(s)? [y/N]"
    if ($answer -notmatch '^[Yy]') {
        Write-Host "Stopping after test. Re-run with -All to process everything."
        return
    }
    $remaining = $episodes | Select-Object -Skip 1
}
else {
    $remaining = $episodes
}

# 2) loop the rest of the folder
$done = 0; $skipped = 0
foreach ($ep in $remaining) {
    try {
        if (Invoke-Episode -Video $ep.FullName) { $done++ } else { $skipped++ }
    }
    catch {
        Write-Warning "  ERROR on $($ep.Name): $_"
        $skipped++
    }
}
Write-Host "`nFinished. Dubbed: $done, skipped: $skipped. Output in $OutputDir"
