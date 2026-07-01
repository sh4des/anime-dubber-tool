# give me some code that does the following
#
# looks at a folder of anime episodes
# looks for embedded or .srt subtitle files
# if it finds any, send them to a function that will convert
# convert them from text to speech using a TTS engine
# remux the audio back into the video file as a new audio track
#
# do a test on a single episode first, then if it works, loop through the entire folder.
# run this from a windows PC with a rtx 5080 gpu to generate the audio + merge it with the existing
#   audio track at 60% volume from the source audio to still hear sounds and effects in the episode,
#   but be louder than existing japanese speech audio.
# the goal is to have a new audio track that is the english dub of the episode, but still have the
#   original japanese audio track in the file as well.
# the final goal is to allow seamless playback on other plex media clients
#
# -----------------------------------------------------------------------------
# Requires (on PATH or set the *_EXE vars below):
#   ffmpeg, ffprobe   (https://www.gyan.dev/ffmpeg/builds/)
#   python            (with srt_to_speech.py deps installed - see that file's header)
# Usage:
#   pwsh ./subtitle-anime.ps1 -Folder "\\10.0.23.105\media\tv\...\MyShow"
#   pwsh ./subtitle-anime.ps1 -Folder "..." -All                    # skip the test prompt
#   pwsh ./subtitle-anime.ps1 -Folder "..." -Scratch "D:\dub-scratch"  # fast local disk
#
# Network shares: the source is copied to -Scratch (local disk, defaults to TEMP),
# all work happens locally, then the finished file is copied back to <Folder>\dubbed.
# Make sure -Scratch has room for ~2x one episode (source copy + output).
# -----------------------------------------------------------------------------

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Folder,

    # Process every episode without the single-episode test/confirm step.
    [switch]$All,

    # Volume the original audio is mixed down to under the English dub (0.6 = 60%).
    [double]$OriginalVolume = 0.6,

    # XTTS built-in speaker name, or use -SpeakerWav for voice cloning.
    [string]$Speaker = "Damien Black",
    [string]$SpeakerWav = "",

    # Speed up dub lines that overrun the next subtitle cue (keeps lip-ish sync).
    [switch]$FitToCues,

    # Local fast disk used as scratch. Source is copied here to avoid slow,
    # repeated reads/writes over the network share; result is copied back after.
    [string]$Scratch = ([IO.Path]::GetTempPath()),

    # Python interpreter to run the TTS helper. Point this at a venv to avoid
    # global site-packages dependency conflicts, e.g.
    #   -Python "C:\source\dw-projects\subtitle-anime-dub\.venv\Scripts\python.exe"
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
# -Verbose is on by default so you can see every step; run with -Verbose:$false to quiet it.
if (-not $PSBoundParameters.ContainsKey('Verbose')) { $VerbosePreference = 'Continue' }

# --- tool locations ----------------------------------------------------------
$FFMPEG  = "ffmpeg"
$FFPROBE = "ffprobe"
# If -Python wasn't given explicitly, prefer a local .venv over the global python
# (the global env has conflicting TTS/transformers deps).
if (-not $PSBoundParameters.ContainsKey('Python')) {
    $venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPy) { $Python = $venvPy }
}
$PYTHON  = $Python
$TTS_SCRIPT = Join-Path $PSScriptRoot "srt_to_speech.py"

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
# the big video file (cheap - runs over the network before we copy anything).
# Returns a plan object, or $null if there is nothing usable (so we can skip the
# copy entirely). Prefers a sidecar .srt, then an embedded *text* subtitle.
function Get-SubtitlePlan {
    param([string]$Video, $Probe)

    $base = [IO.Path]::GetFileNameWithoutExtension($Video)
    $dir  = [IO.Path]::GetDirectoryName($Video)

    # 1) sidecar files next to the video
    foreach ($cand in @("$base.en.srt", "$base.eng.srt", "$base.srt")) {
        $p = Join-Path $dir $cand
        # -LiteralPath: filenames often contain [brackets], which Test-Path
        # otherwise treats as wildcards and fails to match.
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

# Copy a file with progress/throughput logging. Returns the destination path.
function Copy-WithProgress {
    param([string]$Source, [string]$Destination, [string]$Label)
    $srcInfo = Get-Item -LiteralPath $Source
    $sizeMB = [math]::Round($srcInfo.Length / 1MB, 1)
    Write-Host "  $Label ($sizeMB MB)..."
    Write-Verbose "    $Source -> $Destination"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
    $sw.Stop()
    $secs = [math]::Max($sw.Elapsed.TotalSeconds, 0.001)
    Write-Verbose ("    copied in {0:n1}s ({1:n1} MB/s)" -f $secs, ($sizeMB / $secs))
    return $Destination
}

# Calls the Python TTS helper to render the timed English WAV.
function New-DubTrack {
    param([string]$Srt, [double]$Duration, [string]$OutWav)

    $pyArgs = @(
        $TTS_SCRIPT,
        "--srt", $Srt,
        "--out", $OutWav,
        "--duration", $Duration,
        "--language", "en"
    )
    if ($SpeakerWav) { $pyArgs += @("--speaker-wav", $SpeakerWav) }
    else             { $pyArgs += @("--speaker", $Speaker) }
    if ($FitToCues)  { $pyArgs += "--fit" }
    $pyArgs += "--verbose"

    Write-Host "  synthesizing dub on GPU..."
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
function Merge-DubIntoVideo {
    param([string]$Video, [string]$DubWav, $Probe, [string]$OutVideo)

    $audioStreams = @($Probe.streams | Where-Object { $_.codec_type -eq "audio" })
    $origAudioCount = $audioStreams.Count
    $newIdx = $origAudioCount   # output audio index of the mixed English track

    # English track = original(0:a:0) lowered to 60% + dub(1:a) at full,
    # both forced to stereo for broad Plex client support, soft-limited to avoid clipping.
    $filter = "[0:a:0]aformat=channel_layouts=stereo,volume=$OriginalVolume[orig];" +
              "[1:a]aformat=channel_layouts=stereo,volume=1.0[dub];" +
              "[orig][dub]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,alimiter=limit=0.95[mix]"

    Write-Verbose "  mixing new English track: dub@100% + original@$([int]($OriginalVolume*100))% (stereo)"
    Write-Verbose "  filter: $filter"
    Write-Verbose "  original audio tracks kept: $origAudioCount; new English track = a:$newIdx (default)"

    $ff = @(
        "-y", "-v", $FF_LOGLEVEL, "-stats",
        "-i", $Video,
        "-i", $DubWav,
        "-filter_complex", $filter,
        "-map", "0:v",
        "-map", "0:a",
        "-map", "[mix]",
        "-map", "0:s?",
        "-map", "0:t?",
        "-c", "copy",
        "-c:a:$newIdx", "aac", "-b:a:$newIdx", "256k",
        "-metadata:s:a:$newIdx", "language=eng",
        "-metadata:s:a:$newIdx", "title=English Dub (AI)",
        "-disposition:a:$newIdx", "default"
    )
    # Clear the default flag on the original audio tracks so Plex auto-picks English.
    for ($i = 0; $i -lt $origAudioCount; $i++) {
        $ff += @("-disposition:a:$i", "0")
    }
    $ff += @($OutVideo)

    Write-Cmd $FFMPEG $ff
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $FFMPEG @ff
    $sw.Stop()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $OutVideo)) {
        throw "ffmpeg mux failed for $Video"
    }
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
        Write-Host "  already done, skipping."
        return $true
    }

    Write-Host "  [1/6] probing source"
    $probe = Invoke-FFProbeJson -Path $Video
    $dur = Get-VideoDuration $probe
    Write-Verbose ("  duration: {0:n0}s ({1:hh\:mm\:ss})" -f $dur, [TimeSpan]::FromSeconds($dur))

    # Decide subtitle source over the network first - if there's nothing usable,
    # bail before copying gigabytes we'd only throw away.
    Write-Host "  [2/6] locating subtitles"
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
        Write-Host "  [3/6] copying source to local scratch"
        $localVideo = Join-Path $work ([IO.Path]::GetFileName($Video))
        Copy-WithProgress -Source $Video -Destination $localVideo -Label "copy source local" | Out-Null

        Write-Host "  [4/6] extracting subtitles"
        $srt = Export-Subtitle -Plan $plan -ExtractFrom $localVideo -WorkDir $work -Base $base
        if (-not $srt) { return $false }

        Write-Host "  [5/6] generating English dub (TTS)"
        $dubWav = Join-Path $work "$base.dub.wav"
        New-DubTrack -Srt $srt -Duration $dur -OutWav $dubWav

        Write-Host "  [6/6] muxing tracks (local)"
        $localOut = Join-Path $work "$base.dubbed.mkv"
        Merge-DubIntoVideo -Video $localVideo -DubWav $dubWav -Probe $probe -OutVideo $localOut

        # Only now, after a fully successful local build, push the result back.
        # Copy to a .part file and rename on success, so an interrupted transfer
        # never leaves a truncated file that later looks "already done".
        Write-Host "  copying result back to $OutputDir"
        $partFile = "$outFile.part"
        if (Test-Path -LiteralPath $partFile) { Remove-Item -LiteralPath $partFile -Force }
        Copy-WithProgress -Source $localOut -Destination $partFile -Label "copy result back" | Out-Null
        Move-Item -LiteralPath $partFile -Destination $outFile -Force

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
if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# Report scratch location + free space so a full/wrong drive is obvious up front.
$scratchRoot = [IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $Scratch).Path)
$freeGB = try {
    [math]::Round((Get-PSDrive -Name $scratchRoot.TrimEnd(':\')).Free / 1GB, 1)
} catch { "?" }
Write-Host "Local scratch: $Scratch  (drive $scratchRoot, $freeGB GB free)"
Write-Host "Python: $PYTHON"

$episodes = @(
    Get-ChildItem -LiteralPath $Folder -File |
        Where-Object { $VideoExtensions -contains $_.Extension.ToLower() } |
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
