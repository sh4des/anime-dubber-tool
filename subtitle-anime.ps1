# 2024-07-01 9:20pm - TOUCH here to confirm this is a good working copy of the script with a single voice for all dubs
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
# all work happens locally, then the finished file replaces the original episode
# in place (default). Use -UseDubbedFolder to keep originals and write to
# <Folder>\dubbed\ instead. Make sure -Scratch has room for ~2x one episode.
# -----------------------------------------------------------------------------

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Folder,

    # Process every episode without the single-episode test/confirm step.
    [switch]$All,

    # Keep originals untouched; write <Folder>\dubbed\<name>.dubbed.mkv instead
    # of replacing each source file (the old behaviour).
    [switch]$UseDubbedFolder,

    # Before replacing an episode in-place, copy the original to
    # <name>.pre-dub.<ext> once (handy if you want a rollback copy on the share).
    [switch]$BackupOriginal,

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

# Written on every AI dub audio track; used to detect completed episodes on re-run.
$DubTrackTitlePrefix = "English Dub (AI)"

# Where finished files go when -UseDubbedFolder is set.
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

# True when the file already has an AI dub track from a previous successful run.
function Test-HasAiDubTrack {
    param($Probe)
    foreach ($s in @($Probe.streams | Where-Object { $_.codec_type -eq "audio" })) {
        $title = $s.tags.title
        if ($title -and $title.StartsWith($DubTrackTitlePrefix)) { return $true }
    }
    return $false
}

# Remove stale transfer files left by an interrupted copy/replace on the share.
function Clear-StaleTransferArtifacts {
    param([string]$Video)
    $ext = [IO.Path]::GetExtension($Video)
    $staging = "$Video.replacing$ext"
    $part = "$Video.part"

    if (-not (Test-Path -LiteralPath $Video) -and (Test-Path -LiteralPath $staging)) {
        Write-Warning "  recovering original from incomplete replace: $staging -> $Video"
        [System.IO.File]::Move($staging, $Video, $true)
    }
    elseif ((Test-Path -LiteralPath $Video) -and (Test-Path -LiteralPath $staging)) {
        Write-Verbose "  removing leftover staging file: $staging"
        Remove-Item -LiteralPath $staging -Force
    }
    if (Test-Path -LiteralPath $part) {
        Write-Verbose "  removing stale partial upload: $part"
        Remove-Item -LiteralPath $part -Force
    }
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
    $exArgs = @("-y", "-nostdin", "-v", $FF_LOGLEVEL, "-i", $ExtractFrom, "-map", "0:s:$($Plan.RelIndex)", "-c:s", "srt", $outSrt)
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
    # .NET API, not Copy-Item: bracketed filenames are literal over UNC shares.
    [System.IO.File]::Copy($Source, $Destination, $true)
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
#
# Two passes on purpose:
#   1) render the mixed English track to its OWN clean AAC file, and
#   2) pure copy-mux everything together.
# Piping a filtergraph straight into the final container gave the generated
# track irregular timestamps/interleaving, which made players stutter the video
# whenever that track was selected (the original tracks, being copied, were
# fine). Rendering it standalone first gives it clean, continuous timestamps;
# the final mux is then a plain -c copy with no filtering.
function Merge-DubIntoVideo {
    param([string]$Video, [string]$DubWav, $Probe, [string]$OutVideo, [string]$WorkDir)

    $audioStreams = @($Probe.streams | Where-Object { $_.codec_type -eq "audio" })
    $origAudioCount = $audioStreams.Count
    $newIdx = $origAudioCount   # output audio index of the mixed English track
    $mixedAac = Join-Path $WorkDir "mixed_english.m4a"

    # English track = original(0:a:0) at $OriginalVolume + dub(1:a) at full.
    # Both resampled to 48kHz stereo (broad client support), trimmed to the
    # original audio length (duration=first), soft-limited to avoid clipping.
    $filter = "[0:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=$OriginalVolume[orig];" +
              "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=1.0[dub];" +
              "[orig][dub]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95[mix]"

    Write-Verbose "  [pass 1] rendering mixed English track: dub@100% + original@$([int]($OriginalVolume*100))% (stereo, 48kHz AAC)"
    Write-Verbose "  filter: $filter"
    $p1 = @(
        "-y", "-nostdin", "-v", $FF_LOGLEVEL, "-stats",
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
        "-y", "-nostdin", "-v", $FF_LOGLEVEL, "-stats",
        "-i", $Video,
        "-i", $mixedAac,
        "-map", "0:v",
        "-map", "0:a",
        "-map", "1:a",
        "-map", "0:s?",
        "-map", "0:t?",
        "-c", "copy",
        "-metadata:s:a:$newIdx", "language=eng",
        "-metadata:s:a:$newIdx", "title=English Dub (AI)",
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

# Push a fully-built local file to its final location. For in-place mode the
# original is moved aside only after the .part copy is size- and ffprobe-verified,
# so a network drop during the upload never truncates the episode on the share.
function Install-DubbedEpisode {
    param([string]$LocalOut, [string]$Video, [string]$Base)

    if ($UseDubbedFolder) {
        [System.IO.Directory]::CreateDirectory($OutputDir) | Out-Null
        $outFile = Join-Path $OutputDir "$Base.dubbed.mkv"
    }
    else {
        $outFile = $Video
    }

    $partFile = "$outFile.part"
    if (Test-Path -LiteralPath $partFile) { Remove-Item -LiteralPath $partFile -Force }

    $destLabel = if ($UseDubbedFolder) { "copy result to $OutputDir" } else { "copy result in-place" }
    Copy-WithProgress -Source $LocalOut -Destination $partFile -Label $destLabel | Out-Null

    $expectedLen = (Get-Item -LiteralPath $LocalOut).Length
    $gotLen = (Get-Item -LiteralPath $partFile).Length
    if ($gotLen -ne $expectedLen) {
        Remove-Item -LiteralPath $partFile -Force -ErrorAction SilentlyContinue
        throw "transfer size mismatch ($gotLen vs $expectedLen bytes) - original untouched"
    }

    $partProbe = Invoke-FFProbeJson -Path $partFile
    if (-not (Test-HasAiDubTrack $partProbe)) {
        Remove-Item -LiteralPath $partFile -Force -ErrorAction SilentlyContinue
        throw "uploaded file is missing the AI dub track marker - original untouched"
    }

    if ($UseDubbedFolder) {
        [System.IO.File]::Move($partFile, $outFile, $true)
        return $outFile
    }

    $ext = [IO.Path]::GetExtension($Video)
    $bakFile = "$Video.pre-dub$ext"
    $staging = "$Video.replacing$ext"

    if ($BackupOriginal -and -not (Test-Path -LiteralPath $bakFile)) {
        Copy-WithProgress -Source $Video -Destination $bakFile -Label "backup original" | Out-Null
    }

    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Force }
    [System.IO.File]::Move($Video, $staging, $true)
    try {
        [System.IO.File]::Move($partFile, $Video, $false)
    }
    catch {
        if (-not (Test-Path -LiteralPath $Video) -and (Test-Path -LiteralPath $staging)) {
            [System.IO.File]::Move($staging, $Video, $true)
        }
        throw
    }
    Remove-Item -LiteralPath $staging -Force -ErrorAction SilentlyContinue
    return $Video
}

# Full pipeline for one episode. Returns $true on success, $false if skipped.
function Invoke-Episode {
    param([string]$Video)

    Write-Host "`n=== $([IO.Path]::GetFileName($Video)) ===" -ForegroundColor Cyan
    $epSw = [Diagnostics.Stopwatch]::StartNew()
    $base = [IO.Path]::GetFileNameWithoutExtension($Video)
    Clear-StaleTransferArtifacts -Video $Video

    Write-Host "  [1/6] probing source"
    $probe = Invoke-FFProbeJson -Path $Video
    if (Test-HasAiDubTrack $probe) {
        Write-Host "  already has AI dub track, skipping."
        return $true
    }
    if ($UseDubbedFolder) {
        $legacyOut = Join-Path $OutputDir "$base.dubbed.mkv"
        if (Test-Path -LiteralPath $legacyOut) {
            Write-Host "  already done, skipping (output in $OutputDir)."
            return $true
        }
    }
    $legacyBeside = Join-Path ([IO.Path]::GetDirectoryName($Video)) "$base.dubbed.mkv"
    if (Test-Path -LiteralPath $legacyBeside) {
        Write-Host "  legacy dubbed copy beside source, skipping ($([IO.Path]::GetFileName($legacyBeside)))."
        return $true
    }

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
        Merge-DubIntoVideo -Video $localVideo -DubWav $dubWav -Probe $probe -OutVideo $localOut -WorkDir $work

        Write-Host "  installing dubbed episode"
        $outFile = Install-DubbedEpisode -LocalOut $localOut -Video $Video -Base $base

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

# Report scratch location + free space so a full/wrong drive is obvious up front.
$scratchRoot = [IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $Scratch).Path)
$freeGB = try {
    [math]::Round((Get-PSDrive -Name $scratchRoot.TrimEnd(':\')).Free / 1GB, 1)
} catch { "?" }
Write-Host "Local scratch: $Scratch  (drive $scratchRoot, $freeGB GB free)"
Write-Host "Python: $PYTHON"
$outputMode = if ($UseDubbedFolder) { "separate folder: $OutputDir" } else { "in-place (replace source)" }
Write-Host "Output mode: $outputMode$(if ($BackupOriginal -and -not $UseDubbedFolder) { ' + .pre-dub backup' })"

$episodes = @(
    Get-ChildItem -LiteralPath $Folder -File |
        Where-Object { $VideoExtensions -contains $_.Extension.ToLower() } |
        Where-Object { $_.Name -notlike "*.dubbed.mkv" } |
        Where-Object { $_.Name -notlike "*.pre-dub.*" } |
        Where-Object { $_.Name -notlike "*.replacing.*" } |
        Where-Object { $_.Name -notlike "*.part" } |
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
$outWhere = if ($UseDubbedFolder) { $OutputDir } else { $Folder }
Write-Host "`nFinished. Dubbed: $done, skipped: $skipped. Output: $outWhere"
