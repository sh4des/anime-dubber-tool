# move_dubs.ps1 (2026-07-23) - move each <movie>\dubbed\<name>.dubbed.mkv up into
# the movie's source folder, renamed <name>-DUBBED.mkv, so Plex matches it as an
# additional version alongside the original. Same-share move = fast server rename.
$ErrorActionPreference = 'Stop'
$parents = @(
  '\\NAS\media\movies\15-18\The Beast (2024)',
  '\\NAS\media\movies4k\REMUX\Godzilla Minus One 2023 Japanese 2160p Blu-Ray REMUX DD+5 1 Atmos+TrueHD 7 1 Atmos DV-HDR HEVC-TheBiscuitMan',
  '\\NAS\media\movies\18+\Fantastic Planet (1973)',
  '\\NAS\media\movies\Documentaries\Final Account (2021)',
  '\\NAS\media\movies\18+\Come and See (1985)',
  '\\NAS\media\movies\REMUX 1080P\Stalker (1979)',
  '\\NAS\media\movies\18+\Kalashnikov AK 47 (2020)',
  '\\NAS\media\movies\Sam\Roma (2018)',
  '\\NAS\media\movies\15-18\Apocalypto (2006)',
  '\\NAS\media\movies\15-18\The Secret Agent (2025)',
  '\\NAS\media\movies\15-18\Ip Man (2008)'
)
$moved = 0
foreach ($p in $parents) {
  $d = Join-Path $p 'dubbed'
  if (-not (Test-Path -LiteralPath $d)) { Write-Output "SKIP (no dubbed\): $(Split-Path $p -Leaf)"; continue }
  foreach ($f in Get-ChildItem -LiteralPath $d -File -Filter *.mkv) {
    $newName = ($f.Name -replace '\.dubbed\.mkv$','-DUBBED.mkv')
    if ($newName -eq $f.Name) { $newName = "$($f.BaseName)-DUBBED$($f.Extension)" }
    $dest = Join-Path $p $newName
    if (Test-Path -LiteralPath $dest) { Write-Output "EXISTS, skip: $newName"; continue }
    Move-Item -LiteralPath $f.FullName -Destination $dest
    $moved++
    Write-Output ("MOVED -> {0}\{1}" -f (Split-Path $p -Leaf), $newName)
  }
  if (-not (Get-ChildItem -LiteralPath $d -Force -EA SilentlyContinue)) {
    Remove-Item -LiteralPath $d -Force; Write-Output "  (removed empty dubbed\)"
  }
}
Write-Output "=== moved $moved file(s) ==="
