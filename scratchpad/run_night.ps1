# run_night.ps1 - overnight batch (23:00-08:00 AU only), 2026-07-20.
# Runs the 5 tractable remaining titles FIRST (GMO's ~5h Phase A would otherwise
# hog every window), then Godzilla Minus One last. Resume-safe: already-dubbed
# titles (English AI track present) are skipped, so this can be killed at 08:00
# and re-launched at 23:00 to continue.
$app = Join-Path $PSScriptRoot 'dub_foreign_movies.ps1'
# Night 4 (2026-07-23): Roma + Apocalypto were TAGGED in zDUBBED but never actually
# dubbed (confirmed 2026-07-23: no dub file anywhere on disk; sources carry only
# their foreign audio). Re-render them. Profiles cached in dubprofile-local_* so
# Phase A reuses embeddings (fast). After they finish, move -> source as -DUBBED.mkv.
& $app -OnlyTitles @('Roma (2018)','Apocalypto (2006)') -Stage all -Go
