# run_deferred.ps1 - the remaining long/dense titles after the short batch (2026-07-20).
# These grind in Phase A (dialogue-dense diarization) but complete; user chose "let it grind".
# Includes the Kalashnikov retry (its earlier SKIP was a transient SMB clip-write glitch;
# embeddings are cached). Runner processes in its own $Movies array order:
# GMO -> Come and See -> Stalker -> Kalashnikov -> Roma -> Apocalypto.
& (Join-Path $PSScriptRoot 'dub_foreign_movies.ps1') `
    -OnlyTitles @(
        'Godzilla Minus One (2023)',
        'Come and See (1985)',
        'Stalker (1979)',
        'Kalashnikov AK-47 (2020)',
        'Roma (2018)',
        'Apocalypto (2006)'
    ) `
    -Stage all -Go
