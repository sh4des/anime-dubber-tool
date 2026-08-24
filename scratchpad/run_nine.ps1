# run_nine.ps1 - relaunch the 9 non-4K foreign movies (Godzilla Minus One 2160p
# remux deferred after it stalled in Phase A with a spinning CUDA kernel, 2026-07-19).
# Titles passed as a proper array here to avoid Start-Process arg-splitting.
& (Join-Path $PSScriptRoot 'dub_foreign_movies.ps1') `
    -OnlyTitles @(
        'Fantastic Planet (1973)',
        'Final Account (2021)',
        'Come and See (1985)',
        'Stalker (1979)',
        'Kalashnikov AK-47 (2020)',
        'Roma (2018)',
        'Apocalypto (2006)',
        'The Secret Agent (2025)',
        'Ip Man (2008)'
    ) `
    -Stage all -Go
