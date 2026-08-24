# run_short.ps1 - run the SHORT (<2h) foreign movies first (2026-07-19).
# The long (>2h) titles wedge in Phase A (Demucs/diarization) for 60-110+ min
# with no output (GMO, Come and See), so they're deferred to a separate batch.
# Short films profiled+rendered fine (Fantastic Planet, Final Account).
& (Join-Path $PSScriptRoot 'dub_foreign_movies.ps1') `
    -OnlyTitles @(
        'Kalashnikov AK-47 (2020)',
        'The Secret Agent (2025)',
        'Ip Man (2008)'
    ) `
    -Stage all -Go
