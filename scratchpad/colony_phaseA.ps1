# colony_phaseA.ps1 - Phase A (roster/profile) for Colony (2026).
# Wrapper so the space-in-title -OnlyTitles arg survives Start-Process.
# Runs against the LOCAL mkvmerge remux (G:\Transcode\colony-work) per the
# ">2h films wedge in Phase A" recipe; the NAS source is never touched.
& (Join-Path $PSScriptRoot 'dub_foreign_movies.ps1') -OnlyTitles @('Colony (2026)') -Stage profile -Go
