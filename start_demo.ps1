# Root entry point: current source backend on Windows + LiveKit in Docker.
param([switch]$Background, [switch]$Reindex)
& (Join-Path $PSScriptRoot 'scripts/start_demo.ps1') -Background:$Background -Reindex:$Reindex
