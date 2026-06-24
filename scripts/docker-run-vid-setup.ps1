# Build (optional) and run vid_setup in Docker.
param(
    [switch]$Build,
    [string]$Config = "configs/vid_setup_12gb.yaml",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArgs
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
Usage: docker-run-vid-setup.ps1 [-Build] [-Config PATH] [-- runner args]

Run vid_setup in Docker. Video diffusion weights load from Hugging Face on first
run and are cached in the compose hf_cache volume (rnd4impact_vid_hf_cache).

Options:
  -Build               Run docker compose build before starting the container
  -Config PATH         YAML config inside the container (default: configs/vid_setup_12gb.yaml)

Environment:
  `$env:VID_SETUP_CONFIG   Default config when -Config is omitted

Examples:
  .\scripts\docker-run-vid-setup.ps1 -Build
  .\scripts\docker-run-vid-setup.ps1 -Config configs/vid_setup_12gb.yaml
"@
}

if ($RunnerArgs -contains "-h" -or $RunnerArgs -contains "--help" -or $RunnerArgs -contains "-Help") {
    Show-Usage
    exit 0
}

if ($env:VID_SETUP_CONFIG -and -not $PSBoundParameters.ContainsKey("Config")) {
    $Config = $env:VID_SETUP_CONFIG
}

$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root "docker\vid_setup")

if ($Build) {
    docker compose build
}

$dockerArgs = @(
    "compose", "run", "--rm", "vid-setup",
    "python", "vid_setup/vid_setup_runner.py",
    "--config", $Config
) + $RunnerArgs

& docker @dockerArgs
