# Build (optional) and run image_setup in Docker.
param(
    [switch]$Build,
    [string]$Config = "configs/image_setup_sdxl_lightning_16step.yaml",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArgs
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
Usage: docker-run-image-setup.ps1 [-Build] [-Config PATH] [-- runner args]

Run image_setup in Docker. SDXL weights load from Hugging Face on first run
and are cached in the compose hf_cache volume (rnd4impact_image_hf_cache).

Options:
  -Build               Run docker compose build before starting the container
  -Config PATH         YAML config inside the container (default: configs/image_setup_sdxl_fp16.yaml)

Environment:
  `$env:IMAGE_SETUP_CONFIG   Default config when -Config is omitted

Examples:
  .\scripts\docker-run-image-setup.ps1 -Build
  .\scripts\docker-run-image-setup.ps1 -Config configs/image_setup_sdxl_low_vram.yaml
"@
}

if ($RunnerArgs -contains "-h" -or $RunnerArgs -contains "--help" -or $RunnerArgs -contains "-Help") {
    Show-Usage
    exit 0
}

if ($env:IMAGE_SETUP_CONFIG -and -not $PSBoundParameters.ContainsKey("Config")) {
    $Config = $env:IMAGE_SETUP_CONFIG
}

$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root "docker\image_setup")

if ($Build) {
    docker compose build
}

$dockerArgs = @(
    "compose", "run", "--rm", "image-setup",
    "python", "image_setup/image_setup_runner.py",
    "--config", $Config
) + $RunnerArgs

& docker @dockerArgs
