# Build (optional) and run script_setup in Docker.
param(
    [switch]$Build,
    [string]$Config = "configs/script_setup_12gb.yaml",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArgs
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
Usage: docker-run-script-setup.ps1 [-Build] [-Config PATH] [-- runner args]

Run script_setup in Docker. LLM weights load from Hugging Face on first run
and are cached in the compose hf_cache volume (rnd4impact_script_hf_cache).

Options:
  -Build               Run docker compose build before starting the container
  -Config PATH         YAML config inside the container (default: configs/script_setup_12gb.yaml)

Environment:
  `$env:SCRIPT_SETUP_CONFIG   Default config when -Config is omitted

Examples:
  .\scripts\docker-run-script-setup.ps1 -Build
  .\scripts\docker-run-script-setup.ps1 -Config configs/script_setup_12gb.yaml -- -all
  .\scripts\docker-run-script-setup.ps1 -- -4
"@
}

if ($RunnerArgs -contains "-h" -or $RunnerArgs -contains "--help" -or $RunnerArgs -contains "-Help") {
    Show-Usage
    exit 0
}

if ($env:SCRIPT_SETUP_CONFIG -and -not $PSBoundParameters.ContainsKey("Config")) {
    $Config = $env:SCRIPT_SETUP_CONFIG
}

if ($RunnerArgs.Count -eq 0) {
    $RunnerArgs = @("--all")
}

$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root "docker\script_setup")

if ($Build) {
    docker compose build
}

$dockerArgs = @(
    "compose", "run", "--rm", "script-setup",
    "python", "script_setup/script_setup_runner.py",
    "--config", $Config
) + $RunnerArgs

& docker @dockerArgs
