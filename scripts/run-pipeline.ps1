# Run script_setup → image_setup → vid_setup sequentially (native, not Docker).
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Runner = Join-Path $Root "src\pipeline_runner.py"

if (-not (Test-Path $Python)) {
    throw "Missing $Python. Run .\scripts\install.ps1 first."
}

if (-not $env:HF_HUB_DISABLE_XET) {
    $env:HF_HUB_DISABLE_XET = "1"
}

& $Python $Runner @args
