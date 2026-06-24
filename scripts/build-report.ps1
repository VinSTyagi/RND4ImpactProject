param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Name
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
$reportsDir = Join-Path $repoRoot "reports"
$texPath = Join-Path $reportsDir "$Name.tex"

if (-not (Test-Path $texPath)) {
    throw "Report not found: $texPath"
}

Push-Location $reportsDir
try {
    latexmk -pdf -outdir="out/$Name" "$Name.tex"
    $pdf = Join-Path $reportsDir (Join-Path "out" (Join-Path $Name "$Name.pdf"))
    Write-Host "Built: $pdf"
}
finally {
    Pop-Location
}
