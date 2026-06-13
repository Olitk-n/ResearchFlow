$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$processFile = Join-Path $root "work\local-processes.json"

if (-not (Test-Path $processFile)) {
    Write-Host "No ResearchFlow process record was found."
    exit 0
}

$record = Get-Content $processFile -Raw | ConvertFrom-Json
foreach ($name in @("api", "web")) {
    $processId = $record.$name
    if (-not $processId) { continue }

    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id $processId
        Write-Host "Stopped $name process ($processId)."
    }
}

Remove-Item -LiteralPath $processFile
Write-Host "ResearchFlow stopped."
