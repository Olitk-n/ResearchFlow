$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$apiRoot = Join-Path $root "apps\api"
$webRoot = Join-Path $root "apps\web"
$pytest = Join-Path $apiRoot ".venv\Scripts\pytest.exe"
$ruff = Join-Path $apiRoot ".venv\Scripts\ruff.exe"

if (-not (Test-Path $pytest) -or -not (Test-Path $ruff)) {
    throw "Backend virtual environment is missing. Run uv sync --directory apps/api first."
}

Push-Location $apiRoot
try {
    & $pytest -q
    if ($LASTEXITCODE) { exit $LASTEXITCODE }

    & $ruff check app tests
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}

Push-Location $webRoot
try {
    npm run lint
    if ($LASTEXITCODE) { exit $LASTEXITCODE }

    $webIsRunning = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
    if ($webIsRunning) {
        Write-Host "Web application is running; skipping the production build."
    } else {
        npm run build
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
    }
} finally {
    Pop-Location
}
