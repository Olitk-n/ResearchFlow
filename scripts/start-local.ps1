$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env"
$apiRoot = Join-Path $root "apps\api"
$webRoot = Join-Path $root "apps\web"

if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $root ".env.example") $envFile
}

$apiPortLine = Get-Content $envFile | Where-Object { $_ -match "^API_PORT=" } | Select-Object -First 1
$apiPort = if ($apiPortLine) { $apiPortLine.Split("=", 2)[1] } else { "8000" }
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:$apiPort"

function Test-Url([string]$url) {
    try {
        return (Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200
    } catch {
        return $false
    }
}

Write-Host "Building the stable web application..."
Push-Location $webRoot
try {
    npm run build
    if ($LASTEXITCODE) { throw "Frontend build failed." }
} finally {
    Pop-Location
}

if (-not (Test-Url "http://127.0.0.1:$apiPort/health")) {
    $python = Join-Path $apiRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        throw "Run uv sync --directory apps/api first."
    }
    $apiCommand = "& '$python' -m uvicorn app.main:app --host 127.0.0.1 --port $apiPort"
    Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $apiCommand -WorkingDirectory $apiRoot
}

$existingWeb = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
if ($existingWeb) {
    foreach ($connection in $existingWeb) {
        $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($connection.OwningProcess)" -ErrorAction SilentlyContinue
        if ($owner.CommandLine -and $owner.CommandLine.Contains($webRoot)) {
            Stop-Process -Id $owner.ProcessId -Force -ErrorAction SilentlyContinue
            if ($owner.ParentProcessId) {
                Stop-Process -Id $owner.ParentProcessId -Force -ErrorAction SilentlyContinue
            }
        } else {
            throw "Port 3000 is used by another application. Close it before starting ResearchFlow."
        }
    }
    Start-Sleep -Seconds 2
}

$webCommand = "`$env:NEXT_PUBLIC_API_URL='http://127.0.0.1:$apiPort'; npm run start -- -H 127.0.0.1 -p 3000"
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $webCommand -WorkingDirectory $webRoot

for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if (
        (Test-Url "http://127.0.0.1:$apiPort/health") -and
        (Test-Url "http://127.0.0.1:3000")
    ) {
        break
    }
    Start-Sleep -Seconds 1
}

if (-not (Test-Url "http://127.0.0.1:$apiPort/health")) {
    throw "API did not start. Read the visible API window."
}
if (-not (Test-Url "http://127.0.0.1:3000")) {
    throw "Web application did not start. Read the visible web window."
}

$html = (Invoke-WebRequest "http://127.0.0.1:3000" -UseBasicParsing).Content
$cssPath = [regex]::Match($html, 'href="([^"]*_next/static/css/[^"]+)"').Groups[1].Value
if (-not $cssPath -or -not (Test-Url "http://127.0.0.1:3000$cssPath")) {
    throw "Web stylesheet validation failed. Close the web window and run this script again."
}

Write-Host ""
Write-Host "ResearchFlow is ready."
Write-Host "Web: http://localhost:3000"
Write-Host "API: http://127.0.0.1:$apiPort/docs"
Write-Host "Close the two PowerShell windows to stop ResearchFlow."
