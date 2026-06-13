$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env"

if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $root ".env.example") $envFile
    Write-Host "Created .env. Replace SECRET_KEY and ENCRYPTION_KEY before storing API keys."
}

$apiPortLine = Get-Content $envFile | Where-Object { $_ -match "^API_PORT=" } | Select-Object -First 1
$apiPort = if ($apiPortLine) { $apiPortLine.Split("=", 2)[1] } else { "8000" }
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:$apiPort"

$venvUvicorn = Join-Path $root "apps\api\.venv\Scripts\uvicorn.exe"
if (Test-Path $venvUvicorn) {
    Start-Process -FilePath $venvUvicorn `
        -ArgumentList "app.main:app", "--reload", "--host", "127.0.0.1", "--port", $apiPort `
        -WorkingDirectory (Join-Path $root "apps\api")
} else {
    Start-Process -FilePath "uv" `
        -ArgumentList "run", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", $apiPort `
        -WorkingDirectory (Join-Path $root "apps\api")
}

Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory (Join-Path $root "apps\web")

Write-Host "ResearchFlow is starting in two visible terminal windows."
Write-Host "Web: http://localhost:3000"
Write-Host "API: http://127.0.0.1:$apiPort/docs"
Write-Host "Close both terminal windows to stop ResearchFlow."
