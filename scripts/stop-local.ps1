$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$apiRoot = Join-Path $root "apps\api"
$webRoot = Join-Path $root "apps\web"
$stopped = 0

foreach ($process in Get-CimInstance Win32_Process) {
    $command = $process.CommandLine
    if (-not $command) { continue }

    $isApi = $command.Contains($apiRoot) -and $command.Contains("uvicorn")
    $isWeb = $command.Contains($webRoot) -and (
        $command.Contains("next start") -or
        $command.Contains("npm run start")
    )
    if ($isApi -or $isWeb) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        $stopped++
    }
}

if ($stopped) {
    Write-Host "ResearchFlow stopped."
} else {
    Write-Host "ResearchFlow is not running."
}
