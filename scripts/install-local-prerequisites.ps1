$ErrorActionPreference = "Stop"

$isAdmin = (
    New-Object Security.Principal.WindowsPrincipal(
        [Security.Principal.WindowsIdentity]::GetCurrent()
    )
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Requesting administrator permission for WSL2 and Docker Desktop."
    Start-Process powershell.exe `
        -Verb RunAs `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`""
    exit
}

Write-Host "Preparing D:\ResearchFlow for local runtimes and data."
@(
    "D:\ResearchFlow\data",
    "D:\ResearchFlow\docker",
    "D:\ResearchFlow\wsl"
) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl.exe --install --no-distribution

winget install --id Docker.DockerDesktop --source winget `
    --silent --accept-package-agreements --accept-source-agreements `
    --disable-interactivity

Write-Host ""
Write-Host "WSL2 and Docker Desktop are installed or queued."
Write-Host "Restart Windows once, open Docker Desktop, then set:"
Write-Host "  Settings > Resources > Advanced > Disk image location"
Write-Host "  D:\ResearchFlow\docker"
Write-Host ""
Write-Host "After Docker reports Running, restart ResearchFlow."
