$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $root "my-agent"
$frontendDir = Join-Path $root "frontend"
$url = "http://localhost:5173"

if (-not (Test-Path $backendDir)) {
    Write-Host "Backend directory not found: $backendDir"
    exit 1
}

if (-not (Test-Path $frontendDir)) {
    Write-Host "Frontend directory not found: $frontendDir"
    exit 1
}

Write-Host "Starting WorkAgent backend..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$backendDir'; python -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8000"
)

Write-Host "Starting WorkAgent frontend..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$frontendDir'; npm run dev"
)

Write-Host "Opening $url ..."
Start-Sleep -Seconds 5
Start-Process $url

Write-Host ""
Write-Host "WorkAgent is starting."
Write-Host "If the page opens before Vite is ready, refresh it after a few seconds."
