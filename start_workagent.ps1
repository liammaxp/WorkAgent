$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $root "backend"
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
    "Set-Location '$backendDir'; python -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8001"
)

Write-Host "Waiting for backend on http://127.0.0.1:8001 ..."
$backendReady = $false
for ($i = 1; $i -le 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:8001/api/status" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            $backendReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $backendReady) {
    Write-Host "Backend did not become ready on port 8001. Check the backend PowerShell window for errors."
    exit 1
}

Write-Host "Starting WorkAgent frontend..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$frontendDir'; npm run dev"
)

Write-Host "Waiting for frontend on $url ..."
$frontendReady = $false
for ($i = 1; $i -le 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            $frontendReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $frontendReady) {
    Write-Host "Frontend did not become ready on port 5173. Check the frontend PowerShell window for errors."
    exit 1
}

Write-Host "Opening $url ..."
Start-Process $url

Write-Host ""
Write-Host "WorkAgent is starting."
Write-Host "If the page opens before Vite is ready, refresh it after a few seconds."
