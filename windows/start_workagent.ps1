$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$backendDir = Join-Path $root "backend"
$frontendDir = Join-Path $root "frontend"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$pythonCommand = if (Test-Path $venvPython) { $venvPython } else { "python" }
$url = "http://localhost:5173"
$backendUrl = "http://127.0.0.1:8001/api/status"

function Test-HttpReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Uri
    )

    try {
        $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400)
    }
    catch {
        return $false
    }
}

if (-not (Test-Path $backendDir)) {
    Write-Host "Backend directory not found: $backendDir"
    exit 1
}

if (-not (Test-Path $frontendDir)) {
    Write-Host "Frontend directory not found: $frontendDir"
    exit 1
}

if (Test-HttpReady $backendUrl) {
    Write-Host "Backend is already running on http://127.0.0.1:8001."
    $backendReady = $true
}
else {
    Write-Host "Starting WorkAgent backend..."
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "Set-Location '$backendDir'; & '$pythonCommand' -m uvicorn api_server:app --host 127.0.0.1 --port 8001"
    )

    Write-Host "Waiting for backend on http://127.0.0.1:8001 ..."
    $backendReady = $false
    for ($i = 1; $i -le 30; $i++) {
        if (Test-HttpReady $backendUrl) {
            $backendReady = $true
            break
        }
        Start-Sleep -Seconds 1
    }
}

if (-not $backendReady) {
    Write-Host "Backend did not become ready on port 8001. Check the backend PowerShell window for errors."
    exit 1
}

if (Test-HttpReady $url) {
    Write-Host "Frontend is already running on $url."
    $frontendReady = $true
}
else {
    Write-Host "Starting WorkAgent frontend..."
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "Set-Location '$frontendDir'; npm run dev"
    )

    Write-Host "Waiting for frontend on $url ..."
    $frontendReady = $false
    for ($i = 1; $i -le 30; $i++) {
        if (Test-HttpReady $url) {
            $frontendReady = $true
            break
        }
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
