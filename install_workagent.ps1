$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $root "backend"
$frontendDir = Join-Path $root "frontend"
$requirementsFile = Join-Path $backendDir "requirements.txt"
$packageFile = Join-Path $frontendDir "package.json"

function Assert-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$InstallHint
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Host "Required command not found: $Name"
        Write-Host $InstallHint
        exit 1
    }
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,

        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host $FailureMessage
        exit $LASTEXITCODE
    }
}

if (-not (Test-Path $requirementsFile)) {
    Write-Host "Backend requirements file not found: $requirementsFile"
    exit 1
}

if (-not (Test-Path $packageFile)) {
    Write-Host "Frontend package file not found: $packageFile"
    exit 1
}

Assert-Command "python" "Install Python 3 and make sure python is available in PATH."
Assert-Command "npm" "Install Node.js and make sure npm is available in PATH."

Write-Host "Installing WorkAgent backend dependencies..."
Push-Location $backendDir
try {
    Invoke-CheckedCommand { python -m pip install -r requirements.txt } "Backend dependency installation failed."
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Installing WorkAgent frontend dependencies..."
Push-Location $frontendDir
try {
    Invoke-CheckedCommand { npm install } "Frontend dependency installation failed."
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Dependency installation completed successfully."
