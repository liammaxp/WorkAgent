$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $root "backend"
$frontendDir = Join-Path $root "frontend"
$requirementsFile = Join-Path $backendDir "requirements.txt"
$nodeModulesDir = Join-Path $frontendDir "node_modules"
$latexWarmupDir = Join-Path $root "outputs\latex_install_warmup"

function Confirm-Action {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Prompt
    )

    $answer = Read-Host "$Prompt [y/N]"
    return $answer -match '^(y|yes)$'
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

function Remove-DirectoryInsideWorkspace {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not (Test-Path $Path)) {
        Write-Host "$Label not found; skipping."
        return
    }

    $resolvedRoot = [System.IO.Path]::GetFullPath($root)
    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $resolvedPath.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Write-Host "Refusing to remove path outside workspace: $resolvedPath"
        exit 1
    }

    Write-Host "Removing $Label..."
    Remove-Item -LiteralPath $resolvedPath -Recurse -Force
}

function Uninstall-PythonPackages {
    if (-not (Test-Path $requirementsFile)) {
        Write-Host "Backend requirements file not found; skipping Python package uninstall."
        return
    }

    if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
        Write-Host "python not found; skipping Python package uninstall."
        return
    }

    Write-Host "Uninstalling backend Python packages from the current Python environment..."
    Push-Location $backendDir
    try {
        Invoke-CheckedCommand { python -m pip uninstall -r requirements.txt -y } "Python package uninstall failed."
    }
    finally {
        Pop-Location
    }
}

function Test-MiktexInstalled {
    if ((Get-Command "latexmk" -ErrorAction SilentlyContinue) -or
        (Get-Command "xelatex" -ErrorAction SilentlyContinue) -or
        (Get-Command "pdflatex" -ErrorAction SilentlyContinue)) {
        return $true
    }

    if (Get-Command "winget" -ErrorAction SilentlyContinue) {
        $result = winget list --id MiKTeX.MiKTeX --exact 2>$null
        return $LASTEXITCODE -eq 0 -and ($result -match "MiKTeX")
    }

    return $false
}

function Test-StrawberryPerlInstalled {
    if (Get-Command "perl" -ErrorAction SilentlyContinue) {
        return $true
    }

    if (Get-Command "winget" -ErrorAction SilentlyContinue) {
        $result = winget list --id StrawberryPerl.StrawberryPerl --exact 2>$null
        return $LASTEXITCODE -eq 0 -and ($result -match "Strawberry")
    }

    return $false
}

function Uninstall-Miktex {
    if (-not (Test-MiktexInstalled)) {
        Write-Host "MiKTeX/LaTeX compiler not detected; skipping."
        return
    }

    if (-not (Get-Command "winget" -ErrorAction SilentlyContinue)) {
        Write-Host "winget not found; cannot automatically uninstall MiKTeX."
        Write-Host "Uninstall MiKTeX manually from Windows Settings if needed."
        return
    }

    Write-Host "Uninstalling MiKTeX..."
    Invoke-CheckedCommand {
        winget uninstall --id MiKTeX.MiKTeX --exact --silent --accept-source-agreements
    } "MiKTeX uninstall failed."
}

function Uninstall-StrawberryPerl {
    if (-not (Test-StrawberryPerlInstalled)) {
        Write-Host "Strawberry Perl not detected; skipping."
        return
    }

    if (-not (Get-Command "winget" -ErrorAction SilentlyContinue)) {
        Write-Host "winget not found; cannot automatically uninstall Strawberry Perl."
        Write-Host "Uninstall Strawberry Perl manually from Windows Settings if needed."
        return
    }

    Write-Host "Uninstalling Strawberry Perl..."
    Invoke-CheckedCommand {
        winget uninstall --id StrawberryPerl.StrawberryPerl --exact --silent --accept-source-agreements
    } "Strawberry Perl uninstall failed."
}

Write-Host "WorkAgent environment uninstall"
Write-Host "Workspace: $root"
Write-Host ""

Remove-DirectoryInsideWorkspace $nodeModulesDir "frontend node_modules"
Remove-DirectoryInsideWorkspace $latexWarmupDir "LaTeX package warmup files"

Write-Host ""
Write-Host "Python packages were installed into the current Python environment."
Write-Host "Only uninstall them if this Python environment is dedicated to WorkAgent."
if (Confirm-Action "Uninstall backend Python packages from backend/requirements.txt?") {
    Uninstall-PythonPackages
}
else {
    Write-Host "Skipping Python package uninstall."
}

Write-Host ""
Write-Host "MiKTeX may be shared by other LaTeX projects."
if (Confirm-Action "Uninstall MiKTeX/LaTeX toolchain installed for PDF export?") {
    Uninstall-Miktex
}
else {
    Write-Host "Skipping MiKTeX uninstall."
}

Write-Host ""
Write-Host "Strawberry Perl may be shared by other Perl or LaTeX tools."
if (Confirm-Action "Uninstall Strawberry Perl installed for latexmk support?") {
    Uninstall-StrawberryPerl
}
else {
    Write-Host "Skipping Strawberry Perl uninstall."
}

Write-Host ""
Write-Host "WorkAgent environment uninstall completed."
