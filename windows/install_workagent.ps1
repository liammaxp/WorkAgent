$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$backendDir = Join-Path $root "backend"
$frontendDir = Join-Path $root "frontend"
$requirementsFile = Join-Path $backendDir "requirements.txt"
$packageFile = Join-Path $frontendDir "package.json"
$latexWarmupDir = Join-Path $root "outputs\latex_install_warmup"
$venvDir = Join-Path $root ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

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

function Test-LatexCompiler {
    return [bool](
        (Get-Command "latexmk" -ErrorAction SilentlyContinue) -or
        (Get-Command "xelatex" -ErrorAction SilentlyContinue) -or
        (Get-Command "pdflatex" -ErrorAction SilentlyContinue)
    )
}

function Update-SessionPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $extraPaths = @(
        (Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"),
        "C:\Program Files\MiKTeX\miktex\bin\x64",
        "C:\Strawberry\perl\bin",
        "C:\Strawberry\c\bin"
    )
    $env:Path = @($machinePath, $userPath, $env:Path) + $extraPaths -join ";"
}

function Install-PerlRuntime {
    Update-SessionPath
    if (Get-Command "perl" -ErrorAction SilentlyContinue) {
        Write-Host "Perl runtime already available."
        return
    }

    Write-Host ""
    Write-Host "Perl runtime not found. Installing Strawberry Perl for latexmk..."

    if (-not (Get-Command "winget" -ErrorAction SilentlyContinue)) {
        Write-Host "winget is required to install Strawberry Perl automatically."
        Write-Host "Install App Installer from Microsoft Store, then rerun install_workagent.bat."
        exit 1
    }

    Invoke-CheckedCommand {
        winget install --id StrawberryPerl.StrawberryPerl --exact --source winget --silent --accept-package-agreements --accept-source-agreements
    } "Strawberry Perl installation failed."

    Update-SessionPath
    if (Get-Command "perl" -ErrorAction SilentlyContinue) {
        Write-Host "Strawberry Perl installed and perl is available."
        return
    }

    Write-Host "Strawberry Perl was installed, but perl was not found in the current PATH."
    Write-Host "Close and reopen PowerShell or restart Windows, then rerun install_workagent.bat."
    exit 1
}

function Install-LatexToolchain {
    if (Test-LatexCompiler) {
        Write-Host "LaTeX compiler already available."
        Install-PerlRuntime
        return
    }

    Write-Host ""
    Write-Host "LaTeX compiler not found. Installing MiKTeX for PDF export..."

    if (-not (Get-Command "winget" -ErrorAction SilentlyContinue)) {
        Write-Host "winget is required to install MiKTeX automatically."
        Write-Host "Install App Installer from Microsoft Store, then rerun install_workagent.bat."
        exit 1
    }

    Invoke-CheckedCommand {
        winget install --id MiKTeX.MiKTeX --exact --source winget --silent --accept-package-agreements --accept-source-agreements
    } "MiKTeX installation failed."

    Update-SessionPath
    if (Test-LatexCompiler) {
        Write-Host "MiKTeX installed and LaTeX compiler is available."
        Install-PerlRuntime
        return
    }

    Write-Host "MiKTeX was installed, but latexmk/xelatex/pdflatex was not found in the current PATH."
    Write-Host "Close and reopen PowerShell or restart Windows, then rerun install_workagent.bat."
    exit 1
}

function Initialize-LatexPackages {
    Update-SessionPath
    $xelatex = Get-Command "xelatex" -ErrorAction SilentlyContinue
    $pdflatex = Get-Command "pdflatex" -ErrorAction SilentlyContinue
    if (-not $xelatex -and -not $pdflatex) {
        Write-Host "No xelatex or pdflatex command found; skipping LaTeX package warmup."
        return
    }

    Write-Host ""
    Write-Host "Warming up MiKTeX packages for resume PDF export..."

    $initexmf = Get-Command "initexmf" -ErrorAction SilentlyContinue
    if ($initexmf) {
        & $initexmf.Source --set-config-value="[MPM]AutoInstall=1"
        & $initexmf.Source --update-fndb
    }

    New-Item -ItemType Directory -Force -Path $latexWarmupDir | Out-Null
    $warmupTex = Join-Path $latexWarmupDir "workagent_latex_warmup.tex"

    if ($xelatex) {
        $compiler = $xelatex.Source
        $warmupContent = @'
\documentclass[11pt]{article}
\usepackage[margin=0.5in]{geometry}
\usepackage{fontspec}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{tabularx}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{fontawesome5}
\hypersetup{colorlinks=true,urlcolor=blue}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\begin{document}
\section{WorkAgent LaTeX Warmup}
\begin{tabularx}{\textwidth}{X r}
\textbf{Tailored Resume PDF Export} & \href{https://example.com}{example link} \\
\end{tabularx}
\begin{itemize}[leftmargin=*]
\item Common resume packages are installed and ready. \faGithub
\end{itemize}
\end{document}
'@
    }
    else {
        $compiler = $pdflatex.Source
        $warmupContent = @'
\documentclass[11pt]{article}
\usepackage[margin=0.5in]{geometry}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{tabularx}
\usepackage{array}
\usepackage{ragged2e}
\hypersetup{colorlinks=true,urlcolor=blue}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\begin{document}
\section{WorkAgent LaTeX Warmup}
\begin{tabularx}{\textwidth}{X r}
\textbf{Tailored Resume PDF Export} & \href{https://example.com}{example link} \\
\end{tabularx}
\begin{itemize}[leftmargin=*]
\item Common resume packages are installed and ready.
\end{itemize}
\end{document}
'@
    }

    Set-Content -Path $warmupTex -Value $warmupContent -Encoding UTF8
    for ($i = 1; $i -le 2; $i++) {
        Invoke-CheckedCommand {
            & $compiler -interaction=nonstopmode -halt-on-error "-output-directory=$latexWarmupDir" $warmupTex
        } "LaTeX package warmup failed."
    }

    Write-Host "LaTeX package warmup completed."
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
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating WorkAgent virtual environment..."
    Invoke-CheckedCommand { python -m venv $venvDir } "Python virtual environment creation failed."
}
Push-Location $backendDir
try {
    Invoke-CheckedCommand { & $venvPython -m pip install --upgrade pip } "pip upgrade failed."
    Invoke-CheckedCommand { & $venvPython -m pip install -r requirements.txt } "Backend dependency installation failed."
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

Install-LatexToolchain
Initialize-LatexPackages

Write-Host ""
Write-Host "Dependency installation completed successfully."
