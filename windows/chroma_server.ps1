param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("start", "health", "stop", "restart")]
    [string]$Command,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArguments
)

$ErrorActionPreference = "Stop"
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDirectory ".."))
$venvPython = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$pythonCommand = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

Push-Location -LiteralPath $repositoryRoot
try {
    & $pythonCommand -m backend.chroma_server_lifecycle $Command --json @RemainingArguments
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $exitCode
