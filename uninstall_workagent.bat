@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall_workagent.ps1"
set "exitCode=%ERRORLEVEL%"

echo.
if not "%exitCode%"=="0" (
    echo WorkAgent environment uninstall failed. Check the message above.
    pause
    exit /b %exitCode%
)

echo WorkAgent environment uninstall completed.
pause

endlocal
