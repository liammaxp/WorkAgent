@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_workagent.ps1"
set "exitCode=%ERRORLEVEL%"

echo.
if not "%exitCode%"=="0" (
    echo WorkAgent dependency installation failed. Check the message above.
    pause
    exit /b %exitCode%
)

echo WorkAgent dependencies are ready.
pause

endlocal
