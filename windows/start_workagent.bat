@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_workagent.ps1"

endlocal
