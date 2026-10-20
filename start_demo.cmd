@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_demo.ps1" -Background
if errorlevel 1 (
  echo.
  echo Demo startup failed. Read the message above and press any key to close.
  pause >nul
  exit /b 1
)
start "" "http://127.0.0.1:8010"
