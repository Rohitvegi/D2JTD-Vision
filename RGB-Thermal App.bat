@echo off
rem Start the desktop application (no console window).
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
  echo Not installed yet. Run:  powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1"
  pause
  exit /b 1
)
start "" /d "%~dp0" "%~dp0.venv\Scripts\pythonw.exe" -m src.cli app
