@echo off
rem Command-line entry point:  rgbthermal app | doctor | calibrate ... | process ... | joint-view ... | test | version
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Not installed yet. Run:  powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1"
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m src.cli %*
