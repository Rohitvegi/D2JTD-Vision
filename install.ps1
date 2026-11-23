<#
  Install RGB-Thermal Hand-Joint Analysis on Windows (x64).

  .\install.ps1               create .venv with Python 3.12 and the exact package versions, then run the health check
  .\install.ps1 -Offline      use only the local uv package cache (no download)
  .\install.ps1 -Shortcut     also put a desktop shortcut to the application

  Needs: uv (https://docs.astral.sh/uv/) or Python 3.12 with the 'py' launcher.
#>
param([switch]$Offline, [switch]$Shortcut)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
Write-Host "Installing into $PSScriptRoot" -ForegroundColor Cyan

$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) {
    $off = @(); if ($Offline) { $off = @("--offline") }
    & uv venv .venv --python 3.12 @off
    if ($LASTEXITCODE -ne 0) { throw "could not create the Python 3.12 environment (uv venv)" }
    & uv pip install --python .venv\Scripts\python.exe -r requirements.lock.txt @off
    if ($LASTEXITCODE -ne 0) { throw "package installation failed" }
} else {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if (-not $py) { throw "Install uv (recommended) or Python 3.12 first." }
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 not found (py -3.12)" }
    & .venv\Scripts\python.exe -m pip install --upgrade pip
    & .venv\Scripts\python.exe -m pip install -r requirements.lock.txt
    if ($LASTEXITCODE -ne 0) { throw "package installation failed" }
}

Write-Host "`nChecking the installation..." -ForegroundColor Cyan
& .venv\Scripts\python.exe -m src.doctor
$ok = ($LASTEXITCODE -eq 0)

if ($Shortcut) {
    $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "RGB-Thermal Hand Analysis.lnk"
    $ws = New-Object -ComObject WScript.Shell
    $s = $ws.CreateShortcut($lnk)
    $s.TargetPath = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
    $s.Arguments = "-m src.cli app"
    $s.WorkingDirectory = $PSScriptRoot
    $s.Save()
    Write-Host "Desktop shortcut created: $lnk"
}

if ($ok) {
    Write-Host "`nInstalled. Start the application with 'RGB-Thermal App.bat' or: rgbthermal.bat app" -ForegroundColor Green
} else {
    Write-Host "`nInstalled with problems: see the FAIL lines above." -ForegroundColor Yellow
    exit 1
}
