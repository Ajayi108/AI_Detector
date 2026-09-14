# Build the local AI Detector desktop app into a Windows exe folder.
param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Create a local virtual environment for the build.
if (!(Test-Path ".venv")) {
    & $Python -m venv .venv
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

# Build the Tkinter app into a Windows executable folder.
& $VenvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name "AI Detector" `
    --paths "src" `
    "src\ai_detector\app.py"

Write-Host ""
Write-Host "Build complete: dist\AI Detector\AI Detector.exe"
