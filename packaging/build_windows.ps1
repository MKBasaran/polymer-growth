# Build the Windows .exe and wrap it into an Inno Setup installer.
#
# Prereqs:
#   - Windows
#   - Python 3.10 installed and on PATH (matches macOS dev version)
#   - Inno Setup 6+ installed (https://jrsoftware.org/isdl.php) — only required
#     if you want the .exe installer. Without it you still get a portable folder
#     bundle in dist\Polymer Growth Simulator\.
#
# Output:
#   dist\Polymer Growth Simulator\PolymerGrowthSimulator.exe       (portable)
#   dist\PolymerGrowthSimulator-Setup-0.1.0.exe                    (installer)
#
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Version = "0.1.0"

if (-Not (Test-Path ".venv")) {
    Write-Host "Creating venv (Python 3.10)..."
    python -m venv .venv
}

$Py = ".\.venv\Scripts\python.exe"
$Pip = ".\.venv\Scripts\pip.exe"

Write-Host "==> Upgrading pip..."
& $Py -m pip install --quiet --upgrade pip

Write-Host "==> Installing project + PyInstaller..."
& $Pip install --quiet -e ".[gui,dev]"
& $Pip install --quiet "pyinstaller>=6.6,<7.0"

Write-Host "==> Cleaning prior build/dist..."
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist")  { Remove-Item -Recurse -Force "dist"  }

Write-Host "==> Running PyInstaller..."
& $Py -m PyInstaller packaging\polymer_sim.spec --noconfirm --clean --log-level WARN

$BundleDir = "dist\Polymer Growth Simulator"
if (-Not (Test-Path $BundleDir)) {
    Write-Error "PyInstaller did not produce $BundleDir. Build failed."
    exit 1
}

Write-Host "==> Built $BundleDir"

# Try to find Inno Setup. If absent, stop after the portable bundle.
$Iscc = $null
$CandidatePaths = @(
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
)
foreach ($p in $CandidatePaths) {
    if (Test-Path $p) { $Iscc = $p; break }
}

if ($null -eq $Iscc) {
    Write-Host ""
    Write-Host "Inno Setup not found. Skipping installer step." -ForegroundColor Yellow
    Write-Host "Install from https://jrsoftware.org/isdl.php to produce the .exe installer."
    Write-Host "Portable bundle is ready at: $BundleDir"
    exit 0
}

Write-Host "==> Building Inno Setup installer..."
& $Iscc /Qp "/DAppVersion=$Version" packaging\polymer_sim.iss
if ($LASTEXITCODE -ne 0) {
    Write-Error "Inno Setup compilation failed."
    exit 1
}

Write-Host ""
Write-Host "Done."
Write-Host "Portable bundle: $BundleDir"
Write-Host "Installer:       dist\PolymerGrowthSimulator-Setup-$Version.exe"
