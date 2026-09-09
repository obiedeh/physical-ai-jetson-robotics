param(
    [string]$Target = "windows-dev",
    [string]$Output = "reports/inventory/windows_dev.json",
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    throw "Python executable not found at $Python. Run scripts/windows/bootstrap.ps1 first."
}

& $Python -m physical_ai_lab.cli collect-inventory --target $Target --output $Output
