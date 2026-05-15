param(
    [string]$OutputDir = "release",
    [string]$WorkDir = "build"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

function Resolve-BootstrapPython {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return @{
            Executable = $pythonCommand.Source
            Arguments = @()
        }
    }

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) {
        return @{
            Executable = $pyCommand.Source
            Arguments = @("-3")
        }
    }

    throw "Python was not found. Install Python 3 first, then rerun this script."
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating local virtual environment in .venv ..."
    $bootstrapPython = Resolve-BootstrapPython
    & $bootstrapPython.Executable @($bootstrapPython.Arguments + @("-m", "venv", (Join-Path $projectRoot ".venv")))
}

Write-Host "Installing build dependencies ..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements-dev.txt")

$distPath = Join-Path $projectRoot $OutputDir
$workPath = Join-Path $projectRoot $WorkDir
$entryPoint = Join-Path $projectRoot "main\Aruco_crack_len_measurement.py"

Write-Host "Building Windows executable ..."
& $venvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name ArucoCrackMeasurement `
    --distpath $distPath `
    --workpath $workPath `
    $entryPoint

$releaseFolder = Join-Path $distPath "ArucoCrackMeasurement"
$releaseReadme = Join-Path $releaseFolder "README.txt"
@"
This folder contains the packaged Windows application.

Run:
  ArucoCrackMeasurement.exe

Important:
- Keep the entire folder together when copying it to another PC.
- Do not move only the .exe file by itself; it depends on the bundled files in _internal.
"@ | Set-Content -Path $releaseReadme -Encoding UTF8

Write-Host ""
Write-Host "Build complete."
Write-Host "Packaged application folder:"
Write-Host "  $releaseFolder"
