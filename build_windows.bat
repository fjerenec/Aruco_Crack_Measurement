@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1"
if errorlevel 1 (
    echo.
    echo Build failed.
) else (
    echo.
    echo Build finished successfully.
)
pause
