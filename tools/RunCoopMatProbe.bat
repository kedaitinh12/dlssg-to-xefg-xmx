@echo off
setlocal
cd /d "%~dp0"
set "OUT=%~dp0coopmat_probe_report.txt"
if not exist "%~dp0xefg_coopmat_probe.exe" (
    echo [ERROR] xefg_coopmat_probe.exe not found next to this .bat file.
    pause
    exit /b 1
)
echo Running XeFG Vulkan Cooperative-Matrix Probe...
"%~dp0xefg_coopmat_probe.exe" "%~dp0xefg_mfg_programs\vulkan\coopmat" > "%OUT%" 2>&1
type "%OUT%"
echo.
echo Report saved to: "%OUT%"
pause
