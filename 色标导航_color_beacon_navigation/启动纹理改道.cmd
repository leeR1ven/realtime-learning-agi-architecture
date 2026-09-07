@echo off
setlocal
cd /d "%~dp0"
set "TEXTURE_ROUTE_PYTHON=%~dp0..\.venv\Scripts\pythonw.exe"
if not exist "%TEXTURE_ROUTE_PYTHON%" (
    echo Python runtime was not found in the project .venv folder.
    pause
    exit /b 1
)
start "" "%TEXTURE_ROUTE_PYTHON%" "%~dp0route_switch_viewer.py" --model "%~dp0results\route_texture_evaluation.npz" --texture-seed 44017
endlocal
