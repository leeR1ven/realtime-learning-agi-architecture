@echo off
setlocal
cd /d "%~dp0"
set "ROUTE_SWITCH_PYTHON=%~dp0..\.venv\Scripts\pythonw.exe"
if not exist "%ROUTE_SWITCH_PYTHON%" (
    echo Python runtime was not found in the project .venv folder.
    pause
    exit /b 1
)
start "" "%ROUTE_SWITCH_PYTHON%" "%~dp0route_switch_viewer.py"
endlocal
