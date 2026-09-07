@echo off
setlocal
cd /d "%~dp0"
set "NAVIGATION_PYTHON=%~dp0..\.venv\Scripts\pythonw.exe"
if not exist "%NAVIGATION_PYTHON%" (
    echo Python runtime was not found in the project .venv folder.
    pause
    exit /b 1
)
start "" "%NAVIGATION_PYTHON%" "%~dp0viewer.py"
endlocal
