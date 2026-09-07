@echo off
setlocal
cd /d "%~dp0"
set "AGI_PYTHON=%~dp0..\.venv\Scripts\pythonw.exe"
if not exist "%AGI_PYTHON%" (
    echo Python runtime was not found in the project .venv folder.
    pause
    exit /b 1
)
start "" "%AGI_PYTHON%" "%~dp0viewer.py"
endlocal
