@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0results\living_original.npz" (
    "%~dp0..\.venv\Scripts\python.exe" "%~dp0viewer.py" --checkpoint "%~dp0results\living_original.npz" --save-path "%~dp0results\living_original.npz" %*
) else (
    "%~dp0..\.venv\Scripts\python.exe" "%~dp0viewer.py" --checkpoint "%~dp0results\active_body_additive_600.npz" --save-path "%~dp0results\living_original.npz" %*
)
if errorlevel 1 pause
endlocal
