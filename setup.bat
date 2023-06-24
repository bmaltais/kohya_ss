@echo off

set PYTHON_VER=3.10.9

:: Check if Python version meets the recommended version
python --version 2>nul | findstr /b /c:"Python %PYTHON_VER%" >nul
if errorlevel 1 (
    echo Warning: Python version %PYTHON_VER% is recommended.
)

IF NOT EXIST venv (
    echo Creating venv...
    python -m venv venv
)

:: Create the directory if it doesn't exist
mkdir ".\logs\setup" > nul 2>&1

:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat

:: Calling external python program to check for local modules
python .\setup\check_local_modules.py

call .\venv\Scripts\activate.bat

REM Check if the batch was started via double-click
IF /i "%comspec% /c %~0 " equ "%cmdcmdline:"=%" (
    REM echo This script was started by double clicking.
    cmd /k python .\setup\setup_windows.py
) ELSE (
    REM echo This script was started from a command prompt.
    python .\setup\setup_windows.py
)

:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat