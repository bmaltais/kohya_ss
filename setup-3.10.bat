@echo off

IF NOT EXIST venv (
    echo Creating venv...
    py -3.10 -m venv venv
)

:: Create the directory if it doesn't exist
mkdir ".\logs\setup" > nul 2>&1

:: Deactivate the virtual environment to prevent error
call .\venv\Scripts\deactivate.bat

call .\venv\Scripts\activate.bat

REM Check if the batch was started via double-click
IF /i "%comspec% /c %~0 " equ "%cmdcmdline:"=%" (
    REM echo This script was started by double clicking.
    cmd /k python .\setup\setup_windows.py
) ELSE (
    REM echo This script was started from a command prompt.
    python .\setup\setup_windows.py %*
)

:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat