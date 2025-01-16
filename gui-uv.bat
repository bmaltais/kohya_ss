@echo off

set PYTHON_VER=3.10.9

:: Update pip to latest version
pip install --upgrade uv -q

set PATH=%PATH%;%~dp0venv\Lib\site-packages\torch\lib

echo Starting the GUI... this might take some time...

:: If the exit code is 0, run the kohya_gui.py script with the command-line arguments
if %errorlevel% equ 0 (
    REM Check if the batch was started via double-click
    IF /i "%comspec% /c %~0 " equ "%cmdcmdline:"=%" (
        REM echo This script was started by double clicking.
        cmd /k uv run kohya_gui.py --noverify %*
    ) ELSE (
        REM echo This script was started from a command prompt.
        uv run kohya_gui.py --noverify %*
    )
)
