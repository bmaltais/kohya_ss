@echo off

:: Install uv latest version
pip install --upgrade uv -q

set PATH=%PATH%;%~dp0venv\Lib\site-packages\torch\lib

echo Starting the GUI... this might take some time... Especially on 1st run after install or update...

:: Make sure we are on the right sd-scripts commit
git submodule update --init --recursive

:: If the exit code is 0, run the kohya_gui.py script with the command-line arguments
if %errorlevel% equ 0 (
    REM Check if the batch was started via double-click
    IF /i "%comspec% /c %~0 " equ "%cmdcmdline:"=%" (
        REM echo This script was started by double clicking.
        cmd /k uv run --link-mode=copy kohya_gui.py --noverify %*
    ) ELSE (
        REM echo This script was started from a command prompt.
        uv run --link-mode=copy kohya_gui.py --noverify %*
    )
)
