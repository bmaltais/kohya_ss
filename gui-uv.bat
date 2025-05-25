@echo off
set VIRTUAL_ENV=.venv
echo VIRTUAL_ENV is set to %VIRTUAL_ENV%

:: Check if uv is installed
setlocal enabledelayedexpansion
where uv >nul 2>nul
if %errorlevel% neq 0 (
    set /p INSTALL_UV="uv is not installed. We can try to install it for you, or you can install it manually from https://astral.sh/uv before running this script again. Would you like to attempt automatic installation now? (Y/N) "
    if /i "!INSTALL_UV!"=="Y" (
        winget install --id=astral-sh.uv  -e
    ) else (
        echo Okay, please install uv manually from https://astral.sh/uv and then re-run this script. Exiting.
        exit /b 1
    )
)
endlocal

set PATH=%PATH%;%~dp0venv\Lib\site-packages\torch\lib

echo Starting the GUI... this might take some time... Especially on 1st run after install or update...

:: Make sure we are on the right sd-scripts commit
git submodule update --init --recursive

:: If the exit code is 0, run the kohya_gui.py script with the command-line arguments
if %errorlevel% equ 0 (
    REM Check if the batch was started via double-click
    IF /i "%comspec% /c %~0 " equ "%cmdcmdline:"=%" (
        REM echo This script was started by double clicking.
        cmd /k uv run --link-mode=copy --index-strategy unsafe-best-match kohya_gui.py --noverify %*
    ) ELSE (
        REM echo This script was started from a command prompt.
        uv run --link-mode=copy --index-strategy unsafe-best-match kohya_gui.py --noverify %*
    )
)