@echo off
setlocal

REM Define color variables
set "yellow_text=[1;33m"
set "blue_text=[1;34m"
set "reset_text=[0m"

:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat

REM Run pip freeze and capture the output
for /f "delims=" %%I in ('pip freeze') do (
    set "pip_output=%%I"
    goto :CheckModules
)

:CheckModules
REM Check if modules are found in the output
if defined pip_output (
    echo %yellow_text%=============================================================
    echo Modules installed outside the virtual environment were found.
    echo This can cause issues. Please review the installed modules.
    echo.
    echo You can uninstall all local modules with:
    echo.
    echo %blue_text%deactivate
    echo pip freeze ^> uninstall.txt
    echo pip uninstall -y -r uninstall.txt
    echo %yellow_text%=============================================================%reset_text%
)

endlocal

:: Activate the virtual environment
call .\venv\Scripts\activate.bat
set PATH=%PATH%;%~dp0venv\Lib\site-packages\torch\lib

:: Debug info about system
:: python.exe .\tools\debug_info.py

:: Validate the requirements and store the exit code
python.exe .\tools\validate_requirements.py

:: If the exit code is 0, run the kohya_gui.py script with the command-line arguments
if %errorlevel% equ 0 (
    python.exe kohya_gui.py %*
)