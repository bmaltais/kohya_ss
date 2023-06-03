@echo off
:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat

:: Calling external python program to check for local modules
python .\tools\check_local_modules.py --no_question

:: Activate the virtual environment
call .\venv\Scripts\activate.bat
set PATH=%PATH%;%~dp0venv\Lib\site-packages\torch\lib

:: Debug info about system
:: python.exe .\tools\debug_info.py

:: Check if torch_version is 1
findstr /C:"1" ".\logs\status\torch_version" >nul

:: Check the error level to determine if the text was found
if %errorlevel% equ 0 (
    python.exe .\tools\validate_requirements.py -r requirements_windows_torch1.txt
) else (
    python.exe .\tools\validate_requirements.py -r requirements_windows_torch2.txt
)

:: If the exit code is 0, run the kohya_gui.py script with the command-line arguments
if %errorlevel% equ 0 (
    python.exe kohya_gui.py %*
)