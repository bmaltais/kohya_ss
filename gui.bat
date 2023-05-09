@echo off

:: Activate the virtual environment
call .\venv\Scripts\activate.bat
set PATH=%PATH%;%~dp0venv\Lib\site-packages\torch\lib

:: Debug info about system
python.exe .\tools\debug_info.py

:: Validate the requirements and store the exit code
python.exe .\tools\validate_requirements.py

:: If the exit code is 0, run the kohya_gui.py script with the command-line arguments
if %errorlevel% equ 0 (
    python.exe kohya_gui.py %*
)