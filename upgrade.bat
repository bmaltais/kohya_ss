@echo off
:: Check if there are any changes that need to be committed
git status --short
if %errorlevel%==1 (
    echo There are changes that need to be committed. Please stash or undo your changes before running this script.
    exit
)

:: Pull the latest changes from the remote repository
git pull

:: Activate the virtual environment
call .\venv\Scripts\activate.bat

REM Check if torch_version is 1
findstr /C:"1" ".\logs\status\torch_version" >nul

REM Check the error level to determine if the text was found
if %errorlevel% equ 0 (
    echo Torch version 1...
    pip install --use-pep517 --upgrade -r requirements_windows_torch1.txt
) else (
    echo Torch version 2...
    pip install --use-pep517 --upgrade -r requirements_windows_torch2.txt
)
