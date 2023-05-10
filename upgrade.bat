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

:: Upgrade the required packages
pip install --use-pep517 --upgrade -r requirements.txt