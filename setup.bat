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
mkdir ".\logs\status" > nul 2>&1

:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat

:: Calling external python program to check for local modules
python .\tools\check_local_modules.py
if %errorlevel% equ 1 (
    exit /b
)

call .\venv\Scripts\activate.bat

:: Upgrade pip if needed
pip install --upgrade pip

echo.
echo Please choose the version of torch you want to install:
echo [1] - v1 (torch 1.12.1) (Recommended for best compatibility)
echo [2] - v2 (torch 2.0.0) (Experimental, faster but more prone to issues)
set /p choice="Enter your choice (1 or 2): "

if %choice%==1 (
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    del ".\logs\status\torch_version" > nul 2>&1
    echo 1 > ".\logs\status\torch_version"
    pip install --use-pep517 --upgrade -r requirements_windows_torch1.txt
    pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
) else (
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    del ".\logs\status\torch_version" > nul 2>&1
    echo 2 > ".\logs\status\torch_version"
    pip install --use-pep517 --upgrade -r requirements_windows_torch2.txt
    pip install --upgrade xformers==0.0.20
    pip install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl
)

python.exe .\tools\update_bitsandbytes.py

accelerate config
