@echo off

set PYTHON_VER=3.10.9

REM Check if Python version meets the recommended version
python --version 2>nul | findstr /b /c:"Python %PYTHON_VER%" >nul
if errorlevel 1 (
    echo Warning: Python version %PYTHON_VER% is recommended.
)

IF NOT EXIST venv (
    python -m venv venv
) ELSE (
    echo venv folder already exists, skipping creation...
)
call .\venv\Scripts\activate.bat

echo Do you want to uninstall previous versions of torch and associated files before installing?
echo [1] - Yes
echo [2] - No
set /p uninstall_choice="Enter your choice (1 or 2): "

if %uninstall_choice%==1 (
    pip uninstall -y xformers
    pip uninstall -y torch torchvision
)

echo Please choose the version of torch you want to install:
echo [1] - v1 (torch 1.12.1)
echo [2] - v2 (torch 2.0.0)
set /p choice="Enter your choice (1 or 2): "

if %choice%==1 (
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install --use-pep517 --upgrade -r requirements.txt
    pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
) else (
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    pip install --use-pep517 --upgrade -r requirements.txt
    pip install --upgrade  xformers==0.0.17
    rem pip install -U -I --no-deps https://files.pythonhosted.org/packages/d6/f7/02662286419a2652c899e2b3d1913c47723fc164b4ac06a85f769c291013/xformers-0.0.17rc482-cp310-cp310-win_amd64.whl
)

copy /y .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
copy /y .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
copy /y .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config
