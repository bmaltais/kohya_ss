@echo off
setlocal enabledelayedexpansion

:: Set default values for command line options
set Branch=master
set Dir=%~dp0
set GitRepo=https://github.com/kohya/kohya_ss.git
set Interactive=0
set NoGitUpdate=0
set Public=0
set Runpod=0
set SkipSpaceCheck=0
set Verbose=0
set GUI_LISTEN=127.0.0.1
set GUI_USERNAME=
set GUI_PASSWORD=
set GUI_SERVER_PORT=7861
set GUI_INBROWSER=1
set GUI_SHARE=0

:: Parse command line arguments
:arg_loop
if "%~1"=="" goto arg_end
if /i "%~1"=="--branch" (shift & set Branch=%1) & shift & goto arg_loop
if /i "%~1"=="--dir" (shift & set Dir=%1) & shift & goto arg_loop
if /i "%~1"=="--gitrepo" (shift & set GitRepo=%1) & shift & goto arg_loop
if /i "%~1"=="--interactive" (set Interactive=1) & shift & goto arg_loop
if /i "%~1"=="--nogitupdate" (set NoGitUpdate=1) & shift & goto arg_loop
if /i "%~1"=="--public" (set Public=1) & shift & goto arg_loop
if /i "%~1"=="--runpod" (set Runpod=1) & shift & goto arg_loop
if /i "%~1"=="--skipspacecheck" (set SkipSpaceCheck=1) & shift & goto arg_loop
if /i "%~1"=="--verbose" (set /A Verbose=Verbose+1) & shift & goto arg_loop
if /i "%~1"=="--gui-listen" (shift & set GUI_LISTEN=%1) & shift & goto arg_loop
if /i "%~1"=="--gui-username" (shift & set GUI_USERNAME=%1) & shift & goto arg_loop
if /i "%~1"=="--gui-password" (shift & set GUI_PASSWORD=%1) & shift & goto arg_loop
if /i "%~1"=="--gui-server-port" (shift & set GUI_SERVER_PORT=%1) & shift & goto arg_loop
if /i "%~1"=="--gui-inbrowser" (set GUI_INBROWSER=1) & shift & goto arg_loop
if /i "%~1"=="--gui-share" (set GUI_SHARE=1) & shift & goto arg_loop
shift
goto arg_loop

:arg_end

:: Create venv if it doesn't exist
if not exist "%Dir%\venv" (
    python -m venv "%Dir%\venv"
) else (
    echo venv folder already exists, skipping creation...
)

call "%Dir%\venv\Scripts\activate.bat"

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --use-pep517 --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

copy /y "%Dir%\bits

:: Copy required files
xcopy /y /q "%Dir%\bitsandbytes_windows\*.dll" "%Dir%\venv\Lib\site-packages\bitsandbytes\"
copy /y "%Dir%\bitsandbytes_windows\cextension.py" "%Dir%\venv\Lib\site-packages\bitsandbytes\cextension.py"
copy /y "%Dir%\bitsandbytes_windows\main.py" "%Dir%\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py"

:: Run accelerate configuration
accelerate config
