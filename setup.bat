@echo off
setlocal enabledelayedexpansion

rem Check for Python 3.10
for /f "tokens=* USEBACKQ" %%F in (`python --version 2^>^&1`) do (set "py_ver=%%F")
set "py_ver_req=Python 3.10"
if not "%py_ver%"=="%py_ver_req%" (
    echo Error: Python 3.10 is required, but '%py_ver%' is installed.
    echo Please download and install Python 3.10 from https://www.python.org/downloads/
    exit /b 1
)

rem Set default values for command line options
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

rem Parse command line arguments
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

rem we set an Args variable, so we can pass that to the launcher at the end and pass through values
set Args=-b "%Branch%" -d "%Dir%" -f "%ConfigFile%" -g "%GitRepo%" ^
          -i:%Interactive% -n:%NoGitUpdate% -p:%Public% -r:%Runpod% -s:%SkipSpaceCheck% -v %Verbose% ^
          --gui-listen "%GUI_LISTEN%" --gui-username "%GUI_USERNAME%" --gui-password "%GUI_PASSWORD%" ^
          --gui-server-port %GUI_SERVER_PORT% --gui-inbrowser:%GUI_INBROWSER% --gui-share:%GUI_SHARE%

rem Create venv if it doesn't exist
if not exist "%Dir%\venv" (
    python -m venv "%Dir%\venv"
) else (
    echo venv folder already exists, skipping creation...
)

copy /y "%Dir%\bitsandbytes_windows\*.dll" "%Dir%\venv\Lib\site-packages\bitsandbytes\"
copy /y "%Dir%\bitsandbytes_windows\cextension.py" "%Dir%\venv\Lib\site-packages\bitsandbytes\cextension.py"
copy /y "%Dir%\bitsandbytes_windows\main.py" "%Dir%\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py"

rem Call launcher.py with the provided arguments
python "%Dir%\launcher.py" %Args%

