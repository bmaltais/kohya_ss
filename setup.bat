@echo off
setlocal enabledelayedexpansion

rem Define the default values
set Branch=master
set Dir=%~dp0
set File=
set GitRepo=https://github.com/bmaltais/kohya_ss.git
set Interactive=0
set LogDir=
set NoSetup=0
set Public=0
set Repair=0
set Runpod=0
set SetupOnly=0
set SkipSpaceCheck=0
set Update=0
set Verbosity=0
set LISTEN=127.0.0.1
set USERNAME=
set PASSWORD=
set SERVER_PORT=0
set INBROWSER=1
set SHARE=0

rem Load the configuration file from the first existing location
set ConfigFile=
for %%F in (
    "%CD%\install_config.yml"
    "%USERPROFILE%\.kohya_ss\install_config.yml"
    "%~dp0\install_config.yml"
) do (
    if exist "%%F" (
        set ConfigFile=%%F
        goto load_config
    )
)

:load_config
if not "%ConfigFile%"=="" (
    for /f "usebackq tokens=1,2 delims=: " %%a in ("%ConfigFile%") do (
        if "%%a"=="Branch" set Branch=%%b
        if "%%a"=="Dir" set Dir=%%b
        if "%%a"=="File" set File=%%b
        if "%%a"=="GitRepo" set GitRepo=%%b
        if "%%a"=="Interactive" set Interactive=%%b
        if "%%a"=="LogDir" set LogDir=%%b
        if "%%a"=="NoSetup" set NoSetup=%%b
        if "%%a"=="Public" set Public=%%b
        if "%%a"=="Repair" set Runpod=%%b
        if "%%a"=="Runpod" set Runpod=%%b
        if "%%a"=="SetupOnly" set SetupOnly=%%b
        if "%%a"=="SkipSpaceCheck" set SkipSpaceCheck=%%b
        if "%%a"=="Update" set Update=%%b
        if "%%a"=="Verbosity" set Verbosity=%%b
        if "%%a"=="Listen" set LISTEN=%%b
        if "%%a"=="Username" set USERNAME=%%b
        if "%%a"=="Password" set PASSWORD=%%b
        if "%%a"=="ServerPort" set SERVER_PORT=%%b
        if "%%a"=="InBrowser" set INBROWSER=%%b
        if "%%a"=="Share" set SHARE=%%b
    )
)

rem Parse command line arguments and override loaded config file values
:arg_loop
if "%~1" equ "" goto arg_end

echo Parsing: %~1
if /i "%~1"=="--branch" (
    shift
    if not "%~1"=="" (
        set Branch=%1
        echo Branch set to %Branch%
    )
    goto shift_and_continue
)
if /i "%~1"=="--dir" (
    shift
    if not "%~1"=="" (
        set Dir=%1
        echo Dir set to %Dir%
    )
    goto shift_and_continue
)
if /i "%~1"=="--file" (
    shift
    if not "%~1"=="" (
        set File=%1
        echo File set to %File%
    )
    goto shift_and_continue
)
if /i "%~1"=="--git-repo" (
    shift
    if not "%~1"=="" (
        set GitRepo=%1
        echo GitRepo set to %GitRepo%
    )
    goto shift_and_continue
)
if /i "%~1"=="--help" (
    goto print_help
)
if /i "%~1"=="--interactive" (
    set Interactive=1
    echo Interactive set to %Interactive%
    goto shift_and_continue
)
if /i "%~1"=="--log-dir" (
    shift
    if not "%~1"=="" (
        set LogDir=%1
        echo LogDir set to %LogDir%
    )
    goto shift_and_continue
)
if /i "%~1"=="--no-setup" (
    set NoSetup=1
    echo NoSetup set to %NoSetup%
    goto shift_and_continue
)
if /i "%~1"=="--public" (
    set Public=1
    echo Public set to %Public%
    goto shift_and_continue
)
if /i "%~1"=="--repair" (
    set Repair=1
    echo Repair set to %Repair%
    goto shift_and_continue
)
if /i "%~1"=="--runpod" (
    set Runpod=1
    echo Runpod set to %Runpod%
    goto shift_and_continue
)
if /i "%~1"=="--setup-only" (
    set SetupOnly=1
    echo SetupOnly set to %SetupOnly%
    goto shift_and_continue
)
if /i "%~1"=="--skip-space-check" (
    set SkipSpaceCheck=1
    echo SkipSpaceCheck set to %SkipSpaceCheck%
    goto shift_and_continue
)
if /i "%~1"=="--update" (
    set Update=1
    echo Update set to %Update%
    goto shift_and_continue
)
if /i "%~1"=="--verbosity" (
    set /a Verbosity+=1
    echo Verbosity set to !Verbosity!
    goto shift_and_continue
)
if /i "%~1"=="--listen" (
    shift
    if not "%~1"=="" (
        set LISTEN=%1
        echo LISTEN set to %LISTEN%
    )
    goto shift_and_continue
)
if /i "%~1"=="--username" (
    shift
    if not "%~1"=="" (
        set USERNAME=%1
        echo USERNAME set to %USERNAME%
    )
    goto shift_and_continue
)
if /i "%~1"=="--password" (
    shift
    if not "%~1"=="" (
        set PASSWORD=%1
        echo PASSWORD set to %PASSWORD%
    )
    goto shift_and_continue
)
if /i "%~1"=="--server-port" (
    shift
    if not "%~1"=="" (
        set SERVER_PORT=%1
        echo SERVER_PORT set to %SERVER_PORT%
    )
    goto shift_and_continue
)
if /i "%~1"=="--inbrowser" (
    set INBROWSER=1
    echo INBROWSER set to %INBROWSER%
    goto shift_and_continue
)
if /i "%~1"=="--share" (
    set SHARE=1
    echo SHARE set to %SHARE%
    goto shift_and_continue
)

:: Unrecognized argument.
echo Error: Unrecognized argument "%~1"
echo.
call :print_help
exit /b 1

:shift_and_continue
shift
goto arg_loop

:arg_end



rem Bypass the print_help function and skip to executing launcher.py
goto :run_command

:print_help
echo Usage: my_script.bat [OPTIONS]
echo.
echo OPTIONS:
echo --branch           : Specify the Git branch to use. Default is 'master'.
echo --dir              : Specify the working directory. Default is the directory of the script.
echo --file             : Specify the configuration file to be processed.
echo --git-repo         : Specify the Git repository URL. Default is 'https://github.com/bmaltais/kohya_ss.git'.
echo --help             : Display this help.
echo --interactive      : Run in interactive mode.
echo --log-dir          : Set the custom log directory for kohya_ss.
echo --no-setup         : Skip the setup process.
echo --public           : Run in public mode.
echo --repair           : This runs the installation repair operations. These could take a few minutes to run.
echo --runpod           : Run in Runpod mode.
echo --setup-only       : Only run the setup process, do not launch the application.
echo --skip-space-check : Skip the disk space check.
echo --update           : Run the update process.
echo --verbosity        : Increase the verbosity level.
echo --listen           : Specify the GUI listen address. Default is '127.0.0.1'.
echo --username         : Specify the GUI username.
echo --password         : Specify the GUI password.
echo --server-port      : Specify the GUI server port. Default is 7861.
echo --inbrowser        : Open the GUI in the browser.
echo --share            : Enable GUI sharing.
goto :eof

:run_command
rem we set an Args variable, so we can pass that to the launcher at the end and pass through values
:: Prepare Args
set Args=
if not "%Branch%"=="" set Args=%Args% -b "%Branch%"
if not "%Dir%"=="" set Args=%Args% -d "%Dir%"
if not "%File%"=="" set Args=%Args% -f "%File%"
if not "%GitRepo%"=="" set Args=%Args% -g "%GitRepo%"
if %Interactive% EQU 1 set Args=%Args% -i
if %NoSetup% EQU 1 set Args=%Args% -n
if %Public% EQU 1 set Args=%Args% -p
if %Repair% EQU 1 set Args=%Args% --repair
if %Runpod% EQU 1 set Args=%Args% --runpod
if %SkipSpaceCheck% EQU 1 set Args=%Args% -s
if not "%Verbosity%"=="" set Args=%Args% -v %Verbosity%
if %SetupOnly% EQU 1 set Args=%Args% --setup-only
if not "%LISTEN%"=="" set Args=%Args% --listen "%LISTEN%"
if not "%USERNAME%"=="" set Args=%Args% --username "%USERNAME%"
if not "%PASSWORD%"=="" set Args=%Args% --password "%PASSWORD%"
if not "%SERVER_PORT%"=="" set Args=%Args% --server-port %SERVER_PORT%
if %INBROWSER% EQU 1 set Args=%Args% --inbrowser
if %SHARE% EQU 1 set Args=%Args% --share
if not "%LogDir%"=="" set Args=%Args% --log-dir:%LogDir%

rem Call launcher.py with the provided arguments
:: Execute launcher.py with the provided arguments
echo python "%Dir%launcher.py" %Args%
python "%Dir%launcher.py" %Args% || (
  echo.
  echo Python script encountered an error.
  echo Press Enter to continue...
  pause >nul
)

:end