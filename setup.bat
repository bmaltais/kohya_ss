@echo off
setlocal enabledelayedexpansion

set ScriptDir=%~dp0

rem Set the valid possible values for Torch versions here:
set "validTorchVersions=0 1 2"

rem Define the default values
set Branch=master
set Dir=%ScriptDir%
set File=
set GitRepo=https://github.com/bmaltais/kohya_ss.git
set Headless=0
set Interactive=0
set LogDir=%ScriptDir%logs
set NoSetup=0
set Repair=0
set SetupOnly=0
set SkipSpaceCheck=0
set TorchVersion=0
set Update=0
set Verbosity=0
set LISTEN=127.0.0.1
set USERNAME=
set PASSWORD=
set SERVER_PORT=0
set INBROWSER=0
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
        if "%%a"=="Headless" set Headless=%%b
        if "%%a"=="Interactive" set Interactive=%%b
        if "%%a"=="LogDir" set LogDir=%%b
        if "%%a"=="NoSetup" set NoSetup=%%b
        if "%%a"=="Repair" set Repair=%%b
        if "%%a"=="SetupOnly" set SetupOnly=%%b
        if "%%a"=="SkipSpaceCheck" set SkipSpaceCheck=%%b
        if "%%a"=="TorchVersion" set TorchVersion=%%b
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
if "%~1"=="" (
    goto :arg_end
)

rem echo Parsing: %~1
set "arg=%~1"
if /i "%arg%"=="--branch" (
    if not "%~2"=="" (
        set Branch=%~2
        rem echo Branch set to !Branch!
        shift
    )
    shift
    goto arg_loop
)
if /i "%arg%"=="--dir" (
    if not "%~2"=="" (
        set "Dir=%~2"
        rem echo Dir set to !Dir!
        shift
    )
    shift
    goto arg_loop
)
if /i "%arg%"=="--file" (
    if not "%~2"=="" (
        set File=%~2
        rem echo File set to !File!
    shift
    )
    shift
    goto arg_loop
)
if /i "%arg%"=="--git-repo" (
    if not "%~2"=="" (
        set GitRepo=%~2
        rem echo GitRepo set to !GitRepo!
        shift
    )
    shift
    goto arg_loop
)
if /i "%arg%"=="--help" (
    goto print_help
)
if /i "%arg%"=="--headless" (
    set Headless=1
    rem echo Headless set to !Headless!
    shift
    goto arg_loop
)
if /i "%arg%"=="--interactive" (
    set Interactive=1
    rem echo Interactive set to !Interactive!
    shift
    goto arg_loop
)
if /i "%arg%"=="--log-dir" (
    if not "%~2"=="" (
        set LogDir=%~2
        rem echo LogDir set to !LogDir!
        shift
    ) 
    shift
    goto arg_loop
)
if /i "%arg%"=="--no-setup" (
    set NoSetup=1
    rem echo NoSetup set to !NoSetup!
    shift
    goto arg_loop
)
if /i "%arg%"=="--repair" (
    set Repair=1
    rem echo Repair set to !Repair!
    shift
    goto arg_loop
)
if /i "%arg%"=="--setup-only" (
    set SetupOnly=1
    rem echo SetupOnly set to !SetupOnly!
    shift
    goto arg_loop
)
if /i "%arg%"=="--skip-space-check" (
    set SkipSpaceCheck=1
    rem echo SkipSpaceCheck set to !SkipSpaceCheck!
    shift
    goto arg_loop
)
if /i "%arg%"=="--torch-version" (
    if not "%~2"=="" (
        set /a TorchVersion=%~2 2>nul
        if errorlevel 1 (
            echo Error: TorchVersion must be a number.
            exit /b 1
        ) else (
            echo Torch Version set to !TorchVersion!
        )
        set "validVersion=0"
        for %%v in (%validTorchVersions%) do (
            if !TorchVersion! == %%v (
                set "validVersion=1"
            )
        )
        if !validVersion! == 0 (
            echo Error: Invalid value for --torch-version: !TorchVersion!. Valid values are %validTorchVersions%.
            exit /b 1
        )
        shift
    )
    shift
    goto arg_loop
)
if /i "%arg%"=="--update" (
    set Update=1
    rem echo Update set to !Update!
    shift
    goto arg_loop
)
if /i "%arg%"=="--verbosity" (
    if not "%~2"=="" (
        set /a Verbosity=%~2 2>nul
        if errorlevel 1 (
            echo Error: Verbosity must be a number.
            exit /b 1
        ) else (
            echo Verbosity set to !Verbosity!
        )
        shift
    )
    shift
    goto arg_loop
)
if /i "%arg%"=="--listen" (
    if not "%~2"=="" (
        set LISTEN=%~2
        rem echo LISTEN set to !LISTEN!
        shift
    )
    shift
    goto arg_loop
)
if /i "%arg%"=="--username" (
    if not "%~2"=="" (
        set USERNAME=%~2
        rem echo USERNAME set to !USERNAME!
        shift
    )
    shift
    goto arg_loop
)
if /i "%arg%"=="--password" (
    if not "%~2"=="" (
        set PASSWORD=%~2
        rem echo PASSWORD set to !PASSWORD!
        shift
    )
    
    shift
    goto arg_loop
)
if /i "%arg%"=="--server-port" (
    if not "%~2"=="" (
        set SERVER_PORT=%~2
        rem echo SERVER_PORT set to !SERVER_PORT!
        shift
    )
    
    shift
    goto arg_loop
)
if /i "%arg%"=="--inbrowser" (
    set INBROWSER=1
    rem echo INBROWSER set to !INBROWSER!
    shift
    goto arg_loop
)
if /i "%arg%"=="--share" (
    set SHARE=1
    rem echo SHARE set to !SHARE!
    shift
    goto arg_loop
)

rem Unrecognized argument.
echo Error: Unrecognized argument "%~1"
echo.
call :print_help
exit /b 1

:arg_end

rem Bypass the print_help function and prepare arguments for
rem PowerShell and Python scripts and then run setup
goto :preparePowerShellArgs

:print_help
echo Usage: my_script.bat [OPTIONS]
echo.
echo OPTIONS:
echo --branch           : Specify the Git branch to use. Default is 'master'.
echo --dir              : Specify the working directory. Default is the directory of the script.
echo --file             : Specify the configuration file to be processed.
echo --git-repo         : Specify the Git repository URL. Default is 'https://github.com/bmaltais/kohya_ss.git'.
echo --headless         : Headless mode will not display the native windowing toolkit. Useful for remote deployments.
echo --help             : Display this help.
echo --interactive      : Run in interactive mode.
echo --log-dir          : Set the custom log directory for kohya_ss.
echo --no-setup         : Skip the setup process.
echo --repair           : This runs the installation repair operations. These could take a few minutes to run.
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

:preparePowerShellArgs
rem Function to prepare arguments for PowerShell script
set PSArgs=
if not "%Branch%"=="" set "PSArgs=%PSArgs% -Branch %Branch%"
if not "%Dir%"=="" set "PSArgs=%PSArgs% -Dir %Dir%"
if not "%File%"=="" set "PSArgs=%PSArgs% -File %File%"
if not "%GitRepo%"=="" set PSArgs=%PSArgs% -GitRepo %GitRepo%
if %Headless% EQU 1 set "PSArgs=%PSArgs% -Headless"
if %Interactive% EQU 1 set "PSArgs=%PSArgs% -Interactive"
if not "%LogDir%"=="" set "PSArgs=%PSArgs% -LogDir %LogDir%"
if %NoSetup% EQU 1 set "PSArgs=%PSArgs% -NoSetup"
if %Repair% EQU 1 set "PSArgs=%PSArgs% -Repair"
if %SetupOnly% EQU 1 set "PSArgs=%PSArgs% -SetupOnly"
if %SkipSpaceCheck% EQU 1 set "PSArgs=%PSArgs% -SkipSpaceCheck"
if not %TorchVersion% EQU 0 set "PSArgs=%PSArgs% -TorchVersion %TorchVersion%"
if not %Verbosity% EQU 0 set "PSArgs=%PSArgs% -Verbosity %Verbosity%"
if not "%LISTEN%"=="" set "PSArgs=%PSArgs% -Listen %LISTEN%"
if not "%USERNAME%"=="" set "PSArgs=%PSArgs% -Username %USERNAME%"
if not "%PASSWORD%"=="" set "PSArgs=%PSArgs% -Password %PASSWORD%"
if defined SERVER_PORT (
  if not %SERVER_PORT%==0 set "PSArgs=%PSArgs% -ServerPort %SERVER_PORT%"
)
if %INBROWSER% EQU 1 set "PSArgs=%PSArgs% -Inbrowser"
if %SHARE% EQU 1 set "PSArgs=%PSArgs% -Share"
goto :run_setup


:preparePythonArgs
rem Function to prepare arguments for Python script
set PythonArgs=
if not "%Branch%"=="" set "PythonArgs=%PythonArgs% --branch %Branch%"
if not "%Dir%"=="" set "PythonArgs=%PythonArgs% --dir %Dir%"
if not "%File%"=="" set "PythonArgs=%PythonArgs% --file %File%"
if not "%GitRepo%"=="" set PythonArgs=%PythonArgs% --git-repo %GitRepo%
if %Headless% EQU 1 set "PythonArgs=%PythonArgs% --headless"
if %Interactive% EQU 1 set "PythonArgs=%PythonArgs% -i"
if not "%LogDir%"=="" set "PythonArgs=%PythonArgs% --log-dir %LogDir%"
if %NoSetup% EQU 1 set "PythonArgs=%PythonArgs% -n"
if %Repair% EQU 1 set "PythonArgs=%PythonArgs% -r"
if %SetupOnly% EQU 1 set "PythonArgs=%PythonArgs% --setup-only"
if %SkipSpaceCheck% EQU 1 set "PythonArgs=%PythonArgs% --skip-space-check"
if not %TorchVersion% EQU 0 set "PythonArgs=%PythonArgs% --torch-version %TorchVersion%"
if not %Verbosity% EQU -1 set "PythonArgs=%PythonArgs% --verbosity %Verbosity%"
if not "%LISTEN%"=="" set "PythonArgs=%PythonArgs% --listen %LISTEN%"
if not "%USERNAME%"=="" set "PythonArgs=%PythonArgs% --username %USERNAME%"
if not "%PASSWORD%"=="" set "PythonArgs=%PythonArgs% --password %PASSWORD%"
if defined SERVER_PORT (
  if not %SERVER_PORT%==0 set "PythonArgs=%PythonArgs% --server-port %SERVER_PORT%"
)
if %INBROWSER% EQU 1 set "PythonArgs=%PythonArgs% --inbrowser"
if %SHARE% EQU 1 set "PythonArgs=%PythonArgs% --share"
goto :execute_python

:run_setup
rem Check if setup.ps1 exists
if not exist "%ScriptDir%setup.ps1" (
    echo "%ScriptDir%setup.ps1" not found. Falling back to Python script.
    call :run_python
    goto :eof
)

rem Initialize variables to track the success of the PowerShell and Python scripts
set "PowerShell_Success=0"
set "Python_Success=0"

rem Try to run the PowerShell script
set "PSScript=%ScriptDir%setup.ps1"
echo Launching %PSScript%.

powershell -ExecutionPolicy Bypass -NoLogo -NoProfile -File "%PSScript%" -BatchArgs "%PSArgs%"
if not errorlevel 1 (
    echo "PowerShell exited successfully."
    set "PowerShell_Success=1"
) else (
    echo PowerShell script failed or was blocked. Falling back to Python script.
    goto run_python
)

:check_results
if %PowerShell_Success% EQU 0 if %Python_Success% EQU 0 (
    echo "Both PowerShell and Python scripts failed."
) else (
    echo Python script failed. Please check your Python installation and try again.
)

goto :eof

:run_python
rem Check for valid Python executable
for %%a in (python.exe python310.exe python3.exe) do (
    set "py_exe=%%a"
    !py_exe! --version >nul 2>&1 && (
        echo Found !py_exe! executable. Trying to run the Python script.
        goto :preparePythonArgs
    )
)

:execute_python
rem Run the Python script
!py_exe! "%ScriptDir%launcher.py" %PythonArgs%
if not errorlevel 1 (
    echo Python script completed successfully.
    set "Python_Success=1"
)

goto check_results
