# Contributing

- [High Level Launcher.py Overview](#high-level-launcher.py-overview)
- [High Level Setup Scripts Overview](#high-level-setup-scripts-overview)
- [Arguments](#arguments)
  - [Loading Order](#loading-order)
  - [Adding a new argument to the setup workflow](#adding-a-new-command-line-argument-to-the-setup-workflow)
- [Logging](#logging)
  - [Logging Functions](#logging-functions)

<details>
<summary><h1 id="high-level-launcher.py-overview">High Level Launcher.py Overview</h1></summary>

- Ingest configuration files and command-line arguments:
  - Read input configuration files and parse command-line arguments to set up required settings and options


- Set up logging:
  - Define functions and classes for managing logs (write_to_log(), and CustomFormatter)
  - Create log directory based on the given arguments or default settings
  - Configure logging level based on the verbosity count argument
  - Set up logging handlers, including StreamHandler and FileHandler, with custom formatting


- Prepare environment files and folders:
  - Remove existing directories if necessary
  - Create required directories and copy files


* Update Kohya SS files in the target directory:
  - Define the function update_kohya_ss() that handles updating or cloning the Kohya SS repository
  - Check if Git is installed and the target directory exists
  - Perform Git operations with provided credentials (if any), including:
  - Pull updates from the remote repository if a local Git repository is detected and update flag is enabled
  - Clone the repository if no local Git repository is detected, preserving existing venv and logs folders if present
  - Handle errors such as authentication errors or uncommitted changes
  - If Git operations are not successful, proceed with the fallback method of downloading the repository as a zip file
    - Download the repository as a zip file if certain conditions are met:
    - If the update flag is enabled or the target directory is empty or contains only the venv and logs folders
    - If no specific branch is provided or the provided branch is not "master"
    - If no Git repository is provided or the provided repository is the default Kohya SS GitHub repository
  

- Install TensorFlow dependencies using Homebrew (macOS only):
  - Check if Homebrew is installed
  - Install or update necessary dependencies


- Install Python dependencies:
    - Installs TensorFlow dependencies:
      - For macOS ARM64, installs the tensorflow-macos and tensorflow-metal packages.
      - For macOS x86_64, installs the tensorflow package.
    - Installs or repairs Torch if the conditions are met.
    - Copies the requirements.txt to a temporary file for processing.
    - Merges the macOS-specific requirements from requirements_macos.txt into the temporary file if running on macOS.
    - Installs the Python packages listed in the temporary requirements file, using a progress bar.
    - Adds the appropriate scripts directory to the system PATH variable.
    - Removes the temporary requirements file once installation is complete.
    - Creates a flag file to indicate the completion of pip operations.
    - Handles exceptions that occur during pip operations and logs the error messages.
  

- Configure GPU acceleration:
  - Defines a helper function configure_accelerate_manually that calls the accelerate config command to manually configure the package.
  - Selects the source configuration file for accelerate based on the operating system and platform:
    - For macOS ARM64, uses the macos_config.yaml configuration file.
    - For all other platforms, uses the default_config.yaml configuration file.
    - If the script can't place a file, it will fall back to manual accelerate config.
  - If the configuration is not interactive, determines the target configuration file location based on the operating system and environment variables:
    - For Windows, checks the HF_HOME, LOCALAPPDATA, and USERPROFILE environment variables.
    - For other platforms, checks the HF_HOME, XDG_CACHE_HOME, and HOME environment variables.
  - If a target configuration location is found and the file doesn't already exist, creates the necessary directories and copies the source configuration file to the target location.
  - If a target configuration location is not found, informs the user and falls back to manual accelerate config.
  

- Launch kohya_gui.py:
  - Format the appropriate arguments to pass to kohya_gui.py
  - Execute the kohya_gui.py script to start the main web application

</details>

<details>
<summary><h1 id="high-level-setup-scripts-overview">High Level Setup Scripts Overview</h1></summary>

- Get OS information and format parameters
  - Check for illegal or malformed parameters
- Set up logging
- Define global paths for Python and Git executables
- Define software versions for Python
- For Windows:
    - Define software versions for Git and VC Redist
    - Set URLs for Python, Git, and VC Redist installers
    - Set download paths for Python, Git, and VC Redist installers
    - Get MD5 hash for Python installer from the website
    - Get SHA256 hash for Git installer from the website
- Begin the main function
    - Check if required arguments for minimal setup are provided
    - Install Python 3.10 if not already installed
    - Install Python 3.10 Tk if not already installed
    - Install Git if not already installed
    - Install VC Redist if not already installed (Windows only)
    - Update environment Path variable
    - Check for launcher.py in the target installation folder, if specified, or use the one in this script's directory
    - Format parameters for launcher.py's expectations
    - Call launcher.py with the appropriate parameters

</details>

## Arguments

<details>
<summary><h3 id="loading-order">Loading Order</h3></summary>

Logic Order:
1. Define the default list of arguments.
2. Load configuration data from the configuration file.
3. Update default argument values based on the configuration data.
4. Parse command line arguments and update the values accordingly.
5. Normalize paths to ensure they are absolute paths.
6. Replace any placeholder path values with the actual script directory.

Loading Order with the last item loaded taking priority:
1. Script defaults
2. Configuration file (if provided)
3. Command line arguments

Loading order of configuration files with the last file loaded taking priority:
1. install_config.yml located in the config_files/installation subdirectory within the script's directory
2. install_config.yml located in the .kohya_ss folder in the user's home directory
3. install_config.yml located directly within the script's directory
4. Configuration file specified by the -f option (overrides the previous configuration files if specified)

</details>

### Adding a New Argument to the Setup Workflow

<details>
<summary>1. Modify <code>install_config.yml</code></summary>

- Add the new argument under the appropriate section (`setup_arguments` or `gui_arguments`) with a name, description, and default value. For example:
    
   ```yaml
    newArgument:
      description: "Description of the new argument"
      default: "default_value"
   ```
</details>

<details>
<summary>2. Modify <code>setup.sh</code></summary>

- Add the argument to the display_help() function
```bash
Options:
  -b BRANCH, --branch=BRANCH    Select which branch of kohya to check out on new installs.
  ...
  -n, --new-argument            The long description goes here.
EOF
```
- Add a new case in the `getopts` loop to handle the new argument, including the short and long options.

```bash
# Be sure to add your short option letter to the beginning of the getopts section.
# If your new argument requires a value a : will go after your short options.
while getopts ":vb:d:f:g:il:nprst:ux-:" opt; do
  
# Then the new case
n | new-argument) CLI_ARGUMENTS["NewArgument"]="$OPTARG" ;;
   ```

- Add a line to set the default value if it's not in the config file.

```bash
config_NewArgument="${config_NewArgument:-default_value}"
   ```
  
- Add a line to override the config value with CLI arguments.

```bash
NEW_ARGUMENT="$config_NewArgument"
```
- Now add the argument to be passed into launcher.py.
```bash
"$PYTHON_EXEC" launcher.py \
    --branch="$BRANCH" \
    --dir="$DIR" \
    --git-repo="$GIT_REPO" \
    --new-argument="$NEW_ARGUMENT"
```
</details>

<details>
<summary>3. Modify <code>setup.ps1</code></summary>

- Add the documentation for the parameter to the script header
```
.PARAMETER NewArgument
    The long description goes here.
```
- Add a new parameter to the `param` block of the `Get-Parameters` function.
```powershell
    [string]$NewArgument = ""
   ```
- Add a new entry to the `$Defaults` hashtable for the new argument.
```powershell
    'NewArgument' = 'default_value'
   ```
</details>

<details>
<summary>4. Modify <code>launcher.py</code></summary>

- Add a new argument to the `argparse.ArgumentParser` instance.
```python
    parser.add_argument('--new-argument', default=None, help='Description of the new argument')
```

- You can access the information using the args array:
```python
# Of special note, multi-word arguments will be accessed in one of two styles:
# _args.torch_version or getattr(_args, "log-dir")

install_python_dependencies(_args.dir, _args.torch_version, _args.update, _args.repair,
                                        _args.interactive, getattr(_args, "log-dir"))
```
</details>

<details>
<summary>5. Modify <code>setup.bat</code></summary>

- Locate the section where default values for command-line options are set, and add a default value for the new option. For example, if the new option is `--new-argument`, add a line like `set NewOption=default_value`.

```batch
    set NewArgument=default_value
  ```

- Add it to the configuration file loader
```bash
if "%%a"=="NewArgument" set NewArgument=%%b
```

- Locate the section that parses command-line arguments (starting with `:arg_loop` and ending with `goto arg_loop`). Add a new conditional statement to handle the new option. 

```batch
if /i "%arg%"=="--new-argument" (
    set NewArgument=1
    rem echo NewArgument set to !NewArgument!
    shift
    goto arg_loop
)
```

- Finally, add the argument to the PowerShell and Python arguments

```batch
:preparePowerShellArgs
if %NewArgument% EQU 1 set "PSArgs=%PSArgs% -NewArgument"

:preparePythonArgs
if %NewArgument% EQU 1 set "PythonArgs=%PythonArgs% --new-argument"
   ```

</details>

After these steps, the new argument will be properly handled in the installation process in all files on all operating systems.

## Logging

All of the setup scripts, launcher, and kohya_gui have logging capabilities. Below are the equivalent function calls for each language.

All log files will default to the target installation directory /logs. The default target is $scriptDir/logs from wherever you are running the script from.
This can be overriden with the Log Dir (Directory) argument.

### Logging Functions

| Language  | Critical           | Error            | Info            | Warning         | Debug           |
| --------- | ------------------ | ---------------- | --------------- | --------------- | --------------- |
| PowerShell | Write-CriticalLog | Write-ErrorLog   | Write-InfoLog   | Write-WarnLog   | Write-DebugLog  |
| Bash      | log_critical       | log_error        | log_info        | log_warn        | log_debug       |
| Python    | logging.critical() | logging.error()  | logging.info()  | logging.warn()  | logging.debug() |

> **Special note**: Python also has a `write_to_log()` function that will bypass terminal input, but still allow logging strings. Very useful when you want to display a minimal interface, but still need a logging statement.
>
> **Formatting**: PowerShell and Bash log functions accept --color/-ForegroundColor [color] and --no-header/-NoHeader to set the text color and omit the LOG_LEVEL header on a given log string. 
>
> **Setup.bat**: Setup.bat does not have any equivalent logging functions.