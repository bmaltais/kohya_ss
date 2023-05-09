<#
.SYNOPSIS
    Kohya_SS Installation Script for Windows PowerShell and PowerShell 7+.

.DESCRIPTION
    This script automates the installation of Kohya_SS on Windows, macOS, Ubuntu, and RedHat Linux systems. This is the
    install bootstrap file ensuring Python 3.10 and Python 3.10 TK is installed and available.

.EXAMPLE
    # Specifies custom branch, install directory, and git repo
    .\setup.ps1 -Branch dev -Dir C:\workspace\kohya_ss -GitRepo https://mycustom.repo.tld/custom_fork.git

.EXAMPLE
    # Running setup in Debug mode while skipping the available space check
    .\setup.ps1 -Verbosity 3 -SkipSpaceCheck

.PARAMETER Branch
    Select which branch of kohya to check out on new installs.

.PARAMETER Dir
    The full path you want kohya_ss installed to.

.PARAMETER NoSetup
    Skip all setup steps and only validate python requirements then launch GUI.

.PARAMETER File
    The full path to a custom configuration file.

.PARAMETER GitRepo
    You can optionally provide a git repo to check out. Useful for custom forks.

.PARAMETER Headless
    Headless mode will not display the native windowing toolkit. Useful for remote deployments.

.PARAMETER Interactive
    Interactively configure accelerate instead of using default config file.

.PARAMETER LogDir
    Specifies the directory where log files will be stored.

.PARAMETER NoGitUpdate
    Do not update kohya_ss repo. No git pull or clone operations.

.PARAMETER Repair
    This runs the installation repair operations. These could take a few minutes to run.

.PARAMETER SetupOnly
    Do not launch GUI. Only conduct setup operations.

.PARAMETER SkipSpaceCheck
    Skip the 10Gb minimum storage space check.

.PARAMETER TorchVersion
    Configure the major version of Torch.

.PARAMETER Update
    Update kohya_ss with specified branch, repo, or latest kohya_ss if git's unavailable.

.PARAMETER Verbosity
    Increase verbosity levels up to 3.

.PARAMETER Listen
    The IP address the GUI should listen on.

.PARAMETER Username
    The username for the GUI.

.PARAMETER Password
    The password for the GUI.

.PARAMETER ServerPort
    The port number the GUI server should use.

.PARAMETER InBrowser
    Open the GUI in the default web browser.

.PARAMETER Share
    Share the GUI with other users on the network.
#>

[CmdletBinding()]
param (
    [string]$File,
    [string]$Branch,
    [string]$Dir,
    [string]$GitRepo,
    [switch]$Headless,
    [switch]$Interactive,
    [string]$LogDir,
    [switch]$NoSetup,
    [switch]$Repair,
    [switch]$SetupOnly,
    [switch]$SkipSpaceCheck,
    [int]$TorchVersion,
    [int]$Verbosity,
    [switch]$Update,
    [string]$Listen,
    [string]$Username,
    [string]$Password,
    [int]$ServerPort,
    [switch]$Inbrowser,
    [switch]$Share,
    [Parameter(Mandatory = $false)]
    [string]$BatchArgs = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$unboundArgs
)

<#
.SYNOPSIS
    Get-Parameters is a PowerShell function that retrieves and processes input parameters from various sources.

.DESCRIPTION
    Get-Parameters retrieves parameters from configuration files, passed-in parameters, and batch arguments.
    It processes and prioritizes these sources, handling any inconsistencies and ensuring that the final
    configuration is accurate and up-to-date. It also converts relative paths to absolute paths and
    ensures that required modules are imported.

.PARAMETER BoundParameters
    A hashtable containing the parameters passed to the script.

.PARAMETER BatchArgs
    A string containing batch arguments. This parameter is used when the arguments are received from a batch
    file as a single array. Get-BatchArgs is a helper function to parse the batch arguments and return a
    hashtable.

.EXAMPLE
    $Parameters = Get-Parameters -BoundParameters $PSBoundParameters -BatchArgs $BatchArgs
    This example demonstrates how to call Get-Parameters with the necessary parameters.

    To get specific values you would call:
    $Parameters.ValueName

    To get the option passed in as -Dir:
    $Parameters.Dir

.NOTES
    Get-Parameters relies on several helper functions to handle specific tasks, such as Get-BatchArgs
    to parse batch arguments, and Convert-RelativePathsToAbsolute to convert relative paths to absolute paths.
#>
function Get-Parameters {
    param (
        [Parameter(Mandatory = $true)]
        [hashtable]$BoundParameters,
        [Parameter(Mandatory = $false)]
        [string]$BatchArgs = ""
    )

    # Helper function to parse batch arguments and return a hashtable
    function Get-BatchArgs {
        param (
            [Parameter(Mandatory = $true)]
            [string]$Args
        )

        $Result = @{}
        $splitArgs = $Args -split '\s+'

        for ($i = 0; $i -lt $splitArgs.Length; $i++) {
            if ($splitArgs[$i] -match '^-(\w+)') {
                $key = $Matches[1]
                $value = $null

                if ($i + 1 -lt $splitArgs.Length -and -not ($splitArgs[$i + 1] -match '^-(\w+)')) {
                    $value = $splitArgs[$i + 1]
                    $i++
                }
                else {
                    $value = $true
                }

                $Result[$key] = $value
            }
        }

        return $Result
    }

    # Helper function to convert relative paths to absolute
    function Convert-RelativePathsToAbsolute {
        param (
            [Parameter(Mandatory = $true)]
            [hashtable]$Params
        )

        $Result = $Params.Clone()

        foreach ($key in $Params.Keys) {
            $value = $Params[$key]
            if (-not [string]::IsNullOrEmpty($value)) {
                # Check if value doesn't start with a known scheme and contains a path separator
                if (($value -notmatch '^[a-zA-Z][a-zA-Z0-9+.-]*://') -and ($value -match '[/\\]')) {
                    # Expand tilde to the user's home directory if present
                    if ($value.StartsWith('~')) {
                        $value = $value.Replace('~', $HOME)
                    }

                    # Convert relative paths to absolute and normalize
                    if ($value -ne [System.IO.Path]::GetFullPath($value)) {
                        try {
                            $resolvedPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $value))
                        }
                        catch [System.ArgumentException] {
                            Write-Host "Sorry, the path contains illegal characters:`n'$value'`n " -ForegroundColor Red
                            exit 1
                        }
                        catch [System.IO.PathTooLongException] {
                            Write-Host "Sorry, the path is too long:`n'$value'`n" -ForegroundColor Red
                            exit 1
                        }
                        catch {
                            Write-Host "Sorry, there was an error processing the path:`n'$value'." -ForegroundColor Red
                            exit 1
                        }
                        # Add debug output before setting $Result.$key
                        Write-Debug "Absolute Path: $resolvedPath"
                        $Result.$key = $resolvedPath
                    }
                    else {
                        $Result.$key = $value
                    }
                }
            }
        }

        return $Result
    }

    # Check for the existence of the powershell-yaml module and install it if necessary
    try {
        if (-not (Get-Module -ListAvailable -Name 'powershell-yaml')) {
            Install-Module -Name 'powershell-yaml' -Scope CurrentUser -Force
        }
        Import-Module 'powershell-yaml'
    }
    catch {
        Write-Host "Failed to import module powershell-yaml, exiting to avoid corrupted configuration values."
        exit 1
    }

    # Define possible configuration file locations
    $configFileLocations = if ($os.family -eq "Windows") {
        @(
            (Join-Path -Path "$PSScriptRoot\config_files\installation" -ChildPath "install_config.yml")
            (Join-Path -Path "$env:USERPROFILE\.kohya_ss" -ChildPath "install_config.yml"),
            (Join-Path -Path $PSScriptRoot -ChildPath "install_config.yml"),
            $File
        )
    }
    else {
        @(
            (Join-Path -Path "$PSScriptRoot/config_files/installation" -ChildPath "install_config.yml")
            (Join-Path -Path $env:HOME -ChildPath ".kohya_ss/install_config.yml"),
            (Join-Path -Path $PSScriptRoot -ChildPath "install_config.yml"),
            $File
        )
    }

    # Define the default values
    $Defaults = @{
        'Branch'         = 'master'
        'Dir'            = "$PSScriptRoot"
        'GitRepo'        = 'https://github.com/bmaltais/kohya_ss.git'
        'Headless'       = $false
        'Interactive'    = $true
        'LogDir'         = "$PSScriptRoot/logs"
        'NoSetup'        = $false
        'Repair'         = $false
        'SetupOnly'      = $false
        'SkipSpaceCheck' = $false
        'TorchVersion'   = 0
        'Verbosity'      = 0
        'Update'         = $false
        'Listen'         = '127.0.0.1'
        'Username'       = ''
        'Password'       = ''
        'ServerPort'     = 0
        'Inbrowser'      = $false
        'Share'          = $false
    }

    # Initialize the $Config hashtable with the default values
    $Config = $Defaults.Clone()

    # Load configuration files
    foreach ($location in $configFileLocations) {
        Write-Debug "Config file location: $location"
        if (![string]::IsNullOrEmpty($location)) {
            if (Test-Path $location) {
                Write-Debug "Found configuration file at: ${location}"
                $FileConfig = (Get-Content $location | Out-String | ConvertFrom-Yaml)
                foreach ($section in $FileConfig.Keys) {
                    foreach ($item in $FileConfig[$section]) {
                        $lowerKey = $item.name.ToLower()
                        if ($Config.ContainsKey($lowerKey)) {
                            # Check if the value from the config file is an empty string
                            if ($item.value -eq '') {
                                # If so, continue to the next iteration of the loop without changing $Config
                                continue
                            }
                            # Only assign the value from the config file if the corresponding $Config value is the default value
                            if ($Config[$lowerKey] -eq $Defaults[$lowerKey]) {
                                $Config[$lowerKey] = $item.value
                            }
                        }
                    }
                }
            }
        }        
    }

    # Override config with the $Parameters values
    foreach ($key in $Parameters.Keys) {
        $lowerKey = $key.ToLower()
        if ($Config.ContainsKey($lowerKey) -and ($Parameters[$key] -ne "")) {
            $Config[$lowerKey] = $Parameters[$key]
        }
    }

    foreach ($key in $Parameters.kohya_gui_arguments.Keys) {
        $lowerKey = $key.ToLower()
        if ($Config.ContainsKey($lowerKey)) {
            if ($null -ne $Parameters.kohya_gui_arguments[$key] -and $Parameters.kohya_gui_arguments[$key] -ne "") {
                $Config[$lowerKey] = $Parameters.kohya_gui_arguments[$key]
            }
        }
    }

    foreach ($key in $Parameters.setup_arguments.Keys) {
        $lowerKey = $key.ToLower()
        if ($Config.ContainsKey($lowerKey)) {
            if ($null -ne $Parameters.setup_arguments[$key] -and $Parameters.setup_arguments[$key] -ne "") {
                $Config[$lowerKey] = $Parameters.setup_arguments[$key]
            }
        }
    }

    # Uncomment the debug lines below to check parameter values before processing CLI values.
    # Write-Debug "PSBoundParameters:"
    # foreach ($key in $PSBoundParameters.Keys) {
    #     Write-Debug "${key}: $($PSBoundParameters[$key])"
    # }

    # If batch arguments are provided, parse and update BoundParameters
    if (![string]::IsNullOrEmpty($BatchArgs)) {
        $ParsedBatchArgs = Get-BatchArgs -Args $BatchArgs
        $BoundParameters = $BoundParameters + $ParsedBatchArgs
        if ($BoundParameters.ContainsKey("BatchArgs")) {
            $BoundParameters.Remove('BatchArgs')
        }
    }

    # Override config with command-line arguments last
    foreach ($key in $BoundParameters.Keys) {
        $lowerKey = $key.ToLower()
        if ($Config.ContainsKey($lowerKey)) {
            $Config[$lowerKey] = $BoundParameters[$key]
        }
        else {
            if ($key -ne "Debug") {
                Write-Debug "Key '$key' not found in Config"
            }
        }
    }

    # Uncomment the debug lines below to check parameter values after processing CLI values.
    # Write-Debug "Config after PSBoundParameters override:"
    # foreach ($key in $Config.Keys) {
    #     Write-Debug "${key}: $($Config.$key)"
    # }

    # If the default string is detected. change it to current script directory
    if ($Config["Dir"] -eq "_CURRENT_SCRIPT_DIR_") {
        $Config["Dir"] = "$PSScriptRoot"
    }

    # If LogDir is not specified, use Dir/logs
    if ([string]::IsNullOrEmpty($Config["LogDir"])) {
        $Config["LogDir"] = Join-Path $Config["Dir"] "logs"
    }

    $Config = Convert-RelativePathsToAbsolute -Params $Config
    Write-Debug "LogDir: ${Config.LogDir}"

    # Debug output for the final configuration
    Write-Debug "Config: $($Config.LogDir)"
    foreach ($key in $Config.Keys) {
        Write-Debug "${key}: $($Config.$key)"
    }

    return $Config
}

<#
.SYNOPSIS
    Sets up the logging environment.

.DESCRIPTION
    The Set-Logging function prepares the logging environment.
    It sets the current date, time, log level name, log file name, and creates the log directory if it doesn't exist.

.EXAMPLE
    Set-Logging
#>
function Set-Logging {
    param (
        [Parameter(Mandatory = $true)]
        [string]$LogDir
    )

    # Get current date and time
    $currentDate = Get-Date -Format "yyyy-MM-dd"
    $currentTime = Get-Date -Format "HHmm"

    # Set log level name based on verbosity
    $script:logLevelName = switch ($script:Parameters.Verbosity) {
        0 { "error" }
        1 { "warning" }
        2 { "info" }
        3 { "debug" }
        default { "unknown" }
    }
    
    if ($script:Parameters.Verbosity -eq 0) {
        $script:logFilename = "launcher_$($currentTime).log"
    }
    else {
        $script:logFilename = "launcher_$($currentTime)_$($script:logLevelName).log"
    }

    # Create log directory if it doesn't exist
    $LogDir = "$LogDir/$currentDate"
    if (!(Test-Path -Path $LogDir)) {
        New-Item -ItemType Directory -Force -Path $LogDir > $null -ErrorAction 'SilentlyContinue'
    }

    # Define log file path
    $script:logFile = "$LogDir/$script:logFilename"
}

<#
.SYNOPSIS
    Logs a debug message.

.DESCRIPTION
    The Write-DebugLog function logs a debug message to the console and the log file if the verbosity level is 3 or higher.

.PARAMETER message
    The debug message to log.

.EXAMPLE
    Write-DebugLog "This is a debug message."
#>
function Write-DebugLog {
    param (
        [string]$message,
        [System.ConsoleColor]$ForegroundColor = 'DarkGreen'
    )
    
    if ($script:Parameters.Verbosity -ge 3) {
        Write-Host "DEBUG: $message" -ForegroundColor $ForegroundColor
        Add-Content -Path $script:logFile -Value "DEBUG: $message"
    }
}

<#
.SYNOPSIS
    Logs an informational message.

.DESCRIPTION
    The Write-InfoLog function logs an informational message to the console and the log file if the verbosity level is 2 or higher.

.PARAMETER message
    The informational message to log.

.EXAMPLE
    Write-InfoLog "This is an informational message."
#>
function Write-InfoLog {
    param (
        [string]$message,
        [System.ConsoleColor]$ForegroundColor = 'Blue'
    )
    
    if ($script:Parameters.Verbosity -ge 2) {
        Write-Host "INFO: $message" -ForegroundColor $ForegroundColor
        Add-Content -Path $script:logFile -Value "INFO: $message"
    }
}

<#
.SYNOPSIS
Logs a warning message.

.DESCRIPTION
The Write-WarningLog function logs a warning message to the console and the log file if the verbosity level is 1 or higher.

.PARAMETER message
The warning message to log.

.EXAMPLE
Write-WarningLog "This is a warning message."
#>
function Write-WarningLog {
    param (
        [string]$message,
        [System.ConsoleColor]$ForegroundColor = 'Yellow'
    )

    if ($script:Parameters.Verbosity -ge 1) {
        Write-Host "WARN: $message" -ForegroundColor $ForegroundColor
        Add-Content -Path $script:logFile -Value "WARN: $message"
    }
}

<#
.SYNOPSIS
    Logs an error message.

.DESCRIPTION
    The Write-ErrorLog function logs an error message to the console and the log file if the verbosity level is 0 or higher.

.PARAMETER message
    The error message to log.

.EXAMPLE
    Write-ErrorLog "This is an error message."
#>
function Write-ErrorLog {
    param (
        [string]$message,
        [System.ConsoleColor]$ForegroundColor = 'Red'
    )
    
    if ($script:Parameters.Verbosity -ge 0) {
        Write-Host "ERROR: $message" -ForegroundColor $ForegroundColor
        Add-Content -Path $script:logFile -Value "ERROR: $message"
    }
}

<#
.SYNOPSIS
    Logs a critical error message.

.DESCRIPTION
    The Write-CriticalLog function logs a critical error message to the console and the log file regardless of the verbosity level.

.PARAMETER message
    The critical error message to log.

.EXAMPLE
    Write-CriticalLog "This is a critical error message."
#>
function Write-CriticalLog {
    param (
        [Parameter(Mandatory = $false, Position = 0)]
        [string]$message,
        [Parameter(Mandatory = $false)]
        [System.ConsoleColor]$ForegroundColor = 'White',
        [Parameter(Mandatory = $false)]
        [switch]$NoHeader
    )

    if (-not $NoHeader) {
        $message = "CRITICAL: $message"
    }
    
    if ($script:Parameters.Verbosity -ge 0) {
        Write-Host "$message" -ForegroundColor $ForegroundColor
        Add-Content -Path $script:logFile -Value "$message"
    }
}

<#
.SYNOPSIS
    Tests if the specified keys exist in a given hashtable.

.DESCRIPTION
    This function takes a hashtable and an array of required keys as input. It returns true if all the required keys exist in the hashtable, and false otherwise.

.PARAMETER Params
    A hashtable containing key-value pairs.

.PARAMETER RequiredKeys
    An array of strings representing the required keys.

.OUTPUTS
    Boolean
        Returns true if all the required keys exist in the hashtable, and false otherwise.

.EXAMPLE
    $params = @{
        'key1' = 'value1'
        'key2' = 'value2'
        'key3' = 'value3'
    }
    $requiredKeys = @('key1', 'key2', 'key3')
    $result = Test-Value -Params $params -RequiredKeys $requiredKeys
    if ($result) {
        Write-CriticalLog "All required keys are present."
    } else {
        Write-CriticalLog "Some required keys are missing."
    }
#>
function Test-Value {
    param (
        [hashtable]$Params,
        [string[]]$RequiredKeys
    )

    foreach ($key in $RequiredKeys) {
        if ($null -eq $Params.$key) {
            return $false
        }
    }
    return $true
}

<#
.SYNOPSIS
   Checks if the current user has administrator privileges.

.DESCRIPTION
   This function tests if the current user has administrator privileges by attempting to create a new event log source.
   If successful, it will remove the event log source and return $true, otherwise, it returns $false.

.OUTPUTS
   System.Boolean
   Outputs $true if the user has administrator privileges, $false otherwise.
.NOTES
   This function should work on Windows, Linux, macOS, and BSD systems.
#>
function Test-IsAdmin {
    if ($os.family -eq "Windows") {
        # Windows-specific code
        $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object System.Security.Principal.WindowsPrincipal($identity)
        $isAdmin = $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
        return $isAdmin
    }
    else {
        # POSIX-specific code
        $isBSD = (uname -s) -match 'BSD'
        if ($isBSD) {
            # BSD-specific id output format
            $userId = (id -u -r)
        }
        else {
            # Default POSIX id output format
            $userId = (id -u)
        }
        $result = ($userId -eq 0)
        return $result
    }
}


<#
.SYNOPSIS
Returns the appropriate command to elevate a PowerShell command to administrator/root privileges.

.DESCRIPTION
The Get-ElevationCommand function returns a command that can be used to elevate a PowerShell command to administrator/root privileges, depending on the current operating system and user permissions. On Windows, the function returns an empty string if the current user already has administrator privileges, or a command that uses the Start-Process cmdlet to restart the current script with administrator privileges if not. On POSIX systems, the function returns an empty string if the current user is already root, has sudo privileges, or has su privileges, or a command that uses either sudo or su to run the command as root if not.

.PARAMETER None
This function takes no parameters.

.EXAMPLE
$elevate = Get-ElevationCommand
& $elevate apt install -y python3.10-tk
Runs the "apt install -y python3.10-tk" command with elevated privileges, using the appropriate method for the current operating system.

.NOTES
This function should work on Windows, Linux, macOS, and BSD systems.
#>
function Get-ElevationCommand {
    param (
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$args
    )

    if ($os.family -eq "Windows") {
        # On Windows, use the Start-Process cmdlet to run the command as administrator
        if ((Test-IsAdmin) -eq $true) {
            return ""
        }
        else {
            return "Start-Process powershell.exe -Verb RunAs -ArgumentList '-Command',$(&{[scriptblock]::Create($args[0])} '$args[1..$($args.count)]')"
        }
    }
    else {
        # On POSIX systems, check if we're running as root already
        if ($EUID -eq 0) {
            return ""
        }
        else {
            # Check if we have admin privileges
            if ((Test-IsAdmin) -eq $true) {
                return ""
            }
            else {
                # Check if sudo is installed
                if (Get-Command sudo -ErrorAction SilentlyContinue) {
                    # Use sudo to run the command as root
                    return "sudo -S $($args[0]) $($args[1..$($args.count)] -join ' ')"
                }
                else {
                    # Fall back to su to run the command as root
                    return "su -c '$($args -join ' ')'"
                }
            }
        }
    }
}

<#
.SYNOPSIS
Prompts the user to choose between installing a package for the current user only or for all users on the system.

.DESCRIPTION
The Choose-InstallScope function prompts the user to choose between installing a package for the current user only or for all users on the system. If the $Interactive parameter is set to $false, the function will assume that the package should be installed for all users.

.PARAMETER Interactive
If this parameter is set to $true, the function will prompt the user to choose between a local installation (for the current user only) or a global installation (for all users). If this parameter is set to $false, the function will assume that the package should be installed globally.

.OUTPUTS
System.String
Returns the string "user" if the package should be installed for the current user only, or "allusers" if the package should be installed for all users.

.EXAMPLE
PS C:\> $installScope = Choose-InstallScope -Interactive $true
Choose installation option: (1) Local (2) Global
1
PS C:\> $installScope
user
Prompts the user to choose between a local or global installation, and returns the string "user" if the user chooses a local installation or "allusers" if the user chooses a global installation.

.NOTES
This function does not check for administrator/root privileges. The calling code should ensure that the function is called with appropriate permissions.
#>
function Update-InstallScope {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $false)]
        [bool]$Interactive = $false
    )
    
    if ($Interactive) {
        do {
            $installOption = Read-Host -Prompt "Choose installation option: (1) Local (2) Global"
        } while ($installOption -ne "1" -and $installOption -ne "2")
    }
    else {
        $installOption = 2
    }

    if ($installOption -eq 1) {
        $installScope = "user"
    }
    else {
        $installScope = "allusers"
    }

    return $installScope
}



<#
.SYNOPSIS
    Retrieves information about the current operating system.

.DESCRIPTION
    This function returns an object containing the operating system's name, family, and version. It supports Linux, macOS, and Windows systems.

.OUTPUTS
    PSCustomObject
        An object containing the name, family, and version of the operating system.

.EXAMPLE
    $os = Get-OsInfo
    Write-CriticalLog "Operating System: $($os.name)"
    Write-CriticalLog "OS Family: $($os.family)"
    Write-CriticalLog "OS Version: $($os.version)"
#>
function Get-OsInfo {
    $os = @{
        name    = "Unknown"
        family  = "Unknown"
        version = "Unknown"
    }

    if ([System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT) {
        $os.name = "Windows"
        $os.family = "Windows"
        $os.version = [System.Environment]::OSVersion.Version.ToString()
    }
    elseif (Test-Path "/System/Library/CoreServices/SystemVersion.plist") {
        $os.name = "macOS"
        $os.family = "macOS"
        $os.version = "Unknown"
        try {
            $os.version = (Get-Content -Raw -Path "/System/Library/CoreServices/SystemVersion.plist" -ErrorAction Stop | Select-String -Pattern "<string>([\d\.]+)</string>" | ForEach-Object { $_.Matches.Groups[1].Value }) -join ""
        }
        catch {
            Write-Warning "Error reading /System/Library/CoreServices/SystemVersion.plist: $_"
        }
    }
    elseif (Test-Path "/etc/os-release") {
        if (Test-Path "/etc/os-release") {
            try {
                $os_release = Get-Content "/etc/os-release" -Raw -ErrorAction Stop
                $os.name = if ($os_release -match 'ID="?([^"\n]+)') { $matches[1] } else { "Unknown" }
                $os.family = if ($os_release -match 'ID_LIKE="?([^"\n]+)') { $matches[1] } else { "Unknown" }
                $os.version = if ($os_release -match 'VERSION="?([^"\n]+)') { $matches[1] } else { "Unknown" }
            }
            catch {
                Write-Warning "Error reading /etc/os-release: $_"
            }
        }
        elseif (Test-Path "/etc/redhat-release") {
            try {
                $redhat_release = Get-Content "/etc/redhat-release"
                if ($redhat_release -match '([^ ]+) release ([^ ]+)') {
                    $os.name = $matches[1]
                    $os.family = "RedHat"
                    $os.version = $matches[2]
                }
            }
            catch {
                Write-Warning "Error reading /etc/redhat-release: $_"
            }
        }

        if ($os.name -eq "Unknown") {
            try {
                $uname = uname -a
                if ($uname -match "Ubuntu") { $os.name = "Ubuntu"; $os.family = "Ubuntu" }
                elseif ($uname -match "Debian") { $os.name = "Debian"; $os.family = "Debian" }
                elseif ($uname -match "Red Hat" -or $uname -match "CentOS") { $os.name = "RedHat"; $os.family = "RedHat" }
                elseif ($uname -match "Fedora") { $os.name = "Fedora"; $os.family = "Fedora" }
                elseif ($uname -match "SUSE") { $os.name = "openSUSE"; $os.family = "SUSE" }
                elseif ($uname -match "Arch") { $os.name = "Arch"; $os.family = "Arch" }
                else { $os.name = "Generic Linux"; $os.family = "Generic Linux" }
            }
            catch {
                Write-Warning "Error executing uname command: $_"
                $os.name = "Generic Linux"
                $os.family = "Generic Linux"
            }
        }
    }
    return [PSCustomObject]$os
}


<#
.SYNOPSIS
   Checks if Python 3.10 is installed on the system.

.DESCRIPTION
   Verifies if Python 3.10 is installed by checking its version.
   Returns a boolean value based on the presence of Python 3.10.

.EXAMPLE
   $isPython310Installed = Test-Python310Installed

.OUTPUTS
   System.Boolean
   True if Python 3.10 is installed, otherwise False.
#>
function Test-Python310Installed {
    try {
        if ($null -eq $script:pythonPath) {
            Write-DebugLog "Python executable not found."
            return $false
        }

        Write-DebugLog "We are testing this python path: {$script:pythonPath}"
        $script:pythonVersion = & $script:pythonPath --version 2>&1 | Out-String -Stream -ErrorAction Stop
        $script:pythonVersion = $script:pythonVersion -replace '^Python\s', ''

        if ($script:pythonVersion.StartsWith('3.10')) {
            return $true
        }
        else {
            Write-Error "Python version at $script:pythonPath is not 3.10, it's $script:pythonVersion."
            return $false
        }
    }
    catch {
        switch ($_.Exception.GetType().Name) {
            'Win32Exception' {
                Write-Error "Python executable found at $script:pythonPath , but it could not be run. It may be corrupted or there may be a permission issue."
                return $false
            }
            'RuntimeException' {
                if ($_.Exception.Message -like '*The term*is not recognized as the name of a cmdlet*') {
                    Write-DebugLog "Python executable not found at $script:pythonPath ."
                    return $false
                }
                else {
                    Write-Error "An unknown error occurred when trying to run Python at ${pythonPath}: $($_.Exception.Message)"
                    return $false
                }
            }
            default {
                Write-DebugLog "An unknown error occurred when trying to check Python version at ${pythonPath}: $($_.Exception.Message)"
                return $false
            }
        }
    }
}


<#
.SYNOPSIS
   Retrieves the path to the Python 3.10 executable.

.DESCRIPTION
   Searches for Python 3.10 executable in the system and returns its path.
   It handles different platforms and common edge cases such as Homebrew on macOS and FreeBSD.

.EXAMPLE
   $script:pythonPath = Get-PythonExePath

.OUTPUTS
   System.String
   The path to the Python 3.10 executable or $null if not found.
#>
function Get-PythonExePath {
    $pythonCandidates = @("python3.10", "python3", "python")

    if ($os.family -eq "Windows") {
        $pythonCandidates += @("python3.10.exe", "python3.exe", "python.exe")
    }

    $foundPythonPath = $null

    foreach ($candidate in $pythonCandidates) {
        try {
            $script:pythonPath = (Get-Command $candidate -ErrorAction SilentlyContinue).Source
            if ($null -ne $script:pythonPath) {
                $script:pythonVersion = & $script:pythonPath --version 2>&1
                if ($script:pythonVersion -match "^Python 3\.10") {
                    $foundPythonPath = $script:pythonPath
                    break
                }
            }
        }
        catch {
            continue
        }
    }

    if ($null -eq $foundPythonPath) {
        # macOS with Homebrew
        if ($os.family -eq "Darwin") {
            $brewPythonPath = "/usr/local/opt/python@3.10/bin/python3.10"
            if (Test-Path $brewPythonPath) {
                $foundPythonPath = $brewPythonPath
            }
        }

        # FreeBSD
        if ($os.family -eq "FreeBSD") {
            $freebsdPythonPath = "/usr/local/bin/python3.10"
            if (Test-Path $freebsdPythonPath) {
                $foundPythonPath = $freebsdPythonPath
            }
        }
    }

    return $foundPythonPath
}


<#
.SYNOPSIS
Retrieves the MD5 hash of Python 3.10 Windows installer (64-bit) from the official Python download page.

.DESCRIPTION
This function sends a web request to the Python 3.10 release page and extracts the MD5 hash of the Windows installer (64-bit). The MD5 hash is used to verify the integrity of the downloaded file.

.PARAMETER pythonReleasePageUrl
Specifies the URL of the Python 3.10 release page. The default is derived from the pythonInstallerUrl script variable.

.PARAMETER pythonInstallerUrl
Specifies the URL of the Python 3.10 Windows installer (64-bit). The default is the pythonInstallerUrl script variable.

.EXAMPLE
$pythonHash = Get-Python310Md5HashFromWeb

Returns the MD5 hash of the Python 3.10 Windows installer (64-bit).

.INPUTS
None. You cannot pipe input to this function.

.OUTPUTS
System.String. This function returns the MD5 hash of the Python 3.10 Windows installer (64-bit).

.NOTES
If the function fails to obtain the MD5 hash, it writes an error message and exits with a non-zero status.
#>
function Get-Python310Md5HashFromWeb {
    $webContent = Invoke-WebRequest -Uri $script:pythonReleasePageUrl
    $regexPattern = '<td><a href="' + [regex]::Escape($script:pythonInstallerUrl) + '">Windows installer \(64-bit\)</a></td>\s*<td>Windows</td>\s*<td>Recommended</td>\s*<td>([a-f0-9]{32})</td>'
    if ($webContent -match $regexPattern) {
        return $matches[1]
    }
    else {
        Write-Error "Failed to obtain MD5 hash for Python 3.10 from the download page."
        exit 1
    }
}

<#
.SYNOPSIS
   Installs Python 3.10 using the specified installer.

.DESCRIPTION
   This function downloads and installs Python 3.10 using the specified installer URL.
   It also adds Python and pip to the system PATH variable.

.PARAMETER InstallerURL
   The URL of the Python 3.10 installer.

.OUTPUTS
   None
#>
function Install-Python310 {
    param (
        [switch]$Interactive
    )

    function Install-Python310Windows {
        $packageManagerFound = $false

        if (Get-Command "scoop" -ErrorAction SilentlyContinue) {
            # This should have worked, but it looks like scoop might not have Python 3.10 available,
            # so we're just going to use the native method for now.
            #scoop install python@3.10
            #$packageManagerFound = $true
            $packageManagerFound = $false
        }
        elseif (Get-Command "choco" -ErrorAction SilentlyContinue) {
            choco install python --version=3.10
            $packageManagerFound = $true
        }
        elseif (Get-Command "winget" -ErrorAction SilentlyContinue) {
            winget install --id Python.Python --version 3.10.*
            $packageManagerFound = $true
        }

        if (-not $packageManagerFound) {
            if (-not (Test-Path $script:pythonInstallerPath)) {
                try {
                    Invoke-WebRequest -Uri $script:pythonInstallerUrl -OutFile $script:pythonInstallerPath
                }
                catch {
                    Write-Error "Failed to download Python 3.10. Please check your internet connection or provide a pre-downloaded installer."
                    exit 1
                }
            }

            # Compute the MD5 hash of the downloaded file
            $downloadedPythonMd5 = Get-FileHash -Algorithm MD5 -Path $script:pythonInstallerPath | ForEach-Object Hash

            # Check if the computed MD5 hash matches the expected MD5 hash
            if ($downloadedPythonMd5 -ne $script:pythonMd5) {
                Write-Error "MD5 hash mismatch for Python 3.10. The downloaded file may be corrupt or tampered with."
                exit 1
            }
            if (Test-IsAdmin) {
                $installScope = Update-InstallScope($Interactive)

                if ($installScope -eq "user") {
                    $proc = Start-Process $script:pythonInstallerPath -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0" -Wait -PassThru
                }
                else {
                    $proc = Start-Process $script:pythonInstallerPath -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" -Wait -PassThru
                }
                
                $proc.WaitForExit()
                
                if (Test-Path $script:pythonInstallerPath ) { 
                    Remove-Item $script:pythonInstallerPath 
                }
                
                
            }
            else {
                Write-DebugLog "Path: ${script:pythonInstallerPath}"
                # We default to installing at a user level if admin is not detected.
                $proc = Start-Process $script:pythonInstallerPath -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0" -Wait -PassThru
                $proc.WaitForExit()
                if (Test-Path $script:pythonInstallerPath ) { 
                    Remove-Item $script:pythonInstallerPath 
                }
            }
        }
    }
    
    function Install-Python310Mac {
        if (Get-Command "brew" -ErrorAction SilentlyContinue) {
            brew install python@3.10
            brew link --overwrite --force python@3.10
        }
        else {
            Write-CriticalLog "Please install Homebrew first to continue with Python 3.10 installation."
            Write-CriticalLog "You can find that here: https://brew.sh"
            exit 1
        }
    }

    function Install-Python310Linux {
        $elevate = ""
        if (-not ($env:USER -eq "root" -or (id -u) -eq 0 -or $env:EUID -eq 0)) {
            $elevate = "sudo"
        }

        switch ($os.family) {
            "Ubuntu" {
                if (& $elevate apt-get update) {
                    if (!(& $elevate apt-get install -y python3.10)) {
                        Write-Error "Error: Failed to install python via apt. Installation of Python 3.10 aborted."
                    }
                }
                else {
                    Write-Error "Error: Failed to update package list. Installation of Python 3.10 aborted."
                }
            }
            "Debian" {
                if (& $elevate apt-get update) {
                    if (!(& $elevate apt-get install -y python3.10)) {
                        Write-Error "Error: Failed to install python via apt. Installation of Python 3.10 aborted."
                    }
                }
                else {
                    Write-Error "Error: Failed to update package list. Installation of Python 3.10 aborted."
                }
            }
            "RedHat" {
                if (!(& $elevate dnf install -y python3.10)) {
                    Write-Error "Error: Failed to install python via dnf. Installation of Python 3.10 aborted."
                }
            }
            "Arch" {
                if (!(& $elevate pacman -Sy --noconfirm python3.10)) {
                    Write-Error "Error: Failed to install python via pacman. Installation of Python 3.10 aborted."
                }
            }
            "openSUSE" {
                if (!(& $elevate zypper install -y python3.10)) {
                    Write-Error "Error: Failed to install python via zypper. Installation of Python 3.10 aborted."
                }
            }
            default {
                Write-Error "Unsupported Linux distribution. Please install Python 3.10 manually."
                exit 1
            }
        }
    }

    if (Test-Python310Installed) {
        Write-CriticalLog "Python 3.10 is already installed." -NoHeader
        return
    }

    $osPlatform = ""
    if ($PSVersionTable.Platform -eq 'Windows' -or $PSVersionTable.PSEdition -eq 'Desktop') {
        $osPlatform = (Get-WmiObject -Class Win32_OperatingSystem).Caption
    }
    elseif ($PSVersionTable.Platform -eq 'Unix') {
        $osPlatform = (uname -s).ToString()
    }
    elseif ($PSVersionTable.Platform -eq 'MacOS') {
        $osPlatform = (uname -s).ToString()
    }
    else {
        Write-Error "Unsupported operating system. Please install Python 3.10 manually."
        exit 1
    }


    if ($osPlatform -like "*Windows*") {
        Install-Python310Windows
    }
    elseif ($osPlatform -like "*Mac*") {
        Install-Python310Mac
    }
    elseif ($osPlatform -like "*Linux*") {
        Install-Python310Linux
    }
    else {
        Write-Error "Unsupported operating system. Please install Python 3.10 manually."
        exit 1
    }

    # We are updating the environment and python path after installation to ensure it is picked up by the environment every time.
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    $script:pythonPath = Get-PythonExePath

    if (Test-Python310Installed) {
        Write-CriticalLog "Python 3.10 installed successfully." -NoHeader
    }
    else {
        Write-Error 'Failed to install. Please ensure Python 3.10 is installed and available in $PATH.'
        exit 1
    }
}

<#
.SYNOPSIS
   Installs the Python 3 Tk package using pip.

.DESCRIPTION
   This function installs the Tk package for Python 3 using pip.
   It ensures that pip is up to date before installing the package.

.OUTPUTS
   None
#>
function Install-Python3Tk {
    param (
        [ValidateSet('allusers', 'user')]
        [string]$installScope = 'user'
    )

    $osFamily = $os.family.ToLower()

    if ($PSVersionTable.Platform -eq 'Unix') {
        # Linux / macOS installation
    
        # Pre-check: Try to import Tkinter in Python 3.10
        $isTkinterInstalled = $false
        try {
            $tkinterCheckOutput = & $script:pythonPath -c "import tkinter" 2>&1
            if (-not $tkinterCheckOutput) {
                $isTkinterInstalled = $true
            }
        }
        catch {
            Write-CriticalLog "Tkinter not found. Attempting to install."
        }
    
        # Only try to install Tkinter if it is not already installed
        if (-not $isTkinterInstalled) {
            if ($osFamily -match "ubuntu") {
                # Ubuntu installation
                try {
                    Invoke-Expression (Get-ElevationCommand "apt" "update")
                    Invoke-Expression (Get-ElevationCommand "apt" "install" "-y" "python3.10-tk")
                }
                catch {
                    Write-Error "Error: Failed to install Python 3.10 Tk on Ubuntu. $_"
                }
            }
            elseif ($osFamily -match "debian") {
                # Debian installation
                try {
                    Invoke-Expression (Get-ElevationCommand "apt-get" "update")
                    Invoke-Expression (Get-ElevationCommand "apt-get" "install" "-y" "python3.10-tk")
                }
                catch {
                    Write-Error "Error: Failed to install Python 3.10 Tk on Debian. $_"
                }
            }
            elseif ($osFamily -match "redhat") {
                # Red Hat installation
                try {
                    Invoke-Expression (Get-ElevationCommand "dnf" "install" "-y" "python3.10-tkinter")
                }
                catch {
                    Write-Error "Error: Failed to install Python 3.10 Tk on Red Hat. $_"
                }
            }
            elseif ($osFamily -match "arch") {
                # Arch installation
                try {
                    Invoke-Expression (Get-ElevationCommand "pacman" "-S" "--noconfirm" "tk")
                }
                catch {
                    Write-Error "Error: Failed to install Python 3.10 Tk on Arch. $_"
                }
            }
            elseif ($osFamily -match "opensuse") {
                # openSUSE installation
                try {
                    Invoke-Expression (Get-ElevationCommand "zypper" "install" "-y" "python3.10-tk")
                }
                catch {
                    Write-Error "Error: Failed to install Python 3.10 Tk on openSUSE. $_"
                }
            }
            elseif ($osFamily -match "macos") {
                if (Test-Path "/usr/local/bin/brew") {
                    try {
                        # macOS installation using Homebrew
                        Invoke-Expression "brew install python-tk@3.10"
                    }
                    catch {
                        Write-Error "Error: Failed to install Python 3.10 Tk on macOS using Homebrew. $_"
                    }
                }
                else {
                    Write-Error "Unsupported Unix platform or package manager not found."
                }
            }
            else {
                Write-Error "Unsupported Linux distribution. Please install Python 3.10 Tk manually."
            }
        }
        else {
            Write-CriticalLog "Tkinter for Python 3.10 is already installed on this system." -NoHeader
        }
    }
    else { 
        # Windows installation
        if (! (Test-Python310Installed)) {
            Invoke-WebRequest -Uri $script:pythonInstallerUrl -OutFile $script:pythonInstallerPath

            $installScope = Update-InstallScope($Interactive)

            if ($installScope -eq 'allusers') {
                if (Test-IsAdmin) {
                    try {
                        $proc = Start-Process -FilePath $script:pythonInstallerPath -ArgumentList "/passive InstallAllUsers=1 PrependPath=1 Include_tcltk=1" -PassThru
                        $proc.WaitForExit()
                    }
                    catch {
                        Write-Error "Error: Failed to install Python 3.10 Tk for all users on Windows. $_"
                    }
                }
                else {
                    if (Test-IsAdmin) {
                        Write-Warning "Warning: Running as administrator, but 'user' scope is selected. Proceeding with user-scope installation."
                    }
                    try {
                        $proc = Start-Process -FilePath $script:pythonInstallerPath -ArgumentList "/passive InstallAllUsers=0 PrependPath=1 Include_tcltk=1" -PassThru
                        $proc.WaitForExit()
                    }
                    catch {
                        Write-Error "Error: Failed to install Python 3.10 Tk for current user on Windows. $_"
                    }
                }

                if (Test-Path $script:pythonInstallerPath) {
                    Remove-Item $script:pythonInstallerPath -ErrorAction SilentlyContinue
                }
            }
        }
        else {
            Write-CriticalLog "Tkinter for Python 3.10 is already installed on this system." -NoHeader
        }

        # We are updating the environment and python path after installation to ensure it is picked up by the environment every time.
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        $script:pythonPath = Get-PythonExePath      
    }
}

<#
.SYNOPSIS
    Checks whether Git is installed and accessible from the command line.

.DESCRIPTION
    Test-GitInstalled checks whether Git is installed on the system by first attempting to call
    `git --version` from the command line. If this call fails, the function then checks a list of
    common locations where Git might be installed. If Git is found, the function returns $true and
    displays the Git version. If Git is not found or an error occurs, the function returns $false and
    displays a warning message.

.EXAMPLE
    if (Test-GitInstalled) {
        Write-CriticalLog "Git is installed."
    }
    else {
        Write-CriticalLog "Git is not installed."
    }

    This example checks whether Git is installed on the system. If Git is installed, it displays a
    message indicating so. Otherwise, it displays a message stating that Git is not installed.
#>
function Test-GitInstalled {
    # Define common git install locations
    $commonGitLocations = @(
        "/usr/local/git/bin",
        "/usr/bin",
        "/usr/local/bin",
        "/usr/sbin",
        "/sbin",
        "/bin",
        "/usr/local",
        "/opt/X11/bin",
        "/opt/local/bin",
        "C:\Program Files\Git\bin",
        "C:\Program Files (x86)\Git\bin"
    )

    try {
        # Check if git command is available in PATH
        if (Get-Command git -ErrorAction SilentlyContinue) {
            git --version
            return $true
        }
        else {
            # If git not found in PATH, check common install locations
            foreach ($location in $commonGitLocations) {
                if (Test-Path "$location/git") {
                    & "$location/git" --version
                    Write-CriticalLog "Git already installed." -NoHeader
                    return $true
                }
            }
        }
    }
    catch {
        Write-Warning "Failed to execute git --version. Error: $_"
    }

    Write-Warning "Git not found."
    return $false
}

<#
.SYNOPSIS
   Fetches the SHA-256 hash of the specified Git for Windows installer from the GitHub releases page.

.DESCRIPTION
   The Get-GitHashFromWeb function uses the script-level gitVersion variable to fetch the HTML content 
   of the Git for Windows release page for that version. It then uses a regex pattern to find and return 
   the SHA-256 hash of the Git installer file for the specified version.

.PARAMETER gitVersion
   The version number of Git for Windows whose SHA-256 hash should be fetched.
   This is a script-level variable and should be set before calling this function.

.OUTPUTS
   String. Returns the SHA-256 hash of the Git installer file for the specified version.

.EXAMPLE
   $script:gitVersion = "2.40.1"
   $script:gitSha256 = Get-GitHashFromWeb
   This example fetches the SHA-256 hash of the Git for Windows v2.40.1 installer file.

.NOTES
   This function requires the Invoke-WebRequest cmdlet, which may not be available on all systems.
#>
function Get-GitHashFromWeb {
    # Derive the release page URL from the git version
    $gitReleasePageUrl = "https://github.com/git-for-windows/git/releases/tag/v${script:gitVersion}.windows.1"

    # Fetch the HTML content of the release page
    $releasesPage = Invoke-WebRequest -Uri $gitReleasePageUrl
    $html = $releasesPage.Content

    # Define the regex pattern
    $filename = "Git-${script:gitVersion}-64-bit.exe"
    $pattern = "(?<=<td>${filename}</td>\r\n<td>)[a-fA-F0-9]{64}(?=</td>)"

    # Apply the regex pattern to the HTML content
    $hash = [regex]::Match($html, $pattern).Value

    return $hash
}

<#
.SYNOPSIS
   Finds the path of the Git executable in the system.

.DESCRIPTION
   The Get-GitExePath function attempts to locate the Git executable in the system. It first tries to find
   it in the system's registry (for Windows) or using package managers like Homebrew (for macOS) or 
   apt-get, pacman and dnf (for Linux and FreeBSD). If these methods fail, it falls back to default hard-coded paths.

.OUTPUTS
   String. Returns the path of the Git executable if found.

.EXAMPLE
   $gitExePath = Get-GitExePath
   This example retrieves the path of the Git executable in the system.

.NOTES
   This function relies on multiple system-specific commands and may not work on all systems.
   It also attempts to modify the $env:Path environment variable if the Git executable is found.
#>
function Get-GitExePath {
    if ($os.family -eq "Windows") {
        # Try to get the Git path from the registry if Git is installed natively on Windows
        $registryPath = "HKLM:\SOFTWARE\GitForWindows"
        $gitPath = Get-ItemPropertyValue -Path $registryPath -Name InstallPath -ErrorAction SilentlyContinue

        # Try to find Git installation path using Winget
        if (-not $gitPath -and (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
            $wingetResults = winget search git
            $gitResult = $wingetResults | Where-Object { $_.Name -match 'git for windows' } | Select-Object -First 1
            if ($gitResult) {
                $gitPath = $gitResult.InstalledLocation
            }
        }

        # Try to find Git installation path using Scoop
        if (-not $gitPath -and (Get-Command scoop.exe -ErrorAction SilentlyContinue)) {
            $scoopResults = scoop search git
            $gitResult = $scoopResults | Where-Object { $_.Name -match 'git' } | Select-Object -First 1
            if ($gitResult) {
                $gitPath = Join-Path -Path $env:USERPROFILE -ChildPath ('scoop\apps\' + $gitResult.Name + '\' + $gitResult.Version + '\bin')
            }
        }

        # Try to find Git installation path using Chocolatey
        if (-not $gitPath -and (Get-Command choco.exe -ErrorAction SilentlyContinue)) {
            $chocoResults = choco search git
            $gitResult = $chocoResults | Where-Object { $_.Name -match 'git' } | Select-Object -First 1
            if ($gitResult) {
                $gitPath = Join-Path -Path $env:ProgramFiles -ChildPath ('Git\bin')
            }
        }

        # If Git is not installed using any known package manager, fall back to the hard-coded paths
        if (-not $gitPath) {
            $gitPath = Join-Path -Path $env:ProgramFiles -ChildPath 'Git\bin'
            if (-not (Test-Path $gitPath)) {
                $gitPath = Join-Path -Path ${env:ProgramFiles(x86)} -ChildPath 'Git\bin'
            }
        }
    }
    elseif ($os.family -eq "Darwin") {
        # Try to find Git installation path using Homebrew
        if (Get-Command brew -ErrorAction SilentlyContinue) {
            $brewResults = brew list --cask --versions git
            if ($brewResults) {
                $brewResult = $brewResults | Select-Object -First 1
                $gitPath = Join-Path -Path '/usr/local/Caskroom/git' -ChildPath $brewResult
            }
        }

        # If Git is not installed using Homebrew, fall back to the default path
        if (-not $gitPath) {
            $gitPath = '/usr/bin'
        }
    }
    elseif ($os.family -eq "Linux" -or $os.family -eq "FreeBSD") {
        # Try to find Git installation path using the package manager
        if (Get-Command apt-get -ErrorAction SilentlyContinue) {
            $dpkgResults = dpkg -l git
            if ($dpkgResults -match '^ii') {
                $gitPath = Join-Path -Path "/usr/bin" -ChildPath "git"
            }
        }
        elseif (Get-Command pacman -ErrorAction SilentlyContinue) {
            $pacmanResults = pacman -Ql git | Where-Object { $_ -match '/bin/git$' }
            if ($pacmanResults) {
                $gitPath = Split-Path -Parent $pacmanResults
            }
        }
        elseif (Get-Command dnf -ErrorAction SilentlyContinue) {
            $rpmResults = rpm -qa git
            if ($rpmResults) {
                $gitPath = Join-Path -Path "/usr/bin" -ChildPath "git"
            }
        }
        
        # If Git is not installed using the package manager, fall back to the hard-coded path
        if (-not $gitPath) {
            $gitPath = "/usr/bin/git"
        }
        
        # Test if Git executable exists in the resolved path
        if (-not (Test-Path $gitPath)) {
            Write-Error "Git executable not found at path '$gitPath'. Installation may be incomplete or corrupted."
        }
        else {
            $env:Path = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ":" + [System.Environment]::GetEnvironmentVariable("PATH", "User") + ":$gitPath"
            $script:gitPath = $gitPath
        }
        
        if (Test-Path $gitPath) {
            return $gitPath
        }
        
        Write-Error "Could not find Git executable."
    }
}        

<#
.SYNOPSIS
   Installs Git using the specified installer.

.DESCRIPTION
   This function downloads and installs Git using the specified installer URL.
   It also adds Git to the system PATH variable.

.PARAMETER InstallerURL
   The URL of the Git installer.

.OUTPUTS
   None
#>
function Install-Git {
    param (
        [switch]$Interactive
    )    

    function Install-GitWindows {
        $packageManagerFound = $false

        if (Get-Command "scoop" -ErrorAction SilentlyContinue) {
            try {
                scoop install git
                $packageManagerFound = $true
            }
            catch {
                Write-Error "Error: Failed to install Git using Scoop. $_"
            }
        }
    
        if (-not $packageManagerFound -and (Get-Command "choco" -ErrorAction SilentlyContinue)) {
            try {
                choco install git
                $packageManagerFound = $true
            }
            catch {
                Write-Error "Error: Failed to install Git using Chocolatey. $_"
            }
        }
    
        if (-not $packageManagerFound -and (Get-Command "winget" -ErrorAction SilentlyContinue)) {
            try {
                winget install --id Git.Git
                $packageManagerFound = $true
            }
            catch {
                Write-Error "Error: Failed to install Git using Winget. $_"
            }
            if ($packageManagerFound) {
                # Update the environment and Git path after installation to ensure it is picked up by the environment every time
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                $script:gitPath = Get-GitExePath
            }
        }
        
        if (-not $packageManagerFound) {
            if (Test-IsAdmin) {
                $installScope = Update-InstallScope($Interactive)

                if (-not (Test-Path $script:gitInstallerPath)) {
                    try {
                        Invoke-WebRequest -Uri $script:gitUrl -OutFile $script:gitInstallerPath
                    }
                    catch {
                        Write-Error "Failed to download Git. Please check your internet connection or provide a pre-downloaded installer."
                        exit 1
                    }
                }

                # Compute the SHA-256 hash of the downloaded file
                $downloadedGitSha256 = Get-FileHash -Algorithm SHA256 -Path $script:gitInstallerPath | ForEach-Object Hash

                # Check if the computed SHA-256 hash matches the expected SHA-256 hash
                if ($downloadedGitSha256 -ne $script:gitSha256) {
                    Write-Error "SHA-256 hash mismatch for git. The downloaded file may be corrupt or tampered with."
                    exit 1
                }

                if ($installScope -eq "user") {
                    $proc = Start-Process -FilePath $script:gitInstallerPath -ArgumentList "/VERYSILENT", "/NORESTART", "/NOCANCEL", "/SP-", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS", "/COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -PassThru
                }
                else {
                    $proc = Start-Process -FilePath $script:gitInstallerPath -ArgumentList "/VERYSILENT", "/NORESTART", "/NOCANCEL", "/SP-", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS", "/COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -PassThru
                }
                $proc.WaitForExit()
                if (Test-Path $script:gitInstallerPath) {
                    Remove-Item $script:gitInstallerPath
                }
                # Update the environment and Git path after installation to ensure it is picked up by the environment every time
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                $script:gitPath = Get-GitExePath
            }
            else {
                # We default to installing at a user level if admin is not detected.
                $proc = Start-Process -FilePath $script:gitInstallerPath -ArgumentList "/VERYSILENT", "/NORESTART", "/NOCANCEL", "/SP-", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS", "/COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -PassThru
                $proc.WaitForExit()
                if (Test-Path $script:gitInstallerPath) {
                    Remove-Item $script:gitInstallerPath
                }
            }
        }
    }

    function Install-GitMac {
        if (Get-Command "brew" -ErrorAction SilentlyContinue) {
            try {
                Invoke-Expression "brew install git"
            }
            catch {
                Write-Error "Error: Failed to install Git on macOS using Homebrew. $_"
            }
        }
        else {
            Write-Error "Please install Homebrew first to continue with Git installation. You can find that here: https://brew.sh"
            exit 1
        }
    }

    function Install-GitLinux {
        $osFamily = Get-OsFamily
        if ($osFamily -match "debian" -or $osFamily -match "ubuntu") {
            try {
                Invoke-Expression (Get-ElevationCommand "apt-get" "install" "-y" "git")
            }
            catch {
                Write-Error "Error: Failed to install Git on $osFamily. $_"
            }
        }
        elseif ($osFamily -match "redhat") {
            try {
                Invoke-Expression (Get-ElevationCommand "dnf" "install" "-y" "git")
            }
            catch {
                Write-Error "Error: Failed to install Git on RedHat. $_"
            }
        }
        elseif ($osFamily -match "arch") {
            try {
                Invoke-Expression (Get-ElevationCommand "pacman" "-Sy" "--noconfirm" "git")
            }
            catch {
                Write-Error "Error: Failed to install Git on Arch Linux. $_"
            }
        }
        elseif ($osFamily -match "opensuse") {
            try {
                Invoke-Expression (Get-ElevationCommand "zypper" "install" "-y" "git")
            }
            catch {
                Write-Error "Error: Failed to install Git on openSUSE. $_"
            }
        }
        elseif ($osFamily -match "freebsd") {
            try {
                Invoke-Expression (Get-ElevationCommand "pkg" "install" "-y" "git")
            }
            catch {
                Write-Error "Error: Failed to install Git on FreeBSD. $_"
            }
        }
        else {
            Write-Error "Unsupported Linux/Unix distribution. Please install Git manually."
            exit 1
        }
    }

    $osPlatform = ""
    if ($PSVersionTable.Platform -eq 'Windows' -or $PSVersionTable.PSEdition -eq 'Desktop') {
        $osPlatform = (Get-WmiObject -Class Win32_OperatingSystem).Caption
    }
    elseif ($PSVersionTable.Platform -eq 'Unix') {
        $osPlatform = (uname -s).ToString()
    }
    elseif ($PSVersionTable.Platform -eq 'MacOS') {
        $osPlatform = (uname -s).ToString()
    }
    else {
        Write-Error "Unsupported operating system. Please install Git manually."
        exit 1
    }

    if ($osPlatform -like "*Windows*") {
        Install-GitWindows
    }
    elseif ($osPlatform -like "*Mac*") {
        Install-GitMac
    }
    elseif ($osPlatform -like "*Linux*" -or $osPlatform -like "*BSD*") {
        Install-GitLinux
    }
    else {
        Write-Error "Unsupported operating system. Please install Git manually."
        exit 1
    }

    if (Test-GitInstalled) {
        Write-CriticalLog "Git installed successfully." -NoHeader
    }
    else {
        Write-ErrorLog 'Failed to install. Please ensure Git is installed and available in $PATH.'
        exit 1
    }
}

<#
.SYNOPSIS
    Displays a countdown timer and allows users to skip the countdown or cancel the installation.

.DESCRIPTION
    This function displays a countdown timer, decrementing from the given countdown value.
    It also provides users the option to skip the countdown and continue or cancel the installation.
    The user can press 'y' to skip the countdown or 'n' to cancel the installation.

.PARAMETER countdown
    An integer representing the number of seconds for the countdown.

.EXAMPLE
    $continueInstallation = DisplayCountdown -countdown 15
    Displays a 15-second countdown and allows the user to either skip or cancel the installation.

.NOTES
    The countdown is displayed with carriage return (`r) to update the same line.
#>
function DisplayCountdown($countdown) {
    Write-Host "Press 'y' to skip the countdown and continue or 'n' to cancel the installation."
    $continue = $false

    for ($i = 0; $i -lt $countdown; $i++) {
        $remainingTime = $countdown - $i
        Write-Host "`rContinuing in $remainingTime... " -NoNewline

        $keyInfo = $null
        $timeout = 1000

        while (($timeout -gt 0) -and (-not $keyInfo)) {
            $timeout -= 100
            if ([console]::KeyAvailable) {
                $keyInfo = [System.Console]::ReadKey($true)
            }
            Start-Sleep -Milliseconds 100
        }

        if ($keyInfo) {
            $key = $keyInfo.Key
            if ($key -eq "Y") {
                $continue = $true
                break
            }
            elseif ($key -eq "N") {
                $continue = $false
                break
            }
        }
    }

    Write-Host
    return $continue
}

<#
.SYNOPSIS
   Checks if the specified range of Microsoft Visual C++ Redistributable versions is installed.

.DESCRIPTION
   The Test-VCRedistInstalled function checks the registry to determine if any version of
   Microsoft Visual C++ Redistributable within the specified range is installed on the system.
   It supports checking for the Visual C++ <oldest_year>-<newest_year> Redistributable (x64).

.PARAMETER vcRedistOldestYear
   The oldest year of the Visual C++ Redistributable range to search for.

.PARAMETER vcRedistNewestYear
   The newest year of the Visual C++ Redistributable range to search for.

.EXAMPLE
   Test-VCRedistInstalled -vcRedistOldestYear 2015 -vcRedistNewestYear 2022

   This example checks if any version of Microsoft Visual C++ 2015-2022 Redistributable (x64) is installed.
#>
function Test-VCRedistInstalled {
    param(
        [int]$vcRedistOldestYear,
        [int]$vcRedistNewestYear
    )

    $registryKeys = @(
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall',
        'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall'
    )

    $found = $false

    foreach ($key in $registryKeys) {
        $installedSoftware = Get-ChildItem $key |
        Get-ItemProperty |
        Select-Object -Property DisplayName

        $matchingSoftware = $installedSoftware | Where-Object { $_.DisplayName -match "Microsoft Visual C\+\+ $($vcRedistOldestYear)-$($vcRedistNewestYear) Redistributable \(x64\) .*" }

        if ($matchingSoftware) {
            $found = $true
            break
        }
    }

    if ($found) {
        Write-DebugLog "Visual C++ $($vcRedistOldestYear)-$($vcRedistNewestYear) Redistributable is already installed."
    }
    else {
        Write-DebugLog "Visual C++ $($vcRedistOldestYear)-$($vcRedistNewestYear) Redistributable is not installed."
    }

    return $found
}

<#
.SYNOPSIS
    Installs the Visual Studio redistributables on Windows systems.

.DESCRIPTION
    This function downloads and installs the Visual Studio 2015-2022 redistributable on Windows systems. It checks for administrator privileges and prompts the user for elevation if necessary.

.PARAMETER Interactive
    A boolean value that indicates whether to prompt the user for choices during the installation process.

.EXAMPLE
    Install-VCRedistWindows -Interactive $true
#>
function Install-VCRedistWindows {
    if ($os.family -ne "Windows") {
        return
    }

    function Get-VCRedist {
        if (-not (Test-Path $script:vcInstallerPath)) {
            try {
                # This is useful for debugging the VC exe download, but commenting by default out to avoid leaking users' personal information
                # Write-DebugLog "Attempting to download: ${script:vcRedistUrl} to ${script:vcInstallerPath}"
                Invoke-WebRequest -Uri $script:vcRedistUrl -OutFile $script:vcInstallerPath
            }
            catch {
                Write-Error "Failed to download Visual Studio redistributables. Please check your internet connection or provide a pre-downloaded installer."
                exit 1
            }
        }
    }

    if (-not (Test-IsAdmin)) {
        Write-Host "`n`n"
        Write-Warning "Admin privileges are required to install Visual Studio redistributables. The script will attempt to run the installer with elevated privileges."

        $continueInstallation = DisplayCountdown -countdown 15

        if ($continueInstallation) {
            # Continue with normal operations
            Write-Host "Continuing with the installation."
        }
        else {
            Write-Host "Installation cancelled."
            Write-Host "Please manually install VC via the following URL: " -ForegroundColor Yellow -NoNewline
            Write-Host "https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Yellow
            exit 1
        }

        try {
            Get-VCRedist
            $proc = Start-Process -FilePath $script:vcInstallerPath -ArgumentList `/install` `/quiet` `/norestart` '-Verb RunAs' -PassThru -Wait -NoNewWindow
            $proc.WaitForExit()
            $exitCode = $proc.ExitCode

            if ($exitCode -eq 3010) {
                # Exit code 3010 means a reboot is required, so handle this case separately
                Write-CriticalLog "VC Redistributable installation requires a reboot." -ForegroundColor Yellow
                Write-CriticalLog "Exiting script to avoid any issues. Please reboot the computer and run this setup again." -ForegroundColor Red
                exit 0
            }
            elseif ($exitCode -ne 0) {
                # Any other non-zero exit code indicates an error
                Write-ErrorLog "VC Redistributable installation failed with exit code $($proc.ExitCode)."
                Write-CriticalLog "Please try again or manually install VC via the following URL: ." -ForegroundColor Yellow -NoHeader
                Write-CriticalLog "https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Yellow -NoHeader
                exit 1
            }
        }
        catch {
            Write-ErrorLog "Failed to start the installation process with admin privileges. $_"
            exit 1
        }

        return
    }

    Get-VCRedist

    try {
        $proc = Start-Process -FilePath $script:vcInstallerPath -ArgumentList `/install` `/quiet` `/norestart` -PassThru -Wait -NoNewWindow
        $proc.WaitForExit()
        $exitCode = $proc.ExitCode

        if ($exitCode -eq 3010) {
            # Exit code 3010 means a reboot is required, so handle this case separately
            Write-CriticalLog "VC Redistributable installation requires a reboot." -ForegroundColor Yellow
            Write-CriticalLog "Exiting script to avoid any issues. Please reboot the computer and run this setup again." -ForegroundColor Red
            exit 0
        }
        elseif ($exitCode -ne 0) {
            # Any other non-zero exit code indicates an error
            Write-ErrorLog "VC Redistributable installation failed with exit code $($proc.ExitCode)."
            exit 1
        }
    }
    catch {
        Write-Error "Failed to start the installation process. $_"
        exit 1
    }

    try {
        if (Test-Path -Path $script:vcInstallerPath) {
            Remove-Item -Path $script:vcInstallerPath -Force
        }
    }
    catch {
        Write-Error "Failed to remove the installer file. $_"
        exit 1
    }
}

<#
.SYNOPSIS
    Retrieves the built-in parameters of the current PowerShell executable.

.DESCRIPTION
    This function identifies the appropriate PowerShell executable based on the
    platform and version of PowerShell, then parses the built-in parameters from
    the executable's help output.

.OUTPUTS
    [string[]] An array of built-in parameter names.

.EXAMPLE
    $builtInParams = Get-BuiltInParameters
    Returns an array of built-in parameter names for the current PowerShell executable.
#>
function Get-BuiltInParameters {
    if ($os.family -eq "Windows") {
        if ($PSVersionTable.PSVersion.Major -ge 6) {
            $powershellExeName = 'pwsh.exe'
        }
        else {
            $powershellExeName = 'powershell.exe'
        }
    }
    else {
        $powershellExeName = 'pwsh'
    }
    
    $powershellExe = (Get-Command $powershellExeName).Path

    $exeHelp = & $powershellExe -h | Out-String
    $regex = [regex]'-\w+'
    $builtInParams = $regex.Matches($exeHelp) | ForEach-Object { $_.Value.TrimStart('-') }
    return $builtInParams
}

<#
.SYNOPSIS
    Test-Parameters is a PowerShell function that validates input parameters.

.DESCRIPTION
    Test-Parameters checks if the provided parameters are valid and allowed. If invalid parameters are found,
    it outputs an error message and displays the list of valid parameters. It also considers built-in
    PowerShell parameters to avoid false positives.

.PARAMETER Dir
    The directory path containing the target script file for the Get-Help cmdlet.

.PARAMETER CommandString
    A string containing the command-line arguments to be tested.

.PARAMETER ValidParams
    A string array containing the valid parameter names allowed for the script.

.EXAMPLE
    Test-Parameters -Dir $PSScriptRoot -CommandString $MyInvocation.UnboundArguments -ValidParams $validParams
    This example demonstrates how to call Test-Parameters with the necessary parameters.

.NOTES
    Test-Parameters uses built-in parameters lists specific to the PowerShell version and operating system
    to ensure accurate validation.
#>
function Test-Parameters {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Dir,

        [Parameter(Mandatory = $true)]
        [string]$CommandString,

        [Parameter(Mandatory = $true)]
        [string[]]$ValidParams
    )

    # Find the target script for the Get-Help cmdlet
    $file = $(Split-Path -Leaf $PSCommandPath)
    $Script = Join-Path $Dir $file

    # List of built-in parameters to exclude
    if ($os.family -eq "Windows") {
        if ($PSVersionTable.PSVersion.Major -lt 6) {
            $builtInParams = [System.Management.Automation.Cmdlet].GetMethod('get_CommonParameters').Invoke($null, @())
        }
        else {
            $builtInParams = [System.Management.Automation.Internal.CommonParameters].GetProperties().Name
        }
    }
    else {
        $builtInParams = [System.Management.Automation.Internal.CommonParameters].GetProperties().Name
    }
    
    # Add a dash in front of each parameter name
    $builtInParams = $builtInParams | ForEach-Object { "-$_" }

    $globalBuiltInParams = $(Get-BuiltInParameters) | ForEach-Object { "-$_" }
    $allOptions = @($CommandString -split '\s|=+' | Where-Object { $_ -match '^-' -and $_ -notmatch '^-BatchArgs$|^-unboundArgs$' })

    Write-Debug "`nValidParams: $($ValidParams -join ', ')"
    Write-Debug "`nBuilt-in Params: $($builtInParams -join ', ')"
    Write-Debug "`nBuilt-in Global Params: $($globalBuiltInParams -join ', ')"
    Write-Debug "`nAll found Args: $($allOptions -join ', ')"

    # Separate valid and invalid options
    $invalidOptions = @($allOptions | Where-Object {
            $optionName = $_.Trim()
            if ($validParams -notcontains $optionName -and $builtInParams -notcontains $optionName -and $globalBuiltInParams -notcontains $optionName) {
                Write-Host "Invalid option found: $optionName"
                $true
            }
            else {
                $false
            }
        })

    # Check if there are any invalid options
    if ($invalidOptions) {
        if (Test-Path $Script) {
            Get-Help $Script -Parameter * | 
            Select-Object Name, @{Name = 'Description'; Expression = { $_.Description.Text } } | 
            Where-Object { $_.Name -ne 'unboundArgs' -and $_.Name -ne 'BatchArgs' } | 
            Sort-Object Name | Format-List
        }
        else {
            Write-CriticalLog "You can run Get-Help ./setup.ps1 -Full to see all valid parameters." -NoHeader
        }

        Write-CriticalLog "Illegal option(s): $($invalidOptions -join ', ').`nPlease see above for all valid options using only a single - to invoke each one." -ForegroundColor Red
        exit 1
    }
}

<#
.SYNOPSIS
   The main function for the setup.ps1 script.

.DESCRIPTION
   This function orchestrates the entire setup process for the Kohya application.
   It handles parameter loading, checks for administrator privileges, installs Python 3.10 and the Tk package,
   and performs any other necessary setup tasks.

.OUTPUTS
   None
#>
function Main {
    param (
        $Parameters
    )

    begin {
        # Then check to make sure we have the required arguments for minimal setup
        if (-not $Parameters.NoSetup) {
            $requiredKeys = @("Dir", "Branch", "GitRepo")
            if (-not (Test-Value -Params $Parameters -RequiredKeys $requiredKeys)) {
                Write-Error "Error: Some required parameters are missing. Please provide values in the config file or through command line arguments."
                exit 1
            }
        }
    }

    process {
        if (-not $Parameters.NoSetup) {
            if (-not (Test-Python310Installed)) {
                Write-CriticalLog "Installing Python 3.10. This could take a few minutes." -NoHeader
                Install-Python310 -Interactive:$Params.interactive
            }
            else {
                Write-CriticalLog "Python 3.10 is already installed." -NoHeader
            }
            
            if ($os.family -eq "Windows") {
                if (-not (Test-Python310Installed)) {
                    Write-CriticalLog "Installing Python 3.10 Tk. This could take a few minutes." -NoHeader
                    $installScope = Update-InstallScope($Interactive)
                    Install-Python3Tk $installScope
                }
            }
            else {
                Install-Python3Tk
            }

            if (-not (Test-GitInstalled)) {
                Write-CriticalLog "Installing git. This could take a few minutes." -NoHeader
                Install-Git
            }
            else {
                Write-CriticalLog "Git already installed." -NoHeader
            }

            if ($os.family -eq "Windows") {
                Write-DebugLog "Checking for VC version: ${script:vcRedistOldestYear}-${script:vcRedistNewestYear}"
                if (-not (Test-VCRedistInstalled -vcRedistOldestYear $script:vcRedistOldestYear -vcRedistNewestYear $script:vcRedistNewestYear)) {
                    Install-VCRedistWindows
                }
                else {
                    Write-CriticalLog "VC Redist already installed." -NoHeader
                }
            }
        }
    }

    end {
        # Update Path just in case Python was installed during this PowerShell session.
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        $pyExe = Get-PythonExePath
    
        if ($null -ne $pyExe) {
            # Check for launcher.py at the target installation folder if specified via CLI, if not try to test target directory anyway no matter how it was defined, then fall back to current script directory.
            if (![string]::IsNullOrEmpty($Dir) -and (Test-Path -Path (Join-Path -Path $Dir -ChildPath "launcher.py"))) {
                $launcher = Join-Path -Path $Dir -ChildPath "launcher.py"
            }
            elseif (Test-Path -Path (Join-Path -Path $PSScriptRoot -ChildPath "launcher.py")) {
                $launcher = Join-Path -Path $PSScriptRoot -ChildPath "launcher.py"
            }
            else {
                Write-Error "Error: launcher.py not found in provided directory or script directory."
                exit 1
            }

            $installArgs = New-Object System.Collections.ArrayList
            foreach ($key in $Parameters.Keys) {
                $argName = $key

                # Handle PascalCase keys
                if ($argName -cmatch '[a-z][A-Z]') {
                    $argName = -join ($argName.ToCharArray() | ForEach-Object {
                            if (($_ -ge 'A'[0]) -and ($_ -le 'Z'[0])) {
                                "-$($_.ToString().ToLower())"
                            }
                            else {
                                $_.ToString().ToLower()
                            }
                        })
                }
                else {
                    # Handle uppercase keys
                    $argName = $argName.ToLower()
                }

                # Replace underscore with hyphen and prepend with --
                $argName = "--" + ($argName.Replace("_", "-").TrimStart('-'))

                if ($null -ne $Parameters.$key) {
                    Write-DebugLog "Checking parameter: $key, value: $($Parameters.$key), type: $($Parameters.$key.GetType().Name)"
                }
                else {
                    Write-DebugLog "Checking parameter: $key, value: $($Parameters.$key), type: Null"
                }

                if ($Parameters[$key] -is [int]) {
                    # Handle integer parameters
                    $installArgs.Add($argName) | Out-Null
                    $installArgs.Add($Parameters[$key]) | Out-Null
                }
                elseif ($Parameters.$key -is [bool] -or $Parameters.$key -is [switch]) {
                    # Handle boolean and switch parameters
                    if ($Parameters[$key] -eq $true) {
                        $installArgs.Add($argName) | Out-Null
                        Write-DebugLog "Boolean argument: $argName"
                    }
                }
                elseif ($key -eq "verbosity") {
                    # Handle verbosity separately, as -vvvv or --verbosity 4
                    $verbosity = $Parameters.$key
                    if ($verbosity -gt 0) {
                        # produces "-vvvv" for verbosity 4 or "--verbosity 4"
                        # Below is the -vvvv version
                        # $verbosityArg = "-" + ("v" * $verbosity)
                        # Below is the --verbosity [int] version
                        $verbosityArg = "--verbosity"
                        $verbosityValue = [int]$verbosity
                        $installArgs.Add($verbosityArg) | Out-Null
                        $installArgs.Add($verbosityValue) | Out-Null
                    }
                }
                elseif (![string]::IsNullOrEmpty($Parameters.$key)) {
                    $installArgs.Add($argName) | Out-Null
                    $installArgs.Add($Parameters.$key) | Out-Null
                }
            }

            # Call launcher.py with the appropriate parameters
            $launcherFileName = Split-Path $launcher -Leaf
            Write-CriticalLog "Switching to ${launcherFileName}.`n`n" -NoHeader
            $command = "$pyExe -u $launcher $installArgs"
            Write-DebugLog "Running command: $command"
            & $pyExe -u "$launcher" $($installArgs.ToArray())
        }
        else {
            Write-Error "Error: Python 3.10 executable not found. Installation cannot proceed."
            exit 1
        }
    }
}

# ---------------------------------------------------------
# The main execution block of the code starts here
# ---------------------------------------------------------

# Ensure script executes as normal while printing debug statements.
# This only affects the PowerShell script parameter setup as that happens before logging setup.
if ($PSBoundParameters.ContainsKey("Debug")) {
    $DebugPreference = "Continue"
}
else {
    $DebugPreference = "SilentlyContinue"
}

# Call the Get-Parameters function to process the arguments in the intended fashion
# Parameter value assignments should be as follows:
# 1) Command line arguments (user overrides)
# 2) Configuration file (install_config.yml). Default locations:
#    a) OS-specific standard folders
#       Windows: $env:APPDATA, "kohya_ss\install_config.yaml"
#       Non-Windows: $env:HOME/kohya_ss/
#    b) Installation Directory
#    c) The folder this script is run from
# 3) Default values built into the scripts if none specified by user and there is no config file.

# Set a global OS detection for usage in functions
$os = Get-OsInfo
Write-DebugLog "Detected OS Family: {$os.family}."

# Format parameters and setup logging
$script:Parameters = Get-Parameters -BoundParameters $PSBoundParameters -BatchArgs $BatchArgs
Set-Logging -LogDir $script:Parameters.LogDir

# If the -Debug switch is used we want to make sure our custom logging happens too.
if ($PSBoundParameters.ContainsKey("Debug")) {
    $script:Parameters.Verbosity = 3
}

# Check for illegal or malformed parameters
if ($MyInvocation.Line) {
    $commandLine = $MyInvocation.Line
}
else {
    $commandLine = [Environment]::CommandLine
}

# Debug the command line call
Write-Debug "Raw command line call: $commandLine"
Write-Debug "Arguments found: $($PSBoundParameters.Keys)"

# These lines dynamically grab parameter names from the script and filters out our "hidden" arguments.
$parsedScript = [System.Management.Automation.Language.Parser]::ParseFile($PSCommandPath, [ref]$null, [ref]$null)
$paramBlock = $parsedScript.Find({ $args[0] -is [System.Management.Automation.Language.ParamBlockAst] }, $false)
$parameterNames = $paramBlock.Parameters.Name.VariablePath.UserPath | Where-Object { $_ -ne 'unboundArgs' -and $_ -ne 'BatchArgs' }
$validParams = $parameterNames | ForEach-Object { "-$_" }

if ([string]::IsNullOrWhiteSpace($ValidParams) -or $null -eq $ValidParams) {
    Write-Error "Error: Could not parse arguments."
    exit 1
}
else {
    Test-Parameters -Dir $Parameters["Dir"] -CommandString $commandLine -ValidParams $validParams
}

# If all switches came back valid, validate acceptable Torch versions launcher.py will accept
$validTorchVersions = @(0, 1, 2)

if (-not ($validTorchVersions -contains $Parameters.TorchVersion)) {
    Write-CriticalLog "Invalid value for -TorchVersion: $($Parameters.TorchVersion). Valid values are $($validTorchVersions -join ', ')." -ForegroundColor Red
    exit 1
}

# Define global Python path to use in various functions
$script:pythonPath = Get-PythonExePath
$script:gitPath = Get-GitExePath

# Define versions globally for easy modification
# Software Versions
$script:pythonVersion = "3.10.11"
$script:gitVersion = "2.40.1"
$script:vcRedistOldestYear = 2015
$script:vcRedistNewestYear = 2022

# Set the downloads folder for storing pre-req installation files
if ($os.family -eq "Windows") {
    $script:downloadsFolder = Join-Path -Path $env:USERPROFILE -ChildPath 'Downloads'


    $script:pythonInstallerUrl = "https://www.python.org/ftp/python/${script:pythonVersion}/python-${script:pythonVersion}-amd64.exe"
    # Derive the release page URL from the installer URL
    $script:pythonReleasePageUrl = "https://www.python.org/downloads/release/python-$($script:pythonVersion -replace '\.','')/"

    #Grab the Python raw installer file name and path
    $script:pythonInstallerFile = Split-Path -Leaf $script:pythonInstallerUrl
    $script:pythonInstallerPath = Join-Path -Path $script:downloadsFolder -ChildPath $script:pythonInstallerFile

    # Get the MD5 sum from the website for a given version
    $script:pythonMd5 = Get-Python310Md5HashFromWeb

    # Git URL and Hash
    $script:gitUrl = "https://github.com/git-for-windows/git/releases/download/v${script:gitVersion}.windows.1/Git-${script:gitVersion}-64-bit.exe"
    $script:gitSha256 = Get-GitHashFromWeb
    $script:gitInstallerFile = Split-Path -Leaf $script:gitUrl
    $script:gitInstallerPath = Join-Path -Path $script:downloadsFolder -ChildPath $script:gitInstallerFile

    # VC Redist URL and hash
    $script:vcRedistInstallerName = "vc_redist.x64.exe"
    $script:vcRedistUrl = "https://aka.ms/vs/17/release/$($script:vcRedistInstallerName)"
    $script:vcInstallerPath = Join-Path -Path $script:downloadsFolder -ChildPath $script:vcRedistInstallerName
    Write-DebugLog "VC Installer path: $script:vcInstallerPath"
}

# Main entry point to the script
Write-CriticalLog "Beginning main function." -NoHeader
Main -Parameters $script:Parameters
