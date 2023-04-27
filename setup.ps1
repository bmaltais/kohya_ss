<#
.SYNOPSIS
    Kohya_SS Installation Script for Windows PowerShell and PowerShell 7+.

.DESCRIPTION
    This script automates the installation of Kohya_SS on Windows, macOS, Ubuntu, and RedHat Linux systems. This is the
    install bootstrap file ensuring Python 3.10 and Python 3.10 TK is installed and available.

.EXAMPLE
    # Specifies custom branch, install directory, and git repo
    .\setup.ps1 -Branch 'dev' -Dir 'C:\workspace\kohya_ss' -GitRepo 'https://mycustom.repo.tld/custom_fork.git'

.EXAMPLE
    # Maximum verbosity, fully automated installation in a runpod environment skipping the runpod env checks
    .\setup.ps1 -Verbose -SkipSpaceCheck -Runpod

.PARAMETER Branch
    Select which branch of kohya to check out on new installs.

.PARAMETER Dir
    The full path you want kohya_ss installed to.

.PARAMETER NoSetup
    Skip all setup steps and only validate python requirements then launch GUI.

.PARAMETER File
    The full path to the configuration file. If not provided, the script looks for an 'install_config.yaml' file in the script's directory.

.PARAMETER GitRepo
    You can optionally provide a git repo to check out for runpod installation. Useful for custom forks.

.PARAMETER Interactive
    Interactively configure accelerate instead of using default config file.

.PARAMETER LogDir
    Specifies the directory where log files will be stored.

.PARAMETER NoGitUpdate
    Do not update kohya_ss repo. No git pull or clone operations.

.PARAMETER Public
    Expose public URL in runpod mode. Won't have an effect in other modes.

.PARAMETER Runpod
    Forces a runpod installation. Useful if detection fails for any reason.

.PARAMETER SetupOnly
    Do not launch GUI. Only conduct setup operations.

.PARAMETER SkipSpaceCheck
    Skip the 10Gb minimum storage space check.

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
    [string]$File = "",
    [string]$Branch = "",
    [string]$Dir = "",
    [string]$GitRepo = "",
    [switch]$Interactive,
    [string]$LogDir = "",
    [switch]$NoSetup,
    [switch]$Public,
    [switch]$Runpod,
    [switch]$SetupOnly,
    [switch]$SkipSpaceCheck,
    [int]$Verbosity = -1,
    [switch]$Update,
    [string]$LISTEN = "",
    [string]$USERNAME = "",
    [string]$PASSWORD = "",
    [int]$SERVER_PORT,
    [switch]$INBROWSER,
    [switch]$SHARE
)

# Define global Python path to use in various functions
$pythonPath = $null

<#
.SYNOPSIS
   Writes a debug message to the console if the verbosity level is 3 or higher.

.DESCRIPTION
   The function takes a message string as input and writes it to the console 
   as a debug message only if the global verbosity level is set to 3 or higher.

.PARAMETER message
   The debug message to be written to the console.

.EXAMPLE
   Write-LogDebug "This is a debug message."
#>
function Write-LogDebug($message) {
    if ($verbosity -ge 3) {
        Write-Debug $message
    }
}

<#
.SYNOPSIS
   Writes a verbose message to the console if the verbosity level is 2 or higher.

.DESCRIPTION
   The function takes a message string as input and writes it to the console 
   as a verbose message only if the global verbosity level is set to 2 or higher.

.PARAMETER message
   The verbose message to be written to the console.

.EXAMPLE
   Write-LogVerbose "This is a verbose message."
#>
function Write-LogVerbose($message) {
    if ($verbosity -ge 2) {
        Write-Verbose $message
    }
}

<#
.SYNOPSIS
   Writes an informational message to the console if the verbosity level is 1 or higher.

.DESCRIPTION
   The function takes a message string as input and writes it to the console 
   as an informational message only if the global verbosity level is set to 1 or higher.

.PARAMETER message
   The informational message to be written to the console.

.EXAMPLE
   Write-LogInformation "This is an informational message."
#>
function Write-LogInformation($message) {
    if ($verbosity -ge 1) {
        Write-Information $message
    }
}

<#
.SYNOPSIS
   Writes a critical message to the console. This message will always be displayed regardless of the verbosity level.

.DESCRIPTION
   The function takes a message string as input and always writes it to the console 
   as a critical message.

.PARAMETER message
   The critical message to be written to the console.

.EXAMPLE
   Write-LogCritical "This is a critical message."
#>
function Write-LogCritical($message) {
    # Always write critical messages
    Write-Host $message
}

<#
.SYNOPSIS
   Handles the loading of parameter values with a specific order of precedence.

.DESCRIPTION
   This function handles the loading of parameter values in the following order of precedence:
   1. Command-line arguments provided by the user
   2. Values defined in a configuration file
   3. Default values specified in the function

   First, the function checks for the presence of a configuration file in the specified locations.
   If found, the configuration file's values are loaded into a hashtable.
   Then, default values are added to the hashtable for any parameters not defined in the configuration file.
   Finally, any command-line arguments provided by the user are merged into the hashtable,
   overriding the corresponding values from the configuration file or the function's default values.
   If neither a configuration file nor a command-line argument is provided for a parameter,
   the function will fallback to the hard-coded default values.

   To access arguments from the hashtable, use the following syntax:
   $Config['parameter_name']

   For example:
   $Config['setup_branch']
   $Config['setup_dir']
   $Config['gui_listen']

.PARAMETER File
   An optional parameter to specify the path to a configuration file.

.OUTPUTS
   System.Collections.Hashtable
   Outputs a hashtable containing the merged parameter values.
#>
function Get-Parameters {
    param (
        [Parameter(Mandatory = $true)]
        [hashtable]$BoundParameters
    )

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
                    # Convert relative paths to absolute and normalize
                    $FullPath = Join-Path (Get-Location) $value
    
                    if ($PSVersionTable.PSVersion.Major -lt 6) {
                        # Use Resolve-Path for PowerShell 5.1 and older
                        if (!(Test-Path $FullPath)) {
                            # If path does not exist, create a temporary one, resolve it, and then remove it
                            $ParentDir = Split-Path $FullPath -Parent
                            $TestPath = Join-Path $ParentDir "testwrite.tmp"
                            try {
                                $null = New-Item -Path $TestPath -ItemType File -ErrorAction Stop
                                Remove-Item -Path $TestPath -ErrorAction Stop
                            }
                            catch {
                                throw "The script does not have write permissions to create a file in the directory: $ParentDir"
                            }
    
                            $null = New-Item -Path $FullPath -ItemType Directory -Force
                            $AbsolutePath = (Resolve-Path $FullPath).Path
                            Remove-Item -Path $FullPath -Force
                        }
                        else {
                            $AbsolutePath = (Resolve-Path $FullPath).Path
                        }
                    }
                    else {
                        # Use System.IO.Path.GetFullPath for PowerShell 6 and later
                        $AbsolutePath = [System.IO.Path]::GetFullPath($FullPath)
                    }
    
                    $Result[$key] = $AbsolutePath
                }
            }
        }
    
        return $Result
    }


    # Check for the existence of the powershell-yaml module and install it if necessary
    if (-not (Get-Module -ListAvailable -Name 'powershell-yaml')) {
        Install-Module -Name 'powershell-yaml' -Scope CurrentUser -Force
    }
    Import-Module 'powershell-yaml'

    # Define possible configuration file locations
    $configFileLocations = if ($IsWindows) {
        @(
            $File,
            (Join-Path -Path $env:APPDATA -ChildPath "kohya_ss\config_files\installation\install_config.yml"),
            (Join-Path -Path $env:LOCALAPPDATA -ChildPath "kohya_ss\install_config.yml"),
            (Join-Path -Path "$env:USERPROFILE\.kohya_ss" -ChildPath "install_config.yml"),
            (Join-Path -Path $PSScriptRoot -ChildPath "install_config.yml")
        )
    }
    else {
        @(
            $File,
            (Join-Path -Path $env:HOME -ChildPath ".kohya_ss\install_config.yml"),
            (Join-Path -Path "$env:USERPROFILE\kohya_ss" -ChildPath "install_config.yml"),
            (Join-Path -Path $PSScriptRoot -ChildPath "install_config.yml")
        )
    }

    # Define the default values
    $Defaults = @{
        'Branch'         = 'master'
        'Dir'            = "$env:USERPROFILE\.kohya_ss"
        'GitRepo'        = 'https://github.com/bmaltais/kohya_ss.git'
        'Interactive'    = $false
        'LogDir'         = "$env:USERPROFILE\.kohya_ss\logs"
        'NoSetup'        = $false
        'Public'         = $false
        'Runpod'         = $false
        'SetupOnly'      = $false
        'SkipSpaceCheck' = $false
        'Verbosity'      = -1
        'Update'         = $false
        'LISTEN'         = '127.0.0.1'
        'USERNAME'       = ''
        'PASSWORD'       = ''
        'SERVER_PORT'    = 7861
        'INBROWSER'      = $false
        'SHARE'          = $false
    }

    # Load the configuration file from the first existing location
    $Config = @{}
    foreach ($location in $configFileLocations) {
        if (Test-Path $location) {
            $Config = (Get-Content $location | Out-String | ConvertFrom-Yaml)
            break
        }
    }

    # Iterate through the default values and set them if not defined in the config file
    foreach ($key in $Defaults.Keys) {
        if (-not $Config.ContainsKey($key)) {
            $Config[$key] = $Defaults[$key]
        }
    }

    # Update the config with the command-line arguments
    foreach ($key in $Defaults.Keys) {
        if ($BoundParameters.ContainsKey($key)) {
            if ($key -eq 'Verbosity' -and $BoundParameters['Verbosity'] -eq -1) {
                continue
            }
            if ($BoundParameters[$key] -is [switch]) {
                $Config[$key] = $BoundParameters[$key].IsPresent
            }
            else {
                $Config[$key] = $BoundParameters[$key]
            }
        }
    }
    

    $Config = Convert-RelativePathsToAbsolute -Params $Config

    # Output the final configuration
    Write-Host "Config:"
    foreach ($key in $Config.Keys) {
        Write-Host "${key}: $($Config.$key)"
    }
    
    return $Config
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
        Write-Host "All required keys are present."
    } else {
        Write-Host "Some required keys are missing."
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
    if ($PSVersionTable.Platform -eq 'Win32NT') {
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

    if ($IsWindows) {
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
    Write-Host "Operating System: $($os.name)"
    Write-Host "OS Family: $($os.family)"
    Write-Host "OS Version: $($os.version)"
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
    $pythonBinaries = @("python", "python3", "python3.10")

    if ($IsWindows) {
        # Add windows-specific paths
        $paths = @("${Env:ProgramFiles}\Python310", "${Env:ProgramFiles(x86)}\Python310")
        # Add .exe extension for Windows
        $pythonBinaries = $pythonBinaries | ForEach-Object { "$_.exe" }
    }
    else {
        # Unix-like system paths
        $paths = @("/usr/bin", "/usr/local/bin", "${Env:HOME}/.local/bin")
    }

    foreach ($path in $paths) {
        foreach ($binary in $pythonBinaries) {
            $global:pythonPath = Join-Path -Path $path -ChildPath $binary
            if (Test-Path -Path $global:pythonPath ) {
                try {
                    $pythonVersion = & $global:pythonPath  --version 2>&1 | Out-String -Stream -ErrorAction Stop
                    $pythonVersion = $pythonVersion -replace '^Python\s', ''
                    if ($pythonVersion.StartsWith('3.10')) {
                        return $true
                    }
                }
                catch {
                    switch ($_.Exception.GetType().Name) {
                        'Win32Exception' {
                            Write-Error "Python executable found at $global:pythonPath , but it could not be run. It may be corrupted or there may be a permission issue."
                            return $false
                        }
                        'RuntimeException' {
                            if ($_.Exception.Message -like '*The term*is not recognized as the name of a cmdlet*') {
                                Write-Error "Python executable not found at $global:pythonPath ."
                                return $false
                            }
                            else {
                                Write-Error "An unknown error occurred when trying to run Python at ${global:pythonPath }: $($_.Exception.Message)"
                                return $false
                            }
                        }
                        default {
                            Write-Error "An unknown error occurred when trying to check Python version at ${global:pythonPath }: $($_.Exception.Message)"
                            return $false
                        }
                    }
                }
            }
        }
    }

    return $false
}

<#
.SYNOPSIS
   Retrieves the path to the Python 3.10 executable.

.DESCRIPTION
   Searches for Python 3.10 executable in the system and returns its path.
   It handles different platforms and common edge cases such as Homebrew on macOS and FreeBSD.

.EXAMPLE
   $pythonPath = Get-PythonExePath

.OUTPUTS
   System.String
   The path to the Python 3.10 executable or $null if not found.
#>
function Get-PythonExePath {
    $pythonCandidates = @("python3.10", "python3", "python")

    if ($PSVersionTable.Platform -eq 'Windows' -or $PSVersionTable.PSEdition -eq 'Desktop') {
        $pythonCandidates += @("python3.10.exe", "python3.exe", "python.exe")
    }

    $foundPythonPath = $null

    foreach ($candidate in $pythonCandidates) {
        try {
            $pythonPath = (Get-Command $candidate -ErrorAction SilentlyContinue).Source
            if ($null -ne $pythonPath) {
                $pythonVersion = & $pythonPath --version 2>&1
                if ($pythonVersion -match "^Python 3\.10") {
                    $foundPythonPath = $pythonPath
                    break
                }
            }
        }
        catch {
            continue
        }
    }

    if ($null -eq $foundPythonPath) {
        $osInfo = Get-OsInfo

        # macOS with Homebrew
        if ($osInfo.family -eq "Darwin") {
            $brewPythonPath = "/usr/local/opt/python@3.10/bin/python3.10"
            if (Test-Path $brewPythonPath) {
                $foundPythonPath = $brewPythonPath
            }
        }

        # FreeBSD
        if ($osInfo.family -eq "FreeBSD") {
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
            if (Test-IsAdmin) {
                $installScope = Update-InstallScope($Interactive)

                $pythonUrl = "https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe"
                $pythonInstallerName = "python-3.10.0-amd64.exe"
                $downloadsFolder = [Environment]::GetFolderPath('MyDocuments') + "\Downloads"
                $installerPath = Join-Path -Path $downloadsFolder -ChildPath $pythonInstallerName

                if (-not (Test-Path $installerPath)) {
                    try {
                        $pythonUrl = "https://www.python.org/ftp/python/3.10.0/$pythonInstallerName"
                        Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath
                    }
                    catch {
                        Write-Host "Failed to download Python 3.10. Please check your internet connection or provide a pre-downloaded installer."
                        exit 1
                    }
                }

                if ($installScope -eq "user") {
                    Start-Process $installerPath -ArgumentList "/passive InstallAllUsers=0" -Wait
                }
                else {
                    Start-Process $installerPath -ArgumentList "/passive InstallAllUsers=1" -Wait
                }

                Remove-Item $installerPath
            }
            else {
                # We default to installing at a user level if admin is not detected.
                Start-Process $installerPath -ArgumentList "/passive InstallAllUsers=0" -Wait
            }
        }

        function Install-Python310Mac {
            if (Get-Command "brew" -ErrorAction SilentlyContinue) {
                brew install python@3.10
                brew link --overwrite --force python@3.10
            }
            else {
                Write-Host "Please install Homebrew first to continue with Python 3.10 installation."
                Write-Host "You can find that here: https://brew.sh"
                exit 1
            }
        }

        function Install-Python310Linux {
            $elevate = ""
            if (-not ($env:USER -eq "root" -or (id -u) -eq 0 -or $env:EUID -eq 0)) {
                $elevate = "sudo"
            }
            $os = Get-OsInfo

            switch ($os.family) {
                "Ubuntu" {
                    if (& $elevate apt-get update) {
                        if (!(& $elevate apt-get install -y python3.10)) {
                            Write-Host "Error: Failed to install python via apt. Installation of Python 3.10 aborted."
                        }
                    }
                    else {
                        Write-Host "Error: Failed to update package list. Installation of Python 3.10 aborted."
                    }
                }
                "Debian" {
                    if (& $elevate apt-get update) {
                        if (!(& $elevate apt-get install -y python3.10)) {
                            Write-Host "Error: Failed to install python via apt. Installation of Python 3.10 aborted."
                        }
                    }
                    else {
                        Write-Host "Error: Failed to update package list. Installation of Python 3.10 aborted."
                    }
                }
                "RedHat" {
                    if (!(& $elevate dnf install -y python3.10)) {
                        Write-Host "Error: Failed to install python via dnf. Installation of Python 3.10 aborted."
                    }
                }
                "Arch" {
                    if (!(& $elevate pacman -Sy --noconfirm python3.10)) {
                        Write-Host "Error: Failed to install python via pacman. Installation of Python 3.10 aborted."
                    }
                }
                "openSUSE" {
                    if (!(& $elevate zypper install -y python3.10)) {
                        Write-Host "Error: Failed to install python via zypper. Installation of Python 3.10 aborted."
                    }
                }
                default {
                    Write-Host "Unsupported Linux distribution. Please install Python 3.10 manually."
                    exit 1
                }
            }
        }


        if (Test-Python310Installed) {
            Write-Host "Python 3.10 is already installed."
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
            Write-Host "Unsupported operating system. Please install Python 3.10 manually."
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
            Write-Host "Unsupported operating system. Please install Python 3.10 manually."
            exit 1
        }

        if (Test-Python310Installed) {
            Write-Host "Python 3.10 installed successfully."
        }
        else {
            Write-Host 'Failed to install. Please ensure Python 3.10 is installed and available in $PATH.'
            exit 1
        }
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

    $os = Get-OsInfo
    $osFamily = $os.family.ToLower()
    write-host $osFamily

    if ($PSVersionTable.Platform -eq 'Unix') {
        # Linux / macOS installation
    
        # Pre-check: Try to import Tkinter in Python 3.10
        $isTkinterInstalled = $false
        try {
            $tkinterCheckOutput = & $global:pythonPath -c "import tkinter" 2>&1
            if (-not $tkinterCheckOutput) {
                $isTkinterInstalled = $true
            }
        }
        catch {
            Write-Error "Tkinter not found. Attempting to install."
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
                    Write-Host "Error: Failed to install Python 3.10 Tk on Ubuntu. $_"
                }
            }
            elseif ($osFamily -match "debian") {
                # Debian installation
                try {
                    Invoke-Expression (Get-ElevationCommand "apt-get" "update")
                    Invoke-Expression (Get-ElevationCommand "apt-get" "install" "-y" "python3.10-tk")
                }
                catch {
                    Write-Host "Error: Failed to install Python 3.10 Tk on Debian. $_"
                }
            }
            elseif ($osFamily -match "redhat") {
                # Red Hat installation
                try {
                    Invoke-Expression (Get-ElevationCommand "dnf" "install" "-y" "python3.10-tkinter")
                }
                catch {
                    Write-Host "Error: Failed to install Python 3.10 Tk on Red Hat. $_"
                }
            }
            elseif ($osFamily -match "arch") {
                # Arch installation
                try {
                    Invoke-Expression (Get-ElevationCommand "pacman" "-S" "--noconfirm" "tk")
                }
                catch {
                    Write-Host "Error: Failed to install Python 3.10 Tk on Arch. $_"
                }
            }
            elseif ($osFamily -match "opensuse") {
                # openSUSE installation
                try {
                    Invoke-Expression (Get-ElevationCommand "zypper" "install" "-y" "python3.10-tk")
                }
                catch {
                    Write-Host "Error: Failed to install Python 3.10 Tk on openSUSE. $_"
                }
            }
            elseif ($osFamily -match "macos") {
                if (Test-Path "/usr/local/bin/brew") {
                    try {
                        # macOS installation using Homebrew
                        Invoke-Expression "brew install python-tk@3.10"
                    }
                    catch {
                        Write-Host "Error: Failed to install Python 3.10 Tk on macOS using Homebrew. $_"
                    }
                }
                else {
                    Write-Host "Unsupported Unix platform or package manager not found."
                }
            }
            else {
                Write-Host "Unsupported Linux distribution. Please install Python 3.10 Tk manually."
            }
        }
        else {
            Write-Host "Tkinter for Python 3.10 is already installed on this system."
        }
    }
    else {
        # Windows installation
        $pythonInstallerUrl = "https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe"
        $pythonInstallerFile = "python-3.10.0-amd64.exe"
        $downloadsFolder = [Environment]::GetFolderPath('MyDocuments') + "\Downloads"
        $installerPath = Join-Path -Path $downloadsFolder -ChildPath $pythonInstallerFile
        Invoke-WebRequest -Uri $pythonInstallerUrl -OutFile $installerPath

        $installScope = Update-InstallScope($Interactive)

        if ($installScope -eq 'allusers') {
            if (Test-IsAdmin) {
                try {
                    Start-Process -FilePath $installerPath -ArgumentList "/passive InstallAllUsers=1 PrependPath=1 Include_tcltk=1" -Wait
                }
                catch {
                    Write-Host "Error: Failed to install Python 3.10 Tk for all users on Windows. $_"
                }
            }
            else {
                if (Test-IsAdmin) {
                    Write-Host "Warning: Running as administrator, but 'user' scope is selected. Proceeding with user-scope installation."
                }
                try {
                    Start-Process -FilePath $installerPath -ArgumentList "/passive InstallAllUsers=0 PrependPath=1 Include_tcltk=1" -Wait
                }
                catch {
                    Write-Host "Error: Failed to install Python 3.10 Tk for current user on Windows. $_"
                }
            }

            Remove-Item $installerPath
        }
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

    function Test-GitInstalled {
        try {
            git --version
            return $true
        }
        catch {
            return $false
        }
    }

    function Install-GitWindows {
        $packageManagerFound = $false
        $errorMsg = "Package manager detected, but install failed. Not attempting directly install to avoid conflicts. Please install Git manually from https://git-scm.com/download/win and re-run the script."

        if (Get-Command "scoop" -ErrorAction SilentlyContinue) {
            try {
                scoop install git
                $packageManagerFound = $true
            }
            catch {
                $packageManagerFound = $false
                Write-Host $errorMsg -ForegroundColor Red
            }
        }
        if (-not $packageManagerFound -and (Get-Command "choco" -ErrorAction SilentlyContinue)) {
            try {
                choco install git
                $packageManagerFound = $true
            }
            catch {
                $packageManagerFound = $false
                Write-Host $errorMsg -ForegroundColor Red
            }
        }
        if (-not $packageManagerFound -and (Get-Command "winget" -ErrorAction SilentlyContinue)) {
            try {
                winget install --id Git.Git
                $packageManagerFound = $true
            }
            catch {
                $packageManagerFound = $false
                Write-Host $errorMsg -ForegroundColor Red
            }
        }

        if (-not $packageManagerFound) {
            if (Test-IsAdmin) {
                $installScope = Update-InstallScope($Interactive)

                $gitUrl = "https://github.com/git-for-windows/git/releases/download/v2.35.1.windows.1/Git-2.35.1-64-bit.exe"
                $gitInstallerName = "Git-2.35.1-64-bit.exe"
                $downloadsFolder = [Environment]::GetFolderPath('MyDocuments') + "\Downloads"
                $installerPath = Join-Path -Path $downloadsFolder -ChildPath $gitInstallerName

                if (-not (Test-Path $installerPath)) {
                    try {
                        Invoke-WebRequest -Uri $gitUrl -OutFile $installerPath
                    }
                    catch {
                        Write-Host "Failed to download Git. Please check your internet connection or provide a pre-downloaded installer."
                        exit 1
                    }
                }

                if ($installScope -eq "user") {
                    Start-Process $installerPath -ArgumentList "/VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -Wait
                }
                else {
                    Start-Process $installerPath -ArgumentList "/VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -Wait
                }

                Remove-Item $installerPath
            }
            else {
                # We default to installing at a user level if admin is not detected.
                Start-Process $installerPath -ArgumentList "/VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -Wait
            }
        }
    }

    function Install-GitMac {
        if (Get-Command "brew" -ErrorAction SilentlyContinue) {
            brew install git
        }
        else {
            Write-Host "Please install Homebrew first to continue with Git installation."
            Write-Host "You can find that here: https://brew.sh"
            exit 1
        }
    }

    function Install-GitLinux {
        $elevate = ""
        if (-not ($env:USER -eq "root" -or (id -u) -eq 0 -or $env:EUID -eq 0)) {
            $elevate = "sudo"
        }
        $os = Get-OsInfo

        switch ($os.family) {
            "Ubuntu" {
                if (& $elevate apt update) {
                    if (!(& $elevate apt-get install -y git)) {
                        Write-Host "Error: Failed to install Git via apt. Installation of Git aborted."
                    }
                }
                else {
                    Write-Host "Error: Failed to update package list. Installation of Git aborted."
                }
            }
            "Debian" {
                if (& $elevate apt update) {
                    if (!(& $elevate apt-get install -y git)) {
                        Write-Host "Error: Failed to install Git via apt. Installation of Git aborted."
                    }
                }
                else {
                    Write-Host "Error: Failed to update package list. Installation of Git aborted."
                }
            }
            "RedHat" {
                if (!(& $elevate dnf install -y git)) {
                    Write-Host "Error: Failed to install Git via dnf. Installation of Git aborted."
                }
            }
            "Arch" {
                if (!(& $elevate pacman -Sy --noconfirm git)) {
                    Write-Host "Error: Failed to install Git via pacman. Installation of Git aborted."
                }
            }
            "openSUSE" {
                if (!(& $elevate zypper install -y git)) {
                    Write-Host "Error: Failed to install Git via zypper. Installation of Git aborted."
                }
            }
            default {
                Write-Host "Unsupported Linux distribution. Please install Git manually."
                exit 1
            }
        }
    }

    if (Test-GitInstalled) {
        Write-Host "Git is already installed."
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
        Write-Host "Unsupported operating system. Please install Git manually."
        exit 1
    }

    if ($osPlatform -like "*Windows*") {
        Install-GitWindows
    }
    elseif ($osPlatform -like "*Mac*") {
        Install-GitMac
    }
    elseif ($osPlatform -like "*Linux*") {
        Install-GitLinux
    }
    else {
        Write-Host "Unsupported operating system. Please install Git manually."
        exit 1
    }

    if (Test-GitInstalled) {
        Write-Host "Git installed successfully."
    }
    else {
        Write-Host 'Failed to install. Please ensure Git is installed and available in $PATH.'
        exit 1
    }
}

<#
.SYNOPSIS
    Installs the Visual Studio redistributables on Windows systems.

.DESCRIPTION
    This function downloads and installs the Visual Studio 2015, 2017, 2019, and 2022 redistributable on Windows systems. It checks for administrator privileges and prompts the user for elevation if necessary.

.PARAMETER Interactive
    A boolean value that indicates whether to prompt the user for choices during the installation process.

.EXAMPLE
    Install-VCRedistWindows -Interactive $true
#>
function Install-VCRedistWindows {
    $os = Get-OsInfo
    if ($os.Name -ne "Windows") {
        return
    }

    if (-not (Test-IsAdmin)) {
        Write-Host "Admin privileges are required to install Visual Studio redistributables. Please run this script as an administrator."
        exit 1
    }

    $vcRedistUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    $vcRedistInstallerName = "vc_redist.x64.exe"
    $downloadsFolder = "$env:USERPROFILE\Downloads"
    $installerPath = Join-Path -Path $downloadsFolder -ChildPath $vcRedistInstallerName

    if (-not (Test-Path $installerPath)) {
        try {
            Invoke-WebRequest -Uri $vcRedistUrl -OutFile $installerPath
        }
        catch {
            Write-Host "Failed to download Visual Studio redistributables. Please check your internet connection or provide a pre-downloaded installer."
            exit 1
        }
    }

    Start-Process -FilePath $installerPath -ArgumentList "/install", "/quiet", "/norestart" -Wait
    Remove-Item -Path $installerPath -Force
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
        if (-not $Parameters.NoSetup) {
            $requiredKeys = @("Dir", "Branch", "GitRepo")
            if (-not (Test-Value -Params $Parameters -RequiredKeys $requiredKeys)) {
                Write-Host "Error: Some required parameters are missing. Please provide values in the config file or through command line arguments."
                exit 1
            }
        }
    }

    process {
        if (-not $Parameters.NoSetup) {
            if (-not (Test-Python310Installed)) {
                Install-Python310 -Interactive:$Params.interactive
            }
            
            $installScope = Update-InstallScope($Interactive)
            Install-Python3Tk $installScope
        }
    }

    end {
        $pyExe = Get-PythonExePath
    
        if ($null -ne $pyExe) {
            $launcher = Join-Path -Path $Dir -ChildPath "launcher.py"

            # Check if launcher.py exists in the specified directory or in the script directory
            if (!(Test-Path -Path $launcher)) {
                $launcher = Join-Path -Path $PSScriptRoot -ChildPath "launcher.py"
                if (!(Test-Path -Path $launcher)) {
                    Write-Host "Error: launcher.py not found. Please ensure the file exists in the script directory or the specified directory."
                    exit 1
                }
            }

            Write-Host "Params: $($Parameters | Out-String)"

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

                Write-Host "Checking parameter: $key, value: $($Parameters[$key]), type: $($Parameters[$key].GetType().Name)"

                if ($Parameters[$key] -isnot [string]) {
                    # Only add the argument if it is true
                    if ($Parameters[$key] -eq $true) {
                        $installArgs.Add($argName) | Out-Null
                        Write-Host "Boolean argument: $argName"
                    }
                    elseif ($key -eq "verbosity") {
                        # Handle verbosity separately, as -vvvv or --verbosity 4
                        $verbosity = $Parameters[$key]
                        if ($verbosity -gt 0) {
                            # produces "-vvvv" for verbosity 4 or "--verbosity 4"
                            # Uncomment the line that suits your requirement
                            # $verbosityArg = "-" + ("v" * $verbosity)
                            $verbosityArg = "--verbosity"
                            $verbosityValue = [int]$verbosity
                            $installArgs.Add($verbosityArg) | Out-Null
                            $installArgs.Add($verbosityValue) | Out-Null
                        }
                    }
                }

                elseif (![string]::IsNullOrEmpty($Parameters[$key])) {
                    $installArgs.Add($argName) | Out-Null
                    $installArgs.Add($Parameters[$key]) | Out-Null
                }
            }


            # Call launcher.py with the appropriate parameters
            $command = "$pyExe -u $launcher $($installArgs -join ' ')"
            Write-Host "Running command: $command"
            & $pyExe -u "$launcher" $($installArgs.ToArray())
        }


        else {
            Write-Host "Error: Python 3.10 executable not found. Installation cannot proceed."
            exit 1
        }
    }
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
$Parameters = Get-Parameters -BoundParameters $PSBoundParameters
Main -Parameters $Parameters
