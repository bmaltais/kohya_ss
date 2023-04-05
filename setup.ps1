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

.PARAMETER File
  The full path to the configuration file. If not provided, the script looks for an 'install_config.yaml' file in the script's directory.

.PARAMETER GitRepo
  You can optionally provide a git repo to check out for runpod installation. Useful for custom forks.

.PARAMETER Interactive
  Interactively configure accelerate instead of using default config file.

.PARAMETER NoGitUpdate
  Do not update kohya_ss repo. No git pull or clone operations.

.PARAMETER Public
  Expose public URL in runpod mode. Won't have an effect in other modes.

.PARAMETER Runpod
  Forces a runpod installation. Useful if detection fails for any reason.

.PARAMETER SkipSpaceCheck
  Skip the 10Gb minimum storage space check.

.PARAMETER Verbose
  Increase verbosity levels up to 3.

.PARAMETER GUI_LISTEN
  The IP address the GUI should listen on.

.PARAMETER GUI_USERNAME
  The username for the GUI.

.PARAMETER GUI_PASSWORD
  The password for the GUI.

.PARAMETER GUI_SERVER_PORT
  The port number the GUI server should use.

.PARAMETER GUI_INBROWSER
  Open the GUI in the default web browser.

.PARAMETER GUI_SHARE
  Share the GUI with other users on the network.
#>


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
   the function will use the default value specified within the function.

.PARAMETER File
   An optional parameter to specify the path to a configuration file.

.OUTPUTS
   System.Collections.Hashtable
   Outputs a hashtable containing the merged parameter values.
#>

function Get-Parameters {
    param (
        [string]$File = ""
    )

    # Check for the existence of the powershell-yaml module and install it if necessary
    if (-not (Get-Module -ListAvailable -Name 'powershell-yaml')) {
        Install-Module -Name 'powershell-yaml' -Scope CurrentUser -Force
    }
    Import-Module 'powershell-yaml'

    # Define possible configuration file locations
    $configFileLocations = if ($IsWindows) {
        @(
            $File,
            (Join-Path -Path $env:APPDATA -ChildPath "kohya_ss\config_files\installation\install_config.yaml"),
            (Join-Path -Path $env:LOCALAPPDATA -ChildPath "kohya_ss\install_config.yaml"),
            (Join-Path -Path $PSBoundParameters['Dir'] -ChildPath "install_config.yaml"),
            (Join-Path -Path "$env:USERPROFILE\kohya_ss" -ChildPath "install_config.yaml"),
            (Join-Path -Path $PSScriptRoot -ChildPath "install_config.yaml")
        )
    }
    else {
        @(
            $File,
            (Join-Path -Path $env:HOME -ChildPath ".kohya_ss\install_config.yaml"),
            (Join-Path -Path $PSBoundParameters['Dir'] -ChildPath "install_config.yaml"),
            (Join-Path -Path "$env:USERPROFILE\kohya_ss" -ChildPath "install_config.yaml"),
            (Join-Path -Path $PSScriptRoot -ChildPath "install_config.yaml")
        )
    }

    # Load the configuration file from the first existing location
    $Config = @{}
    foreach ($location in $configFileLocations) {
        if (Test-Path -Path $location) {
            $yamlContent = Get-Content -Path $location | ConvertFrom-Yaml
            foreach ($section in $yamlContent.Keys) {
                foreach ($key in $yamlContent[$section].Keys) {
                    if ($yamlContent[$section][$key].ContainsKey('default')) {
                        $Config["${section}_${key}"] = $yamlContent[$section][$key]['default']
                    }
                }
            }
            break
        }
    }

    # Define the default values
    $Defaults = @{
        'setup_branch'          = 'master'
        'setup_dir'             = "$env:USERPROFILE\kohya_ss"
        'setup_gitRepo'         = 'https://github.com/kohya/kohya_ss.git'
        'setup_interactive'     = $false
        'setup_gitUpdate'       = $false
        'setup_public'          = $false
        'setup_runpod'          = $false
        'setup_spaceCheck'      = $false
        'setup_verbosity'       = 0
        'gui_listen'            = '127.0.0.1'
        'gui_username'          = ''
        'gui_password'          = ''
        'gui_server_port'       = 7861
        'gui_inbrowser'         = $true
        'gui_share'             = $false
    }

    # Iterate through the default values and set them if not defined in the config file
    foreach ($key in $Defaults.Keys) {
        if (-not $Config.ContainsKey($key)) {
            $Config[$key] = $Defaults[$key]
        }
    }

    # Merge CLI arguments with the configuration
    $params = $PSBoundParameters.GetEnumerator() | Where-Object { $_.Key -ne "File" }
    foreach ($param in $params) {
        $Config[$param.Key] = $param.Value
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
        if ($null -eq $Params[$key]) {
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
                    return "sudo -S $(get-command $args[0]).source $($args[1..$($args.count)] -join ' ')"
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

    function Test-Python310Installed {
        try {
            $pythonVersion = (python --version) -replace '^Python\s', ''
            return $pythonVersion.StartsWith('3.10')
        }
        catch {
            return $false
        }
    }

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
            $os = Get-LinuxDistribution

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

    $os = Get-LinuxDistribution
    $elevate = Get-ElevationCommand

    if ($PSVersionTable.Platform -eq 'Unix') {
        # Linux / macOS installation
        switch ($os.family) {
            "Ubuntu" {
                # Ubuntu installation
                try {
                    & $elevate apt update
                    & $elevate apt install -y python3.10-tk
                }
                catch {
                    Write-Host "Error: Failed to install Python 3.10 Tk on Ubuntu. $_"
                }
            }
            "Debian" {
                # Debian installation
                try {
                    & $elevate apt-get update
                    & $elevate apt-get install -y python3.10-tk
                }
                catch {
                    Write-Host "Error: Failed to install Python 3.10 Tk on Debian. $_"
                }
            }
            "RedHat" {
                # Red Hat installation
                try {
                    & $elevate dnf install -y python3.10-tkinter
                }
                catch {
                    Write-Host "Error: Failed to install Python 3.10 Tk on Red Hat. $_"
                }
            }
            "Arch" {
                # Arch installation
                try {
                    & $elevate pacman -S --noconfirm tk
                }
                catch {
                    Write-Host "Error: Failed to install Python 3.10 Tk on Arch. $_"
                }
            }
            "openSUSE" {
                # openSUSE installation
                try {
                    & $elevate zypper install -y python3.10-tk
                }
                catch {
                    Write-Host "Error: Failed to install Python 3.10 Tk on openSUSE. $_"
                }
            }
            "macOS" {
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
            default {
                Write-Host "Unsupported Linux distribution. Please install Python 3.10 Tk manually."
            }
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

        if (Get-Command "scoop" -ErrorAction SilentlyContinue) {
            scoop install git
            $packageManagerFound = $true
        }
        elseif (Get-Command "choco" -ErrorAction SilentlyContinue) {
            choco install git
            $packageManagerFound = $true
        }
        elseif (Get-Command "winget" -ErrorAction SilentlyContinue) {
            winget install --id Git.Git
            $packageManagerFound = $true
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
        $os = Get-LinuxDistribution

        switch ($os.family) {
            "Ubuntu" {
                if (& $elevate apt-get update) {
                    if (!(& $elevate apt-get install -y git)) {
                        Write-Host "Error: Failed to install Git via apt. Installation of Git aborted."
                    }
                }
                else {
                    Write-Host "Error: Failed to update package list. Installation of Git aborted."
                }
            }
            "Debian" {
                if (& $elevate apt-get update) {
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
        $requiredKeys = @("Dir", "Branch", "GitRepo")
        if (-not (Test-Value -Params $Parameters -RequiredKeys $requiredKeys)) {
            Write-Host "Error: Some required parameters are missing. Please provide values in the config file or through command line arguments."
            exit 1
        }
    }

    process {
        if (-not (Test-Python310Installed)) {
            Install-Python310 -Interactive:$Parameters.Interactive
        }

        if ($PSVersionTable.Platform -eq 'Unix') {
            $linuxDistribution = Get-LinuxDistribution
        }
        else {
            $linuxDistribution = $null
        }

        Install-Python3Tk -LinuxDistribution $linuxDistribution
    }

    end {
        $pyExe = Get-PythonExePath

        if ($null -ne $pyExe) {
            $launcher = Join-Path -Path $Dir -ChildPath "launcher.py"

            $installArgs = @()
            foreach ($key in $Params.Keys) {
                $argName = "--$(($key.ToLowerInvariant() -replace '[A-Z]', '-$0').ToLower())"
                $installArgs += $argName, $Params[$key]
            }

            # Call launcher.py with the appropriate parameters
            & $pyExe -u "$launcher" $installArgs
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
$Params = Get-Parameters
Main -Parameters $Params

