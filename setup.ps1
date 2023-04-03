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


function Get-Parameters {
    param (
        [string]$File = ""
    )

    # Define possible configuration file locations
    $configFileLocations = if ($IsWindows) {
        @(
            $File,
            (Join-Path -Path $env:APPDATA -ChildPath "kohya_ss\install_config.yaml"),
            (Join-Path -Path $env:LOCALAPPDATA -ChildPath "kohya_ss\install_config.yaml"),
            (Join-Path -Path $PSBoundParameters['Dir'] -ChildPath "install_config.yaml"),
            (Join-Path -Path "$env:USERPROFILE\kohya_ss" -ChildPath "install_config.yaml"),
            (Join-Path -Path $PSScriptRoot -ChildPath "install_config.yaml")
        )
    } else {
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
            $Config = Get-Content -Path $location | ConvertFrom-Yaml
            break
        }
    }

    # Define the default values
    $Defaults = @{
        'Branch' = 'master'
        'Dir' = "$env:USERPROFILE\kohya_ss"
        'GitRepo' = 'https://github.com/kohya/kohya_ss.git'
        'Interactive' = $false
        'NoGitUpdate' = $false
        'Public' = $false
        'Runpod' = $false
        'SkipSpaceCheck' = $false
        'Verbose' = 0
        'GUI_LISTEN' = '127.0.0.1'
        'GUI_USERNAME' = ''
        'GUI_PASSWORD' = ''
        'GUI_SERVER_PORT' = 8080
        'GUI_INBROWSER' = $true
        'GUI_SHARE' = $false
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


function Test-IsAdmin {
    if ($PSVersionTable.Platform -eq 'Win32NT') {
        $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object System.Security.Principal.WindowsPrincipal($identity)
        $isAdmin = $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
        return $isAdmin
    }
    else {
        try {
            $userId = (id -u)
            $result = ($userId -eq 0)
        }
        catch {
            $result = $false
        }
        return $result
    }
}

function Get-LinuxDistribution {
    if (Test-Path "/etc/os-release") {
        $os_release = Get-Content "/etc/os-release"
        if ($os_release -match "Ubuntu") {
            return "Ubuntu"
        }
        elseif ($os_release -match "Red Hat") {
            return "RedHat"
        }
    }
    elseif (Test-Path "/etc/redhat-release") {
        return "RedHat"
    }
    else {
        return "Unknown"
    }
}


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
        }
        else {
            $installScope = "user"
        }

        $pythonUrl = "https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe"
        $$pythonInstallerName = "python-3.10.0-amd64.exe"
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
        } else {
            Start-Process $installerPath -ArgumentList "/passive InstallAllUsers=1" -Wait
        }

        Remove-Item $installerPath
        }
    }

    function Install-Python310Mac {
        if (Get-Command "brew" -ErrorAction SilentlyContinue) {
            brew install python@3.10
            brew link --overwrite --force python@3.10
        }
        else {
            Write-Host "Please install Homebrew first to continue with Python 3.10 installation."
        }
    }

    function Install-Python310Linux {
        $distribution = Get-LinuxDistribution

        if ($distribution -eq "Ubuntu") {
            if (sudo apt-get update) {
                sudo apt-get install -y python3.10
            } else {
                Write-Host "Error: Failed to update package list. Installation of Python 3.10 aborted."
            }
        }
        elseif ($distribution -eq "RedHat") {
            sudo dnf install -y python3.10
        }
        else {
            Write-Host "Unsupported Linux distribution. Please install Python 3.10 manually."
        }
    }

    if (Test-Python310Installed) {
        Write-Host "Python 3.10 is already installed."
        return
    }

    $osPlatform = ""
    if ($PSVersionTable.Platform -eq 'Windows' -or $PSVersionTable.PSEdition -eq 'Desktop') {
        $osPlatform = (Get-WmiObject -Class Win32_OperatingSystem).Caption
    } elseif ($PSVersionTable.Platform -eq 'Unix') {
        $osPlatform = (uname -s).ToString()
    } elseif ($PSVersionTable.Platform -eq 'MacOS') {
        $osPlatform = (uname -s).ToString()
    } else {
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
        Write-Host "Failed to install"
    }
}

function Install-Python3Tk {
    if (Test-IsAdmin) {
        Write-Host "Running as an administrator. Installing packages."

        if ($PSVersionTable.Platform -eq 'Unix') {
            # Linux / macOS installation
            $distribution = Get-LinuxDistribution
                if ($distribution -eq "Ubuntu") {
                    # Ubuntu installation
                    Invoke-Expression "apt-get update"
                    Invoke-Expression "apt-get install -y python3.10-tk"
                }
                elseif ($distribution -eq "RedHat") {
                    # Red Hat installation
                    Invoke-Expression "yum install -y python3.10-tkinter"
                }
            }
            elseif (Test-Path "/usr/local/bin/brew") {
                # macOS installation using Homebrew
                Invoke-Expression "brew install python-tk@3.10"
            }
            else {
                Write-Host "Unsupported Unix platform or package manager not found."
            }
        }
        else {
            # Windows installation
            if (Test-Path "C:\Python310\python.exe") {
                Write-Host "Python 3.10 is already installed."
            }
            else {
                Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe" -OutFile "python-3.10.0-amd64.exe"
                Start-Process -FilePath "python-3.10.0-amd64.exe" -ArgumentList "/passive InstallAllUsers=1 PrependPath=1 Include_tcltk=1" -Wait
                Remove-Item "python-3.10.0-amd64.exe"
        }
    }
    else {
        Write-Host "This script needs to be run as an administrator or via 'Run as administrator' to install packages."
        exit 1
    }
}

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
        } else {
            $linuxDistribution = $null
        }

        Install-Python3Tk -LinuxDistribution $linuxDistribution
    }

    end {
        $pyExe = Get-PythonExePath

        if ($pythonExec -ne $null) {
            $$launcher = Join-Path -Path $Dir -ChildPath "launcher.py"

            $installArgs = @()
            foreach ($key in $Params.Keys) {
                $argName = "--$(($key.ToLowerInvariant() -replace '[A-Z]', '-$0').ToLower())"
                $installArgs += $argName, $Params[$key]
            }

            # Call launcher.py with the appropriate parameters
            & $pythonExec -u "$launcher" $installArgs
        } else {
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

