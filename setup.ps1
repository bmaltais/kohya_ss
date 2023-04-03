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
#>

param (
    [string]$Branch = "master",
    [string]$Dir = "$env:USERPROFILE\kohya_ss",
    [string]$GitRepo = "https://github.com/kohya/kohya_ss.git",
    [switch]$Interactive = $false,
    [switch]$NoGitUpdate = $false,
    [switch]$Public = $false,
    [switch]$Runpod = $false,
    [switch]$SkipSpaceCheck = $false,
    [int]$Verbose = 0,
    [string]$File = (Join-Path -Path $PSScriptRoot -ChildPath "install_config.yaml")
)

function Test-Value {
    param (
        $Parameters
    )

    function Check-YamlModule {
        if (-not (Get-Module -ListAvailable -Name 'powershell-yaml')) {
            try {
                Install-Module -Name powershell-yaml -Scope CurrentUser -ErrorAction Stop
                Write-Host "powershell-yaml module installed successfully."
                return $true
            }
            catch {
                Write-Host "Warning: Unable to install powershell-yaml module. Please install it manually with:"
                Write-Host "Install-Module -Name powershell-yaml -Scope CurrentUser"
                Write-Host "Continuing without loading configuration file..."
                return $false
            }
        }
        return $true
    }

    if (Check-YamlModule) {
        if (Test-Path $Parameters.File) {
            if (Import-Module powershell-yaml) {
                $config = ConvertFrom-Yaml (Get-Content -Raw $Parameters.File)
                $defaultBranch = $config.arguments.Branch.default
                $defaultDir = $config.arguments.Dir.default
                $defaultGitRepo = $config.arguments.GitRepo.default
            }
            else {
                Write-Host "Error: Could not load YAML installation config file."
            }
        }
        else {
            Write-Host "Warning: YAML configuration file not found. Continuing with command line arguments or default values."
        }
    }

    # Check if gitRepo, Branch, and Dir have values
    if (-not $Parameters.GitRepo -or -not $Parameters.Branch -or -not $Parameters.Dir) {
        Write-Host "Error: gitRepo, Branch, and Dir must have a value. Please provide values in the config file or through command line arguments."
        return $false
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
        if (-not (Test-Value $Parameters)) {
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

        if ($pyExe -ne $null) {
            $installArgs = @("launcher.py")

            # Iterate through the parameters and add them to the installArgs array
            foreach ($key in $Parameters.Keys) {
                if ($null -ne $Parameters[$key]) {
                    $installArgs += "-$key", $Parameters[$key]
                }
            }

            & $pyExe $installArgs
        } else {
            Write-Host "Error: Python 3.10 executable not found. Installation cannot proceed."
            exit 1
        }
    }
}



Main -Parameters $PSBoundParameters

