
# Check if Python version meets the recommended version
$pythonVersion = & .\venv\Scripts\python.exe --version 2>$null
if ($pythonVersion -notmatch "^Python $PYTHON_VER") {
    Write-Host "Warning: Python version $PYTHON_VER is recommended."
}

if (-not (Test-Path -Path "venv")) {
    Write-Host "Creating venv..."
    python -m venv venv
}

# Create the directory if it doesn't exist
$null = New-Item -ItemType Directory -Force -Path ".\logs\setup"

# Deactivate the virtual environment
& .\venv\Scripts\deactivate.bat

# Calling external python program to check for local modules
& .\venv\Scripts\python.exe .\setup\check_local_modules.py

& .\venv\Scripts\activate.bat

& .\venv\Scripts\python.exe .\setup\setup_windows.py

# Deactivate the virtual environment
& .\venv\Scripts\deactivate.bat
