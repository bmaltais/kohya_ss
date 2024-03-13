if (-not (Test-Path -Path "venv")) {
    Write-Host "Creating venv..."
    python -m venv venv
}

# Create the directory if it doesn't exist
$null = New-Item -ItemType Directory -Force -Path ".\logs\setup"

# Deactivate the virtual environment
& .\venv\Scripts\deactivate.bat

& .\venv\Scripts\activate.bat

& .\venv\Scripts\python.exe .\setup\setup_windows.py $args

# Deactivate the virtual environment
& .\venv\Scripts\deactivate.bat
