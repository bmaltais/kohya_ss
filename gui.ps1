# Check if a virtual environment is active and deactivate it if necessary
if ($env:VIRTUAL_ENV) {
    # Write-Host "Deactivating the virtual environment to test for modules installed locally..."
    & deactivate
}

# Run pip freeze and capture the output
$pipOutput = & pip freeze

# Check if modules are found in the output
if ($pipOutput) {
    Write-Host " "
    Write-Host -ForegroundColor Yellow -Object "============================================================="
    Write-Host -ForegroundColor Yellow -Object "Modules installed outside the virtual environment were found."
    Write-Host -ForegroundColor Yellow -Object "This can cause issues. Please review the installed modules."
    Write-Host " "
    Write-Host -ForegroundColor Yellow -Object "You can deinstall all the local modules with:"
    Write-Host " "
    Write-Host -ForegroundColor Blue -Object "deactivate"
    Write-Host -ForegroundColor Blue -Object "pip freeze > uninstall.txt"
    Write-Host -ForegroundColor Blue -Object "pip uninstall -y -r uninstall.txt"
    Write-Host -ForegroundColor Yellow -Object "============================================================="
    Write-Host " "
} 

# Activate the virtual environment
# Write-Host "Activating the virtual environment..."
& .\venv\Scripts\activate
$env:PATH += ";$($MyInvocation.MyCommand.Path)\venv\Lib\site-packages\torch\lib"

# Debug info about system
# python.exe .\setup\debug_info.py

# Validate the requirements and store the exit code
python.exe .\setup\validate_requirements.py

# If the exit code is 0, read arguments from gui_parameters.txt (if it exists)
# and run the kohya_gui.py script with the command-line arguments
if ($LASTEXITCODE -eq 0) {
    $argsFromFile = @()
    if (Test-Path .\gui_parameters.txt) {
        $argsFromFile = Get-Content .\gui_parameters.txt -Encoding UTF8 | Where-Object { $_ -notmatch "^#" } | Foreach-Object { $_ -split " " }
    }
    $args_combo = $argsFromFile + $args
    # Write-Host "The arguments passed to this script were: $args_combo"
    python.exe kohya_gui.py $args_combo
}
