# Activate the virtual environment
& .\venv\Scripts\activate
$env:PATH += ";$($MyInvocation.MyCommand.Path)\venv\Lib\site-packages\torch\lib"

# Debug info about system
python.exe .\tools\debug_info.py

# Validate the requirements and store the exit code
python.exe .\tools\validate_requirements.py

# If the exit code is 0, read arguments from gui_parameters.txt (if it exists)
# and run the kohya_gui.py script with the command-line arguments
if ($LASTEXITCODE -eq 0) {
    $argsFromFile = @()
    if (Test-Path .\gui_parameters.txt) {
        $argsFromFile = Get-Content .\gui_parameters.txt -Encoding UTF8 | Where-Object { $_ -notmatch "^#" } | Foreach-Object { $_ -split " " }
    }
    $args_combo = $argsFromFile + $args
    Write-Host "The arguments passed to this script were: $args_combo"
    python.exe kohya_gui.py $args_combo
}
