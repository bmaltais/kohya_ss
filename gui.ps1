# Check if a virtual environment is active and deactivate it if necessary
if ($env:VIRTUAL_ENV) {
    # Write-Host "Deactivating the virtual environment to test for modules installed locally..."
    & deactivate
}

# Activate the virtual environment
# Write-Host "Activating the virtual environment..."
& .\venv\Scripts\activate

python.exe -m pip install --upgrade pip -q

$env:PATH += ";$($MyInvocation.MyCommand.Path)\venv\Lib\site-packages\torch\lib"

Write-Host "Starting the GUI... this might take some time..."

$argsFromFile = @()
if (Test-Path .\gui_parameters.txt) {
    $argsFromFile = Get-Content .\gui_parameters.txt -Encoding UTF8 | Where-Object { $_ -notmatch "^#" } | Foreach-Object { $_ -split " " }
}
$args_combo = $argsFromFile + $args
# Write-Host "The arguments passed to this script were: $args_combo"
python.exe kohya_gui.py $args_combo

