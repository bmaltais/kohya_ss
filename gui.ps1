# Activate the virtual environment
& .\venv\Scripts\activate

# Validate the requirements and store the exit code
python.exe .\tools\validate_requirements.py

# If the exit code is 0, run the kohya_gui.py script with the command-line arguments
if ($LASTEXITCODE -eq 0) {
    python.exe kohya_gui.py $args
}