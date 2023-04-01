#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Validate the requirements and store the exit code
python tools/validate_requirements.py --requirements requirements_macos.txt
exit_code=$?

# If the exit code is 0, run the kohya_gui.py script with the command-line arguments
if [ $exit_code -eq 0 ]; then
    python kohya_gui.py "$@"
fi
