#!/usr/bin/env bash

# Activate the virtual environment
source ./venv/bin/activate
python -V
# If the requirements are validated, run the kohya_gui.py script with the command-line arguments
if python tools/validate_requirements.py; then
    python kohya_gui.py "$@"
fi
