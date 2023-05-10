#!/usr/bin/env bash

# This gets the directory the script is run from so pathing can work relative to the script where needed.
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

# Step into GUI local directory
cd "$SCRIPT_DIR"

# Activate the virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# If the requirements are validated, run the kohya_gui.py script with the command-line arguments
if python  "$SCRIPT_DIR"/tools/validate_requirements.py -r "$SCRIPT_DIR"/requirements.txt; then
    python "$SCRIPT_DIR/kohya_gui.py" "$@"
fi
