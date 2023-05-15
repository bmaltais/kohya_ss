#!/usr/bin/env bash

# If it is run with the sudo command, get the complete LD_LIBRARY_PATH environment variable of the system and assign it to the current environment,
# because it will be used later.
if [ -n "$SUDO_USER" ] || [ -n "$SUDO_COMMAND" ] ; then
    echo "The sudo command resets the non-essential environment variables, we keep the LD_LIBRARY_PATH variable."
    export LD_LIBRARY_PATH=$(sudo -i printenv LD_LIBRARY_PATH)
fi

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
