#!/usr/bin/env bash

# If it is run with the sudo command, get the complete LD_LIBRARY_PATH environment variable of the system and assign it to the current environment,
# because it will be used later.
if [ -n "$SUDO_USER" ] || [ -n "$SUDO_COMMAND" ]; then
    echo "The sudo command resets the non-essential environment variables, we keep the LD_LIBRARY_PATH variable."
    export LD_LIBRARY_PATH=$(sudo -i printenv LD_LIBRARY_PATH)
fi

# This gets the directory the script is run from so pathing can work relative to the script where needed.
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

# Step into GUI local directory
cd "$SCRIPT_DIR" || exit 1

# Activate the virtual environment
source "$SCRIPT_DIR/venv/bin/activate" || exit 1

# Check if LD_LIBRARY_PATH environment variable exists
if [[ -z "${LD_LIBRARY_PATH}" ]]; then
    echo "Warning: LD_LIBRARY_PATH environment variable is not set."
    echo "Certain functionalities like 8bit based optimizers may not work correctly."
    echo "Please ensure that the required libraries are properly configured."
fi

# Determine the requirements file based on the system
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$(uname -m)" == "arm64" ]]; then
        REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_macos_arm64.txt"
    else
        REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_macos_amd64.txt"
    fi
else
    REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_linux.txt"
fi

# Validate the requirements and run the script if successful
if python "$SCRIPT_DIR/setup/validate_requirements.py" -r "$REQUIREMENTS_FILE"; then
    python "$SCRIPT_DIR/kohya_gui.py" "$@"
fi
