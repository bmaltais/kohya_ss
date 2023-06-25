#!/usr/bin/env bash

# This gets the directory the script is run from so pathing can work relative to the script where needed.
SCRIPT_DIR="$(cd -- $(dirname -- "$0") && pwd)"

# Check if the venv folder doesn't exist
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
fi

# Install tk
echo "Install tk..."
apt update -y && apt install -y python3-tk

# Activate the virtual environment
echo "Activate venv..."
source "$SCRIPT_DIR/venv/bin/activate" || exit 1

# Run setup_linux.py script with platform requirements
python "$SCRIPT_DIR/setup/setup_linux.py" --platform-requirements-file=requirements_runpod.txt
