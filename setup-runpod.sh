#!/usr/bin/env bash

# This gets the directory the script is run from so pathing can work relative to the script where needed.
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

# Install tk and python3.10-venv
echo "Installing tk and python3.10-venv..."
apt update -y && apt install -y python3-tk python3.10-venv

# Check if the venv folder doesn't exist
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Creating venv..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

# Activate the virtual environment
echo "Activating venv..."
source "$SCRIPT_DIR/venv/bin/activate" || exit 1

# Run setup_linux.py script with platform requirements
echo "Running setup_linux.py..."
python "$SCRIPT_DIR/setup/setup_linux.py" --platform-requirements-file=requirements_runpod.txt

# Configure accelerate
echo "Configuring accelerate..."
cp "$SCRIPT_DIR/config_files/accelerate/runpod.yaml" "/root/.cache/huggingface/accelerate/default_config.yaml"
echo "To manually configure accelerate run: $SCRIPT_DIR/venv/bin/activate"
