#!/usr/bin/env bash

# This gets the directory the script is run from so pathing can work relative to the script where needed.
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

# Install tk and python3.10-venv
echo "Installing tk and python3.10-venv..."
apt update -y && apt install -y python3-tk python3.10-venv

# Install required libcudnn release 8.7.0.84-1
echo "Installing required libcudnn release 8.7.0.84-1..."
apt install -y libcudnn8=8.7.0.84-1+cuda11.8 libcudnn8-dev=8.7.0.84-1+cuda11.8 --allow-change-held-packages

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
python "$SCRIPT_DIR/setup/setup_linux.py" --platform-requirements-file=requirements_runpod.txt --show_stdout --no_run_accelerate
pip3 cache purge

# Configure accelerate
echo "Configuring accelerate..."
mkdir -p "/root/.cache/huggingface/accelerate"
cp "$SCRIPT_DIR/config_files/accelerate/runpod.yaml" "/root/.cache/huggingface/accelerate/default_config.yaml"

echo "Installation completed... You can start the gui with ./gui.sh --share --headless"

# Deactivate the virtual environment
echo "Deactivating venv..."
deactivate