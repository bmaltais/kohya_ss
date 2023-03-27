#!/bin/bash

# Check if there are any changes that need to be committed
if [[ -n $(git status --short) ]]; then
    echo "There are changes that need to be committed. Please stash or undo your changes before running this script." >&2
    exit 1
fi

# Pull the latest changes from the remote repository
git pull

# Activate the virtual environment
source venv/bin/activate

# Upgrade the required packages
pip install --upgrade -r requirements_macos.txt
