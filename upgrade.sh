#!/bin/bash

# Check if there are any changes that need to be committed
if git status --short | grep -q "^[^ ?][^?]*"; then
    echo "There are changes that need to be committed. Please stash or undo your changes before running this script."
    exit 1
fi

# Pull the latest changes from the remote repository
git pull

# Activate the virtual environment
source venv/bin/activate

# Upgrade the required packages
pip install --use-pep517 --upgrade -r requirements.txt
