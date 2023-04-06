#!/usr/bin/env bash

# Set root dir for the kohya-ss and change working directory
KOHYA_SS_DIR=$(dirname "$(readlink -f -- "$0")")
cd $KOHYA_SS_DIR

# Activate the virtual environment
source "$KOHYA_SS_DIR"/venv/bin/activate

# If the requirements are validated, run the kohya_gui.py script with the command-line arguments
if python "$KOHYA_SS_DIR"/tools/validate_requirements.py -r $KOHYA_SS_DIR/requirements.txt; then
    python "$KOHYA_SS_DIR"/kohya_gui.py "$@"
fi

# step back
cd -
