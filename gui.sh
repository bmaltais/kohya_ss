#!/usr/bin/env bash

# Checks to see if variable is set and non-empty.
# This is defined first, so we can use the function for some default variable values
env_var_exists() {
  if [[ -n "${!1}" ]]; then
    return 0
  else
    return 1
  fi
}

# Need RUNPOD to have a default value before first access
RUNPOD=false
if env_var_exists RUNPOD_POD_ID || env_var_exists RUNPOD_API_KEY; then
  RUNPOD=true
fi

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

if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate" || exit 1
else
    echo "venv folder does not exist. Not activating..."
fi

# Check if LD_LIBRARY_PATH environment variable exists
if [[ -z "${LD_LIBRARY_PATH}" ]]; then
    # Set the ANSI escape sequence for yellow text
    YELLOW='\033[0;33m'
    # Set the ANSI escape sequence to reset text color
    RESET='\033[0m'
    
    echo -e "${YELLOW}Warning: LD_LIBRARY_PATH environment variable is not set.${RESET}"
    echo -e "${YELLOW}Certain functionalities may not work correctly.${RESET}"
    echo -e "${YELLOW}Please ensure that the required libraries are properly configured.${RESET}"
    echo -e " "
    echo -e "${YELLOW}If you use WSL2 you may want to: export LD_LIBRARY_PATH=/usr/lib/wsl/lib/${RESET}"
    echo -e " "
fi

# Determine the requirements file based on the system
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$(uname -m)" == "arm64" ]]; then
        REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_macos_arm64.txt"
    else
        REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_macos_amd64.txt"
    fi
else
    if [ "$RUNPOD" = false ]; then
        if [[ "$@" == *"--use-ipex"* ]]; then
            REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_linux_ipex.txt"
        else
            REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_linux.txt"
        fi
    else
        REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_runpod.txt"
    fi
fi

#Set OneAPI if it's not set by the user
if [[ "$@" == *"--use-ipex"* ]]
then
    echo "Setting OneAPI environment"
    if [ ! -x "$(command -v sycl-ls)" ]
    then
        if [[ -z "$ONEAPI_ROOT" ]]
        then
            ONEAPI_ROOT=/opt/intel/oneapi
        fi
        source $ONEAPI_ROOT/setvars.sh
    fi
    export NEOReadDebugKeys=1
    export ClDeviceGlobalMemSizeAvailablePercent=100
    if [[ -z "$STARTUP_CMD" ]] && [[ -z "$DISABLE_IPEXRUN" ]] && [ -x "$(command -v ipexrun)" ]
    then
        STARTUP_CMD=ipexrun
        if [[ -z "$STARTUP_CMD_ARGS" ]]
        then
            STARTUP_CMD_ARGS="--multi-task-manager taskset --memory-allocator jemalloc"
        fi
    fi
fi

#Set STARTUP_CMD as normal python if not specified
if [[ -z "$STARTUP_CMD" ]]
then
    STARTUP_CMD=python
fi

# Validate the requirements and run the script if successful
if python "$SCRIPT_DIR/setup/validate_requirements.py" -r "$REQUIREMENTS_FILE"; then
    "${STARTUP_CMD}" $STARTUP_CMD_ARGS "$SCRIPT_DIR/kohya_gui.py" "$@"
fi
