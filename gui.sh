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

# Define the directory path for WSL2
lib_path="/usr/lib/wsl/lib/"

# Check if the directory exists
if [ -d "$lib_path" ]; then
    # Check if LD_LIBRARY_PATH is already set
    if [ -z "${LD_LIBRARY_PATH}" ]; then
        # LD_LIBRARY_PATH is not set, set it to the lib_path
        export LD_LIBRARY_PATH="$lib_path"
        # echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
    fi
fi

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

# Check if conda environment is already activated
if [ -n "$CONDA_PREFIX" ]; then
    echo "Using existing conda environment: $CONDA_DEFAULT_ENV"
    echo "Conda environment path: $CONDA_PREFIX"
elif [ -d "$SCRIPT_DIR/venv" ]; then
    echo "Activating venv..."
    source "$SCRIPT_DIR/venv/bin/activate" || exit 1
else
    echo "No conda environment active and venv folder does not exist."
    echo "Please run setup.sh first or activate a conda environment."
    exit 1
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
        elif [ -x "$(command -v nvidia-smi)" ]; then
            REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_linux.txt"
        elif [[ "$@" == *"--use-rocm"* ]] || [ -x "$(command -v rocminfo)" ] || [ -f "/opt/rocm/bin/rocminfo" ]; then
            REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_linux_rocm.txt"
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
    if [[ -z "${DISABLE_VENV_LIBS}" ]]; then
        if [ -n "$CONDA_PREFIX" ]; then
            export LD_LIBRARY_PATH=$(realpath "$CONDA_PREFIX")/lib/:$LD_LIBRARY_PATH
        elif [ -d "$SCRIPT_DIR/venv" ]; then
            export LD_LIBRARY_PATH=$(realpath "$SCRIPT_DIR/venv")/lib/:$LD_LIBRARY_PATH
        fi
    fi
    if [[ -z "${NEOReadDebugKeys}" ]]; then
        export NEOReadDebugKeys=1
    fi
    if [[ -z "${ClDeviceGlobalMemSizeAvailablePercent}" ]]; then
        export ClDeviceGlobalMemSizeAvailablePercent=100
    fi
    if [[ -z "${SYCL_CACHE_PERSISTENT}" ]]; then
        export SYCL_CACHE_PERSISTENT=1
    fi
    if [[ -z "${PYTORCH_ENABLE_XPU_FALLBACK}" ]]; then
        export PYTORCH_ENABLE_XPU_FALLBACK=1
    fi
    if [[ ! -z "${IPEXRUN}" ]] && [ ${IPEXRUN}="True" ] && [ -x "$(command -v ipexrun)" ]
    then
        if [[ -z "$STARTUP_CMD" ]]
        then
            STARTUP_CMD=ipexrun
        fi
        if [[ -z "$STARTUP_CMD_ARGS" ]]
        then
            STARTUP_CMD_ARGS="--multi-task-manager taskset --memory-allocator tcmalloc"
        fi
    fi
fi

#Set STARTUP_CMD as normal python if not specified
if [[ -z "$STARTUP_CMD" ]]
then
    STARTUP_CMD=python
fi

"${STARTUP_CMD}" $STARTUP_CMD_ARGS "$SCRIPT_DIR/kohya_gui.py" "--requirements=""$REQUIREMENTS_FILE" "$@"
