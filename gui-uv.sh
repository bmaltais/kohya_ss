#!/usr/bin/env bash
export VIRTUAL_ENV=.venv

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

# Check if uv is already installed
if ! command -v uv &> /dev/null; then
  read -p "uv is not installed. We can try to install it for you, or you can install it manually from https://astral.sh/uv before running this script again. Would you like to attempt automatic installation now? [Y/n]: " install_uv
  if [[ "$install_uv" =~ ^[Yy]$ ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
  else
    echo "Okay, please install uv manually from https://astral.sh/uv and then re-run this script. Exiting."
    exit 1
  fi
fi

git submodule update --init --recursive
uv run kohya_gui.py --noverify "$@"
