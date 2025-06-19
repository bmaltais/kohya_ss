#!/usr/bin/env bash
set -e
export VIRTUAL_ENV=.venv

env_var_exists() {
  if [[ -n "${!1}" ]]; then
    return 0
  else
    return 1
  fi
}

# Handle LD_LIBRARY_PATH for WSL environments and preserve it when using sudo,
# as these can affect how shared libraries are found.
lib_path="/usr/lib/wsl/lib/"

if [ -d "$lib_path" ]; then
    if [ -z "${LD_LIBRARY_PATH}" ]; then
        export LD_LIBRARY_PATH="$lib_path"
    fi
fi

if [ -n "$SUDO_USER" ] || [ -n "$SUDO_COMMAND" ]; then
    echo "The sudo command resets the non-essential environment variables, we keep the LD_LIBRARY_PATH variable."
    export LD_LIBRARY_PATH=$(sudo -i printenv LD_LIBRARY_PATH)
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1

# Check if --quiet is in the arguments
uv_quiet=""
args=()
for arg in "$@"; do
  if [[ "$arg" == "--quiet" ]]; then
    uv_quiet="--quiet"
  else
    args+=("$arg")
  fi
done

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

echo "Ensuring virtual environment .venv is set up..."
uv venv .venv || { echo "Failed to create or set up the virtual environment. Exiting."; exit 1; }

if [[ "$uv_quiet" == "--quiet" ]]; then
  echo "Notice: uv will run in quiet mode. No indication of the uv module download and install process will be displayed."
fi

git submodule update --init --recursive || { echo "ERROR: Failed to update git submodules. Please check for errors above and try again." >&2; exit 1; }
uv run $uv_quiet kohya_gui.py --noverify "${args[@]}"
