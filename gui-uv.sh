#!/usr/bin/env bash
export VIRTUAL_ENV=.venv

env_var_exists() {
  if [[ -n "${!1}" ]]; then
    return 0
  else
    return 1
  fi
}

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

if [[ "$uv_quiet" == "--quiet" ]]; then
  echo "Notice: uv will run in quiet mode. No indication of the uv module download and install process will be displayed."
fi

git submodule update --init --recursive

# --- bitsandbytes compilation (optional, similar to Colab setup) ---
# The pyproject.toml specifies bitsandbytes, which uv will install.
# However, if you encounter issues with the pre-compiled version,
# you might need to compile it from source, similar to the Colab notebook.
# Uncomment and adapt the following lines if needed.
# Ensure you have the necessary build tools (like CUDA toolkit for cuda11x).

# BUILD_BITSANDBYTES_FROM_SOURCE=false # Set to true to enable
# if [ "$BUILD_BITSANDBYTES_FROM_SOURCE" = true ]; then
#   echo "Attempting to build bitsandbytes from source..."
#   if [ ! -d "bitsandbytes" ]; then
#     git clone -b 0.41.0 https://github.com/TimDettmers/bitsandbytes
#   fi
#   cd bitsandbytes || exit 1
#   # IMPORTANT: Adjust CUDA_VERSION if necessary for your setup (e.g., 118 for CUDA 11.8, 12x for CUDA 12.x)
#   # The Colab notebook used CUDA_VERSION=118
#   # Common options: cuda11x (for 11.0-11.8), cuda12x (for 12.0-12.x)
#   # Ensure the corresponding CUDA toolkit is installed and in PATH.
#   # For ROCm, the build process is different.
#   # Check bitsandbytes documentation for the correct make target for your GPU/driver.
#   # Example for CUDA 11.8:
#   # CUDA_VERSION=118 make cuda11x
#   # Example for CUDA 12.1:
#   # make cuda12x
#   # Then, install using uv within the environment (or pip if uv is not active yet for this part)
#   # Assuming uv environment is active or will be activated by the main command:
#   uv pip install .
#   # Or, if building before activating the main uv environment:
#   # python setup.py install
#   cd .. || exit 1
#   echo "bitsandbytes build attempt finished."
# fi
# --- end of bitsandbytes compilation ---

echo "Launching Kohya GUI via uv..."
uv run $uv_quiet kohya_gui.py --noverify "${args[@]}"
