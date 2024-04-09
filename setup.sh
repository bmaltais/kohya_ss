#!/usr/bin/env bash

# Function to display help information
display_help() {
  cat <<EOF
Kohya_SS Installation Script for POSIX operating systems.

Usage:
  # Specifies custom branch, install directory, and git repo
  setup.sh -b dev -d /workspace/kohya_ss -g https://mycustom.repo.tld/custom_fork.git

  # Same as example 1, but uses long options
  setup.sh --branch=dev --dir=/workspace/kohya_ss --git-repo=https://mycustom.repo.tld/custom_fork.git

  # Maximum verbosity, fully automated installation in a runpod environment skipping the runpod env checks
  setup.sh -vvv --skip-space-check --runpod

Options:
  -b BRANCH, --branch=BRANCH    Select which branch of kohya to check out on new installs.
  -d DIR, --dir=DIR             The full path you want kohya_ss installed to.
  -g REPO, --git_repo=REPO      You can optionally provide a git repo to check out for runpod installation. Useful for custom forks.
  -h, --help                    Show this screen.
  -i, --interactive             Interactively configure accelerate instead of using default config file.
  -n, --no-git-update           Do not update kohya_ss repo. No git pull or clone operations.
  -p, --public                  Expose public URL in runpod mode. Won't have an effect in other modes.
  -r, --runpod                  Forces a runpod installation. Useful if detection fails for any reason.
  -s, --skip-space-check        Skip the 10Gb minimum storage space check.
  -u, --no-gui                  Skips launching the GUI.
  -v, --verbose                 Increase verbosity levels up to 3.
      --use-ipex                Use IPEX with Intel ARC GPUs.
      --use-rocm                Use ROCm with AMD GPUs.
EOF
}

# Helper function to check if variable is set and non-empty
env_var_exists() {
  if [[ -n "${!1}" ]]; then
    return 0
  else
    return 1
  fi
}

# Check if RUNPOD variable should be set
RUNPOD=false
if env_var_exists RUNPOD_POD_ID || env_var_exists RUNPOD_API_KEY; then
  RUNPOD=true
fi

# Directory of the script
SCRIPT_DIR="$(cd -- $(dirname -- "$0") && pwd)"

# Variables defined before the getopts loop, so we have sane default values.
# Default installation locations based on OS and environment
if [[ "$OSTYPE" == "lin"* ]]; then
  if [ "$RUNPOD" = true ]; then
    DIR="/workspace/kohya_ss"
  elif [ -d "$SCRIPT_DIR/.git" ]; then
    DIR="$SCRIPT_DIR"
  elif [ -w "/opt" ]; then
    DIR="/opt/kohya_ss"
  elif env_var_exists HOME; then
    DIR="$HOME/kohya_ss"
  else
    # The last fallback is simply PWD
    DIR="$(PWD)"
  fi
else
  if [ -d "$SCRIPT_DIR/.git" ]; then
    DIR="$SCRIPT_DIR"
  elif env_var_exists HOME; then
    DIR="$HOME/kohya_ss"
  else
    # The last fallback is simply PWD
    DIR="$(PWD)"
  fi
fi

# Variables
BRANCH="master"
GIT_REPO="https://github.com/bmaltais/kohya_ss.git"
INTERACTIVE=false
PUBLIC=false
SKIP_SPACE_CHECK=false
SKIP_GIT_UPDATE=true
SKIP_GUI=false
VERBOSITY=2
MAXVERBOSITY=6
DIR=""
PARENT_DIR=""
VENV_DIR=""
USE_IPEX=false
USE_ROCM=false

# Function to get the distro name
get_distro_name() {
  local line
  if [ -f /etc/os-release ]; then
    # We search for the line starting with ID=
    # Then we remove the ID= prefix to get the name itself
    line="$(grep -Ei '^ID=' /etc/os-release)"
    echo "Raw detected os-release distro line: $line" >&5
    line=${line##*=}
    echo "$line"
    return 0
  elif command -v python >/dev/null; then
    line="$(python -mplatform)"
    echo "$line"
    return 0
  elif command -v python3 >/dev/null; then
    line="$(python3 -mplatform)"
    echo "$line"
    return 0
  else
    line="None"
    echo "$line"
    return 1
  fi
}

# Function to get the distro family
get_distro_family() {
  local line
  if [ -f /etc/os-release ]; then
    if grep -Eiq '^ID_LIKE=' /etc/os-release >/dev/null; then
      line="$(grep -Ei '^ID_LIKE=' /etc/os-release)"
      echo "Raw detected os-release distro family line: $line" >&5
      line=${line##*=}
      echo "$line"
      return 0
    else
      line="None"
      echo "$line"
      return 1
    fi
  else
    line="None"
    echo "$line"
    return 1
  fi
}

# Function to check available storage space
check_storage_space() {
  if [ "$SKIP_SPACE_CHECK" = false ]; then
    if [ "$(size_available)" -lt 10 ]; then
      echo "You have less than 10Gb of free space. This installation may fail."
      MSGTIMEOUT=10 # In seconds
      MESSAGE="Continuing in..."
      echo "Press control-c to cancel the installation."
      for ((i = MSGTIMEOUT; i >= 0; i--)); do
        printf "\r${MESSAGE} %ss. " "${i}"
        sleep 1
      done
    fi
  fi
}

# Function to create symlinks
create_symlinks() {
  local symlink="$1"
  local target_file="$2"

  echo "Checking symlinks now."

  # Check if the symlink exists
  if [ -L "$symlink" ]; then
    # Check if the linked file exists and points to the expected file
    if [ -e "$symlink" ] && [ "$(readlink "$symlink")" == "$target_file" ]; then
      echo "$(basename "$symlink") symlink looks fine. Skipping."
    else
      if [ -f "$target_file" ]; then
        echo "Broken symlink detected. Recreating $(basename "$symlink")."
        rm "$symlink" && ln -s "$target_file" "$symlink"
      else
        echo "$target_file does not exist. Nothing to link."
      fi
    fi
  else
    echo "Linking $(basename "$symlink")."
    ln -s "$target_file" "$symlink"
  fi
}

# Function to install Python dependencies
install_python_dependencies() {
  local TEMP_REQUIREMENTS_FILE

  # Switch to local virtual env
  echo "Switching to virtual Python environment."
  if ! inDocker; then
    if command -v python3.10 >/dev/null; then
      python3.10 -m venv "$DIR/venv"
    elif command -v python3 >/dev/null; then
      python3 -m venv "$DIR/venv"
    else
      echo "Valid python3 or python3.10 binary not found."
      echo "Cannot proceed with the python steps."
      return 1
    fi

    # Activate the virtual environment
    source "$DIR/venv/bin/activate"
  fi

  case "$OSTYPE" in
    "lin"*)
      if [ "$RUNPOD" = true ]; then
        python "$SCRIPT_DIR/setup/setup_linux.py" --platform-requirements-file=requirements_runpod.txt
      elif [ "$USE_IPEX" = true ]; then
        python "$SCRIPT_DIR/setup/setup_linux.py" --platform-requirements-file=requirements_linux_ipex.txt
      elif [ "$USE_ROCM" = true ] || [ -x "$(command -v rocminfo)" ] || [ -f "/opt/rocm/bin/rocminfo" ]; then
        python "$SCRIPT_DIR/setup/setup_linux.py" --platform-requirements-file=requirements_linux_rocm.txt
      else
        python "$SCRIPT_DIR/setup/setup_linux.py" --platform-requirements-file=requirements_linux.txt
      fi
      ;;
    "darwin"*)
      if [[ "$(uname -m)" == "arm64" ]]; then
        python "$SCRIPT_DIR/setup/setup_linux.py" --platform-requirements-file=requirements_macos_arm64.txt
      else
        python "$SCRIPT_DIR/setup/setup_linux.py" --platform-requirements-file=requirements_macos_amd64.txt
      fi
      ;;
  esac

  if [ -n "$VIRTUAL_ENV" ] && ! inDocker; then
    if command -v deactivate >/dev/null; then
      echo "Exiting Python virtual environment."
      deactivate
    else
      echo "deactivate command not found. Could still be in the Python virtual environment."
    fi
  fi
}

# Function to configure accelerate
configure_accelerate() {
  echo "Source accelerate config location: $DIR/config_files/accelerate/default_config.yaml" >&3
  if [ "$INTERACTIVE" = true ]; then
    accelerate config
  else
    if env_var_exists HF_HOME; then
      if [ ! -f "$HF_HOME/accelerate/default_config.yaml" ]; then
        mkdir -p "$HF_HOME/accelerate/" &&
          echo "Target accelerate config location: $HF_HOME/accelerate/default_config.yaml" >&3
        cp "$DIR/config_files/accelerate/default_config.yaml" "$HF_HOME/accelerate/default_config.yaml" &&
          echo "Copied accelerate config file to: $HF_HOME/accelerate/default_config.yaml"
      fi
    elif env_var_exists XDG_CACHE_HOME; then
      if [ ! -f "$XDG_CACHE_HOME/huggingface/accelerate" ]; then
        mkdir -p "$XDG_CACHE_HOME/huggingface/accelerate" &&
          echo "Target accelerate config location: $XDG_CACHE_HOME/accelerate/default_config.yaml" >&3
        cp "$DIR/config_files/accelerate/default_config.yaml" "$XDG_CACHE_HOME/huggingface/accelerate/default_config.yaml" &&
          echo "Copied accelerate config file to: $XDG_CACHE_HOME/huggingface/accelerate/default_config.yaml"
      fi
    elif env_var_exists HOME; then
      if [ ! -f "$HOME/.cache/huggingface/accelerate" ]; then
        mkdir -p "$HOME/.cache/huggingface/accelerate" &&
          echo "Target accelerate config location: $HOME/accelerate/default_config.yaml" >&3
        cp "$DIR/config_files/accelerate/default_config.yaml" "$HOME/.cache/huggingface/accelerate/default_config.yaml" &&
          echo "Copying accelerate config file to: $HOME/.cache/huggingface/accelerate/default_config.yaml"
      fi
    else
      echo "Could not place the accelerate configuration file. Please configure manually."
      sleep 2
      accelerate config
    fi
  fi
}

# Function to update Kohya_SS repo
update_kohya_ss() {
  if [ "$SKIP_GIT_UPDATE" = false ]; then
    if command -v git >/dev/null; then
      # First, we make sure there are no changes that need to be made in git, so no work is lost.
      if [ "$(git -C "$DIR" status --porcelain=v1 2>/dev/null | wc -l)" -gt 0 ] &&
        echo "These files need to be committed or discarded: " >&4 &&
        git -C "$DIR" status >&4; then
        echo "There are changes that need to be committed or discarded in the repo in $DIR."
        echo "Commit those changes or run this script with -n to skip git operations entirely."
        exit 1
      fi

      echo "Attempting to clone $GIT_REPO."
      if [ ! -d "$DIR/.git" ]; then
        echo "Cloning and switching to $GIT_REPO:$BRANCH" >&4
        git -C "$PARENT_DIR" clone -b "$BRANCH" "$GIT_REPO" "$(basename "$DIR")" >&3
        git -C "$DIR" switch "$BRANCH" >&4
      else
        echo "git repo detected. Attempting to update repository instead."
        echo "Updating: $GIT_REPO"
        git -C "$DIR" pull "$GIT_REPO" "$BRANCH" >&3
        if ! git -C "$DIR" switch "$BRANCH" >&4; then
          echo "Branch $BRANCH did not exist. Creating it." >&4
          git -C "$DIR" switch -c "$BRANCH" >&4
        fi
      fi
    else
      echo "You need to install git."
      echo "Rerun this after installing git or run this script with -n to skip the git operations."
    fi
  else
    echo "Skipping git operations."
  fi
}

# Section: Command-line options parsing

while getopts ":vb:d:g:inprus-:" opt; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$opt" = "-" ]; then # long option: reformulate OPT and OPTARG
    opt="${OPTARG%%=*}"     # extract long option name
    OPTARG="${OPTARG#$opt}" # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"    # if long option argument, remove assigning `=`
  fi
  
  case $opt in
  b | branch) BRANCH="$OPTARG" ;;
  d | dir) DIR="$OPTARG" ;;
  g | git-repo) GIT_REPO="$OPTARG" ;;
  i | interactive) INTERACTIVE=true ;;
  n | no-git-update) SKIP_GIT_UPDATE=true ;;
  p | public) PUBLIC=true ;;
  r | runpod) RUNPOD=true ;;
  s | skip-space-check) SKIP_SPACE_CHECK=true ;;
  u | no-gui) SKIP_GUI=true ;;
  v) ((VERBOSITY = VERBOSITY + 1)) ;;
  use-ipex) USE_IPEX=true ;;
  use-rocm) USE_ROCM=true ;;
  h) display_help && exit 0 ;;
  *) display_help && exit 0 ;;
  esac
done
shift $((OPTIND - 1))

# Just in case someone puts in a relative path into $DIR,
# we're going to get the absolute path of that.
if [[ "$DIR" != /* ]] && [[ "$DIR" != ~* ]]; then
  DIR="$(
    cd "$(dirname "$DIR")" || exit 1
    pwd
  )/$(basename "$DIR")"
fi

for v in $( #Start counting from 3 since 1 and 2 are standards (stdout/stderr).
  seq 3 $VERBOSITY
); do
  (("$v" <= "$MAXVERBOSITY")) && eval exec "$v>&2" #Don't change anything higher than the maximum verbosity allowed.
done

for v in $( #From the verbosity level one higher than requested, through the maximum;
  seq $((VERBOSITY + 1)) $MAXVERBOSITY
); do
  (("$v" > "2")) && eval exec "$v>/dev/null" #Redirect these to bitbucket, provided that they don't match stdout and stderr.
done

# Example of how to use the verbosity levels.
# printf "%s\n" "This message is seen at verbosity level 1 and above." >&3
# printf "%s\n" "This message is seen at verbosity level 2 and above." >&4
# printf "%s\n" "This message is seen at verbosity level 3 and above." >&5

# Debug variable dump at max verbosity
echo "BRANCH: $BRANCH
DIR: $DIR
GIT_REPO: $GIT_REPO
INTERACTIVE: $INTERACTIVE
PUBLIC: $PUBLIC
RUNPOD: $RUNPOD
SKIP_SPACE_CHECK: $SKIP_SPACE_CHECK
VERBOSITY: $VERBOSITY
Script directory is ${SCRIPT_DIR}." >&5

# This must be set after the getopts loop to account for $DIR changes.
PARENT_DIR="$(dirname "${DIR}")"
VENV_DIR="$DIR/venv"

if [ -w "$PARENT_DIR" ] && [ ! -d "$DIR" ]; then
  echo "Creating install folder ${DIR}."
  mkdir "$DIR"
fi

if [ ! -w "$DIR" ]; then
  echo "We cannot write to ${DIR}."
  echo "Please ensure the install directory is accurate and you have the correct permissions."
  exit 1
fi

# Shared functions
# This checks for free space on the installation drive and returns that in Gb.
size_available() {
  local folder
  if [ -d "$DIR" ]; then
    folder="$DIR"
  elif [ -d "$PARENT_DIR" ]; then
    folder="$PARENT_DIR"
  elif [ -d "$(echo "$DIR" | cut -d "/" -f2)" ]; then
    folder="$(echo "$DIR" | cut -d "/" -f2)"
  else
    echo "We are assuming a root drive install for space-checking purposes."
    folder='/'
  fi

  local FREESPACEINKB
  FREESPACEINKB="$(df -Pk "$folder" | sed 1d | grep -v used | awk '{ print $4 "\t" }')"
  echo "Detected available space in Kb: $FREESPACEINKB" >&5
  local FREESPACEINGB
  FREESPACEINGB=$((FREESPACEINKB / 1024 / 1024))
  echo "$FREESPACEINGB"
}

isContainerOrPod() {
  local cgroup=/proc/1/cgroup
  test -f $cgroup && (grep -qE ':cpuset:/(docker|kubepods)' $cgroup || grep -q ':/docker/' $cgroup)
}

isDockerBuildkit() {
  local cgroup=/proc/1/cgroup
  test -f $cgroup && grep -q ':cpuset:/docker/buildkit' $cgroup
}

isDockerContainer() {
  [ -e /.dockerenv ]
}

inDocker() {
  if isContainerOrPod || isDockerBuildkit || isDockerContainer; then
    return 0
  else
    return 1
  fi
}

# Start OS-specific detection and work
if [[ "$OSTYPE" == "lin"* ]]; then
  # Check if root or sudo
  root=false
  if [ "$EUID" = 0 ]; then
    root=true
  elif command -v id >/dev/null && [ "$(id -u)" = 0 ]; then
    root=true
  elif [ "$UID" = 0 ]; then
    root=true
  fi

  check_storage_space
  update_kohya_ss

  distro=get_distro_name
  family=get_distro_family
  echo "Raw detected distro string: $distro" >&4
  echo "Raw detected distro family string: $family" >&4

  if "$distro" | grep -qi "Ubuntu" || "$family" | grep -qi "Ubuntu"; then
    echo "Ubuntu detected."
    if [ $(dpkg-query -W -f='${Status}' python3-tk 2>/dev/null | grep -c "ok installed") = 0 ]; then
      # if [ "$root" = true ]; then
        echo "This script needs YOU to install the missing python3-tk packages. Please install with:"
        echo " "
        if [ "$RUNPOD" = true ]; then
          bash apt update -y && apt install -y python3-tk
        else
          echo "sudo apt update -y && sudo apt install -y python3-tk"
        fi
        exit 1
      # else
      #   echo "This script needs to be run as root or via sudo to install packages."
      #   exit 1
      # fi
    else
      echo "Python TK found..."
    fi
  elif "$distro" | grep -Eqi "Fedora|CentOS|Redhat"; then
    echo "Redhat or Redhat base detected."
    if ! rpm -qa | grep -qi python3-tkinter; then
      # if [ "$root" = true ]; then
        echo "This script needs you to install the missing python3-tk packages. Please install with:\n\n"
        echo "sudo dnf install python3-tkinter -y >&3"
        exit 1
      # else
      #   echo "This script needs to be run as root or via sudo to install packages."
      #   exit 1
      # fi
    else
      echo "Python TK found..."
    fi
  elif "$distro" | grep -Eqi "arch" || "$family" | grep -qi "arch"; then
    echo "Arch Linux or Arch base detected."
    if ! pacman -Qi tk >/dev/null; then
      # if [ "$root" = true ]; then
        echo "This script needs you to install the missing python3-tk packages. Please install with:\n\n"
        echo "pacman --noconfirm -S tk >&3"
        exit 1
      # else
      #   echo "This script needs to be run as root or via sudo to install packages."
      #   exit 1
      # fi
    else
      echo "Python TK found..."
    fi
  elif "$distro" | grep -Eqi "opensuse" || "$family" | grep -qi "opensuse"; then
    echo "OpenSUSE detected."
    if ! rpm -qa | grep -qi python-tk; then
      # if [ "$root" = true ]; then
        echo "This script needs you to install the missing python3-tk packages. Please install with:\n\n"
        echo "zypper install -y python-tk >&3"
        exit 1
      # else
      #   echo "This script needs to be run as root or via sudo to install packages."
      #   exit 1
      # fi
    else
      echo "Python TK found..."
    fi
  elif [ "$distro" = "None" ] || [ "$family" = "None" ]; then
    if [ "$distro" = "None" ]; then
      echo "We could not detect your distribution of Linux. Please file a bug report on github with the contents of your /etc/os-release file."
    fi

    if [ "$family" = "None" ]; then
      echo "We could not detect the family of your Linux distribution. Please file a bug report on github with the contents of your /etc/os-release file."
    fi
  fi

  install_python_dependencies

  # We need just a little bit more setup for non-interactive environments
  if [ "$RUNPOD" = true ]; then
    if inDocker; then
      # We get the site-packages from python itself, then cut the string, so no other code changes required.
      VENV_DIR=$(python -c "import site; print(site.getsitepackages()[0])")
      VENV_DIR="${VENV_DIR%/lib/python3.10/site-packages}"
    fi

    # Symlink paths
    libnvinfer_plugin_symlink="$VENV_DIR/lib/python3.10/site-packages/tensorrt/libnvinfer_plugin.so.7"
    libnvinfer_symlink="$VENV_DIR/lib/python3.10/site-packages/tensorrt/libnvinfer.so.7"
    libcudart_symlink="$VENV_DIR/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.11.0"

    #Target file paths
    libnvinfer_plugin_target="$VENV_DIR/lib/python3.10/site-packages/tensorrt/libnvinfer_plugin.so.8"
    libnvinfer_target="$VENV_DIR/lib/python3.10/site-packages/tensorrt/libnvinfer.so.8"
    libcudart_target="$VENV_DIR/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"

    # echo "Checking symlinks now."
    # create_symlinks "$libnvinfer_plugin_symlink" "$libnvinfer_plugin_target"
    # create_symlinks "$libnvinfer_symlink" "$libnvinfer_target"
    # create_symlinks "$libcudart_symlink" "$libcudart_target"

    # if [ -d "${VENV_DIR}/lib/python3.10/site-packages/tensorrt/" ]; then
    #   export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${VENV_DIR}/lib/python3.10/site-packages/tensorrt/"
    # else
    #   echo "${VENV_DIR}/lib/python3.10/site-packages/tensorrt/ not found; not linking library."
    # fi

    # if [ -d "${VENV_DIR}/lib/python3.10/site-packages/tensorrt/" ]; then
    #   export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${VENV_DIR}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/"
    # else
    #   echo "${VENV_DIR}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/ not found; not linking library."
    # fi

    configure_accelerate

    # This is a non-interactive environment, so just directly call gui.sh after all setup steps are complete.
    if [ "$SKIP_GUI" = false ]; then
      if command -v bash >/dev/null; then
        if [ "$PUBLIC" = false ]; then
          bash "$DIR"/gui.sh --headless
          exit 0
        else
          bash "$DIR"/gui.sh --headless --share
          exit 0
        fi
      else
        # This shouldn't happen, but we're going to try to help.
        if [ "$PUBLIC" = false ]; then
          sh "$DIR"/gui.sh --headless
          exit 0
        else
          sh "$DIR"/gui.sh --headless --share
          exit 0
        fi
      fi
    fi
  fi

  echo -e "Setup finished! Run \e[0;92m./gui.sh\e[0m to start."
  echo "Please note if you'd like to expose your public server you need to run ./gui.sh --share"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  # The initial setup script to prep the environment on macOS
  # xformers has been omitted as that is for Nvidia GPUs only

  if ! command -v brew >/dev/null; then
    echo "Please install homebrew first. This is a requirement for the remaining setup."
    echo "You can find that here: https://brew.sh"
    #shellcheck disable=SC2016
    echo 'The "brew" command should be in $PATH to be detected.'
    exit 1
  fi

  check_storage_space

  # Install base python packages
  echo "Installing Python 3.10 if not found."
  if ! brew ls --versions python@3.10 >/dev/null; then
    echo "Installing Python 3.10."
    brew install python@3.10 >&3
  else
    echo "Python 3.10 found!"
  fi
  echo "Installing Python-TK 3.10 if not found."
  if ! brew ls --versions python-tk@3.10 >/dev/null; then
    echo "Installing Python TK 3.10."
    brew install python-tk@3.10 >&3
  else
    echo "Python Tkinter 3.10 found!"
  fi

  update_kohya_ss

  if ! install_python_dependencies; then
    echo "You may need to install Python. The command for this is brew install python@3.10."
  fi

  configure_accelerate
  echo -e "Setup finished! Run ./gui.sh to start."
elif [[ "$OSTYPE" == "cygwin" ]]; then
  # Cygwin is a standalone suite of Linux utilities on Windows
  echo "This hasn't been validated on cygwin yet."
elif [[ "$OSTYPE" == "msys" ]]; then
  # MinGW has the msys environment which is a standalone suite of Linux utilities on Windows
  # "git bash" on Windows may also be detected as msys.
  echo "This hasn't been validated in msys 'mingw' on Windows yet."
fi
