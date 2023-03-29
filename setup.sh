#!/usr/bin/env bash

# This file will be the host environment setup file for all operating systems other than base Windows.

display_help() {
  cat <<EOF
Kohya_SS Installation Script for POSIX operating systems.

The following options are useful in a runpod environment,
but will not affect a local machine install.

Usage:
  setup.sh -b dev -d /workspace/kohya_ss -g https://mycustom.repo.tld/custom_fork.git
  setup.sh --branch=dev --dir=/workspace/kohya_ss --git-repo=https://mycustom.repo.tld/custom_fork.git

Options:
  -b BRANCH, --branch=BRANCH    Select which branch of kohya to checkout on new installs.
  -d DIR, --dir=DIR             The full path you want kohya_ss installed to.
  -g, --git_repo                You can optionally provide a git repo to checkout for runpod installation. Useful for custom forks.
  -r, --runpod                  Forces a runpod installation. Useful if detection fails for any reason.
  -i, --interactive             Interactively configure accelerate instead of using default config file.
  -h, --help                    Show this screen.
EOF
}

# Variables defined before the getopts loop, so we have sane default values.
DIR="/workspace/kohya_ss"
BRANCH="master"
GIT_REPO="https://github.com/bmaltais/kohya_ss.git"
RUNPOD=false
INTERACTIVE=false

while getopts "b:d:g:ir-:" opt; do
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
  r | runpod) RUNPOD=true ;;
  h) display_help && exit 0 ;;
  *) display_help && exit 0 ;;
  esac
done
shift $((OPTIND - 1))

# This must be set after the getopts loop to account for $DIR changes.
PARENT_DIR="$(dirname "${DIR}")"
VENV_DIR="$DIR/venv"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  # Check if root or sudo
  root=false
  if [ "$EUID" = 0 ]; then
    root=true
  elif command -v id >/dev/null && [ "$(id -u)" = 0 ]; then
    root=true
  elif [ "$UID" = 0 ]; then
    root=true
  fi

  env_var_exists() {
    local env_var=
    env_var=$(declare -p "$1")
    if ! [[ -v $1 && $env_var =~ ^declare\ -x ]]; then
      return 1
    fi
  }

  get_distro_name() {
    local line
    if [ -f /etc/os-release ]; then
      # We search for the line starting with ID=
      # Then we remove the ID= prefix to get the name itself
      line="$(grep -Ei '^ID=' /etc/os-release)"
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

  get_distro_family() {
    local line
    if [ -f /etc/os-release ]; then
      # We search for the line starting with ID_LIKE=
      # Then we remove the ID_LIKE= prefix to get the name itself
      # This is the "type" of distro. For example, Ubuntu returns "debian".
      if grep -Eiq '^ID_LIKE=' /etc/os-release >/dev/null; then
        line="$(grep -Ei '^ID_LIKE=' /etc/os-release)"
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

  # This checks for free space on the installation drive and returns that in Gb.
  size_available() {
    local FREESPACEINKB="$(df -Pk "$DIR" | sed 1d | grep -v used | awk '{ print $4 "\t" }')"
    local FREESPACEINGB=$((FREESPACEINKB / 1024 / 1024))
    echo "$FREESPACEINGB"
  }

  if env_var_exists RUNPOD_POD_ID || env_var_exists RUNPOD_API_KEY; then
    RUNPOD=true
  fi

  # Offer a warning and opportunity to cancel the installation if < 10Gb of Free Space detected
  if [ "$(size_available)" -lt 10 ]; then
    echo "You have less than 10Gb of free space. This installation may fail."
    MSGTIMEOUT=10 # In seconds
    MESSAGE="Continuing in..."
    echo "Press control-c to cancel the installation."
    for ((i = $MSGTIMEOUT; i >= 0; i--)); do
      printf "\r${MESSAGE} %ss. " "${i}"
      sleep 1
    done
  fi

  # This is the pre-install work for a kohya installation on a runpod
  if [ "$RUNPOD" = true ]; then
    if [ -d "$VENV_DIR" ]; then
      echo "Pre-existing installation on a runpod detected."
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$VENV_DIR"/lib/python3.10/site-packages/tensorrt/
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$VENV_DIR"/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/
      cd "$DIR" || exit 1
      sed -i "s/interface.launch(\*\*launch_kwargs)/interface.launch(\*\*launch_kwargs,share=True)/g" ./kohya_gui.py
    else
      echo "Clean installation on a runpod detected."
      cd "$PARENT_DIR" || exit 1
      if [ ! -d "$DIR/.git" ]; then
        echo "Cloning $GIT_REPO."
        git clone "$GIT_REPO"
        cd "$DIR" || exit 1
        git checkout "$BRANCH"
      else
        cd "$DIR" || exit 1
        echo "git repo detected. Attempting tp update repo instead."
        echo "Updating: $GIT_REPO"
        git pull "$GIT_REPO"
      fi
    fi
  fi

  distro=get_distro_name
  family=get_distro_family

  echo "Installing Python TK if not found on the system."

  if "$distro" | grep -qi "Ubuntu" || "$family" | grep -qi "Ubuntu"; then
    echo "Ubuntu detected."
    if [ $(dpkg-query -W -f='${Status}' python3-tk 2>/dev/null | grep -c "ok installed") = 0 ]; then
      if [ "$root" = true ]; then
        apt update -y && apt install -y python3-tk
      else
        echo "This script needs to be run as root or via sudo to install packages."
        exit 1
      fi
    else
      echo "Python TK found! Skipping install!"
    fi
  elif "$distro" | grep -Eqi "Fedora|CentOS|Redhat"; then
    echo "Redhat or Redhat base detected."
    if ! rpm -qa | grep -qi python3-tkinter; then
      if [ "$root" = true ]; then
        dnf install python3-tkinter -y
      else
        echo "This script needs to be run as root or via sudo to install packages."
        exit 1
      fi
    fi
  elif "$distro" | grep -Eqi "arch" || "$family" | grep -qi "arch"; then
    echo "Arch Linux or Arch base detected."
    if ! pacman -Qi tk >/dev/null; then
      if [ "$root" = true ]; then
        pacman --noconfirm -S tk
      else
        echo "This script needs to be run as root or via sudo to install packages."
        exit 1
      fi
    fi
  elif "$distro" | grep -Eqi "opensuse" || "$family" | grep -qi "opensuse"; then
    echo "OpenSUSE detected."
    if ! rpm -qa | grep -qi python-tk; then
      if [ "$root" = true ]; then
        zypper install -y python-tk
      else
        echo "This script needs to be run as root or via sudo to install packages."
        exit 1
      fi
    fi
  elif [ "$distro" = "None" ] || [ "$family" = "None" ]; then
    if [ "$distro" = "None" ]; then
      echo "We could not detect your distribution of Linux. Please file a bug report on github with the contents of your /etc/os-release file."
    fi

    if [ "$family" = "None" ]; then
      echo "We could not detect the family of your Linux distribution. Please file a bug report on github with the contents of your /etc/os-release file."
    fi
  fi

  python3 -m venv venv
  source venv/bin/activate
  pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
  pip install --use-pep517 --upgrade -r requirements.txt
  pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/linux/xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl

  # We need this extra package and setup if we are running in a runpod
  if [ "$RUNPOD" = true ]; then
    pip install tensorrt
    ln -s "$VENV_DIR/lib/python3.10/site-packages/tensorrt/libnvinfer_plugin.so.8" \
      "$VENV_DIR/lib/python3.10/site-packages/tensorrt/libnvinfer_plugin.so.7"
    ln -s "$VENV_DIR/lib/python3.10/site-packages/tensorrt/libnvinfer.so.8" \
      "$VENV_DIR/lib/python3.10/site-packages/tensorrt/libnvinfer.so.7"
    ln -s "$VENV_DIR/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12" \
      "$VENV_DIR/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.11.0"

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VENV_DIR/lib/python3.10/site-packages/tensorrt/"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VENV_DIR/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/"

    # Attempt to non-interactively install a default accelerate config file unless specified otherwise.
    # Documentation for order of precedence locations for configuration file for automated installation:
    # https://huggingface.co/docs/accelerate/basic_tutorials/launch#custom-configurations
    if [ "$INTERACTIVE" = true ]; then
      accelerate config
    else
      if env_var_exists HF_HOME; then
        if [ ! -f "$HF_HOME/accelerate/default_config.yaml" ]; then
          mkdir -p "$HF_HOME/accelerate/" &&
            cp ./config_files/accelerate/default_config.yaml "$HF_HOME/accelerate/default_config.yaml" &&
            echo "Copied accelerate config file to: $HF_HOME/accelerate/default_config.yaml"
        fi
      elif env_var_exists XDG_CACHE_HOME; then
        if [ ! -f "$XDG_CACHE_HOME/huggingface/accelerate" ]; then
          mkdir -p "$XDG_CACHE_HOME/huggingface/accelerate" &&
            cp ./config_files/accelerate/default_config.yaml "$XDG_CACHE_HOME/huggingface/accelerate/default_config.yaml" &&
            echo "Copied accelerate config file to: $XDG_CACHE_HOME/huggingface/accelerate/default_config.yaml"
        fi
      elif env_var_exists HOME; then
        if [ ! -f "$HOME/.cache/huggingface/accelerate" ]; then
          mkdir -p "$HOME/.cache/huggingface/accelerate" &&
            cp ./config_files/accelerate/default_config.yaml "$HOME/.cache/huggingface/accelerate/default_config.yaml" &&
            echo "Copying accelerate config file to: $HOME/.cache/huggingface/accelerate/default_config.yaml"
        fi
      else
        echo "Could not place the accelerate configuration file. Please configure manually."
        sleep 2
        accelerate config
      fi
    fi

    # This is a non-interactive environment, so just directly call gui.sh after all setup steps are complete.
    if command -v bash >/dev/null; then
      bash "$DIR"/gui.sh
    else
      # This shouldn't happen, but we're going to try to help.
      sh "$DIR"/gui.sh
    fi
  fi

  echo -e "Setup finished! Run \e[0;92m./gui.sh\e[0m to start."
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

  # Install base python packages
  echo "Installing Python 3.10 if not found."
  if ! brew ls --versions python@3.10 >/dev/null; then
    brew install python@3.10
  else
    echo "Python 3.10 found!"
  fi
  echo "Installing Python-TK 3.10 if not found."
  if ! brew ls --versions python-tk@3.10 >/dev/null; then
    brew install python-tk@3.10
  else
    echo "Python Tkinter 3.10 found!"
  fi

  if command -v python3.10 >/dev/null; then
    python3.10 -m venv venv
    source venv/bin/activate

    # DEBUG ONLY
    #pip install pydevd-pycharm~=223.8836.43

    # Tensorflow installation
    if wget https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl /tmp; then
      python -m pip install tensorflow==0.1a3 -f https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl
      rm -f /tmp/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl
    fi

    pip install torch==2.0.0 torchvision==0.15.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
    python -m pip install --use-pep517 --upgrade -r requirements.txt
    accelerate config
    echo -e "Setup finished! Run ./gui.sh to start."
  else
    echo "Python not found. Please ensure you install Python."
    echo "The brew command for Python 3.10 is: brew install python@3.10"
    exit 1
  fi
elif [[ "$OSTYPE" == "cygwin" ]]; then
  # Cygwin is a standalone suite of Linux utilies on Windows
  echo "This hasn't been validated on cygwin yet."
elif [[ "$OSTYPE" == "msys" ]]; then
  # MinGW has the msys environment which is a standalone suite of Linux utilies on Windows
  # "git bash" on Windows may also be detected as msys.
  echo "This hasn't been validated in msys (mingw) on Windows yet."
fi
