#!/usr/bin/env bash

# This file will be the host environment setup file for all operating systems other than base Windows.

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
  --gui-listen=IP               IP to listen on for connections to Gradio.
  --gui-username=USERNAME       Username for authentication.
  --gui-password=PASSWORD       Password for authentication.
  --gui-server-port=PORT        Port to run the server listener on.
  --gui-inbrowser               Open in browser.
  --gui-share                   Share your installation.
  -v, --verbose                 Increase verbosity levels up to 3.
EOF
}

# This gets the directory the script is run from so pathing can work relative to the script where needed.
SCRIPT_DIR="$(cd -- $(dirname -- "$0") && pwd)"

parse_yaml() {
  local yaml_file="$1"
  local prefix="$2"
  local line key value state

  state="none"

  while IFS= read -r line; do
    if [[ "$line" =~ ^[[:space:]]*name:[[:space:]]*([^[:space:]#]+)$ ]]; then
      key="${BASH_REMATCH[1]}"
      state="searching_default"
    elif [[ "$state" == "searching_default" ]] && [[ "$line" =~ ^[[:space:]]*default:[[:space:]]*(.+)$ ]]; then
      value="${BASH_REMATCH[1]}"
      state="none"
      eval "${prefix}_${key}=\"$value\""
    else
      state="none"
    fi
  done <"$yaml_file"
}

# Use the variables from the configuration file as default values
BRANCH="${config_Branch:-master}"
DIR="${config_Dir:-$HOME/kohya_ss}"
GIT_REPO="${config_GitRepo:-https://github.com/kohya/kohya_ss.git}"
INTERACTIVE="${config_Interactive:-false}"
SKIP_GIT_UPDATE="${config_NoGitUpdate:-false}"
PUBLIC="${config_Public:-false}"
RUNPOD="${config_Runpod:-false}"
SKIP_SPACE_CHECK="${config_SkipSpaceCheck:-false}"
VERBOSE="${config_Verbose:-2}" #Start counting at 2 so that any increase to this will result in a minimum of file descriptor 3.  You should leave this alone.
GUI_LISTEN="${config_GuiListen:-127.0.0.1}"
GUI_USERNAME="${config_GuiUsername:-}"
GUI_PASSWORD="${config_GuiPassword:-}"
GUI_SERVER_PORT="${config_GuiServerPort:-8080}"
GUI_INBROWSER="${config_GuiInbrowser:-false}"
GUI_SHARE="${config_GuiShare:-false}"

MAXVERBOSITY=6 #The highest verbosity we use / allow to be displayed.  Feel free to adjust.

# This code handles the loading of parameter values in the following order of precedence:
# 1. Command-line arguments provided by the user
# 2. Values defined in a configuration file
# 3. Default values specified in the script
#
# First, the code checks for the presence of a configuration file in the specified locations.
# If found, the configuration file's values are loaded as default values for the variables.
# Then, the getopts loop processes any command-line arguments provided by the user.
# If the user has provided a value for a parameter via the command-line, it will override
# the corresponding value from the configuration file or the script's default value.
# If neither a configuration file nor a command-line argument is provided for a parameter,
# the script will use the default value specified within the script.

USER_CONFIG_FILE=""
declare -A CLI_ARGUMENTS

while getopts ":vb:d:f:g:inprs-:" opt; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$opt" = "-" ]; then # long option: reformulate OPT and OPTARG
    opt="${OPTARG%%=*}"     # extract long option name
    OPTARG="${OPTARG#$opt}" # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"    # if long option argument, remove assigning `=`
  fi
  case $opt in
  b | branch) CLI_ARGUMENTS["Branch"]="$OPTARG" ;;
  d | dir) CLI_ARGUMENTS["Dir"]="$OPTARG" ;;
  f | file) USER_CONFIG_FILE="$OPTARG" ;;
  g | git-repo) CLI_ARGUMENTS["GitRepo"]="$OPTARG" ;;
  i | interactive) CLI_ARGUMENTS["Interactive"]="true" ;;
  n | no-git-update) CLI_ARGUMENTS["NoGitUpdate"]="true" ;;
  p | public) CLI_ARGUMENTS["Public"]="true" ;;
  r | runpod) CLI_ARGUMENTS["Runpod"]="true" ;;
  s | skip-space-check) CLI_ARGUMENTS["SkipSpaceCheck"]="true" ;;
  gui-listen) CLI_ARGUMENTS["GuiListen"]="$OPTARG" ;;
  gui-username) CLI_ARGUMENTS["GuiUsername"]="$OPTARG" ;;
  gui-password) CLI_ARGUMENTS["GuiPassword"]="$OPTARG" ;;
  gui-server-port) CLI_ARGUMENTS["GuiServerPort"]="$OPTARG" ;;
  gui-inbrowser) CLI_ARGUMENTS["GuiInbrowser"]="true" ;;
  gui-share) CLI_ARGUMENTS["GuiShare"]="true" ;;
  v) ((CLI_ARGUMENTS["Verbose"] = CLI_ARGUMENTS["Verbose"] + 1)) ;;
  h) display_help && exit 0 ;;
  *) display_help && exit 0 ;;
  esac
done
shift $((OPTIND - 1))

configFileLocations=(
  "$USER_CONFIG_FILE"
  "$HOME/.kohya_ss/install_config.yaml"
  "$DIR/install_config.yaml"
  "$SCRIPT_DIR/install_config.yaml"
)

configFile=""

for location in "${configFileLocations[@]}"; do
  if [ -f "$location" ]; then
    configFile="$location"
    break
  fi
done

if [ -n "$configFile" ]; then
  parse_yaml "$configFile" "config"
fi

# Set default values
config_Branch="${config_Branch:-master}"
config_Dir="${config_Dir:-$HOME/kohya_ss}"
config_GitRepo="${config_GitRepo:-https://github.com/kohya/kohya_ss.git}"
config_Interactive="${config_Interactive:-false}"
config_NoGitUpdate="${config_NoGitUpdate:-false}"
config_Public="${config_Public:-false}"
config_Runpod="${config_Runpod:-false}"
config_SkipSpaceCheck="${config_SkipSpaceCheck:-false}"
config_Verbose="${config_Verbose:-2}"
config_GuiListen="${config_GuiListen:-127.0.0.1}"
config_GuiUsername="${config_GuiUsername:-}"
config_GuiPassword="${config_GuiPassword:-}"
config_GuiServerPort="${config_GuiServerPort:-8080}"
config_GuiInbrowser="${config_GuiInbrowser:-false}"
config_GuiShare="${config_GuiShare:-false}"

# Override config values with CLI arguments
for key in "${!CLI_ARGUMENTS[@]}"; do
  configVar="config_$key"
  eval "$configVar=${CLI_ARGUMENTS[$key]}"
done

# Use the variables from the configuration file as default values
BRANCH="$config_Branch"
DIR="$config_Dir"
GIT_REPO="$config_GitRepo"
INTERACTIVE="$config_Interactive"
SKIP_GIT_UPDATE="$config_NoGitUpdate"
PUBLIC="$config_Public"
RUNPOD="$config_Runpod"
SKIP_SPACE_CHECK="$config_SkipSpaceCheck"
VERBOSE="$config_Verbose"
GUI_LISTEN="$config_GuiListen"
GUI_USERNAME="$config_GuiUsername"
GUI_PASSWORD="$config_GuiPassword"
GUI_SERVER_PORT="$config_GuiServerPort"
GUI_INBROWSER="$config_GuiInbrowser"
GUI_SHARE="$config_GuiShare"

for v in $( #Start counting from 3 since 1 and 2 are standards (stdout/stderr).
  seq 3 $VERBOSE
); do
  (("$v" <= "$VERBOSE")) && eval exec "$v>&2" #Don't change anything higher than the maximum verbosity allowed.
done

for v in $( #From the verbosity level one higher than requested, through the maximum;
  seq $((VERBOSE + 1)) $MAXVERBOSITY
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
Config file location: $USER_CONFIG_FILE
INTERACTIVE: $INTERACTIVE
PUBLIC: $PUBLIC
RUNPOD: $RUNPOD
SKIP_SPACE_CHECK: $SKIP_SPACE_CHECK
VERBOSITY: $VERBOSITY
Script directory is ${SCRIPT_DIR}." >&5

# Shared functions
# Offer a warning and opportunity to cancel the installation if < 10Gb of Free Space detected
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

check_storage_space() {
  if [ "$SKIP_SPACE_CHECK" = false ]; then
    if [ "$(size_available)" -lt "$1" ]; then
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

run_launcher() {
  if command -v python3.10 >/dev/null 2>&1; then
    local PYTHON_EXEC="python3.10"
  elif command -v python3 >/dev/null 2>&1 && [ "$(python3 -c 'import sys; print(sys.version_info[:2])')" = "(3, 10)" ]; then
    local PYTHON_EXEC="python3"
  else
    echo "Error: Python 3.10 is required to run this script. Please install Python 3.10 and try again."
    exit 1
  fi

  "$PYTHON_EXEC" launcher.py \
    --branch="$BRANCH" \
    --dir="$DIR" \
    --gitrepo="$GIT_REPO" \
    --interactive="$INTERACTIVE" \
    --nogitupdate="$SKIP_GIT_UPDATE" \
    --public="$PUBLIC" \
    --runpod="$RUNPOD" \
    --skipspacecheck="$SKIP_SPACE_CHECK" \
    --listen="$GUI_LISTEN" \
    --username="$GUI_USERNAME" \
    --password="$GUI_PASSWORD" \
    --server_port="$GUI_SERVER_PORT" \
    --inbrowser="$GUI_INBROWSER" \
    --share="$GUI_SHARE" \
    --verbose="$VERBOSITY"
}

# Start OS-specific detection and work
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

  # We search for the line starting with ID_LIKE=
  # Then we remove the ID_LIKE= prefix to get the name itself
  # This is the "type" of distro. For example, Ubuntu returns "debian".
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

  distro=get_distro_name
  family=get_distro_family
  echo "Raw detected distro string: $distro" >&4
  echo "Raw detected distro family string: $family" >&4

  echo "Installing Python TK if not found on the system."
  if "$distro" | grep -qi "Ubuntu" || "$family" | grep -qi "Ubuntu"; then
    echo "Ubuntu detected."
    if [ $(dpkg-query -W -f='${Status}' python3-tk 2>/dev/null | grep -c "ok installed") = 0 ]; then
      if [ "$root" = true ]; then
        apt update -y >&3 && apt install -y python3-tk >&3
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
        dnf install python3-tkinter -y >&3
      else
        echo "This script needs to be run as root or via sudo to install packages."
        exit 1
      fi
    fi
  elif "$distro" | grep -Eqi "arch" || "$family" | grep -qi "arch"; then
    echo "Arch Linux or Arch base detected."
    if ! pacman -Qi tk >/dev/null; then
      if [ "$root" = true ]; then
        pacman --noconfirm -S tk >&3
      else
        echo "This script needs to be run as root or via sudo to install packages."
        exit 1
      fi
    fi
  elif "$distro" | grep -Eqi "opensuse" || "$family" | grep -qi "opensuse"; then
    echo "OpenSUSE detected."
    if ! rpm -qa | grep -qi python-tk; then
      if [ "$root" = true ]; then
        zypper install -y python-tk >&3
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

  # Setup should be completed by now. Run the launcher for remaining tasks and running the GUI.
  echo -e "Python setup finished. Running launcher.py to complete installation."
  run_launcher
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

  # Will warn the user if their space is low before installing Python.
  # check_storage_space 1 means it will warn the user if 1gb of storage space or less detected.
  check_storage_space 1

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

  # Setup should be completed by now. Run the launcher for remaining tasks and running the GUI.
  echo -e "Python setup finished. Running launcher.py to complete installation."
  run_launcher
elif [[ "$OSTYPE" == "cygwin" ]]; then
  # Cygwin is a standalone suite of Linux utilities on Windows
  echo "This hasn't been validated on cygwin yet."
elif [[ "$OSTYPE" == "msys" ]]; then
  # MinGW has the msys environment which is a standalone suite of Linux utilities on Windows
  # "git bash" on Windows may also be detected as msys.
  echo "This hasn't been validated in msys (mingw) on Windows yet."
fi
