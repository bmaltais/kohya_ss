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
  "$DIR/config_files/installation/install_config.yaml"
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
is_admin() {
  if [ "$(uname -s)" = "Windows" ] || [ "$(uname -s)" = "MINGW64_NT" ] || [ "$(uname -s)" = "CYGWIN_NT" ]; then
    if net session >/dev/null 2>&1; then
      return 0
    else
      return 1
    fi
  else
    if [ "$EUID" = 0 ] || [ "$(id -u)" = 0 ] || [ "$UID" = 0 ]; then
      return 0
    else
      return 1
    fi
  fi
}

update_install_scope() {
  local interactive="$1"
  local install_option
  local install_scope

  if [ "$interactive" = true ]; then
    while true; do
      read -rp "Choose installation option: (1) Local (2) Global: " install_option
      if [ "$install_option" = "1" ] || [ "$install_option" = "2" ]; then
        break
      fi
    done
  else
    install_option=2
  fi

  if [ "$install_option" = "1" ]; then
    install_scope="user"
  else
    install_scope="allusers"
  fi

  echo "$install_scope"
}

normalize_path() {
  local path="$1"
  os=$(get_os_info)

  case $os in
  "Windows")
    if command -v cygpath >/dev/null 2>&1; then
      path=$(cygpath -m -a "$path" 2>/dev/null || echo "$path")
    else
      # Fallback method for Windows without cygpath
      path="$(cd "$(dirname "$path")" && pwd -W)/$(basename "$path")"
    fi
    ;;
  *)
    # Portable approach for Linux and BSD systems, including minimal environments like Alpine Linux
    # shellcheck disable=SC2164
    path=$(
      cd "$(dirname "$path")"
      pwd
    )/$(basename "$path")
    ;;
  esac

  echo "$path"
}
# Offer a warning and opportunity to cancel the installation if < 10Gb of Free Space detected
size_available() {
  local folder
  os=$(get_os_info)

  case $os in
  "Windows")
    folder="C:"
    ;;
  *)
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
    ;;
  esac

  local FREESPACEINKB
  case $os in
  "Windows")
    FREESPACEINKB=$(wmic logicaldisk where "DeviceID='$folder'" get FreeSpace | awk 'NR==2 {print int($1/1024)}')
    ;;
  *)
    FREESPACEINKB=$(df -Pk "$folder" | sed 1d | grep -v used | awk '{ print $4 "\t" }')
    ;;
  esac

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

get_os_info() {
  os_name="Unknown"
  os_family="Unknown"
  os_version="Unknown"

  case "$(uname -s)" in
  Darwin*)
    os_name="macOS"
    os_family="macOS"
    os_version=$(sw_vers -productVersion)
    ;;
  MINGW64_NT* | MSYS_NT* | CYGWIN_NT*)
    os_name="Windows"
    os_family="Windows"
    os_version=$(systeminfo | grep "^OS Version" | awk -F: '{print $2}' | tr -d '[:space:]')
    ;;
  Linux*)
    if [ -f /etc/os-release ]; then
      os_name=$(grep -oP '(?<=^ID=).*' /etc/os-release | tr -d '"')
      os_family=$(grep -oP '(?<=^ID_LIKE=).*' /etc/os-release | tr -d '"')
      os_version=$(grep -oP '(?<=^VERSION=).*' /etc/os-release | tr -d '"')
    elif [ -f /etc/redhat-release ]; then
      os_name=$(awk '{print $1}' /etc/redhat-release)
      os_family="RedHat"
      os_version=$(awk '{print $3}' /etc/redhat-release)
    fi

    if [ "$os_name" == "Unknown" ]; then
      uname_output=$(uname -a)
      if [[ $uname_output == *"Ubuntu"* ]]; then
        os_name="Ubuntu"
        os_family="Ubuntu"
      elif [[ $uname_output == *"Debian"* ]]; then
        os_name="Debian"
        os_family="Debian"
      elif [[ $uname_output == *"Red Hat"* || $uname_output == *"CentOS"* ]]; then
        os_name="RedHat"
        os_family="RedHat"
      elif [[ $uname_output == *"Fedora"* ]]; then
        os_name="Fedora"
        os_family="Fedora"
      elif [[ $uname_output == *"SUSE"* ]]; then
        os_name="openSUSE"
        os_family="SUSE"
      elif [[ $uname_output == *"Arch"* ]]; then
        os_name="Arch"
        os_family="Arch"
      else
        os_name="Generic Linux"
        os_family="Generic Linux"
      fi
    fi
    ;;
  *)
    os_name="Unknown"
    os_family="Unknown"
    ;;
  esac

  echo "$os_name"
}

install_git_windows() {
  local interactive="$1"
  local package_manager_found=false

  if command -v git >/dev/null 2>&1; then
    echo "Git is already installed."
    return 0
  fi

  if command -v scoop >/dev/null 2>&1; then
    scoop install git
    package_manager_found=true
  elif command -v choco >/dev/null 2>&1; then
    choco install git
    package_manager_found=true
  elif command -v winget >/dev/null 2>&1; then
    winget install --id Git.Git
    package_manager_found=true
  fi

  if [ "$package_manager_found" = false ]; then
    if is_admin; then
      local install_scope=$(update_install_scope "$interactive")
    else
      install_scope="user"
    fi

    local git_url="https://github.com/git-for-windows/git/releases/download/v2.35.1.windows.1/Git-2.35.1-64-bit.exe"
    local git_installer_name="Git-2.35.1-64-bit.exe"
    local downloads_folder="$HOME/Downloads"
    local installer_path="${downloads_folder}/${git_installer_name}"

    if [ ! -f "$installer_path" ]; then
      if ! curl -o "$installer_path" -L "$git_url"; then
        echo "Failed to download Git. Please check your internet connection or provide a pre-downloaded installer."
        exit 1
      fi
    fi

    if [ "$install_scope" = "user" ]; then
      start /wait "$installer_path" /VERYSILENT /NORESTART /LOG /NOICONS /COMPONENTS="icons,ext\reg\shellhere,assoc,assoc_sh"
    else
      start /wait "$installer_path" /VERYSILENT /NORESTART /LOG /NOICONS /COMPONENTS="icons,ext\reg\shellhere,assoc,assoc_sh" /ALLUSERS=1
    fi

    rm -f "$installer_path"
  fi
}

install_git() {
  os=$(get_os_info)

  if command -v git >/dev/null 2>&1; then
    echo "Git is already installed."
    return 0
  fi

  case $os in
  "Windows")
    install_git_windows "$interactive"
    ;;
  "macOS")
    if command -v brew >/dev/null 2>&1; then
      brew install git
    else
      echo "Please install Homebrew first to continue with Git installation."
      echo "You can find that here: https://brew.sh"
      exit 1
    fi
    ;;
  "Ubuntu" | "Debian")
    if is_admin; then
      sudo apt-get update && sudo apt-get install -y git
    else
      echo "Admin privileges are required to install Git. Please run the script as root or with sudo."
      exit 1
    fi
    ;;
  "Fedora" | "CentOS" | "RedHat")
    if is_admin; then
      sudo dnf install -y git
    else
      echo "Admin privileges are required to install Git. Please run the script as root or with sudo."
      exit 1
    fi
    ;;
  "Arch" | "Manjaro")
    if is_admin; then
      sudo pacman -Sy --noconfirm git
    else
      echo "Admin privileges are required to install Git. Please run the script as root or with sudo."
      exit 1
    fi
    ;;
  "openSUSE")
    if is_admin; then
      sudo zypper install -y git
    else
      echo "Admin privileges are required to install Git. Please run the script as root or with sudo."
      exit 1
    fi
    ;;
  *)
    echo "Unsupported operating system. Please install Git manually."
    exit 1
    ;;
  esac
}

install_python310_windows() {
  local interactive="$1"

  if command -v scoop >/dev/null 2>&1; then
    package_manager_found=false
  elif command -v choco >/dev/null 2>&1; then
    choco install python --version=3.10 --params "/IncludeTclTk"
    package_manager_found=true
  elif command -v winget >/dev/null 2>&1; then
    winget install --id Python.Python --version 3.10.*
    package_manager_found=true
  fi

  if [ "$package_manager_found" = false ]; then
    if is_admin; then
      local install_scope=$(update_install_scope "$interactive")
    else
      install_scope="user"
    fi

    local python_url="https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe"
    local python_installer_name="python-3.10.0-amd64.exe"
    local downloads_folder="$HOME/Downloads"
    local installer_path="${downloads_folder}/${python_installer_name}"

    if [ ! -f "$installer_path" ]; then
      if ! curl -o "$installer_path" -L "$python_url"; then
        echo "Failed to download Python 3.10. Please check your internet connection or provide a pre-downloaded installer."
        exit 1
      fi
    fi

    if [ "$install_scope" = "user" ]; then
      start /wait "$installer_path" /passive InstallAllUsers=0 PrependPath=1 Include_tcltk=1
    else
      start /wait "$installer_path" /passive InstallAllUsers=1 PrependPath=1 Include_tcltk=1
    fi

    rm -f "$installer_path"
  fi
}

install_python_and_tk() {
  os=$(get_os_info)

  if command -v python3.10 >/dev/null 2>&1; then
    echo "Python 3.10 is already installed."
  else
    case $os in
    "Windows")
      install_python310_windows "$interactive"
      ;;
    "macOS")
      if command -v brew >/dev/null 2>&1; then
        brew install python@3.10
        brew link --overwrite --force python@3.10
        brew install tcl-tk
      else
        echo "Please install Homebrew first to continue with Python 3.10 and Tk installation."
        echo "You can find that here: https://brew.sh"
        exit 1
      fi
      ;;
    "Ubuntu" | "Debian")
      if is_admin; then
        sudo apt update && sudo apt install -y python3.10 python3.10-tk
      else
        echo "Root privileges are required to install Python and Tk on Ubuntu/Debian. Exiting."
        exit 1
      fi
      ;;
    "Fedora" | "CentOS" | "RedHat")
      if is_admin; then
        sudo dnf install -y python3.10 python3.10-tkinter
      else
        echo "Root privileges are required to install Python and Tk on Fedora/CentOS/RedHat. Exiting."
        exit 1
      fi
      ;;
    "Arch" | "Manjaro")
      # Get the latest 3.10.x version of Python available in the repository
      # shellcheck disable=SC2155
      local latest_python310_version=$(pacman -Si python | grep -oP '3.10\.\d+' | head -n 1)

      if [[ -n "$latest_python310_version" ]]; then
        # Install the latest 3.10.x version of Python along with python-tk
        if is_admin; then
          sudo pacman -Sy --noconfirm "python=${latest_python310_version}" python-tk
        else
          echo "Root privileges are required to install Python and Tk on Arch/Manjaro. Exiting."
          exit 1
        fi
      else
        echo "Python 3.10.x not found in the repository."
        exit 1
      fi
      ;;
    "openSUSE")
      if is_admin; then
        sudo zypper install -y python3.10 python3.10-tk
      else
        echo "Root privileges are required to install Python and Tk on openSUSE. Exiting."
        exit 1
      fi
      ;;
    *)
      echo "Unsupported operating system. Please install Python 3.10 and Python Tk 3.10 manually."
      echo "For manual installation, you can download the official Python tar.gz packages from:"
      echo "https://www.python.org/downloads/source/"
      exit 1
      ;;
    esac
  fi
}

install_vc_redist_windows() {
  os=$(get_os_info)
  if [ "$os" != "Windows" ]; then
    return 0
  fi

  if ! is_admin; then
    echo "Admin privileges are required to install Visual Studio redistributables. Please run this script as an administrator."
    exit 1
  fi

  local vc_redist_url="https://aka.ms/vs/17/release/vc_redist.x64.exe"
  local vc_redist_installer_name="vc_redist.x64.exe"
  local downloads_folder="$HOME/Downloads"
  local installer_path="${downloads_folder}/${vc_redist_installer_name}"

  if [ ! -f "$installer_path" ]; then
    if ! curl -o "$installer_path" -L "$vc_redist_url"; then
      echo "Failed to download Visual Studio redistributables. Please check your internet connection or provide a pre-downloaded installer."
      exit 1
    fi
  fi

  start /wait "$installer_path" /install /quiet /norestart
  rm -f "$installer_path"
}

function main() {
  DIR="$(normalize_path "$DIR")"
  # Warn user and give them a chance to cancel install if less than 5Gb is available on storage device
  check_storage_space 5
  install_git
  install_python_and_tk
  install_vc_redist_windows
  run_launcher
}

main
