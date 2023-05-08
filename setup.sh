#!/usr/bin/env bash

# This file will be the host environment setup file for all operating systems other than base Windows.

# bashsupport disable=GrazieInspection
display_help() {
  cat <<EOF
Kohya_SS Installation Script for POSIX operating systems.

Usage:
  # Specifies custom branch, install directory, and git repo
  setup.sh -b dev -d /workspace/kohya_ss -g https://mycustom.repo.tld/custom_fork.git

  # Same as example 1, but uses long options
  setup.sh --branch=dev --dir=/workspace/kohya_ss --git-repo=https://mycustom.repo.tld/custom_fork.git

  # Running setup in Debug mode while skipping the available space check
    .\setup.ps1 -vvv --skip-space-check

Options:
  -b BRANCH, --branch=BRANCH    Select which branch of kohya to check out on new installs.
  -d DIR, --dir=DIR             The full path you want kohya_ss installed to.
  -f FILE, --file=FILE          Load a custom configuration file.
  -g REPO, --git_repo=REPO      You can optionally provide a git repo to check out. Useful for custom forks.
  -h, --help                    Show this screen.
  -i, --interactive             Interactively configure accelerate instead of using default config file.
  -l LOG_DIR, --log-dir=LOG_DIR Set the custom log directory for kohya_ss.
  -n, --no-setup                Skip all setup steps and only validate python requirements then launch GUI.
  -r, --repair                  This runs the installation repair operations. These could take a few minutes to run.
  -s, --skip-space-check        Skip the 10Gb minimum storage space check.
  -t, --torch-version           Configure the major version of Torch.
  -u, --update                  Update kohya_ss with specified branch, repo, or latest stable if git's unavailable.
  -v                            Increase verbosity levels up to 3. (e.g., -vvv)
  --headless                    Headless mode will not display the native windowing toolkit. Useful for remote deployments.
  --listen=IP                   IP to listen on for connections to Gradio.
  --inbrowser                   Open in browser.
  --password=PASSWORD           Password for authentication.
  --setup-only                  Do not launch GUI. Only conduct setup operations.
  --share                       Share your installation.
  --server-port=PORT            The port number the GUI server should use.
  --username=USERNAME           Username for authentication.
EOF
}

# This gets the directory the script is run from so pathing can work relative to the script where needed.
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

get_abs_filename() {
  # $1 : relative filename
  if [ -d "$(dirname "$1")" ]; then
    echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
  fi
}

# This code handles the loading of parameter values in the following order of precedence:
# 1. Command-line arguments provided by the user
# 2. Values defined in install_config.yml (location order in configFileLocations or defined by -f)
# 3. Default values specified in the script
#
# The default values are specified in the `config_<variable>` variables using the
# `${config_<variable>:-<default_value>}` syntax. If the config file doesn't provide
# a value, the default value will be used.
#
# The config file values are read and stored in the `config_<variable>` variables. If
# the config file provides a value, it will override the default value.
#
# The CLI arguments are stored in the `CLI_ARGUMENTS` associative array.
#
# The loop `for key in "${!CLI_ARGUMENTS[@]}"; do` iterates through the CLI arguments
# and overwrites the corresponding `config_<variable>` variable if a CLI argument
# was provided.
#
# The final values for each option are assigned to their respective script defaults
# (e.g., `BRANCH="$config_Branch"`).

USER_CONFIG_FILE=""
declare -A CLI_ARGUMENTS

while getopts ":vb:d:f:g:il:nprst:ux-:" opt; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$opt" = "-" ]; then   # long option: reformulate OPT and OPTARG
    opt="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#"$opt"}" # extract long option argument (maybe empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case $opt in
  b | branch) CLI_ARGUMENTS["branch"]="$OPTARG" ;;
  d | dir) CLI_ARGUMENTS["dir"]="$OPTARG" ;;
  f | file) USER_CONFIG_FILE="$OPTARG" ;;
  g | git-repo) CLI_ARGUMENTS["gitRepo"]="$OPTARG" ;;
  h | help) display_help && exit 0 ;;
  i | interactive) CLI_ARGUMENTS["interactive"]="true" ;;
  l | log-dir) CLI_ARGUMENTS["logDir"]="$OPTARG" ;;
  n | no-setup) CLI_ARGUMENTS["noSetup"]="true" ;;
  r | repair) CLI_ARGUMENTS["repair"]="true" ;;
  s | skip-space-check) CLI_ARGUMENTS["skipSpaceCheck"]="true" ;;
  t | torch-version) CLI_ARGUMENTS["torchVersion"]="true" ;;
  u | update) CLI_ARGUMENTS["update"]="true" ;;
  v) ((CLI_ARGUMENTS["verbosity"] = CLI_ARGUMENTS["verbosity"] + 1)) ;;
  headless) CLI_ARGUMENTS["headless"]="true" ;;
  inbrowser) CLI_ARGUMENTS["inbrowser"]="true" ;;
  listen) CLI_ARGUMENTS["listen"]="$OPTARG" ;;
  password) CLI_ARGUMENTS["password"]="$OPTARG" ;;
  server-port) CLI_ARGUMENTS["serverPort"]="$OPTARG" ;;
  setup-only) CLI_ARGUMENTS["setupOnly"]="true" ;;
  share) CLI_ARGUMENTS["share"]="true" ;;
  username) CLI_ARGUMENTS["username"]="$OPTARG" ;;
  *) display_help && exit 0 ;;
  esac
done
shift $((OPTIND - 1))

# This reads a YAML configuration file and extracts default argument values.
# It stores these values in variables with a specified prefix.
# The function supports two sections: 'arguments' and 'kohya_gui_arguments'.
# For each argument in the sections, it looks for a 'name' and a 'default' value,
# and then stores the default value in a variable named "${prefix}_<name>"
# or "${prefix}_gui_<name>" for the respective sections.
parse_and_validate_yaml() {
  local yaml_file="$1"
  local prefix="$2"
  local line key value state section

  state="none"
  section="none"

  while IFS= read -r line; do
    # echo "$line"
    if [[ "$line" =~ ^[[:space:]]*(setup_arguments|kohya_gui_arguments):[[:space:]]*$ ]]; then
      section="${BASH_REMATCH[1]}"
      state="none"
      # echo "Section: $section"
    elif [[ "$section" != "none" ]] && [[ "$line" =~ ^[[:space:]]*-?[[:space:]]*name:[[:space:]]*([^[:space:]#]+)$ ]]; then
      key="${BASH_REMATCH[1]}"
      state="searching_value"
      # echo "Key: $key"
    elif [[ "$state" == "searching_value" ]] && [[ "$line" =~ ^[[:space:]]*value:[[:space:]]*([^[:space:]].*[^[:space:]])?[[:space:]]*$ ]]; then
      value="${BASH_REMATCH[1]}"
      # echo "Found value for $key: $value"
      if [[ "$section" == "setup_arguments" ]]; then
        eval "${prefix}_${key}=\"$value\""
      elif [[ "$section" == "kohya_gui_arguments" ]]; then
        eval "${prefix}_${key}=\"$value\""
      fi
      state="none"
    fi
  done <"$yaml_file"
}

configFileLocations=(
  "$SCRIPT_DIR/config_files/installation/install_config.yml"
  "$HOME/.kohya_ss/install_config.yml"
  "$SCRIPT_DIR/install_config.yml"
)

# Load and merge default config files
for location in "${configFileLocations[@]}"; do
  # echo "Parsing $location"
  if [ -f "$location" ]; then
    parse_and_validate_yaml "$location" "config"
    # echo "Parsed $location"
  fi
done

# Load and merge user-specified config file
if [ -n "$USER_CONFIG_FILE" ] && [ -f "$USER_CONFIG_FILE" ]; then
  parse_and_validate_yaml "$USER_CONFIG_FILE" "config"
fi

# Set default values only if they haven't been set by the config files
config_branch="${config_branch:-master}"
config_dir="${config_dir:-$SCRIPT_DIR}"
config_gitRepo="${config_gitRepo:-https://github.com/bmaltais/kohya_ss.git}"
config_headless="${config_headless:-false}"
config_interactive="${config_interactive:-false}"
config_noSetup="${config_noSetup:-false}"
config_repair="${config_repair:-false}"
config_setupOnly="${config_setupOnly:-false}"
config_skipSpaceCheck="${config_skipSpaceCheck:-false}"
config_torchVersion="${config_torchVersion:-1}"
config_update="${config_update:-false}"
config_verbosity="${config_verbosity:-0}"
config_listen="${config_listen:-127.0.0.1}"
config_username="${config_username:-}"
config_password="${config_password:-}"
config_serverPort="${config_serverPort:-0}"
config_inbrowser="${config_inbrowser:-false}"
config_share="${config_share:-false}"

# Override config values with CLI arguments
for key in "${!CLI_ARGUMENTS[@]}"; do
  configVar="config_$key"
  # echo "Processing CLI argument $key with value ${CLI_ARGUMENTS[$key]}"
  if [[ -n ${CLI_ARGUMENTS[$key]} ]]; then # Check if the CLI argument is not empty
    if [[ "$key" == "dir" ]]; then         # If the argument is the directory, convert to an absolute path
      # echo "CLI_ARGUMENTS dir: ${CLI_ARGUMENTS[$key]}"
      # echo "get_abs_filename: $(get_abs_filename "${CLI_ARGUMENTS[$key]}")"
      eval "$configVar=$(get_abs_filename "${CLI_ARGUMENTS[$key]}")"
    else
      eval "$configVar=${CLI_ARGUMENTS[$key]}"
    fi
  fi
done

# After CLI arguments have been processed, check if config_dir is _CURRENT_SCRIPT_DIR_
if [ "$config_dir" == "_CURRENT_SCRIPT_DIR_" ]; then
  config_dir="$SCRIPT_DIR"
fi

# After that, set dependent config values
config_logDir="${config_logDir:-$config_dir/logs}"

# Use the variables from the configuration file as default values
BRANCH="$config_branch"
DIR="$config_dir"
GIT_REPO="$config_gitRepo"
HEADLESS="$config_headless"
INTERACTIVE="$config_interactive"
LOG_DIR="$config_logDir"
NO_SETUP="$config_noSetup"
REPAIR="$config_repair"
SETUP_ONLY="$config_setupOnly"
SKIP_SPACE_CHECK="$config_skipSpaceCheck"
TORCH_VERSION="$config_torchVersion"
UPDATE="$config_update"
VERBOSITY="$config_verbosity"
GUI_LISTEN="$config_listen"
GUI_USERNAME="$config_username"
GUI_PASSWORD="$config_password"
GUI_SERVER_PORT="$config_serverPort"
GUI_INBROWSER="$config_inbrowser"
GUI_SHARE="$config_share"

# We set up logging here to match the format we are using downstream in the Python scripts
CURRENT_DATE=$(date +%Y-%m-%d)
CURRENT_TIME=$(date +%H%M)
LOG_LEVEL_NAME=$(case $VERBOSITY in
  0) echo "error" ;;
  1) echo "warning" ;;
  2) echo "info" ;;
  3) echo "debug" ;;
  *) echo "unknown" ;;
  esac)
if [ "$VERBOSITY" -eq 0 ]; then
  LOG_FILENAME="launcher_${CURRENT_TIME}.log"
else
  LOG_FILENAME="launcher_${CURRENT_TIME}_${LOG_LEVEL_NAME}.log"
fi

if [ ! -d "$LOG_DIR/$CURRENT_DATE/" ]; then
  mkdir -p "$LOG_DIR/$CURRENT_DATE/"
fi

LOG_FILE="$LOG_DIR/$CURRENT_DATE/$LOG_FILENAME"

# The color_code function takes a color name as input and returns the corresponding
# ANSI escape code for that color. These escape codes are used to set the color of
# the text displayed in the terminal.
#
# The parse_logger_args function processes named parameters like --color and --no-header
# in the arguments provided to the logging functions. It returns the corresponding
# color code and header flag (0 for displaying the header, 1 for hiding it).
#
# Example usage:
# log_warn "Warning message without header" --color=blue --no-header
color_code() {
  local color_name=$1
  case $color_name in
  black) echo "30" ;;
  red) echo "31" ;;
  green) echo "32" ;;
  yellow) echo "33" ;;
  blue) echo "34" ;;
  magenta) echo "35" ;;
  cyan) echo "36" ;;
  white) echo "37" ;;
  *) echo "97" ;; # Default to white
  esac
}

parse_logger_args() {
  local args=("$@")
  local color=""
  local no_header="0"

  for i in "${!args[@]}"; do
    case "${args[$i]}" in
    --color)
      i=$((i + 1))
      color=$(color_code "${args[$i]}")
      ;;
    --no-header)
      no_header="1"
      ;;
    esac
  done

  printf "%s %s" "$color" "$no_header"
}

log_debug() {
  local message
  message=$1
  local args
  args=$(parse_logger_args "${@:2}")

  local color
  color=$(echo "$args" | cut -d' ' -f1)
  local no_header
  no_header=$(echo "$args" | cut -d' ' -f2)

  if ((VERBOSITY >= 3)); then
    if ((no_header == 0)); then
      message="DEBUG: $message"
    fi
    printf "\033[${color}m%s\033[0m\n" "$message"
    echo "$message" >>"$LOG_FILE"
  fi
}

log_info() {
  local message
  message=$1
  local args
  args=$(parse_logger_args "${@:2}")

  local color
  color=$(echo "$args" | cut -d' ' -f1)
  local no_header
  no_header=$(echo "$args" | cut -d' ' -f2)

  if ((VERBOSITY >= 2)); then
    if ((no_header == 0)); then
      message="INFO: $message"
    fi
    printf "\033[${color}m%s\033[0m\n" "$message"
    echo "$message" >>"$LOG_FILE"
  fi
}

log_warn() {
  local message
  message=$1
  local args
  args=$(parse_logger_args "${@:2}")

  local color
  color=$(echo "$args" | cut -d' ' -f1)
  local no_header
  no_header=$(echo "$args" | cut -d' ' -f2)

  if ((VERBOSITY >= 1)); then
    if ((no_header == 0)); then
      message="WARNING: $message"
    fi
    printf "\033[${color}m%s\033[0m\n" "$message"
    echo "$message" >>"$LOG_FILE"
  fi
}

log_error() {
  local message
  message=$1
  local args
  args=$(parse_logger_args "${@:2}")

  local color
  color=$(echo "$args" | cut -d' ' -f1)
  local no_header
  no_header=$(echo "$args" | cut -d' ' -f2)

  if ((VERBOSITY >= 0)); then
    if ((no_header == 0)); then
      message="ERROR: $message"
    fi
    printf "\033[${color}m%s\033[0m\n" "$message"
    echo "$message" >>"$LOG_FILE"
  fi
}

log_critical() {
  local message
  message=$1
  local args
  args=$(parse_logger_args "${@:2}")

  local color
  color=$(echo "$args" | cut -d' ' -f1)
  local no_header
  no_header=$(echo "$args" | cut -d' ' -f2)

  if ((VERBOSITY >= 0)); then
    if ((no_header == 0)); then
      message="CRITICAL: $message"
    fi
    printf "\033[${color}m%s\033[0m\n" "$message"
    echo "$message" >>"$LOG_FILE"
  fi
}

log_debug "$0 launching."

# Debug variable dump at max verbosity
log_debug "BRANCH: $BRANCH
DIR: $DIR
GIT_REPO: $GIT_REPO
Config file location: $USER_CONFIG_FILE
HEADLESS: $HEADLESS
INTERACTIVE: $INTERACTIVE
LOG_DIR: $LOG_DIR
REPAIR: $REPAIR
SKIP_SPACE_CHECK: $SKIP_SPACE_CHECK
TORCH_VERSION: $TORCH_VERSION
UPDATE: $UPDATE
Skip Setup: $NO_SETUP
VERBOSITY: $VERBOSITY
Script directory is ${SCRIPT_DIR}."

# Shared functions
get_os_info() {
  declare -A os_info
  os_info["name"]="Unknown"
  os_info["family"]="Unknown"
  os_info["version"]="Unknown"

  case "$(uname -s)" in
  Darwin*)
    os_info["name"]="macOS"
    os_info["family"]="macOS"
    os_info["version"]=$(sw_vers -productVersion)
    ;;
  MINGW64_NT* | MSYS_NT* | CYGWIN_NT*)
    os_info["name"]="Windows"
    os_info["family"]="Windows"
    os_info["version"]=$(systeminfo | grep "^OS Version" | awk -F: '{print $2}' | tr -d '[:space:]')
    ;;
  Linux*)
    if [ -f /etc/os-release ]; then
      os_info["name"]=$(grep -oP '(?<=^ID=).*' /etc/os-release | tr -d '"')
      os_info["family"]=$(grep -oP '(?<=^ID_LIKE=).*' /etc/os-release | tr -d '"')
      os_info["version"]=$(grep -oP '(?<=^VERSION=).*' /etc/os-release | tr -d '"')
    elif [ -f /etc/redhat-release ]; then
      os_info["name"]=$(awk '{print $1}' /etc/redhat-release)
      os_info["family"]="RedHat"
      os_info["version"]=$(awk '{print $3}' /etc/redhat-release)
    fi

    if [ "${os_info["name"]}" == "Unknown" ]; then
      local uname_output
      uname_output=$(uname -a)

      case $uname_output in
      *Ubuntu*)
        os_info["name"]="Ubuntu"
        os_info["family"]="Ubuntu"
        ;;
      *Debian*)
        os_info["name"]="Debian"
        os_info["family"]="Debian"
        ;;
      *Red\ Hat* | *CentOS*)
        os_info["name"]="RedHat"
        os_info["family"]="RedHat"
        ;;
      *Fedora*)
        os_info["name"]="Fedora"
        os_info["family"]="Fedora"
        ;;
      *SUSE*)
        os_info["name"]="openSUSE"
        os_info["family"]="SUSE"
        ;;
      *Arch*)
        os_info["name"]="Arch"
        os_info["family"]="Arch"
        ;;
      *)
        os_info["name"]="Generic Linux"
        os_info["family"]="Generic Linux"
        ;;
      esac
    fi
    ;;

  FreeBSD*)
    os_info["name"]="FreeBSD"
    os_info["family"]="FreeBSD"
    os_info["version"]=$(uname -r)
    ;;
  *)
    os_info["name"]="Unknown"
    os_info["family"]="Unknown"
    ;;
  esac

  declare -p os_info
}

# Eval here to make it global for the other functions
eval "$(get_os_info)"

is_admin() {
  case "${os_info["name"]}" in
  "Windows" | "MINGW64_NT" | "CYGWIN_NT")
    if net session >/dev/null 2>&1; then
      return 0
    else
      return 1
    fi
    ;;
  *)
    if [ "$EUID" = 0 ] || [ "$(id -u)" = 0 ] || [ "$UID" = 0 ]; then
      return 0
    else
      return 1
    fi
    ;;
  esac
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

  case "${os_info["name"]}" in
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

  case "${os_info["family"]}" in
  "Windows")
    if [ -d "$DIR" ]; then
      folder="$DIR"
    elif [ -d "$PARENT_DIR" ]; then
      folder="$PARENT_DIR"
    elif [ -d "$(echo "$DIR" | cut -d '/' -f1)" ]; then
      folder="$(echo "$DIR" | cut -d '/' -f1)"
    else
      echo "We are assuming a C: drive install for space-checking purposes." >&2
      folder='C:'
    fi
    ;;
  *)
    if [ -d "$DIR" ]; then
      folder="$DIR"
    elif [ -d "$PARENT_DIR" ]; then
      folder="$PARENT_DIR"
    elif [ -d "$(echo "$DIR" | cut -d '/' -f2)" ]; then
      folder="$(echo "$DIR" | cut -d '/' -f2)"
    else
      log_info "We are assuming a root drive install for space-checking purposes."
      folder='/'
    fi
    ;;
  esac

  # Return available space in GB
  if [[ "${os_info["family"]}" == "Windows" ]]; then
    local drive
    drive=$(powershell.exe -Command "(Get-Item -Path '$folder').Root.Name")
    drive=${drive::-1}

    # shellcheck disable=SC2046
    powershell.exe -Command "Get-WmiObject -Class Win32_LogicalDisk -Filter 'DeviceID=''$drive''' \
      | Select-Object -ExpandProperty FreeSpace" |
      awk '/[0-9]+/ {print int($1 / 1024 / 1024 / 1024)}'

  else
    df --output=avail -B1G "$folder" | tail -n1 | awk '{print $1}'
  fi
}

check_storage_space() {
  if [ "$SKIP_SPACE_CHECK" = false ]; then
    if [[ ! -z "$(size_available)" && "$(size_available)" =~ ^[0-9]+$ && "$(size_available)" -lt "$1" ]]; then
      log_critical "You have less than 10Gb of free space. This installation may fail." --color red
      local MSGTIMEOUT=10 # In seconds
      local MESSAGE="Continuing in..."
      echo "Press control-c to cancel the installation."
      for ((i = MSGTIMEOUT; i >= 0; i--)); do
        printf "\r${MESSAGE} %ss. " "${i}"
        sleep 1
      done
    fi
  fi
}

package_exists() {
  local package="$1"

  if [[ "${os_info["name"]}" =~ Ubuntu|Debian || "${os_info["family"]}" =~ Ubuntu|Debian ]]; then
    dpkg -s "$package" >/dev/null 2>&1
    return $?
  elif [[ "${os_info["name"]}" =~ (Fedora|CentOS|RedHat) ]] || [[ "${os_info["family"]}" =~ (Fedora|CentOS|RedHat) ]]; then
    rpm -q "$package" >/dev/null 2>&1
    return $?
  elif [[ "${os_info["name"]}" =~ (Arch|Manjaro) ]] || [[ "${os_info["family"]}" =~ (Arch|Manjaro) ]]; then
    pacman -Qi "$package" >/dev/null 2>&1
    return $?
  elif [[ "${os_info["name"]}" =~ openSUSE ]] || [[ "${os_info["family"]}" =~ openSUSE ]]; then
    zypper if "$package" | grep "Installed: Yes" >/dev/null 2>&1
    return $?
  elif [[ "${os_info["name"]}" =~ FreeBSD ]] || [[ "${os_info["family"]}" =~ FreeBSD ]]; then
    pkg info "$package" >/dev/null 2>&1
    return $?
  else
    log_critical "Unsupported operating system for package_exists function." --color red
    return 1
  fi
}

git_is_installed() {
  if command -v git >/dev/null 2>&1; then
    local git_version
    git_version=$(git --version 2>&1)
  else
    return 1
  fi

  if [[ $git_version =~ ^git\ version ]]; then
    return 0
  else
    return 1
  fi
}

install_git_windows() {
  local interactive="$1"
  local package_manager_found=false

  if git_is_installed; then
    log_critical "git is already installed." --no-header --color yellow
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

  if ! "$package_manager_found"; then
    if is_admin; then
      local install_scope
      install_scope=$(update_install_scope "$interactive")
    else
      install_scope="user"
    fi

    local git_url="https://github.com/git-for-windows/git/releases/download/v2.35.1.windows.1/Git-2.35.1-64-bit.exe"
    local git_installer_name="Git-2.35.1-64-bit.exe"
    local downloads_folder="$HOME/Downloads"
    local installer_path="${downloads_folder}/${git_installer_name}"

    if [ ! -f "$installer_path" ]; then
      if ! curl -o "$installer_path" -L "$git_url"; then
        log_critical "Failed to download Git. Please check your internet connection or provide a pre-downloaded installer." --color red
        exit 1
      fi
    fi

    local installer_path_windows
    installer_path_windows=$(cygpath -w "$installer_path")

    if [ "$install_scope" = "user" ]; then
      powershell.exe -Command "Start-Process -FilePath \"$installer_path_windows\" -ArgumentList '/VERYSILENT /NORESTART /LOG=\"$LOG_FILE\" /NOICONS /COMPONENTS=\"icons,ext\\reg\\shellhere,assoc,assoc_sh\"' -Wait -NoNewWindow"
    else
      powershell.exe -Command "Start-Process -FilePath \"$installer_path_windows\" -ArgumentList '/VERYSILENT /NORESTART /LOG=\"$LOG_FILE\" /NOICONS /COMPONENTS=\"icons,ext\\reg\\shellhere,assoc,assoc_sh\" /ALLUSERS=1' -Wait -NoNewWindow"
    fi

    rm -f "$installer_path"
  fi
}

install_git() {
  shopt -s nocasematch
  if git_is_installed; then
    log_critical "git is already installed." --no-header --color yellow
    return 0
  fi

  # Windows
  if [[ "${os_info["name"]}" =~ Windows || "${os_info["family"]}" =~ Windows ]]; then
    install_git_windows "$INTERACTIVE"

  # macOS
  elif [[ "${os_info["name"]}" =~ macOS || "${os_info["family"]}" =~ macOS ]]; then
    if command -v brew >/dev/null 2>&1; then
      ! package_exists git && brew install git
    else
      log_critical "Please install Homebrew first to continue with Git installation." --no-header --color yellow
      log_critical "You can find that here: https://brew.sh" --no-header --color yellow
      exit 1
    fi

  # Ubuntu/Debian
  elif [[ "${os_info["name"]}" =~ Ubuntu|Debian || "${os_info["family"]}" =~ Ubuntu|Debian ]]; then
    if is_admin; then
      ! package_exists git && (sudo apt-get update && sudo apt-get install -y git)
    else
      log_critical "Admin privileges are required to install Git. Please run the script as root or with sudo."--color red
      exit 1
    fi
  # RedHat
  elif [[ "${os_info["name"]}" =~ Fedora|CentOS|RedHat || "${os_info["family"]}" =~ Fedora|CentOS|RedHat ]]; then
    if is_admin; then
      ! package_exists git && sudo dnf install -y git
    else
      log_critical "Admin privileges are required to install Git. Please run the script as root or with sudo."--color red
      exit 1
    fi

  # Arch
  elif [[ "${os_info["name"]}" =~ Arch|Manjaro || "${os_info["family"]}" =~ Arch|Manjaro ]]; then
    if is_admin; then
      ! package_exists git && sudo pacman -Sy --noconfirm git
    else
      log_critical "Admin privileges are required to install Git. Please run the script as root or with sudo."--color red
      exit 1
    fi

  # openSUSE
  elif [[ "${os_info["name"]}" =~ openSUSE || "${os_info["family"]}" =~ openSUSE ]]; then
    if is_admin; then
      ! package_exists git && sudo zypper install -y git
    else
      log_critical "Admin privileges are required to install Git. Please run the script as root or with sudo."--color red
      exit 1
    fi

  # FreeBSD
  elif [[ "${os_info["name"]}" =~ FreeBSD ]]; then
    if is_admin; then
      ! package_exists git && sudo pkg install -y git
    else
      log_critical "Admin privileges are required to install Git. Please run the script as root or with sudo."--color red
      exit 1
    fi
  else
    log_critical "Admin privileges are required to install Git. Please run the script as root or with sudo."--color red
    exit 1
  fi
  shopt -u nocasematch
}

python310_is_installed() {
  if command -v python >/dev/null 2>&1; then
    local python_version
    python_version=$(python --version 2>&1)
  elif command -v py >/dev/null 2>&1; then
    local python_version
    python_version=$(py -3.10 --version 2>&1)
  else
    return 1
  fi

  if [[ $python_version =~ ^Python\ 3\.10 ]]; then
    return 0
  else
    return 1
  fi
}

install_python310_windows() {
  local interactive="$1"

  if python310_is_installed; then
    log_critical "Python 3.10 is already installed." --no-header --color yellow
    return 0
  fi

  if command -v scoop >/dev/null 2>&1; then
    local package_manager_found=false
  elif command -v choco >/dev/null 2>&1; then
    choco install python --version=3.10 --params "/IncludeTclTk"
    local package_manager_found=true
  elif command -v winget >/dev/null 2>&1; then
    winget install --id Python.Python --version 3.10.*
    local package_manager_found=true
  fi

  if [ "$package_manager_found" = false ]; then
    if is_admin; then
      local install_scope
      install_scope=$(update_install_scope "$interactive")
    else
      install_scope="user"
    fi

    local python_url="https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe"
    local python_installer_name="python-3.10.0-amd64.exe"
    local downloads_folder="$HOME/Downloads"
    local installer_path="${downloads_folder}/${python_installer_name}"

    if [ ! -f "$installer_path" ]; then
      if ! curl -o "$installer_path" -L "$python_url"; then
        log_critical "Failed to download Python 3.10. Please check your internet connection or provide a pre-downloaded installer." --color red
        exit 1
      fi
    fi

    local installer_path_windows
    installer_path_windows=$(cygpath -w "$installer_path")

    if [ "$install_scope" = "user" ]; then
      powershell.exe -Command "Start-Process -FilePath \"$installer_path_windows\" -ArgumentList '/VERYSILENT /NORESTART /LOG=\"$LOG_FILE\" /NOICONS /COMPONENTS=\"icons,ext\\reg\\shellhere,assoc,assoc_sh\"' -Wait -NoNewWindow"
    else
      powershell.exe -Command "Start-Process -FilePath \"$installer_path_windows\" -ArgumentList '/VERYSILENT /NORESTART /LOG=\"$LOG_FILE\" /NOICONS /COMPONENTS=\"icons,ext\\reg\\shellhere,assoc,assoc_sh\" /ALLUSERS=1' -Wait -NoNewWindow"
    fi

    rm -f "$installer_path"
  fi
}

install_python_and_tk() {
  local missing_packages=()

  # Ubuntu/Debian
  if [[ "${os_info["name"]}" =~ Ubuntu|Debian || "${os_info["family"]}" =~ Ubuntu|Debian ]]; then
    package_exists python3.10 || missing_packages+=("python3.10")
    package_exists python3-tk || missing_packages+=("python3-tk")
    package_exists python3.10-venv || missing_packages+=("python3.10-venv")
    [[ ${#missing_packages[@]} -ne 0 ]] && sudo apt-get update && sudo apt-get install -y "${missing_packages[@]}"

  # Redhat
  elif [[ "${os_info["name"]}" =~ Fedora|CentOS|RedHat || "${os_info["family"]}" =~ Fedora|CentOS|RedHat ]]; then
    package_exists python310 || missing_packages+=("python310")
    package_exists python3-tkinter || missing_packages+=("python3-tkinter")
    [[ ${#missing_packages[@]} -ne 0 ]] && sudo yum install -y "${missing_packages[@]}"

  # Arch
  elif [[ "${os_info["name"]}" =~ Arch|Manjaro || "${os_info["family"]}" =~ Arch|Manjaro ]]; then
    package_exists python310 || missing_packages+=("python310")
    package_exists tk || missing_packages+=("tk")
    [[ ${#missing_packages[@]} -ne 0 ]] && sudo pacman -Syu --needed "${missing_packages[@]}"

  # openSUSE
  elif [[ "${os_info["name"]}" =~ openSUSE || "${os_info["family"]}" =~ openSUSE ]]; then
    package_exists python310 || missing_packages+=("python310")
    package_exists python3-tk || missing_packages+=("python3-tk")
    [[ ${#missing_packages[@]} -ne 0 ]] && sudo zypper in "${missing_packages[@]}"

  # FreeBSD
  elif [[ "${os_info["name"]}" =~ FreeBSD ]]; then
    package_exists py310-python || missing_packages+=("py310-python")
    package_exists py310-tkinter || missing_packages+=("py310-tkinter")
    [[ ${#missing_packages[@]} -ne 0 ]] && sudo pkg install "${missing_packages[@]}"

  # macOS
  elif [[ "${os_info["name"]}" =~ macOS || "${os_info["family"]}" =~ macOS ]]; then
    if command -v brew >/dev/null 2>&1; then
      ! brew list --versions python@3.10 >/dev/null && missing_packages+=("python@3.10")
      ! brew list --versions tcl-tk >/dev/null && missing_packages+=("tcl-tk")
      [[ ${#missing_packages[@]} -ne 0 ]] && brew install "${missing_packages[@]}"
      brew link --overwrite --force python@3.10
    else
      log_critical "Please install Homebrew first to continue with Python 3.10 and Tk installation." --no-header --color yellow
      log_c "You can find that here: https://brew.sh" --no-header --color yellow
      exit 1
    fi

  # Windows
  elif [[ "${os_info["name"]}" =~ Windows || "${os_info["family"]}" == "Windows" ]]; then
    install_python310_windows "$INTERACTIVE"
  else
    log_critical "Unsupported operating system. Please install Python 3.10 and Python Tk 3.10 manually." --color red
    log_critical "For manual installation, you can download the official Python tar.gz packages from:" --no-header --color yellow
    log_critical "https://www.python.org/downloads/source/" --no-header --color yellow
    exit 1
  fi
}

display_countdown() {
  local countdown=$1
  printf "Press 'y' to skip the countdown and continue or 'n' to cancel the installation.\n"
  local continue_installation=1

  for ((i = 0; i < countdown; i++)); do
    local remaining_time=$((countdown - i))
    printf "\rContinuing in %d... " "$remaining_time"

    local key=""
    local timeout=10

    while ((timeout > 0)); do
      read -r -s -N 1 -t 0.1 key
      if [[ -n $key ]]; then
        break
      fi
      ((timeout--))
    done

    if [[ -n $key ]]; then
      if [[ $key == "y" ]]; then
        continue_installation=0
        break
      elif [[ $key == "n" ]]; then
        continue_installation=1
        break
      fi
    fi
  done

  printf "\n"
  return $continue_installation
}

install_vc_redist_windows() {
  if [ "${os_info["family"]}" != "Windows" ]; then
    return 0
  fi

  local vc_redist_oldest_year=2015
  local vc_redist_newest_year=2022

  if ! is_vc_redist_installed "$vc_redist_oldest_year" "$vc_redist_newest_year"; then
    if ! is_admin; then
      log_critical "Admin privileges are required to install the Visual Studio redistributable. The script will attempt to run with elevated privileges." --color red
      display_countdown 15
      continue_installation=$?

      if ((continue_installation == 0)); then
        printf "Continuing with the installation.\n"
      else
        log_critical "VC Redist Installation cancelled." --color red
        log_critical "Please manually install VC via the following URL: " --no-header --color yellow
        log_critical "https://aka.ms/vs/17/release/vc_redist.x64.exe" --no-header --color yellow
        exit 1
      fi
    fi

    local vc_redist_url="https://aka.ms/vs/17/release/vc_redist.x64.exe"
    local vc_redist_installer_name="vc_redist.x64.exe"
    local downloads_folder="$HOME/Downloads"
    local installer_path="${downloads_folder}/${vc_redist_installer_name}"

    log_critical "Downloading VC Redist installer." --no-header

    if [ ! -f "$installer_path" ]; then
      if ! curl -# -o "$installer_path" -L "$vc_redist_url"; then
        log_critical "Failed to download Visual Studio redistributables. Please check your internet connection or provide a pre-downloaded installer."
        exit 1
      else
        log_critical "Download complete!" --no-header
      fi
    fi

    log_critical "The UAC Prompt for the installer should appear; you may need to find that in the task bar." --no-header --color yellow

    local installer_path_windows
    installer_path_windows=$(cygpath -w "$installer_path")

    local powershell_exit_code
    powershell_exit_code=$(powershell.exe -Command "(Start-Process -FilePath \"$installer_path_windows\" -ArgumentList '/install /quiet /norestart' -Wait -Passthru -NoNewWindow).ExitCode")

    if [ -z "$powershell_exit_code" ]; then
      log_critical "Failed to retrieve exit code from the VC Redist installation process. Exiting the script." --color red
      exit 1
    elif [ "$powershell_exit_code" -eq 3010 ]; then
      # Exit code 3010 means a reboot is required, so handle this case separately
      log_critical "VC Redistributable installation requires a reboot." --color yellow
      log_critical "Exiting script to avoid any issues. Please reboot the computer and run this setup again." --color red
      exit 0
    elif [ "$powershell_exit_code" -ne 0 ]; then
      log_critical "VC Redist installation failed or was cancelled. Exiting the script." --color red
      exit 1
    else
      if [ -f "$installer_path" ]; then
        rm -f "$installer_path"
      fi
    fi

    rm -f "$installer_path"
  else
    log_critical "VC Redist already installed." --no-header --color yellow
  fi
}

is_vc_redist_installed() {
  local vc_redist_oldest_year=$1
  local vc_redist_newest_year=$2
  local registry_keys=("HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall" "HKLM:\\SOFTWARE\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall" "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall")
  local installed=1

  for key in "${registry_keys[@]}"; do
    local installed_software
    installed_software=$(powershell.exe -Command "Get-ChildItem -Path ${key} | Get-ItemProperty | Select-Object -Property DisplayName" | tr -d '\000')

    local matching_software
    matching_software=$(echo "$installed_software" | grep -P "Microsoft Visual C\+\+ ${vc_redist_oldest_year}-${vc_redist_newest_year} Redistributable \(x64\) .*")

    if [ "$matching_software" ]; then
      installed=0
      break
    fi
  done

  return $installed
}

run_launcher() {
  if command -v python3.10 >/dev/null 2>&1; then
    local PYTHON_EXEC="python3.10"
  elif command -v python >/dev/null 2>&1 && [ "$(python -c 'import sys; print(sys.version_info[:2])')" = "(3, 10)" ]; then
    local PYTHON_EXEC="python"
  elif command -v python3 >/dev/null 2>&1 && [ "$(python3 -c 'import sys; print(sys.version_info[:2])')" = "(3, 10)" ]; then
    local PYTHON_EXEC="python3"
  else
    log_critical "Error: Python 3.10 is required to run this script. Please install Python 3.10 and try again." --color red
    exit 1
  fi

  # Print a literal string to give us some space before executing launcher.py
  log_critical $'Launcher.py is now executing.\n\n' --no-header

  # shellcheck disable=SC2046
  "$PYTHON_EXEC" launcher.py \
    --branch="$BRANCH" \
    --dir="$DIR" \
    --git-repo="$GIT_REPO" \
    $([ "$HEADLESS" = "true" ] && echo "--headless") \
    $([ "$INTERACTIVE" = "true" ] && echo "--interactive") \
    --log-dir="$LOG_DIR" \
    $([ "$NO_SETUP" = "true" ] && echo "--no-setup") \
    $([ "$REPAIR" = "true" ] && echo "--repair") \
    $([ "$SETUP_ONLY" = "true" ] && echo "--setup-only") \
    $([ "$SKIP_SPACE_CHECK" = "true" ] && echo "--skipspacecheck") \
    $([ "$UPDATE" = "true" ] && echo "--update") \
    --listen="$GUI_LISTEN" \
    --username="$GUI_USERNAME" \
    --password="$GUI_PASSWORD" \
    --server-port="$GUI_SERVER_PORT" \
    $([ "$GUI_INBROWSER" = "true" ] && echo "--inbrowser") \
    $([ "$GUI_SHARE" = "true" ] && echo "--share") \
    -v "$VERBOSITY"
}

function main() {
  if ! "$NO_SETUP"; then
    DIR="$(normalize_path "$DIR")"

    # Warn user and give them a chance to cancel install if less than 5Gb is available on storage device
    check_storage_space 5
    install_git
    install_python_and_tk
    install_vc_redist_windows
  fi

  run_launcher
}

main
