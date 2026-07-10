#!/usr/bin/env bash
# Tkinter capability preflight helpers for setup.sh.
#
# Detection is based on whether a concrete interpreter can `import tkinter`,
# not on whether a distro package (python3-tk, etc.) is registered with apt/rpm.
# Package names appear only in remediation messages.
#
# Safe to source from tests:
#   source setup/tkinter_preflight.sh
#   python_has_tkinter /path/to/python && echo ok

# Returns 0 if the given interpreter can import tkinter, else 1.
# Arg: absolute path or command name of the Python executable.
python_has_tkinter() {
  local py="${1:?python_has_tkinter requires a python command}"

  # Accept either a command on PATH or an absolute/relative executable path.
  if ! command -v "$py" >/dev/null 2>&1 && [ ! -x "$py" ]; then
    return 1
  fi

  "$py" -c "import tkinter" >/dev/null 2>&1
}

# Resolve which Python the installer should probe for tkinter.
# Prefer an explicit override, then an active conda env, then get_python_command
# if that function is already defined (setup.sh), else a local fallback.
resolve_tkinter_python() {
  if [ -n "${1:-}" ]; then
    echo "$1"
    return 0
  fi
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    echo "${CONDA_PREFIX}/bin/python"
    return 0
  fi
  if declare -F get_python_command >/dev/null 2>&1; then
    get_python_command
    return 0
  fi
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
  elif command -v python3.10 >/dev/null 2>&1; then
    echo "python3.10"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  else
    echo "python"
  fi
}

# Print distro-specific install hints when import fails.
# Arg: one of ubuntu|fedora|arch|opensuse|macos|generic
print_tkinter_install_hint() {
  local family="${1:-generic}"
  echo "This script needs a Python interpreter that can import tkinter."
  echo "The selected interpreter cannot import tkinter. Install the matching Tk bindings, for example:"
  echo " "
  case "$family" in
    ubuntu | debian)
      echo "  sudo apt update -y && sudo apt install -y python3-tk"
      echo "  # or a versioned package matching your interpreter, e.g. python3.11-tk"
      ;;
    fedora | rhel | centos)
      echo "  sudo dnf install -y python3-tkinter"
      ;;
    arch)
      echo "  sudo pacman --noconfirm -S tk"
      ;;
    opensuse)
      echo "  sudo zypper install -y python-tk"
      ;;
    macos)
      echo "  brew install python-tk@3.11   # or python-tk@3.10"
      ;;
    *)
      echo "  Install Tcl/Tk support for your Python build (e.g. python3-tk / python3-tkinter / brew python-tk)."
      ;;
  esac
  echo " "
  echo "Custom builds (pyenv, source, conda) must include Tcl/Tk; installing a system package alone may not help."
}

# Map distro/family strings from get_distro_* to a hint family key.
tkinter_hint_family_from_distro() {
  local distro="${1:-}"
  local family="${2:-}"
  if echo "$distro $family" | grep -Eqi "ubuntu|debian"; then
    echo "ubuntu"
  elif echo "$distro $family" | grep -Eqi "fedora|centos|redhat|rhel"; then
    echo "fedora"
  elif echo "$distro $family" | grep -Eqi "arch"; then
    echo "arch"
  elif echo "$distro $family" | grep -Eqi "opensuse|suse"; then
    echo "opensuse"
  else
    echo "generic"
  fi
}

# Run the full Linux preflight: resolve python, test import, print hints + exit 1 on failure.
# Env:
#   RUNPOD=true  — attempt apt install python3-tk once, then re-check import
# Args (optional):
#   $1 explicit python command
#   $2 distro name string
#   $3 distro family string
require_python_tkinter() {
  local py
  local hint_family
  py="$(resolve_tkinter_python "${1:-}")"
  hint_family="$(tkinter_hint_family_from_distro "${2:-}" "${3:-}")"

  if python_has_tkinter "$py"; then
    echo "Python TK found ($py)..."
    return 0
  fi

  if [ "${RUNPOD:-false}" = true ] && [ "$hint_family" = "ubuntu" ]; then
    echo "Tkinter missing for $py; attempting apt install of python3-tk (RUNPOD)..."
    apt update -y && apt install -y python3-tk || true
    if python_has_tkinter "$py"; then
      echo "Python TK found after install ($py)..."
      return 0
    fi
  fi

  print_tkinter_install_hint "$hint_family"
  return 1
}
