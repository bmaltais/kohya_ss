#!/bin/bash
# The initial setup script to prep the environment on macOS
# xformers has been omitted as that is for Nvidia GPUs only

if ! command -v brew >/dev/null; then
  echo "Please install homebrew first. This is a requirement for the remaining setup."
  echo "You can find that here: https://brew.sh"
  exit 1
fi

# Install base python packages
echo "Installing Python 3.10 if not found."
brew ls --versions python@3.10 >/dev/null || brew install python@3.10
echo "Installing Python-TK 3.10 if not found."
brew ls --versions python-tk@3.10 >/dev/null || brew install python-tk@3.10

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
  python -m pip install --use-pep517 --upgrade -r requirements_macos.txt
  accelerate config
  echo -e "Setup finished! Run ./gui_macos.sh to start."
else
  echo "Python not found. Please ensure you install Python."
  echo "The brew command for Python 3.10 is: brew install python@3.10"
  exit 1
fi