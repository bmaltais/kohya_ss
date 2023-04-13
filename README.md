# Kohya's GUI

This repository provides a Windows-focused Gradio GUI for [Kohya's Stable Diffusion trainers](https://github.com/kohya-ss/sd-scripts). The GUI allows you to set the training parameters and generate and run the required CLI commands to train the model.

If you run on Linux and would like to use the GUI, there is now a port of it as a docker container. You can find the project [here](https://github.com/P2Enjoy/kohya_ss-docker).

### Table of Contents

- [Tutorials](#tutorials)
- [Required Dependencies](#required-dependencies)
  - [Linux/macOS](#linux-and-macos-dependencies)
- [Installation](#installation)
    - [Linux/macOS](#linux-and-macos)
      - [Default Install Locations](#install-location)
    - [Windows](#windows)
    - [CUDNN 8.6](#optional--cudnn-86)
- [Upgrading](#upgrading)
  - [Windows](#windows-upgrade)
  - [Linux/macOS](#linux-and-macos-upgrade)
- [Launching the GUI](#starting-gui-service)
  - [Windows](#launching-the-gui-on-windows)
  - [Linux/macOS](#launching-the-gui-on-linux-and-macos)
  - [Direct Launch via Python Script](#launching-the-gui-directly-using-kohyaguipy)
- [Dreambooth](#dreambooth)
- [Finetune](#finetune)
- [Train Network](#train-network)
- [LoRA](#lora)
- [Troubleshooting](#troubleshooting)
  - [Page File Limit](#page-file-limit)
  - [No module called tkinter](#no-module-called-tkinter)
  - [FileNotFoundError](#filenotfounderror)
- [Change History](#change-history)

## Tutorials

[How to Create a LoRA Part 1: Dataset Preparation](https://www.youtube.com/watch?v=N4_-fB62Hwk):

[![LoRA Part 1 Tutorial](https://img.youtube.com/vi/N4_-fB62Hwk/0.jpg)](https://www.youtube.com/watch?v=N4_-fB62Hwk)

[How to Create a LoRA Part 2: Training the Model](https://www.youtube.com/watch?v=k5imq01uvUY):

[![LoRA Part 2 Tutorial](https://img.youtube.com/vi/k5imq01uvUY/0.jpg)](https://www.youtube.com/watch?v=k5imq01uvUY)

## Required Dependencies

- Install [Python 3.10](https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe) 
  - make sure to tick the box to add Python to the 'PATH' environment variable
- Install [Git](https://git-scm.com/download/win)
- Install [Visual Studio 2015, 2017, 2019, and 2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Linux and macOS dependencies

These dependencies are taken care of via `setup.sh` in the installation section. No additional steps should be needed unless the scripts inform you otherwise.

## Installation

### Runpod
Follow the instructions found in this discussion: https://github.com/bmaltais/kohya_ss/discussions/379

### Linux and macOS
In the terminal, run

```
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
# May need to chmod +x ./setup.sh if you're on a machine with stricter security.
# There are additional options if needed for a runpod environment.
# Call 'setup.sh -h' or 'setup.sh --help' for more information.
./setup.sh
```

Setup.sh help included here:

```bash
Kohya_SS Installation Script for POSIX operating systems.

The following options are useful in a runpod environment,
but will not affect a local machine install.

Usage:
  setup.sh -b dev -d /workspace/kohya_ss -g https://mycustom.repo.tld/custom_fork.git
  setup.sh --branch=dev --dir=/workspace/kohya_ss --git-repo=https://mycustom.repo.tld/custom_fork.git

Options:
  -b BRANCH, --branch=BRANCH    Select which branch of kohya to check out on new installs.
  -d DIR, --dir=DIR             The full path you want kohya_ss installed to.
  -g REPO, --git_repo=REPO      You can optionally provide a git repo to check out for runpod installation. Useful for custom forks.
  -h, --help                    Show this screen.
  -i, --interactive             Interactively configure accelerate instead of using default config file.
  -n, --no-update               Do not update kohya_ss repo. No git pull or clone operations.
  -p, --public                  Expose public URL in runpod mode. Won't have an effect in other modes.
  -r, --runpod                  Forces a runpod installation. Useful if detection fails for any reason.
  -s, --skip-space-check        Skip the 10Gb minimum storage space check.
  -u, --no-gui                  Skips launching the GUI.
  -v, --verbose                 Increase verbosity levels up to 3.
```

#### Install location

The default install location for Linux is where the script is located if a previous installation is detected that location.
Otherwise, it will fall to `/opt/kohya_ss`. If /opt is not writeable, the fallback is `$HOME/kohya_ss`. Lastly, if all else fails it will simply install to the current folder you are in (PWD).

On macOS and other non-Linux machines, it will first try to detect an install where the script is run from and then run setup there if that's detected. 
If a previous install isn't found at that location, then it will default install to `$HOME/kohya_ss` followed by where you're currently at if there's no access to $HOME.
You can override this behavior by specifying an install directory with the -d option.

If you are using the interactive mode, our default values for the accelerate config screen after running the script answer "This machine", "None", "No" for the remaining questions.
These are the same answers as the Windows install.

### Windows
In the terminal, run:

```
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
setup.bat
```

Then configure accelerate with the same answers as in the MacOS instructions when prompted.

### Optional: CUDNN 8.6

This step is optional but can improve the learning speed for NVIDIA 30X0/40X0 owners. It allows for larger training batch size and faster training speed.

Due to the file size, I can't host the DLLs needed for CUDNN 8.6 on Github. I strongly advise you download them for a speed boost in sample generation (almost 50% on 4090 GPU) you can download them [here](https://b1.thefileditch.ch/mwxKTEtelILoIbMbruuM.zip).

To install, simply unzip the directory and place the `cudnn_windows` folder in the root of the this repo.

Run the following commands to install:

```
.\venv\Scripts\activate

python .\tools\cudann_1.8_install.py
```

Once the commands have completed successfully you should be ready to use the new version. MacOS support is not tested and has been mostly taken from https://gist.github.com/jstayco/9f5733f05b9dc29de95c4056a023d645

## Upgrading

The following commands will work from the root directory of the project if you'd prefer to not run scripts.
These commands will work on any OS.
```bash
git pull

.\venv\Scripts\activate

pip install --use-pep517 --upgrade -r requirements.txt
```

### Windows Upgrade
When a new release comes out, you can upgrade your repo with the following commands in the root directory:

```powershell
upgrade.bat
```

### Linux and macOS Upgrade
You can cd into the root directory and simply run

```bash
# Refresh and update everything
./setup.sh

# This will refresh everything, but NOT clone or pull the git repo.
./setup.sh --no-git-update
```

Once the commands have completed successfully you should be ready to use the new version.

# Starting GUI Service

The following command line arguments can be passed to the scripts on any OS to configure the underlying service.
```
--listen: the IP address to listen on for connections to Gradio.
--username: a username for authentication. 
--password: a password for authentication. 
--server_port: the port to run the server listener on. 
--inbrowser: opens the Gradio UI in a web browser. 
--share: shares the Gradio UI.
```

### Launching the GUI on Windows

The two scripts to launch the GUI on Windows are gui.ps1 and gui.bat in the root directory.
You can use whichever script you prefer.

To launch the Gradio UI, run the script in a terminal with the desired command line arguments, for example:

`gui.ps1 --listen 127.0.0.1 --server_port 7860 --inbrowser --share`

or

`gui.bat --listen 127.0.0.1 --server_port 7860 --inbrowser --share`

## Launching the GUI on Linux and macOS

Run the launcher script with the desired command line arguments similar to Windows.
`gui.sh --listen 127.0.0.1 --server_port 7860 --inbrowser --share`

## Launching the GUI directly using kohya_gui.py

To run the GUI directly bypassing the wrapper scripts, simply use this command from the root project directory:

```
.\venv\Scripts\activate

python .\kohya_gui.py
```

## Dreambooth

You can find the dreambooth solution specific here: [Dreambooth README](train_db_README.md)

## Finetune

You can find the finetune solution specific here: [Finetune README](fine_tune_README.md)

## Train Network

You can find the train network solution specific here: [Train network README](train_network_README.md)

## LoRA

Training a LoRA currently uses the `train_network.py` code. You can create a LoRA network by using the all-in-one `gui.cmd` or by running the dedicated LoRA training GUI with:

```
.\venv\Scripts\activate

python lora_gui.py
```

Once you have created the LoRA network, you can generate images via auto1111 by installing [this extension](https://github.com/kohya-ss/sd-webui-additional-networks).

## Troubleshooting

### Page File Limit

- X error relating to `page file`: Increase the page file size limit in Windows.

### No module called tkinter

- Re-install [Python 3.10](https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe) on your system.

### FileNotFoundError

This is usually related to an installation issue. Make sure you do not have any python modules installed locally that could conflict with the ones installed in the venv:

1. Open a new powershell terminal and make sure no venv is active.
2.  Run the following commands:

```
pip freeze > uninstall.txt
pip uninstall -r uninstall.txt
```

This will store a backup file with your current locally installed pip packages and then uninstall them. Then, redo the installation instructions within the kohya_ss venv.

## Change History

* 2023/04/14 (v21.5.3)
    - Fix issue when Adafactor is used as optimizer and LR Warmup is not 0: https://github.com/bmaltais/kohya_ss/issues/617

* 2023/04/09 (v21.5.2)

    - Added support for training with weighted captions. Thanks to AI-Casanova for the great contribution! 
    - Please refer to the PR for details: [PR #336](https://github.com/kohya-ss/sd-scripts/pull/336)
    - Specify the `--weighted_captions` option. It is available for all training scripts except Textual Inversion and XTI.
    - This option is also applicable to token strings of the DreamBooth method.
    - The syntax for weighted captions is almost the same as the Web UI, and you can use things like `(abc)`, `[abc]`, and `(abc:1.23)`. Nesting is also possible.
    - If you include a comma in the parentheses, the parentheses will not be properly matched in the prompt shuffle/dropout, so do not include a comma in the parentheses.
    - Run gui.sh from any place
