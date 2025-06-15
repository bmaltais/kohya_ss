# Windows – Installation (pip method)

Use this method if `uv` is not available or you prefer the traditional approach.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Using Conda](#using-conda-optional)
- [Clone the Repository](#clone-the-repository)
- [Run the Setup Script](#run-the-setup-script)
- [Start the GUI](#start-the-gui)
    - [Available CLI Options](#available-cli-options)
- [Upgrade Instructions](#upgrade-instructions)
- [Optional: Install Location Details](#optional-install-location-details)

## Prerequisites

- **Python 3.10.11**
- **Git** – Required for cloning the repository
- **NVIDIA CUDA Toolkit 12.8**
- **NVIDIA GPU** – Required for training; VRAM needs vary
- **(Optional) NVIDIA cuDNN** – Improves training speed and batch size
- (Optional) Visual Studio Redistributables: [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)

## Installation Steps

1. Install [Python 3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)  
   ✅ Enable the "Add to PATH" option during setup

2. Install [CUDA 12.8 Toolkit](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64)

3. Install [Git](https://git-scm.com/download/win)

4. Install [Visual Studio Redistributables](https://aka.ms/vs/17/release/vc_redist.x64.exe)


## Using Conda (Optional)

If you prefer Conda over `venv`, you can create an environment like this:

```powershell
conda create -n kohyass python=3.10
conda activate kohyass

setup.bat
```

You can also use:

```powershell
setup-3.10.bat
```

Then run:

```powershell
gui.ps1
```

or:

```cmd
gui.bat
```

## Clone the Repository

Clone with submodules:

```cmd
git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
```

> The `--recursive` flag ensures all submodules are fetched.

## Run the Setup Script

Run:

```cmd
setup.bat
```

If you have multiple Python versions installed:

```cmd
setup-3.10.bat
```

During the Accelerate configuration step, use the default values as proposed unless you know your hardware demands otherwise.  
The amount of VRAM on your GPU does **not** impact the values used.

*Optional: cuDNN 8.9.6.50*

These optional steps improve training speed for NVIDIA 30X0/40X0 GPUs. They allow for larger batch sizes and faster training.

Run:

```cmd
setup.bat
```

Then select:

```
2. (Optional) Install cudnn files (if you want to use the latest supported cudnn version)
```
## Start the GUI

If you installed using the `pip` method, use either the `gui.ps1` or `gui.bat` script located in the root directory. Choose the script that suits your preference and run it in a terminal, providing the desired command line arguments. Here's an example:

```powershell
gui.ps1 --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

or

```cmd
gui.bat --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

You can also run `kohya_gui.py` directly with the same flags.

For help:

```cmd
gui.bat --help
```

This method uses a Python virtual environment managed via pip.

### Available CLI Options

```text
  --help                show this help message and exit
  --config CONFIG       Path to the toml config file for interface defaults
  --debug               Debug on
  --listen LISTEN       IP to listen on for connections to Gradio
  --username USERNAME   Username for authentication
  --password PASSWORD   Password for authentication
  --server_port SERVER_PORT
                        Port to run the server listener on
  --inbrowser           Open in browser
  --share               Share the gradio UI
  --headless            Is the server headless
  --language LANGUAGE   Set custom language
  --use-ipex            Use IPEX environment
  --use-rocm            Use ROCm environment
  --do_not_use_shell    Enforce not to use shell=True when running external commands
  --do_not_share        Do not share the gradio UI
  --requirements REQUIREMENTS
                        requirements file to use for validation
  --root_path ROOT_PATH
                        `root_path` for Gradio to enable reverse proxy support. e.g. /kohya_ss
  --noverify            Disable requirements verification
```

## Upgrade Instructions

To upgrade your environment:

```cmd
git pull
setup.bat
```

