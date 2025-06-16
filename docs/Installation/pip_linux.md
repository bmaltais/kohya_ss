# Linux – Installation (pip method)

Use this method if you prefer `pip` or are on macOS.

## Table of Contents

- [Linux – Installation (pip method)](#linux--installation-pip-method)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
    - [Using `conda`](#using-conda)
  - [Clone the Repository](#clone-the-repository)
  - [Run the Setup Script](#run-the-setup-script)
  - [Start the GUI](#start-the-gui)
    - [Available CLI Options](#available-cli-options)
  - [Upgrade Instructions](#upgrade-instructions)
  - [Optional: Install Location Details](#optional-install-location-details)

## Prerequisites

- **Python 3.10.9** (or higher, but below 3.13)
- **Git** – Required for cloning the repository
- **NVIDIA CUDA Toolkit 12.8**
- **NVIDIA GPU** – Required for training; VRAM needs vary
- **(Optional) NVIDIA cuDNN** – Improves training speed and batch size

## Installation Steps

1. Install Python and Git. On Ubuntu 22.04 or later:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv git
```

2. Install [CUDA 12.8 Toolkit](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64)  
   Follow the instructions for your distribution.
3.

> [!NOTE]
> CUDA is usually not required and may not be compatible with Apple Silicon GPUs.

### Using `conda`

If you prefer Conda over `venv`, you can create an environment like this:

```shell
# Create Conda Environment
conda create -n kohyass python=3.11
conda activate kohyass

# Run the Scripts
chmod +x setup.sh
./setup.sh

chmod +x gui.sh
./gui.sh
```

## Clone the Repository

Clone with submodules:

```bash
git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
```

## Run the Setup Script

Make the setup script executable:

```bash
chmod +x setup.sh
```

Run:

```bash
./setup.sh
```

> [!NOTE]
> If you need additional options or information about the runpod environment, you can use `setup.sh -h` or `setup.sh --help` to display the help message.

## Start the GUI

Start with:

```bash
./gui.sh --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

You can also run `kohya_gui.py` directly with the same flags.

For help:

```bash
./gui.sh --help
```

This method uses a standard Python virtual environment.

### Available CLI Options

You can pass the following arguments to `gui.sh` or `kohya_gui.py`:

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

To upgrade, pull the latest changes and rerun setup:

```bash
git pull
./setup.sh
```

## Optional: Install Location Details

On Linux, the setup script will install in the current directory if possible.

If that fails:

- Fallback: `/opt/kohya_ss`
- If not writable: `$HOME/kohya_ss`
- If all fail: stays in the current directory

To override the location, use:

```bash
./setup.sh -d /your/custom/path
```

On macOS, the behavior is similar but defaults to `$HOME/kohya_ss`.

If you use interactive mode, the default Accelerate values are:

- Machine: `This machine`
- Compute: `None`
- Others: `No`
