# Linux – Installation (uv method)

Recommended setup for most Linux users.  
If you have macOS please use **pip method**.

## Table of Contents

- [Linux – Installation (uv method)](#linux--installation-uv-method)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [Clone the Repository](#clone-the-repository)
  - [Start the GUI](#start-the-gui)
    - [Available CLI Options](#available-cli-options)
  - [Upgrade Instructions](#upgrade-instructions)
  - [Optional: Install Location Details](#optional-install-location-details)

## Prerequisites

- **Python 3.10.9** (or higher, but below 3.13)

> [!NOTE]
> The `uv` environment will use the Python version specified in the `.python-version` file at the root of the repository. You can edit this file to change the Python version used by `uv`.

- **Git** – Required for cloning the repository
- **NVIDIA CUDA Toolkit 12.8**
- **NVIDIA GPU** – Required for training; VRAM needs vary
- **(Optional) NVIDIA cuDNN** – Improves training speed and batch size

## Installation Steps

1. Install Python (Make sure you have Python version 3.10.9 or higher (but lower than 3.11.0) installed on your system.)  
   On Ubuntu 22.04 or later:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv git
```

2. Install [CUDA 12.8 Toolkit](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64)  
   Follow the instructions for your distribution.

> [!NOTE]  
> macOS is only supported via the **pip method**.  
> CUDA is usually not required and may not be compatible with Apple Silicon GPUs.

## Clone the Repository

To install the project, you must first clone the repository **with submodules**:

```bash
git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
```

> The `--recursive` flag ensures that all required Git submodules are also cloned.

Run:

```bash
./gui-uv.sh
```

## Start the GUI

To launch the GUI service, run `./gui-uv.sh` or run the `kohya_gui.py` script directly. Use the command line arguments listed below to configure the underlying service.

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

When you run `gui-uv.sh`, it will first check if `uv` is installed on your system. If `uv` is not found, the script will prompt you, asking if you'd like to attempt an automatic installation. You can choose 'Y' (or 'y') to let the script try to install `uv` for you, or 'N' (or 'n') to cancel. If you cancel, you'll need to install `uv` manually from [https://astral.sh/uv](https://astral.sh/uv) before running `gui-uv.sh` again.

```shell
./gui-uv.sh --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

If you are running on a headless server, use:

```shell
./gui-uv.sh --headless --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

This script utilizes the `uv` managed environment.

## Upgrade Instructions

To upgrade your installation to a new version, follow the instructions below.

1. Open a terminal and navigate to the root directory of the project.
2. Pull the latest changes from the repository:

     ```bash
     git pull
     ```

3. Updates to the Python environment are handled automatically when you next run the `gui-uv.sh` script. No separate setup script execution is needed.

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
