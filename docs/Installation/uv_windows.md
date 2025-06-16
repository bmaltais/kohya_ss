# Windows – Installation (uv method)

Recommended for most Windows users.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Clone the Repository](#clone-the-repository)
- [Start the GUI](#start-the-gui)
    - [Available CLI Options](#available-cli-options)
- [Upgrade Instructions](#upgrade-instructions)
- 
## Prerequisites

- [Python 3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) – enable "Add to PATH"
> [!NOTE]
> The `uv` environment will use the Python version specified in the `.python-version` file at the root of the repository. You can edit this file to change the Python version used by `uv`.
- [Git for Windows](https://git-scm.com/download/win)
- [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64)
- **NVIDIA GPU** – Required for training; VRAM needs vary
- **(Optional) NVIDIA cuDNN** – Improves training speed and batch size
- (Optional) Visual Studio Redistributables: [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)

## Installation Steps

1. Install [Python 3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)  
   ✅ Enable the "Add to PATH" option during setup

2. Install [CUDA 12.8 Toolkit](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64)

3. Install [Git](https://git-scm.com/download/win)

4. Install [Visual Studio Redistributables](https://aka.ms/vs/17/release/vc_redist.x64.exe)


## Clone the Repository

Clone with submodules:

```powershell
git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
```
## Start the GUI

To launch the GUI, run:

```cmd
.\gui-uv.bat
```

If `uv` is not installed, the script will prompt you:
- Press `Y` to install `uv` automatically
- Or press `N` to cancel and install `uv` manually from [https://astral.sh/uv](https://astral.sh/uv)

Once installed, you can also start the GUI with additional flags:

```cmd
.\gui-uv.bat --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

This script utilizes the `uv` managed environment and handles dependencies and updates automatically.

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

This script utilizes the `uv` managed environment and automatically handles dependencies and updates.

## Upgrade Instructions

1. Pull the latest changes:

```powershell
git pull
```

2. Run `gui-uv.bat` again. It will update the environment automatically.
