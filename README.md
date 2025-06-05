# Kohya's GUI

[![GitHub stars](https://img.shields.io/github/stars/bmaltais/kohya_ss?style=social)](https://github.com/bmaltais/kohya_ss/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/bmaltais/kohya_ss?style=social)](https://github.com/bmaltais/kohya_ss/network/members)
[![License](https://img.shields.io/github/license/bmaltais/kohya_ss)](LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/bmaltais/kohya_ss)](https://github.com/bmaltais/kohya_ss/issues)

This project provides a user-friendly Gradio-based Graphical User Interface (GUI) for [Kohya's Stable Diffusion training scripts](https://github.com/kohya-ss/sd-scripts). Stable Diffusion training empowers users to customize image generation models by fine-tuning existing models, creating unique artistic styles, and training specialized models like LoRA (Low-Rank Adaptation).

Key features of this GUI include:
*   Easy-to-use interface for setting a wide range of training parameters.
*   Automatic generation of the command-line interface (CLI) commands required to run the training scripts.
*   Support for various training methods, including LoRA, Dreambooth, fine-tuning, and SDXL training.

Support for Linux and macOS is also available. While Linux support is actively maintained through community contributions, macOS compatibility may vary.

## Table of Contents

- [Kohya's GUI](#kohyas-gui)
  - [Table of Contents](#table-of-contents)
  - [🦒 Colab](#-colab)
  - [Installation Methods](#installation-methods)
    - [Using `uv` (Recommended)](#using-uv-recommended)
    - [Using `pip` (Traditional Method)](#using-pip-traditional-method)
    - [Using `conda`](#using-conda)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [Windows](#windows)
      - [Windows Pre-requirements](#windows-pre-requirements)
      - [Setup Windows](#setup-windows)
    - [Linux and macOS](#linux-and-macos)
      - [Linux Pre-requirements](#linux-pre-requirements)
      - [Setup Linux](#setup-linux)
      - [Install Location](#install-location)
    - [Runpod](#runpod)
    - [Novita](#novita)
    - [Docker](#docker)
  - [Upgrading](#upgrading)
    - [Windows Upgrade](#windows-upgrade)
    - [Linux and macOS Upgrade](#linux-and-macos-upgrade)
  - [Starting GUI Service](#starting-gui-service)
    - [Launching the GUI on Windows (pip method)](#launching-the-gui-on-windows-pip-method)
    - [Launching the GUI on Windows (uv method)](#launching-the-gui-on-windows-uv-method)
    - [Launching the GUI on Linux and macOS](#launching-the-gui-on-linux-and-macos)
    - [Launching the GUI on Linux (uv method)](#launching-the-gui-on-linux-uv-method)
  - [Custom Path Defaults](#custom-path-defaults)
  - [LoRA](#lora)
  - [Sample image generation during training](#sample-image-generation-during-training)
  - [Troubleshooting](#troubleshooting)
    - [Page File Limit](#page-file-limit)
    - [No module called tkinter](#no-module-called-tkinter)
    - [LORA Training on TESLA V100 - GPU Utilization Issue](#lora-training-on-tesla-v100---gpu-utilization-issue)
  - [SDXL training](#sdxl-training)
  - [Masked loss](#masked-loss)
  - [Guides](#guides)
    - [Using Accelerate Lora Tab to Select GPU ID](#using-accelerate-lora-tab-to-select-gpu-id)
      - [Starting Accelerate in GUI](#starting-accelerate-in-gui)
      - [Running Multiple Instances (linux)](#running-multiple-instances-linux)
      - [Monitoring Processes](#monitoring-processes)
  - [Interesting Forks](#interesting-forks)
  - [Contributing](#contributing)
  - [License](#license)
  - [Change History](#change-history)
    - [v25.0.3](#v2503)
    - [v25.0.2](#v2502)
    - [v25.0.1](#v2501)
    - [v25.0.0](#v2500)
  
## 🦒 Colab

This Colab notebook was not created or maintained by me; however, it appears to function effectively. The source can be found at: <https://github.com/camenduru/kohya_ss-colab>.

I would like to express my gratitude to camenduru for their valuable contribution. If you encounter any issues with the Colab notebook, please report them on their repository.

| Colab                                                                                                                                                                          | Info               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/kohya_ss-colab/blob/main/kohya_ss_colab.ipynb) | kohya_ss_gui_colab |

## Installation Methods

This project offers two primary methods for installing and running the GUI: using the `uv` package manager (recommended for ease of use and automatic updates) or using the traditional `pip` package manager. Below, you'll find details on both approaches. Please read this section to decide which method best suits your needs before proceeding to the OS-specific installation prerequisites.

**Key Differences:**

*   **`uv` method:**
    *   Simplifies the setup process.
    *   Automatically handles updates when you run `gui-uv.bat` (Windows) or `gui-uv.sh` (Linux).
    *   No need to run `setup.bat` or `setup.sh` after the initial clone.
    *   This is the recommended method for most users on Windows and Linux.
    *   **Not recommended for Runpod or macOS installations.** For these, please use the `pip` method.
*   **`pip` method:**
    *   The traditional method, requiring manual execution of `setup.bat` (Windows) or `setup.sh` (Linux) after cloning and for updates.
    *   Necessary for environments like Runpod and macOS where the `uv` scripts are not intended to be used.

Subsequent sections will detail the specific commands for each method.

### Using `uv` (Recommended)
This method utilizes the `uv` package manager for a streamlined setup and automatic updates. It is the preferred approach for most users on Windows and Linux.

> [!NOTE]
> This method is not intended for runpod or MacOS installation. Use the "pip based package manager" setup instead.

To set up the project, follow these steps:

1. Open a terminal and navigate to the desired installation directory.

2. Clone the repository by running the following command:

   ```shell
   git clone --recursive https://github.com/bmaltais/kohya_ss.git
   ```

3. Change into the `kohya_ss` directory:

   ```shell
   cd kohya_ss
   ```

For Linux, the steps are similar (clone and change directory as above).

### Using `pip` (Traditional Method)
This method uses the traditional `pip` package manager and requires manual script execution for setup and updates. It is necessary for environments like Runpod or macOS, or if you prefer managing your environment with `pip`.

Regardless of your OS, start with these steps:

1. Open a terminal and navigate to the desired installation directory.

2. Clone the repository by running the following command:

   ```shell
   git clone --recursive https://github.com/bmaltais/kohya_ss.git
   ```

3. Change into the `kohya_ss` directory:

   ```shell
   cd kohya_ss
   ```

Then, proceed with OS-specific instructions:

### Using `conda`

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
> [!NOTE]
> For Windows users, the `chmod +x` commands are not necessary. You should run `setup.bat` and subsequently `gui.bat` (or `gui.ps1` if you prefer PowerShell) instead of the `.sh` scripts.

**For Windows:**

*   If you want to use the new uv based version of the script to run the GUI, you do not need to follow this step. On the other hand, if you want to use the legacy "pip" based method, please follow this next step.

    Run one of the following setup script by executing the following command:

    For systems with only python 3.10.11 installed:

   ```shell
   .\setup.bat
   ```

   For systems with only more than one python release installed:

   ```shell
   .\setup-3.10.bat
   ```

    During the accelerate config step, use the default values as proposed during the configuration unless you know your hardware demands otherwise. The amount of VRAM on your GPU does not impact the values used.

*   Optional: CUDNN 8.9.6.50

    The following steps are optional but will improve the learning speed for owners of NVIDIA 30X0/40X0 GPUs. These steps enable larger training batch sizes and faster training speeds.

    Run `.\setup.bat` and select `2. (Optional) Install cudnn files (if you want to use the latest supported cudnn version)`.

**For Linux and macOS:**

*   If you want to use the new uv based version of the script to run the GUI, you do not need to follow this step. On the other hand, if you want to use the legacy "pip" based method, please follow this next step.

    If you encounter permission issues, make the `setup.sh` script executable by running the following command:

   ```shell
   chmod +x ./setup.sh
   ```

   Run the setup script by executing the following command:

   ```shell
   ./setup.sh
   ```

   > [!NOTE]
   > If you need additional options or information about the runpod environment, you can use `setup.sh -h` or `setup.sh --help` to display the help message.

## Prerequisites

Before you begin, ensure you have the following software and hardware:

*   **Python:** Version 3.10.x or 3.11.x. (Python 3.11.9 is used in Windows pre-requirements, Python 3.10.9+ for Linux).
*   **Git:** For cloning the repository and managing updates.
*   **NVIDIA CUDA Toolkit:** Version 12.8 or compatible (as per installation steps).
*   **NVIDIA GPU:** A compatible NVIDIA graphics card is required. VRAM requirements vary depending on the model and training parameters.
*   **(Optional but Recommended) NVIDIA cuDNN:** For accelerated performance on compatible NVIDIA GPUs. (Often included with CUDA Toolkit or installed separately).
*   **For Windows Users:** Visual Studio 2015, 2017, 2019, and 2022 Redistributable.

## Installation

### Windows

#### Windows Pre-requirements

To install the necessary dependencies on a Windows system, follow these steps:

1. Install [Python 3.11.9](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe).
   - During the installation process, ensure that you select the option to add Python to the 'PATH' environment variable.

2. Install [CUDA 12.8 toolkit](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64).

3. Install [Git](https://git-scm.com/download/win).

4. Install the [Visual Studio 2015, 2017, 2019, and 2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

#### Setup Windows

For detailed setup instructions using either `uv` or `pip`, please refer to the 'Installation Methods' section above. Ensure you have met the Windows Pre-requirements before proceeding with either method.

### Linux and macOS

#### Linux Pre-requirements

To install the necessary dependencies on a Linux system, ensure that you fulfill the following requirements:

- Ensure that `venv` support is pre-installed. You can install it on Ubuntu 22.04 using the command:

  ```shell
  apt install python3.10-venv
  ```

- Install the CUDA 12.8 Toolkit by following the instructions provided in [this link](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64).

- Make sure you have Python version 3.10.9 or higher (but lower than 3.11.0) installed on your system.

#### Setup Linux

For detailed setup instructions using either `uv` or `pip`, please refer to the 'Installation Methods' section above. Ensure you have met the Linux Pre-requirements before proceeding with either method.

#### Install Location

Note: The information below regarding install location applies to both `uv` and `pip` installation methods described in the 'Installation Methods' section.

The default installation location on Linux is the directory where the script is located. If a previous installation is detected in that location, the setup will proceed there. Otherwise, the installation will fall back to `/opt/kohya_ss`. If `/opt` is not writable, the fallback location will be `$HOME/kohya_ss`. Finally, if none of the previous options are viable, the installation will be performed in the current directory.

For macOS and other non-Linux systems, the installation process will attempt to detect the previous installation directory based on where the script is run. If a previous installation is not found, the default location will be `$HOME/kohya_ss`. You can override this behavior by specifying a custom installation directory using the `-d` or `--dir` option when running the setup script.

If you choose to use the interactive mode, the default values for the accelerate configuration screen will be "This machine," "None," and "No" for the remaining questions. These default answers are the same as the Windows installation.

### Runpod

See [Runpod Installation Guide](docs/installation_runpod.md) for details.

### Novita

See [Novita Installation Guide](docs/installation_novita.md) for details.

### Docker

See [Docker Installation Guide](docs/installation_docker.md) for details.

## Upgrading

To upgrade your installation to a new version, follow the instructions below.

### Windows Upgrade

If a new release becomes available, you can upgrade your repository by following these steps:

*   **If you are using the `uv`-based installation (`gui-uv.bat`):**
    1.  Pull the latest changes from the repository:
        ```powershell
        git pull
        ```
    2.  Updates to the Python environment are handled automatically when you next run the `gui-uv.bat` script. No separate setup script execution is needed.

*   **If you are using the `pip`-based installation (`gui.bat` or `gui.ps1`):**
    1.  Pull the latest changes from the repository:
        ```powershell
        git pull
        ```
    2.  Run the setup script to update dependencies:
        ```powershell
        .\setup.bat
        ```

### Linux and macOS Upgrade

To upgrade your installation on Linux or macOS, follow these steps:

*   **If you are using the `uv`-based installation (`gui-uv.sh`):**
    1.  Open a terminal and navigate to the root directory of the project.
    2.  Pull the latest changes from the repository:
        ```bash
        git pull
        ```
    3.  Updates to the Python environment are handled automatically when you next run the `gui-uv.sh` script. No separate setup script execution is needed.

*   **If you are using the `pip`-based installation (`gui.sh`):**
    1.  Open a terminal and navigate to the root directory of the project.
    2.  Pull the latest changes from the repository:
        ```bash
        git pull
        ```
    3.  Refresh and update everything by running the setup script:
        ```bash
        ./setup.sh
        ```

## Starting GUI Service

To launch the GUI service, use the script corresponding to your chosen installation method (`uv` or `pip`), or run the `kohya_gui.py` script directly. Use the command line arguments listed below to configure the underlying service.

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

### Launching the GUI on Windows (pip method)

If you installed using the `pip` method, use either the `gui.ps1` or `gui.bat` script located in the root directory. Choose the script that suits your preference and run it in a terminal, providing the desired command line arguments. Here's an example:

```powershell
gui.ps1 --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

or

```powershell
gui.bat --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

### Launching the GUI on Windows (uv method)

If you installed using the `uv` method, use the `gui-uv.bat` script to start the GUI. Follow these steps:

When you run `gui-uv.bat`, it will first check if `uv` is installed on your system. If `uv` is not found, the script will prompt you, asking if you'd like to attempt an automatic installation. You can choose 'Y' to let the script try to install `uv` for you, or 'N' to cancel. If you cancel, you'll need to install `uv` manually from [https://astral.sh/uv](https://astral.sh/uv) before running `gui-uv.bat` again.

```cmd
.\gui-uv.bat
```

or

```powershell
.\gui-uv.bat --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

This script utilizes the `uv` managed environment.

### Launching the GUI on Linux and macOS

If you installed using the `pip` method on Linux or macOS, run the `gui.sh` script located in the root directory. Provide the desired command line arguments as follows:

```bash
./gui.sh --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

### Launching the GUI on Linux (uv method)

If you installed using the `uv` method on Linux, use the `gui-uv.sh` script to start the GUI. Follow these steps:

When you run `gui-uv.sh`, it will first check if `uv` is installed on your system. If `uv` is not found, the script will prompt you, asking if you'd like to attempt an automatic installation. You can choose 'Y' (or 'y') to let the script try to install `uv` for you, or 'N' (or 'n') to cancel. If you cancel, you'll need to install `uv` manually from [https://astral.sh/uv](https://astral.sh/uv) before running `gui-uv.sh` again.

```shell
./gui-uv.sh --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

If you are running on a headless server, use:

```shell
./gui-uv.sh --headless --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

This script utilizes the `uv` managed environment.

## Custom Path Defaults

The repository now provides a default configuration file named `config.toml`. This file is a template that you can customize to suit your needs.

To use the default configuration file, follow these steps:

1. Copy the `config example.toml` file from the root directory of the repository to `config.toml`.
2. Open the `config.toml` file in a text editor.
3. Modify the paths and settings as per your requirements.

This approach allows you to easily adjust the configuration to suit your specific needs to open the desired default folders for each type of folder/file input supported in the GUI.

You can specify the path to your config.toml (or any other name you like) when running the GUI. For instance: ./gui.bat --config c:\my_config.toml

## LoRA

To train a LoRA, you can currently use the `train_network.py` code. You can create a LoRA network by using the all-in-one GUI.

Once you have created the LoRA network, you can generate images using auto1111 by installing [this extension](https://github.com/kohya-ss/sd-webui-additional-networks).

For more detailed information on LoRA training options and advanced configurations, please refer to our LoRA documentation:
- [LoRA Training Guide](docs/LoRA/top_level.md)
- [LoRA Training Options](docs/LoRA/options.md)

## Sample image generation during training

A prompt file might look like this, for example:

```txt
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy, bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy, bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following options can be used:

- `--n`: Negative prompt up to the next option.
- `--w`: Specifies the width of the generated image.
- `--h`: Specifies the height of the generated image.
- `--d`: Specifies the seed of the generated image.
- `--l`: Specifies the CFG scale of the generated image.
- `--s`: Specifies the number of steps in the generation.

The prompt weighting such as `( )` and `[ ]` is working.

## Troubleshooting

If you encounter any issues, refer to the troubleshooting steps below.

### Page File Limit

If you encounter an X error related to the page file, you may need to increase the page file size limit in Windows.

### No module called tkinter

If you encounter an error indicating that the module `tkinter` is not found, try reinstalling Python 3.10 on your system.

### LORA Training on TESLA V100 - GPU Utilization Issue

See [Troubleshooting LORA Training on TESLA V100](docs/troubleshooting_tesla_v100.md) for details.

## SDXL training

For detailed guidance on SDXL training, please refer to the [official sd-scripts documentation](https://github.com/kohya-ss/sd-scripts/blob/main/README.md#sdxl-training) and relevant sections in our [LoRA Training Guide](docs/LoRA/top_level.md).

## Masked loss

The masked loss is supported in each training script. To enable the masked loss, specify the `--masked_loss` option.

> [!WARNING]
> The feature is not fully tested, so there may be bugs. If you find any issues, please open an Issue.

ControlNet dataset is used to specify the mask. The mask images should be the RGB images. The pixel value 255 in R channel is treated as the mask (the loss is calculated only for the pixels with the mask), and 0 is treated as the non-mask. The pixel values 0-255 are converted to 0-1 (i.e., the pixel value 128 is treated as the half weight of the loss). See details for the dataset specification in the [LLLite documentation](./docs/train_lllite_README.md#preparing-the-dataset).

## Guides

The following are guides extracted from issues discussions

### Using Accelerate Lora Tab to Select GPU ID

#### Starting Accelerate in GUI

- Open the kohya GUI on your desired port.
- Open the `Accelerate launch` tab
- Ensure the Multi-GPU checkbox is unchecked.
- Set GPU IDs to the desired GPU (like 1).

#### Running Multiple Instances (linux)

- For tracking multiple processes, use separate kohya GUI instances on different ports (e.g., 7860, 7861).
- Start instances using `nohup ./gui.sh --listen 0.0.0.0 --server_port <port> --headless > log.log 2>&1 &`.

#### Monitoring Processes

- Open each GUI in a separate browser tab.
- For terminal access, use SSH and tools like `tmux` or `screen`.

For more details, visit the [GitHub issue](https://github.com/bmaltais/kohya_ss/issues/2577).

## Interesting Forks

To finetune HunyuanDiT models or create LoRAs, visit this [fork](https://github.com/Tencent/HunyuanDiT/tree/main/kohya_ss-hydit)

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please consider the following:
- For bug reports or feature requests, please open an issue on the [GitHub Issues page](https://github.com/bmaltais/kohya_ss/issues).
- If you'd like to submit code changes, please open a pull request. Ensure your changes are well-tested and follow the existing code style.
- For security-related concerns, please refer to our `SECURITY.md` file.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE.md](LICENSE.md) file for details.

## Change History

### v25.0.3

- Upgrade Gradio, diffusers and huggingface-hub to latest release to fix issue with ASGI.
- Add a new method to setup and run the GUI. You will find two new script for both Windows (gui-uv.bat) and Linux (gui-uv.sh). With those scripts there is no need to run setup.bat or setup.sh anymore.

### v25.0.2

- Force gradio to 5.14.0 or greater so it is updated.

### v25.0.1

- Fix issue with requirements version causing huggingface download issues

### v25.0.0

- Major update: Introduced support for flux.1 and sd3, moving the GUI to align with more recent script functionalities.
- Users preferring the pre-flux.1/sd3 version can check out tag `v24.1.7`.
  ```shell
  git checkout v24.1.7
  ```
- For details on new flux.1 and sd3 parameters, refer to the [sd-scripts README](https://github.com/kohya-ss/sd-scripts/blob/sd3/README.md).
