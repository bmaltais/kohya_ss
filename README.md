# Kohya's GUI

This repository primarily provides a Gradio GUI for [Kohya's Stable Diffusion trainers](https://github.com/kohya-ss/sd-scripts). However, support for Linux OS is also offered through community contributions. macOS support is not optimal at the moment but might work if the conditions are favorable.

The GUI allows you to set the training parameters and generate and run the required CLI commands to train the model.

## Table of Contents

- [Kohya's GUI](#kohyas-gui)
  - [Table of Contents](#table-of-contents)
  - [🦒 Colab](#-colab)
  - [Installation](#installation)
    - [Windows](#windows)
      - [Windows Pre-requirements](#windows-pre-requirements)
      - [Setup Windows](#setup-windows)
      - [Optional: CUDNN 8.9.6.50](#optional-cudnn-89650)
    - [Linux and macOS](#linux-and-macos)
      - [Linux Pre-requirements](#linux-pre-requirements)
      - [Setup Linux](#setup-linux)
      - [Install Location](#install-location)
    - [Runpod](#runpod)
      - [Manual installation](#manual-installation)
      - [Pre-built Runpod template](#pre-built-runpod-template)
    - [Docker](#docker)
      - [Local docker build](#local-docker-build)
      - [ashleykleynhans runpod docker builds](#ashleykleynhans-runpod-docker-builds)
  - [Upgrading](#upgrading)
    - [Windows Upgrade](#windows-upgrade)
    - [Linux and macOS Upgrade](#linux-and-macos-upgrade)
  - [Starting GUI Service](#starting-gui-service)
    - [Launching the GUI on Windows](#launching-the-gui-on-windows)
    - [Launching the GUI on Linux and macOS](#launching-the-gui-on-linux-and-macos)
  - [Custom Path Defaults](#custom-path-defaults)
  - [LoRA](#lora)
  - [Sample image generation during training](#sample-image-generation-during-training)
  - [Troubleshooting](#troubleshooting)
    - [Page File Limit](#page-file-limit)
    - [No module called tkinter](#no-module-called-tkinter)
  - [SDXL training](#sdxl-training)
  - [Change History](#change-history)
    - [2024/03/20 (v23.0.15)](#20240320-v23015)
    - [2024/03/19 (v23.0.14)](#20240319-v23014)
    - [2024/03/19 (v23.0.13)](#20240319-v23013)
    - [2024/03/16 (v23.0.12)](#20240316-v23012)
      - [New Features \& Improvements](#new-features--improvements)
      - [Software Updates](#software-updates)
      - [Recommendations for Users](#recommendations-for-users)
    - [2024/03/13 (v23.0.11)](#20240313-v23011)
    - [2024/03/13 (v23.0.9)](#20240313-v2309)
    - [2024/03/12 (v23.0.8)](#20240312-v2308)
    - [2024/03/12 (v23.0.7)](#20240312-v2307)
    - [2024/03/11 (v23.0.6)](#20240311-v2306)
    - [2024/03/11 (v23.0.5)](#20240311-v2305)
    - [2024/03/10 (v23.0.4)](#20240310-v2304)
    - [2024/03/10 (v23.0.3)](#20240310-v2303)
    - [2024/03/10 (v23.0.2)](#20240310-v2302)
    - [2024/03/09 (v23.0.1)](#20240309-v2301)
    - [2024/03/02 (v23.0.0)](#20240302-v2300)

## 🦒 Colab

This Colab notebook was not created or maintained by me; however, it appears to function effectively. The source can be found at: <https://github.com/camenduru/kohya_ss-colab>.

I would like to express my gratitude to camendutu for their valuable contribution. If you encounter any issues with the Colab notebook, please report them on their repository.

| Colab                                                                                                                                                                          | Info               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/kohya_ss-colab/blob/main/kohya_ss_colab.ipynb) | kohya_ss_gui_colab |

## Installation

### Windows

#### Windows Pre-requirements

To install the necessary dependencies on a Windows system, follow these steps:

1. Install [Python 3.10.11](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe).
   - During the installation process, ensure that you select the option to add Python to the 'PATH' environment variable.

2. Install [CUDA 11.8 toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64).

3. Install [Git](https://git-scm.com/download/win).

4. Install the [Visual Studio 2015, 2017, 2019, and 2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

#### Setup Windows

To set up the project, follow these steps:

1. Open a terminal and navigate to the desired installation directory.

2. Clone the repository by running the following command:

   ```shell
   git clone https://github.com/bmaltais/kohya_ss.git
   ```

3. Change into the `kohya_ss` directory:

   ```shell
   cd kohya_ss
   ```

4. Run the setup script by executing the following command:

   ```shell
   .\setup.bat
   ```

   During the accelerate config step, use the default values as proposed during the configuration unless you know your hardware demands otherwise. The amount of VRAM on your GPU does not impact the values used.

#### Optional: CUDNN 8.9.6.50

The following steps are optional but will improve the learning speed for owners of NVIDIA 30X0/40X0 GPUs. These steps enable larger training batch sizes and faster training speeds.

1. Run `.\setup.bat` and select `2. (Optional) Install cudnn files (if you want to use the latest supported cudnn version)`.

### Linux and macOS

#### Linux Pre-requirements

To install the necessary dependencies on a Linux system, ensure that you fulfill the following requirements:

- Ensure that `venv` support is pre-installed. You can install it on Ubuntu 22.04 using the command:

  ```shell
  apt install python3.10-venv
  ```

- Install the CUDA 11.8 Toolkit by following the instructions provided in [this link](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64).

- Make sure you have Python version 3.10.9 or higher (but lower than 3.11.0) installed on your system.

#### Setup Linux

To set up the project on Linux or macOS, perform the following steps:

1. Open a terminal and navigate to the desired installation directory.

2. Clone the repository by running the following command:

   ```shell
   git clone https://github.com/bmaltais/kohya_ss.git
   ```

3. Change into the `kohya_ss` directory:

   ```shell
   cd kohya_ss
   ```

4. If you encounter permission issues, make the `setup.sh` script executable by running the following command:

   ```shell
   chmod +x ./setup.sh
   ```

5. Run the setup script by executing the following command:

   ```shell
   ./setup.sh
   ```

   Note: If you need additional options or information about the runpod environment, you can use `setup.sh -h` or `setup.sh --help` to display the help message.

#### Install Location

The default installation location on Linux is the directory where the script is located. If a previous installation is detected in that location, the setup will proceed there. Otherwise, the installation will fall back to `/opt/kohya_ss`. If `/opt` is not writable, the fallback location will be `$HOME/kohya_ss`. Finally, if none of the previous options are viable, the installation will be performed in the current directory.

For macOS and other non-Linux systems, the installation process will attempt to detect the previous installation directory based on where the script is run. If a previous installation is not found, the default location will be `$HOME/kohya_ss`. You can override this behavior by specifying a custom installation directory using the `-d` or `--dir` option when running the setup script.

If you choose to use the interactive mode, the default values for the accelerate configuration screen will be "This machine," "None," and "No" for the remaining questions. These default answers are the same as the Windows installation.

### Runpod

#### Manual installation

To install the necessary components for Runpod and run kohya_ss, follow these steps:

1. Select the Runpod pytorch 2.0.1 template. This is important. Other templates may not work.

2. SSH into the Runpod.

3. Clone the repository by running the following command:

   ```shell
   cd /workspace
   git clone https://github.com/bmaltais/kohya_ss.git
   ```

4. Run the setup script:

   ```shell
   cd kohya_ss
   ./setup-runpod.sh
   ```

5. Run the GUI with:

   ```shell
   ./gui.sh --share --headless
   ```

   or with this if you expose 7860 directly via the runpod configuration:

   ```shell
   ./gui.sh --listen=0.0.0.0 --headless
   ```

6. Connect to the public URL displayed after the installation process is completed.

#### Pre-built Runpod template

To run from a pre-built Runpod template, you can:

1. Open the Runpod template by clicking on <https://runpod.io/gsc?template=ya6013lj5a&ref=w18gds2n>.

2. Deploy the template on the desired host.

3. Once deployed, connect to the Runpod on HTTP 3010 to access the kohya_ss GUI. You can also connect to auto1111 on HTTP 3000.

### Docker

#### Local docker build

If you prefer to use Docker, follow the instructions below:

1. Ensure that you have Git and Docker installed on your Windows or Linux system.

2. Open your OS shell (Command Prompt or Terminal) and run the following commands:

   ```bash
   git clone --recursive https://github.com/bmaltais/kohya_ss.git
   cd kohya_ss
   docker compose up -d --build
   ```

   Note: The initial run may take up to 20 minutes to complete.

   Please be aware of the following limitations when using Docker:

   - All training data must be placed in the `dataset` subdirectory, as the Docker container cannot access files from other directories.
   - The file picker feature is not functional. You need to manually set the folder path and config file path.
   - Dialogs may not work as expected, and it is recommended to use unique file names to avoid conflicts.
   - This Dockerfile has been designed to be easily disposable. You can discard the container at any time and docker build it with a new version of the code. To update the system, run update scripts outside of Docker and rebuild using `docker compose down && docker compose up -d --build`.

   If you are running Linux, an alternative Docker container port with fewer limitations is available [here](https://github.com/P2Enjoy/kohya_ss-docker).

#### ashleykleynhans runpod docker builds

You may want to use the following Dockerfile repositories to build the images:

- Standalone Kohya_ss template: <https://github.com/ashleykleynhans/kohya-docker>
- Auto1111 + Kohya_ss GUI template: <https://github.com/ashleykleynhans/stable-diffusion-docker>

## Upgrading

To upgrade your installation to a new version, follow the instructions below.

### Windows Upgrade

If a new release becomes available, you can upgrade your repository by running the following commands from the root directory of the project:

1. Pull the latest changes from the repository:

   ```powershell
   git pull
   ```

2. Run the setup script:

   ```powershell
   .\setup.bat
   ```

### Linux and macOS Upgrade

To upgrade your installation on Linux or macOS, follow these steps:

1. Open a terminal and navigate to the root directory of the project.

2. Pull the latest changes from the repository:

   ```bash
   git pull
   ```

3. Refresh and update everything:

   ```bash
   ./setup.sh
   ```

## Starting GUI Service

To launch the GUI service, you can use the provided scripts or run the `kohya_gui.py` script directly. Use the command line arguments listed below to configure the underlying service.

```text
--listen: Specify the IP address to listen on for connections to Gradio.
--username: Set a username for authentication.
--password: Set a password for authentication.
--server_port: Define the port to run the server listener on.
--inbrowser: Open the Gradio UI in a web browser.
--share: Share the Gradio UI.
--language: Set custom language
```

### Launching the GUI on Windows

On Windows, you can use either the `gui.ps1` or `gui.bat` script located in the root directory. Choose the script that suits your preference and run it in a terminal, providing the desired command line arguments. Here's an example:

```powershell
gui.ps1 --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

or

```powershell
gui.bat --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

### Launching the GUI on Linux and macOS

To launch the GUI on Linux or macOS, run the `gui.sh` script located in the root directory. Provide the desired command line arguments as follows:

```bash
gui.sh --listen 127.0.0.1 --server_port 7860 --inbrowser --share
```

## Custom Path Defaults

You can now specify custom paths more easily:

- Simply copy the `config example.toml` file located in the root directory of the repository to `config.toml`.
- Edit the `config.toml` file to adjust paths and settings according to your preferences.

## LoRA

To train a LoRA, you can currently use the `train_network.py` code. You can create a LoRA network by using the all-in-one GUI.

Once you have created the LoRA network, you can generate images using auto1111 by installing [this extension](https://github.com/kohya-ss/sd-webui-additional-networks).

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

## SDXL training

The documentation in this section will be moved to a separate document later.

## Change History

### 2024/03/21 (v23.0.15)

- Add support for toml dataset configuration fole to all trainers
- Add new setup menu option to install Triton 2.1.0 for Windows
- Add support for LyCORIS BOFT and DoRA and QLyCORIS options for LoHA, LoKr and LoCon
- Fix issue with vae path validation
- Other fixes

### 2024/03/19 (v23.0.14)

- Fix blip caption issue

### 2024/03/19 (v23.0.13)

- Fix issue with image samples.

### 2024/03/16 (v23.0.12)

#### New Features & Improvements

- **Enhanced Logging and Tracking Capabilities**
  - Added support for configuring advanced logging and tracking:
    - `wandb_run_name`: Set a custom name for your Weights & Biases runs to easily identify and organize your experiments.
    - `log_tracker_name` and `log_tracker_config`: Integrate custom logging trackers with your projects. Specify the tracker name and provide its configuration to enable detailed monitoring and logging of your runs.

- **Custom Path Defaults**
  - You can now specify custom paths more easily:
    - Simply copy the `config example.toml` file located in the root directory of the repository to `config.toml`.
    - Edit the `config.toml` file to adjust paths and settings according to your preferences.

#### Software Updates

- **sd-scripts updated to v0.8.5**
  - **Bug Fixes:**
    - Corrected an issue where the value of timestep embedding was incorrect during SDXL training. This fix ensures accurate training progress and results.
    - Addressed a related inference issue with the generation script, improving the reliability of SDXL model outputs.
  - **Note:** The exact impact of this bug is currently unknown, but it's recommended to update to v0.8.5 for anyone engaged in SDXL training to ensure optimal performance and results.

- **Upgrade of `lycoris_lora` Python Module**
  - Updated the `lycoris_lora` module to version 2.2.0.post3. This update may include bug fixes, performance improvements, and new features.

#### Recommendations for Users

- To benefit from the latest features and improvements, users are encouraged to update their installations and configurations accordingly.

### 2024/03/13 (v23.0.11)

- Increase icon size.
- More setup fixes.

### 2024/03/13 (v23.0.9)

- Reworked how setup can be run to improve Stability Matrix support.
- Added support for huggingface-based vea path.

### 2024/03/12 (v23.0.8)

- Add the ability to create output and logs folder if it does not exist

### 2024/03/12 (v23.0.7)

- Fixed minor issues related to functions and file paths.

### 2024/03/11 (v23.0.6)

- Fixed an issue with PYTHON paths that have "spaces" in them.

### 2024/03/11 (v23.0.5)

- Updated python module verification.
- Removed cudnn module installation in Windows.

### 2024/03/10 (v23.0.4)

- Updated bitsandbytes to 0.43.0.
- Added packaging to runpod setup.

### 2024/03/10 (v23.0.3)

- Fixed a bug with setup.
- Enforced proper python version before running the GUI to prevent issues with execution of the GUI.

### 2024/03/10 (v23.0.2)

- Improved validation of the path provided by users before running training.

### 2024/03/09 (v23.0.1)

- Updated bitsandbytes module to 0.43.0 as it provides native Windows support.
- Minor fixes to the code.

### 2024/03/02 (v23.0.0)

- Used sd-scripts release [0.8.4](https://github.com/kohya-ss/sd-scripts/releases/tag/v0.8.4) post commit [fccbee27277d65a8dcbdeeb81787ed4116b92e0b](https://github.com/kohya-ss/sd-scripts/commit/fccbee27277d65a8dcbdeeb81787ed4116b92e0b).
- Major code refactoring thanks to @wkpark. This will make updating sd-scripts cleaner by keeping sd-scripts files separate from the GUI files. This will also make configuration more streamlined with fewer tabs and more accordion elements. Hope you like the new style.
- This new release is implementing a significant structure change, moving all of the sd-scripts written by kohya under a folder called sd-scripts in the root of this project. This folder is a submodule that will be populated during setup or GUI execution.
