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
      - [Get your Docker ready for GPU support](#get-your-docker-ready-for-gpu-support)
        - [Windows](#windows-1)
        - [Linux, OSX](#linux-osx)
      - [Design of our Dockerfile](#design-of-our-dockerfile)
      - [Use the pre-built Docker image](#use-the-pre-built-docker-image)
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
    - [LORA Training on TESLA V100 - GPU Utilization Issue](#lora-training-on-tesla-v100---gpu-utilization-issue)
      - [Issue Summary](#issue-summary)
      - [Potential Solutions](#potential-solutions)
  - [SDXL training](#sdxl-training)
  - [Masked loss](#masked-loss)
  - [Guides](#guides)
    - [Using Accelerate Lora Tab to Select GPU ID](#using-accelerate-lora-tab-to-select-gpu-id)
      - [Starting Accelerate in GUI](#starting-accelerate-in-gui)
      - [Running Multiple Instances (linux)](#running-multiple-instances-linux)
      - [Monitoring Processes](#monitoring-processes)
  - [Interesting Forks](#interesting-forks)
  - [Change History](#change-history)

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

2. Install [CUDA 12.4 toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64).

3. Install [Git](https://git-scm.com/download/win).

4. Install the [Visual Studio 2015, 2017, 2019, and 2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

#### Setup Windows

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

4. Run one of the following setup script by executing the following command:

   For systems with only python 3.10.11 installed:

   ```shell
   .\setup.bat
   ```

   For systems with only more than one python release installed:

   ```shell
   .\setup-3.10.bat
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

- Install the CUDA 12.4 Toolkit by following the instructions provided in [this link](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64).

- Make sure you have Python version 3.10.9 or higher (but lower than 3.11.0) installed on your system.

#### Setup Linux

To set up the project on Linux or macOS, perform the following steps:

1. Open a terminal and navigate to the desired installation directory.

2. Clone the repository by running the following command:

   ```shell
   git clone --recursive https://github.com/bmaltais/kohya_ss.git
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
   git clone --recursive https://github.com/bmaltais/kohya_ss.git
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

#### Get your Docker ready for GPU support

##### Windows

Once you have installed [**Docker Desktop**](https://www.docker.com/products/docker-desktop/), [**CUDA Toolkit**](https://developer.nvidia.com/cuda-downloads), [**NVIDIA Windows Driver**](https://www.nvidia.com.tw/Download/index.aspx), and ensured that your Docker is running with [**WSL2**](https://docs.docker.com/desktop/wsl/#turn-on-docker-desktop-wsl-2), you are ready to go.

Here is the official documentation for further reference.  
<https://docs.nvidia.com/cuda/wsl-user-guide/index.html#nvidia-compute-software-support-on-wsl-2>
<https://docs.docker.com/desktop/wsl/use-wsl/#gpu-support>

##### Linux, OSX

Install an NVIDIA GPU Driver if you do not already have one installed.  
<https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>

Install the NVIDIA Container Toolkit with this guide.  
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

#### Design of our Dockerfile

- It is required that all training data is stored in the `dataset` subdirectory, which is mounted into the container at `/dataset`.
- Please note that the file picker functionality is not available. Instead, you will need to manually input the folder path and configuration file path.
- TensorBoard has been separated from the project.
  - TensorBoard is not included in the Docker image.
  - The "Start TensorBoard" button has been hidden.
  - TensorBoard is launched from a distinct container [as shown here](/docker-compose.yaml#L41).
- The browser won't be launched automatically. You will need to manually open the browser and navigate to [http://localhost:7860/](http://localhost:7860/) and [http://localhost:6006/](http://localhost:6006/)
- This Dockerfile has been designed to be easily disposable. You can discard the container at any time and restart it with the new code version.

#### Use the pre-built Docker image

```bash
git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
docker compose up -d
```

To update the system, do `docker compose down && docker compose up -d --pull always`

#### Local docker build

> [!IMPORTANT]  
> Clone the Git repository ***recursively*** to include submodules:  
> `git clone --recursive https://github.com/bmaltais/kohya_ss.git`

```bash
git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
docker compose up -d --build
```

> [!NOTE]  
> Building the image may take up to 20 minutes to complete.

To update the system, ***checkout to the new code version*** and rebuild using `docker compose down && docker compose up -d --build --pull always`

> If you are running on Linux, an alternative Docker container port with fewer limitations is available [here](https://github.com/P2Enjoy/kohya_ss-docker).

#### ashleykleynhans runpod docker builds

You may want to use the following repositories when running on runpod:

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

#### Issue Summary

When training LORA on a TESLA V100, users reported low GPU utilization. Additionally, there was difficulty in specifying GPUs other than the default for training.

#### Potential Solutions

- **GPU Selection:** Users can specify GPU IDs in the setup configuration to select the desired GPUs for training.
- **Improving GPU Load:** Utilizing `adamW8bit` optimizer and increasing the batch size can help achieve 70-80% GPU utilization without exceeding GPU memory limits.

## SDXL training

The documentation in this section will be moved to a separate document later.

## Masked loss

The masked loss is supported in each training script. To enable the masked loss, specify the `--masked_loss` option.

The feature is not fully tested, so there may be bugs. If you find any issues, please open an Issue.

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

## Change History

Added support for SD3 (Dreambooth and Finetuning) and Flux.1 (Dreambooth, LoRA and Finetuning).

See <https://github.com/kohya-ss/sd-scripts/blob/sd3/README.md> for more details.
