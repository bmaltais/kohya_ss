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
  - [Change History](#change-history)
    - [2024/04/28 (v24.0.9)](#20240428-v2409)
    - [2024/04/26 (v24.0.8)](#20240426-v2408)
    - [2024/04/25 (v24.0.7)](#20240425-v2407)
    - [2024/04/22 (v24.0.6)](#20240422-v2406)
    - [2024/04/19 (v24.0.5)](#20240419-v2405)
      - [New Contributors](#new-contributors)
    - [2024/04/18 (v24.0.4)](#20240418-v2404)
      - [What's Changed](#whats-changed)
      - [New Contributors](#new-contributors-1)
    - [2024/04/24 (v24.0.3)](#20240424-v2403)
    - [2024/04/24 (v24.0.2)](#20240424-v2402)
    - [2024/04/17 (v24.0.1)](#20240417-v2401)
      - [Enhancements](#enhancements)
      - [Security and Stability](#security-and-stability)
      - [Shell Execution](#shell-execution)
      - [Miscellaneous](#miscellaneous)
    - [2024/04/10 (v23.1.5)](#20240410-v2315)
      - [Security Improvements](#security-improvements)
    - [2024/04/08 (v23.1.4)](#20240408-v2314)
    - [2024/04/08 (v23.1.3)](#20240408-v2313)
    - [2024/04/08 (v23.1.2)](#20240408-v2312)
    - [2024/04/07 (v23.1.1)](#20240407-v2311)
    - [2024/04/07 (v23.1.0)](#20240407-v2310)

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
   git clone --recursive https://github.com/bmaltais/kohya_ss.git
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

## Change History

### 2024/04/28 (v24.0.9)

- Update tmp config file to carry date and time info as part of the name. This will allow for easy batching of multiple training commands for users that desire to do so.

### 2024/04/26 (v24.0.8)

- Set `max_train_steps` to 0 if not specified in older `.json` config files.

### 2024/04/25 (v24.0.7)

- Prevent crash if tkinter is not installed
- Fix [24.0.6] Train toml config seed type error #2370
- A new docker container is now built with every new release, eliminating the need for manual building. A big thank you to @jim60105 for his hard work in this area. You can find more information about it in the Docker section of the README.

### 2024/04/22 (v24.0.6)

- Make start and stop buttons visible in headless
- Add validation for lr and optimizer arguments

### 2024/04/19 (v24.0.5)

- Hide tensorboard button if tensorflow module is not installed by @bmaltais in <https://github.com/bmaltais/kohya_ss/pull/2347>
- wd14 captioning issue with undesired tags nor tag replacement by @bmaltais in <https://github.com/bmaltais/kohya_ss/pull/2350>
- Changed logger checkbox to dropdown, renamed use_wandb -> log_with by @ccharest93 in <https://github.com/bmaltais/kohya_ss/pull/2352>

#### New Contributors

- @ccharest93 made their first contribution in <https://github.com/bmaltais/kohya_ss/pull/2352>

### 2024/04/18 (v24.0.4)

#### What's Changed

- Fix options.md heading by @bmaltais in <https://github.com/bmaltais/kohya_ss/pull/2337>
- Use correct file extensions when browsing for model file by @b-fission in <https://github.com/bmaltais/kohya_ss/pull/2323>
- Add argument for Gradio's `root_path` to enable reverse proxy support by @hlky in <https://github.com/bmaltais/kohya_ss/pull/2333>
- 2325 quotes wrapping python path cause subprocess cant find target in v2403 by @bmaltais in <https://github.com/bmaltais/kohya_ss/pull/2338>
- 2330 another seemingly new data validation leads to unusable configs 2403 by @bmaltais in <https://github.com/bmaltais/kohya_ss/pull/2339>
- Fix bad Lora parameters by @bmaltais in <https://github.com/bmaltais/kohya_ss/pull/2341>

#### New Contributors

- @b-fission made their first contribution in <https://github.com/bmaltais/kohya_ss/pull/2323>
- @hlky made their first contribution in <https://github.com/bmaltais/kohya_ss/pull/2333>

### 2024/04/24 (v24.0.3)

- Fix issue with sample prompt creation

### 2024/04/24 (v24.0.2)

- Fixed issue with clip_skip not being passed as an int to sd-scripts when using old config.json files.

### 2024/04/17 (v24.0.1)

#### Enhancements

- **User Interface:** Transitioned the GUI to use a TOML file for argument passing to sd-scripts, significantly enhancing security by eliminating the need for command-line interface (CLI) use for sensitive data.
- **Training Tools:** Improved the training and TensorBoard buttons to provide a more intuitive user experience.
- **HuggingFace Integration:** Integrated a HuggingFace section in all trainer tabs, enabling authentication and use of HuggingFace's advanced AI models.
- **Gradio Upgrade:** Upgraded Gradio to version 4.20.0 to fix a previously identified bug impacting the runpod platform.
- **Metadata Support:** Added functionality for metadata capture within the GUI.

#### Security and Stability

- **Code Refactoring:** Extensively rewrote the code to address various security vulnerabilities, including removing the `shell=True` parameter from process calls.
- **Scheduler Update:** Disabled LR Warmup when using the Constant LR Scheduler to prevent traceback errors associated with sd-scripts.

#### Shell Execution

- **Conditional Shell Usage:** Added support for optional shell usage when executing external sd-scripts commands, tailored to meet specific platform needs and recent security updates.

The `gui.bat` and `gui.sh` scripts now include the `--do_not_use_shell` argument to prevent shell execution (`shell=True`) during external process handling.The GUI will automatically set `use_shell` to True internally, as required for proper execution of external commands. To enforce disabling shell execution, use the `--do_not_use_shell` argument.

- **How to Enable Shell Execution via Config File:**
  1. In the `config.toml` file, set `use_shell` to `true` to enable shell usage as per GUI startup settings.
  **Note:** The `--do_not_use_shell` option will override the `config.toml` settings, setting `use_shell` to False even if it is set to True in the config file.

#### Miscellaneous

- Made various other minor improvements and bug fixes to enhance overall functionality and user experience.
- Fixed an issue with existing LoRA network weights were not properly loaded prior to training

### 2024/04/10 (v23.1.5)

- Fix issue with Textual Inversion configuration file selection.
- Upgrade to gradio 4.19.2 to fix several high security risks associated to earlier versions. This is a major upgrade, moving from 3.x to 4.x. Hoping this will not introduce undorseen issues.
- Upgrade transformers to 4.38.0 to fix a low severity security issue.

#### Security Improvements

- Add explicit --do_not_share parameter to kohya_gui.py to avoid sharing the GUI on platforms like Kaggle.
- Remove shell=True from subprocess calls to avoid security issues when using the GUI.
- Limit caption extensions to a fixed set of extensions to limit the risk of finding and replacing text content in unexpected files.

### 2024/04/08 (v23.1.4)

- Relocate config accordion to the top of the GUI.

### 2024/04/08 (v23.1.3)

- Fix dataset preparation bug.

### 2024/04/08 (v23.1.2)

- Added config.toml support for wd14_caption.

### 2024/04/07 (v23.1.1)

- Added support for Huber loss under the Parameters / Advanced tab.

### 2024/04/07 (v23.1.0)

- Update sd-scripts to 0.8.7
  - The default value of `huber_schedule` in Scheduled Huber Loss is changed from `exponential` to `snr`, which is expected to give better results.

  - Highlights
    - The dependent libraries are updated. Please see [Upgrade](#upgrade) and update the libraries.
      - Especially `imagesize` is newly added, so if you cannot update the libraries immediately, please install with `pip install imagesize==1.4.1` separately.
      - `bitsandbytes==0.43.0`, `prodigyopt==1.0`, `lion-pytorch==0.0.6` are included in the requirements.txt.
        - `bitsandbytes` no longer requires complex procedures as it now officially supports Windows.  
      - Also, the PyTorch version is updated to 2.1.2 (PyTorch does not need to be updated immediately). In the upgrade procedure, PyTorch is not updated, so please manually install or update torch, torchvision, xformers if necessary (see [Upgrade PyTorch](#upgrade-pytorch)).
    - When logging to wandb is enabled, the entire command line is exposed. Therefore, it is recommended to write wandb API key and HuggingFace token in the configuration file (`.toml`). Thanks to bghira for raising the issue.
      - A warning is displayed at the start of training if such information is included in the command line.
      - Also, if there is an absolute path, the path may be exposed, so it is recommended to specify a relative path or write it in the configuration file. In such cases, an INFO log is displayed.
      - See [#1123](https://github.com/kohya-ss/sd-scripts/pull/1123) and PR [#1240](https://github.com/kohya-ss/sd-scripts/pull/1240) for details.
    - Colab seems to stop with log output. Try specifying `--console_log_simple` option in the training script to disable rich logging.
    - Other improvements include the addition of masked loss, scheduled Huber Loss, DeepSpeed support, dataset settings improvements, and image tagging improvements. See below for details.

  - Training scripts
    - `train_network.py` and `sdxl_train_network.py` are modified to record some dataset settings in the metadata of the trained model (`caption_prefix`, `caption_suffix`, `keep_tokens_separator`, `secondary_separator`, `enable_wildcard`).
    - Fixed a bug that U-Net and Text Encoders are included in the state in `train_network.py` and `sdxl_train_network.py`. The saving and loading of the state are faster, the file size is smaller, and the memory usage when loading is reduced.
    - DeepSpeed is supported. PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101)  and [#1139](https://github.com/kohya-ss/sd-scripts/pull/1139) Thanks to BootsofLagrangian! See PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101) for details.
    - The masked loss is supported in each training script. PR [#1207](https://github.com/kohya-ss/sd-scripts/pull/1207) See [Masked loss](#masked-loss) for details.
    - Scheduled Huber Loss has been introduced to each training scripts. PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/) Thanks to kabachuha for the PR and cheald, drhead, and others for the discussion! See the PR and [Scheduled Huber Loss](./docs/train_lllite_README.md#scheduled-huber-loss) for details.
    - The options `--noise_offset_random_strength` and `--ip_noise_gamma_random_strength` are added to each training script. These options can be used to vary the noise offset and ip noise gamma in the range of 0 to the specified value. PR [#1177](https://github.com/kohya-ss/sd-scripts/pull/1177) Thanks to KohakuBlueleaf!
    - The options `--save_state_on_train_end` are added to each training script. PR [#1168](https://github.com/kohya-ss/sd-scripts/pull/1168) Thanks to gesen2egee!
    - The options `--sample_every_n_epochs` and `--sample_every_n_steps` in each training script now display a warning and ignore them when a number less than or equal to `0` is specified. Thanks to S-Del for raising the issue.

  - Dataset settings
    - The [English version of the dataset settings documentation](./docs/config_README-en.md) is added. PR [#1175](https://github.com/kohya-ss/sd-scripts/pull/1175) Thanks to darkstorm2150!
    - The `.toml` file for the dataset config is now read in UTF-8 encoding. PR [#1167](https://github.com/kohya-ss/sd-scripts/pull/1167) Thanks to Horizon1704!
    - Fixed a bug that the last subset settings are applied to all images when multiple subsets of regularization images are specified in the dataset settings. The settings for each subset are correctly applied to each image. PR [#1205](https://github.com/kohya-ss/sd-scripts/pull/1205) Thanks to feffy380!
    - Some features are added to the dataset subset settings.
      - `secondary_separator` is added to specify the tag separator that is not the target of shuffling or dropping.
        - Specify `secondary_separator=";;;"`. When you specify `secondary_separator`, the part is not shuffled or dropped.
      - `enable_wildcard` is added. When set to `true`, the wildcard notation `{aaa|bbb|ccc}` can be used. The multi-line caption is also enabled.
      - `keep_tokens_separator` is updated to be used twice in the caption. When you specify `keep_tokens_separator="|||"`, the part divided by the second `|||` is not shuffled or dropped and remains at the end.
      - The existing features `caption_prefix` and `caption_suffix` can be used together. `caption_prefix` and `caption_suffix` are processed first, and then `enable_wildcard`, `keep_tokens_separator`, shuffling and dropping, and `secondary_separator` are processed in order.
      - See [Dataset config](./docs/config_README-en.md) for details.
    - The dataset with DreamBooth method supports caching image information (size, caption). PR [#1178](https://github.com/kohya-ss/sd-scripts/pull/1178) and [#1206](https://github.com/kohya-ss/sd-scripts/pull/1206) Thanks to KohakuBlueleaf! See [DreamBooth method specific options](./docs/config_README-en.md#dreambooth-specific-options) for details.

  - Image tagging (not implemented yet in the GUI)
    - The support for v3 repositories is added to `tag_image_by_wd14_tagger.py` (`--onnx` option only). PR [#1192](https://github.com/kohya-ss/sd-scripts/pull/1192) Thanks to sdbds!
      - Onnx may need to be updated. Onnx is not installed by default, so please install or update it with `pip install onnx==1.15.0 onnxruntime-gpu==1.17.1` etc. Please also check the comments in `requirements.txt`.
    - The model is now saved in the subdirectory as `--repo_id` in `tag_image_by_wd14_tagger.py` . This caches multiple repo_id models. Please delete unnecessary files under `--model_dir`.
    - Some options are added to `tag_image_by_wd14_tagger.py`.
      - Some are added in PR [#1216](https://github.com/kohya-ss/sd-scripts/pull/1216) Thanks to Disty0!
      - Output rating tags `--use_rating_tags` and `--use_rating_tags_as_last_tag`
      - Output character tags first `--character_tags_first`
      - Expand character tags and series `--character_tag_expand`
      - Specify tags to output first `--always_first_tags`
      - Replace tags `--tag_replacement`
      - See [Tagging documentation](./docs/wd14_tagger_README-en.md) for details.
    - Fixed an error when specifying `--beam_search` and a value of 2 or more for `--num_beams` in `make_captions.py`.

  - About Masked loss
    The masked loss is supported in each training script. To enable the masked loss, specify the `--masked_loss` option.

    The feature is not fully tested, so there may be bugs. If you find any issues, please open an Issue.

    ControlNet dataset is used to specify the mask. The mask images should be the RGB images. The pixel value 255 in R channel is treated as the mask (the loss is calculated only for the pixels with the mask), and 0 is treated as the non-mask. The pixel values 0-255 are converted to 0-1 (i.e., the pixel value 128 is treated as the half weight of the loss). See details for the dataset specification in the [LLLite documentation](./docs/train_lllite_README.md#preparing-the-dataset).

  - About Scheduled Huber Loss
    Scheduled Huber Loss has been introduced to each training scripts. This is a method to improve robustness against outliers or anomalies (data corruption) in the training data.

    With the traditional MSE (L2) loss function, the impact of outliers could be significant, potentially leading to a degradation in the quality of generated images. On the other hand, while the Huber loss function can suppress the influence of outliers, it tends to compromise the reproduction of fine details in images.

    To address this, the proposed method employs a clever application of the Huber loss function. By scheduling the use of Huber loss in the early stages of training (when noise is high) and MSE in the later stages, it strikes a balance between outlier robustness and fine detail reproduction.

    Experimental results have confirmed that this method achieves higher accuracy on data containing outliers compared to pure Huber loss or MSE. The increase in computational cost is minimal.

    The newly added arguments loss_type, huber_schedule, and huber_c allow for the selection of the loss function type (Huber, smooth L1, MSE), scheduling method (exponential, constant, SNR), and Huber's parameter. This enables optimization based on the characteristics of the dataset.

    See PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/) for details.

    - `loss_type`: Specify the loss function type. Choose `huber` for Huber loss, `smooth_l1` for smooth L1 loss, and `l2` for MSE loss. The default is `l2`, which is the same as before.
    - `huber_schedule`: Specify the scheduling method. Choose `exponential`, `constant`, or `snr`. The default is `snr`.
    - `huber_c`: Specify the Huber's parameter. The default is `0.1`.

    Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.`

- Added GUI support for the new parameters listed above.
- Moved accelerate launch parameters to a new `Accelerate launch` accordion above the `Model` accordion.
- Added support for `Debiased Estimation loss` to Dreambooth settings.
- Added support for "Dataset Preparation" defaults via the config.toml file.
- Added a field to allow for the input of extra accelerate launch arguments.
- Added new caption tool from <https://github.com/kainatquaderee>
