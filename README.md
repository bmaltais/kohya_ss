# Kohya's GUI

This repository mostly provides a Gradio GUI for [Kohya's Stable Diffusion trainers](https://github.com/kohya-ss/sd-scripts)... but support for Linux OS is also provided through community contributions. Macos is not great at the moment... but might work if the wind blow in the right direction...

The GUI allows you to set the training parameters and generate and run the required CLI commands to train the model.

## Table of Contents

- [Kohya's GUI](#kohyas-gui)
  - [Table of Contents](#table-of-contents)
  - [🦒 Colab](#-colab)
  - [Installation](#installation)
    - [Windows](#windows)
      - [Windows Pre-requirements](#windows-pre-requirements)
      - [Setup](#setup)
      - [Optional: CUDNN 8.6](#optional-cudnn-86)
    - [Linux and macOS](#linux-and-macos)
      - [Linux Pre-requirements](#linux-pre-requirements)
      - [Setup](#setup-1)
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
  - [Dreambooth](#dreambooth)
  - [Finetune](#finetune)
  - [Train Network](#train-network)
  - [LoRA](#lora)
  - [Sample image generation during training](#sample-image-generation-during-training)
  - [Troubleshooting](#troubleshooting)
    - [Page File Limit](#page-file-limit)
    - [No module called tkinter](#no-module-called-tkinter)
    - [FileNotFoundError](#filenotfounderror)
  - [SDXL training](#sdxl-training)
    - [Training scripts for SDXL](#training-scripts-for-sdxl)
    - [Utility scripts for SDXL](#utility-scripts-for-sdxl)
    - [Tips for SDXL training](#tips-for-sdxl-training)
    - [Format of Textual Inversion embeddings for SDXL](#format-of-textual-inversion-embeddings-for-sdxl)
    - [ControlNet-LLLite](#controlnet-lllite)
    - [Sample image generation during training](#sample-image-generation-during-training-1)
  - [Change History](#change-history)

## 🦒 Colab

This Colab notebook was not created or maintained by me; however, it appears to function effectively. The source can be found at: https://github.com/camenduru/kohya_ss-colab.

I would like to express my gratitude to camendutu for their valuable contribution. If you encounter any issues with the Colab notebook, please report them on their repository.

| Colab                                                                                                                                                                          | Info               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/kohya_ss-colab/blob/main/kohya_ss_colab.ipynb) | kohya_ss_gui_colab |

## Installation

### Windows

#### Windows Pre-requirements

To install the necessary dependencies on a Windows system, follow these steps:

1. Install [Python 3.10](https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe).
   - During the installation process, ensure that you select the option to add Python to the 'PATH' environment variable.

2. Install [Git](https://git-scm.com/download/win).

3. Install the [Visual Studio 2015, 2017, 2019, and 2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

#### Setup

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

   During the accelerate config step use the default values as proposed during the configuration unless you know your hardware demand otherwise. The amount of VRAM on your GPU does not have an impact on the values used.

#### Optional: CUDNN 8.6

The following steps are optional but can improve the learning speed for owners of NVIDIA 30X0/40X0 GPUs. These steps enable larger training batch sizes and faster training speeds.

Please note that the CUDNN 8.6 DLLs needed for this process cannot be hosted on GitHub due to file size limitations. You can download them [here](https://github.com/bmaltais/python-library/raw/main/cudnn_windows.zip) to boost sample generation speed (almost 50% on a 4090 GPU). After downloading the ZIP file, follow the installation steps below:

1. Unzip the downloaded file and place the `cudnn_windows` folder in the root directory of the `kohya_ss` repository.

2. Run .\setup.bat and select the option to install cudnn.

### Linux and macOS

#### Linux Pre-requirements

To install the necessary dependencies on a Linux system, ensure that you fulfill the following requirements:

- Ensure that `venv` support is pre-installed. You can install it on Ubuntu 22.04 using the command:
  ```shell
  apt install python3.10-venv
  ```

- Install the cudNN drivers by following the instructions provided in [this link](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64).

- Make sure you have Python version 3.10.6 or higher (but lower than 3.11.0) installed on your system.

- If you are using WSL2, set the `LD_LIBRARY_PATH` environment variable by executing the following command:
  ```shell
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib/
  ```

#### Setup

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

5. Run the gui with:
   ```shell
   ./gui.sh --share --headless
   ```

   or with this if you expose 7860 directly via the runpod configuration

   ```shell
   ./gui.sh --listen=0.0.0.0 --headless
   ```

6. Connect to the public URL displayed after the installation process is completed.

#### Pre-built Runpod template

To run from a pre-built Runpod template you can:

1. Open the Runpod template by clicking on https://runpod.io/gsc?template=ya6013lj5a&ref=w18gds2n

2. Deploy the template on the desired host

3. Once deployed connect to the Runpod on HTTP 3010 to connect to kohya_ss GUI. You can also connect to auto1111 on HTTP 3000.


### Docker

#### Local docker build

If you prefer to use Docker, follow the instructions below:

1. Ensure that you have Git and Docker installed on your Windows or Linux system.

2. Open your OS shell (Command Prompt or Terminal) and run the following commands:

   ```bash
   git clone https://github.com/bmaltais/kohya_ss.git
   cd kohya_ss
   docker compose create
   docker compose build
   docker compose run --service-ports kohya-ss-gui
   ```

   Note: The initial run may take up to 20 minutes to complete.

   Please be aware of the following limitations when using Docker:

   - All training data must be placed in the `dataset` subdirectory, as the Docker container cannot access files from other directories.
   - The file picker feature is not functional. You need to manually set the folder path and config file path.
   - Dialogs may not work as expected, and it is recommended to use unique file names to avoid conflicts.
   - There is no built-in auto-update support. To update the system, you must run update scripts outside of Docker and rebuild using `docker compose build`.

   If you are running Linux, an alternative Docker container port with fewer limitations is available [here](https://github.com/P2Enjoy/kohya_ss-docker).

#### ashleykleynhans runpod docker builds

You may want to use the following Dockerfile repos to build the images:

   - Standalone Kohya_ss template: https://github.com/ashleykleynhans/kohya-docker
   - Auto1111 + Kohya_ss GUI template: https://github.com/ashleykleynhans/stable-diffusion-docker

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

1. Open a terminal and navigate to the root

 directory of the project.

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

## Dreambooth

For specific instructions on using the Dreambooth solution, please refer to the [Dreambooth README](https://github.com/bmaltais/kohya_ss/blob/master/train_db_README.md).

## Finetune

For specific instructions on using the Finetune solution, please refer to the [Finetune README](https://github.com/bmaltais/kohya_ss/blob/master/fine_tune_README.md).

## Train Network

For specific instructions on training a network, please refer to the [Train network README](https://github.com/bmaltais/kohya_ss/blob/master/train_network_README.md).

## LoRA

To train a LoRA, you can currently use the `train_network.py` code. You can create a LoRA network by using the all-in-one GUI.

Once you have created the LoRA network, you can generate images using auto1111 by installing [this extension](https://github.com/kohya-ss/sd-webui-additional-networks).

The following are the names of LoRA types used in this repository:

1. LoRA-LierLa: LoRA for Linear layers and Conv2d layers with a 1x1 kernel.

2. LoRA-C3Lier: LoRA for Conv2d layers with a 3x3 kernel, in addition to LoRA-LierLa.

LoRA-LierLa is the default LoRA type for `train_network.py` (without `conv_dim` network argument). You can use LoRA-LierLa with our extension for AUTOMATIC1111's Web UI or the built-in LoRA feature of the Web UI.

To use LoRA-C3Lier with the Web UI, please use our extension.

## Sample image generation during training

A prompt file might look like this, for example:

```
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

The prompt weighting such as `( )` and `[ ]` are working.

## Troubleshooting

If you encounter any issues, refer to the troubleshooting steps below.

### Page File Limit

If you encounter an X error related to the page file, you may need to increase the page file size limit in Windows.

### No module called tkinter

If you encounter an error indicating that the module `tkinter` is not found, try reinstalling Python 3.10 on your system.

### FileNotFoundError

If you come across a `FileNotFoundError`, it is likely due to an installation issue. Make sure you do not have any locally installed Python modules that could conflict with the ones installed in the virtual environment. You can uninstall them by following these steps:

1. Open a new PowerShell terminal and ensure that no virtual environment is active.

2. Run the following commands to create a backup file of your locally installed pip packages and then uninstall them:
   ```powershell
   pip freeze > uninstall.txt
   pip uninstall -r uninstall.txt
   ```

   After uninstalling the local packages, redo the installation steps within the `kohya_ss` virtual environment.


## SDXL training

The documentation in this section will be moved to a separate document later.

### Training scripts for SDXL

- `sdxl_train.py` is a script for SDXL fine-tuning. The usage is almost the same as `fine_tune.py`, but it also supports DreamBooth dataset.
  - `--full_bf16` option is added. Thanks to KohakuBlueleaf!
    - This option enables the full bfloat16 training (includes gradients). This option is useful to reduce the GPU memory usage.
    - The full bfloat16 training might be unstable. Please use it at your own risk.
  - The different learning rates for each U-Net block are now supported in sdxl_train.py. Specify with `--block_lr` option. Specify 23 values separated by commas like `--block_lr 1e-3,1e-3 ... 1e-3`.
    - 23 values correspond to `0: time/label embed, 1-9: input blocks 0-8, 10-12: mid blocks 0-2, 13-21: output blocks 0-8, 22: out`.
- `prepare_buckets_latents.py` now supports SDXL fine-tuning.

- `sdxl_train_network.py` is a script for LoRA training for SDXL. The usage is almost the same as `train_network.py`.

- Both scripts has following additional options:
  - `--cache_text_encoder_outputs` and `--cache_text_encoder_outputs_to_disk`: Cache the outputs of the text encoders. This option is useful to reduce the GPU memory usage. This option cannot be used with options for shuffling or dropping the captions.
  - `--no_half_vae`: Disable the half-precision (mixed-precision) VAE. VAE for SDXL seems to produce NaNs in some cases. This option is useful to avoid the NaNs.

- `--weighted_captions` option is not supported yet for both scripts.

- `sdxl_train_textual_inversion.py` is a script for Textual Inversion training for SDXL. The usage is almost the same as `train_textual_inversion.py`.
  - `--cache_text_encoder_outputs` is not supported.
  - There are two options for captions:
    1. Training with captions. All captions must include the token string. The token string is replaced with multiple tokens.
    2. Use `--use_object_template` or `--use_style_template` option. The captions are generated from the template. The existing captions are ignored.
  - See below for the format of the embeddings.

- `--min_timestep` and `--max_timestep` options are added to each training script. These options can be used to train U-Net with different timesteps. The default values are 0 and 1000.

### Utility scripts for SDXL

- `tools/cache_latents.py` is added. This script can be used to cache the latents to disk in advance.
  - The options are almost the same as `sdxl_train.py'. See the help message for the usage.
  - Please launch the script as follows:
    `accelerate launch  --num_cpu_threads_per_process 1 tools/cache_latents.py ...`
  - This script should work with multi-GPU, but it is not tested in my environment.

- `tools/cache_text_encoder_outputs.py` is added. This script can be used to cache the text encoder outputs to disk in advance.
  - The options are almost the same as `cache_latents.py` and `sdxl_train.py`. See the help message for the usage.

- `sdxl_gen_img.py` is added. This script can be used to generate images with SDXL, including LoRA, Textual Inversion and ControlNet-LLLite. See the help message for the usage.

### Tips for SDXL training

- The default resolution of SDXL is 1024x1024.
- The fine-tuning can be done with 24GB GPU memory with the batch size of 1. For 24GB GPU, the following options are recommended __for the fine-tuning with 24GB GPU memory__:
  - Train U-Net only.
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` option and caching latents.
  - Use Adafactor optimizer. RMSprop 8bit or Adagrad 8bit may work. AdamW 8bit doesn't seem to work.
- The LoRA training can be done with 8GB GPU memory (10GB recommended). For reducing the GPU memory usage, the following options are recommended:
  - Train U-Net only.
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` option and caching latents.
  - Use one of 8bit optimizers or Adafactor optimizer.
  - Use lower dim (4 to 8 for 8GB GPU).
- `--network_train_unet_only` option is highly recommended for SDXL LoRA. Because SDXL has two text encoders, the result of the training will be unexpected.
- PyTorch 2 seems to use slightly less GPU memory than PyTorch 1.
- `--bucket_reso_steps` can be set to 32 instead of the default value 64. Smaller values than 32 will not work for SDXL training.

Example of the optimizer settings for Adafactor with the fixed learning rate:

```toml
optimizer_type = "adafactor"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False" ]
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100
learning_rate = 4e-7 # SDXL original learning rate
```

### Format of Textual Inversion embeddings for SDXL

```python
from safetensors.torch import save_file

state_dict = {"clip_g": embs_for_text_encoder_1280, "clip_l": embs_for_text_encoder_768}
save_file(state_dict, file)
```

### ControlNet-LLLite

ControlNet-LLLite, a novel method for ControlNet with SDXL, is added. See [documentation](./docs/train_lllite_README.md) for details.

### Sample image generation during training
  A prompt file might look like this, for example

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following can be used.

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

  The prompt weighting such as `( )` and `[ ]` are working.


## Change History
* 2024/03/02 (v22.7.0)
- Major code refactoring thanks to @wkpark , This will make updating sd-script cleaner by keeping sd-scripts files separate from the GUI files.
* 2024/02/17 (v22.6.2)
- Fix issue with Lora Extract GUI
  - Fix syntax issue where parameter lora_network_weights is actually called network_weights
- Merge sd-scripts v0.8.4 code update
  - Fixed a bug that the VRAM usage without Text Encoder training is larger than before in training scripts for LoRA etc (`train_network.py`, `sdxl_train_network.py`).
    - Text Encoders were not moved to CPU.
  - Fixed typos. Thanks to akx! [PR #1053](https://github.com/kohya-ss/sd-scripts/pull/1053)
  - The log output has been improved. PR [#905](https://github.com/kohya-ss/sd-scripts/pull/905) Thanks to shirayu!
    - The log is formatted by default. The `rich` library is required. Please see [Upgrade](#upgrade) and update the library.
    - If `rich` is not installed, the log output will be the same as before.
    - The following options are available in each training script:
    - `--console_log_simple` option can be used to switch to the previous log output.
    - `--console_log_level` option can be used to specify the log level. The default is `INFO`.
    - `--console_log_file` option can be used to output the log to a file. The default is `None` (output to the console).
  - The sample image generation during multi-GPU training is now done with multiple GPUs. PR [#1061](https://github.com/kohya-ss/sd-scripts/pull/1061) Thanks to DKnight54!
  - The support for mps devices is improved. PR [#1054](https://github.com/kohya-ss/sd-scripts/pull/1054) Thanks to akx! If mps device exists instead of CUDA, the mps device is used automatically.
  - The `--new_conv_rank` option to specify the new rank of Conv2d is added to `networks/resize_lora.py`. PR [#1102](https://github.com/kohya-ss/sd-scripts/pull/1102) Thanks to mgz-dev!
  - An option `--highvram` to disable the optimization for environments with little VRAM is added to the training scripts. If you specify it when there is enough VRAM, the operation will be faster.
    - Currently, only the cache part of latents is optimized.
  - The IPEX support is improved. PR [#1086](https://github.com/kohya-ss/sd-scripts/pull/1086) Thanks to Disty0!
  - Fixed a bug that `svd_merge_lora.py` crashes in some cases. PR [#1087](https://github.com/kohya-ss/sd-scripts/pull/1087) Thanks to mgz-dev!
  - DyLoRA is fixed to work with SDXL. PR [#1126](https://github.com/kohya-ss/sd-scripts/pull/1126) Thanks to tamlog06!
  - The common image generation script `gen_img.py` for SD 1/2 and SDXL is added. The basic functions are the same as the scripts for SD 1/2 and SDXL, but some new features are added.
    - External scripts to generate prompts can be supported. It can be called with `--from_module` option. (The documentation will be added later)
    - The normalization method after prompt weighting can be specified with `--emb_normalize_mode` option. `original` is the original method, `abs` is the normalization with the average of the absolute values, `none` is no normalization.
  - Gradual Latent Hires fix is added to each generation script. See [here](./docs/gen_img_README-ja.md#about-gradual-latent) for details.
  
* 2024/02/15 (v22.6.1)
- Add support for multi-gpu parameters in the GUI under the "Parameters > Advanced" tab.
- Significant rewrite of how parameters are created in the code. I hope I did not break anything in the process... Will make the code easier to update.
- Update TW locallisation
- Update gradio module version to latest 3.x

* 2024/01/27 (v22.6.0)
- Merge sd-scripts v0.8.3 code update
  - Fixed a bug that the training crashes when `--fp8_base` is specified with `--save_state`. PR [#1079](https://github.com/kohya-ss/sd-scripts/pull/1079) Thanks to feffy380!
    - `safetensors` is updated. Please see [Upgrade](#upgrade) and update the library.
  - Fixed a bug that the training crashes when `network_multiplier` is specified with multi-GPU training. PR [#1084](https://github.com/kohya-ss/sd-scripts/pull/1084) Thanks to fireicewolf!
  - Fixed a bug that the training crashes when training ControlNet-LLLite.

- Merge sd-scripts v0.8.2 code update
  - [Experimental] The `--fp8_base` option is added to the training scripts for LoRA etc. The base model (U-Net, and Text Encoder when training modules for Text Encoder) can be trained with fp8. PR [#1057](https://github.com/kohya-ss/sd-scripts/pull/1057) Thanks to KohakuBlueleaf!
    - Please specify `--fp8_base` in `train_network.py` or `sdxl_train_network.py`.
    - PyTorch 2.1 or later is required.
    - If you use xformers with PyTorch 2.1, please see [xformers repository](https://github.com/facebookresearch/xformers) and install the appropriate version according to your CUDA version.
    - The sample image generation during training consumes a lot of memory. It is recommended to turn it off.

  - [Experimental] The network multiplier can be specified for each dataset in the training scripts for LoRA etc.
    - This is an experimental option and may be removed or changed in the future.
    - For example, if you train with state A as `1.0` and state B as `-1.0`, you may be able to generate by switching between state A and B depending on the LoRA application rate.
    - Also, if you prepare five states and train them as `0.2`, `0.4`, `0.6`, `0.8`, and `1.0`, you may be able to generate by switching the states smoothly depending on the application rate.
    - Please specify `network_multiplier` in `[[datasets]]` in `.toml` file.

  - Some options are added to `networks/extract_lora_from_models.py` to reduce the memory usage.
    - `--load_precision` option can be used to specify the precision when loading the model. If the model is saved in fp16, you can reduce the memory usage by specifying `--load_precision fp16` without losing precision.
    - `--load_original_model_to` option can be used to specify the device to load the original model. `--load_tuned_model_to` option can be used to specify the device to load the derived model. The default is `cpu` for both options, but you can specify `cuda` etc. You can reduce the memory usage by loading one of them to GPU. This option is available only for SDXL.

  - The gradient synchronization in LoRA training with multi-GPU is improved. PR [#1064](https://github.com/kohya-ss/sd-scripts/pull/1064) Thanks to KohakuBlueleaf!

  - The code for Intel IPEX support is improved. PR [#1060](https://github.com/kohya-ss/sd-scripts/pull/1060) Thanks to akx!

  - Fixed a bug in multi-GPU Textual Inversion training.

  - `.toml` example for network multiplier

    ```toml
    [general]
    [[datasets]]
    resolution = 512
    batch_size = 8
    network_multiplier = 1.0

    ... subset settings ...

    [[datasets]]
    resolution = 512
    batch_size = 8
    network_multiplier = -1.0

    ... subset settings ...
    ```

- Merge sd-scripts v0.8.1 code update

  - Fixed a bug that the VRAM usage without Text Encoder training is larger than before in training scripts for LoRA etc (`train_network.py`, `sdxl_train_network.py`).
    - Text Encoders were not moved to CPU.

  - Fixed typos. Thanks to akx! [PR #1053](https://github.com/kohya-ss/sd-scripts/pull/1053)

* 2024/01/15 (v22.5.0)
- Merged sd-scripts v0.8.0 updates
  - Diffusers, Accelerate, Transformers and other related libraries have been updated. Please update the libraries with [Upgrade](#upgrade).
    - Some model files (Text Encoder without position_id) based on the latest Transformers can be loaded.
  - `torch.compile` is supported (experimental). PR [#1024](https://github.com/kohya-ss/sd-scripts/pull/1024) Thanks to p1atdev!
    - This feature works only on Linux or WSL.
    - Please specify `--torch_compile` option in each training script.
    - You can select the backend with `--dynamo_backend` option. The default is `"inductor"`. `inductor` or `eager` seems to work.
    - Please use `--spda` option instead of `--xformers` option.
    - PyTorch 2.1 or later is recommended.
    - Please see [PR](https://github.com/kohya-ss/sd-scripts/pull/1024) for details.
  - The session name for wandb can be specified with `--wandb_run_name` option. PR [#1032](https://github.com/kohya-ss/sd-scripts/pull/1032) Thanks to hopl1t!
  - IPEX library is updated. PR [#1030](https://github.com/kohya-ss/sd-scripts/pull/1030) Thanks to Disty0!
  - Fixed a bug that Diffusers format model cannot be saved.
- Fix LoRA config display after load that would sometime hide some of the feilds

* 2024/01/02 (v22.4.1)
- Minor bug fixed and enhancements.

* 2023/12/28 (v22.4.0)
- Fixed to work `tools/convert_diffusers20_original_sd.py`. Thanks to Disty0! PR [#1016](https://github.com/kohya-ss/sd-scripts/pull/1016)
- The issues in multi-GPU training are fixed. Thanks to Isotr0py! PR [#989](https://github.com/kohya-ss/sd-scripts/pull/989) and [#1000](https://github.com/kohya-ss/sd-scripts/pull/1000)
  - `--ddp_gradient_as_bucket_view` and `--ddp_bucket_view`options are added to `sdxl_train.py`. Please specify these options for multi-GPU training.
- IPEX support is updated. Thanks to Disty0!
- Fixed the bug that the size of the bucket becomes less than `min_bucket_reso`. Thanks to Cauldrath! PR [#1008](https://github.com/kohya-ss/sd-scripts/pull/1008)
- `--sample_at_first` option is added to each training script. This option is useful to generate images at the first step, before training. Thanks to shirayu! PR [#907](https://github.com/kohya-ss/sd-scripts/pull/907)
- `--ss` option is added to the sampling prompt in training. You can specify the scheduler for the sampling like `--ss euler_a`. Thanks to shirayu! PR [#906](https://github.com/kohya-ss/sd-scripts/pull/906)
- `keep_tokens_separator` is added to the dataset config. This option is useful to keep (prevent from shuffling) the tokens in the captions. See [#975](https://github.com/kohya-ss/sd-scripts/pull/975) for details. Thanks to Linaqruf!
  - You can specify the separator with an option like `--keep_tokens_separator "|||"` or with `keep_tokens_separator: "|||"` in `.toml`. The tokens before `|||` are not shuffled.
- Attention processor hook is added. See [#961](https://github.com/kohya-ss/sd-scripts/pull/961) for details. Thanks to rockerBOO!
- The optimizer `PagedAdamW` is added. Thanks to xzuyn! PR [#955](https://github.com/kohya-ss/sd-scripts/pull/955)
- NaN replacement in SDXL VAE is sped up. Thanks to liubo0902! PR [#1009](https://github.com/kohya-ss/sd-scripts/pull/1009)
- Fixed the path error in `finetune/make_captions.py`. Thanks to CjangCjengh! PR [#986](https://github.com/kohya-ss/sd-scripts/pull/986)

* 2023/12/20 (v22.3.1)
- Add goto button to manual caption utility
- Add missing options for various LyCORIS training algorithms
- Refactor how feilds are shown or hidden
- Made max value for network and convolution rank 512 except for LyCORIS/LoKr.

* 2023/12/06 (v22.3.0)
- Merge sd-scripts updates:
  - `finetune\tag_images_by_wd14_tagger.py` now supports the separator other than `,` with `--caption_separator` option. Thanks to KohakuBlueleaf! PR [#913](https://github.com/kohya-ss/sd-scripts/pull/913)
  - Min SNR Gamma with V-predicition (SD 2.1) is fixed. Thanks to feffy380! PR[#934](https://github.com/kohya-ss/sd-scripts/pull/934)
    - See [#673](https://github.com/kohya-ss/sd-scripts/issues/673) for details.
  - `--min_diff` and `--clamp_quantile` options are added to `networks/extract_lora_from_models.py`. Thanks to wkpark! PR [#936](https://github.com/kohya-ss/sd-scripts/pull/936)
    - The default values are same as the previous version.
  - Deep Shrink hires fix is supported in `sdxl_gen_img.py` and `gen_img_diffusers.py`.
    - `--ds_timesteps_1` and `--ds_timesteps_2` options denote the timesteps of the Deep Shrink for the first and second stages.
    - `--ds_depth_1` and `--ds_depth_2` options denote the depth (block index) of the Deep Shrink for the first and second stages.
    - `--ds_ratio` option denotes the ratio of the Deep Shrink. `0.5` means the half of the original latent size for the Deep Shrink.
    - `--dst1`, `--dst2`, `--dsd1`, `--dsd2` and `--dsr` prompt options are also available.
  - Add GLoRA support
-