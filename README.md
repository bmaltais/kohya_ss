# Kohya's GUI

This repository mostly provides a Windows-focused Gradio GUI for [Kohya's Stable Diffusion trainers](https://github.com/kohya-ss/sd-scripts)... but support for Linux OS is also provided through community contributions. Macos is not great at the moment.

The GUI allows you to set the training parameters and generate and run the required CLI commands to train the model.

## Table of Contents

1. [Tutorials](#tutorials)
2. [Installation](#installation)
   1. [Windows](#windows)
      1. [Windows Pre-requirements](#windows-pre-requirements)
      2. [Setup](#setup)
      3. [Optional: CUDNN 8.6](#optional-cudnn-86)
   2. [Linux and macOS](#linux-and-macos)
      1. [Linux Pre-requirements](#linux-pre-requirements)
      2. [Setup](#setup-1)
      3. [Install Location](#install-location)
   3. [Runpod](#runpod)
   4. [Docker](#docker)
3. [Upgrading](#upgrading)
   1. [Windows Upgrade](#windows-upgrade)
   2. [Linux and macOS Upgrade](#linux-and-macos-upgrade)
4. [Starting GUI Service](#starting-gui-service)
   1. [Launching the GUI on Windows](#launching-the-gui-on-windows)
   2. [Launching the GUI on Linux and macOS](#launching-the-gui-on-linux-and-macos)
5. [Dreambooth](#dreambooth)
6. [Finetune](#finetune)
7. [Train Network](#train-network)
8. [LoRA](#lora)
9. [Sample image generation during training](#sample-image-generation-during-training)
10. [Troubleshooting](#troubleshooting)
   1. [Page File Limit](#page-file-limit)
   2. [No module called tkinter](#no-module-called-tkinter)
   3. [FileNotFoundError](#filenotfounderror)
11. [Change History](#change-history)

## Tutorials

[How to Create a LoRA Part 1: Dataset Preparation](https://www.youtube.com/watch?v=N4_-fB62Hwk):

[![LoRA Part 1 Tutorial](https://img.youtube.com/vi/N4_-fB62Hwk/0.jpg)](https://www.youtube.com/watch?v=N4_-fB62Hwk)

[How to Create a LoRA Part 2: Training the Model](https://www.youtube.com/watch?v=k5imq01uvUY):
### About SDXL training

The feature of SDXL training is now available in sdxl branch as an experimental feature. 

Summary of the feature:

- `sdxl_train.py` is a script for SDXL fine-tuning. The usage is almost the same as `fine_tune.py`, but it also supports DreamBooth dataset.
  - `--full_bf16` option is added. Thanks to KohakuBlueleaf!
    - This option enables the full bfloat16 training (includes gradients). This option is useful to reduce the GPU memory usage. 
    - However, bitsandbytes==0.35 doesn't seem to support this. Please use a newer version of bitsandbytes or another optimizer.
    - I cannot find bitsandbytes>0.35.0 that works correctly on Windows.
    - In addition, the full bfloat16 training might be unstable. Please use it at your own risk.
- `prepare_buckets_latents.py` now supports SDXL fine-tuning.
- `sdxl_train_network.py` is a script for LoRA training for SDXL. The usage is almost the same as `train_network.py`.
- Both scripts has following additional options:
  - `--cache_text_encoder_outputs`: Cache the outputs of the text encoders. This option is useful to reduce the GPU memory usage. This option cannot be used with options for shuffling or dropping the captions.
  - `--no_half_vae`: Disable the half-precision (mixed-precision) VAE. VAE for SDXL seems to produce NaNs in some cases. This option is useful to avoid the NaNs.
- The image generation during training is now available. However, the VAE for SDXL seems to produce NaNs in some cases when using `fp16`. The images will be black. Currently, the NaNs cannot be avoided even with `--no_half_vae` option. It works with `bf16` or without mixed precision.

- `--weighted_captions` option is not supported yet for both scripts.
- `--min_timestep` and `--max_timestep` options are added to each training script. These options can be used to train U-Net with different timesteps. The default values are 0 and 1000.

- `sdxl_train_textual_inversion.py` is a script for Textual Inversion training for SDXL. The usage is almost the same as `train_textual_inversion.py`.
  - `--cache_text_encoder_outputs` is not supported.
  - `token_string` must be alphabet only currently, due to the limitation of the open-clip tokenizer.
  - There are two options for captions:
    1. Training with captions. All captions must include the token string. The token string is replaced with multiple tokens.
    2. Use `--use_object_template` or `--use_style_template` option. The captions are generated from the template. The existing captions are ignored.
  - See below for the format of the embeddings.
  
- `sdxl_gen_img.py` is added. This script can be used to generate images with SDXL, including LoRA. See the help message for the usage.
  - Textual Inversion is supported, but the name for the embeds in the caption becomes alphabet only. For example, `neg_hand_v1.safetensors` can be activated with `neghandv`.

`requirements.txt` is updated to support SDXL training. 

#### Tips for SDXL training

- The default resolution of SDXL is 1024x1024.
- The fine-tuning can be done with 24GB GPU memory with the batch size of 1. For 24GB GPU, the following options are recommended:
  - Train U-Net only.
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` option and caching latents.
  - Use Adafactor optimizer. RMSprop 8bit or Adagrad 8bit may work. AdamW 8bit doesn't seem to work.
- The LoRA training can be done with 12GB GPU memory.
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

### Format of Textual Inversion embeddings

```python
from safetensors.torch import save_file

state_dict = {"clip_g": embs_for_text_encoder_1280, "clip_l": embs_for_text_encoder_768}
save_file(state_dict, file)
```

### TODO

- [ ] Support conversion of Diffusers SDXL models.
- [ ] Support `--weighted_captions` option.
- [ ] Change `--output_config` option to continue the training.
- [ ] Extend `--full_bf16` for all the scripts.
- [x] Support Textual Inversion training.

## About requirements.txt

[![LoRA Part 2 Tutorial](https://img.youtube.com/vi/k5imq01uvUY/0.jpg)](https://www.youtube.com/watch?v=k5imq01uvUY)

Newer Tutorial: [Generate Studio Quality Realistic Photos By Kohya LoRA Stable Diffusion Training](https://www.youtube.com/watch?v=TpuDOsuKIBo):
The scripts are tested with PyTorch 1.12.1 and 2.0.1, Diffusers 0.17.1.

[![Newer Tutorial: Generate Studio Quality Realistic Photos By Kohya LoRA Stable Diffusion Training](https://user-images.githubusercontent.com/19240467/235306147-85dd8126-f397-406b-83f2-368927fa0281.png)](https://www.youtube.com/watch?v=TpuDOsuKIBo)

Newer Tutorial: [How To Install And Use Kohya LoRA GUI / Web UI on RunPod IO](https://www.youtube.com/watch?v=3uzCNrQao3o):

[![How To Install And Use Kohya LoRA GUI / Web UI on RunPod IO With Stable Diffusion & Automatic1111](https://github-production-user-asset-6210df.s3.amazonaws.com/19240467/238678226-0c9c3f7d-c308-4793-b790-999fdc271372.png)](https://www.youtube.com/watch?v=3uzCNrQao3o)

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
   ```
   git clone https://github.com/bmaltais/kohya_ss.git
   ```

3. Change into the `kohya_ss` directory:
   ```
   cd kohya_ss
   ```

4. Run the setup script by executing the following command:
   ```
   .\setup.bat
   ```

#### Optional: CUDNN 8.6

The following steps are optional but can improve the learning speed for owners of NVIDIA 30X0/40X0 GPUs. These steps enable larger training batch sizes and faster training speeds.

Please note that the CUDNN 8.6 DLLs needed for this process cannot be hosted on GitHub due to file size limitations. You can download them [here](https://github.com/bmaltais/python-library/raw/main/cudnn_windows.zip) to boost sample generation speed (almost 50% on a 4090 GPU). After downloading the ZIP file, follow the installation steps below:

1. Unzip the downloaded file and place the `cudnn_windows` folder in the root directory of the `kohya_ss` repository.

2. Run .\setup.bat and select the option to install cudann.

### Linux and macOS

#### Linux Pre-requirements

To install the necessary dependencies on a Linux system, ensure that you fulfill the following requirements:

- Ensure that `venv` support is pre-installed. You can install it on Ubuntu 22.04 using the command:
  ```
  apt install python3.10-venv
  ```

- Install the cudaNN drivers by following the instructions provided in [this link](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64).

- Make sure you have Python version 3.10.6 or higher (but lower than 3.11.0) installed on your system.

- If you are using WSL2, set the `LD_LIBRARY_PATH` environment variable by executing the following command:
  ```
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib/
  ```

#### Setup

To set up the project on Linux or macOS, perform the following steps:

1. Open a terminal and navigate to the desired installation directory.

2. Clone the repository by running the following command:
   ```
   git clone https://github.com/bmaltais/kohya_ss.git
   ```

3. Change into the `kohya_ss` directory:
   ```
   cd kohya_ss
   ```

4. If you encounter permission issues, make the `setup.sh` script executable by running the following command:
   ```
   chmod +x ./setup.sh
   ```

5. Run the setup script by executing the following command:
   ```
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
   ```
   cd /workspace
   git clone https://github.com/bmaltais/kohya_ss.git
   ```

4. Run the setup script:
   ```
   cd kohya_ss
   ./setup-runpod.sh
   ```

5. Run the gui with:
   ```
   ./gui.sh --share --headless
   ```

   or with this if you expose 7860 directly via the runpod configuration

   ```
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

## Change History

* 2023/07/11 (v21.8.2)
  - Let Tensorboard works in docker #1137
  - Fix for accelerate issue
  - Add SDXL TI training support
  - Rework gui for common layout
  - More LoRA tools to class
  - Add no_half_vae option to TI