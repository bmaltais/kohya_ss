# Kohya's GUI

This repository mostly provides a Windows-focused Gradio GUI for [Kohya's Stable Diffusion trainers](https://github.com/kohya-ss/sd-scripts)... but support for Linux OS is also provided through community contributions. Macos is not great at the moment.

The GUI allows you to set the training parameters and generate and run the required CLI commands to train the model.

## Table of Contents

- [Kohya's GUI](#kohyas-gui)
  - [Table of Contents](#table-of-contents)
  - [Tutorials](#tutorials)
    - [About SDXL training](#about-sdxl-training)
      - [Tips for SDXL training](#tips-for-sdxl-training)
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
  - [Change History](#change-history)


## Tutorials

[How to Create a LoRA Part 1: Dataset Preparation](https://www.youtube.com/watch?v=N4_-fB62Hwk):

[![LoRA Part 1 Tutorial](https://img.youtube.com/vi/N4_-fB62Hwk/0.jpg)](https://www.youtube.com/watch?v=N4_-fB62Hwk)

[How to Create a LoRA Part 2: Training the Model](https://www.youtube.com/watch?v=k5imq01uvUY):

[![LoRA Part 2 Tutorial](https://img.youtube.com/vi/k5imq01uvUY/0.jpg)](https://www.youtube.com/watch?v=k5imq01uvUY)

Newer Tutorial: [Generate Studio Quality Realistic Photos By Kohya LoRA Stable Diffusion Training](https://www.youtube.com/watch?v=TpuDOsuKIBo):
The scripts are tested with PyTorch 1.12.1 and 2.0.1, Diffusers 0.17.1.

[![Newer Tutorial: Generate Studio Quality Realistic Photos By Kohya LoRA Stable Diffusion Training](https://user-images.githubusercontent.com/19240467/235306147-85dd8126-f397-406b-83f2-368927fa0281.png)](https://www.youtube.com/watch?v=TpuDOsuKIBo)

Newer Tutorial: [How To Install And Use Kohya LoRA GUI / Web UI on RunPod IO](https://www.youtube.com/watch?v=3uzCNrQao3o):

[![How To Install And Use Kohya LoRA GUI / Web UI on RunPod IO With Stable Diffusion & Automatic1111](https://github-production-user-asset-6210df.s3.amazonaws.com/19240467/238678226-0c9c3f7d-c308-4793-b790-999fdc271372.png)](https://www.youtube.com/watch?v=3uzCNrQao3o)

First SDXL Tutorial: [First Ever SDXL Training With Kohya LoRA - Stable Diffusion XL Training Will Replace Older Models](https://youtu.be/AY6DMBCIZ3A):

[![First Ever SDXL Training With Kohya LoRA - Stable Diffusion XL Training Will Replace Older Models](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/mG0CvKAzb8o29nr5ye0Br.png)](https://youtu.be/AY6DMBCIZ3A)

### About SDXL training

The feature of SDXL training is now available in sdxl branch as an experimental feature. 

Aug 6, 2023: The feature will be merged into the main branch soon. Following are the changes from the previous version.

- [SAI Model Spec](https://github.com/Stability-AI/ModelSpec) metadata is now supported partially. `hash_sha256` is not supported yet.
  - The main items are set automatically. 
  - You can set title, author, description, license and tags with `--metadata_xxx` options in each training script.
  - Merging scripts also support minimum SAI Model Spec metadata. See the help message for the usage.
  - Metadata editor will be available soon.
- SDXL LoRA has `sdxl_base_v1-0` now  for `ss_base_model_version` metadata item, instead of `v0-9`.

Aug 4, 2023: 

- `bitsandbytes` is now optional. Please install it if you want to use it. The insructions are in the later section.
- `albumentations` is not required anymore.
- An issue for pooled output for Textual Inversion training is fixed.
- `--v_pred_like_loss ratio` option is added. This option adds the loss like v-prediction loss in SDXL training. `0.1` means that the loss is added 10% of the v-prediction loss. The default value is None (disabled).
  - In v-prediction, the loss is higher in the early timesteps (near the noise). This option can be used to increase the loss in the early timesteps.
- Arbitrary options can be used for Diffusers' schedulers. For example `--lr_scheduler_args "lr_end=1e-8"`.
- `sdxl_gen_imgs.py` supports batch size > 1.
- Fix ControlNet to work with attention couple and reginal LoRA in `gen_img_diffusers.py`.

Summary of the feature:

- `tools/cache_latents.py` is added. This script can be used to cache the latents to disk in advance. 
  - The options are almost the same as `sdxl_train.py'. See the help message for the usage.
  - Please launch the script as follows:
    `accelerate launch  --num_cpu_threads_per_process 1 tools/cache_latents.py ...`
  - This script should work with multi-GPU, but it is not tested in my environment.

- `tools/cache_text_encoder_outputs.py` is added. This script can be used to cache the text encoder outputs to disk in advance. 
  - The options are almost the same as `cache_latents.py' and `sdxl_train.py'. See the help message for the usage.

- `sdxl_train.py` is a script for SDXL fine-tuning. The usage is almost the same as `fine_tune.py`, but it also supports DreamBooth dataset.
  - `--full_bf16` option is added. Thanks to KohakuBlueleaf!
    - This option enables the full bfloat16 training (includes gradients). This option is useful to reduce the GPU memory usage. 
    - However, bitsandbytes==0.35 doesn't seem to support this. Please use a newer version of bitsandbytes or another optimizer.
    - I cannot find bitsandbytes>0.35.0 that works correctly on Windows.
    - In addition, the full bfloat16 training might be unstable. Please use it at your own risk.
- `prepare_buckets_latents.py` now supports SDXL fine-tuning.
- `sdxl_train_network.py` is a script for LoRA training for SDXL. The usage is almost the same as `train_network.py`.
- Both scripts has following additional options:
  - `--cache_text_encoder_outputs` and `--cache_text_encoder_outputs_to_disk`: Cache the outputs of the text encoders. This option is useful to reduce the GPU memory usage. This option cannot be used with options for shuffling or dropping the captions.
  - `--no_half_vae`: Disable the half-precision (mixed-precision) VAE. VAE for SDXL seems to produce NaNs in some cases. This option is useful to avoid the NaNs.
- The image generation during training is now available. `--no_half_vae` option also works to avoid black images.

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
  - Use lower dim (-8 for 8GB GPU).
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

## 🦒 Colab

🚦 WIP 🚦

This Colab notebook was not created or maintained by me; however, it appears to function effectively. The source can be found at: https://github.com/camenduru/kohya_ss-colab.

I would like to express my gratitude to camendutu for their valuable contribution. If you encounter any issues with the Colab notebook, please report them on their repository.

| Colab                                                                                                                                                                          | Info                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------- |
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

   During the accelerate config step use the default values as proposed during the configuration unless you know your hardware demand otherwise. Tfe amount of VRAM on your GPU does not have an impact on the values used.

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

* 2023/08/05 (v21.8.7)
  - Merge latest sd-scripts updates.
  - Updated layout? This is up for debate... but I think it make things easier to find. Tab instead of endless scrolling...
  - Fix issue with LoRA merge GUI