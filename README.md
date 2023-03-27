# Kohya's GUI

This repository provides a Windows-focused Gradio GUI for [Kohya's Stable Diffusion trainers](https://github.com/kohya-ss/sd-scripts). The GUI allows you to set the training parameters and generate and run the required CLI commands to train the model.

If you run on Linux and would like to use the GUI, there is now a port of it as a docker container. You can find the project [here](https://github.com/P2Enjoy/kohya_ss-docker).

### Table of Contents

- [Tutorials](https://github.com/bmaltais/kohya_ss#tutorials)
- [Required Dependencies](https://github.com/bmaltais/kohya_ss#required-dependencies)
- [Installation](https://github.com/bmaltais/kohya_ss#installation)
    - [CUDNN 8.6](https://github.com/bmaltais/kohya_ss#optional-cudnn-86)
- [Upgrading](https://github.com/bmaltais/kohya_ss#upgrading)
- [Launching the GUI](https://github.com/bmaltais/kohya_ss#launching-the-gui)
- [Dreambooth](https://github.com/bmaltais/kohya_ss#dreambooth)
- [Finetune](https://github.com/bmaltais/kohya_ss#finetune)
- [Train Network](https://github.com/bmaltais/kohya_ss#train-network)
- [LoRA](https://github.com/bmaltais/kohya_ss#lora)
- [Troubleshooting](https://github.com/bmaltais/kohya_ss#troubleshooting)
  - [Page File Limit](https://github.com/bmaltais/kohya_ss#page-file-limit)
  - [No module called tkinter](https://github.com/bmaltais/kohya_ss#no-module-called-tkinter)
  - [FileNotFoundError](https://github.com/bmaltais/kohya_ss#filenotfounderror)
- [Change History](https://github.com/bmaltais/kohya_ss#change-history)

## Tutorials

[How to Create a LoRA Part 1: Dataset Preparation](https://www.youtube.com/watch?v=N4_-fB62Hwk):

[![LoRA Part 1 Tutorial](https://img.youtube.com/vi/N4_-fB62Hwk/0.jpg)](https://www.youtube.com/watch?v=N4_-fB62Hwk)

[How to Create a LoRA Part 2: Training the Model](https://www.youtube.com/watch?v=k5imq01uvUY):

[![LoRA Part 2 Tutorial](https://img.youtube.com/vi/k5imq01uvUY/0.jpg)](https://www.youtube.com/watch?v=k5imq01uvUY)

## Required Dependencies

- Install [Python 3.10](https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe) 
  - make sure to tick the box to add Python to the 'PATH' environment variable
- Install [Git](https://git-scm.com/download/win)
- Install [Visual Studio 2015, 2017, 2019, and 2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

## Installation

### Runpod
Follow the instructions found in this discussion: https://github.com/bmaltais/kohya_ss/discussions/379

### MacOS
In the terminal, run

```
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
bash macos_setup.sh
```

During the accelerate config screen after running the script answer "This machine", "None", "No" for the remaining questions.

### Ubuntu
In the terminal, run

```
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
bash ubuntu_setup.sh
```

then configure accelerate with the same answers as in the Windows instructions when prompted.

### Windows

Give unrestricted script access to powershell so venv can work:

- Run PowerShell as an administrator
- Run `Set-ExecutionPolicy Unrestricted` and answer 'A'
- Close PowerShell

Open a regular user Powershell terminal and run the following commands:

```powershell
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss

python -m venv venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --use-pep517 --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config
```

### Optional: CUDNN 8.6

This step is optional but can improve the learning speed for NVIDIA 30X0/40X0 owners. It allows for larger training batch size and faster training speed.

Due to the file size, I can't host the DLLs needed for CUDNN 8.6 on Github. I strongly advise you download them for a speed boost in sample generation (almost 50% on 4090 GPU) you can download them [here](https://b1.thefileditch.ch/mwxKTEtelILoIbMbruuM.zip).

To install, simply unzip the directory and place the `cudnn_windows` folder in the root of the this repo.

Run the following commands to install:

```
.\venv\Scripts\activate

python .\tools\cudann_1.8_install.py
```

## Upgrading MacOS

When a new release comes out, you can upgrade your repo with the following commands in the root directory:

```bash
upgrade_macos.sh
```

Once the commands have completed successfully you should be ready to use the new version. MacOS support is not tested and has been mostly taken from https://gist.github.com/jstayco/9f5733f05b9dc29de95c4056a023d645

## Upgrading Windows

When a new release comes out, you can upgrade your repo with the following commands in the root directory:

```powershell
git pull

.\venv\Scripts\activate

pip install --use-pep517 --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

## Launching the GUI using gui.bat or gui.ps1

The script can be run with several optional command line arguments:

--listen: the IP address to listen on for connections to Gradio.
--username: a username for authentication.
--password: a password for authentication.
--server_port: the port to run the server listener on.
--inbrowser: opens the Gradio UI in a web browser.
--share: shares the Gradio UI.

These command line arguments can be passed to the UI function as keyword arguments. To launch the Gradio UI, run the script in a terminal with the desired command line arguments, for example:

`gui.ps1 --listen 127.0.0.1 --server_port 7860 --inbrowser --share`

or

`gui.bat --listen 127.0.0.1 --server_port 7860 --inbrowser --share`

## Launching the GUI using kohya_gui.py

To run the GUI, simply use this command:

```
.\venv\Scripts\activate

python.exe .\kohya_gui.py
```

## Dreambooth

You can find the dreambooth solution specific here: [Dreambooth README](train_db_README.md)

## Finetune

You can find the finetune solution specific here: [Finetune README](fine_tune_README.md)

## Train Network

You can find the train network solution specific here: [Train network README](train_network_README.md)

## LoRA

Training a LoRA currently uses the `train_network.py` code. You can create a LoRA network by using the all-in-one `gui.cmd` or by running the dedicated LoRA training GUI with:

```
.\venv\Scripts\activate

python lora_gui.py
```

Once you have created the LoRA network, you can generate images via auto1111 by installing [this extension](https://github.com/kohya-ss/sd-webui-additional-networks).

## Troubleshooting

### Page File Limit

- X error relating to `page file`: Increase the page file size limit in Windows.

### No module called tkinter

- Re-install [Python 3.10](https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe) on your system.

### FileNotFoundError

This is usually related to an installation issue. Make sure you do not have any python modules installed locally that could conflict with the ones installed in the venv:

1. Open a new powershell terminal and make sure no venv is active.
2.  Run the following commands:

```
pip freeze > uninstall.txt
pip uninstall -r uninstall.txt
```

This will store your a backup file with your current locally installed pip packages and then uninstall them. Then, redo the installation instructions within the kohya_ss venv.

## Change History

* 2023/03/26 (v21.3.6)
    - Fixed the error while images are ended with capital image extensions. Thanks to @kvzn. https://github.com/bmaltais/kohya_ss/pull/454
* 2023/03/26 (v21.3.5)
    - Fix for https://github.com/bmaltais/kohya_ss/issues/230
    - Added detection for Google Colab to not bring up the GUI file/folder window on the platform. Instead it will only use the file/folder path provided in the input field.
* 2023/03/25 (v21.3.4)
    - Added untested support for MacOS base on this gist: https://gist.github.com/jstayco/9f5733f05b9dc29de95c4056a023d645

    Let me know how this work. From the look of it it appear to be well tought out. I modified a few things to make it fit better with the rest of the code in the repo.
    - Fix for issue https://github.com/bmaltais/kohya_ss/issues/433 by implementing default of 0.
    - Removed non applicable save_model_as choices for LoRA and TI.
* 2023/03/24 (v21.3.3)
    - Add support for custom user gui files. THey will be created at installation time or when upgrading is missing. You will see two files in the root of the folder. One named `gui-user.bat` and the other `gui-user.ps1`. Edit the file based on your prefered terminal. Simply add the parameters you want to pass the gui in there and execute it to start the gui with them. Enjoy!
* 2023/03/23 (v21.3.2)
    - Fix issue reported: https://github.com/bmaltais/kohya_ss/issues/439
* 2023/03/23 (v21.3.1)
    - Merge PR to fix refactor naming issue for basic captions. Thank @zrma
* 2023/03/22 (v21.3.0)
    - Add a function to load training config with `.toml` to each training script. Thanks to Linaqruf for this great contribution!
        - Specify `.toml` file with `--config_file`. `.toml` file has `key=value` entries. Keys are same as command line options. See [#241](https://github.com/kohya-ss/sd-scripts/pull/241) for details.
        - All sub-sections are combined to a single dictionary (the section names are ignored.)
        - Omitted arguments are the default values for command line arguments.
        - Command line args override the arguments in `.toml`.
        - With `--output_config` option, you can output current command line options  to the `.toml` specified with`--config_file`. Please use as a template.
    - Add `--lr_scheduler_type` and `--lr_scheduler_args` arguments for custom LR scheduler to each training script. Thanks to Isotr0py! [#271](https://github.com/kohya-ss/sd-scripts/pull/271)
        - Same as the optimizer.
    - Add sample image generation with weight and no length limit. Thanks to mio2333! [#288](https://github.com/kohya-ss/sd-scripts/pull/288)
        - `( )`, `(xxxx:1.2)` and `[ ]` can be used.
    - Fix exception on training model in diffusers format with `train_network.py` Thanks to orenwang! [#290](https://github.com/kohya-ss/sd-scripts/pull/290)
    - Add warning if you are about to overwrite an existing model: https://github.com/bmaltais/kohya_ss/issues/404
    - Add `--vae_batch_size` for faster latents caching to each training script. This  batches VAE calls.
        - Please start with`2` or `4` depending on the size of VRAM.
    - Fix a number of training steps with `--gradient_accumulation_steps` and `--max_train_epochs`. Thanks to tsukimiya!
    - Extract parser setup to external scripts. Thanks to robertsmieja!
    - Fix an issue without `.npz` and with `--full_path` in training.
    - Support extensions with upper cases for images for not Windows environment.
    - Fix `resize_lora.py` to work with LoRA with dynamic rank (including `conv_dim != network_dim`). Thanks to toshiaki!
    - Fix issue: https://github.com/bmaltais/kohya_ss/issues/406
    - Add device support to LoRA extract.
