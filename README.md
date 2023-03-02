# Kohya's GUI

This repository provides a Windows-focused Gradio GUI for [Kohya's Stable Diffusion trainers](https://github.com/kohya-ss/sd-scripts). The GUI allows you to set the training parameters and generate and run the required CLI commands to train the model.

If you run on Linux and would like to use the GUI, there is now a port of it as a docker container. You can find the project [here](https://github.com/P2Enjoy/kohya_ss-docker).

### Table of Contents

- [Tutorials](https://github.com/jonathanzhang53/kohya_ss#tutorials)
- [Required Dependencies](https://github.com/jonathanzhang53/kohya_ss#required-dependencies)
- [Installation](https://github.com/jonathanzhang53/kohya_ss#installation)
    - [CUDNN 8.6](https://github.com/jonathanzhang53/kohya_ss#optional-cudnn-86)
- [Upgrading](https://github.com/jonathanzhang53/kohya_ss#upgrading)
- [Launching the GUI](https://github.com/jonathanzhang53/kohya_ss#launching-the-gui)
- [Dreambooth](https://github.com/jonathanzhang53/kohya_ss#dreambooth)
- [Finetune](https://github.com/jonathanzhang53/kohya_ss#finetune)
- [Train Network](https://github.com/jonathanzhang53/kohya_ss#train-network)
- [LoRA](https://github.com/jonathanzhang53/kohya_ss#lora)
- [Troubleshooting](https://github.com/jonathanzhang53/kohya_ss#troubleshooting)
  - [Page File Limit](https://github.com/jonathanzhang53/kohya_ss#page-file-limit)
  - [No module called tkinter](https://github.com/jonathanzhang53/kohya_ss#no-module-called-tkinter)
  - [FileNotFoundError](https://github.com/jonathanzhang53/kohya_ss#filenotfounderror)
- [Change History](https://github.com/jonathanzhang53/kohya_ss#change-history)

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

## Upgrading

When a new release comes out, you can upgrade your repo with the following commands in the root directory:

```powershell
git pull

.\venv\Scripts\activate

pip install --use-pep517 --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

## Launching the GUI

To run the GUI, simply use this command:

```
.\gui.ps1
```

or you can also do:

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

* 2023/03/02 (v21.1.0):
    - Add LoCon support (https://github.com/KohakuBlueleaf/LoCon.git) to the Dreambooth LoRA tab. This will allow to create a new type of LoRA that include conv layers as part of the LoRA... hence the name LoCon. LoCon will work with the native Auto1111 implementation of LoRA. If you want to use it with the Kohya_ss additionalNetwork you will need to install this other extension... until Kohya_ss support it nativelly: https://github.com/KohakuBlueleaf/a1111-sd-webui-locon
* 2023/03/01 (v21.0.1):
    - Add warning to tensorboard start if the log information is missing
    - Fix issue with 8bitadam on older config file load
* 2023/02/27 (v21.0.0):
    - Add tensorboard start and stop support to the GUI
* 2023/02/26 (v20.8.2):
    - Fix issue https://github.com/bmaltais/kohya_ss/issues/231
    - Change default for seed to random
    - Add support for --share argument to `kohya_gui.py` and `gui.ps1`
    - Implement 8bit adam login to help with the legacy `Use 8bit adam` checkbox that is now superceided by the `Optimizer` dropdown selection. This field will be eventually removed. Kept for now for backward compatibility.
* 2023/02/23 (v20.8.1):
    - Fix instability training issue in `train_network.py`.
        - `fp16` training is probably not affected by this issue.
        - Training with `float` for SD2.x models will work now. Also training with bf16 might be improved.
        - This issue seems to have occurred in [PR#190](https://github.com/kohya-ss/sd-scripts/pull/190).
    - Add some metadata to LoRA model. Thanks to space-nuko!
    - Raise an error if optimizer options conflict (e.g. `--optimizer_type` and `--use_8bit_adam`.)
    - Support ControlNet in `gen_img_diffusers.py` (no documentation yet.)
* 2023/02/22 (v20.8.0):
    - Add gui support for optimizers: `AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdaptation, AdaFactor`
    - Add gui support for `--noise_offset`
    - Refactor optmizer options. Thanks to mgz-dev!
        - Add `--optimizer_type` option for each training script. Please see help. Japanese documentation is [here](https://github-com.translate.goog/kohya-ss/sd-scripts/blob/main/train_network_README-ja.md?_x_tr_sl=fr&_x_tr_tl=en&_x_tr_hl=en-US&_x_tr_pto=wapp#%E3%82%AA%E3%83%97%E3%83%86%E3%82%A3%E3%83%9E%E3%82%A4%E3%82%B6%E3%81%AE%E6%8C%87%E5%AE%9A%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6).
        - `--use_8bit_adam` and `--use_lion_optimizer` options also work and will override the options above for backward compatibility.
    - Add SGDNesterov and its 8bit.
    - Add [D-Adaptation](https://github.com/facebookresearch/dadaptation) optimizer. Thanks to BootsofLagrangian and all! 
        - Please install D-Adaptation optimizer with `pip install dadaptation` (it is not in requirements.txt currently.)
        - Please see https://github.com/kohya-ss/sd-scripts/issues/181 for details.
    - Add AdaFactor optimizer. Thanks to Toshiaki!
    - Extra lr scheduler settings (num_cycles etc.) are working in training scripts other than `train_network.py`.
    - Add `--max_grad_norm` option for each training script for gradient clipping. `0.0` disables clipping. 
    - Symbolic link can be loaded in each training script. Thanks to TkskKurumi!
