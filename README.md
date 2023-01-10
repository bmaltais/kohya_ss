# Kohya's GUI

This repository repository is providing a Gradio GUI for kohya's Stable Diffusion trainers found here: https://github.com/kohya-ss/sd-scripts. The GUI allow you to set the training parameters and generate and run the required CLI command to train the model.

## Required Dependencies

Python 3.10.6+ and Git:

- Python 3.10.6+: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Installation

Open a regular Powershell terminal and type the following inside:

```powershell
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss

python -m venv --system-site-packages venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config

```

### Optional: CUDNN 8.6

This step is optional but can improve the learning speed for NVidia 4090 owners...

Due to the filesize I can't host the DLLs needed for CUDNN 8.6 on Github, I strongly advise you download them for a speed boost in sample generation (almost 50% on 4090) you can download them from here: https://b1.thefileditch.ch/mwxKTEtelILoIbMbruuM.zip

To install simply unzip the directory and place the cudnn_windows folder in the root of the kohya_diffusers_fine_tuning repo.

Run the following command to install:

```
.\venv\Scripts\activate
python .\tools\cudann_1.8_install.py
```

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd kohya_ss
git pull
.\venv\Scripts\activate
pip install --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

## Launching the GUI

To run the GUI you simply use this command:

```
gui.ps1
```

## Dreambooth

You can find the dreambooth solution spercific [Dreambooth README](train_db_README.md)

## Finetune

You can find the finetune solution spercific [Finetune README](fine_tune_README.md)

## Train Network

You can find the train network solution spercific [Train network README](train_network_README.md)

## LoRA

Training a LoRA currently use the `train_network.py` python code. You can create LoRA network by using the all-in-one `gui.cmd` or by running the dedicated LoRA training GUI with:

```
.\venv\Scripts\activate
python lora_gui.py
```

Once you have created the LoRA network you can generate images via auto1111 by installing the extension found here: https://github.com/kohya-ss/sd-webui-additional-networks

## Change history

* 2023/01/11 (v20.2.0):
    - Add support for max token lenght
* 2023/01/10 (v20.1.1):
    - Fix issue with LoRA config loading
* 2023/01/10 (v20.1):
    - Add support for `--output_name` to trainers
    - Refactor code for easier maintenance
* 2023/01/10 (v20.0):
    - Update code base to match latest kohys_ss code upgrade in https://github.com/kohya-ss/sd-scripts
* 2023/01/09 (v19.4.3):
    - Add vae support to dreambooth GUI
    - Add gradient_checkpointing, gradient_accumulation_steps, mem_eff_attn, shuffle_caption to finetune GUI
    - Add gradient_accumulation_steps, mem_eff_attn to dreambooth lora gui
* 2023/01/08 (v19.4.2):
    - Add find/replace option to Basic Caption utility
    - Add resume training and save_state option to finetune UI
* 2023/01/06 (v19.4.1):
    - Emergency fix for new version of gradio causing issues with drop down menus. Please run `pip install -U -r requirements.txt` to fix the issue after pulling this repo.
* 2023/01/06 (v19.4):
    - Add new Utility to Extract a LoRA from a finetuned model
* 2023/01/06 (v19.3.1):
    - Emergency fix for dreambooth_ui no longer working, sorry
    - Add LoRA network merge too GUI. Run `pip install -U -r requirements.txt` after pulling this new release.
* 2023/01/05 (v19.3):
    - Add support for `--clip_skip` option
    - Add missing `detect_face_rotate.py` to tools folder
    - Add `gui.cmd` for easy start of GUI
* 2023/01/02 (v19.2) update:
    - Finetune, add xformers, 8bit adam, min bucket, max bucket, batch size and flip augmentation support for dataset preparation
    - Finetune, add "Dataset preparation" tab to group task specific options
* 2023/01/01 (v19.2) update:
    - add support for color and flip augmentation to "Dreambooth LoRA"
* 2023/01/01 (v19.1) update:
    - merge kohys_ss upstream code  updates
    - rework Dreambooth LoRA GUI
    - fix bug where LoRA network weights were not loaded to properly resume training
* 2022/12/30 (v19) update:
    - support for LoRA network training in kohya_gui.py.
* 2022/12/23 (v18.8) update:
    - Fix for conversion tool issue when the source was an sd1.x diffuser model
    - Other minor code and GUI fix
* 2022/12/22 (v18.7) update:
    - Merge dreambooth and finetune is a common GUI
    - General bug fixes and code improvements
* 2022/12/21 (v18.6.1) update:
    - fix issue with dataset balancing when the number of detected images in the folder is 0

* 2022/12/21 (v18.6) update:
    - add optional GUI authentication support via: `python fine_tune.py --username=<name> --password=<password>`