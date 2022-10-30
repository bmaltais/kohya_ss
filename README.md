# HOWTO

This repo provide all the required config to run the Dreambooth version found in this note: https://note.com/kohya_ss/n/nee3ed1649fb6


## Required Dependencies

Python 3.10.6 and Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

## Installation

Open a Powershell terminal and type the following inside:

```powershell
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
python -m venv --system-site-packages venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

pip install --upgrade diffusers
pip install -r requirements.txt
pip install OmegaConf
pip install pytorch_lightning

pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

# Setup bitsandbytes with Adam8bit support for windows: https://note.com/kohya_ss/n/n47f654dc161e
pip install bitsandbytes==0.35.0
cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config:
- 0
- 0
- NO
- NO
- All
- fp16
```

## Folders configuration

Refer to the note to understand how to create the folde structure. In short it should look like:

```
<wathever top folder name>
|- reg_<class>
    |- <repeat count>_<prompt>
|- train_<class>
   |- <repeat count>_<prompt>
```

Example for `sks dog`

```
my_sks_dog_dreambooth
|- reg_dog
    |- 1_sks dog
|- train_dog
    |- 20_sks dog
```

## Execution

Edit and paste the following in a Powershell terminal:

```powershell
accelerate launch --num_cpu_threads_per_process 6 train_db_fixed_v6.py `
    --pretrained_model_name_or_path="d:\models\v1-5-pruned.ckpt" `
    --train_data_dir="D:\dreambooth\train_bernard\train_man" `
    --train_data_dir="D:\dreambooth\train_bernard\reg_man" `
    --output_dir="D:\dreambooth\train_bernard" `
    --prior_loss_weight=1.0 `
    --resolution=512 `
    --train_batch_size=1 `
    --learning_rate=1e-6 `
    --max_train_steps=2100 `
    --use_8bit_adam `
    --xformers `
    --mixed_precision="fp16" `
    --cache_latents `
    --gradient_checkpointing `
    --save_every_n_epochs=1 
