# HOWTO

This repo provide all the required config to run the Dreambooth version found in this note: https://note.com/kohya_ss/n/nee3ed1649fb6


## Required Dependencies

Python 3.10.6 and Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
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
<arbitrary folder name>
|- <arbitrary class folder name>
    |- <repeat count>_<class>
|- <arbitrary training folder name>
   |- <repeat count>_<token> <class>
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
accelerate launch --num_cpu_threads_per_process 6 train_db_fixed.py `
    --pretrained_model_name_or_path="D:\models\last.ckpt" `
    --train_data_dir="D:\dreambooth\train_bernard\train_man" `
    --reg_data_dir="D:\dreambooth\train_bernard\reg_man" `
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

## Finetuning

If you would rather use model finetuning rather than the dreambooth method you can use a command similat to the following. The advantage of fine tuning is that you do not need to worry about regularization images... but you need to provide captions for every images. The caption will be used to train the model. You can use auto1111 to preprocess your training images and add either BLIP or danbooru captions to them. You then need to edit those to add the name of the model and correct any wrong description.

```
accelerate launch --num_cpu_threads_per_process 6 train_db_fixed-ber.py `
    --pretrained_model_name_or_path="D:\models\alexandrine_teissier_and_bernard_maltais-400-kohya-sd15-v1.ckpt" `
    --train_data_dir="D:\dreambooth\source\alet_et_bernard\landscape-pp" `
    --output_dir="D:\dreambooth\train_alex_and_bernard" `
    --resolution="640,448" `
    --train_batch_size=1 `
    --learning_rate=1e-6 `
    --max_train_steps=550 `
    --use_8bit_adam `
    --xformers `
    --mixed_precision="fp16" `
    --cache_latents `
    --save_every_n_epochs=1 `
    --fine_tuning `
    --enable_bucket `
    --dataset_repeats=200 `
    --seed=23 `
    ---save_precision="fp16"
```

Refer to this url for more details about finetuning: https://note.com/kohya_ss/n/n1269f1e1a54e

## Change history

* 11/7 (v7): Text Encoder supports checkpoint files in different storage formats (it is converted at the time of import, so export will be in normal format). Changed the average value of EPOCH loss to output to the screen. Added a function to save epoch and global step in checkpoint in SD format (add values if there is existing data). The reg_data_dir option is enabled during fine tuning (fine tuning while mixing regularized images). Added dataset_repeats option that is valid for fine tuning (specified when the number of teacher images is small and the epoch is extremely short).
* 11/9 (v8): supports Diffusers 0.7.2. To upgrade diffusers run `pip install --upgrade diffusers[torch]`
* 11/14 (diffusers_fine_tuning v2):
    - script name is now fine_tune.py.
    - Added option to learn Text Encoder --train_text_encoder.
    - The data format of checkpoint at the time of saving can be specified with the --save_precision option. You can choose float, fp16, and bf16.
    - Added a --save_state option to save the learning state (optimizer, etc.) in the middle. It can be resumed with the --resume option.
* 11/18 (v9):
    - Added support for Aspect Ratio Bucketing (enable_bucket option). (--enable_bucket)
    - Added support for selecting data format (fp16/bf16/float) when saving checkpoint (--save_precision)
    - Added support for saving learning state (--save_state, --resume)
    - Added support for logging (--logging_dir)
* 11/21 (v10):
    - Added minimum/maximum resolution specification when using Aspect Ratio Bucketing (min_bucket_reso/max_bucket_reso option).
    - Added extension specification for caption files (caption_extention).
    - Added support for images with .webp extension.
    - Added a function that allows captions to learning images and regularized images.