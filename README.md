# HOWTO

This repo provide all the required config to run the Dreambooth version found in this note: https://note.com/kohya_ss/n/nee3ed1649fb6
The setup of bitsandbytes with Adam8bit support for windows: https://note.com/kohya_ss/n/n47f654dc161e

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
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config

```

Answers to accelerate config:

```txt
- 0
- 0
- NO
- NO
- All
- fp16
```

### Optional: CUDNN 8.6

This step is optional but can improve the learning speed for NVidia 4090 owners...

Due to the filesize I can't host the DLLs needed for CUDNN 8.6 on Github, I strongly advise you download them for a speed boost in sample generation (almost 50% on 4090) you can download them from here: https://b1.thefileditch.ch/mwxKTEtelILoIbMbruuM.zip

To install simply unzip the directory and place the cudnn_windows folder in the root of the kohya_diffusers_fine_tuning repo.

Run the following command to install:

```
python cudann_1.8_install.py
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

## GUI

There is now support for GUI based training using gradio. You can start the GUI interface by running:

```powershell
python .\dreambooth_gui.py
```

## Quickstart screencast

You can find a screen cast on how to use the GUI at the following location:

https://youtu.be/RlvqEKj03WI

## Folders configuration

Refer to the note to understand how to create the folde structure. In short it should look like:

```
<arbitrary folder name>
|- <arbitrary class folder name>
    |- <repeat count>_<class>
|- <arbitrary training folder name>
   |- <repeat count>_<token> <class>
```

Example for `asd dog` where `asd` is the token word and `dog` is the class. In this example the regularization `dog` class images contained in the folder will be repeated only 1 time and the `asd dog` images will be repeated 20 times:

```
my_asd_dog_dreambooth
|- reg_dog
    |- 1_dog
       `- reg_image_1.png
       `- reg_image_2.png
       ...
       `- reg_image_256.png
|- train_dog
    |- 20_asd dog
       `- dog1.png
       ...
       `- dog8.png
```

## Support

Drop by the discord server for support: https://discord.com/channels/1041518562487058594/1041518563242020906

## Change history

* 12/17 (v17.1) update:
    - Adding GUI for kohya_ss called dreambooth_gui.py
    - removing support for `--finetuning` as there is now a dedicated python repo for that. `--fine-tuning` is still there behind the scene until kohya_ss remove it in a future code release.
    - removing cli examples as I will now focus on the GUI for training. People who prefer cli based training can still do that.
* 12/13 (v17) update:
    - Added support for learning to fp16 gradient (experimental function). SD1.x can be trained with 8GB of VRAM. Specify full_fp16 options.
* 12/06 (v16) update:
    - Added support for Diffusers 0.10.2 (use code in Diffusers to learn v-parameterization).
    - Diffusers also supports safetensors.
    - Added support for accelerate 0.15.0.
* 12/05 (v15) update:
    - The script has been divided into two parts
    - Support for SafeTensors format has been added. Install SafeTensors with `pip install safetensors`. The script will automatically detect the format based on the file extension when loading. Use the `--use_safetensors` option if you want to save the model as safetensor.
    - The vae option has been added to load a VAE model separately.
    - The log_prefix option has been added to allow adding a custom string to the log directory name before the date and time.
* 11/30 (v13) update:
    - fix training text encoder at specified step (`--stop_text_encoder_training=<step #>`) that was causing both Unet and text encoder training to stop completely at the specified step rather than continue without text encoding training.
* 11/29 (v12) update:
    - stop training text encoder at specified step (`--stop_text_encoder_training=<step #>`)
    - tqdm smoothing
    - updated fine tuning script to support SD2.0 768/v
* 11/27 (v11) update:
    - DiffUsers 0.9.0 is required. Update with `pip install --upgrade -r requirements.txt` in the virtual environment.
    - The way captions are handled in DreamBooth has changed. When a caption file existed, the file's caption was added to the folder caption until v10, but from v11 it is only the file's caption. Please be careful.
    - Fixed a bug where prior_loss_weight was applied to learning images. Sorry for the inconvenience.
    - Compatible with Stable Diffusion v2.0. Add the `--v2` option. If you are using `768-v-ema.ckpt` or `stable-diffusion-2` instead of `stable-diffusion-v2-base`, add `--v_parameterization` as well. Learn more about other options.
    - Added options related to the learning rate scheduler.
    - You can download and use DiffUsers models directly from Hugging Face. In addition, DiffUsers models can be saved during training.
* 11/21 (v10):
    - Added minimum/maximum resolution specification when using Aspect Ratio Bucketing (min_bucket_reso/max_bucket_reso option).
    - Added extension specification for caption files (caption_extention).
    - Added support for images with .webp extension.
    - Added a function that allows captions to learning images and regularized images.
* 11/18 (v9):
    - Added support for Aspect Ratio Bucketing (enable_bucket option). (--enable_bucket)
    - Added support for selecting data format (fp16/bf16/float) when saving checkpoint (--save_precision)
    - Added support for saving learning state (--save_state, --resume)
    - Added support for logging (--logging_dir)
* 11/14 (diffusers_fine_tuning v2):
    - script name is now fine_tune.py.
    - Added option to learn Text Encoder --train_text_encoder.
    - The data format of checkpoint at the time of saving can be specified with the --save_precision option. You can choose float, fp16, and bf16.
    - Added a --save_state option to save the learning state (optimizer, etc.) in the middle. It can be resumed with the --resume option.
* 11/9 (v8): supports Diffusers 0.7.2. To upgrade diffusers run `pip install --upgrade diffusers[torch]`
* 11/7 (v7): Text Encoder supports checkpoint files in different storage formats (it is converted at the time of import, so export will be in normal format). Changed the average value of EPOCH loss to output to the screen. Added a function to save epoch and global step in checkpoint in SD format (add values if there is existing data). The reg_data_dir option is enabled during fine tuning (fine tuning while mixing regularized images). Added dataset_repeats option that is valid for fine tuning (specified when the number of teacher images is small and the epoch is extremely short).