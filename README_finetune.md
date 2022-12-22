# Kohya_ss Finetune

This python utility provide code to run the diffusers fine tuning version found in this note: https://note.com/kohya_ss/n/nbf7ce8d80f29

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
git clone https://github.com/bmaltais/kohya_diffusers_fine_tuning.git
cd kohya_diffusers_fine_tuning

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
python .\tools\cudann_1.8_install.py
```

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```
.\upgrade.bat
```

or you can do it manually with

```powershell
cd kohya_ss
git pull
.\venv\Scripts\activate
pip install --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

## Folders configuration

Simply put all the images you will want to train on in a single directory. It does not matter what size or aspect ratio they have. It is your choice.

## Captions

Each file need to be accompanied by a caption file describing what the image is about. For example, if you want to train on cute dog pictures you can put `cute dog` as the caption in every file. You can use the `tools\caption.ps1` sample code to help out with that:

```powershell
$folder = "sample"
$file_pattern="*.*"
$caption_text="cute dog"

$files = Get-ChildItem "$folder\$file_pattern" -Include *.png, *.jpg, *.webp -File
foreach ($file in $files) {
    if (-not(Test-Path -Path $folder\"$($file.BaseName).txt" -PathType Leaf)) {
        New-Item -ItemType file -Path $folder -Name "$($file.BaseName).txt" -Value $caption_text
    }
}

You can also use the `Captioning` tool found under the `Utilities` tab in the GUI.
```

## GUI

There is now support for GUI based training using gradio. You can start the complete kohya training GUI interface by running:

```powershell
.\kohya.cmd
```

and select the Finetune tab.

Alternativelly you can use the Finetune focus GUI with

```powershell
.\finetune.cmd
```

## CLI

You can find various examples of how to leverage the `fine_tune.py` in this folder: https://github.com/bmaltais/kohya_ss/tree/master/examples

## Support

Drop by the discord server for support: https://discord.com/channels/1041518562487058594/1041518563242020906

## Change history

* 12/20 (v9.6) update:
    - fix issue with config file save and opening
* 12/19 (v9.5) update:
    - Fix file/folder dialog opening behind the browser window
    - Update GUI layout to be more logical
* 12/18 (v9.4) update:
    - Add WD14 tagging to utilities
* 12/18 (v9.3) update:
    - Add logging option
* 12/18 (v9.2) update:
    - Add BLIP Captioning utility
* 12/18 (v9.1) update:
    - Add Stable Diffusion model conversion utility. Make sure to run `pip upgrade -U -r requirements.txt` after updating to this release as this introduce new pip requirements.
* 12/17 (v9) update:
    - Save model as option added to fine_tune.py
    - Save model as option added to GUI
    - Retirement of cli based documentation. Will focus attention to GUI based training
* 12/13 (v8):
    - WD14Tagger now works on its own.
    - Added support for learning to fp16 up to the gradient. Go to "Building the environment and preparing scripts for Diffusers for more info".
* 12/10 (v7):
    - We have added support for Diffusers 0.10.2.
    - In addition, we have made other fixes.
    - For more information, please see the section on "Building the environment and preparing scripts for Diffusers" in our documentation.
* 12/6 (v6): We have responded to reports that some models experience an error when saving in SafeTensors format.
* 12/5 (v5):
    - .safetensors format is now supported. Install SafeTensors as "pip install safetensors". When loading, it is automatically determined by extension. Specify use_safetensors options when saving.
    - Added an option to add any string before the date and time log directory name log_prefix.
    - Cleaning scripts now work without either captions or tags.
* 11/29 (v4):
    - DiffUsers 0.9.0 is required. Update as "pip install -U diffusers[torch]==0.9.0" in the virtual environment, and update the dependent libraries as "pip install --upgrade -r requirements.txt" if other errors occur.
    - Compatible with Stable Diffusion v2.0. Add the --v2 option when training (and pre-fetching latents). If you are using 768-v-ema.ckpt or stable-diffusion-2 instead of stable-diffusion-v2-base, add --v_parameterization as well when learning. Learn more about other options.
    - The minimum resolution and maximum resolution of the bucket can be specified when pre-fetching latents.
    - Corrected the calculation formula for loss (fixed that it was increasing according to the batch size).
    - Added options related to the learning rate scheduler.
    - So that you can download and learn DiffUsers models directly from Hugging Face. In addition, DiffUsers models can be saved during training.
    - Available even if the clean_captions_and_tags.py is only a caption or a tag.
    - Other minor fixes such as changing the arguments of the noise scheduler during training.
* 11/23 (v3):
    - Added WD14Tagger tagging script.
    - A log output function has been added to the fine_tune.py. Also, fixed the double shuffling of data.
    - Fixed misspelling of options for each script (caption_extention→caption_extension will work for the time being, even if it remains outdated).
