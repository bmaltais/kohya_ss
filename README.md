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

* 2023/02/09 (v20.7.1)
    - Caption dropout is supported in ``train_db.py``, ``fine_tune.py`` and ``train_network.py``. Thanks to forestsource!
        - ``--caption_dropout_rate`` option specifies the dropout rate for captions (0~1.0, 0.1 means 10% chance for dropout). If dropout occurs, the image is trained with the empty caption. Default is 0 (no dropout).
        - ``--caption_dropout_every_n_epochs`` option specifies how many epochs to drop captions. If ``3`` is specified, in epoch 3, 6, 9 ..., images are trained with all captions empty. Default is None (no dropout).
        - ``--caption_tag_dropout_rate`` option specified the dropout rate for tags (comma separated tokens) (0~1.0, 0.1 means 10% chance for dropout). If dropout occurs, the tag is removed from the caption. If ``--keep_tokens`` option is set, these tokens (tags) are not dropped. Default is 0 (no droupout).
        - The bulk image downsampling script is added. Documentation is [here](https://github.com/kohya-ss/sd-scripts/blob/main/train_network_README-ja.md#%E7%94%BB%E5%83%8F%E3%83%AA%E3%82%B5%E3%82%A4%E3%82%BA%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%97%E3%83%88) (in Jpanaese). Thanks to bmaltais!
        - Typo check is added. Thanks to shirayu!
    - Add option to autolaunch the GUI in a browser and set the server_port. USe either `gui.ps1 --inbrowser --server_port 3456`or `gui.cmd -inbrowser -server_port 3456`
* 2023/02/06 (v20.7.0)
    - `--bucket_reso_steps` and `--bucket_no_upscale` options are added to training scripts (fine tuning, DreamBooth, LoRA and Textual Inversion) and `prepare_buckets_latents.py`.
    - `--bucket_reso_steps` takes the steps for buckets in aspect ratio bucketing. Default is 64, same as before.
        - Any value greater than or equal to 1 can be specified; 64 is highly recommended and a value divisible by 8 is recommended.
        - If less than 64 is specified, padding will occur within U-Net. The result is unknown.
        - If you specify a value that is not divisible by 8, it will be truncated to divisible by 8 inside VAE, because the size of the latent is 1/8 of the image size.
    - If the `--bucket_no_upscale` option is specified, images smaller than the bucket size will be processed without upscaling.
        - Internally, a bucket smaller than the image size is created (for example, if the image is 300x300 and `bucket_reso_steps=64`, the bucket is 256x256). The image will be trimmed.
        - Implementation of [#130](https://github.com/kohya-ss/sd-scripts/issues/130).
        - Images with an area larger than the maximum size specified by `--resolution` are downsampled to the max bucket size.
    - Now the number of data in each batch is limited to the number of actual images (not duplicated). Because a certain bucket may contain smaller number of actual images, so the batch may contain same (duplicated) images.
    - `--random_crop` now also works with buckets enabled.
        - Instead of always cropping the center of the image, the image is shifted left, right, up, and down to be used as the training data. This is expected to train to the edges of the image.
        - Implementation of discussion [#34](https://github.com/kohya-ss/sd-scripts/discussions/34).
* 2023/02/04 (v20.6.1)
    - Add new LoRA resize GUI
    - `--persistent_data_loader_workers` option is added to `fine_tune.py`, `train_db.py` and `train_network.py`. This option may significantly reduce the waiting time between epochs. Thanks to hitomi!
    - `--debug_dataset` option is now working on non-Windows environment. Thanks to tsukimiya!
    - `networks/resize_lora.py` script is added. This can approximate the higher-rank (dim) LoRA model by a lower-rank LoRA model, e.g. 128 to 4. Thanks to mgz-dev!
        - `--help` option shows usage.
        - Currently the metadata is not copied. This will be fixed in the near future.
* 2023/02/03 (v20.6.0)
    - Increase max LoRA rank (dim) size to 1024.
    - Update finetune preprocessing scripts.
        - `.bmp` and `.jpeg` are supported. Thanks to breakcore2 and p1atdev!
        - The default weights of `tag_images_by_wd14_tagger.py` is now `SmilingWolf/wd-v1-4-convnext-tagger-v2`. You can specify another model id from `SmilingWolf` by `--repo_id` option. Thanks to SmilingWolf for the great work.
        - To change the weight, remove `wd14_tagger_model` folder, and run the script again.
        - `--max_data_loader_n_workers` option is added to each script. This option uses the DataLoader for data loading to speed up loading, 20%~30% faster.
        - Please specify 2 or 4, depends on the number of CPU cores.
        - `--recursive` option is added to `merge_dd_tags_to_metadata.py` and `merge_captions_to_metadata.py`, only works with `--full_path`.
        - `make_captions_by_git.py` is added. It uses [GIT microsoft/git-large-textcaps](https://huggingface.co/microsoft/git-large-textcaps) for captioning. 
        - `requirements.txt` is updated. If you use this script, [please update the libraries](https://github.com/kohya-ss/sd-scripts#upgrade).
        - Usage is almost the same as `make_captions.py`, but batch size should be smaller.
        - `--remove_words` option removes as much text as possible (such as `the word "XXXX" on it`).
        - `--skip_existing` option is added to `prepare_buckets_latents.py`. Images with existing npz files are ignored by this option.
        - `clean_captions_and_tags.py` is updated to remove duplicated or conflicting tags, e.g. `shirt` is removed when `white shirt` exists. if `black hair` is with `red hair`, both are removed.
    - Tag frequency is added to the metadata in `train_network.py`. Thanks to space-nuko!
        - __All tags and number of occurrences of the tag are recorded.__ If you do not want it, disable metadata storing with `--no_metadata` option.
* 2023/01/30 (v20.5.2):
  - Add `--lr_scheduler_num_cycles` and `--lr_scheduler_power` options for `train_network.py` for cosine_with_restarts and polynomial learning rate schedulers. Thanks to mgz-dev!
  - Fixed U-Net `sample_size` parameter to `64` when converting from SD to Diffusers format, in `convert_diffusers20_original_sd.py`
* 2023/01/27 (v20.5.1):
    - Fix [issue #70](https://github.com/bmaltais/kohya_ss/issues/70)
    - Fix [issue #71](https://github.com/bmaltais/kohya_ss/issues/71)
* 2023/01/26 (v20.5.0):
    - Add new `Dreambooth TI` tab for training of Textual Inversion embeddings
    - Add Textual Inversion training. Documentation is [here](./train_ti_README-ja.md) (in Japanese.)
* 2023/01/22 (v20.4.1):
    - Add new tool to verify LoRA weights produced by the trainer. Can be found under "Dreambooth LoRA/Tools/Verify LoRA"
* 2023/01/22 (v20.4.0):
    - Add support for `network_alpha` under the Training tab and support for `--training_comment` under the Folders tab.
    - Add `--network_alpha` option to specify `alpha` value to prevent underflows for stable training. Thanks to CCRcmcpe!
        - Details of the issue are described [here](https://github.com/kohya-ss/sd-webui-additional-networks/issues/49).
        - The default value is `1`, scale `1 / rank (or dimension)`. Set same value as `network_dim` for same behavior to old version.
        - LoRA with a large dimension (rank) seems to require a higher learning rate with `alpha=1` (e.g. 1e-3 for 128-dim, still investigating).　
    - For generating images in Web UI, __the latest version of the extension `sd-webui-additional-networks` (v0.3.0 or later) is required for the models trained with this release or later.__
    - Add logging for the learning rate for U-Net and Text Encoder independently, and for running average epoch loss. Thanks to mgz-dev!  
    - Add more metadata such as dataset/reg image dirs, session ID, output name etc... See [this pull request](https://github.com/kohya-ss/sd-scripts/pull/77) for details. Thanks to space-nuko!
        - __Now the metadata includes the folder name (the basename of the folder contains image files, not the full path).__ If you do not want it, disable metadata storing with `--no_metadata` option.
    - Add `--training_comment` option. You can specify an arbitrary string and refer to it by the extension.

It seems that the Stable Diffusion web UI now supports image generation using the LoRA model learned in this repository.

Note: At this time, it appears that models learned with version 0.4.0 are not supported. If you want to use the generation function of the web UI, please continue to use version 0.3.2. Also, it seems that LoRA models for SD2.x are not supported.

* 2023/01/16 (v20.3.0):
  - Fix a part of LoRA modules are not trained when `gradient_checkpointing` is enabled. 
  - Add `--save_last_n_epochs_state` option. You can specify how many state folders to keep, apart from how many models to keep. Thanks to shirayu!
  - Fix Text Encoder training stops at `max_train_steps` even if `max_train_epochs` is set in `train_db.py`.
  - Added script to check LoRA weights. You can check weights by `python networks\check_lora_weights.py <model file>`. If some modules are not trained, the value is `0.0` like following. 
    - `lora_te_text_model_encoder_layers_11_*` is not trained with `clip_skip=2`, so `0.0` is okay for these modules.

- example result of `check_lora_weights.py`, Text Encoder and a part of U-Net are not trained:
```
number of LoRA-up modules: 264
lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight,0.0
lora_te_text_model_encoder_layers_0_mlp_fc2.lora_up.weight,0.0
lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_up.weight,0.0
:
lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight,0.0
lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight,0.0
lora_unet_mid_block_attentions_0_proj_in.lora_up.weight,0.003503334941342473
lora_unet_mid_block_attentions_0_proj_out.lora_up.weight,0.004308608360588551
:
```

- all modules are trained:
```
number of LoRA-up modules: 264
lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight,0.0028684409335255623
lora_te_text_model_encoder_layers_0_mlp_fc2.lora_up.weight,0.0029794853180646896
lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_up.weight,0.002507600700482726
lora_te_text_model_encoder_layers_0_self_attn_out_proj.lora_up.weight,0.002639499492943287
:
```

* 2023/01/16 (v20.2.1):
    - Merging the latest code update from kohya
    - Added `--max_train_epochs` and `--max_data_loader_n_workers` option for each training script.
    - If you specify the number of training epochs with `--max_train_epochs`, the number of steps is calculated from the number of epochs automatically.
    - You can set the number of workers for DataLoader with `--max_data_loader_n_workers`, default is 8. The lower number may reduce the main memory usage and the time between epochs, but may cause slower data loading (training).
    - Fix loading some VAE or .safetensors as VAE is failed for `--vae` option. Thanks to Fannovel16!
    - Add negative prompt scaling for `gen_img_diffusers.py` You can set another conditioning scale to the negative prompt with `--negative_scale` option, and `--nl` option for the prompt. Thanks to laksjdjf!
    - Refactoring of GUI code and fixing mismatch... and possibly introducing bugs...
* 2023/01/11 (v20.2.0):
    - Add support for max token length
* 2023/01/10 (v20.1.1):
    - Fix issue with LoRA config loading
* 2023/01/10 (v20.1):
    - Add support for `--output_name` to trainers
    - Refactor code for easier maintenance
* 2023/01/10 (v20.0):
    - Update code base to match [latest kohys_ss code upgrade](https://github.com/kohya-ss/sd-scripts)
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