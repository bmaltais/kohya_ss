# Kohya's GUI

This repository provides a Windows-focused Gradio GUI for [Kohya's Stable Diffusion trainers](https://github.com/kohya-ss/sd-scripts). The GUI allows you to set the training parameters and generate and run the required CLI commands to train the model.

If you run on Linux and would like to use the GUI, there is now a port of it as a docker container. You can find the project [here](https://github.com/P2Enjoy/kohya_ss-docker).

### Table of Contents

- [Tutorials](#tutorials)
- [Required Dependencies](#required-dependencies)
  - [Linux/macOS](#linux-and-macos-dependencies)
- [Installation](#installation)
    - [Linux/macOS](#linux-and-macos)
      - [Default Install Locations](#install-location)
    - [Windows](#windows)
    - [CUDNN 8.6](#optional--cudnn-86)
- [Upgrading](#upgrading)
  - [Windows](#windows-upgrade)
  - [Linux/macOS](#linux-and-macos-upgrade)
- [Launching the GUI](#starting-gui-service)
  - [Windows](#launching-the-gui-on-windows)
  - [Linux/macOS](#launching-the-gui-on-linux-and-macos)
  - [Direct Launch via Python Script](#launching-the-gui-directly-using-kohyaguipy)
- [Dreambooth](#dreambooth)
- [Finetune](#finetune)
- [Train Network](#train-network)
- [LoRA](#lora)
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

## Required Dependencies

- Install [Python 3.10](https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe) 
  - make sure to tick the box to add Python to the 'PATH' environment variable
- Install [Git](https://git-scm.com/download/win)
- Install [Visual Studio 2015, 2017, 2019, and 2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Linux and macOS dependencies

These dependencies are taken care of via `setup.sh` in the installation section. No additional steps should be needed unless the scripts inform you otherwise.

## Installation

### Runpod
Follow the instructions found in this discussion: https://github.com/bmaltais/kohya_ss/discussions/379

### Linux and macOS
In the terminal, run

```
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
# May need to chmod +x ./setup.sh if you're on a machine with stricter security.
# There are additional options if needed for a runpod environment.
# Call 'setup.sh -h' or 'setup.sh --help' for more information.
./setup.sh
```

Setup.sh help included here:

```bash
Kohya_SS Installation Script for POSIX operating systems.

The following options are useful in a runpod environment,
but will not affect a local machine install.

Usage:
  setup.sh -b dev -d /workspace/kohya_ss -g https://mycustom.repo.tld/custom_fork.git
  setup.sh --branch=dev --dir=/workspace/kohya_ss --git-repo=https://mycustom.repo.tld/custom_fork.git

Options:
  -b BRANCH, --branch=BRANCH    Select which branch of kohya to check out on new installs.
  -d DIR, --dir=DIR             The full path you want kohya_ss installed to.
  -g REPO, --git_repo=REPO      You can optionally provide a git repo to check out for runpod installation. Useful for custom forks.
  -h, --help                    Show this screen.
  -i, --interactive             Interactively configure accelerate instead of using default config file.
  -n, --no-update               Do not update kohya_ss repo. No git pull or clone operations.
  -p, --public                  Expose public URL in runpod mode. Won't have an effect in other modes.
  -r, --runpod                  Forces a runpod installation. Useful if detection fails for any reason.
  -s, --skip-space-check        Skip the 10Gb minimum storage space check.
  -u, --no-gui                  Skips launching the GUI.
  -v, --verbose                 Increase verbosity levels up to 3.
```

#### Install location

The default install location for Linux is where the script is located if a previous installation is detected that location.
Otherwise, it will fall to `/opt/kohya_ss`. If /opt is not writeable, the fallback is `$HOME/kohya_ss`. Lastly, if all else fails it will simply install to the current folder you are in (PWD).

On macOS and other non-Linux machines, it will first try to detect an install where the script is run from and then run setup there if that's detected. 
If a previous install isn't found at that location, then it will default install to `$HOME/kohya_ss` followed by where you're currently at if there's no access to $HOME.
You can override this behavior by specifying an install directory with the -d option.

If you are using the interactive mode, our default values for the accelerate config screen after running the script answer "This machine", "None", "No" for the remaining questions.
These are the same answers as the Windows install.

### Windows
In the terminal, run:

```
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
setup.bat
```

Then configure accelerate with the same answers as in the MacOS instructions when prompted.

### Optional: CUDNN 8.6

This step is optional but can improve the learning speed for NVIDIA 30X0/40X0 owners. It allows for larger training batch size and faster training speed.

Due to the file size, I can't host the DLLs needed for CUDNN 8.6 on Github. I strongly advise you download them for a speed boost in sample generation (almost 50% on 4090 GPU) you can download them [here](https://b1.thefileditch.ch/mwxKTEtelILoIbMbruuM.zip).

To install, simply unzip the directory and place the `cudnn_windows` folder in the root of the this repo.

Run the following commands to install:

```
.\venv\Scripts\activate

python .\tools\cudann_1.8_install.py
```

Once the commands have completed successfully you should be ready to use the new version. MacOS support is not tested and has been mostly taken from https://gist.github.com/jstayco/9f5733f05b9dc29de95c4056a023d645

## Upgrading

The following commands will work from the root directory of the project if you'd prefer to not run scripts.
These commands will work on any OS.
```bash
git pull

.\venv\Scripts\activate

pip install --use-pep517 --upgrade -r requirements.txt
```

### Windows Upgrade
When a new release comes out, you can upgrade your repo with the following commands in the root directory:

```powershell
upgrade.bat
```

### Linux and macOS Upgrade
You can cd into the root directory and simply run

```bash
# Refresh and update everything
./setup.sh

# This will refresh everything, but NOT clone or pull the git repo.
./setup.sh --no-git-update
```

Once the commands have completed successfully you should be ready to use the new version.

# Starting GUI Service

The following command line arguments can be passed to the scripts on any OS to configure the underlying service.
```
--listen: the IP address to listen on for connections to Gradio.
--username: a username for authentication. 
--password: a password for authentication. 
--server_port: the port to run the server listener on. 
--inbrowser: opens the Gradio UI in a web browser. 
--share: shares the Gradio UI.
```

### Launching the GUI on Windows

The two scripts to launch the GUI on Windows are gui.ps1 and gui.bat in the root directory.
You can use whichever script you prefer.

To launch the Gradio UI, run the script in a terminal with the desired command line arguments, for example:

`gui.ps1 --listen 127.0.0.1 --server_port 7860 --inbrowser --share`

or

`gui.bat --listen 127.0.0.1 --server_port 7860 --inbrowser --share`

## Launching the GUI on Linux and macOS

Run the launcher script with the desired command line arguments similar to Windows.
`gui.sh --listen 127.0.0.1 --server_port 7860 --inbrowser --share`

## Launching the GUI directly using kohya_gui.py

To run the GUI directly bypassing the wrapper scripts, simply use this command from the root project directory:

```
.\venv\Scripts\activate

python .\kohya_gui.py
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

This will store a backup file with your current locally installed pip packages and then uninstall them. Then, redo the installation instructions within the kohya_ss venv.

## Change History

* 2023/04/09 (v21.5.2)

    - Added support for training with weighted captions. Thanks to AI-Casanova for the great contribution! 
    - Please refer to the PR for details: [PR #336](https://github.com/kohya-ss/sd-scripts/pull/336)
    - Specify the `--weighted_captions` option. It is available for all training scripts except Textual Inversion and XTI.
    - This option is also applicable to token strings of the DreamBooth method.
    - The syntax for weighted captions is almost the same as the Web UI, and you can use things like `(abc)`, `[abc]`, and `(abc:1.23)`. Nesting is also possible.
    - If you include a comma in the parentheses, the parentheses will not be properly matched in the prompt shuffle/dropout, so do not include a comma in the parentheses.
  
* 2023/04/08 (v21.5.1)
    - Integrate latest sd-scripts updates. Not integrated in the GUI. Will consider if you think it is wort integrating. At the moment you can add the required parameters using the `Additional parameters` field under the `Advanced Configuration` accordion in the `Training Parameters` tab:
        - There may be bugs because I changed a lot. If you cannot revert the script to the previous version when a problem occurs, please wait for the update for a while.
    - There may be bugs because I changed a lot. If you cannot revert the script to the previous version when a problem occurs, please wait for the update for a while.

        - Added a feature to upload model and state to HuggingFace. Thanks to ddPn08 for the contribution! [PR #348](https://github.com/kohya-ss/sd-scripts/pull/348)
        - When `--huggingface_repo_id` is specified, the model is uploaded to HuggingFace at the same time as saving the model.
        - Please note that the access token is handled with caution. Please refer to the [HuggingFace documentation](https://huggingface.co/docs/hub/security-tokens).
        - For example, specify other arguments as follows.
            - `--huggingface_repo_id "your-hf-name/your-model" --huggingface_path_in_repo "path" --huggingface_repo_type model --huggingface_repo_visibility private --huggingface_token hf_YourAccessTokenHere`
        - If `public` is specified for `--huggingface_repo_visibility`, the repository will be public. If the option is omitted or `private` (or anything other than `public`) is specified, it will be private.
        - If you specify `--save_state` and `--save_state_to_huggingface`, the state will also be uploaded.
        - If you specify `--resume` and `--resume_from_huggingface`, the state will be downloaded from HuggingFace and resumed.
            - In this case, the `--resume` option is `--resume {repo_id}/{path_in_repo}:{revision}:{repo_type}`. For example: `--resume_from_huggingface --resume your-hf-name/your-model/path/test-000002-state:main:model`
        - If you specify `--async_upload`, the upload will be done asynchronously.
        - Added the documentation for applying LoRA to generate with the standard pipeline of Diffusers.   [training LoRA](https://github-com.translate.goog/kohya-ss/sd-scripts/blob/main/train_network_README-ja.md?_x_tr_sl=fr&_x_tr_tl=en&_x_tr_hl=en-US&_x_tr_pto=wapp#diffusers%E3%81%AEpipeline%E3%81%A7%E7%94%9F%E6%88%90%E3%81%99%E3%82%8B) (Google translate from Japanese)
        - Support for Attention Couple and regional LoRA in `gen_img_diffusers.py`.
        - If you use ` AND ` to separate the prompts, each sub-prompt is sequentially applied to LoRA. `--mask_path` is treated as a mask image. The number of sub-prompts and the number of LoRA must match.
    - Resolved bug https://github.com/bmaltais/kohya_ss/issues/554
* 2023/04/07 (v21.5.0)
    - Update MacOS and Linux install scripts. Thanks @jstayco
    - Update windows upgrade ps1 and bat
    - Update kohya_ss sd-script code to latest release... this is a big one so it might cause some training issue. If you find that this release is causing issues for you you can go back to the previous release with `git checkout v21.4.2` and then run the upgrade script for your platform. Here is the list of changes in the new sd-scripts:
        - There may be bugs because I changed a lot. If you cannot revert the script to the previous version when a problem occurs, please wait for the update for a while.
        - The learning rate and dim (rank) of each block may not work with other modules (LyCORIS, etc.) because the module needs to be changed.

        - Fix some bugs and add some features.
            - Fix an issue that `.json` format dataset config files cannot be read.  [issue #351](https://github.com/kohya-ss/sd-scripts/issues/351) Thanks to rockerBOO!
            - Raise an error when an invalid `--lr_warmup_steps` option is specified (when warmup is not valid for the specified scheduler).  [PR #364](https://github.com/kohya-ss/sd-scripts/pull/364)  Thanks to shirayu!
            - Add `min_snr_gamma` to metadata in `train_network.py`. [PR #373](https://github.com/kohya-ss/sd-scripts/pull/373) Thanks to rockerBOO!
            - Fix the data type handling in `fine_tune.py`. This may fix an error that occurs in some environments when using xformers, npz format cache, and mixed_precision.

        - Add options to `train_network.py` to specify block weights for learning rates. [PR #355](https://github.com/kohya-ss/sd-scripts/pull/355) Thanks to u-haru for the great contribution!
            - Specify the weights of 25 blocks for the full model.
            - No LoRA corresponds to the first block, but 25 blocks are specified for compatibility with 'LoRA block weight' etc. Also, if you do not expand to conv2d3x3, some blocks do not have LoRA, but please specify 25 values ​​for the argument for consistency.
            - Specify the following arguments with `--network_args`.
            - `down_lr_weight` : Specify the learning rate weight of the down blocks of U-Net. The following can be specified.
            - The weight for each block: Specify 12 numbers such as `"down_lr_weight=0,0,0,0,0,0,1,1,1,1,1,1"`.
            - Specify from preset: Specify such as `"down_lr_weight=sine"` (the weights by sine curve). sine, cosine, linear, reverse_linear, zeros can be specified. Also, if you add `+number` such as `"down_lr_weight=cosine+.25"`, the specified number is added (such as 0.25~1.25).
            - `mid_lr_weight` : Specify the learning rate weight of the mid block of U-Net. Specify one number such as `"down_lr_weight=0.5"`.
            - `up_lr_weight` : Specify the learning rate weight of the up blocks of U-Net. The same as down_lr_weight.
            - If you omit the some arguments, the 1.0 is used. Also, if you set the weight to 0, the LoRA modules of that block are not created.
            - `block_lr_zero_threshold` : If the weight is not more than this value, the LoRA module is not created. The default is 0.

        - Add options to `train_network.py` to specify block dims (ranks) for variable rank.
            - Specify 25 values ​​for the full model of 25 blocks. Some blocks do not have LoRA, but specify 25 values ​​always.
            - Specify the following arguments with `--network_args`.
            - `block_dims` : Specify the dim (rank) of each block. Specify 25 numbers such as `"block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"`.
            - `block_alphas` : Specify the alpha of each block. Specify 25 numbers as with block_dims. If omitted, the value of network_alpha is used.
            - `conv_block_dims` : Expand LoRA to Conv2d 3x3 and specify the dim (rank) of each block.
            - `conv_block_alphas` : Specify the alpha of each block when expanding LoRA to Conv2d 3x3. If omitted, the value of conv_alpha is used.
    - Add GUI support for new features introduced above by kohya_ss. Those will be visible only if the LoRA is of type `Standard` or `kohya LoCon`. You will find the new parameters under the `Advanced Configuration` accordion in the `Training parameters` tab.
    - Various improvements to linux and macos srtup scripts thanks to @Oceanswave and @derVedro
    - Integrated sd-scripts commits into commit history. Thanks to @Cauldrath
* 2023/04/02 (v21.4.2)
    - removes TensorFlow from requirements.txt for Darwin platforms as pip does not support advanced conditionals like CPU architecture. The logic is now defined in setup.sh to avoid version bump headaches, and the selection logic is in the pre-existing pip function. Additionally, the release includes the addition of the tensorflow-metal package for M1+ Macs, which enables GPU acceleration per Apple's documentation. Thanks @jstayco
* 2023/04/01 (v21.4.1)
    - Fix type for linux install by @bmaltais in https://github.com/bmaltais/kohya_ss/pull/517
    - Fix .gitignore by @bmaltais in https://github.com/bmaltais/kohya_ss/pull/518
* 2023/04/01 (v21.4.0)
    - Improved linux and macos installation and updates script. See README for more details. Many thanks to @jstayco and @Galunid for the great PR!
    - Fix issue with "missing library" error.
* 2023/04/01 (v21.3.9)
    - Update how setup is done on Windows by introducing a setup.bat script. This will make it easier to install/re-install on Windows if needed. Many thanks to @missionfloyd for his PR: https://github.com/bmaltais/kohya_ss/pull/496
* 2023/03/30 (v21.3.8)
    - Fix issue with LyCORIS version not being found: https://github.com/bmaltais/kohya_ss/issues/481
* 2023/03/29 (v21.3.7)
    - Allow for 0.1 increment in Network and Conv alpha values: https://github.com/bmaltais/kohya_ss/pull/471 Thanks to @srndpty
    - Updated Lycoris module version
* 2023/03/28 (v21.3.6)
    - Fix issues when `--persistent_data_loader_workers` is specified.
        - The batch members of the bucket are not shuffled.
        - `--caption_dropout_every_n_epochs` does not work.
        - These issues occurred because the epoch transition was not recognized correctly. Thanks to u-haru for reporting the issue.
    - Fix an issue that images are loaded twice in Windows environment.
    - Add Min-SNR Weighting strategy. Details are in [#308](https://github.com/kohya-ss/sd-scripts/pull/308). Thank you to AI-Casanova for this great work!
        - Add `--min_snr_gamma` option to training scripts, 5 is recommended by paper.
        - The Min SNR gamma fields can be found under the advanced training tab in all trainers.
    - Fixed the error while images are ended with capital image extensions. Thanks to @kvzn. https://github.com/bmaltais/kohya_ss/pull/454
* 2023/03/26 (v21.3.5)
    - Fix for https://github.com/bmaltais/kohya_ss/issues/230
    - Added detection for Google Colab to not bring up the GUI file/folder window on the platform. Instead it will only use the file/folder path provided in the input field.
* 2023/03/25 (v21.3.4)
    - Added untested support for MacOS base on this gist: https://gist.github.com/jstayco/9f5733f05b9dc29de95c4056a023d645

    Let me know how this work. From the look of it it appear to be well thought out. I modified a few things to make it fit better with the rest of the code in the repo.
    - Fix for issue https://github.com/bmaltais/kohya_ss/issues/433 by implementing default of 0.
    - Removed non applicable save_model_as choices for LoRA and TI.
* 2023/03/24 (v21.3.3)
    - Add support for custom user gui files. THey will be created at installation time or when upgrading is missing. You will see two files in the root of the folder. One named `gui-user.bat` and the other `gui-user.ps1`. Edit the file based on your preferred terminal. Simply add the parameters you want to pass the gui in there and execute it to start the gui with them. Enjoy!
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
