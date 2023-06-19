# Kohya's GUI

This repository provides a Windows-focused Gradio GUI for [Kohya's Stable Diffusion trainers](https://github.com/kohya-ss/sd-scripts). The GUI allows you to set the training parameters and generate and run the required CLI commands to train the model.

### Table of Contents

- [Tutorials](#tutorials)
* [Training guide - common](./docs/train_README-ja.md) : data preparation, options etc... 
  * [Chinese version](./docs/train_README-zh.md)
  * [Dataset config](./docs/config_README-ja.md) 
  * [DreamBooth training guide](./docs/train_db_README-ja.md)
  * [Step by Step fine-tuning guide](./docs/fine_tune_README_ja.md):
  * [Training LoRA](./docs/train_network_README-ja.md)
  * [training Textual Inversion](./docs/train_ti_README-ja.md)
  * [Image generation](./docs/gen_img_README-ja.md)
  * [Model conversion](https://note.com/kohya_ss/n/n374f316fe4ad)
- [Required Dependencies](#required-dependencies)
  - [Linux/macOS](#linux-and-macos-dependencies)
- [Installation](#installation)
    - [Docker](#docker)
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

Newer Tutorial: [Generate Studio Quality Realistic Photos By Kohya LoRA Stable Diffusion Training](https://www.youtube.com/watch?v=TpuDOsuKIBo):

[![Newer Tutorial: Generate Studio Quality Realistic Photos By Kohya LoRA Stable Diffusion Training](https://user-images.githubusercontent.com/19240467/235306147-85dd8126-f397-406b-83f2-368927fa0281.png)](https://www.youtube.com/watch?v=TpuDOsuKIBo)

Newer Tutorial: [How To Install And Use Kohya LoRA GUI / Web UI on RunPod IO](https://www.youtube.com/watch?v=3uzCNrQao3o):

[![How To Install And Use Kohya LoRA GUI / Web UI on RunPod IO With Stable Diffusion & Automatic1111](https://github-production-user-asset-6210df.s3.amazonaws.com/19240467/238678226-0c9c3f7d-c308-4793-b790-999fdc271372.png)](https://www.youtube.com/watch?v=3uzCNrQao3o)

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

### Docker
Docker is supported on Windows and Linux distributions. However this method currently only supports Nvidia GPUs. 
Run the following commands in your OS shell after installing [git](https://git-scm.com/download/) and [docker](https://www.docker.com/products/docker-desktop/):
```bash
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
docker compose build
docker compose run --service-ports kohya-ss-gui
```

This will take a while (up to 20 minutes) on the first run.

The following limitations apply:
* All training data must be added to the `dataset` subdirectory, the docker container cannot access any other files
* The file picker does not work
  * Cannot select folders, folder path must be set manually like e.g. /dataset/my_lora/img
  * Cannot select config file, it must be loaded via path instead like e.g. /dataset/my_config.json  
* Dialogs do not work
  * Make sure your file names are unique as this happens when asking if an existing file should be overridden
* No auto-update support. Must run update scripts outside docker manually and then rebuild with `docker compose build`.


If you run on Linux, there is an alternative docker container port with less limitations. You can find the project [here](https://github.com/P2Enjoy/kohya_ss-docker).

### Linux and macOS
#### Linux pre-requirements

venv support need to be pre-installed. Can be done on ubuntu 22.04 with `apt install python3.10-venv`

Make sure to use a version of python >= 3.10.6 and < 3.11.0

#### Setup

In the terminal, run

```bash
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

- Install [Python 3.10](https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe) 
  - make sure to tick the box to add Python to the 'PATH' environment variable
- Install [Git](https://git-scm.com/download/win)
- Install [Visual Studio 2015, 2017, 2019, and 2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

In the terminal, run:

```
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
.\setup.bat
```

If this is a 1st install answer No when asked `Do you want to uninstall previous versions of torch and associated files before installing`.


Then configure accelerate with the same answers as in the MacOS instructions when prompted.

### Optional: CUDNN 8.6

This step is optional but can improve the learning speed for NVIDIA 30X0/40X0 owners. It allows for larger training batch size and faster training speed.

Due to the file size, I can't host the DLLs needed for CUDNN 8.6 on Github. I strongly advise you download them for a speed boost in sample generation (almost 50% on 4090 GPU) you can download them [here](https://github.com/bmaltais/python-library/raw/main/cudnn_windows.zip).

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

### Naming of LoRA

The LoRA supported by `train_network.py` has been named to avoid confusion. The documentation has been updated. The following are the names of LoRA types in this repository.

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers)

    LoRA for Linear layers and Conv2d layers with 1x1 kernel

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers)

    In addition to 1., LoRA for Conv2d layers with 3x3 kernel 
    
LoRA-LierLa is the default LoRA type for `train_network.py` (without `conv_dim` network arg). LoRA-LierLa can be used with [our extension](https://github.com/kohya-ss/sd-webui-additional-networks) for AUTOMATIC1111's Web UI, or with the built-in LoRA feature of the Web UI.

To use LoRA-C3Lier with Web UI, please use our extension.

## Sample image generation during training
A prompt file might look like this, for example

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following can be used.

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

  The prompt weighting such as `( )` and `[ ]` are working.

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

* 2023/06/19 (v21.7.10)
- Quick fix for linux GUI startup where it would try to install darwin requirements on top of linux. Ugly fix but work. Hopefulle some linux user will improve via a PR.
* 2023/06/18 (v21.7.9)
- Implement temporary fix for validation of image dataset. Will no longer stop execution but will let training continue... this is changed to avoid stopping training on false positive... yet still raise awaireness that something might be wrong with the image dataset structure.
* 2023/06/14 (v21.7.8)
- Add tkinter to dockerised version (thanks to @burdokow)
- Add option to create caption files from folder names to the `group_images.py` tool.
- Prodigy optimizer is supported in each training script. It is a member of D-Adaptation and is effective for DyLoRA training. [PR #585](https://github.com/kohya-ss/sd-scripts/pull/585) Please see the PR for details. Thanks to sdbds!
  - Install the package with `pip install prodigyopt`. Then specify the option like `--optimizer_type="prodigy"`.
- Arbitrary Dataset is supported in each training script (except XTI). You can use it by defining a Dataset class that returns images and captions.
  - Prepare a Python script and define a class that inherits `train_util.MinimalDataset`. Then specify the option like `--dataset_class package.module.DatasetClass` in each training script.
  - Please refer to `MinimalDataset` for implementation. I will prepare a sample later.
- The following features have been added to the generation script.
  - Added an option `--highres_fix_disable_control_net` to disable ControlNet in the 2nd stage of Highres. Fix. Please try it if the image is disturbed by some ControlNet such as Canny.
  - Added Variants similar to sd-dynamic-propmpts in the prompt.
    - If you specify `{spring|summer|autumn|winter}`, one of them will be randomly selected.
    - If you specify `{2$$chocolate|vanilla|strawberry}`, two of them will be randomly selected.
    - If you specify `{1-2$$ and $$chocolate|vanilla|strawberry}`, one or two of them will be randomly selected and connected by ` and `.
    - You can specify the number of candidates in the range `0-2`. You cannot omit one side like `-2` or `1-`.
    - It can also be specified for the prompt option.
    - If you specify `e` or `E`, all candidates will be selected and the prompt will be repeated multiple times (`--images_per_prompt` is ignored). It may be useful for creating X/Y plots.
    - You can also specify `--am {e$$0.2|0.4|0.6|0.8|1.0},{e$$0.4|0.7|1.0} --d 1234`. In this case, 15 prompts will be generated with 5*3.
    - There is no weighting function.
- Add pre and posfix to wd14
* 2023/06/12 (v21.7.7)
- Add `Print only` button to all training tabs
- Sort json file vars for easier visual search
- Fixed a bug where clip skip did not work when training with weighted captions (`--weighted_captions` specified) and when generating sample images during training.
- Add verification and reporting of bad dataset folder name structure for DB, LoRA and TI training.
- Some docker build fix.
* 2023/06/06 (v21.7.6)
- Small UI improvements
- Fix `train_network.py` to probably work with older versions of LyCORIS.
- `gen_img_diffusers.py` now supports `BREAK` syntax.
- Add Lycoris iA3, LoKr and DyLoRA support to the UI
- Upgrade LuCORIS python module to 0.1.6
* 2023/06/05 (v21 7.5)
- Fix reported issue with LoHA: https://github.com/bmaltais/kohya_ss/issues/922
* 2023/06/05 (v21.7.4)
- Add manual accelerate config option
- Remove the ability to switch between torch 1 and 2 as it was causing errors with the venv
* 2023/06/04 (v21.7.3)
- Add accelerate configuration from file
- Fix issue with torch uninstallation resulting in Error sometimes
- Fix broken link to cudann files
* 2023/06/04 (v21.7.2)
- Improve handling of legacy installations
* 2023/06/04 (v21.7.1)
- This is mostly an update to the whole setup method for kohya_ss. I got fedup with all the issues from the batch file method and leveraged the great work of vladimandic to improve the whole setup experience.

There is now a new menu in setup.bat that will appear:

```
Kohya_ss GUI setup menu:

0. Cleanup the venv
1. Install kohya_ss gui [torch 1]
2. Install kohya_ss gui [torch 2]
3. Start GUI in browser
4. Quit

Enter your choice:
```

The only obscure option might be option 0. This will help cleanup a corrupted venv without having to delete de folder. This van be really usefull for cases where nothing is working anymore and you should re-install from scratch. Just run the venv cleanup then select the version of kohya_ss GUI you want to instal (torch1 or 2).

You can also start the GUI right from the setup menu using option 3.

After pulling a new version you can either re-run `setup.bat` and install the version you want... or just run `gui.bat` and it will update the python modules as required.

Hope this is useful.

* 2023/06/04 (v21.7.0)
- Max Norm Regularization is now available in `train_network.py`. [PR #545](https://github.com/kohya-ss/sd-scripts/pull/545) Thanks to AI-Casanova!
  - Max Norm Regularization is a technique to stabilize network training by limiting the norm of network weights. It may be effective in suppressing overfitting of LoRA and improving stability when used with other LoRAs. See PR for details.
  - Specify as `--scale_weight_norms=1.0`. It seems good to try from `1.0`.
  - The networks other than LoRA in this repository (such as LyCORIS) do not support this option.

- Three types of dropout have been added to `train_network.py` and LoRA network.
  - Dropout is a technique to suppress overfitting and improve network performance by randomly setting some of the network outputs to 0.
  - `--network_dropout` is a normal dropout at the neuron level. In the case of LoRA, it is applied to the output of down. Proposed in [PR #545](https://github.com/kohya-ss/sd-scripts/pull/545) Thanks to AI-Casanova!
    - `--network_dropout=0.1` specifies the dropout probability to `0.1`.
    - Note that the specification method is different from LyCORIS.
  - For LoRA network, `--network_args` can specify `rank_dropout` to dropout each rank with specified probability. Also `module_dropout` can be specified to dropout each module with specified probability.
    - Specify as `--network_args "rank_dropout=0.2" "module_dropout=0.1"`.
  - `--network_dropout`, `rank_dropout`, and `module_dropout` can be specified at the same time.
  - Values of 0.1 to 0.3 may be good to try. Values greater than 0.5 should not be specified.
  - `rank_dropout` and `module_dropout` are original techniques of this repository. Their effectiveness has not been verified yet.
  - The networks other than LoRA in this repository (such as LyCORIS) do not support these options.
  
- Added an option `--scale_v_pred_loss_like_noise_pred` to scale v-prediction loss like noise prediction in each training script.
  - By scaling the loss according to the time step, the weights of global noise prediction and local noise prediction become the same, and the improvement of details may be expected.
  - See [this article](https://xrg.hatenablog.com/entry/2023/06/02/202418) by xrg for details (written in Japanese). Thanks to xrg for the great suggestion!

* 2023/06/03 (v21.6.5)
- Fix dreambooth issue with new logging
- Update setup and upgrade scripts
- Adding test folder

* 2023/05/28 (v21.5.15)
- Show warning when image caption file does not exist during training. [PR #533](https://github.com/kohya-ss/sd-scripts/pull/533) Thanks to TingTingin!
  - Warning is also displayed when using class+identifier dataset. Please ignore if it is intended.
- `train_network.py` now supports merging network weights before training. [PR #542](https://github.com/kohya-ss/sd-scripts/pull/542) Thanks to u-haru!
  - `--base_weights` option specifies LoRA or other model files (multiple files are allowed) to merge.
  - `--base_weights_multiplier` option specifies multiplier of the weights to merge (multiple values are allowed). If omitted or less than `base_weights`, 1.0 is used.
  - This is useful for incremental learning. See PR for details.
- Show warning and continue training when uploading to HuggingFace fails.
