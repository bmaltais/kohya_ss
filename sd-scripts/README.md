# sd-scripts

[English](./README.md) / [日本語](./README-ja.md)

## Table of Contents
<details>
<summary>Click to expand</summary>

- [Introduction](#introduction)
  - [Supported Models](#supported-models)
  - [Features](#features)
  - [Sponsors](#sponsors)
  - [Support the Project](#support-the-project)
- [Documentation](#documentation)
  - [Training Documentation (English and Japanese)](#training-documentation-english-and-japanese)
  - [Other Documentation (English and Japanese)](#other-documentation-english-and-japanese)
- [For Developers Using AI Coding Agents](#for-developers-using-ai-coding-agents)
- [Windows Installation](#windows-installation)
  - [Windows Required Dependencies](#windows-required-dependencies)
  - [Installation Steps](#installation-steps)
  - [About requirements.txt and PyTorch](#about-requirementstxt-and-pytorch)
  - [xformers installation (optional)](#xformers-installation-optional)
- [Linux/WSL2 Installation](#linuxwsl2-installation)
  - [DeepSpeed installation (experimental, Linux or WSL2 only)](#deepspeed-installation-experimental-linux-or-wsl2-only)
- [Upgrade](#upgrade)
  - [Upgrade PyTorch](#upgrade-pytorch)
- [Credits](#credits)
- [License](#license)

</details>

## Introduction

This repository contains training, generation and utility scripts for Stable Diffusion and other image generation models.

### Sponsors

We are grateful to the following companies for their generous sponsorship:

<a href="https://aihub.co.jp/top-en">
  <img src="./images/logo_aihub.png" alt="AiHUB Inc." title="AiHUB Inc." height="100px">
</a>

### Support the Project

If you find this project helpful, please consider supporting its development via [GitHub Sponsors](https://github.com/sponsors/kohya-ss/). Your support is greatly appreciated!

### Change History

- **Version 0.10.0 (2026-01-19):**
    - `sd3` branch is merged to `main` branch. From this version, FLUX.1 and SD3/SD3.5 etc. are supported in the `main` branch.
    - There are still some missing parts in the documentation, so please let us know if you find any issues via Issues etc.
    - The `sd3` branch will be maintained as a development branch synchronized with `dev` for the time being.

### Supported Models

* **Stable Diffusion 1.x/2.x**
* **SDXL**
* **SD3/SD3.5**
* **FLUX.1**
* **LUMINA**
* **HunyuanImage-2.1**

### Features

* LoRA training
* Fine-tuning (native training, DreamBooth): except for HunyuanImage-2.1
* Textual Inversion training: SD/SDXL
* Image generation
* Other utilities such as model conversion, image tagging, LoRA merging, etc.

## Documentation

### Training Documentation (English and Japanese)

* [LoRA Training Overview](./docs/train_network.md)
* [Dataset config](./docs/config_README-en.md) / [Japanese version](./docs/config_README-ja.md)
* [Advanced Training](./docs/train_network_advanced.md)
* [SDXL Training](./docs/sdxl_train_network.md)
* [SD3 Training](./docs/sd3_train_network.md)
* [FLUX.1 Training](./docs/flux_train_network.md)
* [LUMINA Training](./docs/lumina_train_network.md)
* [HunyuanImage-2.1 Training](./docs/hunyuan_image_train_network.md)
* [Fine-tuning](./docs/fine_tune.md)
* [Textual Inversion Training](./docs/train_textual_inversion.md)
* [ControlNet-LLLite Training](./docs/train_lllite_README.md) / [Japanese version](./docs/train_lllite_README-ja.md)
* [Validation](./docs/validation.md)
* [Masked Loss Training](./docs/masked_loss_README.md) / [Japanese version](./docs/masked_loss_README-ja.md)

### Other Documentation (English and Japanese)

* [Image generation](./docs/gen_img_README.md) / [Japanese version](./docs/gen_img_README-ja.md)
* [Tagging images with WD14 Tagger](./docs/wd14_tagger_README-en.md) / [Japanese version](./docs/wd14_tagger_README-ja.md)

## For Developers Using AI Coding Agents

This repository provides recommended instructions to help AI agents like Claude and Gemini understand our project context and coding standards.

To use them, you need to opt-in by creating your own configuration file in the project root.

**Quick Setup:**

1.  Create a `CLAUDE.md` and/or `GEMINI.md` file in the project root.
2.  Add the following line to your `CLAUDE.md` to import the repository's recommended prompt:

    ```markdown
    @./.ai/claude.prompt.md
    ```

    or for Gemini:

    ```markdown
    @./.ai/gemini.prompt.md
    ```

3.  You can now add your own personal instructions below the import line (e.g., `Always respond in Japanese.`).

This approach ensures that you have full control over the instructions given to your agent while benefiting from the shared project context. Your `CLAUDE.md` and `GEMINI.md` are already listed in `.gitignore`, so they won't be committed to the repository.

## Windows Installation

### Windows Required Dependencies

Python 3.10.x and Git:

- Python 3.10.x: Download Windows installer (64-bit) from https://www.python.org/downloads/windows/
- git: Download latest installer from https://git-scm.com/download/win

Python 3.11.x, and 3.12.x will work but not tested.

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

### Installation Steps

Open a regular Powershell terminal and type the following inside:

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt

accelerate config
```

If `python -m venv` shows only `python`, change `python` to `py`.

Note: `bitsandbytes`, `prodigyopt` and `lion-pytorch` are included in the requirements.txt. If you'd like to use another version, please install it manually.

This installation is for CUDA 12.4. If you use a different version of CUDA, please install the appropriate version of PyTorch. For example, if you use CUDA 12.1, please install `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121`.

Answers to accelerate config:

```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

If you'd like to use bf16, please answer `bf16` to the last question.

Note: Some user reports ``ValueError: fp16 mixed precision requires a GPU`` is occurred in training. In this case, answer `0` for the 6th question: 
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`` 

(Single GPU with id `0` will be used.)

## About requirements.txt and PyTorch

The file does not contain requirements for PyTorch. Because the version of PyTorch depends on the environment, it is not included in the file. Please install PyTorch first according to the environment. See installation instructions below.

The scripts are tested with PyTorch 2.6.0. PyTorch 2.6.0 or later is required.

For RTX 50 series GPUs, PyTorch 2.8.0 with CUDA 12.8/12.9 should be used. `requirements.txt` will work with this version.

### xformers installation (optional)

To install xformers, run the following command in your activated virtual environment:

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu124
```

Please change the CUDA version in the URL according to your environment if necessary. xformers may not be available for some GPU architectures.

## Linux/WSL2 Installation

Linux or WSL2 installation steps are almost the same as Windows. Just change `venv\Scripts\activate` to `source venv/bin/activate`.

Note: Please make sure that NVIDIA driver and CUDA toolkit are installed in advance.

### DeepSpeed installation (experimental, Linux or WSL2 only)
  
To install DeepSpeed, run the following command in your activated virtual environment:

```bash
pip install deepspeed==0.16.7 
```

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

### Upgrade PyTorch

If you want to upgrade PyTorch, you can upgrade it with `pip install` command in [Windows Installation](#windows-installation) section.

## Credits

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for great work!

The LoRA expansion to Conv2d 3x3 was initially released by cloneofsimo and its effectiveness was demonstrated at [LoCon](https://github.com/KohakuBlueleaf/LoCon) by KohakuBlueleaf. Thank you so much KohakuBlueleaf!

## License

The majority of scripts is licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's and LoCon), however portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause
