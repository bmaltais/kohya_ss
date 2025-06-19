# Kohya's GUI

[![GitHub stars](https://img.shields.io/github/stars/bmaltais/kohya_ss?style=social)](https://github.com/bmaltais/kohya_ss/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/bmaltais/kohya_ss?style=social)](https://github.com/bmaltais/kohya_ss/network/members)
[![License](https://img.shields.io/github/license/bmaltais/kohya_ss)](LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/bmaltais/kohya_ss)](https://github.com/bmaltais/kohya_ss/issues)

This is a GUI and CLI for training diffusion models.

This project provides a user-friendly Gradio-based Graphical User Interface (GUI) for [Kohya's Stable Diffusion training scripts](https://github.com/kohya-ss/sd-scripts). 
Stable Diffusion training empowers users to customize image generation models by fine-tuning existing models, creating unique artistic styles, 
and training specialized models like LoRA (Low-Rank Adaptation).

Key features of this GUI include:
*   Easy-to-use interface for setting a wide range of training parameters.
*   Automatic generation of the command-line interface (CLI) commands required to run the training scripts.
*   Support for various training methods, including LoRA, Dreambooth, fine-tuning, and SDXL training.

Support for Linux and macOS is also available. While Linux support is actively maintained through community contributions, macOS compatibility may vary.

## Table of Contents

- [Installation Options](#installation-options)
  - [Local Installation Overview](#local-installation-overview)
    - [`uv` vs `pip` – What's the Difference?](#uv-vs-pip--whats-the-difference)
  - [Cloud Installation Overview](#cloud-installation-overview)
    - [Colab](#-colab)
    - [Runpod, Novita, Docker](#runpod-novita-docker)
- [Custom Path Defaults](#custom-path-defaults)
    - [LoRA](#lora)
  - [Sample image generation during training](#sample-image-generation-during-training)
  - [Troubleshooting](#troubleshooting)
  - [Page File Limit](#page-file-limit)
  - [No module called tkinter](#no-module-called-tkinter)
  - [LORA Training on TESLA V100 - GPU Utilization Issue](#lora-training-on-tesla-v100---gpu-utilization-issue)
- [SDXL training](#sdxl-training)
- [Masked loss](#masked-loss)
- [Guides](#guides)
  - [Using Accelerate Lora Tab to Select GPU ID](#using-accelerate-lora-tab-to-select-gpu-id)
    - [Starting Accelerate in GUI](#starting-accelerate-in-gui)
    - [Running Multiple Instances (linux)](#running-multiple-instances-linux)
    - [Monitoring Processes](#monitoring-processes)
- [Interesting Forks](#interesting-forks)
- [Contributing](#contributing)
- [License](#license)
- [Change History](#change-history)
  - [v25.0.3](#v2503)
  - [v25.0.2](#v2502)
  - [v25.0.1](#v2501)
  - [v25.0.0](#v2500)


## Installation Options

You can run `kohya_ss` either **locally on your machine** or via **cloud-based solutions** like Colab or Runpod.

- If you have a GPU-equipped PC and want full control: install it locally using `uv` or `pip`.
- If your system doesn’t meet requirements or you prefer a browser-based setup: use Colab or a paid GPU provider like Runpod or Novita.
- If you are a developer or DevOps user, Docker is also supported.

---

### Local Installation Overview

You can install `kohya_ss` locally using either the `uv` or `pip` method. Choose one depending on your platform and preferences:

| Platform     | Recommended Method | Instructions                                |
|--------------|----------------|---------------------------------------------|
| Linux        | `uv`           | [uv_linux.md](./docs/Installation/uv_linux.md) |
| Linux or Mac | `pip`              | [pip_linux.md](./docs/Installation/pip_linux.md)               |
| Windows      | `uv`           | [uv_windows.md](./docs/Installation/uv_windows.md)             |
| Windows      | `pip`          | [pip_windows.md](./docs/Installation/pip_windows.md)           |

#### `uv` vs `pip` – What's the Difference?

- `uv` is faster and isolates dependencies more cleanly, ideal if you want minimal setup hassle.
- `pip` is more traditional, easier to debug if issues arise, and works better with some IDEs or Python tooling.
- If unsure: try `uv`. If it doesn't work for you, fall back to `pip`.

### Cloud Installation Overview

#### 🦒 Colab

For browser-based training without local setup, use this Colab notebook:  
<https://github.com/camenduru/kohya_ss-colab>

- No installation required
- Free to use (GPU availability may vary)
- Maintained by **camenduru**, not the original author

| Colab                                                                                                                                                                          | Info               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/kohya_ss-colab/blob/main/kohya_ss_colab.ipynb) | kohya_ss_gui_colab |

> 💡 If you encounter issues, please report them on camenduru’s repo.

**Special thanks**  
I would like to express my gratitude to camenduru for their valuable contribution.

#### Runpod, Novita, Docker

These options are for users running training on hosted GPU infrastructure or containers.

- **[Runpod setup](docs/runpod_setup.md)** – Ready-made GPU background training via templates.
- **[Novita setup](docs/novita_setup.md)** – Similar to Runpod, but integrated into the Novita UI.
- **[Docker setup](docs/docker.md)** – For developers/sysadmins using containerized environments.


## Custom Path Defaults with `config.toml`

The GUI supports a configuration file named `config.toml` that allows you to set default paths for many of the input fields. This is useful for avoiding repetitive manual selection of directories every time you start the GUI.

**Purpose of `config.toml`:**

*   Pre-fill default directory paths for pretrained models, datasets, output folders, LoRA models, etc.
*   Streamline your workflow by having the GUI remember your preferred locations.

**How to Use and Customize:**

1.  **Create your configuration file:**
    *   In the root directory of the `kohya_ss` repository, you'll find a file named `config example.toml`.
    *   Copy this file and rename the copy to `config.toml`. This `config.toml` file will be automatically loaded when the GUI starts.
2.  **Edit `config.toml`:**
    *   Open `config.toml` with a text editor.
    *   The file uses TOML (Tom's Obvious, Minimal Language) format, which consists of `key = "value"` pairs.
    *   Modify the paths for the keys according to your local directory structure.
    *   **Important:**
        *   Use absolute paths (e.g., `C:/Users/YourName/StableDiffusion/Models` or `/home/yourname/sd-models`).
        *   Alternatively, you can use paths relative to the `kohya_ss` root directory.
        *   Ensure you use forward slashes (`/`) for paths, even on Windows, as this is generally more compatible with TOML and Python.
        *   Make sure the specified directories exist on your system.

**Structure of `config.toml`:**

The `config.toml` file can have several sections, typically corresponding to different training modes or general settings. Common keys you might want to set include:

*   `model_dir`: Default directory for loading base Stable Diffusion models.
*   `lora_model_dir`: Default directory for saving and loading LoRA models.
*   `output_dir`: Default base directory for training outputs (images, logs, model checkpoints).
*   `dataset_dir`: A general default if you store all your datasets in one place.
*   Specific input paths for different training tabs like Dreambooth, Finetune, LoRA, etc. (e.g., `db_model_dir`, `ft_source_model_name_or_path`).

**Example Configurations:**

Here's an example snippet of what your `config.toml` might look like:

```toml
# General settings
model_dir = "C:/ai_stuff/stable-diffusion-webui/models/Stable-diffusion"
lora_model_dir = "C:/ai_stuff/stable-diffusion-webui/models/Lora"
vae_dir = "C:/ai_stuff/stable-diffusion-webui/models/VAE"
output_dir = "C:/ai_stuff/kohya_ss_outputs"
logging_dir = "C:/ai_stuff/kohya_ss_outputs/logs"

# Dreambooth specific paths
db_model_dir = "C:/ai_stuff/stable-diffusion-webui/models/Stable-diffusion"
db_reg_image_dir = "C:/ai_stuff/datasets/dreambooth_regularization_images"
# Add other db_... paths as needed

# Finetune specific paths
ft_model_dir = "C:/ai_stuff/stable-diffusion-webui/models/Stable-diffusion"
# Add other ft_... paths as needed

# LoRA / LoCon specific paths
lc_model_dir = "C:/ai_stuff/stable-diffusion-webui/models/Stable-diffusion" # Base model for LoRA training
lc_output_dir = "C:/ai_stuff/kohya_ss_outputs/lora"
lc_dataset_dir = "C:/ai_stuff/datasets/my_lora_project"
# Add other lc_... paths as needed

# You can find a comprehensive list of all available keys in the `config example.toml` file.
# Refer to it to customize paths for all supported options in the GUI.
```

**Using a Custom Config File Path:**

If you prefer to name your configuration file differently or store it in another location, you can specify its path using the `--config` command-line argument when launching the GUI:

*   On Windows: `gui.bat --config D:/my_configs/kohya_settings.toml`
*   On Linux/macOS: `./gui.sh --config /home/user/my_configs/kohya_settings.toml`

By effectively using `config.toml`, you can significantly speed up your training setup process. Always refer to the `config example.toml` for the most up-to-date list of configurable paths.

## LoRA

To train a LoRA, you can currently use the `train_network.py` code. You can create a LoRA network by using the all-in-one GUI.

Once you have created the LoRA network, you can generate images using auto1111 by installing [this extension](https://github.com/kohya-ss/sd-webui-additional-networks).

For more detailed information on LoRA training options and advanced configurations, please refer to our LoRA documentation:
- [LoRA Training Guide](docs/LoRA/top_level.md)
- [LoRA Training Options](docs/LoRA/options.md)

## Sample image generation during training

A prompt file might look like this, for example:

```txt
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy, bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy, bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following options can be used:

- `--n`: Negative prompt up to the next option.
- `--w`: Specifies the width of the generated image.
- `--h`: Specifies the height of the generated image.
- `--d`: Specifies the seed of the generated image.
- `--l`: Specifies the CFG scale of the generated image.
- `--s`: Specifies the number of steps in the generation.

The prompt weighting such as `( )` and `[ ]` is working.

## Troubleshooting

If you encounter any issues, refer to the troubleshooting steps below.

### Page File Limit

If you encounter an X error related to the page file, you may need to increase the page file size limit in Windows.

### No module called tkinter

If you encounter an error indicating that the module `tkinter` is not found, try reinstalling Python 3.10 on your system.

### LORA Training on TESLA V100 - GPU Utilization Issue

See [Troubleshooting LORA Training on TESLA V100](docs/troubleshooting_tesla_v100.md) for details.

## SDXL training

For detailed guidance on SDXL training, please refer to the [official sd-scripts documentation](https://github.com/kohya-ss/sd-scripts/blob/main/README.md#sdxl-training) and relevant sections in our [LoRA Training Guide](docs/LoRA/top_level.md).

## Masked loss

The masked loss is supported in each training script. To enable the masked loss, specify the `--masked_loss` option.

> [!WARNING]
> The feature is not fully tested, so there may be bugs. If you find any issues, please open an Issue.

ControlNet dataset is used to specify the mask. The mask images should be the RGB images. The pixel value 255 in R channel is treated as the mask (the loss is calculated only for the pixels with the mask), and 0 is treated as the non-mask. The pixel values 0-255 are converted to 0-1 (i.e., the pixel value 128 is treated as the half weight of the loss). See details for the dataset specification in the [LLLite documentation](./docs/train_lllite_README.md#preparing-the-dataset).

## Guides

The following are guides extracted from issues discussions

### Using Accelerate Lora Tab to Select GPU ID

#### Starting Accelerate in GUI

- Open the kohya GUI on your desired port.
- Open the `Accelerate launch` tab
- Ensure the Multi-GPU checkbox is unchecked.
- Set GPU IDs to the desired GPU (like 1).

#### Running Multiple Instances (linux)

- For tracking multiple processes, use separate kohya GUI instances on different ports (e.g., 7860, 7861).
- Start instances using `nohup ./gui.sh --listen 0.0.0.0 --server_port <port> --headless > log.log 2>&1 &`.

#### Monitoring Processes

- Open each GUI in a separate browser tab.
- For terminal access, use SSH and tools like `tmux` or `screen`.

For more details, visit the [GitHub issue](https://github.com/bmaltais/kohya_ss/issues/2577).

## Interesting Forks

To finetune HunyuanDiT models or create LoRAs, visit this [fork](https://github.com/Tencent/HunyuanDiT/tree/main/kohya_ss-hydit)

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please consider the following:
- For bug reports or feature requests, please open an issue on the [GitHub Issues page](https://github.com/bmaltais/kohya_ss/issues).
- If you'd like to submit code changes, please open a pull request. Ensure your changes are well-tested and follow the existing code style.
- For security-related concerns, please refer to our `SECURITY.md` file.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE.md](LICENSE.md) file for details.

## Change History

### v25.0.3

- Upgrade Gradio, diffusers and huggingface-hub to latest release to fix issue with ASGI.
- Add a new method to setup and run the GUI. You will find two new script for both Windows (gui-uv.bat) and Linux (gui-uv.sh). With those scripts there is no need to run setup.bat or setup.sh anymore.

### v25.0.2

- Force gradio to 5.14.0 or greater so it is updated.

### v25.0.1

- Fix issue with requirements version causing huggingface download issues

### v25.0.0

- Major update: Introduced support for flux.1 and sd3, moving the GUI to align with more recent script functionalities.
- Users preferring the pre-flux.1/sd3 version can check out tag `v24.1.7`.
  ```shell
  git checkout v24.1.7
  ```
- For details on new flux.1 and sd3 parameters, refer to the [sd-scripts README](https://github.com/kohya-ss/sd-scripts/blob/sd3/README.md).
