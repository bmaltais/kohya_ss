# Kohya's GUI

[![GitHub stars](https://img.shields.io/github/stars/bmaltais/kohya_ss?style=social)](https://github.com/bmaltais/kohya_ss/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/bmaltais/kohya_ss?style=social)](https://github.com/bmaltais/kohya_ss/network/members)
[![License](https://img.shields.io/github/license/bmaltais/kohya_ss)](LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/bmaltais/kohya_ss)](https://github.com/bmaltais/kohya_ss/issues)

A comprehensive GUI and CLI toolkit for training Stable Diffusion models, LoRAs, and other diffusion model variants.

## Overview

This project provides a user-friendly **Gradio-based interface** for [Kohya's Stable Diffusion training scripts](https://github.com/kohya-ss/sd-scripts), making it accessible for both beginners and advanced users to fine-tune diffusion models.

**Key Features:**
- **Easy-to-use GUI** for configuring training parameters
- **Automatic CLI command generation** for advanced users
- **Multiple training methods**: LoRA, Dreambooth, Fine-tuning, SDXL, Flux.1, SD3
- **Cross-platform support**: Windows, Linux, macOS
- **Flexible deployment**: Local installation, Docker, or cloud-based

## Table of Contents

- [Quick Start](#quick-start)
- [Installation Options](#installation-options)
  - [Local Installation](#local-installation)
  - [Docker Installation](#docker-installation)
  - [Cloud-Based Solutions](#cloud-based-solutions)
- [Configuration](#configuration)
- [Training Features](#training-features)
  - [LoRA Training](#lora-training)
  - [SDXL Training](#sdxl-training)
  - [Sample Image Generation](#sample-image-generation)
  - [Masked Loss](#masked-loss)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)
- [Change History](#change-history)

## Quick Start

Choose your preferred installation method:

| Method | Best For | Time to Setup |
|--------|----------|---------------|
| **Docker** | Quick start, consistency across systems | 5-10 minutes |
| **uv (Recommended)** | Latest features, faster dependency management | 10-15 minutes |
| **pip** | Traditional Python users, easier debugging | 15-20 minutes |
| **Cloud (Colab)** | No local GPU, testing, or limited resources | 2-5 minutes |

**Fastest way to get started:**

```bash
# Docker (if you have Docker + NVIDIA GPU)
git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
docker compose up -d
# Access GUI at http://localhost:7860

# OR Local installation with uv (Linux/Windows)
git clone https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
# See installation guides below for platform-specific steps
```

## Installation Options

### Local Installation

Install `kohya_ss` directly on your machine for maximum flexibility and performance.

#### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **RAM**: 16GB minimum (32GB recommended for SDXL)
- **Storage**: 20GB+ free space
- **Python**: 3.10 or 3.11 (3.12 not yet supported)

#### Installation Methods

| Platform     | Recommended | Alternative | Installation Guide |
|--------------|-------------|-------------|-------------------|
| **Windows**  | uv | pip | [uv_windows.md](./docs/Installation/uv_windows.md) / [pip_windows.md](./docs/Installation/pip_windows.md) |
| **Linux**    | uv | pip | [uv_linux.md](./docs/Installation/uv_linux.md) / [pip_linux.md](./docs/Installation/pip_linux.md) |
| **macOS**    | pip | uv | [pip_linux.md](./docs/Installation/pip_linux.md) |

#### `uv` vs `pip` - Which Should I Choose?

**Use `uv` if:**
- You want the fastest installation and updates
- You prefer automatic dependency isolation
- You're setting up a new environment
- You want minimal configuration hassle

**Use `pip` if:**
- You're experienced with Python package management
- You need fine-grained control over dependencies
- You're integrating with existing Python tooling
- You encounter issues with `uv`

**Still unsure?** Start with `uv`. If you encounter problems, fall back to `pip`.

### Docker Installation

**Best for:** Consistent environment, easy updates, isolation from system Python.

Docker provides the fastest and most reliable way to run Kohya_ss with all dependencies pre-configured.

#### Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit (Linux) or WSL2 with GPU support (Windows)

#### Quick Start with Docker

```bash
# Clone repository with submodules
git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss

# Start services
docker compose up -d

# Access the GUI
# Kohya GUI: http://localhost:7860
# TensorBoard: http://localhost:6006
```

#### Updating Docker Installation

```bash
# Stop containers
docker compose down

# Pull latest images and restart
docker compose up -d --pull always
```

**Complete Docker documentation:** [docs/docker.md](./docs/docker.md)

**Platform-specific setup:**
- **Windows**: [Docker Desktop + WSL2 GPU Setup](./docs/docker.md#windows)
- **Linux**: [NVIDIA Container Toolkit Setup](./docs/docker.md#linux)
- **macOS**: Docker does not support NVIDIA GPUs (use cloud or native installation)

### Cloud-Based Solutions

No local GPU? Use these cloud alternatives:

#### Google Colab (Free)

**Pros:** Free GPU access, no installation required, browser-based
**Cons:** Session limits, may disconnect, shared resources

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/kohya_ss-colab/blob/main/kohya_ss_colab.ipynb)

- **Repository:** <https://github.com/camenduru/kohya_ss-colab>
- **Maintained by:** camenduru (community contributor)
- **Note:** Report Colab-specific issues to camenduru's repository

**Special thanks to camenduru for maintaining the Colab version!**

#### RunPod (Paid)

**Pros:** Dedicated GPUs, persistent storage, no session limits
**Cons:** Costs money, requires account setup

- **Setup Guide:** [docs/installation_runpod.md](docs/installation_runpod.md)
- **Templates available** with pre-configured environments

#### Novita (Paid)

**Pros:** Integrated UI, easy setup, good for beginners
**Cons:** Costs money, platform-specific

- **Setup Guide:** [docs/installation_novita.md](docs/installation_novita.md)

## Configuration

### Custom Path Defaults with `config.toml`

Streamline your workflow by setting default paths for models, datasets, and outputs.

#### Quick Setup

1. **Copy the example configuration:**
   ```bash
   cp "config example.toml" config.toml
   ```

2. **Edit `config.toml`** with your preferred paths:
   ```toml
   # Example configuration
   model_dir = "C:/ai/models/Stable-diffusion"
   lora_model_dir = "C:/ai/models/Lora"
   output_dir = "C:/ai/outputs"
   dataset_dir = "C:/ai/datasets"
   ```

3. **Use absolute paths** or paths relative to the kohya_ss root directory

4. **Use forward slashes** (/) even on Windows for compatibility

#### Configuration Structure

The `config.toml` file supports multiple sections for different training modes:

```toml
# General settings
model_dir = "/path/to/models"
lora_model_dir = "/path/to/lora"
vae_dir = "/path/to/vae"
output_dir = "/path/to/outputs"
logging_dir = "/path/to/logs"

# Dreambooth specific
db_model_dir = "/path/to/models"
db_reg_image_dir = "/path/to/regularization"

# LoRA specific
lc_model_dir = "/path/to/models"
lc_output_dir = "/path/to/outputs/lora"
lc_dataset_dir = "/path/to/datasets"

# See 'config example.toml' for complete list of options
```

#### Using Custom Config Path

Specify a different config file location:

```bash
# Windows
gui.bat --config D:/my_configs/kohya_settings.toml

# Linux/macOS
./gui.sh --config /home/user/my_configs/kohya_settings.toml
```

**Full configuration reference:** See `config example.toml` in the root directory

## Training Features

### LoRA Training

LoRA (Low-Rank Adaptation) allows efficient fine-tuning of Stable Diffusion models with minimal computational requirements.

**Training a LoRA:**
1. Use the GUI's LoRA training tab
2. Configure dataset and parameters
3. Start training via `train_network.py`

**Using trained LoRAs:**
- Install [Additional Networks extension](https://github.com/kohya-ss/sd-webui-additional-networks) for Auto1111
- Load LoRA in your preferred Stable Diffusion UI

**Documentation:**
- [LoRA Training Guide](docs/LoRA/top_level.md) - Comprehensive overview
- [LoRA Training Options](docs/LoRA/options.md) - Advanced configuration

### SDXL Training

Support for Stable Diffusion XL model training with optimized settings.

**Resources:**
- [Official SDXL Training Guide](https://github.com/kohya-ss/sd-scripts/blob/main/README.md#sdxl-training)
- [LoRA Training Guide](docs/LoRA/top_level.md) (includes SDXL sections)

### Sample Image Generation

Generate sample images during training to monitor progress and quality.

#### Creating a Prompt File

Create a text file with prompts and generation parameters:

```txt
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy, bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy, bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

#### Available Options

- `--n`: Negative prompt (text to avoid)
- `--w`: Image width in pixels
- `--h`: Image height in pixels
- `--d`: Seed for reproducibility
- `--l`: CFG scale (guidance strength)
- `--s`: Number of sampling steps

**Note:** Prompt weighting with `()` and `[]` is supported.

### Masked Loss

Enable masked loss to train only specific regions of images.

**Activation:** Add `--masked_loss` option in training configuration

**How it works:**
- Uses ControlNet dataset format
- RGB mask images where Red channel value determines weight
  - 255 (full weight) = train this area
  - 0 (no weight) = ignore this area
  - 128 (half weight) = partial training
- Pixel values 0-255 map to loss weights 0.0-1.0

**Documentation:** [LLLite Training Guide](./docs/train_lllite_README.md#preparing-the-dataset)

**Warning:** This feature is experimental. Please report issues on GitHub.

## Troubleshooting

### Common Issues

#### Page File Limit (Windows)

**Symptom:** Error about page file size

**Solution:** Increase Windows virtual memory (page file) size:
1. System Properties > Advanced > Performance Settings
2. Virtual Memory > Change
3. Set custom size (16GB+ recommended)

#### No module called 'tkinter'

**Symptom:** Import error for tkinter module

**Solutions:**
- **Windows:** Reinstall Python 3.10 or 3.11 with "tcl/tk" option enabled
- **Linux:** `sudo apt-get install python3-tk`
- **macOS:** Reinstall Python from python.org (not Homebrew)

#### GPU Not Being Used / Low GPU Utilization

**Symptoms:** Training is slow, GPU usage at 0-10%

**Solutions:**
1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch GPU access:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```
3. Increase batch size
4. Disable CPU offloading options
5. See: [Tesla V100 Troubleshooting](docs/troubleshooting_tesla_v100.md)

#### Out of Memory Errors

**Solutions:**
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision training (fp16)
- Lower resolution
- Enable CPU offloading
- Close other GPU applications

#### Docker-Specific Issues

See the comprehensive [Docker Troubleshooting Guide](./docs/docker.md#troubleshooting) for:
- GPU not detected in container
- Permission denied errors
- Volume mount issues
- Port conflicts

### Getting Help

If you're stuck:

1. **Search existing issues:** <https://github.com/bmaltais/kohya_ss/issues>
2. **Check documentation:** See `/docs` directory
3. **Open a new issue** with:
   - Operating system and version
   - Installation method (Docker/uv/pip)
   - Python version
   - Full error message and logs
   - Steps to reproduce

## Advanced Usage

### Accelerate Configuration for Multi-GPU

Use the Accelerate tab in the GUI to configure multi-GPU training:

1. Open the "Accelerate launch" tab
2. For single GPU: Uncheck "Multi-GPU", set GPU ID (e.g., "0" or "1")
3. For multi-GPU: Check "Multi-GPU", configure device IDs

#### Running Multiple Instances (Linux)

Run separate GUI instances for different training jobs:

```bash
# Start first instance on port 7860
nohup ./gui.sh --listen 0.0.0.0 --server_port 7860 --headless > log_7860.log 2>&1 &

# Start second instance on port 7861
nohup ./gui.sh --listen 0.0.0.0 --server_port 7861 --headless > log_7861.log 2>&1 &
```

**Monitoring:** Use `tmux` or `screen` for terminal management

**More details:** [GitHub Issue #2577](https://github.com/bmaltais/kohya_ss/issues/2577)

### Command-Line Usage

The GUI generates CLI commands that can be run directly:

```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows

# Run training script directly
python sd-scripts/train_network.py \
  --pretrained_model_name_or_path=/path/to/model.safetensors \
  --train_data_dir=/path/to/dataset \
  --output_dir=/path/to/output \
  # ... additional parameters
```

### Using Different Python Versions

Kohya_ss supports Python 3.10 and 3.11:

```bash
# Create environment with specific version
uv venv --python 3.11
# or
python3.11 -m venv venv
```

## Interesting Forks

Community-maintained variants with additional features:

- **HunyuanDiT Support:** Fine-tune HunyuanDiT models
  - Repository: <https://github.com/Tencent/HunyuanDiT/tree/main/kohya_ss-hydit>

## Contributing

Contributions are welcome! Help improve Kohya_ss by:

**Reporting Issues:**
- Use [GitHub Issues](https://github.com/bmaltais/kohya_ss/issues)
- Include detailed reproduction steps
- Provide system information and logs

**Submitting Code:**
- Fork the repository
- Create a feature branch
- Follow existing code style
- Test thoroughly before submitting PR
- Document new features

**Security Issues:**
- See [SECURITY.md](SECURITY.md) for responsible disclosure

## License

This project is licensed under the **Apache License 2.0**.

See [LICENSE.md](LICENSE.md) for complete terms.

## Change History

### v25.2.1 (Current)

- Latest stable release
- Python 3.11 support
- Updated dependencies

### v25.0.3

- Upgraded Gradio, diffusers, and huggingface-hub to fix ASGI issues
- New simplified setup scripts:
  - `gui-uv.bat` (Windows) and `gui-uv.sh` (Linux)
  - No need to run separate setup scripts anymore

### v25.0.2

- Forced Gradio upgrade to 5.14.0+ for critical updates

### v25.0.1

- Fixed requirements versioning issues affecting Hugging Face downloads

### v25.0.0

- **Major update:** Added support for Flux.1 and SD3
- Aligned GUI with latest sd-scripts features
- Breaking changes: Previous workflows may need adjustment

**Note:** For pre-Flux.1/SD3 version, checkout tag `v24.1.7`:
```bash
git checkout v24.1.7
```

**Flux.1 and SD3 Parameters:**
- See [sd-scripts README](https://github.com/kohya-ss/sd-scripts/blob/sd3/README.md)

### Older Versions

For complete version history, see [GitHub Releases](https://github.com/bmaltais/kohya_ss/releases).

---

## Quick Reference

### Important Links

- **Main Repository:** <https://github.com/bmaltais/kohya_ss>
- **SD-Scripts (Core Training):** <https://github.com/kohya-ss/sd-scripts>
- **Issues & Support:** <https://github.com/bmaltais/kohya_ss/issues>
- **Colab Version:** <https://github.com/camenduru/kohya_ss-colab>

### Default Ports

- **Kohya GUI:** 7860
- **TensorBoard:** 6006

### File Locations

- **Config:** `config.toml` (root directory)
- **Training Scripts:** `sd-scripts/` (submodule)
- **Documentation:** `docs/`
- **Examples:** `examples/`

### Supported Models

- Stable Diffusion 1.x, 2.x
- Stable Diffusion XL (SDXL)
- Stable Diffusion 3 (SD3)
- Flux.1
- Custom fine-tuned models

### Training Methods

- LoRA (Low-Rank Adaptation)
- Dreambooth
- Fine-tuning
- Textual Inversion
- LLLite

---

**Need help?** Check the [documentation](./docs/) or open an [issue](https://github.com/bmaltais/kohya_ss/issues)!
