# Docker Setup Guide for Kohya_ss

This guide provides comprehensive instructions for running Kohya_ss in Docker containers.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Prerequisites

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (compute capability 7.0+)
- **RAM**: Minimum 16GB recommended
- **Storage**: At least 50GB free space for models and datasets
- **OS**: Linux, Windows 10/11 with WSL2, or macOS (limited support)

### Required Software

#### Windows

1. **Docker Desktop** (version 4.0+)
   - Download from: <https://www.docker.com/products/docker-desktop/>
   - Ensure WSL2 backend is enabled

2. **NVIDIA CUDA Toolkit**
   - Download from: <https://developer.nvidia.com/cuda-downloads>
   - Version 12.8 or compatible

3. **NVIDIA Windows Driver**
   - Download from: <https://www.nvidia.com/Download/index.aspx>
   - Version 525.60.11 or newer

4. **WSL2 with GPU Support**
   - Enable WSL2: <https://docs.docker.com/desktop/wsl/#turn-on-docker-desktop-wsl-2>
   - Verify GPU support: <https://docs.docker.com/desktop/wsl/use-wsl/#gpu-support>

**Official Documentation:**
- <https://docs.nvidia.com/cuda/wsl-user-guide/index.html#nvidia-compute-software-support-on-wsl-2>

#### Linux

1. **Docker Engine** or **Docker Desktop**
   - Install guide: <https://docs.docker.com/engine/install/>

2. **NVIDIA GPU Driver**
   - Install the latest driver for your GPU
   - Guide: <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>

3. **NVIDIA Container Toolkit**
   - Required for GPU access in containers
   - Install guide: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

#### macOS

Docker on macOS does not support NVIDIA GPU acceleration. For GPU-accelerated training on Mac:
- Use cloud-based solutions (see [Cloud Alternatives](#cloud-alternatives))
- Or install natively using the installation guides in `/docs/Installation/`

## Quick Start

### Using Pre-built Images (Recommended)

This is the fastest way to get started. The images are automatically built and published to GitHub Container Registry.

```bash
# Clone the repository recursively (important!)
git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss

# Start the services
docker compose up -d

# View logs
docker compose logs -f
```

**Access the GUI:**
- Kohya GUI: <http://localhost:7860>
- TensorBoard: <http://localhost:6006>

### Building Locally

If you need to modify the Dockerfile or want to build from source:

```bash
# Clone recursively to include submodules
git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss

# Build and start
docker compose up -d --build
```

**Note:** Initial build may take 15-30 minutes depending on your internet connection and hardware.

## Configuration

### Environment Variables

Create a `.env` file in the root directory to customize settings:

```bash
# .env file example
TENSORBOARD_PORT=6006
UID=1000
```

**Available Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `TENSORBOARD_PORT` | Port for TensorBoard web interface | `6006` |
| `UID` | User ID for file permissions | `1000` |

### User ID Configuration

The `UID` parameter is critical for file permissions. To find your user ID:

```bash
# Linux/macOS/WSL
id -u

# Then set it in docker-compose.yaml or .env
```

If you encounter permission errors, ensure the UID in docker-compose.yaml matches your host user ID.

### Volume Mounts

The Docker setup uses the following directory structure:

```
kohya_ss/
├── dataset/              # Your training datasets
│   ├── images/          # Training images
│   ├── logs/            # TensorBoard logs
│   ├── outputs/         # Trained models output
│   └── regularization/  # Regularization images
├── models/              # Pre-trained models
└── .cache/              # Cache directories
    ├── config/
    ├── user/
    ├── triton/
    ├── nv/
    └── keras/
```

**Important:** All training data must be placed in the `dataset/` directory or its subdirectories.

### Directory Setup

Before first use, ensure these directories exist:

```bash
mkdir -p dataset/images dataset/logs dataset/outputs dataset/regularization
mkdir -p models
mkdir -p .cache/{config,user,triton,nv,keras}
```

## Usage

### Starting the Services

```bash
# Start in detached mode
docker compose up -d

# Start with logs visible
docker compose up

# Start only specific service
docker compose up -d kohya-ss-gui
```

### Stopping the Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (warning: deletes data)
docker compose down -v
```

### Updating

To update to the latest version:

```bash
# Pull latest images
docker compose down
docker compose pull
docker compose up -d

# Or with auto-pull
docker compose down && docker compose up -d --pull always
```

If you're building locally:

```bash
# Update code
git pull
git submodule update --init --recursive

# Rebuild and restart
docker compose down
docker compose up -d --build --pull always
```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f kohya-ss-gui

# Last 100 lines
docker compose logs --tail=100
```

## Troubleshooting

### GPU Not Detected

**Symptoms:** Training is slow, no GPU utilization in `nvidia-smi`

**Solutions:**

1. Verify GPU is visible to Docker:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
   ```

2. Check NVIDIA Container Toolkit:
   ```bash
   # Linux
   nvidia-ctk --version

   # If not installed, see prerequisites
   ```

3. Windows WSL2 users:
   - Ensure Docker Desktop is using WSL2 backend
   - Verify CUDA is working in WSL: `nvidia-smi` in WSL terminal

### Permission Denied Errors

**Symptoms:** Cannot read/write files in mounted volumes

**Solutions:**

1. Check your user ID:
   ```bash
   id -u
   ```

2. Update docker-compose.yaml:
   ```yaml
   services:
     kohya-ss-gui:
       user: YOUR_UID:0  # Replace YOUR_UID with actual UID
       build:
         args:
           - UID=YOUR_UID  # Same here
   ```

3. Fix ownership of existing files:
   ```bash
   sudo chown -R YOUR_UID:YOUR_UID dataset/ models/ .cache/
   ```

### Out of Memory Errors

**Symptoms:** Container crashes, training fails with OOM

**Solutions:**

1. Add memory limits to docker-compose.yaml:
   ```yaml
   services:
     kohya-ss-gui:
       deploy:
         resources:
           limits:
             memory: 32G  # Adjust based on your system
   ```

2. Reduce batch size in training parameters
3. Use gradient checkpointing
4. Enable CPU offloading in training settings

### Container Won't Start

**Symptoms:** Container exits immediately or shows errors

**Solutions:**

1. Check logs:
   ```bash
   docker compose logs kohya-ss-gui
   ```

2. Verify all submodules are cloned:
   ```bash
   git submodule update --init --recursive
   ```

3. Remove old containers and images:
   ```bash
   docker compose down
   docker system prune -a
   docker compose up -d --build
   ```

### File Picker Not Working

**Note:** This is a known limitation of the Docker setup.

**Workaround:** Manually type the full path instead of using the file picker. Paths should be relative to `/app` or `/dataset`:

Examples:
- Training images: `/dataset/images/my_dataset`
- Model output: `/dataset/outputs/my_model`
- Pretrained model: `/app/models/sd_xl_base_1.0.safetensors`

### TensorBoard Not Accessible

**Symptoms:** Cannot access TensorBoard at localhost:6006

**Solutions:**

1. Check if container is running:
   ```bash
   docker compose ps
   ```

2. Verify logs are being written:
   ```bash
   ls -la dataset/logs/
   ```

3. Check port conflicts:
   ```bash
   # Linux/macOS
   sudo lsof -i :6006

   # Windows PowerShell
   netstat -ano | findstr :6006
   ```

4. Change port in .env file if needed:
   ```bash
   echo "TENSORBOARD_PORT=6007" > .env
   docker compose down && docker compose up -d
   ```

## Advanced Configuration

### Custom CUDA Version

If you need a different CUDA version, modify the Dockerfile:

```dockerfile
# Line 39-40
ENV CUDA_VERSION=12.8
ENV NVIDIA_REQUIRE_CUDA=cuda>=12.8

# Line 61
ENV UV_INDEX=https://download.pytorch.org/whl/cu128
```

### Resource Limits

Add resource limits to prevent container from consuming all system resources:

```yaml
# docker-compose.yaml
services:
  kohya-ss-gui:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          cpus: '4'
          memory: 16G
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ["0"]  # Specific GPU
```

### Multiple GPU Setup

To use specific GPUs:

```yaml
# Use GPU 0 and 1
device_ids: ["0", "1"]

# Use all GPUs
device_ids: ["all"]
```

In the container, you can also use `CUDA_VISIBLE_DEVICES`:

```yaml
environment:
  CUDA_VISIBLE_DEVICES: "0,1"
```

### Restart Policies

Add automatic restart on failure:

```yaml
services:
  kohya-ss-gui:
    restart: unless-stopped
  tensorboard:
    restart: unless-stopped
```

### Using Different Base Images

For development or debugging, you can switch base images:

```dockerfile
# Use full CUDA toolkit instead of minimal
FROM docker.io/nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base
```

## Docker Design Philosophy

This Docker setup follows these principles:

1. **Disposable Containers**: Containers can be destroyed and recreated at any time. All important data is stored in mounted volumes.

2. **Data Separation**: Training data, models, and outputs are kept outside the container in the `dataset/` directory.

3. **No Built-in File Picker**: Due to container isolation, the GUI file picker is disabled. Use manual path entry instead.

4. **Separate TensorBoard**: TensorBoard runs in its own container for better resource isolation and easier updates.

5. **Minimal Image Size**: Only essential CUDA libraries are included to reduce image size from ~8GB to ~3GB.

## Cloud Alternatives

If Docker on your local machine isn't suitable:

- **RunPod**: See [docs/installation_runpod.md](installation_runpod.md)
- **Novita**: See [docs/installation_novita.md](installation_novita.md)
- **Colab**: See [README.md](../README.md#-colab) for free cloud-based option

## Community Docker Builds

Alternative Docker implementations with different features:

- **P2Enjoy's Linux-optimized build**: <https://github.com/P2Enjoy/kohya_ss-docker>
  - Fewer limitations on Linux
  - Different architecture

- **Ashley Kleynhans' RunPod templates**:
  - Standalone: <https://github.com/ashleykleynhans/kohya-docker>
  - With Auto1111: <https://github.com/ashleykleynhans/stable-diffusion-docker>

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review container logs: `docker compose logs`
3. Search existing issues: <https://github.com/bmaltais/kohya_ss/issues>
4. Open a new issue with:
   - Your OS and Docker version
   - Complete error logs
   - Steps to reproduce

## Performance Tips

1. **Use SSD storage** for dataset and model directories
2. **Increase Docker memory limit** in Docker Desktop settings (Windows/macOS)
3. **Use tmpfs for temporary files** (already configured in docker-compose.yaml)
4. **Enable BuildKit** for faster builds:
   ```bash
   export DOCKER_BUILDKIT=1
   ```
5. **Use pillow-simd** (automatically enabled on x86_64 in Dockerfile)

## Security Notes

1. The container runs as a non-root user (UID 1000 by default)
2. Only necessary ports are exposed
3. Sensitive data should not be included in the image build
4. Use `.dockerignore` to exclude credentials and secrets
5. Keep base images updated for security patches
