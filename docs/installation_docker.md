### Docker

#### Get your Docker ready for GPU support

##### Windows

Once you have installed [**Docker Desktop**](https://www.docker.com/products/docker-desktop/), [**CUDA Toolkit**](https://developer.nvidia.com/cuda-downloads), [**NVIDIA Windows Driver**](https://www.nvidia.com.tw/Download/index.aspx), and ensured that your Docker is running with [**WSL2**](https://docs.docker.com/desktop/wsl/#turn-on-docker-desktop-wsl-2), you are ready to go.

Here is the official documentation for further reference.  
<https://docs.nvidia.com/cuda/wsl-user-guide/index.html#nvidia-compute-software-support-on-wsl-2>
<https://docs.docker.com/desktop/wsl/use-wsl/#gpu-support>

##### Linux, OSX

Install an NVIDIA GPU Driver if you do not already have one installed.  
<https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>

Install the NVIDIA Container Toolkit with this guide.  
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

#### Design of our Dockerfile

- It is required that all training data is stored in the `dataset` subdirectory, which is mounted into the container at `/dataset`.
- Please note that the file picker functionality is not available. Instead, you will need to manually input the folder path and configuration file path.
- TensorBoard has been separated from the project.
  - TensorBoard is not included in the Docker image.
  - The "Start TensorBoard" button has been hidden.
  - TensorBoard is launched from a distinct container [as shown here](/docker-compose.yaml#L41).
- The browser won't be launched automatically. You will need to manually open the browser and navigate to [http://localhost:7860/](http://localhost:7860/) and [http://localhost:6006/](http://localhost:6006/)
- This Dockerfile has been designed to be easily disposable. You can discard the container at any time and restart it with the new code version.

#### Use the pre-built Docker image

```bash
git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
docker compose up -d
```

To update the system, do `docker compose down && docker compose up -d --pull always`

#### Local docker build

> [!IMPORTANT]  
> Clone the Git repository ***recursively*** to include submodules:  
> `git clone --recursive https://github.com/bmaltais/kohya_ss.git`

```bash
git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
docker compose up -d --build
```

> [!NOTE]  
> Building the image may take up to 20 minutes to complete.

To update the system, ***checkout to the new code version*** and rebuild using `docker compose down && docker compose up -d --build --pull always`

> [!NOTE]
> If you are running on Linux, an alternative Docker container port with fewer limitations is available [here](https://github.com/P2Enjoy/kohya_ss-docker).

#### ashleykleynhans runpod docker builds

You may want to use the following repositories when running on runpod:

- Standalone Kohya_ss template: <https://github.com/ashleykleynhans/kohya-docker>
- Auto1111 + Kohya_ss GUI template: <https://github.com/ashleykleynhans/stable-diffusion-docker>
