# syntax=docker/dockerfile:1
ARG UID=1000
ARG VERSION=EDGE
ARG RELEASE=0

FROM python:3.10-slim as build

# RUN mount cache for multi-arch: https://github.com/docker/buildx/issues/549#issuecomment-1788297892
ARG TARGETARCH
ARG TARGETVARIANT

WORKDIR /app

# Install under /root/.local
ENV PIP_USER="true"
ARG PIP_NO_WARN_SCRIPT_LOCATION=0
ARG PIP_ROOT_USER_ACTION="ignore"

# Install build dependencies
RUN --mount=type=cache,id=apt-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=aptlists-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/lib/apt/lists \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends python3-launchpadlib git curl

# Install PyTorch
# The versions must align and be in sync with the requirements_linux_docker.txt
# hadolint ignore=SC2102
RUN --mount=type=cache,id=pip-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/root/.cache/pip \
    pip install -U --extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.nvidia.com \
    torch==2.1.2 torchvision==0.16.2 \
    xformers==0.0.23.post1 \
    ninja \
    pip setuptools wheel

# Install requirements
RUN --mount=type=cache,id=pip-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/root/.cache/pip \
    --mount=source=requirements_linux_docker.txt,target=requirements_linux_docker.txt \
    --mount=source=requirements.txt,target=requirements.txt \
    --mount=source=setup/docker_setup.py,target=setup.py \
    --mount=source=sd-scripts,target=sd-scripts,rw \
    pip install -r requirements_linux_docker.txt -r requirements.txt

# Replace pillow with pillow-simd (Only for x86)
ARG TARGETPLATFORM
RUN --mount=type=cache,id=apt-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=aptlists-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/lib/apt/lists \
    if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    apt-get update && apt-get install -y --no-install-recommends zlib1g-dev libjpeg62-turbo-dev build-essential && \
    pip uninstall -y pillow && \
    CC="cc -mavx2" pip install -U --force-reinstall pillow-simd; \
    fi

FROM python:3.10-slim as final

ARG TARGETARCH
ARG TARGETVARIANT

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

WORKDIR /tmp

ENV CUDA_VERSION=12.1.1
ENV NV_CUDA_CUDART_VERSION=12.1.105-1
ENV NVIDIA_REQUIRE_CUDA=cuda>=12.1
ENV NV_CUDA_COMPAT_PACKAGE=cuda-compat-12-1

# Install CUDA partially
ADD https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb .
RUN --mount=type=cache,id=apt-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=aptlists-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/lib/apt/lists \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb && \
    sed -i 's/^Components: main$/& contrib/' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    # Installing the whole CUDA typically increases the image size by approximately **8GB**.
    # To decrease the image size, we opt to install only the necessary libraries.
    # Here is the package list for your reference: https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64
    # !If you experience any related issues, replace the following line with `cuda-12-1` to obtain the complete CUDA package.
    cuda-cudart-12-1=${NV_CUDA_CUDART_VERSION} ${NV_CUDA_COMPAT_PACKAGE} libcusparse-12-1 libnvjitlink-12-1

# Install runtime dependencies
RUN --mount=type=cache,id=apt-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=aptlists-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/lib/apt/lists \
    apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libjpeg62 libtcl8.6 libtk8.6 libgoogle-perftools-dev dumb-init

# Fix missing libnvinfer7
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer.so.7 && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7

# Create user
ARG UID
RUN groupadd -g $UID $UID && \
    useradd -l -u $UID -g $UID -m -s /bin/sh -N $UID

# Create directories with correct permissions
RUN install -d -m 775 -o $UID -g 0 /dataset && \
    install -d -m 775 -o $UID -g 0 /licenses && \
    install -d -m 775 -o $UID -g 0 /app

# Copy licenses (OpenShift Policy)
COPY --link --chmod=775 LICENSE.md /licenses/LICENSE.md

# Copy dependencies and code (and support arbitrary uid for OpenShift best practice)
COPY --link --chown=$UID:0 --chmod=775 --from=build /root/.local /home/$UID/.local
COPY --link --chown=$UID:0 --chmod=775 . /app

ENV PATH="/usr/local/cuda/lib:/usr/local/cuda/lib64:/home/$UID/.local/bin:$PATH"
ENV PYTHONPATH="${PYTHONPATH}:/home/$UID/.local/lib/python3.10/site-packages" 
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LD_PRELOAD=libtcmalloc.so
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Rich logging
# https://rich.readthedocs.io/en/stable/console.html#interactive-mode
ENV FORCE_COLOR="true"
ENV COLUMNS="100"

WORKDIR /app

VOLUME [ "/dataset" ]

# 7860: Kohya GUI
EXPOSE 7860

USER $UID

STOPSIGNAL SIGINT

# Use dumb-init as PID 1 to handle signals properly
ENTRYPOINT ["dumb-init", "--"]
CMD ["python3", "kohya_gui.py", "--listen", "0.0.0.0", "--server_port", "7860", "--headless"]

ARG VERSION
ARG RELEASE
LABEL name="bmaltais/kohya_ss" \
    vendor="bmaltais" \
    maintainer="bmaltais" \
    # Dockerfile source repository
    url="https://github.com/bmaltais/kohya_ss" \
    version=${VERSION} \
    # This should be a number, incremented with each change
    release=${RELEASE} \
    io.k8s.display-name="kohya_ss" \
    summary="Kohya's GUI: This repository provides a Gradio GUI for Kohya's Stable Diffusion trainers(https://github.com/kohya-ss/sd-scripts)." \
    description="The GUI allows you to set the training parameters and generate and run the required CLI commands to train the model. This is the docker image for Kohya's GUI. For more information about this tool, please visit the following website: https://github.com/bmaltais/kohya_ss."