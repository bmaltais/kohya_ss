# syntax=docker/dockerfile:1

ARG UID=1000
ARG VERSION=EDGE
ARG RELEASE=0

########################################
# Base stage
########################################
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04 AS base

ARG TARGETARCH
ARG TARGETVARIANT

WORKDIR /tmp

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python 3.11
RUN apt-get update && \
    apt-get install -y software-properties-common curl gnupg && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

########################################
# Build stage
########################################
FROM base AS build

ARG TARGETARCH
ARG TARGETVARIANT

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/venv
ENV VIRTUAL_ENV=/venv
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_INDEX=https://download.pytorch.org/whl/cu126

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-launchpadlib git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install large dependencies first
RUN --mount=type=cache,id=uv-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/root/.cache/uv \
    uv venv --system-site-packages /venv && \
    uv pip install --no-deps \
    torch==2.7.0+cu126 \
    triton>=3.1.0 \
    tensorflow>=2.16.1 \
    onnxruntime-gpu==1.19.2

# Sync project dependencies
RUN --mount=type=cache,id=uv-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=sd-scripts,target=sd-scripts,rw \
    uv sync --frozen --no-dev --no-install-project --no-editable

# Replace pillow with pillow-simd (Only on x86_64)
ARG TARGETPLATFORM
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
    zlib1g-dev libjpeg62-turbo-dev build-essential && \
    uv pip uninstall pillow && \
    CC="cc -mavx2" uv pip install pillow-simd && \
    apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

########################################
# Final stage
########################################
FROM base AS final

ARG TARGETARCH
ARG TARGETVARIANT

WORKDIR /tmp

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libjpeg62 libtcl8.6 libtk8.6 \
    libgoogle-perftools-dev dumb-init && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix missing libnvinfer7 links
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer.so.7 || true && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7 || true

# Create user
ARG UID
RUN groupadd -g $UID $UID && \
    useradd -l -u $UID -g $UID -m -s /bin/sh -N $UID

# Create necessary directories
RUN install -d -m 775 -o $UID -g 0 /dataset && \
    install -d -m 775 -o $UID -g 0 /licenses && \
    install -d -m 775 -o $UID -g 0 /app && \
    install -d -m 775 -o $UID -g 0 /venv

# Copy licenses
COPY --link --chmod=775 LICENSE.md /licenses/LICENSE.md

# Copy venv and app files
COPY --link --chown=$UID:0 --chmod=775 --from=build /venv /venv
COPY --link --chown=$UID:0 --chmod=775 . /app

# Environment setup
ENV PATH="/venv/bin${PATH:+:${PATH}}"
ENV PYTHONPATH="/venv/lib/python3.11/site-packages"
ENV LD_LIBRARY_PATH="/venv/lib/python3.11/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
ENV LD_PRELOAD=libtcmalloc.so
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV FORCE_COLOR="true"
ENV COLUMNS="100"

WORKDIR /app

VOLUME [ "/dataset" ]

EXPOSE 7860

USER $UID
STOPSIGNAL SIGINT

ENTRYPOINT ["dumb-init", "--"]
CMD ["python3", "kohya_gui.py", "--listen", "0.0.0.0", "--server_port", "7860", "--headless", "--noverify"]

ARG VERSION
ARG RELEASE
LABEL name="bmaltais/kohya_ss" \
    vendor="bmaltais" \
    maintainer="bmaltais" \
    url="https://github.com/bmaltais/kohya_ss" \
    version=${VERSION} \
    release=${RELEASE} \
    io.k8s.display-name="kohya_ss" \
    summary="Kohya's GUI: Gradio frontend for Stable Diffusion training scripts" \
    description="GUI for setting training parameters and generating CLI commands for Stable Diffusion models. See https://github.com/bmaltais/kohya_ss for details."
