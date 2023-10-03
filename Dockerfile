FROM nvcr.io/nvidia/pytorch:23.09-py3 as base
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

RUN apt update && apt-get install --no-install-recommends -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt-get install --no-install-recommends -y git curl libgl1 libglib2.0-0 libgoogle-perftools-dev \
    python3.11-dev python3.11-tk python3-html5lib python3-apt python3-pip python3.11-distutils && \
    rm -rf /var/lib/apt/lists/*

# Set python 3.11 and cuda 12.2 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 3 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    update-alternatives --set cuda /usr/local/cuda-12.2

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

WORKDIR /app
RUN python3 -m pip install wheel

COPY ./requirements.txt ./requirements_linux_docker.txt ./
COPY ./setup/docker_setup.py ./setup.py

# Install requirements and Replace pillow with pillow-simd
RUN python3 -m pip install -r ./requirements_linux_docker.txt && \
    python3 -m pip install -r ./requirements.txt && \
    python3 -m pip uninstall -y pillow && \
    CC="cc -mavx2" python3 -m pip install -U --force-reinstall pillow-simd

# Fix missing libnvinfer7
USER root
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer.so.7 && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7

RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser: /app
USER appuser
COPY --chown=appuser . .

STOPSIGNAL SIGINT
ENV LD_PRELOAD=libtcmalloc.so
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PATH="$PATH:/home/appuser/.local/bin:/usr/local/cuda-12/bin"
ENV CUDA_HOME=/usr/local/cuda-12
CMD python3 "./kohya_gui.py" ${CLI_ARGS} --listen 0.0.0.0 --server_port 7860
