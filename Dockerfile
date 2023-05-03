FROM nvcr.io/nvidia/pytorch:23.04-py3 as base
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

RUN apt-get update && \
    apt-get install -y git curl python3-venv python3-tk libgl1 libglib2.0-0 libgoogle-perftools-dev && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash appuser
USER appuser

WORKDIR /app

RUN python3 -m venv ./venv && . ./venv/bin/activate && \
    python3 -m pip install wheel

# Install requirements
COPY requirements.txt setup.py .
RUN . ./venv/bin/activate && \
    python3 -m pip install --no-cache-dir --use-pep517 -U -r requirements.txt

# Upgrade to Torch 2.0
RUN . ./venv/bin/activate && \
    python3 -m pip install --no-cache-dir --use-pep517 --no-deps -U triton==2.0.0 torch>=2.0.0+cu121 xformers==0.0.17 \
	                       --extra-index-url https://download.pytorch.org/whl/cu121

# Fix missing libnvinfer7
USER root
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer.so.7 && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7

USER appuser
COPY --chown=appuser . .

RUN sed -i 's/import library.huggingface_util/# import library.huggingface_util/g' train_network.py && \
    sed -i 's/import library.huggingface_util/# import library.huggingface_util/g' library/train_util.py

STOPSIGNAL SIGINT
ENV LD_PRELOAD=libtcmalloc.so
CMD . ./venv/bin/activate && python3 "./kohya_gui.py" ${CLI_ARGS} --listen 0.0.0.0 --server_port 7860
