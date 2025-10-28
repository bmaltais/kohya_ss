# Python 3.11 + PyTorch 2.7.1 + CUDA 12.8 + CuDNN 9.5
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# 기본 작업 경로 설정
WORKDIR /app

# 필수 패키지 설치
RUN sed -i 's|archive.ubuntu.com|mirror.kakao.com|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y apt-utils && \
    apt-get install -y --no-install-recommends git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Python 패키지 캐싱 방지
ENV PIP_NO_CACHE_DIR=1

# pip 업그레이드 및 공통 유틸 설치
RUN pip install --upgrade pip setuptools wheel

# kohya_ss 전체 복사 (모델 포함)
COPY . /app/sdxl_train_captioner
# requirements.txt 설치
WORKDIR /app/sdxl_train_captioner

# 2. xformers
RUN pip install xformers==0.0.31

RUN pip install --no-cache-dir -r requirements.txt
# 문제 발생 시 버전 고정: ==2.7.4.post1
RUN pip install flash-attn --no-build-isolation 

RUN mkdir -p /app/sdxl_train_captioner/dataset
RUN mkdir -p /app/sdxl_train_captioner/models

# 모델 파일 복사 (미리 포함시킬 가중치)
COPY ./models /app/sdxl_train_captioner/models

WORKDIR /app/sdxl_train_captioner/sd-scripts

RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["bash", "entrypoint.sh"]
