NVIDIA Studio Driver(SDR) : Windows 10/11 → 531.79 / 536.67 등 :
아래서 기종 선택하고 Studio Driver 선택하고 검색 버튼
https://www.nvidia.com/ko-kr/geforce/drivers/
제일 낮은 버전이 아마 괜찮을 듯 함.

## 1. 호환 버전
CUDA 12.4 : https://developer.nvidia.com/cuda-12-4-0-download-archive

CcuDNN v9.5.0 : https://developer.nvidia.com/cudnn-9-5-0-download-archive

## 2. CuDNN 설치
cuDNN (예) C:\Program Files\NVIDIA\CUDNN\v9.5\bin 폴더 안에는 Cuda Major 버전에 대응되는 라이브러리들이 있습니다.
해당폴더 하위의 파일들을 CUDA Toolkit이 설치된 경로 내의 해당 폴더에 복사합니다.

예시: 
C:\Program Files\NVIDIA\CUDNN\v9.5\bin\12.6 아래의 모든 dll 파일을 
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin으로 복사합니다. 

[README.md](../sdxl_train_captioner_runtime/README.md)
## 3. SDXL 모델 다운로드
- 도커 컨테이너가 실행될 때 models 하위에 StableDiffusion XL 1.0 모델이 다운로드 됩니다.
- 만약에 해당 URL 지원이 종료 된 경우, 허깅페이지 또는 CIVITAI에서 다운로드 하세요.
- 현재 사용가능한 다운로드 주소는 아래와 같습니다.
- https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

## 학습방법 1: 폴더명 규칙 사용 (자동)
./train.sh config.json 0

# 방법 2: 강제로 15번 반복
./train.sh config.json 0 15

# 방법 3: 강제로 20번 반복
./train.sh config.json 0 20


3가지 요소 비교
1️⃣ 폴더 식별자 (예: 15_alice)
목적: Kohya 학습 시스템이 이미지를 분류하고 관리하는 용도
training/
├── 15_alice/          ← "alice"는 내부 식별용
│   ├── img1.jpg
│   └── img1.txt
└── 10_background/     ← "background"는 내부 식별용
    ├── bg1.jpg
    └── bg1.txt
특징:

학습 시 로그에만 표시됨
LoRA 모델이나 trigger word와 무관
단순히 폴더 구분용

2️⃣ --output_name (예: karina)
목적: 저장되는 LoRA 파일명

3️⃣ LoRA 태그명
- 학습에 사용되는 캡션 tag + 문장에서 가장 많이 발견되는 Unique Word가 태그명이 됩니다.
- 일반적으로 캡션의 제일 앞에 배치하고 그 뒤에 콤마를 찍고 나머지를 서술합니다.




이 저장소에는 Stable Diffusion용 훈련, 생성 및 유틸리티 스크립트가 포함되어 있습니다.

변경 내역은 페이지 하단으로 이동했습니다.

최신 업데이트: 2025-03-21 (버전 0.9.1)

일본어판 README는 여기

개발 버전은 dev 브랜치에 있습니다. 최신 변경 사항은 dev 브랜치를 확인해 주세요.

FLUX.1 및 SD3/SD3.5 지원은 sd3 브랜치에서 이루어집니다. 해당 모델을 훈련하려면 sd3 브랜치를 사용해 주세요.

더 쉬운 사용법(GUI 및 PowerShell 스크립트 등)을 원하시면 bmaltais가 관리하는 저장소를 방문해 주세요. @bmaltais 님께 감사드립니다!

이 저장소에는 다음 스크립트가 포함되어 있습니다:

- DreamBooth 훈련 (U-Net 및 텍스트 인코더 포함)
- 미세 조정 (네이티브 훈련) (U-Net 및 텍스트 인코더 포함)
- LoRA 훈련
- 텍스트 역전 훈련
- 이미지 생성
- 모델 변환 (1.x 및 2.x, Stable Diffusion ckpt/safetensors 및 Diffusers 지원)


## requirements.txt 파일 안내

이 파일에는 PyTorch 요구 사항이 포함되어 있지 않습니다. PyTorch 버전은 환경에 따라 달라지므로 별도로 관리됩니다. 먼저 환경에 맞는 PyTorch를 설치해 주세요. 설치 방법은 아래를 참고하세요.

스크립트는 PyTorch 2.1.2로 테스트되었습니다. PyTorch 2.2 이상도 작동합니다. 적절한 버전의 PyTorch와 xformers를 설치해 주세요.

## 사용법 문서 링크

대부분의 문서는 일본어로 작성되었습니다.

[darkstorm2150님의 영어 번역본은 여기](https://github.com/darkstorm2150/sd-scripts#links-to-usage-documentation)에서 확인하실 수 있습니다. darkstorm2150님께 감사드립니다!

* [훈련 가이드 - 공통](sd-scripts/docs/train_README-ja.md) : 데이터 준비, 옵션 등...
  * [중국어 버전](sd-scripts/docs/train_README-zh.md)
* [SDXL 훈련](sd-scripts/docs/train_SDXL-en.md) (영어 버전)
* [데이터셋 구성](sd-scripts/docs/config_README-ja.md)
  * [영어 버전](sd-scripts/docs/config_README-en.md)
* [DreamBooth 훈련 가이드](sd-scripts/docs/train_db_README-ja.md)
* [단계별 미세 조정 가이드](sd-scripts/docs/fine_tune_README_ja.md):
* [LoRA 훈련](sd-scripts/docs/train_network_README-ja.md)
* [텍스트 역전 훈련](sd-scripts/docs/train_ti_README-ja.md)
* [이미지 생성](sd-scripts/docs/gen_img_README-ja.md)
* note.com [모델 변환](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windows Required Dependencies

## Windows 필수 종속성

Python 3.10.6 및 Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

Python 3.10.x, 3.11.x, 3.12.x도 작동하지만 테스트되지 않았습니다.

venv가 작동하도록 PowerShell에 제한 없는 스크립트 실행 권한 부여:

- 관리자 권한 PowerShell 창 열기
- `Set-ExecutionPolicy Unrestricted` 입력 후 A 선택
- 관리자 권한 PowerShell 창 닫기

## Windows 설치

일반 PowerShell 터미널을 열고 다음 명령어를 입력하세요:

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

accelerate config
```

`python -m venv` 명령어 실행 시 `python`만 표시된다면, `python`을 `py`로 변경하십시오.

참고: 현재 `bitsandbytes==0.44.0`, `prodigyopt==1.0` 및 `lion-pytorch==0.0.6`이 requirements.txt에 포함되어 있습니다. 다른 버전을 사용하려면 수동으로 설치하십시오.

이 설치는 CUDA 11.8용입니다. 다른 버전의 CUDA를 사용하는 경우, 해당 버전의 PyTorch와 xformers를 설치하십시오. 예를 들어, CUDA 12를 사용하는 경우 `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121` 및 `pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121`를 실행하십시오.

PyTorch 2.2 이상을 사용하는 경우 `torch==2.1.2`, `torchvision==0.16.2`, `xformers==0.0.23.post1`을 적절한 버전으로 변경하십시오.

<!-- 
cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
-->
accelerate config에 대한 답변:

```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

bf16을 사용하려면 마지막 질문에 `bf16`이라고 답변해 주세요.

참고: 일부 사용자가 훈련 중 ``ValueError: fp16 혼합 정밀도는 GPU가 필요합니다`` 오류가 발생한다고 보고했습니다. 이 경우, 여섯 번째 질문에 `0`을 입력하세요:
``이 머신에서 훈련에 사용할 GPU(ID 기준)를 쉼표로 구분된 목록으로 입력하세요? [all]:``

(ID `0`의 단일 GPU가 사용됩니다.)

## 업그레이드

새 버전이 출시되면 다음 명령어로 저장소를 업그레이드할 수 있습니다:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

명령어가 성공적으로 완료되면 새 버전을 사용할 준비가 된 것입니다.

### PyTorch 업그레이드

PyTorch를 업그레이드하려면 [Windows 설치](#windows-installation) 섹션의 `pip install` 명령어로 업그레이드할 수 있습니다. PyTorch를 업그레이드할 때 `xformers`도 함께 업그레이드해야 합니다.

## 크레딧

LoRA 구현은 [cloneofsimo의 저장소](https://github.com/cloneofsimo/lora)를 기반으로 합니다. 훌륭한 작업에 감사드립니다!

Conv2d 3x3에 대한 LoRA 확장은 cloneofsimo에 의해 처음 공개되었으며, 그 효과는 KohakuBlueleaf에 의해 [LoCon](https://github.com/KohakuBlueleaf/LoCon)에서 입증되었습니다. KohakuBlueleaf님께 진심으로 감사드립니다!

## 라이선스

대부분의 스크립트는 ASL 2.0 라이선스 하에 배포됩니다(Diffusers, cloneofsimo 및 LoCon의 코드 포함). 다만 프로젝트의 일부 구성 요소는 별도의 라이선스 조건이 적용됩니다:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause


## Change History

### Mar 21, 2025 /  2025-03-21 Version 0.9.1

- Fixed a bug where some of LoRA modules for CLIP Text Encoder were not trained. Thank you Nekotekina for PR [#1964](https://github.com/kohya-ss/sd-scripts/pull/1964)
  - The LoRA modules for CLIP Text Encoder are now 264 modules, which is the same as before. Only 88 modules were trained in the previous version. 

### Jan 17, 2025 /  2025-01-17 Version 0.9.0

- __important__ The dependent libraries are updated. Please see [Upgrade](#upgrade) and update the libraries.
  - bitsandbytes, transformers, accelerate and huggingface_hub are updated. 
  - If you encounter any issues, please report them.

- The dev branch is merged into main. The documentation is delayed, and I apologize for that. I will gradually improve it.
- The state just before the merge is released as Version 0.8.8, so please use it if you encounter any issues.
- The following changes are included.

#### 변경 사항

## 추가 정보

### LoRA 명명 규칙

`train_network.py`에서 지원하는 LoRA의 명칭을 혼동을 피하기 위해 변경하였습니다. 관련 문서도 업데이트되었습니다. 본 저장소에서 사용하는 LoRA 유형의 명칭은 다음과 같습니다.

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers)

    LoRA for Linear layers and Conv2d layers with 1x1 kernel

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers)

    In addition to 1., LoRA for Conv2d layers with 3x3 kernel 
    
LoRA-LierLa는 `train_network.py`의 기본 LoRA 유형입니다(네트워크 인자 `conv_dim` 제외). 
<!-- 
LoRA-LierLa can be used with [our extension](https://github.com/kohya-ss/sd-webui-additional-networks) for AUTOMATIC1111's Web UI, or with the built-in LoRA feature of the Web UI.

To use LoRA-C3Lier with Web UI, please use our extension. 
-->

### 훈련 중 샘플 이미지 생성
  예를 들어 프롬프트 파일은 다음과 같을 수 있습니다

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  `#`로 시작하는 줄은 주석입니다. 프롬프트 뒤에 `--n`과 같은 옵션을 사용하여 생성된 이미지의 옵션을 지정할 수 있습니다. 다음을 사용할 수 있습니다.

  * `--n` 다음 옵션까지 프롬프트를 음수로 지정합니다.
  * `--w` 생성된 이미지의 너비를 지정합니다.
  * `--h` 생성된 이미지의 높이를 지정합니다.
  * `--d` 생성된 이미지의 시드(seed)를 지정합니다.
  * `--l` 생성된 이미지의 CFG 스케일을 지정합니다.
  * `--s` 생성 과정의 단계 수를 지정합니다.

  `( )` 및 `[ ]`와 같은 프롬프트 가중치 기능이 작동합니다.



