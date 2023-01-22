This repository contains training, generation and utility scripts for Stable Diffusion.

## Updates

__Stable Diffusion web UI now seems to support LoRA trained by ``sd-scripts``.__ Thank you for great work!!!

Note: Currently the LoRA models trained by release 0.4.0 does not seem to be supported. If you use Web UI native LoRA support, please use release 0.3.2 for now. 

The LoRA models for SD 2.x is not supported too.

- 22 Jan. 2023
  - Add ``--network_alpha`` option to specify ``alpha`` value to prevent underflows for stable training. Thanks to CCRcmcpe!
    - Details of the issue are described in https://github.com/kohya-ss/sd-webui-additional-networks/issues/49 .
    - The default value is ``1``, scale ``1 / rank (or dimension)``. Set same value as ``network_dim`` for same behavior to old version.
    - LoRA with a large dimension (rank) seems to require a higher learning rate with ``alpha=1`` (e.g. 1e-3 for 128-dim, still investigating).　
  - Add logging for the learning rate for U-Net and Text Encoder independently, and for running average epoch loss. Thanks to mgz-dev!  
  - Add more metadata such as dataset/reg image dirs, session ID, output name etc... See https://github.com/kohya-ss/sd-scripts/pull/77 for details. Thanks to space-nuko!
    - __Now the metadata includes the folder name (the basename of the folder contains image files, not fullpath).__ If you do not want it, disable metadata storing with ``--no_metadata`` option.
  - Add ``--training_comment`` option. You can specify an arbitrary string and refer to it by the extension.

Stable Diffusion web UI本体で当リポジトリで学習したLoRAモデルによる画像生成がサポートされたようです。

注：現時点ではversion 0.4.0で学習したモデルはサポートされないようです。Web UI本体の生成機能を使う場合には、version 0.3.2を引き続きご利用ください。またSD2.x用のLoRAモデルもサポートされないようです。

- 2023/1/22
  - アンダーフローを防ぎ安定して学習するための ``alpha`` 値を指定する、``--network_alpha`` オプションを追加しました。CCRcmcpe 氏に感謝します。
    - 問題の詳細はこちらをご覧ください： https://github.com/kohya-ss/sd-webui-additional-networks/issues/49
    - デフォルト値は ``1`` で、重みを ``1 / rank (dimension・次元数)`` します。``network_dim`` と同じ値を指定すると旧バージョンと同じ動作になります。
    -  ``alpha=1``の場合、次元数（rank）の多いLoRAモジュールでは学習率を高めにしたほうが良いようです（128次元で1e-3など）。
  - U-Net と Text Encoder のそれぞれの学習率、エポックの平均lossをログに記録するようになりました。mgz-dev 氏に感謝します。
  - 画像ディレクトリ、セッションID、出力名などいくつかの項目がメタデータに追加されました（詳細は https://github.com/kohya-ss/sd-scripts/pull/77 を参照）。space-nuko氏に感謝します。
    - __メタデータにフォルダ名が含まれるようになりました（画像を含むフォルダの名前のみで、フルパスではありません）。__ もし望まない場合には ``--no_metadata`` オプションでメタデータの記録を止めてください。
  - ``--training_comment`` オプションを追加しました。任意の文字列を指定でき、Web UI拡張から参照できます。

Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.
最近の更新情報は [Release](https://github.com/kohya-ss/sd-scripts/releases) をご覧ください。

##

[日本語版README](./README-ja.md)

For easier use (GUI and PowerShell scripts etc...), please visit [the repository maintained by bmaltais](https://github.com/bmaltais/kohya_ss). Thanks to @bmaltais!

This repository contains the scripts for:

* DreamBooth training, including U-Net and Text Encoder
* fine-tuning (native training), including U-Net and Text Encoder
* LoRA training
* image generation
* model conversion (supports 1.x and 2.x, Stable Diffision ckpt/safetensors and Diffusers)

## About requirements.txt

These files do not contain requirements for PyTorch. Because the versions of them depend on your environment. Please install PyTorch at first (see installation guide below.) 

The scripts are tested with PyTorch 1.12.1 and 1.13.0, Diffusers 0.10.2.

## Links to how-to-use documents

All documents are in Japanese currently, and CUI based.

* [DreamBooth training guide](./train_db_README-ja.md)
* [Step by Step fine-tuning guide](./fine_tune_README_ja.md):
Including BLIP captioning and tagging by DeepDanbooru or WD14 tagger
* [training LoRA](./train_network_README-ja.md)
* note.com [Image generation](https://note.com/kohya_ss/n/n2693183a798e)
* note.com [Model conversion](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windows Required Dependencies

Python 3.10.6 and Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Windows Installation

Open a regular Powershell terminal and type the following inside:

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv --system-site-packages venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config

```

Answers to accelerate config:

```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

note: Some user reports ``ValueError: fp16 mixed precision requires a GPU`` is occurred in training. In this case, answer `0` for the 6th question: 
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`` 

(Single GPU with id `0` will be used.)

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

## Credits

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for great work!!!

## License

The majority of scripts is licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's), however portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause
