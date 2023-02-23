This repository contains training, generation and utility scripts for Stable Diffusion.

[__Change History__](#change-history) is moved to the bottom of the page.
更新履歴は[ページ末尾](#change-history)に移しました。

[日本語版README](./README-ja.md)

For easier use (GUI and PowerShell scripts etc...), please visit [the repository maintained by bmaltais](https://github.com/bmaltais/kohya_ss). Thanks to @bmaltais!

This repository contains the scripts for:

* DreamBooth training, including U-Net and Text Encoder
* Fine-tuning (native training), including U-Net and Text Encoder
* LoRA training
* Texutl Inversion training
* Image generation
* Model conversion (supports 1.x and 2.x, Stable Diffision ckpt/safetensors and Diffusers)

__Stable Diffusion web UI now seems to support LoRA trained by ``sd-scripts``.__ (SD 1.x based only) Thank you for great work!!! 

## About requirements.txt

These files do not contain requirements for PyTorch. Because the versions of them depend on your environment. Please install PyTorch at first (see installation guide below.) 

The scripts are tested with PyTorch 1.12.1 and 1.13.0, Diffusers 0.10.2.

## Links to how-to-use documents

All documents are in Japanese currently, and CUI based.

* [DreamBooth training guide](./train_db_README-ja.md)
* [Step by Step fine-tuning guide](./fine_tune_README_ja.md):
Including BLIP captioning and tagging by DeepDanbooru or WD14 tagger
* [training LoRA](./train_network_README-ja.md)
* [training Textual Inversion](./train_ti_README-ja.md)
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

python -m venv venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config
```

update: ``python -m venv venv`` is seemed to be safer than ``python -m venv --system-site-packages venv`` (some user have packages in global python).

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

### about PyTorch and xformers

Other versions of PyTorch and xformers seem to have problems with training.
If there is no other reason, please install the specified version.

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

## Credits

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for great work!!!

## License

The majority of scripts is licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's), however portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause

## Change History

- 23 Feb. 2023, 2023/2/23:
  - Fix instability training issue in ``train_network.py``.
    - ``fp16`` training is probably not affected by this issue.
    - Training with ``float`` for SD2.x models will work now. Also training with ``bf16`` might be improved. 
    - This issue seems to have occurred in [PR#190](https://github.com/kohya-ss/sd-scripts/pull/190).
  - Add some metadata to LoRA model. Thanks to space-nuko!
  - Raise an error if optimizer options conflict (e.g. ``--optimizer_type`` and ``--use_8bit_adam``.) 
  - Support ControlNet in ``gen_img_diffusers.py`` (no documentation yet.)
  - ``train_network.py`` で学習が不安定になる不具合を修正しました。
    - ``fp16`` 精度での学習には恐らくこの問題は影響しません。
    - ``float`` 精度での SD2.x モデルの学習が正しく動作するようになりました。また ``bf16`` 精度の学習も改善する可能性があります。
    - この問題は [PR#190](https://github.com/kohya-ss/sd-scripts/pull/190) から起きていたようです。
  - いくつかのメタデータを LoRA モデルに追加しました。 space-nuko 氏に感謝します。
  - オプティマイザ関係のオプションが矛盾していた場合、エラーとするように修正しました（例: ``--optimizer_type`` と ``--use_8bit_adam``）。
  - ``gen_img_diffusers.py`` で ControlNet をサポートしました（ドキュメントはのちほど追加します）。

- 22 Feb. 2023, 2023/2/22:
  - Refactor optmizer options. Thanks to mgz-dev!
    - Add ``--optimizer_type`` option for each training script. Please see help. Japanese documentation is [here](https://github.com/kohya-ss/sd-scripts/blob/main/train_network_README-ja.md#%E3%82%AA%E3%83%97%E3%83%86%E3%82%A3%E3%83%9E%E3%82%A4%E3%82%B6%E3%81%AE%E6%8C%87%E5%AE%9A%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6).
    - ``--use_8bit_adam`` and ``--use_lion_optimizer`` options also work, but override above option.
  - Add SGDNesterov and its 8bit.
  - Add [D-Adaptation](https://github.com/facebookresearch/dadaptation) optimizer. Thanks to BootsofLagrangian and all! 
    - Please install D-Adaptation optimizer with ``pip install dadaptation`` (it is not in requirements.txt currently.)
    - Please see https://github.com/kohya-ss/sd-scripts/issues/181 for details.
  - Add AdaFactor optimizer. Thanks to Toshiaki!
  - Extra lr scheduler settings (num_cycles etc.) are working in training scripts other than ``train_network.py``.
  - Add ``--max_grad_norm`` option for each training script for gradient clipping. ``0.0`` disables clipping. 
  - Symbolic link can be loaded in each training script. Thanks to TkskKurumi!
  - オプティマイザ関連のオプションを見直しました。mgz-dev氏に感謝します。
    - ``--optimizer_type`` を各学習スクリプトに追加しました。ドキュメントは[こちら](https://github.com/kohya-ss/sd-scripts/blob/main/train_network_README-ja.md#%E3%82%AA%E3%83%97%E3%83%86%E3%82%A3%E3%83%9E%E3%82%A4%E3%82%B6%E3%81%AE%E6%8C%87%E5%AE%9A%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6)。
    - ``--use_8bit_adam`` と ``--use_lion_optimizer`` のオプションは依然として動作しますがoptimizer_typeを上書きしますのでご注意ください。
  - SGDNesterov オプティマイザおよびその8bit版を追加しました。
  - [D-Adaptation](https://github.com/facebookresearch/dadaptation) オプティマイザを追加しました。BootsofLagrangian 氏および諸氏に感謝します。
    - ``pip install dadaptation`` コマンドで別途インストールが必要です（現時点ではrequirements.txtに含まれておりません）。
    - こちらのissueもあわせてご覧ください。 https://github.com/kohya-ss/sd-scripts/issues/181 
  - AdaFactor オプティマイザを追加しました。Toshiaki氏に感謝します。 
  - 追加のスケジューラ設定（num_cycles等）が ``train_network.py`` 以外の学習スクリプトでも使えるようになりました。
  - 勾配クリップ時の最大normを指定する ``--max_grad_norm`` オプションを追加しました。``0.0``を指定するとクリップしなくなります。
  - 各学習スクリプトでシンボリックリンクが読み込めるようになりました。TkskKurumi氏に感謝します。

Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.
最近の更新情報は [Release](https://github.com/kohya-ss/sd-scripts/releases) をご覧ください。
