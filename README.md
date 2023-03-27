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

All documents are in Japanese currently.

* [Training guide - common](./train_README-ja.md) : data preparation, options etc...
    * [Dataset config](./config_README-ja.md)
* [DreamBooth training guide](./train_db_README-ja.md)
* [Step by Step fine-tuning guide](./fine_tune_README_ja.md):
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

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for great work!

The LoRA expansion to Conv2d 3x3 was initially released by cloneofsimo and its effectiveness was demonstrated at [LoCon](https://github.com/KohakuBlueleaf/LoCon) by KohakuBlueleaf. Thank you so much KohakuBlueleaf!

## License

The majority of scripts is licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's and LoCon), however portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause

## Change History

- 27 Mar. 2023, 2023/3/27:
  - Fix issues when `--persistent_data_loader_workers` is specified.
    - The batch members of the bucket are not shuffled.
    - `--caption_dropout_every_n_epochs` does not work.
    - These issues occurred because the epoch transition was not recognized correctly. Thanks to u-haru for reporting the issue.
  - Fix an issue that images are loaded twice in Windows environment.
  - Add Min-SNR Weighting strategy. Details are in [#308](https://github.com/kohya-ss/sd-scripts/pull/308). Thank you to AI-Casanova for this great work!
    - Add `--min_snr_gamma` option to training scripts, 5 is recommended by paper.

  - Add tag warmup. Details are in [#322](https://github.com/kohya-ss/sd-scripts/pull/322). Thanks to u-haru!
    - Add `token_warmup_min` and `token_warmup_step` to dataset settings.
    - Gradually increase the number of tokens from `token_warmup_min` to `token_warmup_step`.
    - For example, if `token_warmup_min` is `3` and `token_warmup_step` is `10`, the first step will use the first 3 tokens, and the 10th step will use all tokens.
  - Fix a bug in `resize_lora.py`. Thanks to mgz-dev! [#328](https://github.com/kohya-ss/sd-scripts/pull/328)  
  - Add `--debug_dataset` option to step to the next step with `S` key and to the next epoch with `E` key.
  - Fix other bugs.

  - `--persistent_data_loader_workers` を指定した時の各種不具合を修正しました。
    - `--caption_dropout_every_n_epochs` が効かない。
    - バケットのバッチメンバーがシャッフルされない。
    - エポックの遷移が正しく認識されないために発生していました。ご指摘いただいたu-haru氏に感謝します。
  - Windows環境で画像が二重に読み込まれる不具合を修正しました。
  - Min-SNR Weighting strategyを追加しました。 詳細は [#308](https://github.com/kohya-ss/sd-scripts/pull/308) をご参照ください。AI-Casanova氏の素晴らしい貢献に感謝します。
    - `--min_snr_gamma` オプションを学習スクリプトに追加しました。論文では5が推奨されています。
  - タグのウォームアップを追加しました。詳細は [#322](https://github.com/kohya-ss/sd-scripts/pull/322) をご参照ください。u-haru氏に感謝します。
    - データセット設定に `token_warmup_min` と `token_warmup_step` を追加しました。
    - `token_warmup_min` で指定した数のトークン（カンマ区切りの文字列）から、`token_warmup_step` で指定したステップまで、段階的にトークンを増やしていきます。
    - たとえば `token_warmup_min`に `3` を、`token_warmup_step` に `10` を指定すると、最初のステップでは最初から3個のトークンが使われ、10ステップ目では全てのトークンが使われます。
  - `resize_lora.py` の不具合を修正しました。mgz-dev氏に感謝します。[#328](https://github.com/kohya-ss/sd-scripts/pull/328)  
  - `--debug_dataset` オプションで、`S`キーで次のステップへ、`E`キーで次のエポックへ進めるようにしました。
  - その他の不具合を修正しました。


- 21 Mar. 2023, 2023/3/21:
  - Add `--vae_batch_size` for faster latents caching to each training script. This  batches VAE calls.
    - Please start with`2` or `4` depending on the size of VRAM.
  - Fix a number of training steps with `--gradient_accumulation_steps` and `--max_train_epochs`. Thanks to tsukimiya!
  - Extract parser setup to external scripts. Thanks to robertsmieja!
  - Fix an issue without `.npz` and with `--full_path` in training.
  - Support extensions with upper cases for images for not Windows environment.
  - Fix `resize_lora.py` to work with LoRA with dynamic rank (including `conv_dim != network_dim`). Thanks to toshiaki!
  - latentsのキャッシュを高速化する`--vae_batch_size` オプションを各学習スクリプトに追加しました。VAE呼び出しをバッチ化します。
    -VRAMサイズに応じて、`2` か `4` 程度から試してください。
  - `--gradient_accumulation_steps` と `--max_train_epochs` を指定した時、当該のepochで学習が止まらない不具合を修正しました。tsukimiya氏に感謝します。
  - 外部のスクリプト用に引数parserの構築が関数化されました。robertsmieja氏に感謝します。
  - 学習時、`--full_path` 指定時に `.npz` が存在しない場合の不具合を解消しました。
  - Windows以外の環境向けに、画像ファイルの大文字の拡張子をサポートしました。
  - `resize_lora.py` を dynamic rank （rankが各LoRAモジュールで異なる場合、`conv_dim` が `network_dim` と異なる場合も含む）の時に正しく動作しない不具合を修正しました。toshiaki氏に感謝します。

## Sample image generation during training
  A prompt file might look like this, for example

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following can be used.

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

  The prompt weighting such as `( )` and `[ ]` are working.

## サンプル画像生成
プロンプトファイルは例えば以下のようになります。

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  `#` で始まる行はコメントになります。`--n` のように「ハイフン二個＋英小文字」の形でオプションを指定できます。以下が使用可能できます。

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

  `( )` や `[ ]` などの重みづけも動作します。

Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.
最近の更新情報は [Release](https://github.com/kohya-ss/sd-scripts/releases) をご覧ください。
