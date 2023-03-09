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

- 2 Mar. 2023, 2023/3/2:
  - There may be problems due to major changes. If you cannot revert back to the previous version when problems occur, please do not update for a while.
  - Dependencies are updated, Please [upgrade](#upgrade) the repo.
  - Add detail dataset config feature by extra config file. Thanks to fur0ut0 for this great contribution!
    - Documentation is [here](./config_README-ja.md) (only in Japanese currently.)
    - Specify ``.toml`` file with ``--dataset_config`` option.
    - The previous options for dataset can be used as is.
    - There might be a bug due to the large scale of update, please report any problems if you find.
  - Add feature to generate sample images in the middle of training for each training scripts.
    - ``--sample_every_n_steps`` and ``--sample_every_n_epochs`` options: frequency to generate.
    - ``--sample_prompts`` option: the file contains prompts (each line generates one image.)
      - The prompt is subset of ``gen_img_diffusers.py``. The prompt options ``w, h, d, l, s, n`` are supported.
    - ``--sample_sampler`` option: sampler (scheduler) for generating, such as ddim or k_euler. See help for useable samplers.
  - Add ``--tokenizer_cache_dir`` to each training and generation scripts to cache Tokenizer locally from Diffusers.
    - Scripts will support offline training/generation after caching.
  - Support letents upscaling for highres. fix, and VAE batch size in ``gen_img_diffusers.py`` (no documentation yet.)

  - Sample image generation:
    A prompt file might look like this, for example

    ```
    # prompt 1
    masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

    # prompt 2
    masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
    ```

    Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following can be used.

    * `--n` Negative prompt up to the next option.
    * `--w` Specifies the width of the generated image.
    * `--h` Specifies the height of the generated image.
    * `--d` Specifies the seed of the generated image.
    * `--l` Specifies the CFG scale of the generated image.
    * `--s` Specifies the number of steps in the generation.

    The prompt weighting such as `( )` and `[ ]` are not working.

  - 大きく変更したため不具合があるかもしれません。問題が起きた時にスクリプトを前のバージョンに戻せない場合は、しばらく更新を控えてください。
  - ライブラリを更新しました。[アップグレード](https://github.com/kohya-ss/sd-scripts/blob/main/README-ja.md#%E3%82%A2%E3%83%83%E3%83%97%E3%82%B0%E3%83%AC%E3%83%BC%E3%83%89)に従って更新してください。
  - 設定ファイルによるデータセット定義機能を追加しました。素晴らしいPRを提供していただいた fur0ut0 氏に感謝します。
    - ドキュメントは[こちら](./config_README-ja.md)。
    - ``--dataset_config`` オプションで ``.toml`` ファイルを指定してください。
    - 今までのオプションはそのまま使えます。
    - 大規模なアップデートのため、もし不具合がありましたらご報告ください。
  - 学習の途中でサンプル画像を生成する機能を各学習スクリプトに追加しました。
    - ``--sample_every_n_steps`` と ``--sample_every_n_epochs`` オプション：生成頻度を指定
    - ``--sample_prompts`` オプション：プロンプトを記述したファイルを指定（1行ごとに1枚の画像を生成）
      - プロンプトには ``gen_img_diffusers.py`` のプロンプトオプションの一部、 ``w, h, d, l, s, n`` が使えます。
    - ``--sample_sampler`` オプション：ddim や k_euler などの sampler (scheduler) を指定します。使用できる sampler についてはヘルプをご覧ください。
  - ``--tokenizer_cache_dir`` オプションを各学習スクリプトおよび生成スクリプトに追加しました。Diffusers から Tokenizer を取得してきてろーかるに保存します。
    - 一度キャッシュしておくことでオフライン学習、生成ができるかもしれません。
  - ``gen_img_diffusers.py`` で highres. fix での letents upscaling と VAE のバッチサイズ指定に対応しました。

  - サンプル画像生成：
    プロンプトファイルは例えば以下のようになります。

    ```
    # prompt 1
    masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

    # prompt 2
    masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
    ```

    `#` で始まる行はコメントになります。`--n` のように「ハイフン二個＋英小文字」の形でオプションを指定できます。以下が使用可能できます。

    * `--n` Negative prompt up to the next option.
    * `--w` Specifies the width of the generated image.
    * `--h` Specifies the height of the generated image.
    * `--d` Specifies the seed of the generated image.
    * `--l` Specifies the CFG scale of the generated image.
    * `--s` Specifies the number of steps in the generation.

    `( )` や `[ ]` などの重みづけは動作しません。

Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.
最近の更新情報は [Release](https://github.com/kohya-ss/sd-scripts/releases) をご覧ください。
