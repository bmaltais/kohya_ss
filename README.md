This repository contains training, generation and utility scripts for Stable Diffusion.

## Updates

- January 14, 2023, 2023/1/14
  - Fix loading some VAE or .safetensors as VAE is failed for ``--vae`` option. Thanks to Fannovel16!
  - Add negative prompt scaling for ``gen_img_diffusers.py`` You can set another conditioning scale to the negative prompt with ``--negative_scale`` option, and ``--nl`` option for the prompt. Thanks to laksjdjf!
  - ``--vae`` オプションに一部のVAEや .safetensors 形式のモデルを指定するとエラーになる不具合を修正しました。Fannovel16氏に感謝します。
  - ``gen_img_diffusers.py`` に、ネガティブプロンプトに異なる guidance scale を設定できる ``--negative_scale`` オプションを追加しました。プロンプトからは ``--nl`` で指定できます。laksjdjf氏に感謝します。
- January 12, 2023, 2023/1/12
  - Metadata is saved on the model (.safetensors only) (model name, VAE name, training steps, learning rate etc.) The metadata will be able to inspect by sd-webui-additional-networks extension in near future. If you do not want to save it, specify ``no_metadata`` option.
  - メタデータが保存されるようになりました（ .safetensors 形式の場合のみ）（モデル名、VAE 名、ステップ数、学習率など）。近日中に拡張から確認できるようになる予定です。メタデータを保存したくない場合は ``no_metadata`` オプションをしてしてください。
  
**January 9, 2023: Important information about the update can be found at [the end of the page](#updates-jan-9-2023).**

**20231/1/9: 更新情報が[ページ末尾](#更新情報-202319)にありますのでご覧ください。**

[日本語版README](./README-ja.md)

##

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

note: Some user reports ``ValueError: fp16 mixed precision requires a GPU`` is occured in training. In this case, answer `0` for the 6th question: 
``What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:`` 

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


# Updates: Jan 9. 2023 

All training scripts are updated. 

## Breaking Changes

- The ``fine_tuning`` option in ``train_db.py`` is removed. Please use DreamBooth with captions or ``fine_tune.py``.
- The Hypernet feature in ``fine_tune.py`` is removed, will be implemented in ``train_network.py`` in future.

## Features, Improvements and Bug Fixes

### for all script: train_db.py, fine_tune.py and train_network.py

- Added ``output_name`` option. The name of output file can be specified.
    - With ``--output_name style1``, the output file is like ``style1_000001.ckpt`` (or ``.safetensors``) for each epoch and ``style1.ckpt`` for last.
    - If ommitted (default), same to previous. ``epoch-000001.ckpt`` and ``last.ckpt``.
- Added ``save_last_n_epochs`` option. Keep only latest n files for the checkpoints and the states. Older files are removed. (Thanks to shirayu!)
    - If the options are ``--save_every_n_epochs=2 --save_last_n_epochs=3``, in the end of epoch 8, ``epoch-000008.ckpt`` is created and ``epoch-000002.ckpt`` is removed.

### train_db.py

- Added ``max_token_length`` option. Captions can have more than 75 tokens.

### fine_tune.py

- The script now works without .npz files. If .npz is not found, the scripts get the latents with VAE.
    - You can omit ``prepare_buckets_latents.py`` in preprocessing. However, it is recommended if you train more than 1 or 2 epochs.
    - ``--resolution`` option is required to specify the training resolution.
- Added ``cache_latents`` and ``color_aug`` options.

### train_network.py

- Now ``--gradient_checkpointing`` is effective for U-Net and Text Encoder.
    - The memory usage is reduced. The larger batch size is avilable, but the training speed will be slow.
    - The training might be possible with 6GB VRAM for dimension=4 with batch size=1.

Documents are not updated now, I will update one by one.

# 更新情報 (2023/1/9)

学習スクリプトを更新しました。

## 削除された機能
- ``train_db.py`` の ``fine_tuning`` は削除されました。キャプション付きの DreamBooth または ``fine_tune.py`` を使ってください。
- ``fine_tune.py`` の Hypernet学習の機能は削除されました。将来的に``train_network.py``に追加される予定です。

## その他の機能追加、バグ修正など

### 学習スクリプトに共通: train_db.py, fine_tune.py and train_network.py

- ``output_name``オプションを追加しました。保存されるモデルファイルの名前を指定できます。
    - ``--output_name style1``と指定すると、エポックごとに保存されるファイル名は``style1_000001.ckpt`` (または ``.safetensors``) に、最後に保存されるファイル名は``style1.ckpt``になります。
    - 省略時は今までと同じです（``epoch-000001.ckpt``および``last.ckpt``）。
- ``save_last_n_epochs``オプションを追加しました。最新の n ファイル、stateだけ保存し、古いものは削除します。（shirayu氏に感謝します。)
    - たとえば``--save_every_n_epochs=2 --save_last_n_epochs=3``と指定した時、8エポック目の終了時には、``epoch-000008.ckpt``が保存され``epoch-000002.ckpt``が削除されます。

### train_db.py

- ``max_token_length``オプションを追加しました。75文字を超えるキャプションが使えるようになります。

### fine_tune.py

- .npzファイルがなくても動作するようになりました。.npzファイルがない場合、VAEからlatentsを取得して動作します。
    -  ``prepare_buckets_latents.py``を前処理で実行しなくても良くなります。ただし事前取得をしておいたほうが、2エポック以上学習する場合にはトータルで高速です。
    - この場合、解像度を指定するために``--resolution``オプションが必要です。
- ``cache_latents``と``color_aug``オプションを追加しました。

### train_network.py

- ``--gradient_checkpointing``がU-NetとText Encoderにも有効になりました。
    - メモリ消費が減ります。バッチサイズを大きくできますが、トータルでの学習時間は長くなるかもしれません。
    - dimension=4のLoRAはバッチサイズ1で6GB VRAMで学習できるかもしれません。

ドキュメントは未更新ですが少しずつ更新の予定です。
