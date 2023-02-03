This repository contains training, generation and utility scripts for Stable Diffusion.

## Updates

__Stable Diffusion web UI now seems to support LoRA trained by ``sd-scripts``.__ Thank you for great work!!! 

Note: The LoRA models for SD 2.x is not supported too in Web UI.

- 3 Feb. 2023, 2023/2/3
  - Update finetune preprocessing scripts.
    - ``.bmp`` and ``.jpeg`` are supported. Thanks to breakcore2 and p1atdev!
    - The default weights of ``tag_images_by_wd14_tagger.py`` is now ``SmilingWolf/wd-v1-4-convnext-tagger-v2``. You can specify another model id from ``SmilingWolf`` by ``--repo_id`` option. Thanks to SmilingWolf for the great work.
      - To change the weight, remove ``wd14_tagger_model`` folder, and run the script again.
    - ``--max_data_loader_n_workers`` option is added to each script. This option uses the DataLoader for data loading to speed up loading, 20%~30% faster.
      - Please specify 2 or 4, depends on the number of CPU cores.
    - ``--recursive`` option is added to ``merge_dd_tags_to_metadata.py`` and ``merge_captions_to_metadata.py``, only works with ``--full_path``.
    - ``make_captions_by_git.py`` is added. It uses [GIT microsoft/git-large-textcaps](https://huggingface.co/microsoft/git-large-textcaps) for captioning. 
      - Usage is almost the same as ``make_captions.py``, but batch size should be smaller.
      - ``--remove_words`` option removes as much text as possible (such as ``the word "XXXX" on it``).
    - ``--skip_existing`` option is added to ``prepare_buckets_latents.py``. Images with existing npz files are ignored by this option.
    - ``clean_captions_and_tags.py`` is updated to remove duplicated or conflicting tags, e.g. ``shirt`` is removed when ``white shirt`` exists. if ``black hair`` is with ``red hair``, both are removed.
  - Tag frequency is added to the metadata in ``train_network.py``. Thanks to space-nuko!
    - __All tags and number of occurrences of the tag are recorded.__ If you do not want it, disable metadata storing with ``--no_metadata`` option.
  
  - fine tuning用の前処理スクリプト群を更新しました。
    - 拡張子 ``.bmp`` と ``.jpeg`` をサポートしました。breakcore2氏およびp1atdev氏に感謝します。
    - ``tag_images_by_wd14_tagger.py`` のデフォルトの重みを ``SmilingWolf/wd-v1-4-convnext-tagger-v2`` に更新しました。他の ``SmilingWolf`` 氏の重みも ``--repo_id`` オプションで指定可能です。SmilingWolf氏に感謝します。
      - 重みを変更するときには ``wd14_tagger_model`` フォルダを削除してからスクリプトを再実行してください。
    - ``--max_data_loader_n_workers`` オプションが各スクリプトに追加されました。DataLoaderを用いることで読み込み処理を並列化し、処理を20~30%程度高速化します。
      - CPUのコア数に応じて2~4程度の値を指定してください。
    - ``--recursive`` オプションを ``merge_dd_tags_to_metadata.py`` と ``merge_captions_to_metadata.py`` に追加しました。``--full_path`` を指定したときのみ使用可能です。
    - ``make_captions_by_git.py`` を追加しました。[GIT microsoft/git-large-textcaps](https://huggingface.co/microsoft/git-large-textcaps) を用いてキャプションニングを行います。
      - 使用法は ``make_captions.py``とほぼ同じですがバッチサイズは小さめにしてください。
      - ``--remove_words`` オプションを指定するとテキスト読み取りを可能な限り削除します（``the word "XXXX" on it``のようなもの）。    
    - ``--skip_existing`` を ``prepare_buckets_latents.py`` に追加しました。すでにnpzファイルがある画像の処理をスキップします。
    - ``clean_captions_and_tags.py``を重複タグや矛盾するタグを削除するよう機能追加しました。例：``white shirt`` タグがある場合、 ``shirt`` タグは削除されます。また``black hair``と``red hair``の両方がある場合、両方とも削除されます。
  - ``train_network.py``で使用されているタグと回数をメタデータに記録するようになりました。space-nuko氏に感謝します。
    - __すべてのタグと回数がメタデータに記録されます__ 望まない場合には``--no_metadata option``オプションでメタデータの記録を停止してください。
    
- 29 Jan. 2023, 2023/1/29
  - Add ``--lr_scheduler_num_cycles`` and ``--lr_scheduler_power`` options for ``train_network.py`` for cosine_with_restarts and polynomial learning rate schedulers. Thanks to mgz-dev!
  - Fixed U-Net ``sample_size`` parameter to ``64`` when converting from SD to Diffusers format, in ``convert_diffusers20_original_sd.py``
  - ``--lr_scheduler_num_cycles`` と ``--lr_scheduler_power`` オプションを ``train_network.py`` に追加しました。前者は cosine_with_restarts、後者は polynomial の学習率スケジューラに有効です。mgz-dev氏に感謝します。
  - ``convert_diffusers20_original_sd.py`` で SD 形式から Diffusers に変換するときの U-Net の ``sample_size`` パラメータを ``64`` に修正しました。
- 26 Jan. 2023, 2023/1/26
  - Add Textual Inversion training. Documentation is [here](./train_ti_README-ja.md) (in Japanese.)
  - Textual Inversionの学習をサポートしました。ドキュメントは[こちら](./train_ti_README-ja.md)。
- 24 Jan. 2023, 2023/1/24
  - Change the default save format to ``.safetensors`` for ``train_network.py``.
  - Add ``--save_n_epoch_ratio`` option to specify how often to save. Thanks to forestsource! 
    - For example, if 5 is specified, 5 (or 6) files will be saved in training.
  - Add feature to pre-calculate hash to reduce loading time in the extension. Thanks to space-nuko!
  - Add bucketing metadata. Thanks to space-nuko!
  - Fix an error with bf16 model in ``gen_img_diffusers.py``.
  - ``train_network.py`` のモデル保存形式のデフォルトを ``.safetensors`` に変更しました。
  - モデルを保存する頻度を指定する ``--save_n_epoch_ratio`` オプションが追加されました。forestsource氏に感謝します。
    - たとえば 5 を指定すると、学習終了までに合計で5個（または6個）のファイルが保存されます。
  - 拡張でモデル読み込み時間を短縮するためのハッシュ事前計算の機能を追加しました。space-nuko氏に感謝します。
  - メタデータにbucket情報が追加されました。space-nuko氏に感謝します。
  - ``gen_img_diffusers.py`` でbf16形式のモデルを読み込んだときのエラーを修正しました。

Stable Diffusion web UI本体で当リポジトリで学習したLoRAモデルによる画像生成がサポートされたようです。

注：SD2.x用のLoRAモデルはサポートされないようです。

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
