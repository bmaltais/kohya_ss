This repository contains training, generation and utility scripts for Stable Diffusion.

## Updates

__Stable Diffusion web UI now seems to support LoRA trained by ``sd-scripts``.__ Thank you for great work!!! 

Note: The LoRA models for SD 2.x is not supported too in Web UI.

- 10 Feb. 2023, 2023/2/10:
  - Updated ``requirements.txt`` to prevent upgrading with pip taking a long time or failure to upgrade.
  - pipでの更新が長時間掛かったり、更新に失敗したりするのを防ぐため、``requirements.txt``を更新しました。
- 9 Feb. 2023, 2023/2/9:
  - Caption dropout is supported in ``train_db.py``, ``fine_tune.py`` and ``train_network.py``. Thanks to forestsource!
    - ``--caption_dropout_rate`` option specifies the dropout rate for captions (0~1.0, 0.1 means 10% chance for dropout). If dropout occurs, the image is trained with the empty caption. Default is 0 (no dropout).
    - ``--caption_dropout_every_n_epochs`` option specifies how many epochs to drop captions. If ``3`` is specified, in epoch 3, 6, 9 ..., images are trained with all captions empty. Default is None (no dropout).
    - ``--caption_tag_dropout_rate`` option specified the dropout rate for tags (comma separated tokens) (0~1.0, 0.1 means 10% chance for dropout). If dropout occurs, the tag is removed from the caption. If ``--keep_tokens`` option is set, these tokens (tags) are not dropped. Default is 0 (no droupout).
    - The bulk image downsampling script is added. Documentation is [here](https://github.com/kohya-ss/sd-scripts/blob/main/train_network_README-ja.md#%E7%94%BB%E5%83%8F%E3%83%AA%E3%82%B5%E3%82%A4%E3%82%BA%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%97%E3%83%88) (in Jpanaese). Thanks to bmaltais!
    - Typo check is added. Thanks to shirayu!
  - キャプションのドロップアウトを``train_db.py``、``fine_tune.py``、``train_network.py``の各スクリプトに追加しました。forestsource氏に感謝します。
    - ``--caption_dropout_rate``オプションでキャプションのドロップアウト率を指定します（0~1.0、 0.1を指定すると10%の確率でドロップアウト）。ドロップアウトされた場合、画像は空のキャプションで学習されます。デフォルトは 0 （ドロップアウトなし）です。
    - ``--caption_dropout_every_n_epochs`` オプションで何エポックごとにキャプションを完全にドロップアウトするか指定します。たとえば``3``を指定すると、エポック3、6、9……で、すべての画像がキャプションなしで学習されます。デフォルトは None （ドロップアウトなし）です。
    - ``--caption_tag_dropout_rate`` オプションで各タグ（カンマ区切りの各部分）のドロップアウト率を指定します（0~1.0、 0.1を指定すると10%の確率でドロップアウト）。ドロップアウトが起きるとそのタグはそのときだけキャプションから取り除かれて学習されます。``--keep_tokens`` オプションを指定していると、シャッフルされない部分のタグはドロップアウトされません。デフォルトは 0 （ドロップアウトなし）です。
    - 画像の一括縮小スクリプトを追加しました。ドキュメントは [こちら](https://github.com/kohya-ss/sd-scripts/blob/main/train_network_README-ja.md#%E7%94%BB%E5%83%8F%E3%83%AA%E3%82%B5%E3%82%A4%E3%82%BA%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%97%E3%83%88) です。bmaltais氏に感謝します。
    - 誤字チェッカが追加されました。shirayu氏に感謝します。
    
- 6 Feb. 2023, 2023/2/6：
  - ``--bucket_reso_steps`` and ``--bucket_no_upscale`` options are added to training scripts (fine tuning, DreamBooth, LoRA and Textual Inversion) and ``prepare_buckets_latents.py``.
  - ``--bucket_reso_steps`` takes the steps for buckets in aspect ratio bucketing. Default is 64, same as before.
    - Any value greater than or equal to 1 can be specified; 64 is highly recommended and a value divisible by 8 is recommended.
    - If less than 64 is specified, padding will occur within U-Net. The result is unknown.
    - If you specify a value that is not divisible by 8, it will be truncated to divisible by 8 inside VAE, because the size of the latent is 1/8 of the image size.
  - If ``--bucket_no_upscale`` option is specified, images smaller than the bucket size will be processed without upscaling.
    - Internally, a bucket smaller than the image size is created (for example, if the image is 300x300 and ``bucket_reso_steps=64``, the bucket is 256x256). The image will be trimmed.
    - Implementation of [#130](https://github.com/kohya-ss/sd-scripts/issues/130).
    - Images with an area larger than the maximum size specified by ``--resolution`` are downsampled to the max bucket size.
  - Now the number of data in each batch is limited to the number of actual images (not duplicated). Because a certain bucket may contain smaller number of actual images, so the batch may contain same (duplicated) images.
  - ``--random_crop`` now also works with buckets enabled.
    - Instead of always cropping the center of the image, the image is shifted left, right, up, and down to be used as the training data. This is expected to train to the edges of the image.
    - Implementation of discussion [#34](https://github.com/kohya-ss/sd-scripts/discussions/34).

  - ``--bucket_reso_steps``および``--bucket_no_upscale``オプションを、学習スクリプトおよび``prepare_buckets_latents.py``に追加しました。
  - ``--bucket_reso_steps``オプションでは、bucketの解像度の単位を指定できます。デフォルトは64で、今までと同じ動作です。
    - 1以上の任意の値を指定できます。基本的には64を推奨します。64以外の値では、8で割り切れる値を推奨します。
    - 64未満を指定するとU-Netの内部でpaddingが発生します。どのような結果になるかは未知数です。
    - 8で割り切れない値を指定すると余りはVAE内部で切り捨てられます。
  - ``--bucket_no_upscale``オプションを指定すると、bucketサイズよりも小さい画像は拡大せずそのまま処理します。
    - 内部的には画像サイズ以下のサイズのbucketを作成します（たとえば画像が300x300で``bucket_reso_steps=64``の場合、256x256のbucket）。余りは都度trimmingされます。
    - [#130](https://github.com/kohya-ss/sd-scripts/issues/130) を実装したものです。
    - ``--resolution``で指定した最大サイズよりも面積が大きい画像は、最大サイズと同じ面積になるようアスペクト比を維持したまま縮小され、そのサイズを元にbucketが作られます。
  - これらのオプションによりbucketが細分化され、ひとつのバッチ内に同一画像が重複して存在することが増えたため、バッチサイズを``そのbucketの画像種類数``までに制限する機能を追加しました。
    - たとえば繰り返し回数10で、あるbucketに1枚しか画像がなく、バッチサイズが10以上のとき、今まではepoch内で、同一画像を10枚含むバッチが1回だけ使用されていました。
    - 機能追加後はepoch内にサイズ1のバッチが10回、使用されます。
  - ``--random_crop``がbucketを有効にした場合にも機能するようになりました。
    - 常に画像の中央を切り取るのではなく、左右、上下にずらして教師データにします。これにより画像端まで学習されることが期待されます。
    - discussionの[#34](https://github.com/kohya-ss/sd-scripts/discussions/34)を実装したものです。
    

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
