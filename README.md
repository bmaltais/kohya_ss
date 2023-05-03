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

Most of the documents are written in Japanese.

* [Training guide - common](./train_README-ja.md) : data preparation, options etc... 
  * [Chinese version](./train_README-zh.md)
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

### Optional: Use Lion8bit

For Lion8bit, you need to upgrade `bitsandbytes` to 0.38.0 or later. Uninstall `bitsandbytes`, and for Windows, install the Windows version whl file from [here](https://github.com/jllllll/bitsandbytes-windows-webui) or other sources, like:

```powershell
pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl
```

For upgrading, upgrade this repo with `pip install .`, and upgrade necessary packages manually.

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

### 3 May 2023, 2023/05/03

-  When saving v2 models in Diffusers format in training scripts and conversion scripts, it was found that the U-Net configuration is different from those of Hugging Face's stabilityai models (this repository is `"use_linear_projection": false`, stabilityai is `true`). Please note that the weight shapes are different, so please be careful when using the weight files directly. We apologize for the inconvenience.
    - Since the U-Net model is created based on the configuration, it should not cause any problems in training or inference.
    - Added `--unet_use_linear_projection` option to `convert_diffusers20_original_sd.py` script. If you specify this option, you can save a Diffusers format model with the same configuration as stabilityai's model from an SD format model (a single `*.safetensors` or `*.ckpt` file). Unfortunately, it is not possible to convert a Diffusers format model to the same format.

- Lion8bit optimizer is supported. [PR #447](https://github.com/kohya-ss/sd-scripts/pull/447) Thanks to sdbds!
  - Currently it is optional because you need to update `bitsandbytes` version. See "Optional: Use Lion8bit" in installation instructions to use it.
- Multi-GPU training with DDP is supported in each training script. [PR #448](https://github.com/kohya-ss/sd-scripts/pull/448) Thanks to Isotr0py!
- Multi resolution noise (pyramid noise) is supported in each training script. [PR #471](https://github.com/kohya-ss/sd-scripts/pull/471) Thanks to pamparamm!
  - See PR and this page [Multi-Resolution Noise for Diffusion Model Training](https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2) for details.

- 学習スクリプトや変換スクリプトでDiffusers形式でv2モデルを保存するとき、U-Netの設定がHugging Faceのstabilityaiのモデルと異なることがわかりました（当リポジトリでは `"use_linear_projection": false`、stabilityaiは`true`）。重みの形状が異なるため、直接重みファイルを利用する場合にはご注意ください。ご不便をお掛けし申し訳ありません。
  - U-Netのモデルは設定に基づいて作成されるため、通常、学習や推論で問題になることはないと思われます。
  - `convert_diffusers20_original_sd.py`スクリプトに`--unet_use_linear_projection`オプションを追加しました。これを指定するとSD形式のモデル（単一の`*.safetensors`または`*.ckpt`ファイル）から、stabilityaiのモデルと同じ形状の重みファイルを持つDiffusers形式モデルが保存できます。なお、Diffusers形式のモデルを同形式に変換することはできません。

- Lion8bitオプティマイザがサポートされました。[PR #447](https://github.com/kohya-ss/sd-scripts/pull/447) sdbds氏に感謝します。
  - `bitsandbytes`のバージョンを更新する必要があるため、現在はオプションです。使用するにはインストール手順の「[オプション：Lion8bitを使う](./README-ja.md#オプションlion8bitを使う)」を参照してください。
- 各学習スクリプトでDDPによるマルチGPU学習がサポートされました。[PR #448](https://github.com/kohya-ss/sd-scripts/pull/448) Isotr0py氏に感謝します。
- Multi resolution noise (pyramid noise) が各学習スクリプトでサポートされました。[PR #471](https://github.com/kohya-ss/sd-scripts/pull/471) pamparamm氏に感謝します。
  - 詳細はPRおよびこちらのページ [Multi-Resolution Noise for Diffusion Model Training](https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2) を参照してください。

### 30 Apr. 2023, 2023/04/30

- Added Chinese translation of [DreamBooth guide](./train_db_README-zh.md) and [LoRA guide](./train_network_README-zh.md). [PR #459](https://github.com/kohya-ss/sd-scripts/pull/459) Thanks to tomj2ee!
- Added [documentation](./gen_img_README-ja.md) for image generation script `gen_img_diffusers.py` (Japanese version only).
- 中国語版の[DreamBoothガイド](./train_db_README-zh.md)と[LoRAガイド](./train_network_README-zh.md)が追加されました。 [PR #459](https://github.com/kohya-ss/sd-scripts/pull/459) tomj2ee氏に感謝します。
- 画像生成スクリプト `gen_img_diffusers.py`の簡単な[ドキュメント](./gen_img_README-ja.md)を追加しました（日本語版のみ）。 

### 26 Apr. 2023, 2023/04/26

- Added [Chinese translation](./train_README-zh.md) of training guide. [PR #445](https://github.com/kohya-ss/sd-scripts/pull/445) Thanks to tomj2ee!
- `tag_images_by_wd14_tagger.py` can now get arguments from outside. [PR #453](https://github.com/kohya-ss/sd-scripts/pull/453) Thanks to mio2333!
- 学習に関するドキュメントの[中国語版](./train_README-zh.md)が追加されました。 [PR #445](https://github.com/kohya-ss/sd-scripts/pull/445) tomj2ee氏に感謝します。
- `tag_images_by_wd14_tagger.py`の引数を外部から取得できるようになりました。 [PR #453](https://github.com/kohya-ss/sd-scripts/pull/453) mio2333氏に感謝します。

### 25 Apr. 2023, 2023/04/25

- Please do not update for a while if you cannot revert the repository to the previous version when something goes wrong, because the model saving part has been changed.
- Added `--save_every_n_steps` option to each training script. The model is saved every specified steps.
  - `--save_last_n_steps` option can be used to save only the specified number of models (old models will be deleted).
  - If you specify the `--save_state` option, the state will also be saved at the same time. You can specify the number of steps to keep the state with the `--save_last_n_steps_state` option (the same value as `--save_last_n_steps` is used if omitted).
  - You can use the epoch-based model saving and state saving options together.
  - Not tested in multi-GPU environment. Please report any bugs.
- `--cache_latents_to_disk` option automatically enables `--cache_latents` option when specified. [#438](https://github.com/kohya-ss/sd-scripts/issues/438)
- Fixed a bug in `gen_img_diffusers.py` where latents upscaler would fail with a batch size of 2 or more.
  
- モデル保存部分を変更していますので、何か不具合が起きた時にリポジトリを前のバージョンに戻せない場合には、しばらく更新を控えてください。
- 各学習スクリプトに`--save_every_n_steps`オプションを追加しました。指定ステップごとにモデルを保存します。
  - `--save_last_n_steps`オプションに数値を指定すると、そのステップ数のモデルのみを保存します（古いモデルは削除されます）。
  - `--save_state`オプションを指定するとstateも同時に保存します。`--save_last_n_steps_state`オプションでstateを残すステップ数を指定できます（省略時は`--save_last_n_steps`と同じ値が使われます）。
  - エポックごとのモデル保存、state保存のオプションと共存できます。
  - マルチGPU環境でのテストを行っていないため、不具合等あればご報告ください。
- `--cache_latents_to_disk`オプションが指定されたとき、`--cache_latents`オプションが自動的に有効になるようにしました。 [#438](https://github.com/kohya-ss/sd-scripts/issues/438)
- `gen_img_diffusers.py`でlatents upscalerがバッチサイズ2以上でエラーとなる不具合を修正しました。


Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.
最近の更新情報は [Release](https://github.com/kohya-ss/sd-scripts/releases) をご覧ください。

### Naming of LoRA

The LoRA supported by `train_network.py` has been named to avoid confusion. The documentation has been updated. The following are the names of LoRA types in this repository.

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers)

    LoRA for Linear layers and Conv2d layers with 1x1 kernel

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers)

    In addition to 1., LoRA for Conv2d layers with 3x3 kernel 
    
LoRA-LierLa is the default LoRA type for `train_network.py` (without `conv_dim` network arg). LoRA-LierLa can be used with [our extension](https://github.com/kohya-ss/sd-webui-additional-networks) for AUTOMATIC1111's Web UI, or with the built-in LoRA feature of the Web UI.

To use LoRA-C3Liar with Web UI, please use our extension.

### LoRAの名称について

`train_network.py` がサポートするLoRAについて、混乱を避けるため名前を付けました。ドキュメントは更新済みです。以下は当リポジトリ内の独自の名称です。

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers、リエラと読みます)

    Linear 層およびカーネルサイズ 1x1 の Conv2d 層に適用されるLoRA

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers、セリアと読みます)

    1.に加え、カーネルサイズ 3x3 の Conv2d 層に適用されるLoRA

LoRA-LierLa は[Web UI向け拡張](https://github.com/kohya-ss/sd-webui-additional-networks)、またはAUTOMATIC1111氏のWeb UIのLoRA機能で使用することができます。

LoRA-C3Liarを使いWeb UIで生成するには拡張を使用してください。

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

