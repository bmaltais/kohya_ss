__SDXL is now supported. The sdxl branch has been merged into the main branch. If you update the repository, please follow the upgrade instructions. Also, the version of accelerate has been updated, so please run accelerate config again.__ The documentation for SDXL training is [here](./README.md#sdxl-training).

This repository contains training, generation and utility scripts for Stable Diffusion.

[__Change History__](#change-history) is moved to the bottom of the page. 
更新履歴は[ページ末尾](#change-history)に移しました。

[日本語版READMEはこちら](./README-ja.md)

For easier use (GUI and PowerShell scripts etc...), please visit [the repository maintained by bmaltais](https://github.com/bmaltais/kohya_ss). Thanks to @bmaltais!

This repository contains the scripts for:

* DreamBooth training, including U-Net and Text Encoder
* Fine-tuning (native training), including U-Net and Text Encoder
* LoRA training
* Textual Inversion training
* Image generation
* Model conversion (supports 1.x and 2.x, Stable Diffision ckpt/safetensors and Diffusers)

## About requirements.txt

These files do not contain requirements for PyTorch. Because the versions of them depend on your environment. Please install PyTorch at first (see installation guide below.) 

The scripts are tested with Pytorch 2.0.1. 1.12.1 is not tested but should work.

## Links to usage documentation

Most of the documents are written in Japanese.

[English translation by darkstorm2150 is here](https://github.com/darkstorm2150/sd-scripts#links-to-usage-documentation). Thanks to darkstorm2150!

* [Training guide - common](./docs/train_README-ja.md) : data preparation, options etc... 
  * [Chinese version](./docs/train_README-zh.md)
* [Dataset config](./docs/config_README-ja.md) 
* [DreamBooth training guide](./docs/train_db_README-ja.md)
* [Step by Step fine-tuning guide](./docs/fine_tune_README_ja.md):
* [training LoRA](./docs/train_network_README-ja.md)
* [training Textual Inversion](./docs/train_ti_README-ja.md)
* [Image generation](./docs/gen_img_README-ja.md)
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

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.20

accelerate config
```

__Note:__ Now bitsandbytes is optional. Please install any version of bitsandbytes as needed. Installation instructions are in the following section.

<!-- 
cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
-->
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

### Optional: Use `bitsandbytes` (8bit optimizer)

For 8bit optimizer, you need to install `bitsandbytes`. For Linux, please install `bitsandbytes` as usual (0.41.1 or later is recommended.)

For Windows, there are several versions of `bitsandbytes`:

- `bitsandbytes` 0.35.0: Stable version. AdamW8bit is available. `full_bf16` is not available.
- `bitsandbytes` 0.41.1: Lion8bit, PagedAdamW8bit and PagedLion8bit are available. `full_bf16` is available.

Note: `bitsandbytes`above 0.35.0 till 0.41.0 seems to have an issue: https://github.com/TimDettmers/bitsandbytes/issues/659

Follow the instructions below to install `bitsandbytes` for Windows.

### bitsandbytes 0.35.0 for Windows

Open a regular Powershell terminal and type the following inside:

```powershell
cd sd-scripts
.\venv\Scripts\activate
pip install bitsandbytes==0.35.0

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
```

This will install `bitsandbytes` 0.35.0 and copy the necessary files to the `bitsandbytes` directory.

### bitsandbytes 0.41.1 for Windows

Install the Windows version whl file from [here](https://github.com/jllllll/bitsandbytes-windows-webui) or other sources, like:

```powershell
python -m pip install bitsandbytes==0.41.1 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```

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


## SDXL training

The documentation in this section will be moved to a separate document later.

### Training scripts for SDXL

- `sdxl_train.py` is a script for SDXL fine-tuning. The usage is almost the same as `fine_tune.py`, but it also supports DreamBooth dataset.
  - `--full_bf16` option is added. Thanks to KohakuBlueleaf!
    - This option enables the full bfloat16 training (includes gradients). This option is useful to reduce the GPU memory usage. 
    - The full bfloat16 training might be unstable. Please use it at your own risk.
  - The different learning rates for each U-Net block are now supported in sdxl_train.py. Specify with `--block_lr` option. Specify 23 values separated by commas like `--block_lr 1e-3,1e-3 ... 1e-3`.
    - 23 values correspond to `0: time/label embed, 1-9: input blocks 0-8, 10-12: mid blocks 0-2, 13-21: output blocks 0-8, 22: out`.
- `prepare_buckets_latents.py` now supports SDXL fine-tuning.

- `sdxl_train_network.py` is a script for LoRA training for SDXL. The usage is almost the same as `train_network.py`.

- Both scripts has following additional options:
  - `--cache_text_encoder_outputs` and `--cache_text_encoder_outputs_to_disk`: Cache the outputs of the text encoders. This option is useful to reduce the GPU memory usage. This option cannot be used with options for shuffling or dropping the captions.
  - `--no_half_vae`: Disable the half-precision (mixed-precision) VAE. VAE for SDXL seems to produce NaNs in some cases. This option is useful to avoid the NaNs.

- `--weighted_captions` option is not supported yet for both scripts.

- `sdxl_train_textual_inversion.py` is a script for Textual Inversion training for SDXL. The usage is almost the same as `train_textual_inversion.py`.
  - `--cache_text_encoder_outputs` is not supported.
  - There are two options for captions:
    1. Training with captions. All captions must include the token string. The token string is replaced with multiple tokens.
    2. Use `--use_object_template` or `--use_style_template` option. The captions are generated from the template. The existing captions are ignored.
  - See below for the format of the embeddings.

- `--min_timestep` and `--max_timestep` options are added to each training script. These options can be used to train U-Net with different timesteps. The default values are 0 and 1000.

### Utility scripts for SDXL

- `tools/cache_latents.py` is added. This script can be used to cache the latents to disk in advance. 
  - The options are almost the same as `sdxl_train.py'. See the help message for the usage.
  - Please launch the script as follows:
    `accelerate launch  --num_cpu_threads_per_process 1 tools/cache_latents.py ...`
  - This script should work with multi-GPU, but it is not tested in my environment.

- `tools/cache_text_encoder_outputs.py` is added. This script can be used to cache the text encoder outputs to disk in advance. 
  - The options are almost the same as `cache_latents.py` and `sdxl_train.py`. See the help message for the usage.

- `sdxl_gen_img.py` is added. This script can be used to generate images with SDXL, including LoRA, Textual Inversion and ControlNet-LLLite. See the help message for the usage.

### Tips for SDXL training

- The default resolution of SDXL is 1024x1024.
- The fine-tuning can be done with 24GB GPU memory with the batch size of 1. For 24GB GPU, the following options are recommended __for the fine-tuning with 24GB GPU memory__:
  - Train U-Net only.
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` option and caching latents.
  - Use Adafactor optimizer. RMSprop 8bit or Adagrad 8bit may work. AdamW 8bit doesn't seem to work.
- The LoRA training can be done with 8GB GPU memory (10GB recommended). For reducing the GPU memory usage, the following options are recommended:
  - Train U-Net only.
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` option and caching latents.
  - Use one of 8bit optimizers or Adafactor optimizer.
  - Use lower dim (4 to 8 for 8GB GPU).
- `--network_train_unet_only` option is highly recommended for SDXL LoRA. Because SDXL has two text encoders, the result of the training will be unexpected.
- PyTorch 2 seems to use slightly less GPU memory than PyTorch 1.
- `--bucket_reso_steps` can be set to 32 instead of the default value 64. Smaller values than 32 will not work for SDXL training.

Example of the optimizer settings for Adafactor with the fixed learning rate:
```toml
optimizer_type = "adafactor"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False" ]
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100
learning_rate = 4e-7 # SDXL original learning rate
```

### Format of Textual Inversion embeddings for SDXL

```python
from safetensors.torch import save_file

state_dict = {"clip_g": embs_for_text_encoder_1280, "clip_l": embs_for_text_encoder_768}
save_file(state_dict, file)
```

### ControlNet-LLLite

ControlNet-LLLite, a novel method for ControlNet with SDXL, is added. See [documentation](./docs/train_lllite_README.md) for details.


## Change History

### Jan 27, 2024 / 2024/1/27: v0.8.3

- Fixed a bug that the training crashes when `--fp8_base` is specified with `--save_state`. PR [#1079](https://github.com/kohya-ss/sd-scripts/pull/1079) Thanks to feffy380!
  - `safetensors` is updated. Please see [Upgrade](#upgrade) and update the library.
- Fixed a bug that the training crashes when `network_multiplier` is specified with multi-GPU training. PR [#1084](https://github.com/kohya-ss/sd-scripts/pull/1084) Thanks to fireicewolf!
- Fixed a bug that the training crashes when training ControlNet-LLLite.

- `--fp8_base` 指定時に `--save_state` での保存がエラーになる不具合が修正されました。 PR [#1079](https://github.com/kohya-ss/sd-scripts/pull/1079) feffy380 氏に感謝します。
  - `safetensors` がバージョンアップされていますので、[Upgrade](#upgrade) を参照し更新をお願いします。
- 複数 GPU での学習時に `network_multiplier` を指定するとクラッシュする不具合が修正されました。 PR [#1084](https://github.com/kohya-ss/sd-scripts/pull/1084) fireicewolf 氏に感謝します。
- ControlNet-LLLite の学習がエラーになる不具合を修正しました。 

### Jan 23, 2024 / 2024/1/23: v0.8.2

- [Experimental] The `--fp8_base` option is added to the training scripts for LoRA etc. The base model (U-Net, and Text Encoder when training modules for Text Encoder) can be trained with fp8. PR [#1057](https://github.com/kohya-ss/sd-scripts/pull/1057) Thanks to KohakuBlueleaf!
  - Please specify `--fp8_base` in `train_network.py` or `sdxl_train_network.py`.
  - PyTorch 2.1 or later is required.
  - If you use xformers with PyTorch 2.1, please see [xformers repository](https://github.com/facebookresearch/xformers) and install the appropriate version according to your CUDA version.
  - The sample image generation during training consumes a lot of memory. It is recommended to turn it off.

- [Experimental] The network multiplier can be specified for each dataset in the training scripts for LoRA etc.
  - This is an experimental option and may be removed or changed in the future.
  - For example, if you train with state A as `1.0` and state B as `-1.0`, you may be able to generate by switching between state A and B depending on the LoRA application rate.
  - Also, if you prepare five states and train them as `0.2`, `0.4`, `0.6`, `0.8`, and `1.0`, you may be able to generate by switching the states smoothly depending on the application rate.
  - Please specify `network_multiplier` in `[[datasets]]` in `.toml` file.
- Some options are added to `networks/extract_lora_from_models.py` to reduce the memory usage.
  - `--load_precision` option can be used to specify the precision when loading the model. If the model is saved in fp16, you can reduce the memory usage by specifying `--load_precision fp16` without losing precision.
  - `--load_original_model_to` option can be used to specify the device to load the original model. `--load_tuned_model_to` option can be used to specify the device to load the derived model. The default is `cpu` for both options, but you can specify `cuda` etc. You can reduce the memory usage by loading one of them to GPU. This option is available only for SDXL.

- The gradient synchronization in LoRA training with multi-GPU is improved. PR [#1064](https://github.com/kohya-ss/sd-scripts/pull/1064) Thanks to KohakuBlueleaf!
- The code for Intel IPEX support is improved. PR [#1060](https://github.com/kohya-ss/sd-scripts/pull/1060) Thanks to akx!
- Fixed a bug in multi-GPU Textual Inversion training.

- （実験的）　LoRA等の学習スクリプトで、ベースモデル（U-Net、および Text Encoder のモジュール学習時は Text Encoder も）の重みを fp8 にして学習するオプションが追加されました。 PR [#1057](https://github.com/kohya-ss/sd-scripts/pull/1057) KohakuBlueleaf 氏に感謝します。
  - `train_network.py` または `sdxl_train_network.py` で `--fp8_base` を指定してください。
  - PyTorch 2.1 以降が必要です。
  - PyTorch 2.1 で xformers を使用する場合は、[xformers のリポジトリ](https://github.com/facebookresearch/xformers) を参照し、CUDA バージョンに応じて適切なバージョンをインストールしてください。
  - 学習中のサンプル画像生成はメモリを大量に消費するため、オフにすることをお勧めします。
- (実験的)　LoRA 等の学習で、データセットごとに異なるネットワーク適用率を指定できるようになりました。 
  - 実験的オプションのため、将来的に削除または仕様変更される可能性があります。
  - たとえば状態 A を `1.0`、状態 B を `-1.0` として学習すると、LoRA の適用率に応じて状態 A と B を切り替えつつ生成できるかもしれません。
  - また、五段階の状態を用意し、それぞれ `0.2`、`0.4`、`0.6`、`0.8`、`1.0` として学習すると、適用率でなめらかに状態を切り替えて生成できるかもしれません。 
  - `.toml` ファイルで `[[datasets]]` に `network_multiplier` を指定してください。
- `networks/extract_lora_from_models.py` に使用メモリ量を削減するいくつかのオプションを追加しました。 
  - `--load_precision` で読み込み時の精度を指定できます。モデルが fp16 で保存されている場合は `--load_precision fp16` を指定して精度を変えずにメモリ量を削減できます。
  - `--load_original_model_to` で元モデルを読み込むデバイスを、`--load_tuned_model_to` で派生モデルを読み込むデバイスを指定できます。デフォルトは両方とも `cpu` ですがそれぞれ `cuda` 等を指定できます。片方を GPU に読み込むことでメモリ量を削減できます。SDXL の場合のみ有効です。
- マルチ GPU での LoRA 等の学習時に勾配の同期が改善されました。 PR [#1064](https://github.com/kohya-ss/sd-scripts/pull/1064) KohakuBlueleaf 氏に感謝します。
- Intel IPEX サポートのコードが改善されました。PR [#1060](https://github.com/kohya-ss/sd-scripts/pull/1060) akx 氏に感謝します。
- マルチ GPU での Textual Inversion 学習の不具合を修正しました。

- `.toml` example for network multiplier / ネットワーク適用率の `.toml` の記述例

```toml
[general]
[[datasets]]
resolution = 512
batch_size = 8
network_multiplier = 1.0

... subset settings ...

[[datasets]]
resolution = 512
batch_size = 8
network_multiplier = -1.0

... subset settings ...
```


### Jan 17, 2024 / 2024/1/17: v0.8.1

- Fixed a bug that the VRAM usage without Text Encoder training is larger than before in training scripts for LoRA etc (`train_network.py`, `sdxl_train_network.py`).
  - Text Encoders were not moved to CPU.
- Fixed typos. Thanks to akx! [PR #1053](https://github.com/kohya-ss/sd-scripts/pull/1053)

- LoRA 等の学習スクリプト（`train_network.py`、`sdxl_train_network.py`）で、Text Encoder を学習しない場合の VRAM 使用量が以前に比べて大きくなっていた不具合を修正しました。 
  - Text Encoder が GPU に保持されたままになっていました。
- 誤字が修正されました。 [PR #1053](https://github.com/kohya-ss/sd-scripts/pull/1053) akx 氏に感謝します。

### Jan 15, 2024 / 2024/1/15: v0.8.0

- Diffusers, Accelerate, Transformers and other related libraries have been updated. Please update the libraries with [Upgrade](#upgrade).
  - Some model files (Text Encoder without position_id) based on the latest Transformers can be loaded.
- `torch.compile` is supported (experimental). PR [#1024](https://github.com/kohya-ss/sd-scripts/pull/1024) Thanks to p1atdev!
  - This feature works only on Linux or WSL.
  - Please specify `--torch_compile` option in each training script.
  - You can select the backend with `--dynamo_backend` option. The default is `"inductor"`. `inductor` or `eager` seems to work.
  - Please use `--sdpa` option instead of `--xformers` option.
  - PyTorch 2.1 or later is recommended.
  - Please see [PR](https://github.com/kohya-ss/sd-scripts/pull/1024) for details.
- The session name for wandb can be specified with `--wandb_run_name` option. PR [#1032](https://github.com/kohya-ss/sd-scripts/pull/1032) Thanks to hopl1t!
- IPEX library is updated. PR [#1030](https://github.com/kohya-ss/sd-scripts/pull/1030) Thanks to Disty0!
- Fixed a bug that Diffusers format model cannot be saved.

- Diffusers、Accelerate、Transformers 等の関連ライブラリを更新しました。[Upgrade](#upgrade) を参照し更新をお願いします。
  - 最新の Transformers を前提とした一部のモデルファイル（Text Encoder が position_id を持たないもの）が読み込めるようになりました。
- `torch.compile` がサポートされしました（実験的）。 PR [#1024](https://github.com/kohya-ss/sd-scripts/pull/1024) p1atdev 氏に感謝します。
  - Linux または WSL でのみ動作します。
  - 各学習スクリプトで `--torch_compile` オプションを指定してください。
  - `--dynamo_backend` オプションで使用される backend を選択できます。デフォルトは `"inductor"` です。 `inductor` または `eager` が動作するようです。
  - `--xformers` オプションとは互換性がありません。 代わりに `--sdpa` オプションを使用してください。
  - PyTorch 2.1以降を推奨します。
  - 詳細は [PR](https://github.com/kohya-ss/sd-scripts/pull/1024) をご覧ください。
- wandb 保存時のセッション名が各学習スクリプトの `--wandb_run_name` オプションで指定できるようになりました。 PR [#1032](https://github.com/kohya-ss/sd-scripts/pull/1032) hopl1t 氏に感謝します。
- IPEX ライブラリが更新されました。[PR #1030](https://github.com/kohya-ss/sd-scripts/pull/1030) Disty0 氏に感謝します。
- Diffusers 形式でのモデル保存ができなくなっていた不具合を修正しました。


Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.
最近の更新情報は [Release](https://github.com/kohya-ss/sd-scripts/releases) をご覧ください。

### Naming of LoRA

The LoRA supported by `train_network.py` has been named to avoid confusion. The documentation has been updated. The following are the names of LoRA types in this repository.

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers)

    LoRA for Linear layers and Conv2d layers with 1x1 kernel

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers)

    In addition to 1., LoRA for Conv2d layers with 3x3 kernel 
    
LoRA-LierLa is the default LoRA type for `train_network.py` (without `conv_dim` network arg). LoRA-LierLa can be used with [our extension](https://github.com/kohya-ss/sd-webui-additional-networks) for AUTOMATIC1111's Web UI, or with the built-in LoRA feature of the Web UI.

To use LoRA-C3Lier with Web UI, please use our extension.

### LoRAの名称について

`train_network.py` がサポートするLoRAについて、混乱を避けるため名前を付けました。ドキュメントは更新済みです。以下は当リポジトリ内の独自の名称です。

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers、リエラと読みます)

    Linear 層およびカーネルサイズ 1x1 の Conv2d 層に適用されるLoRA

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers、セリアと読みます)

    1.に加え、カーネルサイズ 3x3 の Conv2d 層に適用されるLoRA

LoRA-LierLa は[Web UI向け拡張](https://github.com/kohya-ss/sd-webui-additional-networks)、またはAUTOMATIC1111氏のWeb UIのLoRA機能で使用することができます。

LoRA-C3Lierを使いWeb UIで生成するには拡張を使用してください。

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

