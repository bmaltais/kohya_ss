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

### Dec 24, 2023 / 2023/12/24

- Fixed to work `tools/convert_diffusers20_original_sd.py`. Thanks to Disty0! PR [#1016](https://github.com/kohya-ss/sd-scripts/pull/1016)

- `tools/convert_diffusers20_original_sd.py` が動かなくなっていたのが修正されました。Disty0 氏に感謝します。 PR [#1016](https://github.com/kohya-ss/sd-scripts/pull/1016) 


### Dec 21, 2023 / 2023/12/21

- The issues in multi-GPU training are fixed. Thanks to Isotr0py! PR [#989](https://github.com/kohya-ss/sd-scripts/pull/989) and [#1000](https://github.com/kohya-ss/sd-scripts/pull/1000)
  - `--ddp_gradient_as_bucket_view` and `--ddp_bucket_view`options are added to `sdxl_train.py`. Please specify these options for multi-GPU training.
- IPEX support is updated. Thanks to Disty0!
- Fixed the bug that the size of the bucket becomes less than `min_bucket_reso`. Thanks to Cauldrath! PR [#1008](https://github.com/kohya-ss/sd-scripts/pull/1008)
- `--sample_at_first` option is added to each training script. This option is useful to generate images at the first step, before training. Thanks to shirayu! PR [#907](https://github.com/kohya-ss/sd-scripts/pull/907)
- `--ss` option is added to the sampling prompt in training. You can specify the scheduler for the sampling like `--ss euler_a`. Thanks to shirayu! PR [#906](https://github.com/kohya-ss/sd-scripts/pull/906)
- `keep_tokens_separator` is added to the dataset config. This option is useful to keep (prevent from shuffling) the tokens in the captions. See [#975](https://github.com/kohya-ss/sd-scripts/pull/975) for details. Thanks to Linaqruf!
  - You can specify the separator with an option like `--keep_tokens_separator "|||"` or with `keep_tokens_separator: "|||"` in `.toml`. The tokens before `|||` are not shuffled.
- Attention processor hook is added. See [#961](https://github.com/kohya-ss/sd-scripts/pull/961) for details. Thanks to rockerBOO!
- The optimizer `PagedAdamW` is added. Thanks to xzuyn! PR [#955](https://github.com/kohya-ss/sd-scripts/pull/955)
- NaN replacement in SDXL VAE is sped up. Thanks to liubo0902! PR [#1009](https://github.com/kohya-ss/sd-scripts/pull/1009)
- Fixed the path error in `finetune/make_captions.py`. Thanks to CjangCjengh! PR [#986](https://github.com/kohya-ss/sd-scripts/pull/986)

- マルチGPUでの学習の不具合を修正しました。Isotr0py 氏に感謝します。 PR [#989](https://github.com/kohya-ss/sd-scripts/pull/989) および [#1000](https://github.com/kohya-ss/sd-scripts/pull/1000)
  - `sdxl_train.py` に `--ddp_gradient_as_bucket_view` と `--ddp_bucket_view` オプションが追加されました。マルチGPUでの学習時にはこれらのオプションを指定してください。
- IPEX サポートが更新されました。Disty0 氏に感謝します。
- Aspect Ratio Bucketing で bucket のサイズが `min_bucket_reso` 未満になる不具合を修正しました。Cauldrath 氏に感謝します。 PR [#1008](https://github.com/kohya-ss/sd-scripts/pull/1008)
- 各学習スクリプトに `--sample_at_first` オプションが追加されました。学習前に画像を生成することで、学習結果が比較しやすくなります。shirayu 氏に感謝します。 PR [#907](https://github.com/kohya-ss/sd-scripts/pull/907)
- 学習時のプロンプトに `--ss` オプションが追加されました。`--ss euler_a` のようにスケジューラを指定できます。shirayu 氏に感謝します。 PR [#906](https://github.com/kohya-ss/sd-scripts/pull/906)
- データセット設定に `keep_tokens_separator` が追加されました。キャプション内のトークンをどの位置までシャッフルしないかを指定できます。詳細は [#975](https://github.com/kohya-ss/sd-scripts/pull/975) を参照してください。Linaqruf 氏に感謝します。
  - オプションで `--keep_tokens_separator "|||"` のように指定するか、`.toml` で `keep_tokens_separator: "|||"` のように指定します。`|||` の前のトークンはシャッフルされません。
- Attention processor hook が追加されました。詳細は [#961](https://github.com/kohya-ss/sd-scripts/pull/961) を参照してください。rockerBOO 氏に感謝します。
- オプティマイザ `PagedAdamW` が追加されました。xzuyn 氏に感謝します。 PR [#955](https://github.com/kohya-ss/sd-scripts/pull/955)
- 学習時、SDXL VAE で NaN が発生した時の置き換えが高速化されました。liubo0902 氏に感謝します。 PR [#1009](https://github.com/kohya-ss/sd-scripts/pull/1009)
- `finetune/make_captions.py` で相対パス指定時のエラーが修正されました。CjangCjengh 氏に感謝します。 PR [#986](https://github.com/kohya-ss/sd-scripts/pull/986)

### Dec 3, 2023 / 2023/12/3

- `finetune\tag_images_by_wd14_tagger.py` now supports the separator other than `,` with `--caption_separator` option. Thanks to KohakuBlueleaf! PR [#913](https://github.com/kohya-ss/sd-scripts/pull/913)
- Min SNR Gamma with V-predicition (SD 2.1) is fixed. Thanks to feffy380! PR[#934](https://github.com/kohya-ss/sd-scripts/pull/934)
  - See [#673](https://github.com/kohya-ss/sd-scripts/issues/673) for details.
- `--min_diff` and `--clamp_quantile` options are added to `networks/extract_lora_from_models.py`. Thanks to wkpark! PR [#936](https://github.com/kohya-ss/sd-scripts/pull/936)
  - The default values are same as the previous version.
- Deep Shrink hires fix is supported in `sdxl_gen_img.py` and `gen_img_diffusers.py`.
  - `--ds_timesteps_1` and `--ds_timesteps_2` options denote the timesteps of the Deep Shrink for the first and second stages.
  - `--ds_depth_1` and `--ds_depth_2` options denote the depth (block index) of the Deep Shrink for the first and second stages.
  - `--ds_ratio` option denotes the ratio of the Deep Shrink. `0.5` means the half of the original latent size for the Deep Shrink.
  - `--dst1`, `--dst2`, `--dsd1`, `--dsd2` and `--dsr` prompt options are also available.

- `finetune\tag_images_by_wd14_tagger.py` で `--caption_separator` オプションでカンマ以外の区切り文字を指定できるようになりました。KohakuBlueleaf 氏に感謝します。 PR [#913](https://github.com/kohya-ss/sd-scripts/pull/913)
- V-predicition (SD 2.1) での Min SNR Gamma が修正されました。feffy380 氏に感謝します。 PR[#934](https://github.com/kohya-ss/sd-scripts/pull/934)
  - 詳細は [#673](https://github.com/kohya-ss/sd-scripts/issues/673) を参照してください。
- `networks/extract_lora_from_models.py` に `--min_diff` と `--clamp_quantile` オプションが追加されました。wkpark 氏に感謝します。 PR [#936](https://github.com/kohya-ss/sd-scripts/pull/936)
  - デフォルト値は前のバージョンと同じです。
- `sdxl_gen_img.py` と `gen_img_diffusers.py` で Deep Shrink hires fix をサポートしました。
  - `--ds_timesteps_1` と `--ds_timesteps_2` オプションは Deep Shrink の第一段階と第二段階の timesteps を指定します。
  - `--ds_depth_1` と `--ds_depth_2` オプションは Deep Shrink の第一段階と第二段階の深さ（ブロックの index）を指定します。
  - `--ds_ratio` オプションは Deep Shrink の比率を指定します。`0.5` を指定すると Deep Shrink 適用時の latent は元のサイズの半分になります。
  - `--dst1`、`--dst2`、`--dsd1`、`--dsd2`、`--dsr` プロンプトオプションも使用できます。

### Nov 5, 2023 / 2023/11/5

- `sdxl_train.py` now supports different learning rates for each Text Encoder.
  - Example:
    - `--learning_rate 1e-6`: train U-Net only
    - `--train_text_encoder --learning_rate 1e-6`: train U-Net and two Text Encoders with the same learning rate (same as the previous version)
    - `--train_text_encoder --learning_rate 1e-6 --learning_rate_te1 1e-6 --learning_rate_te2 1e-6`: train U-Net and two Text Encoders with the different learning rates
    - `--train_text_encoder --learning_rate 0 --learning_rate_te1 1e-6 --learning_rate_te2 1e-6`: train two Text Encoders only 
    - `--train_text_encoder --learning_rate 1e-6 --learning_rate_te1 1e-6 --learning_rate_te2 0`: train U-Net and one Text Encoder only
    - `--train_text_encoder --learning_rate 0 --learning_rate_te1 0 --learning_rate_te2 1e-6`: train one Text Encoder only

- `train_db.py` and `fine_tune.py` now support different learning rates for Text Encoder. Specify with `--learning_rate_te` option. 
  - To train Text Encoder with `fine_tune.py`, specify `--train_text_encoder` option too. `train_db.py` trains Text Encoder by default.

- Fixed the bug that Text Encoder is not trained when block lr is specified in `sdxl_train.py`.

- Debiased Estimation loss is added to each training script. Thanks to sdbds!
  - Specify `--debiased_estimation_loss` option to enable it. See PR [#889](https://github.com/kohya-ss/sd-scripts/pull/889) for details.
- Training of Text Encoder is improved in `train_network.py` and `sdxl_train_network.py`. Thanks to KohakuBlueleaf! PR [#895](https://github.com/kohya-ss/sd-scripts/pull/895)
- The moving average of the loss is now displayed in the progress bar in each training script. Thanks to shirayu! PR [#899](https://github.com/kohya-ss/sd-scripts/pull/899)
- PagedAdamW32bit optimizer is supported. Specify `--optimizer_type=PagedAdamW32bit`. Thanks to xzuyn! PR [#900](https://github.com/kohya-ss/sd-scripts/pull/900)
- Other bug fixes and improvements.

- `sdxl_train.py` で、二つのText Encoderそれぞれに独立した学習率が指定できるようになりました。サンプルは上の英語版を参照してください。
- `train_db.py` および `fine_tune.py` で Text Encoder に別の学習率を指定できるようになりました。`--learning_rate_te` オプションで指定してください。
  - `fine_tune.py` で Text Encoder を学習するには `--train_text_encoder` オプションをあわせて指定してください。`train_db.py` はデフォルトで学習します。
- `sdxl_train.py` で block lr を指定すると Text Encoder が学習されない不具合を修正しました。
- Debiased Estimation loss が各学習スクリプトに追加されました。sdbsd 氏に感謝します。
  - `--debiased_estimation_loss` を指定すると有効になります。詳細は PR [#889](https://github.com/kohya-ss/sd-scripts/pull/889) を参照してください。
- `train_network.py` と `sdxl_train_network.py` でText Encoderの学習が改善されました。KohakuBlueleaf 氏に感謝します。 PR [#895](https://github.com/kohya-ss/sd-scripts/pull/895)
- 各学習スクリプトで移動平均のlossがプログレスバーに表示されるようになりました。shirayu 氏に感謝します。 PR [#899](https://github.com/kohya-ss/sd-scripts/pull/899)
- PagedAdamW32bit オプティマイザがサポートされました。`--optimizer_type=PagedAdamW32bit` と指定してください。xzuyn 氏に感謝します。 PR [#900](https://github.com/kohya-ss/sd-scripts/pull/900)
- その他のバグ修正と改善。


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

