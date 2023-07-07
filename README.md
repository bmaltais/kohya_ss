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

__Stable Diffusion web UI now seems to support LoRA trained by ``sd-scripts``.__ Thank you for great work!!! 

## About SDXL training

The feature of SDXL training is now available in sdxl branch as an experimental feature. 

Summary of the feature:

- `sdxl_train.py` is a script for SDXL fine-tuning. The usage is almost the same as `fine_tune.py`, but it also supports DreamBooth dataset.
  - `prepare_buckets_latents.py` now supports SDXL fine-tuning.
- `sdxl_train_network.py` is a script for LoRA training for SDXL. The usage is almost the same as `train_network.py`.
- Both scripts has following additional options:
  - `--cache_text_encoder_outputs`: Cache the outputs of the text encoders. This option is useful to reduce the GPU memory usage. This option cannot be used with options for shuffling or dropping the captions.
  - `--no_half_vae`: Disable the half-precision (mixed-precision) VAE. VAE for SDXL seems to produce NaNs in some cases. This option is useful to avoid the NaNs.
- The image generation during training is now available. However, the VAE for SDXL seems to produce NaNs in some cases when using `fp16`. The images will be black. Currently, the NaNs cannot be avoided even with `--no_half_vae` option. It works with `bf16` or without mixed precision.
- `--weighted_captions` option is not supported yet.
- `--min_timestep` and `--max_timestep` options are added to each training script. These options can be used to train U-Net with different timesteps. The default values are 0 and 1000.

`requirements.txt` is updated to support SDXL training. 

### Tips for SDXL training

- The default resolution of SDXL is 1024x1024.
- The fine-tuning can be done with 24GB GPU memory with the batch size of 1. For 24GB GPU, the following options are recommended:
  - Train U-Net only.
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` option and caching latents.
  - Use Adafactor optimizer. RMSprop 8bit or Adagrad 8bit may work. AdamW 8bit doesn't seem to work.
- The LoRA training can be done with 12GB GPU memory.
- `--network_train_unet_only` option is highly recommended for SDXL LoRA. Because SDXL has two text encoders, the result of the training will be unexpected.
- PyTorch 2 seems to use slightly less GPU memory than PyTorch 1.

Example of the optimizer settings for Adafactor with the fixed learning rate:
```
optimizer_type = "adafactor"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False" ]
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100
learning_rate = 4e-7 # SDXL original learning rate
```

## About requirements.txt

These files do not contain requirements for PyTorch. Because the versions of them depend on your environment. Please install PyTorch at first (see installation guide below.) 

The scripts are tested with PyTorch 1.12.1 and 2.0.1, Diffusers 0.17.1.

## Links to how-to-use documents

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

### Experimental: Use PyTorch 2.0

In this case, you need to install PyTorch 2.0 and xformers 0.0.20. Instead of the above, please type the following:

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.20

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config
```

Answers to accelerate config should be the same as above.

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

### 15 Jun. 2023, 2023/06/15

- Prodigy optimizer is supported in each training script. It is a member of D-Adaptation and is effective for DyLoRA training. [PR #585](https://github.com/kohya-ss/sd-scripts/pull/585) Please see the PR for details. Thanks to sdbds!
  - Install the package with `pip install prodigyopt`. Then specify the option like `--optimizer_type="prodigy"`.
- Arbitrary Dataset is supported in each training script (except XTI). You can use it by defining a Dataset class that returns images and captions.
  - Prepare a Python script and define a class that inherits `train_util.MinimalDataset`. Then specify the option like `--dataset_class package.module.DatasetClass` in each training script.
  - Please refer to `MinimalDataset` for implementation. I will prepare a sample later.
- The following features have been added to the generation script.
  - Added an option `--highres_fix_disable_control_net` to disable ControlNet in the 2nd stage of Highres. Fix. Please try it if the image is disturbed by some ControlNet such as Canny.
  - Added Variants similar to sd-dynamic-propmpts in the prompt.
    - If you specify `{spring|summer|autumn|winter}`, one of them will be randomly selected.
    - If you specify `{2$$chocolate|vanilla|strawberry}`, two of them will be randomly selected.
    - If you specify `{1-2$$ and $$chocolate|vanilla|strawberry}`, one or two of them will be randomly selected and connected by ` and `.
    - You can specify the number of candidates in the range `0-2`. You cannot omit one side like `-2` or `1-`.
    - It can also be specified for the prompt option.
    - If you specify `e` or `E`, all candidates will be selected and the prompt will be repeated multiple times (`--images_per_prompt` is ignored). It may be useful for creating X/Y plots.
    - You can also specify `--am {e$$0.2|0.4|0.6|0.8|1.0},{e$$0.4|0.7|1.0} --d 1234`. In this case, 15 prompts will be generated with 5*3.
    - There is no weighting function.

- 各学習スクリプトでProdigyオプティマイザがサポートされました。D-Adaptationの仲間でDyLoRAの学習に有効とのことです。 [PR #585](https://github.com/kohya-ss/sd-scripts/pull/585)  詳細はPRをご覧ください。sdbds氏に感謝します。
  - `pip install prodigyopt` としてパッケージをインストールしてください。また `--optimizer_type="prodigy"` のようにオプションを指定します。
- 各学習スクリプトで任意のDatasetをサポートしました（XTIを除く）。画像とキャプションを返すDatasetクラスを定義することで、学習スクリプトから利用できます。
  - Pythonスクリプトを用意し、`train_util.MinimalDataset`を継承するクラスを定義してください。そして各学習スクリプトのオプションで `--dataset_class package.module.DatasetClass` のように指定してください。
  - 実装方法は `MinimalDataset` を参考にしてください。のちほどサンプルを用意します。
- 生成スクリプトに以下の機能追加を行いました。
  - Highres. Fixの2nd stageでControlNetを無効化するオプション `--highres_fix_disable_control_net` を追加しました。Canny等一部のControlNetで画像が乱れる場合にお試しください。
  - プロンプトでsd-dynamic-propmptsに似たVariantをサポートしました。
    - `{spring|summer|autumn|winter}` のように指定すると、いずれかがランダムに選択されます。
    - `{2$$chocolate|vanilla|strawberry}` のように指定すると、いずれか2個がランダムに選択されます。
    - `{1-2$$ and $$chocolate|vanilla|strawberry}` のように指定すると、1個か2個がランダムに選択され ` and ` で接続されます。
    - 個数のレンジ指定では`0-2`のように0個も指定可能です。`-2`や`1-`のような片側の省略はできません。
    - プロンプトオプションに対しても指定可能です。
    - `{e$$chocolate|vanilla|strawberry}` のように`e`または`E`を指定すると、すべての候補が選択されプロンプトが複数回繰り返されます（`--images_per_prompt`は無視されます）。X/Y plotの作成に便利かもしれません。
    - `--am {e$$0.2|0.4|0.6|0.8|1.0},{e$$0.4|0.7|1.0} --d 1234`のような指定も可能です。この場合、5*3で15回のプロンプトが生成されます。
    - Weightingの機能はありません。

### 8 Jun. 2023, 2023/06/08

- Fixed a bug where clip skip did not work when training with weighted captions (`--weighted_captions` specified) and when generating sample images during training.
- 重みづけキャプションでの学習時（`--weighted_captions`指定時）および学習中のサンプル画像生成時にclip skipが機能しない不具合を修正しました。

### 6 Jun. 2023, 2023/06/06

- Fix `train_network.py` to probably work with older versions of LyCORIS.
- `gen_img_diffusers.py` now supports `BREAK` syntax.
- `train_network.py`がLyCORISの以前のバージョンでも恐らく動作するよう修正しました。
- `gen_img_diffusers.py` で `BREAK` 構文をサポートしました。

### 3 Jun. 2023, 2023/06/03

- Max Norm Regularization is now available in `train_network.py`. [PR #545](https://github.com/kohya-ss/sd-scripts/pull/545) Thanks to AI-Casanova!
  - Max Norm Regularization is a technique to stabilize network training by limiting the norm of network weights. It may be effective in suppressing overfitting of LoRA and improving stability when used with other LoRAs. See PR for details.
  - Specify as `--scale_weight_norms=1.0`. It seems good to try from `1.0`.
  - The networks other than LoRA in this repository (such as LyCORIS) do not support this option.

- Three types of dropout have been added to `train_network.py` and LoRA network.
  - Dropout is a technique to suppress overfitting and improve network performance by randomly setting some of the network outputs to 0.
  - `--network_dropout` is a normal dropout at the neuron level. In the case of LoRA, it is applied to the output of down. Proposed in [PR #545](https://github.com/kohya-ss/sd-scripts/pull/545) Thanks to AI-Casanova!
    - `--network_dropout=0.1` specifies the dropout probability to `0.1`.
    - Note that the specification method is different from LyCORIS.
  - For LoRA network, `--network_args` can specify `rank_dropout` to dropout each rank with specified probability. Also `module_dropout` can be specified to dropout each module with specified probability.
    - Specify as `--network_args "rank_dropout=0.2" "module_dropout=0.1"`.
  - `--network_dropout`, `rank_dropout`, and `module_dropout` can be specified at the same time.
  - Values of 0.1 to 0.3 may be good to try. Values greater than 0.5 should not be specified.
  - `rank_dropout` and `module_dropout` are original techniques of this repository. Their effectiveness has not been verified yet.
  - The networks other than LoRA in this repository (such as LyCORIS) do not support these options.

- Added an option `--scale_v_pred_loss_like_noise_pred` to scale v-prediction loss like noise prediction in each training script.
  - By scaling the loss according to the time step, the weights of global noise prediction and local noise prediction become the same, and the improvement of details may be expected.
  - See [this article](https://xrg.hatenablog.com/entry/2023/06/02/202418) by xrg for details (written in Japanese). Thanks to xrg for the great suggestion!

- Max Norm Regularizationが`train_network.py`で使えるようになりました。[PR #545](https://github.com/kohya-ss/sd-scripts/pull/545) AI-Casanova氏に感謝します。
  - Max Norm Regularizationは、ネットワークの重みのノルムを制限することで、ネットワークの学習を安定させる手法です。LoRAの過学習の抑制、他のLoRAと併用した時の安定性の向上が期待できるかもしれません。詳細はPRを参照してください。
  - `--scale_weight_norms=1.0`のように `--scale_weight_norms` で指定してください。`1.0`から試すと良いようです。
  - LyCORIS等、当リポジトリ以外のネットワークは現時点では未対応です。

- `train_network.py` およびLoRAに計三種類のdropoutを追加しました。
  - dropoutはネットワークの一部の出力をランダムに0にすることで、過学習の抑制、ネットワークの性能向上等を図る手法です。
  - `--network_dropout` はニューロン単位の通常のdropoutです。LoRAの場合、downの出力に対して適用されます。[PR #545](https://github.com/kohya-ss/sd-scripts/pull/545) で提案されました。AI-Casanova氏に感謝します。
    - `--network_dropout=0.1` などとすることで、dropoutの確率を指定できます。
    - LyCORISとは指定方法が異なりますのでご注意ください。
  - LoRAの場合、`--network_args`に`rank_dropout`を指定することで各rankを指定確率でdropoutします。また同じくLoRAの場合、`--network_args`に`module_dropout`を指定することで各モジュールを指定確率でdropoutします。
    - `--network_args "rank_dropout=0.2" "module_dropout=0.1"` のように指定します。
  - `--network_dropout`、`rank_dropout` 、 `module_dropout` は同時に指定できます。
  - それぞれの値は0.1~0.3程度から試してみると良いかもしれません。0.5を超える値は指定しない方が良いでしょう。
  - `rank_dropout`および`module_dropout`は当リポジトリ独自の手法です。有効性の検証はまだ行っていません。
  - これらのdropoutはLyCORIS等、当リポジトリ以外のネットワークは現時点では未対応です。

- 各学習スクリプトにv-prediction lossをnoise predictionと同様の値にスケールするオプション`--scale_v_pred_loss_like_noise_pred`を追加しました。
  - タイムステップに応じてlossをスケールすることで、 大域的なノイズの予測と局所的なノイズの予測の重みが同じになり、ディテールの改善が期待できるかもしれません。
  - 詳細はxrg氏のこちらの記事をご参照ください：[noise_predictionモデルとv_predictionモデルの損失 - 勾配降下党青年局](https://xrg.hatenablog.com/entry/2023/06/02/202418) xrg氏の素晴らしい記事に感謝します。

### 31 May 2023, 2023/05/31

- Show warning when image caption file does not exist during training. [PR #533](https://github.com/kohya-ss/sd-scripts/pull/533) Thanks to TingTingin!
  - Warning is also displayed when using class+identifier dataset. Please ignore if it is intended.
- `train_network.py` now supports merging network weights before training. [PR #542](https://github.com/kohya-ss/sd-scripts/pull/542) Thanks to u-haru!
  - `--base_weights` option specifies LoRA or other model files (multiple files are allowed) to merge.
  - `--base_weights_multiplier` option specifies multiplier of the weights to merge (multiple values are allowed). If omitted or less than `base_weights`, 1.0 is used.
  - This is useful for incremental learning. See PR for details.
- Show warning and continue training when uploading to HuggingFace fails.

- 学習時に画像のキャプションファイルが存在しない場合、警告が表示されるようになりました。 [PR #533](https://github.com/kohya-ss/sd-scripts/pull/533) TingTingin氏に感謝します。
  - class+identifier方式のデータセットを利用している場合も警告が表示されます。意図している通りの場合は無視してください。
- `train_network.py` に学習前にモデルにnetworkの重みをマージする機能が追加されました。 [PR #542](https://github.com/kohya-ss/sd-scripts/pull/542) u-haru氏に感謝します。
  - `--base_weights` オプションでLoRA等のモデルファイル（複数可）を指定すると、それらの重みをマージします。
  - `--base_weights_multiplier` オプションでマージする重みの倍率（複数可）を指定できます。省略時または`base_weights`よりも数が少ない場合は1.0になります。
  - 差分追加学習などにご利用ください。詳細はPRをご覧ください。
- HuggingFaceへのアップロードに失敗した場合、警告を表示しそのまま学習を続行するよう変更しました。

### 25 May 2023, 2023/05/25

- [D-Adaptation v3.0](https://github.com/facebookresearch/dadaptation) is now supported. [PR #530](https://github.com/kohya-ss/sd-scripts/pull/530) Thanks to sdbds!
  - `--optimizer_type` now accepts `DAdaptAdamPreprint`, `DAdaptAdanIP`, and `DAdaptLion`.
  - `DAdaptAdam` is now new. The old `DAdaptAdam` is available with `DAdaptAdamPreprint`.
  - Simply specifying `DAdaptation` will use `DAdaptAdamPreprint` (same behavior as before).
  - You need to install D-Adaptation v3.0. After activating venv, please do `pip install -U dadaptation`.
  - See PR and D-Adaptation documentation for details.
- [D-Adaptation v3.0](https://github.com/facebookresearch/dadaptation)がサポートされました。 [PR #530](https://github.com/kohya-ss/sd-scripts/pull/530)  sdbds氏に感謝します。
  - `--optimizer_type`に`DAdaptAdamPreprint`、`DAdaptAdanIP`、`DAdaptLion` が追加されました。
  - `DAdaptAdam`が新しくなりました。今までの`DAdaptAdam`は`DAdaptAdamPreprint`で使用できます。
  - 単に `DAdaptation` を指定すると`DAdaptAdamPreprint`が使用されます（今までと同じ動き）。
  - D-Adaptation v3.0のインストールが必要です。venvを有効にした後 `pip install -U dadaptation` としてください。
  - 詳細はPRおよびD-Adaptationのドキュメントを参照してください。

### 22 May 2023, 2023/05/22

- Fixed several bugs.
  - The state is saved even when the `--save_state` option is not specified in `fine_tune.py` and `train_db.py`. [PR #521](https://github.com/kohya-ss/sd-scripts/pull/521) Thanks to akshaal!
  - Cannot load LoRA without `alpha`. [PR #527](https://github.com/kohya-ss/sd-scripts/pull/527) Thanks to Manjiz!
  - Minor changes to console output during sample generation. [PR #515](https://github.com/kohya-ss/sd-scripts/pull/515) Thanks to yanhuifair!
- The generation script now uses xformers for VAE as well.
- いくつかのバグ修正を行いました。
  -  `fine_tune.py`と`train_db.py`で`--save_state`オプション未指定時にもstateが保存される。 [PR #521](https://github.com/kohya-ss/sd-scripts/pull/521) akshaal氏に感謝します。
  - `alpha`を持たないLoRAを読み込めない。[PR #527](https://github.com/kohya-ss/sd-scripts/pull/527) Manjiz氏に感謝します。
  - サンプル生成時のコンソール出力の軽微な変更。[PR #515](https://github.com/kohya-ss/sd-scripts/pull/515) yanhuifair氏に感謝します。
- 生成スクリプトでVAEについてもxformersを使うようにしました。

### 16 May 2023, 2023/05/16

- Fixed an issue where an error would occur if the encoding of the prompt file was different from the default. [PR #510](https://github.com/kohya-ss/sd-scripts/pull/510) Thanks to sdbds!
  - Please save the prompt file in UTF-8.
- プロンプトファイルのエンコーディングがデフォルトと異なる場合にエラーが発生する問題を修正しました。 [PR #510](https://github.com/kohya-ss/sd-scripts/pull/510) sdbds氏に感謝します。
  - プロンプトファイルはUTF-8で保存してください。

### 15 May 2023, 2023/05/15

- Added [English translation of documents](https://github.com/darkstorm2150/sd-scripts#links-to-usage-documentation) by darkstorm2150. Thank you very much!
- The prompt for sample generation during training can now be specified in `.toml` or `.json`. [PR #504](https://github.com/kohya-ss/sd-scripts/pull/504) Thanks to Linaqruf!
  - For details on prompt description, please see the PR.

- darkstorm2150氏に[ドキュメント類を英訳](https://github.com/darkstorm2150/sd-scripts#links-to-usage-documentation)していただきました。ありがとうございます！
- 学習中のサンプル生成のプロンプトを`.toml`または`.json`で指定可能になりました。 [PR #504](https://github.com/kohya-ss/sd-scripts/pull/504) Linaqruf氏に感謝します。
  - プロンプト記述の詳細は当該PRをご覧ください。

### 11 May 2023, 2023/05/11

- Added an option `--dim_from_weights` to `train_network.py` to automatically determine the dim(rank) from the weight file. [PR #491](https://github.com/kohya-ss/sd-scripts/pull/491) Thanks to AI-Casanova!
  - It is useful in combination with `resize_lora.py`. Please see the PR for details.
- Fixed a bug where the noise resolution was incorrect with Multires noise. [PR #489](https://github.com/kohya-ss/sd-scripts/pull/489) Thanks to sdbds!
  - Please see the PR for details.
- The image generation scripts can now use img2img and highres fix at the same time.
- Fixed a bug where the hint image of ControlNet was incorrectly BGR instead of RGB in the image generation scripts.
- Added a feature to the image generation scripts to use the memory-efficient VAE.
  - If you specify a number with the `--vae_slices` option, the memory-efficient VAE will be used. The maximum output size will be larger, but it will be slower. Please specify a value of about `16` or `32`.
  - The implementation of the VAE is in `library/slicing_vae.py`.

- `train_network.py`にdim(rank)を重みファイルから自動決定するオプション`--dim_from_weights`が追加されました。 [PR #491](https://github.com/kohya-ss/sd-scripts/pull/491) AI-Casanova氏に感謝します。
  - `resize_lora.py`と組み合わせると有用です。詳細はPRもご参照ください。
- Multires noiseでノイズ解像度が正しくない不具合が修正されました。 [PR #489](https://github.com/kohya-ss/sd-scripts/pull/489)  sdbds氏に感謝します。
  - 詳細は当該PRをご参照ください。
- 生成スクリプトでimg2imgとhighres fixを同時に使用できるようにしました。
- 生成スクリプトでControlNetのhint画像が誤ってBGRだったのをRGBに修正しました。
- 生成スクリプトで省メモリ化VAEを使えるよう機能追加しました。
  - `--vae_slices`オプションに数値を指定すると、省メモリ化VAEを用います。出力可能な最大サイズが大きくなりますが、遅くなります。`16`または`32`程度の値を指定してください。
  - VAEの実装は`library/slicing_vae.py`にあります。

### 7 May 2023, 2023/05/07

- The documentation has been moved to the `docs` folder. If you have links, please change them.
- Removed `gradio` from `requirements.txt`.
- DAdaptAdaGrad, DAdaptAdan, and DAdaptSGD are now supported by DAdaptation. [PR#455](https://github.com/kohya-ss/sd-scripts/pull/455) Thanks to sdbds!
  - DAdaptation needs to be installed. Also, depending on the optimizer, DAdaptation may need to be updated. Please update with `pip install --upgrade dadaptation`.
- Added support for pre-calculation of LoRA weights in image generation scripts. Specify `--network_pre_calc`.
  - The prompt option `--am` is available. Also, it is disabled when Regional LoRA is used.
- Added Adaptive noise scale to each training script. Specify a number with `--adaptive_noise_scale` to enable it.
  - __Experimental option. It may be removed or changed in the future.__
  - This is an original implementation that automatically adjusts the value of the noise offset according to the absolute value of the mean of each channel of the latents. It is expected that appropriate noise offsets will be set for bright and dark images, respectively.
  - Specify it together with `--noise_offset`.
  - The actual value of the noise offset is calculated as `noise_offset + abs(mean(latents, dim=(2,3))) * adaptive_noise_scale`. Since the latent is close to a normal distribution, it may be a good idea to specify a value of about 1/10 to the same as the noise offset.
  - Negative values can also be specified, in which case the noise offset will be clipped to 0 or more.
- Other minor fixes.

- ドキュメントを`docs`フォルダに移動しました。リンク等を張られている場合は変更をお願いいたします。
- `requirements.txt`から`gradio`を削除しました。
- DAdaptationで新しくDAdaptAdaGrad、DAdaptAdan、DAdaptSGDがサポートされました。[PR#455](https://github.com/kohya-ss/sd-scripts/pull/455) sdbds氏に感謝します。
  - dadaptationのインストールが必要です。またオプティマイザによってはdadaptationの更新が必要です。`pip install --upgrade dadaptation`で更新してください。
- 画像生成スクリプトでLoRAの重みの事前計算をサポートしました。`--network_pre_calc`を指定してください。
  - プロンプトオプションの`--am`が利用できます。またRegional LoRA使用時には無効になります。
- 各学習スクリプトにAdaptive noise scaleを追加しました。`--adaptive_noise_scale`で数値を指定すると有効になります。
  - __実験的オプションです。将来的に削除、仕様変更される可能性があります。__
  - Noise offsetの値を、latentsの各チャネルの平均値の絶対値に応じて自動調整するオプションです。独自の実装で、明るい画像、暗い画像に対してそれぞれ適切なnoise offsetが設定されることが期待されます。
  - `--noise_offset` と同時に指定してください。
  - 実際のNoise offsetの値は `noise_offset + abs(mean(latents, dim=(2,3))) * adaptive_noise_scale` で計算されます。 latentは正規分布に近いためnoise_offsetの1/10～同程度の値を指定するとよいかもしれません。
  - 負の値も指定でき、その場合はnoise offsetは0以上にclipされます。
- その他の細かい修正を行いました。

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

