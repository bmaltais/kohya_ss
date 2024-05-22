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

The file does not contain requirements for PyTorch. Because the version of PyTorch depends on the environment, it is not included in the file. Please install PyTorch first according to the environment. See installation instructions below.

The scripts are tested with Pytorch 2.1.2. 2.0.1 and 1.12.1 is not tested but should work.

## Links to usage documentation

Most of the documents are written in Japanese.

[English translation by darkstorm2150 is here](https://github.com/darkstorm2150/sd-scripts#links-to-usage-documentation). Thanks to darkstorm2150!

* [Training guide - common](./docs/train_README-ja.md) : data preparation, options etc... 
  * [Chinese version](./docs/train_README-zh.md)
* [SDXL training](./docs/train_SDXL-en.md) (English version)
* [Dataset config](./docs/config_README-ja.md) 
  * [English version](./docs/config_README-en.md)
* [DreamBooth training guide](./docs/train_db_README-ja.md)
* [Step by Step fine-tuning guide](./docs/fine_tune_README_ja.md):
* [Training LoRA](./docs/train_network_README-ja.md)
* [Training Textual Inversion](./docs/train_ti_README-ja.md)
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

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

accelerate config
```

If `python -m venv` shows only `python`, change `python` to `py`.

__Note:__ Now `bitsandbytes==0.43.0`, `prodigyopt==1.0` and `lion-pytorch==0.0.6` are included in the requirements.txt. If you'd like to use the another version, please install it manually.

This installation is for CUDA 11.8. If you use a different version of CUDA, please install the appropriate version of PyTorch and xformers. For example, if you use CUDA 12, please install `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121` and `pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121`.

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

If you'd like to use bf16, please answer `bf16` to the last question.

Note: Some user reports ``ValueError: fp16 mixed precision requires a GPU`` is occurred in training. In this case, answer `0` for the 6th question: 
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`` 

(Single GPU with id `0` will be used.)

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

### Upgrade PyTorch

If you want to upgrade PyTorch, you can upgrade it with `pip install` command in [Windows Installation](#windows-installation) section. `xformers` is also required to be upgraded when PyTorch is upgraded.

## Credits

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for great work!

The LoRA expansion to Conv2d 3x3 was initially released by cloneofsimo and its effectiveness was demonstrated at [LoCon](https://github.com/KohakuBlueleaf/LoCon) by KohakuBlueleaf. Thank you so much KohakuBlueleaf!

## License

The majority of scripts is licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's and LoCon), however portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause


## Change History

### Apr 7, 2024 / 2024-04-07: v0.8.7

- The default value of `huber_schedule` in Scheduled Huber Loss is changed from `exponential` to `snr`, which is expected to give better results.

- Scheduled Huber Loss の `huber_schedule` のデフォルト値を `exponential` から、より良い結果が期待できる `snr` に変更しました。

### Apr 7, 2024 / 2024-04-07: v0.8.6

#### Highlights

- The dependent libraries are updated. Please see [Upgrade](#upgrade) and update the libraries.
  - Especially `imagesize` is newly added, so if you cannot update the libraries immediately, please install with `pip install imagesize==1.4.1` separately.
  - `bitsandbytes==0.43.0`, `prodigyopt==1.0`, `lion-pytorch==0.0.6` are included in the requirements.txt.
    - `bitsandbytes` no longer requires complex procedures as it now officially supports Windows.  
  - Also, the PyTorch version is updated to 2.1.2 (PyTorch does not need to be updated immediately). In the upgrade procedure, PyTorch is not updated, so please manually install or update torch, torchvision, xformers if necessary (see [Upgrade PyTorch](#upgrade-pytorch)).
- When logging to wandb is enabled, the entire command line is exposed. Therefore, it is recommended to write wandb API key and HuggingFace token in the configuration file (`.toml`). Thanks to bghira for raising the issue.
  - A warning is displayed at the start of training if such information is included in the command line.
  - Also, if there is an absolute path, the path may be exposed, so it is recommended to specify a relative path or write it in the configuration file. In such cases, an INFO log is displayed.
  - See [#1123](https://github.com/kohya-ss/sd-scripts/pull/1123) and PR [#1240](https://github.com/kohya-ss/sd-scripts/pull/1240) for details.
- Colab seems to stop with log output. Try specifying `--console_log_simple` option in the training script to disable rich logging.
- Other improvements include the addition of masked loss, scheduled Huber Loss, DeepSpeed support, dataset settings improvements, and image tagging improvements. See below for details.

#### Training scripts

- `train_network.py` and `sdxl_train_network.py` are modified to record some dataset settings in the metadata of the trained model (`caption_prefix`, `caption_suffix`, `keep_tokens_separator`, `secondary_separator`, `enable_wildcard`).
- Fixed a bug that U-Net and Text Encoders are included in the state in `train_network.py` and `sdxl_train_network.py`. The saving and loading of the state are faster, the file size is smaller, and the memory usage when loading is reduced.
- DeepSpeed is supported. PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101)  and [#1139](https://github.com/kohya-ss/sd-scripts/pull/1139) Thanks to BootsofLagrangian! See PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101) for details.
- The masked loss is supported in each training script. PR [#1207](https://github.com/kohya-ss/sd-scripts/pull/1207) See [Masked loss](#about-masked-loss) for details.
- Scheduled Huber Loss has been introduced to each training scripts. PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/) Thanks to kabachuha for the PR and cheald, drhead, and others for the discussion! See the PR and [Scheduled Huber Loss](#about-scheduled-huber-loss) for details.
- The options `--noise_offset_random_strength` and `--ip_noise_gamma_random_strength` are added to each training script. These options can be used to vary the noise offset and ip noise gamma in the range of 0 to the specified value. PR [#1177](https://github.com/kohya-ss/sd-scripts/pull/1177) Thanks to KohakuBlueleaf!
- The options `--save_state_on_train_end` are added to each training script. PR [#1168](https://github.com/kohya-ss/sd-scripts/pull/1168) Thanks to gesen2egee!
- The options `--sample_every_n_epochs` and `--sample_every_n_steps` in each training script now display a warning and ignore them when a number less than or equal to `0` is specified. Thanks to S-Del for raising the issue.

#### Dataset settings

- The [English version of the dataset settings documentation](./docs/config_README-en.md) is added. PR [#1175](https://github.com/kohya-ss/sd-scripts/pull/1175) Thanks to darkstorm2150!
- The `.toml` file for the dataset config is now read in UTF-8 encoding. PR [#1167](https://github.com/kohya-ss/sd-scripts/pull/1167) Thanks to Horizon1704!
- Fixed a bug that the last subset settings are applied to all images when multiple subsets of regularization images are specified in the dataset settings. The settings for each subset are correctly applied to each image. PR [#1205](https://github.com/kohya-ss/sd-scripts/pull/1205) Thanks to feffy380!
- Some features are added to the dataset subset settings.
  - `secondary_separator` is added to specify the tag separator that is not the target of shuffling or dropping. 
    - Specify `secondary_separator=";;;"`. When you specify `secondary_separator`, the part is not shuffled or dropped. 
  - `enable_wildcard` is added. When set to `true`, the wildcard notation `{aaa|bbb|ccc}` can be used. The multi-line caption is also enabled.
  - `keep_tokens_separator` is updated to be used twice in the caption. When you specify `keep_tokens_separator="|||"`, the part divided by the second `|||` is not shuffled or dropped and remains at the end.
  - The existing features `caption_prefix` and `caption_suffix` can be used together. `caption_prefix` and `caption_suffix` are processed first, and then `enable_wildcard`, `keep_tokens_separator`, shuffling and dropping, and `secondary_separator` are processed in order.
  - See [Dataset config](./docs/config_README-en.md) for details.
- The dataset with DreamBooth method supports caching image information (size, caption). PR [#1178](https://github.com/kohya-ss/sd-scripts/pull/1178) and [#1206](https://github.com/kohya-ss/sd-scripts/pull/1206) Thanks to KohakuBlueleaf! See [DreamBooth method specific options](./docs/config_README-en.md#dreambooth-specific-options) for details.

#### Image tagging

- The support for v3 repositories is added to `tag_image_by_wd14_tagger.py` (`--onnx` option only). PR [#1192](https://github.com/kohya-ss/sd-scripts/pull/1192) Thanks to sdbds!
  - Onnx may need to be updated. Onnx is not installed by default, so please install or update it with `pip install onnx==1.15.0 onnxruntime-gpu==1.17.1` etc. Please also check the comments in `requirements.txt`.
- The model is now saved in the subdirectory as `--repo_id` in `tag_image_by_wd14_tagger.py` . This caches multiple repo_id models. Please delete unnecessary files under `--model_dir`.
- Some options are added to `tag_image_by_wd14_tagger.py`.
  - Some are added in PR [#1216](https://github.com/kohya-ss/sd-scripts/pull/1216) Thanks to Disty0!
  - Output rating tags `--use_rating_tags` and `--use_rating_tags_as_last_tag`
  - Output character tags first `--character_tags_first`
  - Expand character tags and series `--character_tag_expand`
  - Specify tags to output first `--always_first_tags`
  - Replace tags `--tag_replacement`
  - See [Tagging documentation](./docs/wd14_tagger_README-en.md) for details.
- Fixed an error when specifying `--beam_search` and a value of 2 or more for `--num_beams` in `make_captions.py`.

#### About Masked loss

The masked loss is supported in each training script. To enable the masked loss, specify the `--masked_loss` option.

The feature is not fully tested, so there may be bugs. If you find any issues, please open an Issue.

ControlNet dataset is used to specify the mask. The mask images should be the RGB images. The pixel value 255 in R channel is treated as the mask (the loss is calculated only for the pixels with the mask), and 0 is treated as the non-mask. The pixel values 0-255 are converted to 0-1 (i.e., the pixel value 128 is treated as the half weight of the loss). See details for the dataset specification in the [LLLite documentation](./docs/train_lllite_README.md#preparing-the-dataset).

#### About Scheduled Huber Loss

Scheduled Huber Loss has been introduced to each training scripts. This is a method to improve robustness against outliers or anomalies (data corruption) in the training data.

With the traditional MSE (L2) loss function, the impact of outliers could be significant, potentially leading to a degradation in the quality of generated images. On the other hand, while the Huber loss function can suppress the influence of outliers, it tends to compromise the reproduction of fine details in images.

To address this, the proposed method employs a clever application of the Huber loss function. By scheduling the use of Huber loss in the early stages of training (when noise is high) and MSE in the later stages, it strikes a balance between outlier robustness and fine detail reproduction.

Experimental results have confirmed that this method achieves higher accuracy on data containing outliers compared to pure Huber loss or MSE. The increase in computational cost is minimal.

The newly added arguments loss_type, huber_schedule, and huber_c allow for the selection of the loss function type (Huber, smooth L1, MSE), scheduling method (exponential, constant, SNR), and Huber's parameter. This enables optimization based on the characteristics of the dataset.

See PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/) for details.

- `loss_type`: Specify the loss function type. Choose `huber` for Huber loss, `smooth_l1` for smooth L1 loss, and `l2` for MSE loss. The default is `l2`, which is the same as before.
- `huber_schedule`: Specify the scheduling method. Choose `exponential`, `constant`, or `snr`. The default is `snr`.
- `huber_c`: Specify the Huber's parameter. The default is `0.1`.

Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.

#### 主要な変更点

- 依存ライブラリが更新されました。[アップグレード](./README-ja.md#アップグレード) を参照しライブラリを更新してください。
  - 特に `imagesize` が新しく追加されていますので、すぐにライブラリの更新ができない場合は `pip install imagesize==1.4.1` で個別にインストールしてください。
  - `bitsandbytes==0.43.0`、`prodigyopt==1.0`、`lion-pytorch==0.0.6` が requirements.txt に含まれるようになりました。
    - `bitsandbytes` が公式に Windows をサポートしたため複雑な手順が不要になりました。
  - また PyTorch のバージョンを 2.1.2 に更新しました。PyTorch はすぐに更新する必要はありません。更新時は、アップグレードの手順では PyTorch が更新されませんので、torch、torchvision、xformers を手動でインストールしてください。
- wandb へのログ出力が有効の場合、コマンドライン全体が公開されます。そのため、コマンドラインに wandb の API キーや HuggingFace のトークンなどが含まれる場合、設定ファイル（`.toml`）への記載をお勧めします。問題提起していただいた bghira 氏に感謝します。
  - このような場合には学習開始時に警告が表示されます。
  - また絶対パスの指定がある場合、そのパスが公開される可能性がありますので、相対パスを指定するか設定ファイルに記載することをお勧めします。このような場合は INFO ログが表示されます。
  - 詳細は [#1123](https://github.com/kohya-ss/sd-scripts/pull/1123) および PR [#1240](https://github.com/kohya-ss/sd-scripts/pull/1240) をご覧ください。
- Colab での動作時、ログ出力で停止してしまうようです。学習スクリプトに `--console_log_simple` オプションを指定し、rich のロギングを無効してお試しください。
- その他、マスクロス追加、Scheduled Huber Loss 追加、DeepSpeed 対応、データセット設定の改善、画像タグ付けの改善などがあります。詳細は以下をご覧ください。

#### 学習スクリプト

- `train_network.py` および `sdxl_train_network.py` で、学習したモデルのメタデータに一部のデータセット設定が記録されるよう修正しました（`caption_prefix`、`caption_suffix`、`keep_tokens_separator`、`secondary_separator`、`enable_wildcard`）。
- `train_network.py` および `sdxl_train_network.py` で、state に U-Net および Text Encoder が含まれる不具合を修正しました。state の保存、読み込みが高速化され、ファイルサイズも小さくなり、また読み込み時のメモリ使用量も削減されます。
- DeepSpeed がサポートされました。PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101) 、[#1139](https://github.com/kohya-ss/sd-scripts/pull/1139) BootsofLagrangian 氏に感謝します。詳細は PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101) をご覧ください。
- 各学習スクリプトでマスクロスをサポートしました。PR [#1207](https://github.com/kohya-ss/sd-scripts/pull/1207) 詳細は [マスクロスについて](#マスクロスについて) をご覧ください。
- 各学習スクリプトに Scheduled Huber Loss を追加しました。PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/) ご提案いただいた kabachuha 氏、および議論を深めてくださった cheald 氏、drhead 氏を始めとする諸氏に感謝します。詳細は当該 PR および [Scheduled Huber Loss について](#scheduled-huber-loss-について) をご覧ください。
- 各学習スクリプトに、noise offset、ip noise gammaを、それぞれ 0~指定した値の範囲で変動させるオプション `--noise_offset_random_strength` および `--ip_noise_gamma_random_strength` が追加されました。 PR [#1177](https://github.com/kohya-ss/sd-scripts/pull/1177) KohakuBlueleaf 氏に感謝します。
- 各学習スクリプトに、学習終了時に state を保存する `--save_state_on_train_end` オプションが追加されました。 PR [#1168](https://github.com/kohya-ss/sd-scripts/pull/1168) gesen2egee 氏に感謝します。
- 各学習スクリプトで `--sample_every_n_epochs` および `--sample_every_n_steps` オプションに `0` 以下の数値を指定した時、警告を表示するとともにそれらを無視するよう変更しました。問題提起していただいた S-Del 氏に感謝します。

#### データセット設定

- データセット設定の `.toml` ファイルが UTF-8 encoding で読み込まれるようになりました。PR [#1167](https://github.com/kohya-ss/sd-scripts/pull/1167) Horizon1704 氏に感謝します。
- データセット設定で、正則化画像のサブセットを複数指定した時、最後のサブセットの各種設定がすべてのサブセットの画像に適用される不具合が修正されました。それぞれのサブセットの設定が、それぞれの画像に正しく適用されます。PR [#1205](https://github.com/kohya-ss/sd-scripts/pull/1205) feffy380 氏に感謝します。
- データセットのサブセット設定にいくつかの機能を追加しました。
  - シャッフルの対象とならないタグ分割識別子の指定 `secondary_separator` を追加しました。`secondary_separator=";;;"` のように指定します。`secondary_separator` で区切ることで、その部分はシャッフル、drop 時にまとめて扱われます。
  - `enable_wildcard` を追加しました。`true` にするとワイルドカード記法 `{aaa|bbb|ccc}` が使えます。また複数行キャプションも有効になります。
  - `keep_tokens_separator` をキャプション内に 2 つ使えるようにしました。たとえば `keep_tokens_separator="|||"` と指定したとき、`1girl, hatsune miku, vocaloid ||| stage, mic ||| best quality, rating: general` とキャプションを指定すると、二番目の `|||` で分割された部分はシャッフル、drop されず末尾に残ります。
  - 既存の機能 `caption_prefix` と `caption_suffix` とあわせて使えます。`caption_prefix` と `caption_suffix` は一番最初に処理され、その後、ワイルドカード、`keep_tokens_separator`、シャッフルおよび drop、`secondary_separator` の順に処理されます。
  - 詳細は [データセット設定](./docs/config_README-ja.md) をご覧ください。
- DreamBooth 方式の DataSet で画像情報（サイズ、キャプション）をキャッシュする機能が追加されました。PR [#1178](https://github.com/kohya-ss/sd-scripts/pull/1178)、[#1206](https://github.com/kohya-ss/sd-scripts/pull/1206) KohakuBlueleaf 氏に感謝します。詳細は [データセット設定](./docs/config_README-ja.md#dreambooth-方式専用のオプション) をご覧ください。
- データセット設定の[英語版ドキュメント](./docs/config_README-en.md) が追加されました。PR [#1175](https://github.com/kohya-ss/sd-scripts/pull/1175) darkstorm2150 氏に感謝します。

#### 画像のタグ付け

- `tag_image_by_wd14_tagger.py` で v3 のリポジトリがサポートされました（`--onnx` 指定時のみ有効）。 PR [#1192](https://github.com/kohya-ss/sd-scripts/pull/1192) sdbds 氏に感謝します。
  - Onnx のバージョンアップが必要になるかもしれません。デフォルトでは Onnx はインストールされていませんので、`pip install onnx==1.15.0 onnxruntime-gpu==1.17.1` 等でインストール、アップデートしてください。`requirements.txt` のコメントもあわせてご確認ください。
- `tag_image_by_wd14_tagger.py` で、モデルを`--repo_id` のサブディレクトリに保存するようにしました。これにより複数のモデルファイルがキャッシュされます。`--model_dir` 直下の不要なファイルは削除願います。
- `tag_image_by_wd14_tagger.py` にいくつかのオプションを追加しました。
  - 一部は PR [#1216](https://github.com/kohya-ss/sd-scripts/pull/1216) で追加されました。Disty0 氏に感謝します。
  - レーティングタグを出力する `--use_rating_tags` および `--use_rating_tags_as_last_tag`
  - キャラクタタグを最初に出力する `--character_tags_first`
  - キャラクタタグとシリーズを展開する `--character_tag_expand`
  - 常に最初に出力するタグを指定する `--always_first_tags`
  - タグを置換する `--tag_replacement`
  - 詳細は [タグ付けに関するドキュメント](./docs/wd14_tagger_README-ja.md) をご覧ください。
- `make_captions.py` で `--beam_search` を指定し `--num_beams` に2以上の値を指定した時のエラーを修正しました。

#### マスクロスについて

各学習スクリプトでマスクロスをサポートしました。マスクロスを有効にするには `--masked_loss` オプションを指定してください。

機能は完全にテストされていないため、不具合があるかもしれません。その場合は Issue を立てていただけると助かります。

マスクの指定には ControlNet データセットを使用します。マスク画像は RGB 画像である必要があります。R チャンネルのピクセル値 255 がロス計算対象、0 がロス計算対象外になります。0-255 の値は、0-1 の範囲に変換されます（つまりピクセル値 128 の部分はロスの重みが半分になります）。データセットの詳細は [LLLite ドキュメント](./docs/train_lllite_README-ja.md#データセットの準備) をご覧ください。

#### Scheduled Huber Loss について

各学習スクリプトに、学習データ中の異常値や外れ値（data corruption）への耐性を高めるための手法、Scheduled Huber Lossが導入されました。

従来のMSE（L2）損失関数では、異常値の影響を大きく受けてしまい、生成画像の品質低下を招く恐れがありました。一方、Huber損失関数は異常値の影響を抑えられますが、画像の細部再現性が損なわれがちでした。

この手法ではHuber損失関数の適用を工夫し、学習の初期段階（ノイズが大きい場合）ではHuber損失を、後期段階ではMSEを用いるようスケジューリングすることで、異常値耐性と細部再現性のバランスを取ります。

実験の結果では、この手法が純粋なHuber損失やMSEと比べ、異常値を含むデータでより高い精度を達成することが確認されています。また計算コストの増加はわずかです。

具体的には、新たに追加された引数loss_type、huber_schedule、huber_cで、損失関数の種類（Huber, smooth L1, MSE）とスケジューリング方法（exponential, constant, SNR）を選択できます。これによりデータセットに応じた最適化が可能になります。

詳細は PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/) をご覧ください。

- `loss_type` : 損失関数の種類を指定します。`huber` で Huber損失、`smooth_l1` で smooth L1 損失、`l2` で MSE 損失を選択します。デフォルトは `l2` で、従来と同様です。
- `huber_schedule` : スケジューリング方法を指定します。`exponential` で指数関数的、`constant` で一定、`snr` で信号対雑音比に基づくスケジューリングを選択します。デフォルトは `snr` です。
- `huber_c` : Huber損失のパラメータを指定します。デフォルトは `0.1` です。

PR 内でいくつかの比較が共有されています。この機能を試す場合、最初は `--loss_type smooth_l1 --huber_schedule snr --huber_c 0.1` などで試してみるとよいかもしれません。

最近の更新情報は [Release](https://github.com/kohya-ss/sd-scripts/releases) をご覧ください。

## Additional Information

### Naming of LoRA

The LoRA supported by `train_network.py` has been named to avoid confusion. The documentation has been updated. The following are the names of LoRA types in this repository.

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers)

    LoRA for Linear layers and Conv2d layers with 1x1 kernel

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers)

    In addition to 1., LoRA for Conv2d layers with 3x3 kernel 
    
LoRA-LierLa is the default LoRA type for `train_network.py` (without `conv_dim` network arg). 
<!-- 
LoRA-LierLa can be used with [our extension](https://github.com/kohya-ss/sd-webui-additional-networks) for AUTOMATIC1111's Web UI, or with the built-in LoRA feature of the Web UI.

To use LoRA-C3Lier with Web UI, please use our extension. 
-->

### Sample image generation during training
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
