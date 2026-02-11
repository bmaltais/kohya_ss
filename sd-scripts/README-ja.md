## リポジトリについて
Stable Diffusionの学習、画像生成、その他のスクリプトを入れたリポジトリです。

[README in English](./README.md) ←更新情報はこちらにあります

開発中のバージョンはdevブランチにあります。最新の変更点はdevブランチをご確認ください。

FLUX.1およびSD3/SD3.5対応はsd3ブランチで行っています。それらの学習を行う場合はsd3ブランチをご利用ください。

GUIやPowerShellスクリプトなど、より使いやすくする機能が[bmaltais氏のリポジトリ](https://github.com/bmaltais/kohya_ss)で提供されています（英語です）のであわせてご覧ください。bmaltais氏に感謝します。

以下のスクリプトがあります。

* DreamBooth、U-NetおよびText Encoderの学習をサポート
* fine-tuning、同上
* LoRAの学習をサポート
* 画像生成
* モデル変換（Stable Diffision ckpt/safetensorsとDiffusersの相互変換）

## 使用法について

* [学習について、共通編](./docs/train_README-ja.md) : データ整備やオプションなど
    * [データセット設定](./docs/config_README-ja.md)
* [SDXL学習](./docs/train_SDXL-en.md) （英語版）
* [DreamBoothの学習について](./docs/train_db_README-ja.md)
* [fine-tuningのガイド](./docs/fine_tune_README_ja.md):
* [LoRAの学習について](./docs/train_network_README-ja.md)
* [Textual Inversionの学習について](./docs/train_ti_README-ja.md)
* [画像生成スクリプト](./docs/gen_img_README-ja.md)
* note.com [モデル変換スクリプト](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windowsでの動作に必要なプログラム

Python 3.10.6およびGitが必要です。

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

Python 3.10.x、3.11.x、3.12.xでも恐らく動作しますが、3.10.6でテストしています。

PowerShellを使う場合、venvを使えるようにするためには以下の手順でセキュリティ設定を変更してください。
（venvに限らずスクリプトの実行が可能になりますので注意してください。）

- PowerShellを管理者として開きます。
- 「Set-ExecutionPolicy Unrestricted」と入力し、Yと答えます。
- 管理者のPowerShellを閉じます。

## Windows環境でのインストール

スクリプトはPyTorch 2.1.2でテストしています。PyTorch 2.2以降でも恐らく動作します。

（なお、python -m venv～の行で「python」とだけ表示された場合、py -m venv～のようにpythonをpyに変更してください。）

PowerShellを使う場合、通常の（管理者ではない）PowerShellを開き以下を順に実行します。

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

コマンドプロンプトでも同一です。

注：`bitsandbytes==0.44.0`、`prodigyopt==1.0`、`lion-pytorch==0.0.6` は `requirements.txt` に含まれるようになりました。他のバージョンを使う場合は適宜インストールしてください。

この例では PyTorch および xfomers は2.1.2／CUDA 11.8版をインストールします。CUDA 12.1版やPyTorch 1.12.1を使う場合は適宜書き換えください。たとえば CUDA 12.1版の場合は `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121` および `pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121` としてください。

PyTorch 2.2以降を用いる場合は、`torch==2.1.2` と `torchvision==0.16.2` 、および `xformers==0.0.23.post1` を適宜変更してください。

accelerate configの質問には以下のように答えてください。（bf16で学習する場合、最後の質問にはbf16と答えてください。）

```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

※場合によって ``ValueError: fp16 mixed precision requires a GPU`` というエラーが出ることがあるようです。この場合、6番目の質問（
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:``）に「0」と答えてください。（id `0`のGPUが使われます。）

## アップグレード

新しいリリースがあった場合、以下のコマンドで更新できます。

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

コマンドが成功すれば新しいバージョンが使用できます。

## 謝意

LoRAの実装は[cloneofsimo氏のリポジトリ](https://github.com/cloneofsimo/lora)を基にしたものです。感謝申し上げます。

Conv2d 3x3への拡大は [cloneofsimo氏](https://github.com/cloneofsimo/lora) が最初にリリースし、KohakuBlueleaf氏が [LoCon](https://github.com/KohakuBlueleaf/LoCon) でその有効性を明らかにしたものです。KohakuBlueleaf氏に深く感謝します。

## ライセンス

スクリプトのライセンスはASL 2.0ですが（Diffusersおよびcloneofsimo氏のリポジトリ由来のものも同様）、一部他のライセンスのコードを含みます。

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause

## その他の情報

### LoRAの名称について

`train_network.py` がサポートするLoRAについて、混乱を避けるため名前を付けました。ドキュメントは更新済みです。以下は当リポジトリ内の独自の名称です。

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers、リエラと読みます)

    Linear 層およびカーネルサイズ 1x1 の Conv2d 層に適用されるLoRA

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers、セリアと読みます)

    1.に加え、カーネルサイズ 3x3 の Conv2d 層に適用されるLoRA

デフォルトではLoRA-LierLaが使われます。LoRA-C3Lierを使う場合は `--network_args` に `conv_dim` を指定してください。

<!-- 
LoRA-LierLa は[Web UI向け拡張](https://github.com/kohya-ss/sd-webui-additional-networks)、またはAUTOMATIC1111氏のWeb UIのLoRA機能で使用することができます。

LoRA-C3Lierを使いWeb UIで生成するには拡張を使用してください。
-->

### 学習中のサンプル画像生成

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
