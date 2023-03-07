## リポジトリについて
Stable Diffusionの学習、画像生成、その他のスクリプトを入れたリポジトリです。

[README in English](./README.md) ←更新情報はこちらにあります

GUIやPowerShellスクリプトなど、より使いやすくする機能が[bmaltais氏のリポジトリ](https://github.com/bmaltais/kohya_ss)で提供されています（英語です）のであわせてご覧ください。bmaltais氏に感謝します。

以下のスクリプトがあります。

* DreamBooth、U-NetおよびText Encoderの学習をサポート
* fine-tuning、同上
* 画像生成
* モデル変換（Stable Diffision ckpt/safetensorsとDiffusersの相互変換）

## 使用法について

当リポジトリ内およびnote.comに記事がありますのでそちらをご覧ください（将来的にはすべてこちらへ移すかもしれません）。

* [DreamBoothの学習について](./train_db_README-ja.md)
* [fine-tuningのガイド](./fine_tune_README_ja.md):
BLIPによるキャプショニングと、DeepDanbooruまたはWD14 taggerによるタグ付けを含みます
* [LoRAの学習について](./train_network_README-ja.md)
* [Textual Inversionの学習について](./train_ti_README-ja.md)
* note.com [画像生成スクリプト](https://note.com/kohya_ss/n/n2693183a798e)
* note.com [モデル変換スクリプト](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windowsでの動作に必要なプログラム

Python 3.10.6およびGitが必要です。

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

PowerShellを使う場合、venvを使えるようにするためには以下の手順でセキュリティ設定を変更してください。
（venvに限らずスクリプトの実行が可能になりますので注意してください。）

- PowerShellを管理者として開きます。
- 「Set-ExecutionPolicy Unrestricted」と入力し、Yと答えます。
- 管理者のPowerShellを閉じます。

## Windows環境でのインストール

以下の例ではPyTorchは1.12.1／CUDA 11.6版をインストールします。CUDA 11.3版やPyTorch 1.13を使う場合は適宜書き換えください。

（なお、python -m venv～の行で「python」とだけ表示された場合、py -m venv～のようにpythonをpyに変更してください。）

通常の（管理者ではない）PowerShellを開き以下を順に実行します。

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

<!-- 
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --use-pep517 --upgrade -r requirements.txt
pip install -U -I --no-deps xformers==0.0.16
-->

コマンドプロンプトでは以下になります。


```bat
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

copy /y .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
copy /y .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
copy /y .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config
```

（注:``python -m venv venv`` のほうが ``python -m venv --system-site-packages venv`` より安全そうなため書き換えました。globalなpythonにパッケージがインストールしてあると、後者だといろいろと問題が起きます。）

accelerate configの質問には以下のように答えてください。（bf16で学習する場合、最後の質問にはbf16と答えてください。）

※0.15.0から日本語環境では選択のためにカーソルキーを押すと落ちます（……）。数字キーの0、1、2……で選択できますので、そちらを使ってください。

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

### PyTorchとxformersのバージョンについて

他のバージョンでは学習がうまくいかない場合があるようです。特に他の理由がなければ指定のバージョンをお使いください。

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

## ライセンス

スクリプトのライセンスはASL 2.0ですが（Diffusersおよびcloneofsimo氏のリポジトリ由来のものも同様）、一部他のライセンスのコードを含みます。

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause


