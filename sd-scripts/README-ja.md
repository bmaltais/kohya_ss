# sd-scripts

[English](./README.md) / [日本語](./README-ja.md)

## 目次

<details>
<summary>クリックすると展開します</summary>

- [はじめに](#はじめに)
  - [スポンサー](#スポンサー)
  - [スポンサー募集のお知らせ](#スポンサー募集のお知らせ)
  - [更新履歴](#更新履歴)
  - [サポートモデル](#サポートモデル)
  - [機能](#機能)
- [ドキュメント](#ドキュメント)
  - [学習ドキュメント（英語および日本語）](#学習ドキュメント英語および日本語)
  - [その他のドキュメント](#その他のドキュメント)
  - [旧ドキュメント（日本語）](#旧ドキュメント日本語)
- [AIコーディングエージェントを使う開発者の方へ](#aiコーディングエージェントを使う開発者の方へ)
- [Windows環境でのインストール](#windows環境でのインストール)
  - [Windowsでの動作に必要なプログラム](#windowsでの動作に必要なプログラム)
  - [インストール手順](#インストール手順)
  - [requirements.txtとPyTorchについて](#requirementstxtとpytorchについて)
  - [xformersのインストール（オプション）](#xformersのインストールオプション)
- [Linux/WSL2環境でのインストール](#linuxwsl2環境でのインストール)
  - [DeepSpeedのインストール（実験的、LinuxまたはWSL2のみ）](#deepspeedのインストール実験的linuxまたはwsl2のみ)
- [アップグレード](#アップグレード)
  - [PyTorchのアップグレード](#pytorchのアップグレード)
- [謝意](#謝意)
- [ライセンス](#ライセンス)

</details>

## はじめに

Stable Diffusion等の画像生成モデルの学習、モデルによる画像生成、その他のスクリプトを入れたリポジトリです。

### スポンサー

このプロジェクトを支援してくださる企業・団体の皆様に深く感謝いたします。

<a href="https://aihub.co.jp/">
  <img src="./images/logo_aihub.png" alt="AiHUB株式会社" title="AiHUB株式会社" height="100px">
</a>

### スポンサー募集のお知らせ

このプロジェクトがお役に立ったなら、ご支援いただけると嬉しく思います。 [GitHub Sponsors](https://github.com/sponsors/kohya-ss/)で受け付けています。

### 更新履歴

- **Version 0.10.0 (2026-01-19):**
  - `sd3`ブランチを`main`ブランチにマージしました。このバージョンからFLUX.1およびSD3/SD3.5等のモデルが`main`ブランチでサポートされます。
  - ドキュメントにはまだ不備があるため、お気づきの点はIssue等でお知らせください。
  - `sd3`ブランチは当面、`dev`ブランチと同期して開発ブランチとして維持します。

### サポートモデル

* **Stable Diffusion 1.x/2.x**
* **SDXL**
* **SD3/SD3.5**
* **FLUX.1**
* **LUMINA**
* **HunyuanImage-2.1**

### 機能

* LoRA学習
* fine-tuning（DreamBooth）：HunyuanImage-2.1以外のモデル
* Textual Inversion学習：SD/SDXL
* 画像生成
* その他、モデル変換やタグ付け、LoRAマージなどのユーティリティ

## ドキュメント

### 学習ドキュメント（英語および日本語）

日本語は折りたたまれているか、別のドキュメントにあります。

* [LoRA学習の概要](./docs/train_network.md)
* [データセット設定](./docs/config_README-ja.md) / [英語版](./docs/config_README-en.md)
* [高度な学習オプション](./docs/train_network_advanced.md)
* [SDXL学習](./docs/sdxl_train_network.md)
* [SD3学習](./docs/sd3_train_network.md)
* [FLUX.1学習](./docs/flux_train_network.md)
* [LUMINA学習](./docs/lumina_train_network.md)
* [HunyuanImage-2.1学習](./docs/hunyuan_image_train_network.md)
* [Fine-tuning](./docs/fine_tune.md)
* [Textual Inversion学習](./docs/train_textual_inversion.md)
* [ControlNet-LLLite学習](./docs/train_lllite_README-ja.md) / [英語版](./docs/train_lllite_README.md)
* [Validation](./docs/validation.md)
* [マスク損失学習](./docs/masked_loss_README-ja.md) / [英語版](./docs/masked_loss_README.md)

### その他のドキュメント

* [画像生成スクリプト](./docs/gen_img_README-ja.md) / [英語版](./docs/gen_img_README.md)
* [WD14 Taggerによる画像タグ付け](./docs/wd14_tagger_README-ja.md) / [英語版](./docs/wd14_tagger_README-en.md)

### 旧ドキュメント（日本語）

* [学習について、共通編](./docs/train_README-ja.md) : データ整備やオプションなど
* [DreamBoothの学習について](./docs/train_db_README-ja.md)

## AIコーディングエージェントを使う開発者の方へ

This repository provides recommended instructions to help AI agents like Claude and Gemini understand our project context and coding standards.

To use them, you need to opt-in by creating your own configuration file in the project root.

**Quick Setup:**

1.  Create a `CLAUDE.md` and/or `GEMINI.md` file in the project root.
2.  Add the following line to your `CLAUDE.md` to import the repository's recommended prompt:

    ```markdown
    @./.ai/claude.prompt.md
    ```

    or for Gemini:

    ```markdown
    @./.ai/gemini.prompt.md
    ```

3.  You can now add your own personal instructions below the import line (e.g., `Always respond in Japanese.`).

This approach ensures that you have full control over the instructions given to your agent while benefiting from the shared project context. Your `CLAUDE.md` and `GEMINI.md` are already listed in `.gitignore`, so they won't be committed to the repository.

このリポジトリでは、AIコーディングエージェント（例：Claude、Geminiなど）がプロジェクトのコンテキストやコーディング標準を理解できるようにするための推奨プロンプトを提供しています。

それらを使用するには、プロジェクトディレクトリに設定ファイルを作成して明示的に有効にする必要があります。

**簡単なセットアップ手順:**

1.  プロジェクトルートに `CLAUDE.md` や `GEMINI.md` ファイルを作成します。
2.  `CLAUDE.md` に以下の行を追加して、リポジトリの推奨プロンプトをインポートします。

    ```markdown
    @./.ai/claude.prompt.md
    ```

    またはGeminiの場合:

    ```markdown
    @./.ai/gemini.prompt.md
    ``` 
3.  インポート行の下に、独自の指示を追加できます（例：`常に日本語で応答してください。`）。

この方法により、エージェントに与える指示を各開発者が管理しつつ、リポジトリの推奨コンテキストを活用できます。`CLAUDE.md` および `GEMINI.md` は `.gitignore` に登録されているため、リポジトリにコミットされることはありません。

## Windows環境でのインストール

### Windowsでの動作に必要なプログラム

Python 3.10.xおよびGitが必要です。

- Python 3.10.x: https://www.python.org/downloads/windows/ からWindows installer (64-bit)をダウンロード
- git: https://git-scm.com/download/win から最新版をダウンロード

Python 3.11.x、3.12.xでも恐らく動作します（未テスト）。

PowerShellを使う場合、venvを使えるようにするためには以下の手順でセキュリティ設定を変更してください。
（venvに限らずスクリプトの実行が可能になりますので注意してください。）

- PowerShellを管理者として開きます。
- 「Set-ExecutionPolicy Unrestricted」と入力し、Yと答えます。
- 管理者のPowerShellを閉じます。

### インストール手順

PowerShellを使う場合、通常の（管理者ではない）PowerShellを開き以下を順に実行します。

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt

accelerate config
```

コマンドプロンプトでも同一です。

（なお、python -m venv～の行で「python」とだけ表示された場合、py -m venv～のようにpythonをpyに変更してください。）

注：`bitsandbytes`、`prodigyopt`、`lion-pytorch` は `requirements.txt` に含まれています。

この例ではCUDA 12.4版をインストールします。異なるバージョンのCUDAを使用する場合は、適切なバージョンのPyTorchをインストールしてください。たとえばCUDA 12.1版の場合は `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121` としてください。

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

### requirements.txtとPyTorchについて

PyTorchは環境によってバージョンが異なるため、requirements.txtには含まれていません。前述のインストール手順を参考に、環境に合わせてPyTorchをインストールしてください。

スクリプトはPyTorch 2.6.0でテストしています。PyTorch 2.6.0以降が必要です。

RTX 50シリーズGPUの場合、PyTorch 2.8.0とCUDA 12.8/12.9を使用してください。`requirements.txt`はこのバージョンでも動作します。

### xformersのインストール（オプション）

xformersをインストールするには、仮想環境を有効にした状態で以下のコマンドを実行してください。

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu124
```

必要に応じてCUDAバージョンを変更してください。一部のGPUアーキテクチャではxformersが利用できない場合があります。

## Linux/WSL2環境でのインストール

LinuxまたはWSL2環境でのインストール手順はWindows環境とほぼ同じです。`venv\Scripts\activate` の部分を `source venv/bin/activate` に変更してください。

※NVIDIAドライバやCUDAツールキットなどは事前にインストールしておいてください。

### DeepSpeedのインストール（実験的、LinuxまたはWSL2のみ）

DeepSpeedをインストールするには、仮想環境を有効にした状態で以下のコマンドを実行してください。

```bash
pip install deepspeed==0.16.7
```

## アップグレード

新しいリリースがあった場合、以下のコマンドで更新できます。

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

コマンドが成功すれば新しいバージョンが使用できます。

### PyTorchのアップグレード

PyTorchをアップグレードする場合は、[Windows環境でのインストール](#windows環境でのインストール)のセクションの`pip install`コマンドを参考にしてください。

## 謝意

LoRAの実装は[cloneofsimo氏のリポジトリ](https://github.com/cloneofsimo/lora)を基にしたものです。感謝申し上げます。

Conv2d 3x3への拡大は [cloneofsimo氏](https://github.com/cloneofsimo/lora) が最初にリリースし、KohakuBlueleaf氏が [LoCon](https://github.com/KohakuBlueleaf/LoCon) でその有効性を明らかにしたものです。KohakuBlueleaf氏に深く感謝します。

## ライセンス

スクリプトのライセンスはASL 2.0ですが（Diffusersおよびcloneofsimo氏のリポジトリ由来のものも同様）、一部他のライセンスのコードを含みます。

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause
