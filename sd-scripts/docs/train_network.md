# How to use the LoRA training script `train_network.py` / LoRA学習スクリプト `train_network.py` の使い方

This document explains the basic procedures for training LoRA (Low-Rank Adaptation) models using `train_network.py` included in the `sd-scripts` repository.

<details>
<summary>日本語</summary>
このドキュメントでは、`sd-scripts` リポジトリに含まれる `train_network.py` を使用して LoRA (Low-Rank Adaptation) モデルを学習する基本的な手順について解説します。
</details>

## 1. Introduction / はじめに

`train_network.py` is a script for training additional networks such as LoRA on Stable Diffusion models (v1.x, v2.x). It allows for additional training on the original model with a low computational cost, enabling the creation of models that reproduce specific characters or art styles.

This guide focuses on LoRA training and explains the basic configuration items.

**Prerequisites:**

* The `sd-scripts` repository has been cloned and the Python environment has been set up.
* The training dataset has been prepared. (For dataset preparation, please refer to [this guide](link/to/dataset/doc))

<details>
<summary>日本語</summary>

`train_network.py` は、Stable Diffusion モデル（v1.x, v2.x）に対して、LoRA などの追加ネットワークを学習させるためのスクリプトです。少ない計算コストで元のモデルに追加学習を行い、特定のキャラクターや画風を再現するモデルを作成できます。

このガイドでは、LoRA 学習に焦点を当て、基本的な設定項目を中心に説明します。

**前提条件:**

*   `sd-scripts` リポジトリのクローンと Python 環境のセットアップが完了していること。
*   学習用データセットの準備が完了していること。（データセットの準備については[こちら](link/to/dataset/doc)を参照してください）
</details>

## 2. Preparation / 準備

Before starting training, you will need the following files:

1. **Training script:** `train_network.py`
2. **Dataset definition file (.toml):** A file in TOML format that describes the configuration of the training dataset.

### About the Dataset Definition File / データセット定義ファイルについて

The dataset definition file (`.toml`) contains detailed settings such as the directory of images to use, repetition count, caption settings, resolution buckets (optional), etc.

For more details on how to write the dataset definition file, please refer to the [Dataset Configuration Guide](./config_README-en.md).

In this guide, we will use a file named `my_dataset_config.toml` as an example.

<details>
<summary>日本語</summary>

学習を開始する前に、以下のファイルが必要です。

1.  **学習スクリプト:** `train_network.py`
2.  **データセット定義ファイル (.toml):** 学習データセットの設定を記述した TOML 形式のファイル。

**データセット定義ファイルについて**

データセット定義ファイル (`.toml`) には、使用する画像のディレクトリ、繰り返し回数、キャプションの設定、Aspect Ratio Bucketing（任意）などの詳細な設定を記述します。

データセット定義ファイルの詳しい書き方については、[データセット設定ガイド](./config_README-ja.md)を参照してください。

ここでは、例として `my_dataset_config.toml` という名前のファイルを使用することにします。
</details>

## 3. Running the Training / 学習の実行

Training is started by executing `train_network.py` from the terminal. When executing, various training settings are specified as command-line arguments.

Below is a basic command-line execution example:

```bash
accelerate launch --num_cpu_threads_per_process 1 train_network.py 
 --pretrained_model_name_or_path="<path to Stable Diffusion model>" 
 --dataset_config="my_dataset_config.toml" 
 --output_dir="<output directory for training results>" 
 --output_name="my_lora" 
 --save_model_as=safetensors 
 --network_module=networks.lora 
 --network_dim=16 
 --network_alpha=1 
 --learning_rate=1e-4 
 --optimizer_type="AdamW8bit" 
 --lr_scheduler="constant" 
 --sdpa  
 --max_train_epochs=10 
 --save_every_n_epochs=1 
 --mixed_precision="fp16" 
 --gradient_checkpointing 
```

In reality, you need to write this in a single line, but it's shown with line breaks for readability (on Linux or Mac, you can add `\` at the end of each line to break lines). For Windows, either write it in a single line without breaks or add `^` at the end of each line.

Next, we'll explain the main command-line arguments.

<details>
<summary>日本語</summary>

学習は、ターミナルから `train_network.py` を実行することで開始します。実行時には、学習に関する様々な設定をコマンドライン引数として指定します。

以下に、基本的なコマンドライン実行例を示します。

実際には1行で書く必要がありますが、見やすさのために改行しています（Linux や Mac では `\` を行末に追加することで改行できます）。Windows の場合は、改行せずに1行で書くか、`^` を行末に追加してください。

次に、主要なコマンドライン引数について解説します。
</details>

### 3.1. Main Command-Line Arguments / 主要なコマンドライン引数

#### Model Related / モデル関連

* `--pretrained_model_name_or_path="<path to model>"` **[Required]**
  * Specifies the Stable Diffusion model to be used as the base for training. You can specify the path to a local `.ckpt` or `.safetensors` file, or a directory containing a Diffusers format model. You can also specify a Hugging Face Hub model ID (e.g., `"stabilityai/stable-diffusion-2-1-base"`).
* `--v2`
  * Specify this when the base model is Stable Diffusion v2.x.
* `--v_parameterization`
  * Specify this when training with a v-prediction model (such as v2.x 768px models).

#### Dataset Related / データセット関連

* `--dataset_config="<path to configuration file>"`
  * Specifies the path to a `.toml` file describing the dataset configuration. (For details on dataset configuration, see [here](link/to/dataset/config/doc))
  * It's also possible to specify dataset settings from the command line, but using a `.toml` file is recommended as it becomes lengthy.

#### Output and Save Related / 出力・保存関連

* `--output_dir="<output directory>"` **[Required]**
  * Specifies the directory where trained LoRA models, sample images, logs, etc. will be output.
* `--output_name="<output filename>"` **[Required]**
  * Specifies the filename of the trained LoRA model (excluding the extension).
* `--save_model_as="safetensors"`
  * Specifies the format for saving the model. You can choose from `safetensors` (recommended), `ckpt`, or `pt`. The default is `safetensors`.
* `--save_every_n_epochs=1`
  * Saves the model every specified number of epochs. If not specified, only the final model will be saved.
* `--save_every_n_steps=1000`
  * Saves the model every specified number of steps. If both epoch and step saving are specified, both will be saved.

#### LoRA Parameters / LoRA パラメータ

* `--network_module=networks.lora` **[Required]**
  * Specifies the type of network to train. For LoRA, specify `networks.lora`.
* `--network_dim=16` **[Required]**
  * Specifies the rank (dimension) of LoRA. Higher values increase expressiveness but also increase file size and computational cost. Values between 4 and 128 are commonly used. There is no default (module dependent).
* `--network_alpha=1`
  * Specifies the alpha value for LoRA. This parameter is related to learning rate scaling. It is generally recommended to set it to about half the value of `network_dim`, but it can also be the same value as `network_dim`. The default is 1. Setting it to the same value as `network_dim` will result in behavior similar to older versions.
* `--network_args`
  * Used to specify additional parameters specific to the LoRA module. For example, to use Conv2d (3x3) LoRA (LoRA-C3Lier), specify the following in `--network_args`. Use `conv_dim` to specify the rank for Conv2d (3x3) and `conv_alpha` for alpha.
    ```
    --network_args "conv_dim=4" "conv_alpha=1"
    ```

    If alpha is omitted as shown below, it defaults to 1.
    ```
    --network_args "conv_dim=4"
    ```

#### Training Parameters / 学習パラメータ

* `--learning_rate=1e-4`
  * Specifies the learning rate. For LoRA training (when alpha value is 1), relatively higher values (e.g., from `1e-4` to `1e-3`) are often used.
* `--unet_lr=1e-4`
  * Used to specify a separate learning rate for the LoRA modules in the U-Net part. If not specified, the value of `--learning_rate` is used.
* `--text_encoder_lr=1e-5`
  * Used to specify a separate learning rate for the LoRA modules in the Text Encoder part. If not specified, the value of `--learning_rate` is used. A smaller value than that for U-Net is recommended.
* `--optimizer_type="AdamW8bit"`
  * Specifies the optimizer to use for training. Options include `AdamW8bit` (requires `bitsandbytes`), `AdamW`, `Lion` (requires `lion-pytorch`), `DAdaptation` (requires `dadaptation`), and `Adafactor`. `AdamW8bit` is memory-efficient and widely used.
* `--lr_scheduler="constant"`
  * Specifies the learning rate scheduler. This is the method for changing the learning rate as training progresses. Options include `constant` (no change), `cosine` (cosine curve), `linear` (linear decay), `constant_with_warmup` (constant with warmup), and `cosine_with_restarts`. `constant`, `cosine`, and `constant_with_warmup` are commonly used.
* `--lr_warmup_steps=500`
  * Specifies the number of warmup steps for the learning rate scheduler. This is the period during which the learning rate gradually increases at the start of training. Valid when the `lr_scheduler` supports warmup.
* `--max_train_steps=10000`
  * Specifies the total number of training steps. If `max_train_epochs` is specified, that takes precedence.
* `--max_train_epochs=12`
  * Specifies the number of training epochs. If this is specified, `max_train_steps` is ignored.
* `--sdpa`
  * Uses Scaled Dot-Product Attention. This can reduce memory usage and improve training speed for LoRA training.
* `--mixed_precision="fp16"`
  * Specifies the mixed precision training setting. Options are `no` (disabled), `fp16` (half precision), and `bf16` (bfloat16). If your GPU supports it, specifying `fp16` or `bf16` can improve training speed and reduce memory usage.
* `--gradient_accumulation_steps=1`
  * Specifies the number of steps to accumulate gradients. This effectively increases the batch size to `train_batch_size * gradient_accumulation_steps`. Set a larger value if GPU memory is insufficient. Usually `1` is fine.

#### Others / その他

* `--seed=42`
  * Specifies the random seed. Set this if you want to ensure reproducibility of the training.
* `--logging_dir="<log directory>"`
  * Specifies the directory to output logs for TensorBoard, etc. If not specified, logs will not be output.
* `--log_prefix="<prefix>"`
  * Specifies the prefix for the subdirectory name created within `logging_dir`.
* `--gradient_checkpointing`
  * Enables Gradient Checkpointing. This can significantly reduce memory usage but slightly decreases training speed. Useful when memory is limited.
* `--clip_skip=1`
  * Specifies how many layers to skip from the last layer of the Text Encoder. Specifying `2` will use the output from the second-to-last layer. `None` or `1` means no skip (uses the last layer). Check the recommended value for the model you are training.

<details>
<summary>日本語</summary>

#### モデル関連

*   `--pretrained_model_name_or_path="<モデルのパス>"` **[必須]**
    *   学習のベースとなる Stable Diffusion モデルを指定します。ローカルの `.ckpt` または `.safetensors` ファイルのパス、あるいは Diffusers 形式モデルのディレクトリを指定できます。Hugging Face Hub のモデル ID (例: `"stabilityai/stable-diffusion-2-1-base"`) も指定可能です。
*   `--v2`
    *   ベースモデルが Stable Diffusion v2.x の場合に指定します。
*   `--v_parameterization`
    *   v-prediction モデル（v2.x の 768px モデルなど）で学習する場合に指定します。

#### データセット関連

*   `--dataset_config="<設定ファイルのパス>"` 
    *   データセット設定を記述した `.toml` ファイルのパスを指定します。（データセット設定の詳細は[こちら](link/to/dataset/config/doc)）
    *   コマンドラインからデータセット設定を指定することも可能ですが、長くなるため `.toml` ファイルを使用することを推奨します。

#### 出力・保存関連

*   `--output_dir="<出力先ディレクトリ>"` **[必須]**
    *   学習済み LoRA モデルやサンプル画像、ログなどが出力されるディレクトリを指定します。
*   `--output_name="<出力ファイル名>"` **[必須]**
    *   学習済み LoRA モデルのファイル名（拡張子を除く）を指定します。
*   `--save_model_as="safetensors"`
    *   モデルの保存形式を指定します。`safetensors` (推奨), `ckpt`, `pt` から選択できます。デフォルトは `safetensors` です。
*   `--save_every_n_epochs=1`
    *   指定したエポックごとにモデルを保存します。省略するとエポックごとの保存は行われません（最終モデルのみ保存）。
*   `--save_every_n_steps=1000`
    *   指定したステップごとにモデルを保存します。エポック指定 (`save_every_n_epochs`) と同時に指定された場合、両方とも保存されます。

#### LoRA パラメータ

*   `--network_module=networks.lora` **[必須]**
    *   学習するネットワークの種別を指定します。LoRA の場合は `networks.lora` を指定します。
*   `--network_dim=16` **[必須]**
    *   LoRA のランク (rank / 次元数) を指定します。値が大きいほど表現力は増しますが、ファイルサイズと計算コストが増加します。一般的には 4〜128 程度の値が使われます。デフォルトは指定されていません（モジュール依存）。
*   `--network_alpha=1`
    *   LoRA のアルファ値 (alpha) を指定します。学習率のスケーリングに関係するパラメータで、一般的には `network_dim` の半分程度の値を指定することが推奨されますが、`network_dim` と同じ値を指定する場合もあります。デフォルトは 1 です。`network_dim` と同じ値に設定すると、旧バージョンと同様の挙動になります。

* `--network_args`
    *   LoRA モジュールに特有の追加パラメータを指定するために使用します。例えば、Conv2d (3x3) の LoRA  (LoRA-C3Lier) を使用する場合は`--network_args` に以下のように指定してください。`conv_dim` で Conv2d (3x3) の rank を、`conv_alpha` で alpha を指定します。
        ```
        --network_args "conv_dim=4" "conv_alpha=1"
        ```
        以下のように alpha を省略した時は1になります。
        ```
        --network_args "conv_dim=4"
        ```

#### 学習パラメータ

*   `--learning_rate=1e-4`
    *   学習率を指定します。LoRA 学習では（アルファ値が1の場合）比較的高めの値（例: `1e-4`から`1e-3`）が使われることが多いです。
*   `--unet_lr=1e-4`
    *   U-Net 部分の LoRA モジュールに対する学習率を個別に指定する場合に使用します。指定しない場合は `--learning_rate` の値が使用されます。
*   `--text_encoder_lr=1e-5`
    *   Text Encoder 部分の LoRA モジュールに対する学習率を個別に指定する場合に使用します。指定しない場合は `--learning_rate` の値が使用されます。U-Net よりも小さめの値が推奨されます。
*   `--optimizer_type="AdamW8bit"`
    *   学習に使用するオプティマイザを指定します。`AdamW8bit` (要 `bitsandbytes`), `AdamW`, `Lion` (要 `lion-pytorch`), `DAdaptation` (要 `dadaptation`), `Adafactor` などが選択可能です。`AdamW8bit` はメモリ効率が良く、広く使われています。
*   `--lr_scheduler="constant"`
    *   学習率スケジューラを指定します。学習の進行に合わせて学習率を変化させる方法です。`constant` (変化なし), `cosine` (コサインカーブ), `linear` (線形減衰), `constant_with_warmup` (ウォームアップ付き定数), `cosine_with_restarts` などが選択可能です。`constant`や`cosine` 、 `constant_with_warmup` がよく使われます。
*   `--lr_warmup_steps=500`
    *   学習率スケジューラのウォームアップステップ数を指定します。学習開始時に学習率を徐々に上げていく期間です。`lr_scheduler` がウォームアップをサポートする場合に有効です。
*   `--max_train_steps=10000`
    *   学習の総ステップ数を指定します。`max_train_epochs` が指定されている場合はそちらが優先されます。
*   `--max_train_epochs=12`
    *   学習のエポック数を指定します。これを指定すると `max_train_steps` は無視されます。
*   `--sdpa`
    *   Scaled Dot-Product Attention を使用します。LoRA の学習において、メモリ使用量を削減し、学習速度を向上させることができます。
*   `--mixed_precision="fp16"`
    *   混合精度学習の設定を指定します。`no` (無効), `fp16` (半精度), `bf16` (bfloat16) から選択できます。GPU が対応している場合は `fp16` または `bf16` を指定することで、学習速度の向上とメモリ使用量の削減が期待できます。
*   `--gradient_accumulation_steps=1`
    *   勾配を累積するステップ数を指定します。実質的なバッチサイズを `train_batch_size * gradient_accumulation_steps` に増やす効果があります。GPU メモリが足りない場合に大きな値を設定します。通常は `1` で問題ありません。

#### その他

*   `--seed=42`
    *   乱数シードを指定します。学習の再現性を確保したい場合に設定します。
*   `--logging_dir="<ログディレクトリ>"`
    *   TensorBoard などのログを出力するディレクトリを指定します。指定しない場合、ログは出力されません。
*   `--log_prefix="<プレフィックス>"`
    *   `logging_dir` 内に作成されるサブディレクトリ名の接頭辞を指定します。
*   `--gradient_checkpointing`
    *   Gradient Checkpointing を有効にします。メモリ使用量を大幅に削減できますが、学習速度は若干低下します。メモリが厳しい場合に有効です。
*   `--clip_skip=1`
    *   Text Encoder の最後の層から数えて何層スキップするかを指定します。`2` を指定すると最後から 2 層目の出力を使用します。`None` または `1` はスキップなし（最後の層を使用）を意味します。学習対象のモデルの推奨する値を確認してください。
</details>

### 3.2. Starting the Training / 学習の開始

After setting the necessary arguments and executing the command, training will begin. The progress of the training will be output to the console. If `logging_dir` is specified, you can visually check the training status (loss, learning rate, etc.) with TensorBoard.

```bash
tensorboard --logdir <directory specified by logging_dir>
```

<details>
<summary>日本語</summary>

必要な引数を設定し、コマンドを実行すると学習が開始されます。学習の進行状況はコンソールに出力されます。`logging_dir` を指定した場合は、TensorBoard などで学習状況（損失や学習率など）を視覚的に確認できます。
</details>

## 4. Using the Trained Model / 学習済みモデルの利用

Once training is complete, a LoRA model file (`.safetensors` or `.ckpt`) with the name specified by `output_name` will be saved in the directory specified by `output_dir`.

This file can be used with GUI tools such as AUTOMATIC1111/stable-diffusion-webui, ComfyUI, etc.

<details>
<summary>日本語</summary>

学習が完了すると、`output_dir` で指定したディレクトリに、`output_name` で指定した名前の LoRA モデルファイル (`.safetensors` または `.ckpt`) が保存されます。

このファイルは、AUTOMATIC1111/stable-diffusion-webui 、ComfyUI などの GUI ツールで利用できます。
</details>

## 5. Other Features / その他の機能

`train_network.py` has many other options not introduced here.

* Sample image generation (`--sample_prompts`, `--sample_every_n_steps`, etc.)
* More detailed optimizer settings (`--optimizer_args`, etc.)
* Caption preprocessing (`--shuffle_caption`, `--keep_tokens`, etc.)
* Additional network settings (`--network_args`, etc.)

For these features, please refer to the script's help (`python train_network.py --help`) or other documents in the repository.

<details>
<summary>日本語</summary>

`train_network.py` には、ここで紹介した以外にも多くのオプションがあります。

*   サンプル画像の生成 (`--sample_prompts`, `--sample_every_n_steps` など)
*   より詳細なオプティマイザ設定 (`--optimizer_args` など)
*   キャプションの前処理 (`--shuffle_caption`, `--keep_tokens` など)
*   ネットワークの追加設定 (`--network_args` など)

これらの機能については、スクリプトのヘルプ (`python train_network.py --help`) やリポジトリ内の他のドキュメントを参照してください。
</details>

## 6. Additional Information / 追加情報

### Naming of LoRA

The LoRA supported by `train_network.py` has been named to avoid confusion. The documentation has been updated. The following are the names of LoRA types in this repository.

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers)

    LoRA for Linear layers and Conv2d layers with 1x1 kernel

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers)

    In addition to 1., LoRA for Conv2d layers with 3x3 kernel 
    
LoRA-LierLa is the default LoRA type for `train_network.py` (without `conv_dim` network arg). 

<details>
<summary>日本語</summary>

`train_network.py` がサポートするLoRAについて、混乱を避けるため名前を付けました。ドキュメントは更新済みです。以下は当リポジトリ内の独自の名称です。

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers、リエラと読みます)

    Linear 層およびカーネルサイズ 1x1 の Conv2d 層に適用されるLoRA

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers、セリアと読みます)

    1.に加え、カーネルサイズ 3x3 の Conv2d 層に適用されるLoRA

デフォルトではLoRA-LierLaが使われます。LoRA-C3Lierを使う場合は `--network_args` に `conv_dim` を指定してください。

</details>