Status: reviewed

# LoRA Training Guide for HunyuanImage-2.1 using `hunyuan_image_train_network.py` / `hunyuan_image_train_network.py` を用いたHunyuanImage-2.1モデルのLoRA学習ガイド

This document explains how to train LoRA models for the HunyuanImage-2.1 model using `hunyuan_image_train_network.py` included in the `sd-scripts` repository.

<details>
<summary>日本語</summary>

このドキュメントでは、`sd-scripts`リポジトリに含まれる`hunyuan_image_train_network.py`を使用して、HunyuanImage-2.1モデルに対するLoRA (Low-Rank Adaptation) モデルを学習する基本的な手順について解説します。

</details>

## 1. Introduction / はじめに

`hunyuan_image_train_network.py` trains additional networks such as LoRA on the HunyuanImage-2.1 model, which uses a transformer-based architecture (DiT) different from Stable Diffusion. Two text encoders, Qwen2.5-VL and byT5, and a dedicated VAE are used.

This guide assumes you know the basics of LoRA training. For common options see [train_network.py](train_network.md) and [sdxl_train_network.py](sdxl_train_network.md).

**Prerequisites:**

* The repository is cloned and the Python environment is ready.
* A training dataset is prepared. See the dataset configuration guide.

<details>
<summary>日本語</summary>

`hunyuan_image_train_network.py`はHunyuanImage-2.1モデルに対してLoRAなどの追加ネットワークを学習させるためのスクリプトです。HunyuanImage-2.1はStable Diffusionとは異なるDiT (Diffusion Transformer) アーキテクチャを持つ画像生成モデルであり、このスクリプトを使用することで、特定のキャラクターや画風を再現するLoRAモデルを作成できます。

このガイドは、基本的なLoRA学習の手順を理解しているユーザーを対象としています。基本的な使い方や共通のオプションについては、[`train_network.py`のガイド](train_network.md)を参照してください。また一部のパラメータは [`sdxl_train_network.py`](sdxl_train_network.md) や [`flux_train_network.py`](flux_train_network.md) と同様のものがあるため、そちらも参考にしてください。

**前提条件:**

* `sd-scripts`リポジトリのクローンとPython環境のセットアップが完了していること。
* 学習用データセットの準備が完了していること。（データセットの準備については[データセット設定ガイド](config_README-ja.md)を参照してください）

</details>

## 2. Differences from `train_network.py` / `train_network.py` との違い

`hunyuan_image_train_network.py` is based on `train_network.py` but adapted for HunyuanImage-2.1. Main differences include:

* **Target model:** HunyuanImage-2.1 model.
* **Model structure:** HunyuanImage-2.1 uses a Transformer-based architecture (DiT). It uses two text encoders (Qwen2.5-VL and byT5) and a dedicated VAE.
* **Required arguments:** Additional arguments for the DiT model, Qwen2.5-VL, byT5, and VAE model files.
* **Incompatible options:** Some Stable Diffusion-specific arguments (e.g., `--v2`, `--clip_skip`, `--max_token_length`) are not used.
* **HunyuanImage-2.1-specific arguments:** Additional arguments for specific training parameters like flow matching.

<details>
<summary>日本語</summary>

`hunyuan_image_train_network.py`は`train_network.py`をベースに、HunyuanImage-2.1モデルに対応するための変更が加えられています。主な違いは以下の通りです。

* **対象モデル:** HunyuanImage-2.1モデルを対象とします。
* **モデル構造:** HunyuanImage-2.1はDiTベースのアーキテクチャを持ちます。Text EncoderとしてQwen2.5-VLとbyT5の二つを使用し、専用のVAEを使用します。
* **必須の引数:** DiTモデル、Qwen2.5-VL、byT5、VAEの各モデルファイルを指定する引数が追加されています。
* **一部引数の非互換性:** Stable Diffusion向けの引数の一部（例: `--v2`, `--clip_skip`, `--max_token_length`）は使用されません。
* **HunyuanImage-2.1特有の引数:** Flow Matchingなど、特有の学習パラメータを指定する引数が追加されています。

</details>

## 3. Preparation / 準備

Before starting training you need:

1. **Training script:** `hunyuan_image_train_network.py`
2. **HunyuanImage-2.1 DiT model file:** Base DiT model `.safetensors` file.
3. **Text Encoder model files:**
   - Qwen2.5-VL model file (`--text_encoder`).
   - byT5 model file (`--byt5`).
4. **VAE model file:** HunyuanImage-2.1-compatible VAE model `.safetensors` file (`--vae`).
5. **Dataset definition file (.toml):** TOML format file describing training dataset configuration.

### Downloading Required Models

To train HunyuanImage-2.1 models, you need to download the following model files:

- **DiT Model**: Download from the [Tencent HunyuanImage-2.1](https://huggingface.co/tencent/HunyuanImage-2.1/) repository. Use `dit/hunyuanimage2.1.safetensors`.
- **Text Encoders and VAE**: Download from the [Comfy-Org/HunyuanImage_2.1_ComfyUI](https://huggingface.co/Comfy-Org/HunyuanImage_2.1_ComfyUI) repository:
  - Qwen2.5-VL: `split_files/text_encoders/qwen_2.5_vl_7b.safetensors`
  - byT5: `split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors`
  - VAE: `split_files/vae/hunyuan_image_2.1_vae_fp16.safetensors`

<details>
<summary>日本語</summary>

学習を開始する前に、以下のファイルが必要です。

1. **学習スクリプト:** `hunyuan_image_train_network.py`
2. **HunyuanImage-2.1 DiTモデルファイル:** 学習のベースとなるDiTモデルの`.safetensors`ファイル。
3. **Text Encoderモデルファイル:**
   - Qwen2.5-VLモデルファイル (`--text_encoder`)。
   - byT5モデルファイル (`--byt5`)。
4. **VAEモデルファイル:** HunyuanImage-2.1に対応するVAEモデルの`.safetensors`ファイル (`--vae`)。
5. **データセット定義ファイル (.toml):** 学習データセットの設定を記述したTOML形式のファイル。（詳細は[データセット設定ガイド](config_README-ja.md)を参照してください）。

**必要なモデルのダウンロード**

HunyuanImage-2.1モデルを学習するためには、以下のモデルファイルをダウンロードする必要があります：

- **DiTモデル**: [Tencent HunyuanImage-2.1](https://huggingface.co/tencent/HunyuanImage-2.1/) リポジトリから `dit/hunyuanimage2.1.safetensors` をダウンロードします。
- **Text EncoderとVAE**: [Comfy-Org/HunyuanImage_2.1_ComfyUI](https://huggingface.co/Comfy-Org/HunyuanImage_2.1_ComfyUI) リポジトリから以下をダウンロードします：
  - Qwen2.5-VL: `split_files/text_encoders/qwen_2.5_vl_7b.safetensors`
  - byT5: `split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors`
  - VAE: `split_files/vae/hunyuan_image_2.1_vae_fp16.safetensors`

</details>

## 4. Running the Training / 学習の実行

Run `hunyuan_image_train_network.py` from the terminal with HunyuanImage-2.1 specific arguments. Here's a basic command example:

```bash
accelerate launch --num_cpu_threads_per_process 1 hunyuan_image_train_network.py \
  --pretrained_model_name_or_path="<path to HunyuanDiT model>" \
  --text_encoder="<path to Qwen2.5-VL model>" \
  --byt5="<path to byT5 model>" \
  --vae="<path to VAE model>" \
  --dataset_config="my_hunyuan_dataset_config.toml" \
  --output_dir="<output directory>" \
  --output_name="my_hunyuan_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora_hunyuan_image \
  --network_dim=16 \
  --network_alpha=1 \
  --network_train_unet_only \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW8bit" \
  --lr_scheduler="constant" \
  --attn_mode="torch" \
  --split_attn \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --model_prediction_type="raw" \
  --discrete_flow_shift=5.0 \
  --blocks_to_swap=18 \
  --cache_text_encoder_outputs \
  --cache_latents
```

**HunyuanImage-2.1 training does not support LoRA modules for Text Encoders, so `--network_train_unet_only` is required.**

<details>
<summary>日本語</summary>

学習は、ターミナルから`hunyuan_image_train_network.py`を実行することで開始します。基本的なコマンドラインの構造は`train_network.py`と同様ですが、HunyuanImage-2.1特有の引数を指定する必要があります。

コマンドラインの例は英語のドキュメントを参照してください。

</details>

### 4.1. Explanation of Key Options / 主要なコマンドライン引数の解説

The script adds HunyuanImage-2.1 specific arguments. For common arguments (like `--output_dir`, `--output_name`, `--network_module`, etc.), see the [`train_network.py` guide](train_network.md).

#### Model-related [Required]

* `--pretrained_model_name_or_path="<path to HunyuanDiT model>"` **[Required]**
  - Specifies the path to the base DiT model `.safetensors` file.
* `--text_encoder="<path to Qwen2.5-VL model>"` **[Required]**
  - Specifies the path to the Qwen2.5-VL Text Encoder model file. Should be `bfloat16`.
* `--byt5="<path to byT5 model>"` **[Required]**
  - Specifies the path to the byT5 Text Encoder model file. Should be `float16`.
* `--vae="<path to VAE model>"` **[Required]**
  - Specifies the path to the HunyuanImage-2.1-compatible VAE model `.safetensors` file.

#### HunyuanImage-2.1 Training Parameters

* `--network_train_unet_only` **[Required]**
  - Specifies that only the DiT model will be trained. LoRA modules for Text Encoders are not supported.
* `--discrete_flow_shift=<float>`
  - Specifies the shift value for the scheduler used in Flow Matching. Default is `5.0`.
* `--model_prediction_type=<choice>`
  - Specifies what the model predicts. Choose from `raw`, `additive`, `sigma_scaled`. Default and recommended is `raw`.
* `--timestep_sampling=<choice>`
  - Specifies the sampling method for timesteps (noise levels) during training. Choose from `sigma`, `uniform`, `sigmoid`, `shift`, `flux_shift`. Default is `sigma`.
* `--sigmoid_scale=<float>`
  - Scale factor when `timestep_sampling` is set to `sigmoid`, `shift`, or `flux_shift`. Default is `1.0`.

#### Memory/Speed Related

* `--attn_mode=<choice>`
  - Specifies the attention implementation to use. Options are `torch`, `xformers`, `flash`, `sageattn`. Default is `torch` (use scaled dot product attention). Each library must be installed separately other than `torch`. If using `xformers`, also specify `--split_attn` if the batch size is more than 1.
* `--split_attn`
  - Splits the batch during attention computation to process one item at a time, reducing VRAM usage by avoiding attention mask computation. Can improve speed when using `torch`. Required when using `xformers` with batch size greater than 1.
* `--fp8_scaled`
  - Enables training the DiT model in scaled FP8 format. This can significantly reduce VRAM usage (can run with as little as 8GB VRAM when combined with `--blocks_to_swap`), but the training results may vary. This is a newer alternative to the unsupported `--fp8_base` option. See [Musubi Tuner's documentation](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/advanced_config.md#fp8-weight-optimization-for-models--%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E9%87%8D%E3%81%BF%E3%81%AEfp8%E3%81%B8%E3%81%AE%E6%9C%80%E9%81%A9%E5%8C%96) for details.
* `--fp8_vl`
  - Use FP8 for the VLM (Qwen2.5-VL) text encoder.
* `--text_encoder_cpu`
  - Runs the text encoders on CPU to reduce VRAM usage. This is useful when VRAM is insufficient (less than 12GB). Encoding one text may take a few minutes (depending on CPU). It is highly recommended to use this option with `--cache_text_encoder_outputs_to_disk` to avoid repeated encoding every time training starts. **In addition, increasing `--num_cpu_threads_per_process` in the `accelerate launch` command, like `--num_cpu_threads_per_process=8` or `16`, can speed up encoding in some environments.**
* `--blocks_to_swap=<integer>` **[Experimental Feature]**
  - Setting to reduce VRAM usage by swapping parts of the model (Transformer blocks) between CPU and GPU. Specify the number of blocks to swap as an integer (e.g., `18`). Larger values reduce VRAM usage but decrease training speed. Adjust according to your GPU's VRAM capacity. Can be used with `gradient_checkpointing`.
* `--cache_text_encoder_outputs`
  - Caches the outputs of Qwen2.5-VL and byT5. This reduces memory usage.
* `--cache_latents`, `--cache_latents_to_disk`
  - Caches the outputs of VAE. Similar functionality to [sdxl_train_network.py](sdxl_train_network.md).
* `--vae_chunk_size=<integer>`
  - Enables chunked processing in the VAE to reduce VRAM usage during encoding and decoding. Specify the chunk size as an integer (e.g., `16`). Larger values use more VRAM but are faster. Default is `None` (no chunking). This option is useful when VRAM is limited (e.g., 8GB or 12GB).

<details>
<summary>日本語</summary>

[`train_network.py`のガイド](train_network.md)で説明されている引数に加え、以下のHunyuanImage-2.1特有の引数を指定します。共通の引数（`--output_dir`, `--output_name`, `--network_module`, `--network_dim`, `--network_alpha`, `--learning_rate`など）については、上記ガイドを参照してください。

コマンドラインの例と詳細な引数の説明は英語のドキュメントを参照してください。

</details>

## 5. Using the Trained Model / 学習済みモデルの利用

After training, a LoRA model file is saved in `output_dir` and can be used in inference environments supporting HunyuanImage-2.1.

<details>
<summary>日本語</summary>

学習が完了すると、指定した`output_dir`にLoRAモデルファイル（例: `my_hunyuan_lora.safetensors`）が保存されます。このファイルは、HunyuanImage-2.1モデルに対応した推論環境で使用できます。

</details>

## 6. Advanced Settings / 高度な設定

### 6.1. VRAM Usage Optimization / VRAM使用量の最適化

HunyuanImage-2.1 is a large model, so GPUs without sufficient VRAM require optimization.

#### Recommended Settings by GPU Memory

Based on testing with the pull request, here are recommended VRAM optimization settings:

| GPU Memory | Recommended Settings |
|------------|---------------------|
| 40GB+ VRAM | Standard settings (no special optimization needed) |
| 24GB VRAM  | `--fp8_scaled --blocks_to_swap 9` |
| 12GB VRAM  | `--fp8_scaled --blocks_to_swap 32` |
| 8GB VRAM   | `--fp8_scaled --blocks_to_swap 37` |

#### Key VRAM Reduction Options

- **`--fp8_scaled`**: Enables training the DiT in scaled FP8 format. This is the recommended FP8 option for HunyuanImage-2.1, replacing the unsupported `--fp8_base` option. Essential for <40GB VRAM environments.
- **`--fp8_vl`**: Use FP8 for the VLM (Qwen2.5-VL) text encoder.
- **`--blocks_to_swap <number>`**: Swaps blocks between CPU and GPU to reduce VRAM usage. Higher numbers save more VRAM but reduce training speed. Up to 37 blocks can be swapped for HunyuanImage-2.1.
- **`--cpu_offload_checkpointing`**: Offloads gradient checkpoints to CPU. Can reduce VRAM usage but decreases training speed. Cannot be used with `--blocks_to_swap`.
- **Using Adafactor optimizer**: Can reduce VRAM usage more than 8bit AdamW:
  ```
  --optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --lr_scheduler constant_with_warmup --max_grad_norm 0.0
  ```

<details>
<summary>日本語</summary>

HunyuanImage-2.1は大きなモデルであるため、十分なVRAMを持たないGPUでは工夫が必要です。

#### GPU別推奨設定

Pull Requestのテスト結果に基づく推奨VRAM最適化設定：

| GPU Memory | 推奨設定 |
|------------|---------|
| 40GB+ VRAM | 標準設定（特別な最適化不要） |
| 24GB VRAM  | `--fp8_scaled --blocks_to_swap 9` |
| 12GB VRAM  | `--fp8_scaled --blocks_to_swap 32` |
| 8GB VRAM   | `--fp8_scaled --blocks_to_swap 37` |

主要なVRAM削減オプション：
- `--fp8_scaled`: DiTをスケールされたFP8形式で学習（推奨されるFP8オプション、40GB VRAM未満の環境では必須）
- `--fp8_vl`: VLMテキストエンコーダにFP8を使用
- `--blocks_to_swap`: CPUとGPU間でブロックをスワップ（最大37ブロック）
- `--cpu_offload_checkpointing`: 勾配チェックポイントをCPUにオフロード
- Adafactorオプティマイザの使用

</details>

### 6.2. Important HunyuanImage-2.1 LoRA Training Settings / HunyuanImage-2.1 LoRA学習の重要な設定

HunyuanImage-2.1 training has several settings that can be specified with arguments:

#### Timestep Sampling Methods

The `--timestep_sampling` option specifies how timesteps (0-1) are sampled:

- `sigma`: Sigma-based like SD3 (Default)
- `uniform`: Uniform random
- `sigmoid`: Sigmoid of normal distribution random
- `shift`: Sigmoid value of normal distribution random with shift.
- `flux_shift`: Shift sigmoid value of normal distribution random according to resolution.

#### Model Prediction Processing

The `--model_prediction_type` option specifies how to interpret and process model predictions:

- `raw`: Use as-is **[Recommended, Default]**
- `additive`: Add to noise input
- `sigma_scaled`: Apply sigma scaling

#### Recommended Settings

Based on experiments, the default settings work well:
```
--model_prediction_type raw --discrete_flow_shift 5.0
```

<details>
<summary>日本語</summary>

HunyuanImage-2.1の学習には、引数で指定できるいくつかの設定があります。詳細な説明とコマンドラインの例は英語のドキュメントを参照してください。

主要な設定オプション：
- タイムステップのサンプリング方法（`--timestep_sampling`）
- モデル予測の処理方法（`--model_prediction_type`）
- 推奨設定の組み合わせ

</details>

### 6.3. Regular Expression-based Rank/LR Configuration / 正規表現によるランク・学習率の指定

You can specify ranks (dims) and learning rates for LoRA modules using regular expressions. This allows for more flexible and fine-grained control.

These settings are specified via the `network_args` argument.

*   `network_reg_dims`: Specify ranks for modules matching a regular expression. The format is a comma-separated string of `pattern=rank`.
    *   Example: `--network_args "network_reg_dims=attn.*.q_proj=4,attn.*.k_proj=4"`
*   `network_reg_lrs`: Specify learning rates for modules matching a regular expression. The format is a comma-separated string of `pattern=lr`.
    *   Example: `--network_args "network_reg_lrs=down_blocks.1=1e-4,up_blocks.2=2e-4"`

**Notes:**

*   To find the correct module names for the patterns, you may need to inspect the model structure.
*   Settings via `network_reg_dims` and `network_reg_lrs` take precedence over the global `--network_dim` and `--learning_rate` settings.
*   If a module name matches multiple patterns, the setting from the last matching pattern in the string will be applied.

<details>
<summary>日本語</summary>

正規表現を用いて、LoRAのモジュールごとにランク（dim）や学習率を指定することができます。これにより、柔軟できめ細やかな制御が可能になります。

これらの設定は `network_args` 引数で指定します。

*   `network_reg_dims`: 正規表現にマッチするモジュールに対してランクを指定します。
*   `network_reg_lrs`: 正規表現にマッチするモジュールに対して学習率を指定します。

**注意点:**

*   パターンのための正確なモジュール名を見つけるには、モデルの構造を調べる必要があるかもしれません。
*   `network_reg_dims` および `network_reg_lrs` での設定は、全体設定である `--network_dim` や `--learning_rate` よりも優先されます。
*   あるモジュール名が複数のパターンにマッチした場合、文字列の中で後方にあるパターンの設定が適用されます。

</details>

### 6.4. Multi-Resolution Training / マルチ解像度トレーニング

You can define multiple resolutions in the dataset configuration file, with different batch sizes for each resolution.

**Note:** This feature is available, but it is **not recommended** as the HunyuanImage-2.1 base model was not trained with multi-resolution capabilities. Using it may lead to unexpected results.

Configuration file example:
```toml
[general]
shuffle_caption = true
caption_extension = ".txt"

[[datasets]]
batch_size = 2
enable_bucket = true
resolution = [1024, 1024]

  [[datasets.subsets]]
  image_dir = "path/to/image/directory"
  num_repeats = 1

[[datasets]]
batch_size = 1
enable_bucket = true
resolution = [1280, 768]

  [[datasets.subsets]]
  image_dir = "path/to/another/directory"
  num_repeats = 1
```

<details>
<summary>日本語</summary>

データセット設定ファイルで複数の解像度を定義できます。各解像度に対して異なるバッチサイズを指定することができます。

**注意:** この機能は利用可能ですが、HunyuanImage-2.1のベースモデルはマルチ解像度で学習されていないため、**非推奨**です。使用すると予期しない結果になる可能性があります。

設定ファイルの例は英語のドキュメントを参照してください。

</details>

### 6.5. Validation / 検証

You can calculate validation loss during training using a validation dataset to evaluate model generalization performance. This feature works the same as in other training scripts. For details, please refer to the [Validation Guide](validation.md).

<details>
<summary>日本語</summary>

学習中に検証データセットを使用して損失 (Validation Loss) を計算し、モデルの汎化性能を評価できます。この機能は他の学習スクリプトと同様に動作します。詳細は[検証ガイド](validation.md)を参照してください。

</details>

## 7. Other Training Options / その他の学習オプション

- **`--ip_noise_gamma`**: Use `--ip_noise_gamma` and `--ip_noise_gamma_random_strength` to adjust Input Perturbation noise gamma values during training. See Stable Diffusion 3 training options for details.

- **`--loss_type`**: Specifies the loss function for training. The default is `l2`.
  - `l1`: L1 loss.
  - `l2`: L2 loss (mean squared error).
  - `huber`: Huber loss.
  - `smooth_l1`: Smooth L1 loss.

- **`--huber_schedule`**, **`--huber_c`**, **`--huber_scale`**: These are parameters for Huber loss. They are used when `--loss_type` is `huber` or `smooth_l1`.

- **`--weighting_scheme`**, **`--logit_mean`**, **`--logit_std`**, **`--mode_scale`**: These options allow you to adjust the loss weighting for each timestep. For details, refer to the [`sd3_train_network.md` guide](sd3_train_network.md).

- **`--fused_backward_pass`**: Fuses the backward pass and optimizer step to reduce VRAM usage.

<details>
<summary>日本語</summary>

- **`--ip_noise_gamma`**: Input Perturbationノイズのガンマ値を調整します。
- **`--loss_type`**: 学習に用いる損失関数を指定します。
- **`--huber_schedule`**, **`--huber_c`**, **`--huber_scale`**: Huber損失のパラメータです。
- **`--weighting_scheme`**, **`--logit_mean`**, **`--logit_std`**, **`--mode_scale`**: 各タイムステップの損失の重み付けを調整します。
- **`--fused_backward_pass`**: バックワードパスとオプティマイザステップを融合してVRAM使用量を削減します。

</details>

## 8. Using the Inference Script / 推論スクリプトの使用法

The `hunyuan_image_minimal_inference.py` script allows you to generate images using trained LoRA models. Here's a basic usage example:

```bash
python hunyuan_image_minimal_inference.py \
  --dit "<path to hunyuanimage2.1.safetensors>" \
  --text_encoder "<path to qwen_2.5_vl_7b.safetensors>" \
  --byt5 "<path to byt5_small_glyphxl_fp16.safetensors>" \
  --vae "<path to hunyuan_image_2.1_vae_fp16.safetensors>" \
  --lora_weight "<path to your trained LoRA>" \
  --lora_multiplier 1.0 \
  --attn_mode "torch" \
  --prompt "A cute cartoon penguin in a snowy landscape" \
  --image_size 2048 2048 \
  --infer_steps 50 \
  --guidance_scale 3.5 \
  --flow_shift 5.0 \
  --seed 542017 \
  --save_path "output_image.png"
```

**Key Options:**
- `--fp8_scaled`: Use scaled FP8 format for reduced VRAM usage during inference
- `--blocks_to_swap`: Swap blocks to CPU to reduce VRAM usage
- `--image_size`: Resolution in **height width**  (inference is most stable at 2560x1536, 2304x1792, 2048x2048, 1792x2304, 1536x2560 according to the official repo)
- `--guidance_scale`: CFG scale (default: 3.5)
- `--flow_shift`: Flow matching shift parameter (default: 5.0)
- `--text_encoder_cpu`: Run the text encoders on CPU to reduce VRAM usage
- `--vae_chunk_size`: Chunk size for VAE decoding to reduce memory usage (default: None, no chunking). 16 is recommended if enabled.
- `--apg_start_step_general` and `--apg_start_step_ocr`: Start steps for APG (Adaptive Projected Guidance) if using APG during inference. `5` and `38` are the official recommended values for 50 steps. If this value exceeds `--infer_steps`, APG will not be applied.
- `--guidance_rescale`: Rescales the guidance for steps before APG starts. Default is `0.0` (no rescaling). If you use this option, a value around `0.5` might be good starting point.
- `--guidance_rescale_apg`: Rescales the guidance for APG. Default is `0.0` (no rescaling). This option doesn't seem to have a large effect, but if you use it, a value around `0.5` might be a good starting point.

`--split_attn` is not supported (since inference is done one at a time). `--fp8_vl` is not supported, please use CPU for the text encoder if VRAM is insufficient.

<details>
<summary>日本語</summary>

`hunyuan_image_minimal_inference.py`スクリプトを使用して、学習したLoRAモデルで画像を生成できます。基本的な使用例は英語のドキュメントを参照してください。

**主要なオプション:**
- `--fp8_scaled`: VRAM使用量削減のためのスケールFP8形式
- `--blocks_to_swap`: VRAM使用量削減のためのブロックスワップ
- `--image_size`: 解像度（2048x2048で最も安定）
- `--guidance_scale`: CFGスケール（推奨: 3.5）
- `--flow_shift`: Flow Matchingシフトパラメータ（デフォルト: 5.0）
- `--text_encoder_cpu`: テキストエンコーダをCPUで実行してVRAM使用量削減
- `--vae_chunk_size`: VAEデコーディングのチャンクサイズ（デフォルト: None、チャンク処理なし）。有効にする場合は16を推奨。
- `--apg_start_step_general` と `--apg_start_step_ocr`: 推論中にAPGを使用する場合の開始ステップ。50ステップの場合、公式推奨値はそれぞれ5と38です。この値が`--infer_steps`を超えると、APGは適用されません。
- `--guidance_rescale`: APG開始前のステップに対するガイダンスのリスケーリング。デフォルトは0.0（リスケーリングなし）。使用する場合、0.5程度から始めて調整してください。
- `--guidance_rescale_apg`: APGに対するガイダンスのリスケーリング。デフォルトは0.0（リスケーリングなし）。このオプションは大きな効果はないようですが、使用する場合は0.5程度から始めて調整してください。

`--split_attn`はサポートされていません（1件ずつ推論するため）。`--fp8_vl`もサポートされていません。VRAMが不足する場合はテキストエンコーダをCPUで実行してください。

</details>

## 9. Related Tools / 関連ツール

### `networks/convert_hunyuan_image_lora_to_comfy.py`

A script to convert LoRA models to ComfyUI-compatible format. The formats differ slightly, so conversion is necessary. You can convert from the sd-scripts format to ComfyUI format with:

```bash
python networks/convert_hunyuan_image_lora_to_comfy.py path/to/source.safetensors path/to/destination.safetensors
```

Using the `--reverse` option allows conversion in the opposite direction (ComfyUI format to sd-scripts format). However, reverse conversion is only possible for LoRAs converted by this script. LoRAs created with other training tools cannot be converted.

<details>
<summary>日本語</summary>

**`networks/convert_hunyuan_image_lora_to_comfy.py`**

LoRAモデルをComfyUI互換形式に変換するスクリプト。わずかに形式が異なるため、変換が必要です。以下の指定で、sd-scriptsの形式からComfyUI形式に変換できます。

```bash
python networks/convert_hunyuan_image_lora_to_comfy.py path/to/source.safetensors path/to/destination.safetensors
```

`--reverse`オプションを付けると、逆変換（ComfyUI形式からsd-scripts形式）も可能です。ただし、逆変換ができるのはこのスクリプトで変換したLoRAに限ります。他の学習ツールで作成したLoRAは変換できません。

</details>

## 10. Others / その他

`hunyuan_image_train_network.py` includes many features common with `train_network.py`, such as sample image generation (`--sample_prompts`, etc.) and detailed optimizer settings. For these features, refer to the [`train_network.py` guide](train_network.md#5-other-features--その他の機能) or the script help (`python hunyuan_image_train_network.py --help`).

<details>
<summary>日本語</summary>

`hunyuan_image_train_network.py`には、サンプル画像の生成 (`--sample_prompts`など) や詳細なオプティマイザ設定など、`train_network.py`と共通の機能も多く存在します。これらについては、[`train_network.py`のガイド](train_network.md#5-other-features--その他の機能)やスクリプトのヘルプ (`python hunyuan_image_train_network.py --help`) を参照してください。

</details>
