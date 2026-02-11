Status: reviewed

# LoRA Training Guide for FLUX.1 using `flux_train_network.py` / `flux_train_network.py` を用いたFLUX.1モデルのLoRA学習ガイド

This document explains how to train LoRA models for the FLUX.1 model using `flux_train_network.py` included in the `sd-scripts` repository.

<details>
<summary>日本語</summary>

このドキュメントでは、`sd-scripts`リポジトリに含まれる`flux_train_network.py`を使用して、FLUX.1モデルに対するLoRA (Low-Rank Adaptation) モデルを学習する基本的な手順について解説します。

</details>

## 1. Introduction / はじめに

`flux_train_network.py` trains additional networks such as LoRA on the FLUX.1 model, which uses a transformer-based architecture different from Stable Diffusion. Two text encoders, CLIP-L and T5-XXL, and a dedicated AutoEncoder are used.

This guide assumes you know the basics of LoRA training. For common options see [train_network.py](train_network.md) and [sdxl_train_network.py](sdxl_train_network.md).

**Prerequisites:**

* The repository is cloned and the Python environment is ready.
* A training dataset is prepared. See the dataset configuration guide.

<details>
<summary>日本語</summary>

`flux_train_network.py`は、FLUX.1モデルに対してLoRAなどの追加ネットワークを学習させるためのスクリプトです。FLUX.1はStable Diffusionとは異なるアーキテクチャを持つ画像生成モデルであり、このスクリプトを使用することで、特定のキャラクターや画風を再現するLoRAモデルを作成できます。

このガイドは、基本的なLoRA学習の手順を理解しているユーザーを対象としています。基本的な使い方や共通のオプションについては、[`train_network.py`のガイド](train_network.md)を参照してください。また一部のパラメータは [`sdxl_train_network.py`](sdxl_train_network.md) と同様のものがあるため、そちらも参考にしてください。

**前提条件:**

* `sd-scripts`リポジトリのクローンとPython環境のセットアップが完了していること。
* 学習用データセットの準備が完了していること。（データセットの準備については[データセット設定ガイド](link/to/dataset/config/doc)を参照してください）

</details>

## 2. Differences from `train_network.py` / `train_network.py` との違い

`flux_train_network.py` is based on `train_network.py` but adapted for FLUX.1. Main differences include:

* **Target model:** FLUX.1 model (dev or schnell version).
* **Model structure:** Unlike Stable Diffusion, FLUX.1 uses a Transformer-based architecture with two text encoders (CLIP-L and T5-XXL) and a dedicated AutoEncoder (AE) instead of VAE.
* **Required arguments:** Additional arguments for FLUX.1 model, CLIP-L, T5-XXL, and AE model files.
* **Incompatible options:** Some Stable Diffusion-specific arguments (e.g., `--v2`, `--clip_skip`, `--max_token_length`) are not used in FLUX.1 training.
* **FLUX.1-specific arguments:** Additional arguments for FLUX.1-specific training parameters like timestep sampling and guidance scale.

<details>
<summary>日本語</summary>

`flux_train_network.py`は`train_network.py`をベースに、FLUX.1モデルに対応するための変更が加えられています。主な違いは以下の通りです。

* **対象モデル:** FLUX.1モデル（dev版またはschnell版）を対象とします。
* **モデル構造:** Stable Diffusionとは異なり、FLUX.1はTransformerベースのアーキテクチャを持ちます。Text EncoderとしてCLIP-LとT5-XXLの二つを使用し、VAEの代わりに専用のAutoEncoder (AE) を使用します。
* **必須の引数:** FLUX.1モデル、CLIP-L、T5-XXL、AEの各モデルファイルを指定する引数が追加されています。
* **一部引数の非互換性:** Stable Diffusion向けの引数の一部（例: `--v2`, `--clip_skip`, `--max_token_length`）はFLUX.1の学習では使用されません。
* **FLUX.1特有の引数:** タイムステップのサンプリング方法やガイダンススケールなど、FLUX.1特有の学習パラメータを指定する引数が追加されています。

</details>

## 3. Preparation / 準備

Before starting training you need:

1. **Training script:** `flux_train_network.py`
2. **FLUX.1 model file:** Base FLUX.1 model `.safetensors` file (e.g., `flux1-dev.safetensors`).
3. **Text Encoder model files:**
   - CLIP-L model `.safetensors` file (e.g., `clip_l.safetensors`)
   - T5-XXL model `.safetensors` file (e.g., `t5xxl.safetensors`)
4. **AutoEncoder model file:** FLUX.1-compatible AE model `.safetensors` file (e.g., `ae.safetensors`).
5. **Dataset definition file (.toml):** TOML format file describing training dataset configuration (e.g., `my_flux_dataset_config.toml`).

### Downloading Required Models

To train FLUX.1 models, you need to download the following model files:

- **DiT, AE**: Download from the [black-forest-labs/FLUX.1 dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) repository. Use `flux1-dev.safetensors` and `ae.safetensors`. The weights in the subfolder are in Diffusers format and cannot be used.
- **Text Encoder 1 (T5-XXL), Text Encoder 2 (CLIP-L)**: Download from the [ComfyUI FLUX Text Encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) repository. Please use `t5xxl_fp16.safetensors` for T5-XXL. Thanks to ComfyUI for providing these models.

To train Chroma models, you need to download the Chroma model file from the following repository:

- **Chroma Base**: Download from the [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) repository. Use `Chroma.safetensors`.

We have tested Chroma training with the weights from the [lodestones/Chroma](https://huggingface.co/lodestones/Chroma) repository. 

AE and T5-XXL models are same as FLUX.1, so you can use the same files. CLIP-L model is not used for Chroma training, so you can omit the `--clip_l` argument.

<details>
<summary>日本語</summary>

学習を開始する前に、以下のファイルが必要です。

1. **学習スクリプト:** `flux_train_network.py`
2. **FLUX.1モデルファイル:** 学習のベースとなるFLUX.1モデルの`.safetensors`ファイル（例: `flux1-dev.safetensors`）。
3. **Text Encoderモデルファイル:**
   - CLIP-Lモデルの`.safetensors`ファイル。例として`clip_l.safetensors`を使用します。
   - T5-XXLモデルの`.safetensors`ファイル。例として`t5xxl.safetensors`を使用します。
4. **AutoEncoderモデルファイル:** FLUX.1に対応するAEモデルの`.safetensors`ファイル。例として`ae.safetensors`を使用します。
5. **データセット定義ファイル (.toml):** 学習データセットの設定を記述したTOML形式のファイル。（詳細は[データセット設定ガイド](link/to/dataset/config/doc)を参照してください）。例として`my_flux_dataset_config.toml`を使用します。

**必要なモデルのダウンロード**

FLUX.1モデルを学習するためには、以下のモデルファイルをダウンロードする必要があります。

- **DiT, AE**: [black-forest-labs/FLUX.1 dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) リポジトリからダウンロードします。`flux1-dev.safetensors`と`ae.safetensors`を使用してください。サブフォルダ内の重みはDiffusers形式であり、使用できません。
- **Text Encoder 1 (T5-XXL), Text Encoder 2 (CLIP-L)**: [ComfyUI FLUX Text Encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) リポジトリからダウンロードします。T5-XXLには`t5xxl_fp16.safetensors`を使用してください。これらのモデルを提供いただいたComfyUIに感謝します。

Chromaモデルを学習する場合は、以下のリポジトリからChromaモデルファイルをダウンロードする必要があります。

- **Chroma Base**: [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) リポジトリからダウンロードします。`Chroma.safetensors`を使用してください。

Chromaの学習のテストは [lodestones/Chroma](https://huggingface.co/lodestones/Chroma) リポジトリの重みを使用して行いました。

AEとT5-XXLモデルはFLUX.1と同じものを使用できるため、同じファイルを使用します。CLIP-LモデルはChroma学習では使用されないため、`--clip_l`引数は省略できます。

</details>

## 4. Running the Training / 学習の実行

Run `flux_train_network.py` from the terminal with FLUX.1 specific arguments. Here's a basic command example:

```bash
accelerate launch --num_cpu_threads_per_process 1 flux_train_network.py \
  --pretrained_model_name_or_path="<path to FLUX.1 model>" \
  --clip_l="<path to CLIP-L model>" \
  --t5xxl="<path to T5-XXL model>" \
  --ae="<path to AE model>" \
  --dataset_config="my_flux_dataset_config.toml" \
  --output_dir="<output directory>" \
  --output_name="my_flux_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora_flux \
  --network_dim=16 \
  --network_alpha=1 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW8bit" \
  --lr_scheduler="constant" \
  --sdpa \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --guidance_scale=1.0 \
  --timestep_sampling="flux_shift" \
  --model_prediction_type="raw" \
  --blocks_to_swap=18 \
  --cache_text_encoder_outputs \
  --cache_latents
```

### Training Chroma Models

If you want to train a Chroma model, specify `--model_type=chroma`. Chroma does not use CLIP-L, so the `--clip_l` argument is not needed. T5XXL and AE are same as FLUX.1. The command would look like this:

```bash
accelerate launch --num_cpu_threads_per_process 1 flux_train_network.py \
  --pretrained_model_name_or_path="<path to Chroma model>" \
  --model_type=chroma \
  --t5xxl="<path to T5-XXL model>" \
  --ae="<path to AE model>" \
  --dataset_config="my_flux_dataset_config.toml" \
  --output_dir="<output directory>" \
  --output_name="my_chroma_lora" \
  --guidance_scale=0.0 \
  --timestep_sampling="sigmoid" \
  --apply_t5_attn_mask \
  ...
```

Note that for Chroma models, `--guidance_scale=0.0` is required to disable guidance scale, and `--apply_t5_attn_mask` is needed to apply attention masks for T5XXL Text Encoder.

The sample image generation during training requires specifying a negative prompt. Also, set `--g 0` to disable embedded guidance scale and `--l 4.0` to set the CFG scale. For example:

```
Japanese shrine in the summer forest. --n low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors --w 512 --h 512 --d 1 --l 4.0 --g 0.0 --s 20
```

<details>
<summary>日本語</summary>

学習は、ターミナルから`flux_train_network.py`を実行することで開始します。基本的なコマンドラインの構造は`train_network.py`と同様ですが、FLUX.1特有の引数を指定する必要があります。

コマンドラインの例は英語のドキュメントを参照してください。

#### Chromaモデルの学習

Chromaモデルを学習したい場合は、`--model_type=chroma`を指定します。ChromaはCLIP-Lを使用しないため、`--clip_l`引数は不要です。T5XXLとAEはFLUX.1と同様です。

コマンドラインの例は英語のドキュメントを参照してください。

学習中のサンプル画像生成には、ネガティブプロンプトを指定してください。また `--g 0` を指定して埋め込みガイダンススケールを無効化し、`--l 4.0` を指定してCFGスケールを設定します。

</details>

### 4.1. Explanation of Key Options / 主要なコマンドライン引数の解説

The script adds FLUX.1 specific arguments. For common arguments (like `--output_dir`, `--output_name`, `--network_module`, etc.), see the [`train_network.py` guide](train_network.md).

#### Model-related [Required]

* `--pretrained_model_name_or_path="<path to FLUX.1/Chroma model>"` **[Required]**
  - Specifies the path to the base FLUX.1 or Chroma model `.safetensors` file. Diffusers format directories are not currently supported.
* `--model_type=<model type>`
  - Specifies the type of base model for training. Choose from `flux` or `chroma`. Default is `flux`.
* `--clip_l="<path to CLIP-L model>"` **[Required when flux is selected]**
  - Specifies the path to the CLIP-L Text Encoder model `.safetensors` file. Not needed when `--model_type=chroma`.
* `--t5xxl="<path to T5-XXL model>"` **[Required]**
  - Specifies the path to the T5-XXL Text Encoder model `.safetensors` file.
* `--ae="<path to AE model>"` **[Required]**
  - Specifies the path to the FLUX.1-compatible AutoEncoder model `.safetensors` file.

#### FLUX.1 Training Parameters

* `--guidance_scale=<float>`
  - FLUX.1 dev version is distilled with specific guidance scale values, but for training, specify `1.0` to disable guidance scale. Default is `3.5`, so be sure to specify this. Usually ignored for schnell version.
  - Chroma requires `--guidance_scale=0.0` to disable guidance scale.
* `--timestep_sampling=<choice>`
  - Specifies the sampling method for timesteps (noise levels) during training. Choose from `sigma`, `uniform`, `sigmoid`, `shift`, `flux_shift`. Default is `sigma`. Recommended is `flux_shift`. For Chroma models, `sigmoid` is recommended.
* `--sigmoid_scale=<float>`
  - Scale factor when `timestep_sampling` is set to `sigmoid`, `shift`, or `flux_shift`. Default and recommended value is `1.0`.
* `--model_prediction_type=<choice>`
  - Specifies what the model predicts. Choose from `raw` (use prediction as-is), `additive` (add to noise input), `sigma_scaled` (apply sigma scaling). Default is `sigma_scaled`. Recommended is `raw`.
* `--discrete_flow_shift=<float>`
  - Specifies the shift value for the scheduler used in Flow Matching. Default is `3.0`. This value is ignored when `timestep_sampling` is set to other than `shift`.

#### Memory/Speed Related

* `--fp8_base` 
  - Enables training in FP8 format for FLUX.1, CLIP-L, and T5-XXL. This can significantly reduce VRAM usage, but the training results may vary. 
* `--blocks_to_swap=<integer>` **[Experimental Feature]**
  - Setting to reduce VRAM usage by swapping parts of the model (Transformer blocks) between CPU and GPU. Specify the number of blocks to swap as an integer (e.g., `18`). Larger values reduce VRAM usage but decrease training speed. Adjust according to your GPU's VRAM capacity. Can be used with `gradient_checkpointing`.
  - Cannot be used with `--cpu_offload_checkpointing`.
* `--cache_text_encoder_outputs`
  - Caches the outputs of CLIP-L and T5-XXL. This reduces memory usage.
* `--cache_latents`, `--cache_latents_to_disk`
  - Caches the outputs of AE. Similar functionality to [sdxl_train_network.py](sdxl_train_network.md).

#### Incompatible/Deprecated Arguments

* `--v2`, `--v_parameterization`, `--clip_skip`: These are Stable Diffusion-specific arguments and are not used in FLUX.1 training.
* `--max_token_length`: This is an argument for Stable Diffusion v1/v2. For FLUX.1, use `--t5xxl_max_token_length`.
* `--split_mode`: Deprecated argument. Use `--blocks_to_swap` instead.

<details>
<summary>日本語</summary>

[`train_network.py`のガイド](train_network.md)で説明されている引数に加え、以下のFLUX.1特有の引数を指定します。共通の引数（`--output_dir`, `--output_name`, `--network_module`, `--network_dim`, `--network_alpha`, `--learning_rate`など）については、上記ガイドを参照してください。

コマンドラインの例と詳細な引数の説明は英語のドキュメントを参照してください。

</details>

### 4.2. Starting Training / 学習の開始

Training begins once you run the command with the required options. Log checking is the same as in [`train_network.py`](train_network.md#32-starting-the-training--学習の開始).

<details>
<summary>日本語</summary>

必要な引数を設定し、コマンドを実行すると学習が開始されます。基本的な流れやログの確認方法は[`train_network.py`のガイド](train_network.md#32-starting-the-training--学習の開始)と同様です。

</details>

## 5. Using the Trained Model / 学習済みモデルの利用

After training, a LoRA model file is saved in `output_dir` and can be used in inference environments supporting FLUX.1 (e.g. ComfyUI + Flux nodes).

<details>
<summary>日本語</summary>

学習が完了すると、指定した`output_dir`にLoRAモデルファイル（例: `my_flux_lora.safetensors`）が保存されます。このファイルは、FLUX.1モデルに対応した推論環境（例: ComfyUI + ComfyUI-FluxNodes）で使用できます。

</details>

## 6. Advanced Settings / 高度な設定

### 6.1. VRAM Usage Optimization / VRAM使用量の最適化

FLUX.1 is a relatively large model, so GPUs without sufficient VRAM require optimization. Here are settings to reduce VRAM usage (with `--fp8_base`):

#### Recommended Settings by GPU Memory

| GPU Memory | Recommended Settings |
|------------|---------------------|
| 24GB VRAM | Basic settings work fine (batch size 2) |
| 16GB VRAM | Set batch size to 1 and use `--blocks_to_swap` |
| 12GB VRAM | Use `--blocks_to_swap 16` and 8bit AdamW |
| 10GB VRAM | Use `--blocks_to_swap 22`, recommend fp8 format for T5XXL |
| 8GB VRAM | Use `--blocks_to_swap 28`, recommend fp8 format for T5XXL |

#### Key VRAM Reduction Options

- **`--fp8_base`**: Enables training in FP8 format.

- **`--blocks_to_swap <number>`**: Swaps blocks between CPU and GPU to reduce VRAM usage. Higher numbers save more VRAM but reduce training speed. FLUX.1 supports up to 35 blocks for swapping.

- **`--cpu_offload_checkpointing`**: Offloads gradient checkpoints to CPU. Can reduce VRAM usage by up to 1GB but decreases training speed by about 15%. Cannot be used with `--blocks_to_swap`. Chroma models do not support this option.

- **Using Adafactor optimizer**: Can reduce VRAM usage more than 8bit AdamW:
  ```
  --optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --lr_scheduler constant_with_warmup --max_grad_norm 0.0
  ```

- **Using T5XXL fp8 format**: For GPUs with less than 10GB VRAM, using fp8 format T5XXL checkpoints is recommended. Download `t5xxl_fp8_e4m3fn.safetensors` from [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) (use without `scaled`).

- **FP8/FP16 Mixed Training [Experimental]**: Specify `--fp8_base_unet` to train the FLUX.1 model in FP8 format while training Text Encoders (CLIP-L/T5XXL) in BF16/FP16 format. This can further reduce VRAM usage.

<details>
<summary>日本語</summary>

FLUX.1モデルは比較的大きなモデルであるため、十分なVRAMを持たないGPUでは工夫が必要です。VRAM使用量を削減するための設定の詳細は英語のドキュメントを参照してください。

主要なVRAM削減オプション：
- `--fp8_base`: FP8形式での学習を有効化
- `--blocks_to_swap`: CPUとGPU間でブロックをスワップ
- `--cpu_offload_checkpointing`: 勾配チェックポイントをCPUにオフロード  
- Adafactorオプティマイザの使用
- T5XXLのfp8形式の使用
- FP8/FP16混合学習（実験的機能）

</details>

### 6.2. Important FLUX.1 LoRA Training Settings / FLUX.1 LoRA学習の重要な設定

FLUX.1 training has many unknowns, and several settings can be specified with arguments:

#### Timestep Sampling Methods

The `--timestep_sampling` option specifies how timesteps (0-1) are sampled:

- `sigma`: Sigma-based like SD3
- `uniform`: Uniform random
- `sigmoid`: Sigmoid of normal distribution random (similar to x-flux, AI-toolkit)
- `shift`: Sigmoid value of normal distribution random with shift. The `--discrete_flow_shift` setting is used to shift the sigmoid value.
- `flux_shift`: Shift sigmoid value of normal distribution random according to resolution (similar to FLUX.1 dev inference).

`--discrete_flow_shift` only applies when `--timestep_sampling` is set to `shift`.

#### Model Prediction Processing

The `--model_prediction_type` option specifies how to interpret and process model predictions:

- `raw`: Use as-is (similar to x-flux) **[Recommended]**
- `additive`: Add to noise input
- `sigma_scaled`: Apply sigma scaling (similar to SD3)

#### Recommended Settings

Based on experiments, the following settings work well:
```
--timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0
```

For Chroma models, the following settings are recommended:
```
--timestep_sampling sigmoid --model_prediction_type raw --guidance_scale 0.0
```

**About Guidance Scale**: FLUX.1 dev version is distilled with specific guidance scale values, but for training, specify `--guidance_scale 1.0` to disable guidance scale. Chroma requires `--guidance_scale 0.0` to disable guidance scale because it is not distilled.

<details>
<summary>日本語</summary>

FLUX.1の学習には多くの未知の点があり、いくつかの設定は引数で指定できます。詳細な説明とコマンドラインの例は英語のドキュメントを参照してください。

主要な設定オプション：
- タイムステップのサンプリング方法（`--timestep_sampling`）
- モデル予測の処理方法（`--model_prediction_type`）
- 推奨設定の組み合わせ

</details>

### 6.3. Layer-specific Rank Configuration / 各層に対するランク指定

You can specify different ranks (network_dim) for each layer of FLUX.1. This allows you to emphasize or disable LoRA effects for specific layers.

Specify the following network_args to set ranks for each layer. Setting 0 disables LoRA for that layer:

| network_args | Target Layer |
|--------------|--------------|
| img_attn_dim | DoubleStreamBlock img_attn |
| txt_attn_dim | DoubleStreamBlock txt_attn |
| img_mlp_dim | DoubleStreamBlock img_mlp |
| txt_mlp_dim | DoubleStreamBlock txt_mlp |
| img_mod_dim | DoubleStreamBlock img_mod |
| txt_mod_dim | DoubleStreamBlock txt_mod |
| single_dim | SingleStreamBlock linear1 and linear2 |
| single_mod_dim | SingleStreamBlock modulation |

Example usage:
```
--network_args "img_attn_dim=4" "img_mlp_dim=8" "txt_attn_dim=2" "txt_mlp_dim=2" "img_mod_dim=2" "txt_mod_dim=2" "single_dim=4" "single_mod_dim=2"
```

To apply LoRA to FLUX conditioning layers, specify `in_dims` in network_args as a comma-separated list of 5 numbers:

```
--network_args "in_dims=[4,2,2,2,4]"
```

Each number corresponds to `img_in`, `time_in`, `vector_in`, `guidance_in`, `txt_in`. The example above applies LoRA to all conditioning layers with ranks of 4 for `img_in` and `txt_in`, and ranks of 2 for others.

<details>
<summary>日本語</summary>

FLUX.1の各層に対して異なるランク（network_dim）を指定できます。これにより、特定の層に対してLoRAの効果を強調したり、無効化したりできます。

詳細な設定方法とコマンドラインの例は英語のドキュメントを参照してください。

</details>

### 6.4. Block Selection for Training / 学習するブロックの指定

You can specify which blocks to train using `train_double_block_indices` and `train_single_block_indices` in network_args. Indices are 0-based. Default is to train all blocks if omitted.

Specify indices as integer lists like `0,1,5,8` or integer ranges like `0,1,4-5,7`:
- Double blocks: 19 blocks, valid range 0-18
- Single blocks: 38 blocks, valid range 0-37
- Specify `all` to train all blocks
- Specify `none` to skip training blocks

Example usage:
```
--network_args "train_double_block_indices=0,1,8-12,18" "train_single_block_indices=3,10,20-25,37"
```

Or:
```
--network_args "train_double_block_indices=none" "train_single_block_indices=10-15"
```

<details>
<summary>日本語</summary>

FLUX.1 LoRA学習では、network_argsの`train_double_block_indices`と`train_single_block_indices`を指定することで、学習するブロックを指定できます。

詳細な設定方法とコマンドラインの例は英語のドキュメントを参照してください。

</details>

### 6.5. Regular Expression-based Rank/LR Configuration / 正規表現によるランク・学習率の指定

You can specify ranks (dims) and learning rates for LoRA modules using regular expressions. This allows for more flexible and fine-grained control than specifying by layer.

These settings are specified via the `network_args` argument.

*   `network_reg_dims`: Specify ranks for modules matching a regular expression. The format is a comma-separated string of `pattern=rank`.
    *   Example: `--network_args "network_reg_dims=single.*_modulation.*=4,img_attn=8"`
    *   This sets the rank to 4 for modules whose names contain `single` and contain `_modulation`, and to 8 for modules containing `img_attn`.
*   `network_reg_lrs`: Specify learning rates for modules matching a regular expression. The format is a comma-separated string of `pattern=lr`.
    *   Example: `--network_args "network_reg_lrs=single_blocks_(\d|10)_=1e-3,double_blocks=2e-3"`
    *   This sets the learning rate to `1e-3` for modules whose names contain `single_blocks` followed by a digit (`0` to `9`) or `10`, and to `2e-3` for modules whose names contain `double_blocks`.

**Notes:**

*   Settings via `network_reg_dims` and `network_reg_lrs` take precedence over the global `--network_dim` and `--learning_rate` settings.
*   If a module name matches multiple patterns, the setting from the last matching pattern in the string will be applied.
*   These settings are applied after the block-specific training settings (`train_double_block_indices`, `train_single_block_indices`).

<details>
<summary>日本語</summary>

正規表現を用いて、LoRAのモジュールごとにランク（dim）や学習率を指定することができます。これにより、層ごとの指定よりも柔軟できめ細やかな制御が可能になります。

これらの設定は `network_args` 引数で指定します。

*   `network_reg_dims`: 正規表現にマッチするモジュールに対してランクを指定します。`pattern=rank` という形式の文字列をカンマで区切って指定します。
    *   例: `--network_args "network_reg_dims=single.*_modulation.*=4,img_attn=8"`
    *   この例では、名前に `single` で始まり `_modulation` を含むモジュールのランクを4に、`img_attn` を含むモジュールのランクを8に設定します。
*   `network_reg_lrs`: 正規表現にマッチするモジュールに対して学習率を指定します。`pattern=lr` という形式の文字列をカンマで区切って指定します。
    *   例: `--network_args "network_reg_lrs=single_blocks_(\d|10)_=1e-3,double_blocks=2e-3"`
    *   この例では、名前が `single_blocks` で始まり、後に数字（`0`から`9`）または`10`が続くモジュールの学習率を `1e-3` に、`double_blocks` を含むモジュールの学習率を `2e-3` に設定します。
**注意点:**

*   `network_reg_dims` および `network_reg_lrs` での設定は、全体設定である `--network_dim` や `--learning_rate` よりも優先されます。
*   あるモジュール名が複数のパターンにマッチした場合、文字列の中で後方にあるパターンの設定が適用されます。
*   これらの設定は、ブロック指定（`train_double_block_indices`, `train_single_block_indices`）が適用された後に行われます。

</details>

### 6.6. Text Encoder LoRA Support / Text Encoder LoRAのサポート

FLUX.1 LoRA training supports training CLIP-L and T5XXL LoRA:

- To train only FLUX.1: specify `--network_train_unet_only`
- To train FLUX.1 and CLIP-L: omit `--network_train_unet_only`
- To train FLUX.1, CLIP-L, and T5XXL: omit `--network_train_unet_only` and add `--network_args "train_t5xxl=True"`

You can specify individual learning rates for CLIP-L and T5XXL with `--text_encoder_lr`. For example, `--text_encoder_lr 1e-4 1e-5` sets the first value for CLIP-L and the second for T5XXL. Specifying one value uses the same learning rate for both. If `--text_encoder_lr` is not specified, the default `--learning_rate` is used for both.

<details>
<summary>日本語</summary>

FLUX.1 LoRA学習は、CLIP-LとT5XXL LoRAのトレーニングもサポートしています。

詳細な設定方法とコマンドラインの例は英語のドキュメントを参照してください。

</details>

### 6.7. Multi-Resolution Training / マルチ解像度トレーニング

You can define multiple resolutions in the dataset configuration file, with different batch sizes for each resolution.

Configuration file example:
```toml
[general]
# Common settings
flip_aug = true
color_aug = false
keep_tokens_separator= "|||"
shuffle_caption = false
caption_tag_dropout_rate = 0
caption_extension = ".txt"

[[datasets]]
# First resolution settings
batch_size = 2
enable_bucket = true
resolution = [1024, 1024]

  [[datasets.subsets]]
  image_dir = "path/to/image/directory"
  num_repeats = 1

[[datasets]]
# Second resolution settings
batch_size = 3
enable_bucket = true
resolution = [768, 768]

  [[datasets.subsets]]
  image_dir = "path/to/image/directory"
  num_repeats = 1
```

<details>
<summary>日本語</summary>

データセット設定ファイルで複数の解像度を定義できます。各解像度に対して異なるバッチサイズを指定することができます。

設定ファイルの例は英語のドキュメントを参照してください。

</details>

### 6.8. Validation / 検証

You can calculate validation loss during training using a validation dataset to evaluate model generalization performance.

To set up validation, add a `validation_split` and optionally `validation_seed` to your dataset configuration TOML file. 

```toml
validation_seed = 42 # [Optional] Validation seed, otherwise uses training seed for validation split .
enable_bucket = true
resolution = [1024, 1024]

[[datasets]]
  [[datasets.subsets]]
  # This directory will use 100% of the images for training
  image_dir = "path/to/image/directory"

[[datasets]]
validation_split = 0.1 # Split between 0.0 and 1.0 where 1.0 will use the full subset as a validation dataset

  [[datasets.subsets]]
  # This directory will split 10% to validation and 90% to training
  image_dir = "path/to/image/second-directory"

[[datasets]]
validation_split = 1.0 # Will use this full subset as a validation subset. 

  [[datasets.subsets]]
  # This directory will use the 100% to validation and 0% to training
  image_dir = "path/to/image/full_validation"
```

**Notes:**

* Validation loss calculation uses fixed timestep sampling and random seeds to reduce loss variation due to randomness for more stable evaluation.
* Currently, validation loss is not supported when using Schedule-Free optimizers (`AdamWScheduleFree`, `RAdamScheduleFree`, `ProdigyScheduleFree`).

<details>
<summary>日本語</summary>

学習中に検証データセットを使用して損失 (Validation Loss) を計算し、モデルの汎化性能を評価できます。

詳細な設定方法とコマンドラインの例は英語のドキュメントを参照してください。

</details>

## 7. Additional Options / 追加オプション

### 7.1. Other FLUX.1-specific Options / その他のFLUX.1特有のオプション

- **T5 Attention Mask Application**: Specify `--apply_t5_attn_mask` to apply attention masks during T5XXL Text Encoder training and inference. Not recommended due to limited inference environment support. **For Chroma models, this option is required.**

- **IP Noise Gamma**: Use `--ip_noise_gamma` and `--ip_noise_gamma_random_strength` to adjust Input Perturbation noise gamma values during training. See Stable Diffusion 3 training options for details.

- **LoRA-GGPO Support**: Use LoRA-GGPO (Gradient Group Proportion Optimizer) to stabilize LoRA training:
  ```bash
  --network_args "ggpo_sigma=0.03" "ggpo_beta=0.01"
  ```

- **Q/K/V Projection Layer Splitting [Experimental]**: Specify `--network_args "split_qkv=True"` to individually split and apply LoRA to Q/K/V (and SingleStreamBlock Text) projection layers within Attention layers.

<details>
<summary>日本語</summary>

その他のFLUX.1特有のオプション：
- T5 Attention Maskの適用（Chromaモデルでは必須）
- IPノイズガンマ
- LoRA-GGPOサポート
- Q/K/V射影層の分割（実験的機能）

詳細な設定方法とコマンドラインの例は英語のドキュメントを参照してください。

</details>

### 7.2. Dataset-related Additional Options / データセット関連の追加オプション

#### Interpolation Method for Resizing

You can specify the interpolation method when resizing dataset images to training resolution. Specify `interpolation_type` in the `[[datasets]]` or `[general]` section of the dataset configuration TOML file.

Available values: `bicubic` (default), `bilinear`, `lanczos`, `nearest`, `area`

```toml
[[datasets]]
resolution = [1024, 1024]
enable_bucket = true
interpolation_type = "lanczos" # Example: Use Lanczos interpolation
# ...
```

<details>
<summary>日本語</summary>

データセットの画像を学習解像度にリサイズする際の補間方法を指定できます。

設定方法とオプションの詳細は英語のドキュメントを参照してください。

</details>

### 7.3. Other Training Options / その他の学習オプション

- **`--controlnet_model_name_or_path`**: Specifies the path to a ControlNet model compatible with FLUX.1. This allows for training a LoRA that works in conjunction with ControlNet. This is an advanced feature and requires a compatible ControlNet model.

- **`--loss_type`**: Specifies the loss function for training. The default is `l2`.
  - `l1`: L1 loss.
  - `l2`: L2 loss (mean squared error).
  - `huber`: Huber loss.
  - `smooth_l1`: Smooth L1 loss.

- **`--huber_schedule`**, **`--huber_c`**, **`--huber_scale`**: These are parameters for Huber loss. They are used when `--loss_type` is set to `huber` or `smooth_l1`.

- **`--t5xxl_max_token_length`**: Specifies the maximum token length for the T5-XXL text encoder. For details, refer to the [`sd3_train_network.md` guide](sd3_train_network.md).

- **`--weighting_scheme`**, **`--logit_mean`**, **`--logit_std`**, **`--mode_scale`**: These options allow you to adjust the loss weighting for each timestep. For details, refer to the [`sd3_train_network.md` guide](sd3_train_network.md).

- **`--fused_backward_pass`**: Fuses the backward pass and optimizer step to reduce VRAM usage. For details, refer to the [`sdxl_train_network.md` guide](sdxl_train_network.md).

<details>
<summary>日本語</summary>

- **`--controlnet_model_name_or_path`**: FLUX.1互換のControlNetモデルへのパスを指定します。これにより、ControlNetと連携して動作するLoRAを学習できます。これは高度な機能であり、互換性のあるControlNetモデルが必要です。
- **`--loss_type`**: 学習に用いる損失関数を指定します。デフォルトは `l2` です。
  - `l1`: L1損失。
  - `l2`: L2損失（平均二乗誤差）。
  - `huber`: Huber損失。
  - `smooth_l1`: Smooth L1損失。
- **`--huber_schedule`**, **`--huber_c`**, **`--huber_scale`**: これらはHuber損失のパラメータです。`--loss_type` が `huber` または `smooth_l1` の場合に使用されます。
- **`--t5xxl_max_token_length`**: T5-XXLテキストエンコーダの最大トークン長を指定します。詳細は [`sd3_train_network.md` ガイド](sd3_train_network.md) を参照してください。
- **`--weighting_scheme`**, **`--logit_mean`**, **`--logit_std`**, **`--mode_scale`**: これらのオプションは、各タイムステップの損失の重み付けを調整するために使用されます。詳細は [`sd3_train_network.md` ガイド](sd3_train_network.md) を参照してください。
- **`--fused_backward_pass`**: バックワードパスとオプティマイザステップを融合してVRAM使用量を削減します。詳細は [`sdxl_train_network.md` ガイド](sdxl_train_network.md) を参照してください。

</details>

## 8. Related Tools / 関連ツール

Several related scripts are provided for models trained with `flux_train_network.py` and to assist with the training process:

* **`networks/flux_extract_lora.py`**: Extracts LoRA models from the difference between trained and base models.
* **`convert_flux_lora.py`**: Converts trained LoRA models to other formats like Diffusers (AI-Toolkit) format. When trained with Q/K/V split option, converting with this script can reduce model size.
* **`networks/flux_merge_lora.py`**: Merges trained LoRA models into FLUX.1 base models.
* **`flux_minimal_inference.py`**: Simple inference script for generating images with trained LoRA models. You can specify `flux` or `chroma` with the `--model_type` argument.

<details>
<summary>日本語</summary>

`flux_train_network.py` で学習したモデルや、学習プロセスに役立つ関連スクリプトが提供されています：

* **`networks/flux_extract_lora.py`**: 学習済みモデルとベースモデルの差分から LoRA モデルを抽出。
* **`convert_flux_lora.py`**: 学習した LoRA モデルを Diffusers (AI-Toolkit) 形式など他の形式に変換。
* **`networks/flux_merge_lora.py`**: 学習した LoRA モデルを FLUX.1 ベースモデルにマージ。
* **`flux_minimal_inference.py`**: 学習した LoRA モデルを適用して画像を生成するシンプルな推論スクリプト。
  `--model_type` 引数で `flux` または `chroma` を指定できます。

</details>

## 9. Others / その他

`flux_train_network.py` includes many features common with `train_network.py`, such as sample image generation (`--sample_prompts`, etc.) and detailed optimizer settings. For these features, refer to the [`train_network.py` guide](train_network.md#5-other-features--その他の機能) or the script help (`python flux_train_network.py --help`).

<details>
<summary>日本語</summary>

`flux_train_network.py`には、サンプル画像の生成 (`--sample_prompts`など) や詳細なオプティマイザ設定など、`train_network.py`と共通の機能も多く存在します。これらについては、[`train_network.py`のガイド](train_network.md#5-other-features--その他の機能)やスクリプトのヘルプ (`python flux_train_network.py --help`) を参照してください。

</details>
