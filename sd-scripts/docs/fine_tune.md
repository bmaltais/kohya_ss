# Fine-tuning Guide

This document explains how to perform fine-tuning on various model architectures using the `*_train.py` scripts.

<details>
<summary>日本語</summary>

# Fine-tuning ガイド

このドキュメントでは、`*_train.py` スクリプトを用いた、各種モデルアーキテクチャのFine-tuningの方法について解説します。

</details>

### Difference between Fine-tuning and LoRA tuning

This repository supports two methods for additional model training: **Fine-tuning** and **LoRA (Low-Rank Adaptation)**. Each method has distinct features and advantages.

**Fine-tuning** is a method that retrains all (or most) of the weights of a pre-trained model.
- **Pros**: It can improve the overall expressive power of the model and is suitable for learning styles or concepts that differ significantly from the original model.
- **Cons**:
    - It requires a large amount of VRAM and computational cost.
    - The saved file size is large (same as the original model).
    - It is prone to "overfitting," where the model loses the diversity of the original model if over-trained.
- **Corresponding scripts**: Scripts named `*_train.py`, such as `sdxl_train.py`, `sd3_train.py`, `flux_train.py`, and `lumina_train.py`.

**LoRA tuning** is a method that freezes the model's weights and only trains a small additional network called an "adapter."
- **Pros**:
    - It allows for fast training with low VRAM and computational cost.
    - It is considered resistant to overfitting because it trains fewer weights.
    - The saved file (LoRA network) is very small, ranging from tens to hundreds of MB, making it easy to manage.
    - Multiple LoRAs can be used in combination.
- **Cons**: Since it does not train the entire model, it may not achieve changes as significant as fine-tuning.
- **Corresponding scripts**: Scripts named `*_train_network.py`, such as `sdxl_train_network.py`, `sd3_train_network.py`, and `flux_train_network.py`.

| Feature | Fine-tuning | LoRA tuning |
|:---|:---|:---|
| **Training Target** | All model weights | Additional network (adapter) only |
| **VRAM/Compute Cost**| High | Low |
| **Training Time** | Long | Short |
| **File Size** | Large (several GB) | Small (few MB to hundreds of MB) |
| **Overfitting Risk** | High | Low |
| **Suitable Use Case** | Major style changes, concept learning | Adding specific characters or styles |

Generally, it is recommended to start with **LoRA tuning** if you want to add a specific character or style. **Fine-tuning** is a valid option for more fundamental style changes or aiming for a high-quality model.

<details>
<summary>日本語</summary>

### Fine-tuningとLoRA学習の違い

このリポジトリでは、モデルの追加学習手法として**Fine-tuning**と**LoRA (Low-Rank Adaptation)**学習の2種類をサポートしています。それぞれの手法には異なる特徴と利点があります。

**Fine-tuning**は、事前学習済みモデルの重み全体（または大部分）を再学習する手法です。
- **利点**: モデル全体の表現力を向上させることができ、元のモデルから大きく変化した画風やコンセプトの学習に適しています。
- **欠点**:
    - 学習には多くのVRAMと計算コストが必要です。
    - 保存されるファイルサイズが大きくなります（元のモデルと同じサイズ）。
    - 学習させすぎると、元のモデルが持っていた多様性が失われる「過学習（overfitting）」に陥りやすい傾向があります。
- **対応スクリプト**: `sdxl_train.py`, `sd3_train.py`, `flux_train.py`, `lumina_train.py` など、`*_train.py` という命名規則のスクリプトが対応します。

**LoRA学習**は、モデルの重みは凍結（固定）したまま、「アダプター」と呼ばれる小さな追加ネットワークのみを学習する手法です。
- **利点**:
    - 少ないVRAMと計算コストで高速に学習できます。
    - 学習する重みが少ないため、過学習に強いとされています。
    - 保存されるファイル（LoRAネットワーク）は数十〜数百MBと非常に小さく、管理が容易です。
    - 複数のLoRAを組み合わせて使用することも可能です。
- **欠点**: モデル全体を学習するわけではないため、Fine-tuningほどの大きな変化は期待できない場合があります。
- **対応スクリプト**: `sdxl_train_network.py`, `sd3_train_network.py`, `flux_train_network.py` など、`*_train_network.py` という命名規則のスクリプトが対応します。

| 特徴 | Fine-tuning | LoRA学習 |
|:---|:---|:---|
| **学習対象** | モデルの全重み | 追加ネットワーク（アダプター）のみ |
| **VRAM/計算コスト**| 大 | 小 |
| **学習時間** | 長 | 短 |
| **ファイルサイズ** | 大（数GB） | 小（数MB〜数百MB） |
| **過学習リスク** | 高 | 低 |
| **適した用途** | 大規模な画風変更、コンセプト学習 | 特定のキャラ、画風の追加学習 |

一般的に、特定のキャラクターや画風を追加したい場合は**LoRA学習**から試すことが推奨されます。より根本的な画風の変更や、高品質なモデルを目指す場合は**Fine-tuning**が有効な選択肢となります。

</details>

--- 

### Fine-tuning for each architecture

Fine-tuning updates the entire weights of the model, so it has different options and considerations than LoRA tuning. This section describes the fine-tuning scripts for major architectures.

The basic command structure is common to all architectures.

```bash
accelerate launch --mixed_precision bf16 {script_name}.py \
  --pretrained_model_name_or_path <path_to_model> \
  --dataset_config <path_to_config.toml> \
  --output_dir <output_directory> \
  --output_name <model_output_name> \
  --save_model_as safetensors \
  --max_train_steps 10000 \
  --learning_rate 1e-5 \
  --optimizer_type AdamW8bit
```

<details>
<summary>日本語</summary>

### 各アーキテクチャのFine-tuning

Fine-tuningはモデルの重み全体を更新するため、LoRA学習とは異なるオプションや考慮事項があります。ここでは主要なアーキテクチャごとのFine-tuningスクリプトについて説明します。

基本的なコマンドの構造は、どのアーキテクチャでも共通です。

```bash
accelerate launch --mixed_precision bf16 {script_name}.py \
  --pretrained_model_name_or_path <path_to_model> \
  --dataset_config <path_to_config.toml> \
  --output_dir <output_directory> \
  --output_name <model_output_name> \
  --save_model_as safetensors \
  --max_train_steps 10000 \
  --learning_rate 1e-5 \
  --optimizer_type AdamW8bit
```

</details>

#### SDXL (`sdxl_train.py`)

Performs fine-tuning for SDXL models. It is possible to train both the U-Net and the Text Encoders.

**Key Options:**

- `--train_text_encoder`: Includes the weights of the Text Encoders (CLIP ViT-L and OpenCLIP ViT-bigG) in the training. Effective for significant style changes or strongly learning specific concepts.
- `--learning_rate_te1`, `--learning_rate_te2`: Set individual learning rates for each Text Encoder.
- `--block_lr`: Divides the U-Net into 23 blocks and sets a different learning rate for each block. This allows for advanced adjustments, such as strengthening or weakening the learning of specific layers. (Not available in LoRA tuning).

**Command Example:**

```bash
accelerate launch --mixed_precision bf16 sdxl_train.py \
  --pretrained_model_name_or_path "sd_xl_base_1.0.safetensors" \
  --dataset_config "dataset_config.toml" \
  --output_dir "output" \
  --output_name "sdxl_finetuned" \
  --train_text_encoder \
  --learning_rate 1e-5 \
  --learning_rate_te1 5e-6 \
  --learning_rate_te2 2e-6
```

<details>
<summary>日本語</summary>

#### SDXL (`sdxl_train.py`)

SDXLモデルのFine-tuningを行います。U-NetとText Encoderの両方を学習させることが可能です。

**主要なオプション:**

- `--train_text_encoder`: Text Encoder（CLIP ViT-LとOpenCLIP ViT-bigG）の重みを学習対象に含めます。画風を大きく変えたい場合や、特定の概念を強く学習させたい場合に有効です。
- `--learning_rate_te1`, `--learning_rate_te2`: それぞれのText Encoderに個別の学習率を設定します。
- `--block_lr`: U-Netを23個のブロックに分割し、ブロックごとに異なる学習率を設定できます。特定の層の学習を強めたり弱めたりする高度な調整が可能です。（LoRA学習では利用できません）

**コマンド例:**

```bash
accelerate launch --mixed_precision bf16 sdxl_train.py \
  --pretrained_model_name_or_path "sd_xl_base_1.0.safetensors" \
  --dataset_config "dataset_config.toml" \
  --output_dir "output" \
  --output_name "sdxl_finetuned" \
  --train_text_encoder \
  --learning_rate 1e-5 \
  --learning_rate_te1 5e-6 \
  --learning_rate_te2 2e-6
```

</details>

#### SD3 (`sd3_train.py`)

Performs fine-tuning for Stable Diffusion 3 Medium models. SD3 consists of three Text Encoders (CLIP-L, CLIP-G, T5-XXL) and a MMDiT (equivalent to U-Net), which can be targeted for training.

**Key Options:**

- `--train_text_encoder`: Enables training for CLIP-L and CLIP-G.
- `--train_t5xxl`: Enables training for T5-XXL. T5-XXL is a very large model and requires a lot of VRAM for training.
- `--blocks_to_swap`: A memory optimization feature to reduce VRAM usage. It swaps some blocks of the MMDiT to CPU memory during training. Useful for using larger batch sizes in low VRAM environments. (Also available in LoRA tuning).
- `--num_last_block_to_freeze`: Freezes the weights of the last N blocks of the MMDiT, excluding them from training. Useful for maintaining model stability while focusing on learning in the lower layers.

**Command Example:**

```bash
accelerate launch --mixed_precision bf16 sd3_train.py \
  --pretrained_model_name_or_path "sd3_medium.safetensors" \
  --dataset_config "dataset_config.toml" \
  --output_dir "output" \
  --output_name "sd3_finetuned" \
  --train_text_encoder \
  --learning_rate 4e-6 \
  --blocks_to_swap 10
```

<details>
<summary>日本語</summary>

#### SD3 (`sd3_train.py`)

Stable Diffusion 3 MediumモデルのFine-tuningを行います。SD3は3つのText Encoder（CLIP-L, CLIP-G, T5-XXL）とMMDiT（U-Netに相当）で構成されており、これらを学習対象にできます。

**主要なオプション:**

- `--train_text_encoder`: CLIP-LとCLIP-Gの学習を有効にします。
- `--train_t5xxl`: T5-XXLの学習を有効にします。T5-XXLは非常に大きなモデルのため、学習には多くのVRAMが必要です。
- `--blocks_to_swap`: VRAM使用量を削減するためのメモリ最適化機能です。MMDiTの一部のブロックを学習中にCPUメモリに退避（スワップ）させます。VRAMが少ない環境で大きなバッチサイズを使いたい場合に有効です。（LoRA学習でも利用可能）
- `--num_last_block_to_freeze`: MMDiTの最後のNブロックの重みを凍結し、学習対象から除外します。モデルの安定性を保ちつつ、下位層を中心に学習させたい場合に有効です。

**コマンド例:**

```bash
accelerate launch --mixed_precision bf16 sd3_train.py \
  --pretrained_model_name_or_path "sd3_medium.safetensors" \
  --dataset_config "dataset_config.toml" \
  --output_dir "output" \
  --output_name "sd3_finetuned" \
  --train_text_encoder \
  --learning_rate 4e-6 \
  --blocks_to_swap 10
```

</details>

#### FLUX.1 (`flux_train.py`)

Performs fine-tuning for FLUX.1 models. FLUX.1 is internally composed of two Transformer blocks (Double Blocks, Single Blocks).

**Key Options:**

- `--blocks_to_swap`: Similar to SD3, this feature swaps Transformer blocks to the CPU for memory optimization.
- `--blockwise_fused_optimizers`: An experimental feature that aims to streamline training by applying individual optimizers to each block.

**Command Example:**

```bash
accelerate launch --mixed_precision bf16 flux_train.py \
  --pretrained_model_name_or_path "FLUX.1-dev.safetensors" \
  --dataset_config "dataset_config.toml" \
  --output_dir "output" \
  --output_name "flux1_finetuned" \
  --learning_rate 1e-5 \
  --blocks_to_swap 18
```

<details>
<summary>日本語</summary>

#### FLUX.1 (`flux_train.py`)

FLUX.1モデルのFine-tuningを行います。FLUX.1は内部的に2つのTransformerブロック（Double Blocks, Single Blocks）で構成されています。

**主要なオプション:**

- `--blocks_to_swap`: SD3と同様に、メモリ最適化のためにTransformerブロックをCPUにスワップする機能です。
- `--blockwise_fused_optimizers`: 実験的な機能で、各ブロックに個別のオプティマイザを適用し、学習を効率化することを目指します。

**コマンド例:**

```bash
accelerate launch --mixed_precision bf16 flux_train.py \
  --pretrained_model_name_or_path "FLUX.1-dev.safetensors" \
  --dataset_config "dataset_config.toml" \
  --output_dir "output" \
  --output_name "flux1_finetuned" \
  --learning_rate 1e-5 \
  --blocks_to_swap 18
```

</details>

#### Lumina (`lumina_train.py`)

Performs fine-tuning for Lumina-Next DiT models.

**Key Options:**

- `--use_flash_attn`: Enables Flash Attention to speed up computation.
- `lumina_train.py` is relatively new, and many of its options are shared with other scripts. Training can be performed following the basic command pattern.

**Command Example:**

```bash
accelerate launch --mixed_precision bf16 lumina_train.py \
  --pretrained_model_name_or_path "Lumina-Next-DiT-B.safetensors" \
  --dataset_config "dataset_config.toml" \
  --output_dir "output" \
  --output_name "lumina_finetuned" \
  --learning_rate 1e-5
```

<details>
<summary>日本語</summary>

#### Lumina (`lumina_train.py`)

Lumina-Next DiTモデルのFine-tuningを行います。

**主要なオプション:**

- `--use_flash_attn`: Flash Attentionを有効にし、計算を高速化します。
- `lumina_train.py`は比較的新しく、オプションは他のスクリプトと共通化されている部分が多いです。基本的なコマンドパターンに従って学習を行えます。

**コマンド例:**

```bash
accelerate launch --mixed_precision bf16 lumina_train.py \
  --pretrained_model_name_or_path "Lumina-Next-DiT-B.safetensors" \
  --dataset_config "dataset_config.toml" \
  --output_dir "output" \
  --output_name "lumina_finetuned" \
  --learning_rate 1e-5
```

</details>

--- 

### Differences between Fine-tuning and LoRA tuning per architecture

| Architecture | Key Features/Options Specific to Fine-tuning | Main Differences from LoRA tuning |
|:---|:---|:---|
| **SDXL** | `--block_lr` | Only fine-tuning allows for granular control over the learning rate for each U-Net block. |
| **SD3** | `--train_text_encoder`, `--train_t5xxl`, `--num_last_block_to_freeze` | Only fine-tuning can train the entire Text Encoders. LoRA only trains the adapter parts. |
| **FLUX.1** | `--blockwise_fused_optimizers` | Since fine-tuning updates the entire model's weights, more experimental optimizer options are available. |
| **Lumina** | (Few specific options) | Basic training options are common, but fine-tuning differs in that it updates the entire model's foundation. |

<details>
<summary>日本語</summary>

### アーキテクチャごとのFine-tuningとLoRA学習の違い

| アーキテクチャ | Fine-tuning特有の主要機能・オプション | LoRA学習との主な違い |
|:---|:---|:---|
| **SDXL** | `--block_lr` | U-Netのブロックごとに学習率を細かく制御できるのはFine-tuningのみです。 |
| **SD3** | `--train_text_encoder`, `--train_t5xxl`, `--num_last_block_to_freeze` | Text Encoder全体を学習対象にできるのはFine-tuningです。LoRAではアダプター部分のみ学習します。 |
| **FLUX.1** | `--blockwise_fused_optimizers` | Fine-tuningではモデル全体の重みを更新するため、より実験的なオプティマイザの選択肢が用意されています。 |
| **Lumina** | （特有のオプションは少ない） | 基本的な学習オプションは共通ですが、Fine-tuningはモデルの基盤全体を更新する点で異なります。 |

</details>
