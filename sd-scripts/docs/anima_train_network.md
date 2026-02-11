# LoRA Training Guide for Anima using `anima_train_network.py` / `anima_train_network.py` を用いたAnima モデルのLoRA学習ガイド

This document explains how to train LoRA (Low-Rank Adaptation) models for Anima using `anima_train_network.py` in the `sd-scripts` repository.

<details>
<summary>日本語</summary>

このドキュメントでは、`sd-scripts`リポジトリに含まれる`anima_train_network.py`を使用して、Anima モデルに対するLoRA (Low-Rank Adaptation) モデルを学習する基本的な手順について解説します。

</details>

## 1. Introduction / はじめに

`anima_train_network.py` trains additional networks such as LoRA for Anima models. Anima adopts a DiT (Diffusion Transformer) architecture based on the MiniTrainDIT design with Rectified Flow training. It uses a Qwen3-0.6B text encoder, an LLM Adapter (6-layer transformer bridge from Qwen3 to T5-compatible space), and a Qwen-Image VAE (16-channel, 8x spatial downscale). 

Qwen-Image VAE and Qwen-Image VAE have same architecture, but [official Anima weight is named for Qwen-Image VAE](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/vae).

This guide assumes you already understand the basics of LoRA training. For common usage and options, see the [train_network.py guide](train_network.md). Some parameters are similar to those in [`sd3_train_network.py`](sd3_train_network.md) and [`flux_train_network.py`](flux_train_network.md).

**Prerequisites:**

* The `sd-scripts` repository has been cloned and the Python environment is ready.
* A training dataset has been prepared. See the [Dataset Configuration Guide](./config_README-en.md).
* Anima model files for training are available.

<details>
<summary>日本語</summary>

`anima_train_network.py`は、Anima モデルに対してLoRAなどの追加ネットワークを学習させるためのスクリプトです。AnimaはMiniTrainDIT設計に基づくDiT (Diffusion Transformer) アーキテクチャを採用しており、Rectified Flow学習を使用します。テキストエンコーダーとしてQwen3-0.6B、LLM Adapter (Qwen3からT5互換空間への6層Transformerブリッジ)、およびQwen-Image VAE (16チャンネル、8倍空間ダウンスケール) を使用します。

Qwen-Image VAEとQwen-Image VAEは同じアーキテクチャですが、[Anima公式の重みはQwen-Image VAE用](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/vae)のようです。

このガイドは、基本的なLoRA学習の手順を理解しているユーザーを対象としています。基本的な使い方や共通のオプションについては、[`train_network.py`のガイド](train_network.md)を参照してください。また一部のパラメータは [`sd3_train_network.py`](sd3_train_network.md) や [`flux_train_network.py`](flux_train_network.md) と同様のものがあるため、そちらも参考にしてください。

**前提条件:**

* `sd-scripts`リポジトリのクローンとPython環境のセットアップが完了していること。
* 学習用データセットの準備が完了していること。（データセットの準備については[データセット設定ガイド](./config_README-en.md)を参照してください）
* 学習対象のAnimaモデルファイルが準備できていること。
</details>

## 2. Differences from `train_network.py` / `train_network.py` との違い

`anima_train_network.py` is based on `train_network.py` but modified for Anima. Main differences are:

* **Target models:** Anima DiT models.
* **Model structure:** Uses a MiniTrainDIT (Transformer based) instead of U-Net. Employs a single text encoder (Qwen3-0.6B), an LLM Adapter that bridges Qwen3 embeddings to T5-compatible cross-attention space, and a Qwen-Image VAE (16-channel latent space with 8x spatial downscale).
* **Arguments:** Uses the common `--pretrained_model_name_or_path` for the DiT model path, `--qwen3` for the Qwen3 text encoder, and `--vae` for the Qwen-Image VAE. The LLM adapter and T5 tokenizer can be specified separately with `--llm_adapter_path` and `--t5_tokenizer_path`.
* **Incompatible arguments:** Stable Diffusion v1/v2 options such as `--v2`, `--v_parameterization` and `--clip_skip` are not used. `--fp8_base` is not supported.
* **Timestep sampling:** Uses the same `--timestep_sampling` options as FLUX training (`sigma`, `uniform`, `sigmoid`, `shift`, `flux_shift`).
* **LoRA:** Uses regex-based module selection and per-module rank/learning rate control (`network_reg_dims`, `network_reg_lrs`) instead of per-component arguments. Module exclusion/inclusion is controlled by `exclude_patterns` and `include_patterns`.

<details>
<summary>日本語</summary>

`anima_train_network.py`は`train_network.py`をベースに、Anima モデルに対応するための変更が加えられています。主な違いは以下の通りです。

* **対象モデル:** Anima DiTモデルを対象とします。
* **モデル構造:** U-Netの代わりにMiniTrainDIT (Transformerベース) を使用します。テキストエンコーダーとしてQwen3-0.6B、Qwen3埋め込みをT5互換のクロスアテンション空間に変換するLLM Adapter、およびQwen-Image VAE (16チャンネル潜在空間、8倍空間ダウンスケール) を使用します。
* **引数:** DiTモデルのパスには共通引数`--pretrained_model_name_or_path`を、Qwen3テキストエンコーダーには`--qwen3`を、Qwen-Image VAEには`--vae`を使用します。LLM AdapterとT5トークナイザーはそれぞれ`--llm_adapter_path`、`--t5_tokenizer_path`で個別に指定できます。
* **一部引数の非互換性:** Stable Diffusion v1/v2向けの引数（例: `--v2`, `--v_parameterization`, `--clip_skip`）は使用されません。`--fp8_base`はサポートされていません。
* **タイムステップサンプリング:** FLUX学習と同じ`--timestep_sampling`オプション（`sigma`、`uniform`、`sigmoid`、`shift`、`flux_shift`）を使用します。
* **LoRA:** コンポーネント別の引数の代わりに、正規表現ベースのモジュール選択とモジュール単位のランク/学習率制御（`network_reg_dims`、`network_reg_lrs`）を使用します。モジュールの除外/包含は`exclude_patterns`と`include_patterns`で制御します。
</details>

## 3. Preparation / 準備

The following files are required before starting training:

1. **Training script:** `anima_train_network.py`
2. **Anima DiT model file:** `.safetensors` file for the base DiT model.
3. **Qwen3-0.6B text encoder:** Either a HuggingFace model directory, or a single `.safetensors` file (uses the bundled config files in `configs/qwen3_06b/`).
4. **Qwen-Image VAE model file:** `.safetensors` or `.pth` file for the VAE.
5. **LLM Adapter model file (optional):** `.safetensors` file. If not provided separately, the adapter is loaded from the DiT file if the key `llm_adapter.out_proj.weight` exists.
6. **T5 Tokenizer (optional):** If not specified, uses the bundled tokenizer at `configs/t5_old/`.
7. **Dataset definition file (.toml):** Dataset settings in TOML format. (See the [Dataset Configuration Guide](./config_README-en.md).) In this document we use `my_anima_dataset_config.toml` as an example.

Model files can be obtained from the [Anima HuggingFace repository](https://huggingface.co/circlestone-labs/Anima).

**Notes:**
* The T5 tokenizer only needs the tokenizer files (not the T5 model weights). It uses the vocabulary from `google/t5-v1_1-xxl`.

<details>
<summary>日本語</summary>

学習を開始する前に、以下のファイルが必要です。

1. **学習スクリプト:** `anima_train_network.py`
2. **Anima DiTモデルファイル:** ベースとなるDiTモデルの`.safetensors`ファイル。
3. **Qwen3-0.6Bテキストエンコーダー:** HuggingFaceモデルディレクトリまたは単体の`.safetensors`ファイル（バンドル版の`configs/qwen3_06b/`の設定ファイルが使用されます）。
4. **Qwen-Image VAEモデルファイル:** VAEの`.safetensors`または`.pth`ファイル。
5. **LLM Adapterモデルファイル（オプション）:** `.safetensors`ファイル。個別に指定しない場合、DiTファイル内に`llm_adapter.out_proj.weight`キーが存在すればそこから読み込まれます。
6. **T5トークナイザー（オプション）:** 指定しない場合、`configs/t5_old/`のバンドル版トークナイザーを使用します。
7. **データセット定義ファイル (.toml):** 学習データセットの設定を記述したTOML形式のファイル。（詳細は[データセット設定ガイド](./config_README-en.md)を参照してください）。例として`my_anima_dataset_config.toml`を使用します。

モデルファイルは[HuggingFaceのAnimaリポジトリ](https://huggingface.co/circlestone-labs/Anima)から入手できます。

**注意:**
* T5トークナイザーを別途指定する場合、トークナイザーファイルのみ必要です（T5モデルの重みは不要）。`google/t5-v1_1-xxl`の語彙を使用します。
</details>

## 4. Running the Training / 学習の実行

Execute `anima_train_network.py` from the terminal to start training. The overall command-line format is the same as `train_network.py`, but Anima specific options must be supplied.

Example command:

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_network.py \
  --pretrained_model_name_or_path="<path to Anima DiT model>" \
  --qwen3="<path to Qwen3-0.6B model or directory>" \
  --vae="<path to Qwen-Image VAE model>" \
  --dataset_config="my_anima_dataset_config.toml" \
  --output_dir="<output directory>" \
  --output_name="my_anima_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora_anima \
  --network_dim=8 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW8bit" \
  --lr_scheduler="constant" \
  --timestep_sampling="sigmoid" \
  --discrete_flow_shift=1.0 \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --vae_chunk_size=64 \
  --vae_disable_cache
```

*(Write the command on one line or use `\` or `^` for line breaks.)*

**Note:** `--vae_chunk_size` and `--vae_disable_cache` are custom options in this repository to reduce memory usage of the Qwen-Image VAE.

<details>
<summary>日本語</summary>

学習は、ターミナルから`anima_train_network.py`を実行することで開始します。基本的なコマンドラインの構造は`train_network.py`と同様ですが、Anima特有の引数を指定する必要があります。

コマンドラインの例は英語のドキュメントを参照してください。

※実際には1行で書くか、適切な改行文字（`\` または `^`）を使用してください。

注意: `--vae_chunk_size`および`--vae_disable_cache`は当リポジトリ独自のオプションで、Qwen-Image VAEのメモリ使用量を削減するために使用します。

</details>

### 4.1. Explanation of Key Options / 主要なコマンドライン引数の解説

Besides the arguments explained in the [train_network.py guide](train_network.md), specify the following Anima specific options. For shared options (`--output_dir`, `--output_name`, `--network_module`, etc.), see that guide.

#### Model Options [Required] / モデル関連 [必須]

* `--pretrained_model_name_or_path="<path to Anima DiT model>"` **[Required]**
  - Path to the Anima DiT model `.safetensors` file. The model config (channels, blocks, heads) is auto-detected from the state dict. ComfyUI format with `net.` prefix is supported.
* `--qwen3="<path to Qwen3-0.6B model>"` **[Required]**
  - Path to the Qwen3-0.6B text encoder. Can be a HuggingFace model directory or a single `.safetensors` file. The text encoder is always frozen during training.
* `--vae="<path to Qwen-Image VAE model>"` **[Required]**
  - Path to the Qwen-Image VAE model `.safetensors` or `.pth` file. Fixed config: `dim=96, z_dim=16`.

#### Model Options [Optional] / モデル関連 [オプション]

* `--llm_adapter_path="<path to LLM adapter>"` *[Optional]*
  - Path to a separate LLM adapter weights file. If omitted, the adapter is loaded from the DiT file when the key `llm_adapter.out_proj.weight` exists.
* `--t5_tokenizer_path="<path to T5 tokenizer>"` *[Optional]*
  - Path to the T5 tokenizer directory. If omitted, uses the bundled config at `configs/t5_old/`.

#### Anima Training Parameters / Anima 学習パラメータ

* `--timestep_sampling=<choice>`
  - Timestep sampling method. Choose from `sigma`, `uniform`, `sigmoid` (default), `shift`, `flux_shift`. Same options as FLUX training. See the [flux_train_network.py guide](flux_train_network.md) for details on each method.
* `--discrete_flow_shift=<float>`
  - Shift for the timestep distribution in Rectified Flow training. Default `1.0`. This value is used when `--timestep_sampling` is set to **`shift`**. The shift formula is `t_shifted = (t * shift) / (1 + (shift - 1) * t)`.
* `--sigmoid_scale=<float>`
  - Scale factor when `--timestep_sampling` is set to `sigmoid`, `shift`, or `flux_shift`. Default `1.0`.
* `--qwen3_max_token_length=<integer>`
  - Maximum token length for the Qwen3 tokenizer. Default `512`.
* `--t5_max_token_length=<integer>`
  - Maximum token length for the T5 tokenizer. Default `512`.
* `--attn_mode=<choice>`
  - Attention implementation to use. Choose from `torch` (default), `xformers`, `flash`, `sageattn`. `xformers` requires `--split_attn`. `sageattn` does not support training (inference only). This option overrides `--xformers`.
* `--split_attn`
  - Split attention computation to reduce memory usage. Required when using `--attn_mode xformers`.
  
#### Component-wise Learning Rates / コンポーネント別学習率

These options set separate learning rates for each component of the Anima model. They are primarily used for full fine-tuning. Set to `0` to freeze a component:

* `--self_attn_lr=<float>` - Learning rate for self-attention layers. Default: same as `--learning_rate`.
* `--cross_attn_lr=<float>` - Learning rate for cross-attention layers. Default: same as `--learning_rate`.
* `--mlp_lr=<float>` - Learning rate for MLP layers. Default: same as `--learning_rate`.
* `--mod_lr=<float>` - Learning rate for AdaLN modulation layers. Default: same as `--learning_rate`. Note: modulation layers are not included in LoRA by default.
* `--llm_adapter_lr=<float>` - Learning rate for LLM adapter layers. Default: same as `--learning_rate`.

For LoRA training, use `network_reg_lrs` in `--network_args` instead. See [Section 5.2](#52-regex-based-rank-and-learning-rate-control--正規表現によるランク学習率の制御).

#### Memory and Speed / メモリ・速度関連

* `--blocks_to_swap=<integer>`
  - Number of Transformer blocks to swap between CPU and GPU. More blocks reduce VRAM but slow training. Maximum values depend on model size:
    - 28-block model: max **26** (Anima-Preview)
    - 36-block model: max **34**
    - 20-block model: max **18**
  - Cannot be used with `--cpu_offload_checkpointing` or `--unsloth_offload_checkpointing`.
* `--unsloth_offload_checkpointing`
  - Offload activations to CPU RAM using async non-blocking transfers (faster than `--cpu_offload_checkpointing`). Cannot be combined with `--cpu_offload_checkpointing` or `--blocks_to_swap`.
* `--cache_text_encoder_outputs`
  - Cache Qwen3 text encoder outputs to reduce VRAM usage. Recommended when not training text encoder LoRA.
* `--cache_text_encoder_outputs_to_disk`
  - Cache text encoder outputs to disk. Auto-enables `--cache_text_encoder_outputs`.
* `--cache_latents`, `--cache_latents_to_disk`
  - Cache Qwen-Image VAE latent outputs.
* `--vae_chunk_size=<integer>`
  - Chunk size for Qwen-Image VAE processing. Reduces VRAM usage at the cost of speed. Default is no chunking.
* `--vae_disable_cache`
  - Disable internal caching in Qwen-Image VAE to reduce VRAM usage.
  
#### Incompatible or Unsupported Options / 非互換・非サポートの引数

* `--v2`, `--v_parameterization`, `--clip_skip` - Options for Stable Diffusion v1/v2 that are not used for Anima training.
* `--fp8_base` - Not supported for Anima. If specified, it will be disabled with a warning.

<details>
<summary>日本語</summary>

[`train_network.py`のガイド](train_network.md)で説明されている引数に加え、以下のAnima特有の引数を指定します。共通の引数については、上記ガイドを参照してください。

#### モデル関連 [必須]

* `--pretrained_model_name_or_path="<path to Anima DiT model>"` **[必須]** - Anima DiTモデルの`.safetensors`ファイルのパスを指定します。モデルの設定はstate dictから自動検出されます。`net.`プレフィックス付きのComfyUIフォーマットもサポートしています。
* `--qwen3="<path to Qwen3-0.6B model>"` **[必須]** - Qwen3-0.6Bテキストエンコーダーのパスを指定します。HuggingFaceモデルディレクトリまたは単体の`.safetensors`ファイルが使用できます。
* `--vae="<path to Qwen-Image VAE model>"` **[必須]** - Qwen-Image VAEモデルのパスを指定します。

#### モデル関連 [オプション]

* `--llm_adapter_path="<path to LLM adapter>"` *[オプション]* - 個別のLLM Adapterの重みファイルのパス。
* `--t5_tokenizer_path="<path to T5 tokenizer>"` *[オプション]* - T5トークナイザーディレクトリのパス。

#### Anima 学習パラメータ

* `--timestep_sampling` - タイムステップのサンプリング方法。`sigma`、`uniform`、`sigmoid`（デフォルト）、`shift`、`flux_shift`から選択。FLUX学習と同じオプションです。各方法の詳細は[flux_train_network.pyのガイド](flux_train_network.md)を参照してください。
* `--discrete_flow_shift` - Rectified Flow学習のタイムステップ分布シフト。デフォルト`1.0`。`--timestep_sampling`が`shift`の場合に使用されます。
* `--sigmoid_scale` - `sigmoid`、`shift`、`flux_shift`タイムステップサンプリングのスケール係数。デフォルト`1.0`。
* `--qwen3_max_token_length` - Qwen3トークナイザーの最大トークン長。デフォルト`512`。
* `--t5_max_token_length` - T5トークナイザーの最大トークン長。デフォルト`512`。
* `--attn_mode` - 使用するAttentionの実装。`torch`（デフォルト）、`xformers`、`flash`、`sageattn`から選択。`xformers`は`--split_attn`の指定が必要です。`sageattn`はトレーニングをサポートしていません（推論のみ）。
* `--split_attn` - メモリ使用量を減らすためにattention時にバッチを分割します。`--attn_mode xformers`使用時に必要です。

#### コンポーネント別学習率

これらのオプションは、Animaモデルの各コンポーネントに個別の学習率を設定します。主にフルファインチューニング用です。`0`に設定するとそのコンポーネントをフリーズします：

* `--self_attn_lr` - Self-attention層の学習率。
* `--cross_attn_lr` - Cross-attention層の学習率。
* `--mlp_lr` - MLP層の学習率。
* `--mod_lr` - AdaLNモジュレーション層の学習率。モジュレーション層はデフォルトではLoRAに含まれません。
* `--llm_adapter_lr` - LLM Adapter層の学習率。

LoRA学習の場合は、`--network_args`の`network_reg_lrs`を使用してください。[セクション5.2](#52-regex-based-rank-and-learning-rate-control--正規表現によるランク学習率の制御)を参照。

#### メモリ・速度関連

* `--blocks_to_swap` - TransformerブロックをCPUとGPUでスワップしてVRAMを節約。`--cpu_offload_checkpointing`および`--unsloth_offload_checkpointing`とは併用できません。
* `--unsloth_offload_checkpointing` - 非同期転送でアクティベーションをCPU RAMにオフロード。`--cpu_offload_checkpointing`および`--blocks_to_swap`とは併用できません。
* `--cache_text_encoder_outputs` - Qwen3の出力をキャッシュしてメモリ使用量を削減。
* `--cache_latents`, `--cache_latents_to_disk` - Qwen-Image VAEの出力をキャッシュ。
* `--vae_chunk_size` - Qwen-Image VAEのチャンク処理サイズ。メモリ使用量を削減しますが速度が低下します。デフォルトはチャンク処理なし。
* `--vae_disable_cache` - Qwen-Image VAEの内部キャッシュを無効化してメモリ使用量を削減します。

#### 非互換・非サポートの引数

* `--v2`, `--v_parameterization`, `--clip_skip` - Stable Diffusion v1/v2向けの引数。Animaの学習では使用されません。
* `--fp8_base` - Animaではサポートされていません。指定した場合、警告とともに無効化されます。
</details>

### 4.2. Starting Training / 学習の開始

After setting the required arguments, run the command to begin training. The overall flow and how to check logs are the same as in the [train_network.py guide](train_network.md#32-starting-the-training--学習の開始).

<details>
<summary>日本語</summary>

必要な引数を設定したら、コマンドを実行して学習を開始します。全体の流れやログの確認方法は、[train_network.pyのガイド](train_network.md#32-starting-the-training--学習の開始)と同様です。

</details>

## 5. LoRA Target Modules / LoRAの学習対象モジュール

When training LoRA with `anima_train_network.py`, the following modules are targeted by default:

* **DiT Blocks (`Block`)**: Self-attention (`self_attn`), cross-attention (`cross_attn`), and MLP (`mlp`) layers within each transformer block. Modulation (`adaln_modulation`), norm, embedder, and final layers are excluded by default.
* **Embedding layers (`PatchEmbed`, `TimestepEmbedding`) and Final layer (`FinalLayer`)**: Excluded by default but can be included using `include_patterns`.
* **LLM Adapter Blocks (`LLMAdapterTransformerBlock`)**: Only when `--network_args "train_llm_adapter=True"` is specified.
* **Text Encoder (Qwen3)**: Only when `--network_train_unet_only` is NOT specified and `--cache_text_encoder_outputs` is NOT used.

The LoRA network module is `networks.lora_anima`.

### 5.1. Module Selection with Patterns / パターンによるモジュール選択

By default, the following modules are excluded from LoRA via the built-in exclude pattern:
```
.*(_modulation|_norm|_embedder|final_layer).*
```

You can customize which modules are included or excluded using regex patterns in `--network_args`:

* `exclude_patterns` - Exclude modules matching these patterns (in addition to the default exclusion).
* `include_patterns` - Force-include modules matching these patterns, overriding exclusion.

Patterns are matched against the full module name using `re.fullmatch()`.

Example to include the final layer:
```
--network_args "include_patterns=['.*final_layer.*']"
```

Example to additionally exclude MLP layers:
```
--network_args "exclude_patterns=['.*mlp.*']"
```

### 5.2. Regex-based Rank and Learning Rate Control / 正規表現によるランク・学習率の制御

You can specify different ranks (network_dim) and learning rates for modules matching specific regex patterns:

* `network_reg_dims`: Specify ranks for modules matching a regular expression. The format is a comma-separated string of `pattern=rank`.
    * Example: `--network_args "network_reg_dims=.*self_attn.*=8,.*cross_attn.*=4,.*mlp.*=8"`
    * This sets the rank to 8 for self-attention modules, 4 for cross-attention modules, and 8 for MLP modules.
* `network_reg_lrs`: Specify learning rates for modules matching a regular expression. The format is a comma-separated string of `pattern=lr`.
    * Example: `--network_args "network_reg_lrs=.*self_attn.*=1e-4,.*cross_attn.*=5e-5"`
    * This sets the learning rate to `1e-4` for self-attention modules and `5e-5` for cross-attention modules.

**Notes:**

* Settings via `network_reg_dims` and `network_reg_lrs` take precedence over the global `--network_dim` and `--learning_rate` settings.
* Patterns are matched using `re.fullmatch()` against the module's original name (e.g., `blocks.0.self_attn.q_proj`).

### 5.3. LLM Adapter LoRA / LLM Adapter LoRA

To apply LoRA to the LLM Adapter blocks:

```
--network_args "train_llm_adapter=True"
```

### 5.4. Other Network Args / その他のネットワーク引数

* `--network_args "verbose=True"` - Print all LoRA module names and their dimensions.
* `--network_args "rank_dropout=0.1"` - Rank dropout rate.
* `--network_args "module_dropout=0.1"` - Module dropout rate.
* `--network_args "loraplus_lr_ratio=2.0"` - LoRA+ learning rate ratio.
* `--network_args "loraplus_unet_lr_ratio=2.0"` - LoRA+ learning rate ratio for DiT only.
* `--network_args "loraplus_text_encoder_lr_ratio=2.0"` - LoRA+ learning rate ratio for text encoder only.

<details>
<summary>日本語</summary>

`anima_train_network.py`でLoRAを学習させる場合、デフォルトでは以下のモジュールが対象となります。

* **DiTブロック (`Block`)**: 各Transformerブロック内のSelf-attention（`self_attn`）、Cross-attention（`cross_attn`）、MLP（`mlp`）層。モジュレーション（`adaln_modulation`）、norm、embedder、final layerはデフォルトで除外されます。
* **埋め込み層 (`PatchEmbed`, `TimestepEmbedding`) と最終層 (`FinalLayer`)**: デフォルトで除外されますが、`include_patterns`で含めることができます。
* **LLM Adapterブロック (`LLMAdapterTransformerBlock`)**: `--network_args "train_llm_adapter=True"`を指定した場合のみ。
* **テキストエンコーダー (Qwen3)**: `--network_train_unet_only`を指定せず、かつ`--cache_text_encoder_outputs`を使用しない場合のみ。

### 5.1. パターンによるモジュール選択

デフォルトでは以下のモジュールが組み込みの除外パターンによりLoRAから除外されます：
```
.*(_modulation|_norm|_embedder|final_layer).*
```

`--network_args`で正規表現パターンを使用して、含めるモジュールと除外するモジュールをカスタマイズできます：

* `exclude_patterns` - これらのパターンにマッチするモジュールを除外（デフォルトの除外に追加）。
* `include_patterns` - これらのパターンにマッチするモジュールを強制的に含める（除外を上書き）。

パターンは`re.fullmatch()`を使用して完全なモジュール名に対してマッチングされます。

### 5.2. 正規表現によるランク・学習率の制御

正規表現にマッチするモジュールに対して、異なるランクや学習率を指定できます：

* `network_reg_dims`: 正規表現にマッチするモジュールに対してランクを指定します。`pattern=rank`形式の文字列をカンマで区切って指定します。
    * 例: `--network_args "network_reg_dims=.*self_attn.*=8,.*cross_attn.*=4,.*mlp.*=8"`
* `network_reg_lrs`: 正規表現にマッチするモジュールに対して学習率を指定します。`pattern=lr`形式の文字列をカンマで区切って指定します。
    * 例: `--network_args "network_reg_lrs=.*self_attn.*=1e-4,.*cross_attn.*=5e-5"`

**注意点:**
* `network_reg_dims`および`network_reg_lrs`での設定は、全体設定である`--network_dim`や`--learning_rate`よりも優先されます。
* パターンはモジュールのオリジナル名（例: `blocks.0.self_attn.q_proj`）に対して`re.fullmatch()`でマッチングされます。

### 5.3. LLM Adapter LoRA

LLM AdapterブロックにLoRAを適用するには：`--network_args "train_llm_adapter=True"`

### 5.4. その他のネットワーク引数

* `verbose=True` - 全LoRAモジュール名とdimを表示
* `rank_dropout` - ランクドロップアウト率
* `module_dropout` - モジュールドロップアウト率
* `loraplus_lr_ratio` - LoRA+学習率比率
* `loraplus_unet_lr_ratio` - DiT専用のLoRA+学習率比率
* `loraplus_text_encoder_lr_ratio` - テキストエンコーダー専用のLoRA+学習率比率

</details>

## 6. Using the Trained Model / 学習済みモデルの利用

When training finishes, a LoRA model file (e.g. `my_anima_lora.safetensors`) is saved in the directory specified by `output_dir`. Use this file with inference environments that support Anima, such as ComfyUI with appropriate nodes.

<details>
<summary>日本語</summary>

学習が完了すると、指定した`output_dir`にLoRAモデルファイル（例: `my_anima_lora.safetensors`）が保存されます。このファイルは、Anima モデルに対応した推論環境（例: ComfyUI + 適切なノード）で使用できます。

</details>

## 7. Advanced Settings / 高度な設定

### 7.1. VRAM Usage Optimization / VRAM使用量の最適化

Anima models can be large, so GPUs with limited VRAM may require optimization:

#### Key VRAM Reduction Options

- **`--blocks_to_swap <number>`**: Swaps blocks between CPU and GPU to reduce VRAM usage. Higher numbers save more VRAM but reduce training speed. See model-specific max values in section 4.1.

- **`--unsloth_offload_checkpointing`**: Offloads gradient checkpoints to CPU using async non-blocking transfers. Faster than `--cpu_offload_checkpointing`. Cannot be combined with `--blocks_to_swap`.

- **`--gradient_checkpointing`**: Standard gradient checkpointing to reduce VRAM at the cost of compute.

- **`--cache_text_encoder_outputs`**: Caches Qwen3 outputs so the text encoder can be freed from VRAM during training.

- **`--cache_latents`**: Caches Qwen-Image VAE outputs so the VAE can be freed from VRAM during training.

- **Using Adafactor optimizer**: Can reduce VRAM usage:
  ```
  --optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --lr_scheduler constant_with_warmup --max_grad_norm 0.0
  ```

<details>
<summary>日本語</summary>

Animaモデルは大きい場合があるため、VRAMが限られたGPUでは最適化が必要です。

主要なVRAM削減オプション：
- `--blocks_to_swap`: CPUとGPU間でブロックをスワップ
- `--unsloth_offload_checkpointing`: 非同期転送でアクティベーションをCPUにオフロード
- `--gradient_checkpointing`: 標準的な勾配チェックポイント
- `--cache_text_encoder_outputs`: Qwen3の出力をキャッシュ
- `--cache_latents`: Qwen-Image VAEの出力をキャッシュ
- Adafactorオプティマイザの使用

</details>

### 7.2. Training Settings / 学習設定

#### Timestep Sampling

The `--timestep_sampling` option specifies how timesteps are sampled. The available methods are the same as FLUX training:

- `sigma`: Sigma-based sampling like SD3.
- `uniform`: Uniform random sampling from [0, 1].
- `sigmoid` (default): Sample from Normal(0,1), multiply by `sigmoid_scale`, apply sigmoid. Good general-purpose option.
- `shift`: Like `sigmoid`, but applies the discrete flow shift formula: `t_shifted = (t * shift) / (1 + (shift - 1) * t)`.
- `flux_shift`: Resolution-dependent shift used in FLUX training.

See the [flux_train_network.py guide](flux_train_network.md) for detailed descriptions.

#### Discrete Flow Shift

The `--discrete_flow_shift` option (default `1.0`) only applies when `--timestep_sampling` is set to `shift`. The formula is:

```
t_shifted = (t * shift) / (1 + (shift - 1) * t)
```

#### Loss Weighting

The `--weighting_scheme` option specifies loss weighting by timestep:

- `uniform` (default): Equal weight for all timesteps.
- `sigma_sqrt`: Weight by `sigma^(-2)`.
- `cosmap`: Weight by `2 / (pi * (1 - 2*sigma + 2*sigma^2))`.
- `none`: Same as uniform.
- `logit_normal`, `mode`: Additional schemes from SD3 training. See the [`sd3_train_network.md` guide](sd3_train_network.md) for details.

#### Caption Dropout

Caption dropout uses the `caption_dropout_rate` setting from the dataset configuration (per-subset in TOML). When using `--cache_text_encoder_outputs`, the dropout rate is stored with each cached entry and applied during training, so caption dropout is compatible with text encoder output caching.

<details>
<summary>日本語</summary>

#### タイムステップサンプリング

`--timestep_sampling`でタイムステップのサンプリング方法を指定します。FLUX学習と同じ方法が利用できます：

- `sigma`: SD3と同様のシグマベースサンプリング。
- `uniform`: [0, 1]の一様分布からサンプリング。
- `sigmoid`（デフォルト）: 正規分布からサンプリングし、sigmoidを適用。汎用的なオプション。
- `shift`: `sigmoid`と同様だが、離散フローシフトの式を適用。
- `flux_shift`: FLUX学習で使用される解像度依存のシフト。

詳細は[flux_train_network.pyのガイド](flux_train_network.md)を参照してください。

#### 離散フローシフト

`--discrete_flow_shift`（デフォルト`1.0`）は`--timestep_sampling`が`shift`の場合のみ適用されます。

#### 損失の重み付け

`--weighting_scheme`でタイムステップごとの損失の重み付けを指定します。

#### キャプションドロップアウト

キャプションドロップアウトにはデータセット設定（TOMLでのサブセット単位）の`caption_dropout_rate`を使用します。`--cache_text_encoder_outputs`使用時は、ドロップアウト率が各キャッシュエントリとともに保存され、学習中に適用されるため、テキストエンコーダー出力キャッシュとの互換性があります。

</details>

### 7.3. Text Encoder LoRA Support / Text Encoder LoRAのサポート

Anima LoRA training supports training Qwen3 text encoder LoRA:

- To train only DiT: specify `--network_train_unet_only`
- To train DiT and Qwen3: omit `--network_train_unet_only` and do NOT use `--cache_text_encoder_outputs`

You can specify a separate learning rate for Qwen3 with `--text_encoder_lr`. If not specified, the default `--learning_rate` is used.

Note: When `--cache_text_encoder_outputs` is used, text encoder outputs are pre-computed and the text encoder is removed from GPU, so text encoder LoRA cannot be trained.

<details>
<summary>日本語</summary>

Anima LoRA学習では、Qwen3テキストエンコーダーのLoRAもトレーニングできます。

- DiTのみ学習: `--network_train_unet_only`を指定
- DiTとQwen3を学習: `--network_train_unet_only`を省略し、`--cache_text_encoder_outputs`を使用しない

Qwen3に個別の学習率を指定するには`--text_encoder_lr`を使用します。未指定の場合は`--learning_rate`が使われます。

注意: `--cache_text_encoder_outputs`を使用する場合、テキストエンコーダーの出力が事前に計算されGPUから解放されるため、テキストエンコーダーLoRAは学習できません。

</details>

## 8. Other Training Options / その他の学習オプション

- **`--loss_type`**: Loss function for training. Default `l2`.
  - `l1`: L1 loss.
  - `l2`: L2 loss (mean squared error).
  - `huber`: Huber loss.
  - `smooth_l1`: Smooth L1 loss.

- **`--huber_schedule`**, **`--huber_c`**, **`--huber_scale`**: Parameters for Huber loss when `--loss_type` is `huber` or `smooth_l1`.

- **`--ip_noise_gamma`**, **`--ip_noise_gamma_random_strength`**: Input Perturbation noise gamma values.

- **`--fused_backward_pass`**: Fuses the backward pass and optimizer step to reduce VRAM usage. Only works with Adafactor. For details, see the [`sdxl_train_network.py` guide](sdxl_train_network.md).

- **`--weighting_scheme`**, **`--logit_mean`**, **`--logit_std`**, **`--mode_scale`**: Timestep loss weighting options. For details, refer to the [`sd3_train_network.md` guide](sd3_train_network.md).

<details>
<summary>日本語</summary>

- **`--loss_type`**: 学習に用いる損失関数。デフォルト`l2`。`l1`, `l2`, `huber`, `smooth_l1`から選択。
- **`--huber_schedule`**, **`--huber_c`**, **`--huber_scale`**: Huber損失のパラメータ。
- **`--ip_noise_gamma`**: Input Perturbationノイズガンマ値。
- **`--fused_backward_pass`**: バックワードパスとオプティマイザステップの融合。
- **`--weighting_scheme`** 等: タイムステップ損失の重み付け。詳細は[`sd3_train_network.md`](sd3_train_network.md)を参照。

</details>

## 9. Others / その他

### Metadata Saved in LoRA Models

The following metadata is saved in the LoRA model file:

* `ss_weighting_scheme`
* `ss_logit_mean`
* `ss_logit_std`
* `ss_mode_scale`
* `ss_timestep_sampling`
* `ss_sigmoid_scale`
* `ss_discrete_flow_shift`

<details>
<summary>日本語</summary>

`anima_train_network.py`には、サンプル画像の生成 (`--sample_prompts`など) や詳細なオプティマイザ設定など、`train_network.py`と共通の機能も多く存在します。これらについては、[`train_network.py`のガイド](train_network.md#5-other-features--その他の機能)やスクリプトのヘルプ (`python anima_train_network.py --help`) を参照してください。

### LoRAモデルに保存されるメタデータ

以下のメタデータがLoRAモデルファイルに保存されます：

* `ss_weighting_scheme`
* `ss_logit_mean`
* `ss_logit_std`
* `ss_mode_scale`
* `ss_timestep_sampling`
* `ss_sigmoid_scale`
* `ss_discrete_flow_shift`

</details>
