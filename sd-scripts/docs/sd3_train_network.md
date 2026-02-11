# LoRA Training Guide for Stable Diffusion 3/3.5 using `sd3_train_network.py` / `sd3_train_network.py` を用いたStable Diffusion 3/3.5モデルのLoRA学習ガイド

This document explains how to train LoRA (Low-Rank Adaptation) models for Stable Diffusion 3 (SD3) and Stable Diffusion 3.5 (SD3.5) using `sd3_train_network.py` in the `sd-scripts` repository.

## 1. Introduction / はじめに

`sd3_train_network.py` trains additional networks such as LoRA for SD3/3.5 models. SD3 adopts a new architecture called MMDiT (Multi-Modal Diffusion Transformer), so its structure differs from previous Stable Diffusion models. With this script you can create LoRA models specialized for SD3/3.5.

This guide assumes you already understand the basics of LoRA training. For common usage and options, see the [train_network.py guide](train_network.md). Some parameters are the same as those in [`sdxl_train_network.py`](sdxl_train_network.md).

**Prerequisites:**

* The `sd-scripts` repository has been cloned and the Python environment is ready.
* A training dataset has been prepared. See the [Dataset Configuration Guide](link/to/dataset/config/doc).
* SD3/3.5 model files for training are available.

<details>
<summary>日本語</summary>

`sd3_train_network.py`は、Stable Diffusion 3/3.5モデルに対してLoRAなどの追加ネットワークを学習させるためのスクリプトです。SD3は、MMDiT (Multi-Modal Diffusion Transformer) と呼ばれる新しいアーキテクチャを採用しており、従来のStable Diffusionモデルとは構造が異なります。このスクリプトを使用することで、SD3/3.5モデルに特化したLoRAモデルを作成できます。

このガイドは、基本的なLoRA学習の手順を理解しているユーザーを対象としています。基本的な使い方や共通のオプションについては、[`train_network.py`のガイド](train_network.md)を参照してください。また一部のパラメータは [`sdxl_train_network.py`](sdxl_train_network.md) と同様のものがあるため、そちらも参考にしてください。

**前提条件:**

*   `sd-scripts`リポジトリのクローンとPython環境のセットアップが完了していること。
*   学習用データセットの準備が完了していること。（データセットの準備については[データセット設定ガイド](link/to/dataset/config/doc)を参照してください）
*   学習対象のSD3/3.5モデルファイルが準備できていること。
</details>

## 2. Differences from `train_network.py` / `train_network.py` との違い

`sd3_train_network.py` is based on `train_network.py` but modified for SD3/3.5. Main differences are:

* **Target models:** Stable Diffusion 3 and 3.5 Medium/Large.
* **Model structure:** Uses MMDiT (Transformer based) instead of U-Net and employs three text encoders: CLIP-L, CLIP-G and T5-XXL. The VAE is not compatible with SDXL.
* **Arguments:** Options exist to specify the SD3/3.5 model, text encoders and VAE. With a single `.safetensors` file, these paths are detected automatically, so separate paths are optional.
* **Incompatible arguments:** Stable Diffusion v1/v2 options such as `--v2`, `--v_parameterization` and `--clip_skip` are not used.
* **SD3 specific options:** Additional parameters for attention masks, dropout rates, positional embedding adjustments (for SD3.5), timestep sampling and loss weighting.

<details>
<summary>日本語</summary>
`sd3_train_network.py`は`train_network.py`をベースに、SD3/3.5モデルに対応するための変更が加えられています。主な違いは以下の通りです。

*   **対象モデル:** Stable Diffusion 3, 3.5 Medium / Large モデルを対象とします。
*   **モデル構造:** U-Netの代わりにMMDiT (Transformerベース) を使用します。Text EncoderとしてCLIP-L, CLIP-G, T5-XXLの三つを使用します。VAEはSDXLと互換性がありません。
*   **引数:** SD3/3.5モデル、Text Encoder群、VAEを指定する引数があります。ただし、単一ファイルの`.safetensors`形式であれば、内部で自動的に分離されるため、個別のパス指定は必須ではありません。
*   **一部引数の非互換性:** Stable Diffusion v1/v2向けの引数（例: `--v2`, `--v_parameterization`, `--clip_skip`）はSD3/3.5の学習では使用されません。
*   **SD3特有の引数:** Text Encoderのアテンションマスクやドロップアウト率、Positional Embeddingの調整（SD3.5向け）、タイムステップのサンプリングや損失の重み付けに関する引数が追加されています。
</details>

## 3. Preparation / 準備

The following files are required before starting training:

1. **Training script:** `sd3_train_network.py`
2. **SD3/3.5 model file:** `.safetensors` file for the base model and paths to each text encoder. Single-file format can also be used.
3. **Dataset definition file (.toml):** Dataset settings in TOML format. (See the [Dataset Configuration Guide](link/to/dataset/config/doc).) In this document we use `my_sd3_dataset_config.toml` as an example.

<details>
<summary>日本語</summary>
学習を開始する前に、以下のファイルが必要です。

1.  **学習スクリプト:** `sd3_train_network.py`
2.  **SD3/3.5モデルファイル:** 学習のベースとなるSD3/3.5モデルの`.safetensors`ファイル。またText Encoderをそれぞれ対応する引数でパスを指定します。
    * 単一ファイル形式も使用可能です。
3.  **データセット定義ファイル (.toml):** 学習データセットの設定を記述したTOML形式のファイル。（詳細は[データセット設定ガイド](link/to/dataset/config/doc)を参照してください）。
    *   例として`my_sd3_dataset_config.toml`を使用します。
</details>

## 4. Running the Training / 学習の実行

Execute `sd3_train_network.py` from the terminal to start training. The overall command-line format is the same as `train_network.py`, but SD3/3.5 specific options must be supplied.

Example command:

```bash
accelerate launch --num_cpu_threads_per_process 1 sd3_train_network.py \
  --pretrained_model_name_or_path="<path to SD3 model>" \
  --clip_l="<path to CLIP-L model>" \
  --clip_g="<path to CLIP-G model>" \
  --t5xxl="<path to T5-XXL model>" \
  --dataset_config="my_sd3_dataset_config.toml" \
  --output_dir="<output directory for training results>" \
  --output_name="my_sd3_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora \
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
  --weighting_scheme="uniform" \
  --blocks_to_swap=32
```

*(Write the command on one line or use `\` or `^` for line breaks.)*

<details>
<summary>日本語</summary>

学習は、ターミナルから`sd3_train_network.py`を実行することで開始します。基本的なコマンドラインの構造は`train_network.py`と同様ですが、SD3/3.5特有の引数を指定する必要があります。

以下に、基本的なコマンドライン実行例を示します。

```bash
accelerate launch --num_cpu_threads_per_process 1 sd3_train_network.py
 --pretrained_model_name_or_path="<path to SD3 model>"
 --clip_l="<path to CLIP-L model>"
 --clip_g="<path to CLIP-G model>"
 --t5xxl="<path to T5-XXL model>"
 --dataset_config="my_sd3_dataset_config.toml"
 --output_dir="<output directory for training results>"
 --output_name="my_sd3_lora"
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
 --weighting_scheme="uniform"
 --blocks_to_swap=32
```

※実際には1行で書くか、適切な改行文字（`\` または `^`）を使用してください。

</details>

### 4.1. Explanation of Key Options / 主要なコマンドライン引数の解説

Besides the arguments explained in the [train_network.py guide](train_network.md), specify the following SD3/3.5 options. For shared options (`--output_dir`, `--output_name`, etc.), see that guide.

#### Model Options / モデル関連

* `--pretrained_model_name_or_path="<path to SD3 model>"` **required** – Path to the SD3/3.5 model.
* `--clip_l`, `--clip_g`, `--t5xxl`, `--vae` – Skip these if the base model is a single file; otherwise specify each `.safetensors` path. `--vae` is usually unnecessary unless you use a different VAE.

#### SD3/3.5 Training Parameters / SD3/3.5 学習パラメータ

* `--t5xxl_max_token_length=<integer>` – Max token length for T5-XXL. Default `256`.
* `--apply_lg_attn_mask` – Apply an attention mask to CLIP-L/CLIP-G outputs.
* `--apply_t5_attn_mask` – Apply an attention mask to T5-XXL outputs.
* `--clip_l_dropout_rate`, `--clip_g_dropout_rate`, `--t5_dropout_rate` – Dropout rates for the text encoders. Default `0.0`.
* `--pos_emb_random_crop_rate=<float>` **[SD3.5]** – Probability of randomly cropping the positional embedding.
* `--enable_scaled_pos_embed` **[SD3.5][experimental]** – Scale positional embeddings when training with multiple resolutions.
* `--training_shift=<float>` – Shift applied to the timestep distribution. Default `1.0`.
* `--weighting_scheme=<choice>` – Weighting method for loss by timestep. Default `uniform`.
* `--logit_mean=<float>` – Mean value for `logit_normal` weighting scheme. Default `0.0`.
* `--logit_std=<float>` – Standard deviation for `logit_normal` weighting scheme. Default `1.0`.
* `--mode_scale=<float>` – Scale factor for `mode` weighting scheme. Default `1.29`.

#### Memory and Speed / メモリ・速度関連

* `--blocks_to_swap=<integer>` **[experimental]** – Swap a number of Transformer blocks between CPU and GPU. More blocks reduce VRAM but slow training. Cannot be used with `--cpu_offload_checkpointing`.
* `--cache_text_encoder_outputs` – Caches the outputs of the text encoders to reduce VRAM usage and speed up training. This is particularly effective for SD3, which uses three text encoders. Recommended when not training the text encoder LoRA. For more details, see the [`sdxl_train_network.py` guide](sdxl_train_network.md).
* `--cache_text_encoder_outputs_to_disk` – Caches the text encoder outputs to disk when the above option is enabled.
* `--t5xxl_device=<device>` **[not supported yet]** – Specifies the device for T5-XXL model. If not specified, uses accelerator's device.
* `--t5xxl_dtype=<dtype>` **[not supported yet]** – Specifies the dtype for T5-XXL model. If not specified, uses default dtype from mixed precision.
* `--save_clip` **[not supported yet]** – Saves CLIP models to checkpoint (unified checkpoint format not yet supported).
* `--save_t5xxl` **[not supported yet]** – Saves T5-XXL model to checkpoint (unified checkpoint format not yet supported).

#### Incompatible or Deprecated Options / 非互換・非推奨の引数

* `--v2`, `--v_parameterization`, `--clip_skip` – Options for Stable Diffusion v1/v2 that are not used for SD3/3.5.

<details>
<summary>日本語</summary>

[`train_network.py`のガイド](train_network.md)で説明されている引数に加え、以下のSD3/3.5特有の引数を指定します。共通の引数については、上記ガイドを参照してください。

#### モデル関連

*   `--pretrained_model_name_or_path="<path to SD3 model>"` **[必須]**
    *   学習のベースとなるSD3/3.5モデルの`.safetensors`ファイルのパスを指定します。
*   `--clip_l`, `--clip_g`, `--t5xxl`, `--vae`:
    *   ベースモデルが単一ファイル形式の場合、これらの指定は不要です（自動的にモデル内部から読み込まれます）。
    *   Text Encoderが別ファイルとして提供されている場合は、それぞれの`.safetensors`ファイルのパスを指定します。`--vae` はベースモデルに含まれているため、通常は指定する必要はありません（明示的に異なるVAEを使用する場合のみ指定）。

#### SD3/3.5 学習パラメータ

*   `--t5xxl_max_token_length=<integer>` – T5-XXLで使用するトークンの最大長を指定します。デフォルトは`256`です。
*   `--apply_lg_attn_mask` – CLIP-L/CLIP-Gの出力にパディング用のマスクを適用します。
*   `--apply_t5_attn_mask` – T5-XXLの出力にパディング用のマスクを適用します。
*   `--clip_l_dropout_rate`, `--clip_g_dropout_rate`, `--t5_dropout_rate` – 各Text Encoderのドロップアウト率を指定します。デフォルトは`0.0`です。
*   `--pos_emb_random_crop_rate=<float>` **[SD3.5向け]** – Positional Embeddingにランダムクロップを適用する確率を指定します。
*   `--enable_scaled_pos_embed` **[SD3.5向け][実験的機能]** – マルチ解像度学習時に解像度に応じてPositional Embeddingをスケーリングします。
*   `--training_shift=<float>` – タイムステップ分布を調整するためのシフト値です。デフォルトは`1.0`です。
*   `--weighting_scheme=<choice>` – タイムステップに応じた損失の重み付け方法を指定します。デフォルトは`uniform`です。
*   `--logit_mean=<float>` – `logit_normal`重み付けスキームの平均値です。デフォルトは`0.0`です。
*   `--logit_std=<float>` – `logit_normal`重み付けスキームの標準偏差です。デフォルトは`1.0`です。
*   `--mode_scale=<float>` – `mode`重み付けスキームのスケール係数です。デフォルトは`1.29`です。

#### メモリ・速度関連

*   `--blocks_to_swap=<integer>` **[実験的機能]** – TransformerブロックをCPUとGPUでスワップしてVRAMを節約します。`--cpu_offload_checkpointing`とは併用できません。
*   `--cache_text_encoder_outputs` – Text Encoderの出力をキャッシュし、VRAM使用量削減と学習高速化を図ります。SD3は3つのText Encoderを持つため特に効果的です。Text EncoderのLoRAを学習しない場合に推奨されます。詳細は[`sdxl_train_network.py`のガイド](sdxl_train_network.md)を参照してください。
*   `--cache_text_encoder_outputs_to_disk` – 上記オプションと併用し、Text Encoderの出力をディスクにキャッシュします。
*   `--t5xxl_device=<device>` **[未サポート]** – T5-XXLモデルのデバイスを指定します。指定しない場合はacceleratorのデバイスを使用します。
*   `--t5xxl_dtype=<dtype>` **[未サポート]** – T5-XXLモデルのdtypeを指定します。指定しない場合はデフォルトのdtype（mixed precisionから）を使用します。
*   `--save_clip` **[未サポート]** – CLIPモデルをチェックポイントに保存します（統合チェックポイント形式は未サポート）。
*   `--save_t5xxl` **[未サポート]** – T5-XXLモデルをチェックポイントに保存します（統合チェックポイント形式は未サポート）。

#### 非互換・非推奨の引数

*   `--v2`, `--v_parameterization`, `--clip_skip` – Stable Diffusion v1/v2向けの引数のため、SD3/3.5学習では使用されません。

</details>

### 4.2. Starting Training / 学習の開始

After setting the required arguments, run the command to begin training. The overall flow and how to check logs are the same as in the [train_network.py guide](train_network.md#32-starting-the-training--学習の開始).

<details>
<summary>日本語</summary>

必要な引数を設定したら、コマンドを実行して学習を開始します。全体の流れやログの確認方法は、[train_network.pyのガイド](train_network.md#32-starting-the-training--学習の開始)と同様です。

</details>

## 5. LoRA Target Modules / LoRAの学習対象モジュール

When training LoRA with `sd3_train_network.py`, the following modules are targeted by default:

*   **MMDiT (replaces U-Net)**:
    *   `qkv` (Query, Key, Value) matrices and `proj_out` (output projection) in the attention blocks.
*   **final_layer**:
    *   The output layer at the end of MMDiT.

By using `--network_args`, you can apply more detailed controls, such as setting different ranks (dimensions) for each module.

### Specify rank for each layer in SD3 LoRA / 各層のランクを指定する

You can specify the rank for each layer in SD3 by specifying the following network_args. If you specify `0`, LoRA will not be applied to that layer.

When network_args is not specified, the default value (`network_dim`) is applied, same as before.

|network_args|target layer|
|---|---|
|context_attn_dim|attn in context_block|
|context_mlp_dim|mlp in context_block|
|context_mod_dim|adaLN_modulation in context_block|
|x_attn_dim|attn in x_block|
|x_mlp_dim|mlp in x_block|
|x_mod_dim|adaLN_modulation in x_block|

`"verbose=True"` is also available for debugging. It shows the rank of each layer.

example: 
```
--network_args "context_attn_dim=2" "context_mlp_dim=3" "context_mod_dim=4" "x_attn_dim=5" "x_mlp_dim=6" "x_mod_dim=7" "verbose=True"
```

You can apply LoRA to the conditioning layers of SD3 by specifying `emb_dims` in network_args. When specifying, be sure to specify 6 numbers in `[]` as a comma-separated list.

example: 
```
--network_args "emb_dims=[2,3,4,5,6,7]"
```

Each number corresponds to `context_embedder`, `t_embedder`, `x_embedder`, `y_embedder`, `final_layer_adaLN_modulation`, `final_layer_linear`. The above example applies LoRA to all conditioning layers, with rank 2 for `context_embedder`, 3 for `t_embedder`, 4 for `context_embedder`, 5 for `y_embedder`, 6 for `final_layer_adaLN_modulation`, and 7 for `final_layer_linear`.

If you specify `0`, LoRA will not be applied to that layer. For example, `[4,0,0,4,0,0]` applies LoRA only to `context_embedder` and `y_embedder`.

### Specify blocks to train in SD3 LoRA training

You can specify the blocks to train in SD3 LoRA training by specifying `train_block_indices` in network_args. The indices are 0-based. The default (when omitted) is to train all blocks. The indices are specified as a list of integers or a range of integers, like `0,1,5,8` or `0,1,4-5,7`. 

The number of blocks depends on the model. The valid range is 0-(the number of blocks - 1). `all` is also available to train all blocks, `none` is also available to train no blocks.

example: 
```
--network_args "train_block_indices=1,2,6-8" 
```

<details>
<summary>日本語</summary>

`sd3_train_network.py`でLoRAを学習させる場合、デフォルトでは以下のモジュールが対象となります。

*   **MMDiT (U-Netの代替)**:
    *   Attentionブロック内の`qkv`（Query, Key, Value）行列と、`proj_out`（出力Projection）。
*   **final_layer**:
    *   MMDiTの最後にある出力層。

`--network_args` を使用することで、モジュールごとに異なるランク（次元数）を設定するなど、より詳細な制御が可能です。

### SD3 LoRAで各層のランクを指定する

各層のランクを指定するには、`--network_args`オプションを使用します。`0`を指定すると、その層にはLoRAが適用されません。

network_argsが指定されない場合、デフォルト値（`network_dim`）が適用されます。

|network_args|target layer|
|---|---|
|context_attn_dim|attn in context_block|
|context_mlp_dim|mlp in context_block|
|context_mod_dim|adaLN_modulation in context_block|
|x_attn_dim|attn in x_block|
|x_mlp_dim|mlp in x_block|
|x_mod_dim|adaLN_modulation in x_block|

`"verbose=True"`を指定すると、各層のランクが表示されます。

例：

```bash
--network_args "context_attn_dim=2" "context_mlp_dim=3" "context_mod_dim=4" "x_attn_dim=5" "x_mlp_dim=6" "x_mod_dim=7" "verbose=True"
```

また、`emb_dims`を指定することで、SD3の条件付け層にLoRAを適用することもできます。指定する際は、必ず`[]`内にカンマ区切りで6つの数字を指定してください。

```bash
--network_args "emb_dims=[2,3,4,5,6,7]"
```

各数字は、`context_embedder`、`t_embedder`、`x_embedder`、`y_embedder`、`final_layer_adaLN_modulation`、`final_layer_linear`に対応しています。上記の例では、すべての条件付け層にLoRAを適用し、`context_embedder`に2、`t_embedder`に3、`x_embedder`に4、`y_embedder`に5、`final_layer_adaLN_modulation`に6、`final_layer_linear`に7のランクを設定しています。

`0`を指定すると、その層にはLoRAが適用されません。例えば、`[4,0,0,4,0,0]`と指定すると、`context_embedder`と`y_embedder`のみにLoRAが適用されます。

</details>


## 6. Using the Trained Model / 学習済みモデルの利用

When training finishes, a LoRA model file (e.g. `my_sd3_lora.safetensors`) is saved in the directory specified by `output_dir`. Use this file with inference environments that support SD3/3.5, such as ComfyUI.

<details>
<summary>日本語</summary>

学習が完了すると、指定した`output_dir`にLoRAモデルファイル（例: `my_sd3_lora.safetensors`）が保存されます。このファイルは、SD3/3.5モデルに対応した推論環境（例: ComfyUIなど）で使用できます。

</details>


## 7. Others / その他

`sd3_train_network.py` shares many features with `train_network.py`, such as sample image generation (`--sample_prompts`, etc.) and detailed optimizer settings. For these, see the [train_network.py guide](train_network.md#5-other-features--その他の機能) or run `python sd3_train_network.py --help`.

<details>
<summary>日本語</summary>

`sd3_train_network.py`には、サンプル画像の生成 (`--sample_prompts`など) や詳細なオプティマイザ設定など、`train_network.py`と共通の機能も多く存在します。これらについては、[`train_network.py`のガイド](train_network.md#5-other-features--その他の機能)やスクリプトのヘルプ (`python sd3_train_network.py --help`) を参照してください。

</details>
