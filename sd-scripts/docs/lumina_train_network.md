# LoRA Training Guide for Lumina Image 2.0 using `lumina_train_network.py` / `lumina_train_network.py` を用いたLumina Image 2.0モデルのLoRA学習ガイド

This document explains how to train LoRA (Low-Rank Adaptation) models for Lumina Image 2.0 using `lumina_train_network.py` in the `sd-scripts` repository.

## 1. Introduction / はじめに

`lumina_train_network.py` trains additional networks such as LoRA for Lumina Image 2.0 models. Lumina Image 2.0 adopts a Next-DiT (Next-generation Diffusion Transformer) architecture, which differs from previous Stable Diffusion models. It uses a single text encoder (Gemma2) and a dedicated AutoEncoder (AE).

This guide assumes you already understand the basics of LoRA training. For common usage and options, see [the train_network.py guide](./train_network.md). Some parameters are similar to those in [`sd3_train_network.py`](sd3_train_network.md) and [`flux_train_network.py`](flux_train_network.md).

**Prerequisites:**

* The `sd-scripts` repository has been cloned and the Python environment is ready.
* A training dataset has been prepared. See the [Dataset Configuration Guide](./config_README-en.md).
* Lumina Image 2.0 model files for training are available.

<details>
<summary>日本語</summary>

`lumina_train_network.py`は、Lumina Image 2.0モデルに対してLoRAなどの追加ネットワークを学習させるためのスクリプトです。Lumina Image 2.0は、Next-DiT (Next-generation Diffusion Transformer) と呼ばれる新しいアーキテクチャを採用しており、従来のStable Diffusionモデルとは構造が異なります。テキストエンコーダーとしてGemma2を単体で使用し、専用のAutoEncoder (AE) を使用します。

このガイドは、基本的なLoRA学習の手順を理解しているユーザーを対象としています。基本的な使い方や共通のオプションについては、`train_network.py`のガイド（作成中）を参照してください。また一部のパラメータは [`sd3_train_network.py`](sd3_train_network.md) や [`flux_train_network.py`](flux_train_network.md) と同様のものがあるため、そちらも参考にしてください。

**前提条件:**

*   `sd-scripts`リポジトリのクローンとPython環境のセットアップが完了していること。
*   学習用データセットの準備が完了していること。（データセットの準備については[データセット設定ガイド](./config_README-en.md)を参照してください）
*   学習対象のLumina Image 2.0モデルファイルが準備できていること。
</details>

## 2. Differences from `train_network.py` / `train_network.py` との違い

`lumina_train_network.py` is based on `train_network.py` but modified for Lumina Image 2.0. Main differences are:

* **Target models:** Lumina Image 2.0 models.
* **Model structure:** Uses Next-DiT (Transformer based) instead of U-Net and employs a single text encoder (Gemma2). The AutoEncoder (AE) is not compatible with SDXL/SD3/FLUX.
* **Arguments:** Options exist to specify the Lumina Image 2.0 model, Gemma2 text encoder and AE. With a single `.safetensors` file, these components are typically provided separately.
* **Incompatible arguments:** Stable Diffusion v1/v2 options such as `--v2`, `--v_parameterization` and `--clip_skip` are not used.
* **Lumina specific options:** Additional parameters for timestep sampling, model prediction type, discrete flow shift, and system prompt.

<details>
<summary>日本語</summary>
`lumina_train_network.py`は`train_network.py`をベースに、Lumina Image 2.0モデルに対応するための変更が加えられています。主な違いは以下の通りです。

*   **対象モデル:** Lumina Image 2.0モデルを対象とします。
*   **モデル構造:** U-Netの代わりにNext-DiT (Transformerベース) を使用します。Text EncoderとしてGemma2を単体で使用し、専用のAutoEncoder (AE) を使用します。
*   **引数:** Lumina Image 2.0モデル、Gemma2 Text Encoder、AEを指定する引数があります。通常、これらのコンポーネントは個別に提供されます。
*   **一部引数の非互換性:** Stable Diffusion v1/v2向けの引数（例: `--v2`, `--v_parameterization`, `--clip_skip`）はLumina Image 2.0の学習では使用されません。
*   **Lumina特有の引数:** タイムステップのサンプリング、モデル予測タイプ、離散フローシフト、システムプロンプトに関する引数が追加されています。
</details>

## 3. Preparation / 準備

The following files are required before starting training:

1. **Training script:** `lumina_train_network.py`
2. **Lumina Image 2.0 model file:** `.safetensors` file for the base model.
3. **Gemma2 text encoder file:** `.safetensors` file for the text encoder.
4. **AutoEncoder (AE) file:** `.safetensors` file for the AE.
5. **Dataset definition file (.toml):** Dataset settings in TOML format. (See the [Dataset Configuration Guide](./config_README-en.md). In this document we use `my_lumina_dataset_config.toml` as an example.


**Model Files:**
* Lumina Image 2.0: `lumina-image-2.safetensors` ([full precision link](https://huggingface.co/rockerBOO/lumina-image-2/blob/main/lumina-image-2.safetensors)) or `lumina_2_model_bf16.safetensors` ([bf16 link](https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/blob/main/split_files/diffusion_models/lumina_2_model_bf16.safetensors))
* Gemma2 2B (fp16): `gemma-2-2b.safetensors` ([link](https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/blob/main/split_files/text_encoders/gemma_2_2b_fp16.safetensors))
* AutoEncoder: `ae.safetensors` ([link](https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/blob/main/split_files/vae/ae.safetensors)) (same as FLUX)


<details>
<summary>日本語</summary>
学習を開始する前に、以下のファイルが必要です。

1.  **学習スクリプト:** `lumina_train_network.py`
2.  **Lumina Image 2.0モデルファイル:** 学習のベースとなるLumina Image 2.0モデルの`.safetensors`ファイル。
3.  **Gemma2テキストエンコーダーファイル:** Gemma2テキストエンコーダーの`.safetensors`ファイル。
4.  **AutoEncoder (AE) ファイル:** AEの`.safetensors`ファイル。
5.  **データセット定義ファイル (.toml):** 学習データセットの設定を記述したTOML形式のファイル。（詳細は[データセット設定ガイド](./config_README-en.md)を参照してください）。
    *   例として`my_lumina_dataset_config.toml`を使用します。

**モデルファイル** は英語ドキュメントの通りです。

</details>

## 4. Running the Training / 学習の実行

Execute `lumina_train_network.py` from the terminal to start training. The overall command-line format is the same as `train_network.py`, but Lumina Image 2.0 specific options must be supplied.

Example command:

```bash
accelerate launch --num_cpu_threads_per_process 1 lumina_train_network.py \
  --pretrained_model_name_or_path="lumina-image-2.safetensors" \
  --gemma2="gemma-2-2b.safetensors" \
  --ae="ae.safetensors" \
  --dataset_config="my_lumina_dataset_config.toml" \
  --output_dir="./output" \
  --output_name="my_lumina_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora_lumina \
  --network_dim=8 \
  --network_alpha=8 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW" \
  --lr_scheduler="constant" \
  --timestep_sampling="nextdit_shift" \
  --discrete_flow_shift=6.0 \
  --model_prediction_type="raw" \
  --system_prompt="You are an assistant designed to generate high-quality images based on user prompts." \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs
```

*(Write the command on one line or use `\` or `^` for line breaks.)*

<details>
<summary>日本語</summary>
学習は、ターミナルから`lumina_train_network.py`を実行することで開始します。基本的なコマンドラインの構造は`train_network.py`と同様ですが、Lumina Image 2.0特有の引数を指定する必要があります。

以下に、基本的なコマンドライン実行例を示します。

```bash
accelerate launch --num_cpu_threads_per_process 1 lumina_train_network.py \
  --pretrained_model_name_or_path="lumina-image-2.safetensors" \
  --gemma2="gemma-2-2b.safetensors" \
  --ae="ae.safetensors" \
  --dataset_config="my_lumina_dataset_config.toml" \
  --output_dir="./output" \
  --output_name="my_lumina_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora_lumina \
  --network_dim=8 \
  --network_alpha=8 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW" \
  --lr_scheduler="constant" \
  --timestep_sampling="nextdit_shift" \
  --discrete_flow_shift=6.0 \
  --model_prediction_type="raw" \
  --system_prompt="You are an assistant designed to generate high-quality images based on user prompts." \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs
```

※実際には1行で書くか、適切な改行文字（`\` または `^`）を使用してください。
</details>

### 4.1. Explanation of Key Options / 主要なコマンドライン引数の解説

Besides the arguments explained in the [train_network.py guide](train_network.md), specify the following Lumina Image 2.0 options. For shared options (`--output_dir`, `--output_name`, etc.), see that guide.

#### Model Options / モデル関連

* `--pretrained_model_name_or_path="<path to Lumina model>"` **required** – Path to the Lumina Image 2.0 model.
* `--gemma2="<path to Gemma2 model>"` **required** – Path to the Gemma2 text encoder `.safetensors` file.
* `--ae="<path to AE model>"` **required** – Path to the AutoEncoder `.safetensors` file.

#### Lumina Image 2.0 Training Parameters / Lumina Image 2.0 学習パラメータ

* `--gemma2_max_token_length=<integer>` – Max token length for Gemma2. Default is 256.
* `--timestep_sampling=<choice>` – Timestep sampling method. Options: `sigma`, `uniform`, `sigmoid`, `shift`, `nextdit_shift`. Default `shift`. **Recommended: `nextdit_shift`**
* `--discrete_flow_shift=<float>` – Discrete flow shift for the Euler Discrete Scheduler. Default `6.0`.
* `--model_prediction_type=<choice>` – Model prediction processing method. Options: `raw`, `additive`, `sigma_scaled`. Default `raw`. **Recommended: `raw`**
* `--system_prompt=<string>` – System prompt to prepend to all prompts. Recommended: `"You are an assistant designed to generate high-quality images based on user prompts."` or `"You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts."`
* `--use_flash_attn` – Use Flash Attention. Requires `pip install flash-attn` (may not be supported in all environments). If installed correctly, it speeds up training. 
* `--use_sage_attn` – Use Sage Attention for the model.
* `--sample_batch_size=<integer>` – Batch size to use for sampling, defaults to `--training_batch_size` value. Sample batches are bucketed by width, height, guidance scale, and seed.
* `--sigmoid_scale=<float>` – Scale factor for sigmoid timestep sampling. Default `1.0`.

#### Memory and Speed / メモリ・速度関連

* `--blocks_to_swap=<integer>` **[experimental]** – Swap a number of Transformer blocks between CPU and GPU. More blocks reduce VRAM but slow training. Cannot be used with `--cpu_offload_checkpointing`.
* `--cache_text_encoder_outputs` – Cache Gemma2 outputs to reduce memory usage.
* `--cache_latents`, `--cache_latents_to_disk` – Cache AE outputs.
* `--fp8_base` – Use FP8 precision for the base model.

#### Network Arguments / ネットワーク引数

For Lumina Image 2.0, you can specify different dimensions for various components:

* `--network_args` can include:
  * `"attn_dim=4"` – Attention dimension
  * `"mlp_dim=4"` – MLP dimension  
  * `"mod_dim=4"` – Modulation dimension
  * `"refiner_dim=4"` – Refiner blocks dimension
  * `"embedder_dims=[4,4,4]"` – Embedder dimensions for x, t, and caption embedders

#### Incompatible or Deprecated Options / 非互換・非推奨の引数

* `--v2`, `--v_parameterization`, `--clip_skip` – Options for Stable Diffusion v1/v2 that are not used for Lumina Image 2.0.

<details>
<summary>日本語</summary>

[`train_network.py`のガイド](train_network.md)で説明されている引数に加え、以下のLumina Image 2.0特有の引数を指定します。共通の引数については、上記ガイドを参照してください。

#### モデル関連

*   `--pretrained_model_name_or_path="<path to Lumina model>"` **[必須]**
    *   学習のベースとなるLumina Image 2.0モデルの`.safetensors`ファイルのパスを指定します。
*   `--gemma2="<path to Gemma2 model>"` **[必須]**
    *   Gemma2テキストエンコーダーの`.safetensors`ファイルのパスを指定します。
*   `--ae="<path to AE model>"` **[必須]**
    *   AutoEncoderの`.safetensors`ファイルのパスを指定します。

#### Lumina Image 2.0 学習パラメータ

*   `--gemma2_max_token_length=<integer>` – Gemma2で使用するトークンの最大長を指定します。デフォルトは256です。
*   `--timestep_sampling=<choice>` – タイムステップのサンプリング方法を指定します。`sigma`, `uniform`, `sigmoid`, `shift`, `nextdit_shift`から選択します。デフォルトは`shift`です。**推奨: `nextdit_shift`**
*   `--discrete_flow_shift=<float>` – Euler Discrete Schedulerの離散フローシフトを指定します。デフォルトは`6.0`です。
*   `--model_prediction_type=<choice>` – モデル予測の処理方法を指定します。`raw`, `additive`, `sigma_scaled`から選択します。デフォルトは`raw`です。**推奨: `raw`**
*   `--system_prompt=<string>` – 全てのプロンプトに前置するシステムプロンプトを指定します。推奨: `"You are an assistant designed to generate high-quality images based on user prompts."` または `"You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts."`
*   `--use_flash_attn` – Flash Attentionを使用します。`pip install flash-attn`でインストールが必要です（環境によってはサポートされていません）。正しくインストールされている場合は、指定すると学習が高速化されます。
*   `--use_sage_attn` – Sage Attentionを使用します。
*   `--sample_batch_size=<integer>` – サンプリングに使用するバッチサイズ。デフォルトは `--training_batch_size` の値です。サンプルバッチは、幅、高さ、ガイダンススケール、シードによってバケット化されます。
*   `--sigmoid_scale=<float>` – sigmoidタイムステップサンプリングのスケール係数を指定します。デフォルトは`1.0`です。

#### メモリ・速度関連

*   `--blocks_to_swap=<integer>` **[実験的機能]** – TransformerブロックをCPUとGPUでスワップしてVRAMを節約します。`--cpu_offload_checkpointing`とは併用できません。
*   `--cache_text_encoder_outputs` – Gemma2の出力をキャッシュしてメモリ使用量を削減します。
*   `--cache_latents`, `--cache_latents_to_disk` – AEの出力をキャッシュします。
*   `--fp8_base` – ベースモデルにFP8精度を使用します。

#### ネットワーク引数

Lumina Image 2.0では、各コンポーネントに対して異なる次元を指定できます：

*   `--network_args` には以下を含めることができます：
    *   `"attn_dim=4"` – アテンション次元
    *   `"mlp_dim=4"` – MLP次元
    *   `"mod_dim=4"` – モジュレーション次元
    *   `"refiner_dim=4"` – リファイナーブロック次元
    *   `"embedder_dims=[4,4,4]"` – x、t、キャプションエンベッダーのエンベッダー次元

#### 非互換・非推奨の引数

*   `--v2`, `--v_parameterization`, `--clip_skip` – Stable Diffusion v1/v2向けの引数のため、Lumina Image 2.0学習では使用されません。
</details>

### 4.2. Starting Training / 学習の開始

After setting the required arguments, run the command to begin training. The overall flow and how to check logs are the same as in the [train_network.py guide](train_network.md#32-starting-the-training--学習の開始).

## 5. Using the Trained Model / 学習済みモデルの利用

When training finishes, a LoRA model file (e.g. `my_lumina_lora.safetensors`) is saved in the directory specified by `output_dir`. Use this file with inference environments that support Lumina Image 2.0, such as ComfyUI with appropriate nodes.

### Inference with scripts in this repository / このリポジトリのスクリプトを使用した推論

The inference script is also available. The script is `lumina_minimal_inference.py`. See `--help` for options. 

```
python lumina_minimal_inference.py --pretrained_model_name_or_path path/to/lumina.safetensors  --gemma2_path path/to/gemma.safetensors" --ae_path  path/to/flux_ae.safetensors  --output_dir path/to/output_dir --offload --seed 1234 --prompt "Positive prompt" --system_prompt "You are an assistant designed to generate high-quality images based on user prompts."  --negative_prompt "negative prompt"  
```

`--add_system_prompt_to_negative_prompt` option can be used to add the system prompt to the negative prompt.

`--lora_weights` option can be used to specify the LoRA weights file, and optional multiplier (like `path;1.0`).

## 6. Others / その他

`lumina_train_network.py` shares many features with `train_network.py`, such as sample image generation (`--sample_prompts`, etc.) and detailed optimizer settings. For these, see the [train_network.py guide](train_network.md#5-other-features--その他の機能) or run `python lumina_train_network.py --help`.

### 6.1. Recommended Settings / 推奨設定

Based on the contributor's recommendations, here are the suggested settings for optimal training:

**Key Parameters:**
* `--timestep_sampling="nextdit_shift"`
* `--discrete_flow_shift=6.0`
* `--model_prediction_type="raw"`
* `--mixed_precision="bf16"`

**System Prompts:**
* General purpose: `"You are an assistant designed to generate high-quality images based on user prompts."`
* High image-text alignment: `"You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts."`

**Sample Prompts:**
Sample prompts can include CFG truncate (`--ctr`) and Renorm CFG (`-rcfg`) parameters:
* `--ctr 0.25 --rcfg 1.0` (default values)

<details>
<summary>日本語</summary>

必要な引数を設定し、コマンドを実行すると学習が開始されます。基本的な流れやログの確認方法は[`train_network.py`のガイド](train_network.md#32-starting-the-training--学習の開始)と同様です。

学習が完了すると、指定した`output_dir`にLoRAモデルファイル（例: `my_lumina_lora.safetensors`）が保存されます。このファイルは、Lumina Image 2.0モデルに対応した推論環境（例: ComfyUI + 適切なノード）で使用できます。

当リポジトリ内の推論スクリプトを用いて推論することも可能です。スクリプトは`lumina_minimal_inference.py`です。オプションは`--help`で確認できます。記述例は英語版のドキュメントをご確認ください。

`lumina_train_network.py`には、サンプル画像の生成 (`--sample_prompts`など) や詳細なオプティマイザ設定など、`train_network.py`と共通の機能も多く存在します。これらについては、[`train_network.py`のガイド](train_network.md#5-other-features--その他の機能)やスクリプトのヘルプ (`python lumina_train_network.py --help`) を参照してください。

### 6.1. 推奨設定

コントリビューターの推奨に基づく、最適な学習のための推奨設定：

**主要パラメータ:**
* `--timestep_sampling="nextdit_shift"`
* `--discrete_flow_shift=6.0`
* `--model_prediction_type="raw"`
* `--mixed_precision="bf16"`

**システムプロンプト:**
* 汎用目的: `"You are an assistant designed to generate high-quality images based on user prompts."`
* 高い画像-テキスト整合性: `"You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts."`

**サンプルプロンプト:**
サンプルプロンプトには CFG truncate (`--ctr`) と Renorm CFG (`--rcfg`) パラメータを含めることができます：
* `--ctr 0.25 --rcfg 1.0` (デフォルト値)

</details>