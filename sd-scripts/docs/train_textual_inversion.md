# How to use Textual Inversion training scripts / Textual Inversion学習スクリプトの使い方

This document explains how to train Textual Inversion embeddings using the `train_textual_inversion.py` and `sdxl_train_textual_inversion.py` scripts included in the `sd-scripts` repository.

<details>
<summary>日本語</summary>
このドキュメントでは、`sd-scripts` リポジトリに含まれる `train_textual_inversion.py` および `sdxl_train_textual_inversion.py` を使用してTextual Inversionの埋め込みを学習する方法について解説します。
</details>

## 1. Introduction / はじめに

[Textual Inversion](https://textual-inversion.github.io/) is a technique that teaches Stable Diffusion new concepts by learning new token embeddings. Instead of fine-tuning the entire model, it only optimizes the text encoder's token embeddings, making it a lightweight approach to teaching the model specific characters, objects, or artistic styles.

**Available Scripts:**
- `train_textual_inversion.py`: For Stable Diffusion v1.x and v2.x models
- `sdxl_train_textual_inversion.py`: For Stable Diffusion XL models

**Prerequisites:**
* The `sd-scripts` repository has been cloned and the Python environment has been set up.
* The training dataset has been prepared. For dataset preparation, please refer to the [Dataset Configuration Guide](config_README-en.md).

<details>
<summary>日本語</summary>

[Textual Inversion](https://textual-inversion.github.io/) は、新しいトークンの埋め込みを学習することで、Stable Diffusionに新しい概念を教える技術です。モデル全体をファインチューニングする代わりに、テキストエンコーダのトークン埋め込みのみを最適化するため、特定のキャラクター、オブジェクト、芸術的スタイルをモデルに教えるための軽量なアプローチです。

**利用可能なスクリプト:**
- `train_textual_inversion.py`: Stable Diffusion v1.xおよびv2.xモデル用
- `sdxl_train_textual_inversion.py`: Stable Diffusion XLモデル用

**前提条件:**
* `sd-scripts` リポジトリのクローンとPython環境のセットアップが完了していること。
* 学習用データセットの準備が完了していること。データセットの準備については[データセット設定ガイド](config_README-en.md)を参照してください。
</details>

## 2. Basic Usage / 基本的な使用方法

### 2.1. For Stable Diffusion v1.x/v2.x Models / Stable Diffusion v1.x/v2.xモデル用

```bash
accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py \
  --pretrained_model_name_or_path="path/to/model.safetensors" \
  --dataset_config="dataset_config.toml" \
  --output_dir="output" \
  --output_name="my_textual_inversion" \
  --save_model_as="safetensors" \
  --token_string="mychar" \
  --init_word="girl" \
  --num_vectors_per_token=4 \
  --max_train_steps=1600 \
  --learning_rate=1e-6 \
  --optimizer_type="AdamW8bit" \
  --mixed_precision="fp16" \
  --cache_latents \
  --sdpa
```

### 2.2. For SDXL Models / SDXLモデル用

```bash
accelerate launch --num_cpu_threads_per_process 1 sdxl_train_textual_inversion.py \
  --pretrained_model_name_or_path="path/to/sdxl_model.safetensors" \
  --dataset_config="dataset_config.toml" \
  --output_dir="output" \
  --output_name="my_sdxl_textual_inversion" \
  --save_model_as="safetensors" \
  --token_string="mychar" \
  --init_word="girl" \
  --num_vectors_per_token=4 \
  --max_train_steps=1600 \
  --learning_rate=1e-6 \
  --optimizer_type="AdamW8bit" \
  --mixed_precision="fp16" \
  --cache_latents \
  --sdpa
```

<details>
<summary>日本語</summary>
上記のコマンドは実際には1行で書く必要がありますが、見やすさのために改行しています（LinuxやMacでは行末に `\` を追加することで改行できます）。Windowsの場合は、改行せずに1行で書くか、`^` を行末に追加してください。
</details>

## 3. Key Command-Line Arguments / 主要なコマンドライン引数

### 3.1. Textual Inversion Specific Arguments / Textual Inversion固有の引数

#### Core Parameters / コアパラメータ

* `--token_string="mychar"` **[Required]**
  * Specifies the token string used in training. This must not exist in the tokenizer's vocabulary. In your training prompts, include this token string (e.g., if token_string is "mychar", use prompts like "mychar 1girl").
  * 学習時に使用されるトークン文字列を指定します。tokenizerの語彙に存在しない文字である必要があります。学習時のプロンプトには、このトークン文字列を含める必要があります（例：token_stringが"mychar"なら、"mychar 1girl"のようなプロンプトを使用）。

* `--init_word="girl"`
  * Specifies the word to use for initializing the embedding vector. Choose a word that is conceptually close to what you want to teach. Must be a single token.
  * 埋め込みベクトルの初期化に使用する単語を指定します。教えたい概念に近い単語を選ぶとよいでしょう。単一のトークンである必要があります。

* `--num_vectors_per_token=4`
  * Specifies how many embedding vectors to use for this token. More vectors provide greater expressiveness but consume more tokens from the 77-token limit.
  * このトークンに使用する埋め込みベクトルの数を指定します。多いほど表現力が増しますが、77トークン制限からより多くのトークンを消費します。

* `--weights="path/to/existing_embedding.safetensors"`
  * Loads pre-trained embeddings to continue training from. Optional parameter for transfer learning.
  * 既存の埋め込みを読み込んで、そこから追加で学習します。転移学習のオプションパラメータです。

#### Template Options / テンプレートオプション

* `--use_object_template`
  * Ignores captions and uses predefined object templates (e.g., "a photo of a {}"). Same as the original implementation.
  * キャプションを無視して、事前定義された物体用テンプレート（例："a photo of a {}"）を使用します。公式実装と同じです。

* `--use_style_template`
  * Ignores captions and uses predefined style templates (e.g., "a painting in the style of {}"). Same as the original implementation.
  * キャプションを無視して、事前定義されたスタイル用テンプレート（例："a painting in the style of {}"）を使用します。公式実装と同じです。

### 3.2. Model and Dataset Arguments / モデル・データセット引数

For common model and dataset arguments, please refer to [LoRA Training Guide](train_network.md#31-main-command-line-arguments--主要なコマンドライン引数). The following arguments work the same way:

* `--pretrained_model_name_or_path`
* `--dataset_config`
* `--v2`, `--v_parameterization`
* `--resolution`
* `--cache_latents`, `--vae_batch_size`
* `--enable_bucket`, `--min_bucket_reso`, `--max_bucket_reso`

<details>
<summary>日本語</summary>
一般的なモデル・データセット引数については、[LoRA学習ガイド](train_network.md#31-main-command-line-arguments--主要なコマンドライン引数)を参照してください。以下の引数は同様に動作します：

* `--pretrained_model_name_or_path`
* `--dataset_config`
* `--v2`, `--v_parameterization`
* `--resolution`
* `--cache_latents`, `--vae_batch_size`
* `--enable_bucket`, `--min_bucket_reso`, `--max_bucket_reso`
</details>

### 3.3. Training Parameters / 学習パラメータ

For training parameters, please refer to [LoRA Training Guide](train_network.md#31-main-command-line-arguments--主要なコマンドライン引数). Textual Inversion typically uses these settings:

* `--learning_rate=1e-6`: Lower learning rates are often used compared to LoRA training
* `--max_train_steps=1600`: Fewer steps are usually sufficient
* `--optimizer_type="AdamW8bit"`: Memory-efficient optimizer
* `--mixed_precision="fp16"`: Reduces memory usage

**Note:** Textual Inversion has lower memory requirements compared to full model fine-tuning, so you can often use larger batch sizes.

<details>
<summary>日本語</summary>
学習パラメータについては、[LoRA学習ガイド](train_network.md#31-main-command-line-arguments--主要なコマンドライン引数)を参照してください。Textual Inversionでは通常以下の設定を使用します：

* `--learning_rate=1e-6`: LoRA学習と比べて低い学習率がよく使用されます
* `--max_train_steps=1600`: より少ないステップで十分な場合が多いです
* `--optimizer_type="AdamW8bit"`: メモリ効率的なオプティマイザ
* `--mixed_precision="fp16"`: メモリ使用量を削減

**注意:** Textual Inversionはモデル全体のファインチューニングと比べてメモリ要件が低いため、多くの場合、より大きなバッチサイズを使用できます。
</details>

## 4. Dataset Preparation / データセット準備

### 4.1. Dataset Configuration / データセット設定

Create a TOML configuration file as described in the [Dataset Configuration Guide](config_README-en.md). Here's an example for Textual Inversion:

```toml
[general]
shuffle_caption = false
caption_extension = ".txt"
keep_tokens = 1

[[datasets]]
resolution = 512                    # 1024 for SDXL
batch_size = 4                      # Can use larger values than LoRA training
enable_bucket = true

  [[datasets.subsets]]
  image_dir = "path/to/images"
  caption_extension = ".txt"
  num_repeats = 10
```

### 4.2. Caption Guidelines / キャプションガイドライン

**Important:** Your captions must include the token string you specified. For example:

* If `--token_string="mychar"`, captions should be like: "mychar, 1girl, blonde hair, blue eyes"
* The token string can appear anywhere in the caption, but including it is essential

You can verify that your token string is being recognized by using `--debug_dataset`, which will show token IDs. Look for tokens with IDs ≥ 49408 (these are the new custom tokens).

<details>
<summary>日本語</summary>

**重要:** キャプションには指定したトークン文字列を含める必要があります。例：

* `--token_string="mychar"` の場合、キャプションは "mychar, 1girl, blonde hair, blue eyes" のようにします
* トークン文字列はキャプション内のどこに配置しても構いませんが、含めることが必須です

`--debug_dataset` を使用してトークン文字列が認識されているかを確認できます。これによりトークンIDが表示されます。ID ≥ 49408 のトークン（これらは新しいカスタムトークン）を探してください。
</details>

## 5. Advanced Configuration / 高度な設定

### 5.1. Multiple Token Vectors / 複数トークンベクトル

When using `--num_vectors_per_token` > 1, the system creates additional token variations:
- `--token_string="mychar"` with `--num_vectors_per_token=4` creates: "mychar", "mychar1", "mychar2", "mychar3"

For generation, you can use either the base token or all tokens together.

### 5.2. Memory Optimization / メモリ最適化

* Use `--cache_latents` to cache VAE outputs and reduce VRAM usage
* Use `--gradient_checkpointing` for additional memory savings
* For SDXL, use `--cache_text_encoder_outputs` to cache text encoder outputs
* Consider using `--mixed_precision="bf16"` on newer GPUs (RTX 30 series and later)

### 5.3. Training Tips / 学習のコツ

* **Learning Rate:** Start with 1e-6 and adjust based on results. Lower rates often work better than LoRA training.
* **Steps:** 1000-2000 steps are usually sufficient, but this varies by dataset size and complexity.
* **Batch Size:** Textual Inversion can handle larger batch sizes than full fine-tuning due to lower memory requirements.
* **Templates:** Use `--use_object_template` for characters/objects, `--use_style_template` for artistic styles.

<details>
<summary>日本語</summary>

* **学習率:** 1e-6から始めて、結果に基づいて調整してください。LoRA学習よりも低い率がよく機能します。
* **ステップ数:** 通常1000-2000ステップで十分ですが、データセットのサイズと複雑さによって異なります。
* **バッチサイズ:** メモリ要件が低いため、Textual Inversionは完全なファインチューニングよりも大きなバッチサイズを処理できます。
* **テンプレート:** キャラクター/オブジェクトには `--use_object_template`、芸術的スタイルには `--use_style_template` を使用してください。
</details>

## 6. Usage After Training / 学習後の使用方法

The trained Textual Inversion embeddings can be used in:

* **Automatic1111 WebUI:** Place the `.safetensors` file in the `embeddings` folder
* **ComfyUI:** Use the embedding file with appropriate nodes
* **Other Diffusers-based applications:** Load using the embedding path

In your prompts, simply use the token string you trained (e.g., "mychar") and the model will use the learned embedding.

<details>
<summary>日本語</summary>

学習したTextual Inversionの埋め込みは以下で使用できます：

* **Automatic1111 WebUI:** `.safetensors` ファイルを `embeddings` フォルダに配置
* **ComfyUI:** 適切なノードで埋め込みファイルを使用
* **その他のDiffusersベースアプリケーション:** 埋め込みパスを使用して読み込み

プロンプトでは、学習したトークン文字列（例："mychar"）を単純に使用するだけで、モデルが学習した埋め込みを使用します。
</details>

## 7. Troubleshooting / トラブルシューティング

### Common Issues / よくある問題

1. **Token string already exists in tokenizer**
   * Use a unique string that doesn't exist in the model's vocabulary
   * Try adding numbers or special characters (e.g., "mychar123")

2. **No improvement after training**
   * Ensure your captions include the token string
   * Try adjusting the learning rate (lower values like 5e-7)
   * Increase the number of training steps

   * Use `--cache_latents`

<details>
<summary>日本語</summary>

1. **トークン文字列がtokenizerに既に存在する**
   * モデルの語彙に存在しない固有の文字列を使用してください
   * 数字や特殊文字を追加してみてください（例："mychar123"）

2. **学習後に改善が見られない**
   * キャプションにトークン文字列が含まれていることを確認してください
   * 学習率を調整してみてください（5e-7のような低い値）
   * 学習ステップ数を増やしてください

3. **メモリ不足エラー**
   * データセット設定でバッチサイズを減らしてください
   * `--gradient_checkpointing` を使用してください
   * `--cache_latents` を使用してください
</details>

For additional training options and advanced configurations, please refer to the [LoRA Training Guide](train_network.md) as many parameters are shared between training methods.