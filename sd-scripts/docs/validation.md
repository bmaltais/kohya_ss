# Validation Loss

Validation loss is a crucial metric for monitoring the training process of a model. It helps you assess how well your model is generalizing to data it hasn't seen during training, which is essential for preventing overfitting. By periodically evaluating the model on a separate validation dataset, you can gain insights into its performance and make more informed decisions about when to stop training or adjust hyperparameters.

This feature provides a stable and reliable validation loss metric by ensuring the validation process is deterministic.

<details>
<summary>日本語</summary>

Validation loss（検証損失）は、モデルの学習過程を監視するための重要な指標です。モデルが学習中に見ていないデータに対してどの程度汎化できているかを評価するのに役立ち、過学習を防ぐために不可欠です。個別の検証データセットで定期的にモデルを評価することで、そのパフォーマンスに関する洞察を得て、学習をいつ停止するか、またはハイパーパラメータを調整するかについて、より多くの情報に基づいた決定を下すことができます。

この機能は、検証プロセスが決定論的であることを保証することにより、安定して信頼性の高い検証損失指標を提供します。

</details>

## How It Works

When validation is enabled, a portion of your dataset is set aside specifically for this purpose. The script then runs a validation step at regular intervals, calculating the loss on this validation data.

To ensure that the validation loss is a reliable indicator of model performance, the process is deterministic. This means that for every validation run, the same random seed is used for noise generation and timestep selection. This consistency ensures that any fluctuations in the validation loss are due to changes in the model's weights, not random variations in the validation process itself.

The average loss across all validation steps is then logged, providing a single, clear metric to track.

For more technical details, please refer to the original pull request: [PR #1903](https://github.com/kohya-ss/sd-scripts/pull/1903).

<details>
<summary>日本語</summary>

検証が有効になると、データセットの一部がこの目的のために特別に確保されます。スクリプトは定期的な間隔で検証ステップを実行し、この検証データに対する損失を計算します。

検証損失がモデルのパフォーマンスの信頼できる指標であることを保証するために、プロセスは決定論的です。つまり、すべての検証実行で、ノイズ生成とタイムステップ選択に同じランダムシードが使用されます。この一貫性により、検証損失の変動が、検証プロセス自体のランダムな変動ではなく、モデルの重みの変化によるものであることが保証されます。

すべての検証ステップにわたる平均損失がログに記録され、追跡するための単一の明確な指標が提供されます。

より技術的な詳細については、元のプルリクエストを参照してください: [PR #1903](https://github.com/kohya-ss/sd-scripts/pull/1903).

</details>

## How to Use

### Enabling Validation

There are two primary ways to enable validation:

1.  **Using a Dataset Config File (Recommended)**: You can specify a validation set directly within your dataset `.toml` file. This method offers the most control, allowing you to designate entire directories as validation sets or split a percentage of a specific subset for validation.

    To use a whole directory for validation, add a subset and set `validation_split = 1.0`.

    **Example: Separate Validation Set**
    ```toml
    [[datasets]]
      # ... training subset ...
      [[datasets.subsets]]
        image_dir = "path/to/train_images"
        # ... other settings ...

      # Validation subset
      [[datasets.subsets]]
        image_dir = "path/to/validation_images"
        validation_split = 1.0  # Use this entire subset for validation
    ```

    To use a fraction of a subset for validation, set `validation_split` to a value between 0.0 and 1.0.

    **Example: Splitting a Subset**
    ```toml
    [[datasets]]
      # ... dataset settings ...
      [[datasets.subsets]]
        image_dir = "path/to/images"
        validation_split = 0.1  # Use 10% of this subset for validation
    ```

2.  **Using a Command-Line Argument**: For a simpler setup, you can use the `--validation_split` argument. This will take a random percentage of your *entire* training dataset for validation. This method is ignored if `validation_split` is defined in your dataset config file.

    **Example Command:**
    ```bash
    accelerate launch train_network.py ... --validation_split 0.1
    ```
    This command will use 10% of the total training data for validation.

<details>
<summary>日本語</summary>

### 検証を有効にする

検証を有効にする主な方法は2つあります。

1.  **データセット設定ファイルを使用する（推奨）**: データセットの`.toml`ファイル内で直接検証セットを指定できます。この方法は最も制御性が高く、ディレクトリ全体を検証セットとして指定したり、特定のサブセットのパーセンテージを検証用に分割したりすることができます。

    ディレクトリ全体を検証に使用するには、サブセットを追加して`validation_split = 1.0`と設定します。

    **例：個別の検証セット**
    ```toml
    [[datasets]]
      # ... training subset ...
      [[datasets.subsets]]
        image_dir = "path/to/train_images"
        # ... other settings ...

      # Validation subset
      [[datasets.subsets]]
        image_dir = "path/to/validation_images"
        validation_split = 1.0  # このサブセット全体を検証に使用します
    ```

    サブセットの一部を検証に使用するには、`validation_split`を0.0から1.0の間の値に設定します。

    **例：サブセットの分割**
    ```toml
    [[datasets]]
      # ... dataset settings ...
      [[datasets.subsets]]
        image_dir = "path/to/images"
        validation_split = 0.1  # このサブセットの10%を検証に使用します
    ```

2.  **コマンドライン引数を使用する**: より簡単な設定のために、`--validation_split`引数を使用できます。これにより、*全*学習データセットのランダムなパーセンテージが検証に使用されます。この方法は、データセット設定ファイルで`validation_split`が定義されている場合は無視されます。

    **コマンド例:**
    ```bash
    accelerate launch train_network.py ... --validation_split 0.1
    ```
    このコマンドは、全学習データの10%を検証に使用します。

</details>

### Configuration Options

| Argument                    | TOML Option         | Description                                                                                                                            |
| --------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `--validation_split`        | `validation_split`  | The fraction of the dataset to use for validation. The command-line argument applies globally, while the TOML option applies per-subset. The TOML setting takes precedence. |
| `--validate_every_n_steps`  |                     | Run validation every N steps.                                                                                                          |
| `--validate_every_n_epochs` |                     | Run validation every N epochs. If not specified, validation runs once per epoch by default.                                            |
| `--max_validation_steps`    |                     | The maximum number of batches to use for a single validation run. If not set, the entire validation dataset is used.                     |
| `--validation_seed`         | `validation_seed`   | A specific seed for the validation dataloader shuffling. If not set in the TOML file, the main training `--seed` is used.                 |

<details>
<summary>日本語</summary>

### 設定オプション

| 引数                        | TOMLオプション      | 説明                                                                                                                                   |
| --------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `--validation_split`        | `validation_split`  | 検証に使用するデータセットの割合。コマンドライン引数は全体に適用され、TOMLオプションはサブセットごとに適用されます。TOML設定が優先されます。 |
| `--validate_every_n_steps`  |                     | Nステップごとに検証を実行します。                                                                                                      |
| `--validate_every_n_epochs` |                     | Nエポックごとに検証を実行します。指定しない場合、デフォルトでエポックごとに1回検証が実行されます。                                       |
| `--max_validation_steps`    |                     | 1回の検証実行に使用するバッチの最大数。設定しない場合、検証データセット全体が使用されます。                                            |
| `--validation_seed`         | `validation_seed`   | 検証データローダーのシャッフル用の特定のシード。TOMLファイルで設定されていない場合、メインの学習`--seed`が使用されます。                 |

</details>

### Viewing the Results

The validation loss is logged to your tracking tool of choice (TensorBoard or Weights & Biases). Look for the metric `loss/validation` to monitor the performance.

<details>
<summary>日本語</summary>

### 結果の表示

検証損失は、選択した追跡ツール（TensorBoardまたはWeights & Biases）に記録されます。パフォーマンスを監視するには、`loss/validation`という指標を探してください。

</details>

### Practical Example

Here is a complete example of how to run a LoRA training with validation enabled:

**1. Prepare your `dataset_config.toml`:**

```toml
[general]
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = "1024,1024"
batch_size = 2

  [[datasets.subsets]]
  image_dir = 'path/to/your_images'
  caption_extension = '.txt'
  num_repeats = 10

  [[datasets.subsets]]
  image_dir = 'path/to/your_validation_images'
  caption_extension = '.txt'
  validation_split = 1.0 # Use this entire subset for validation
```

**2. Run the training command:**

```bash
accelerate launch sdxl_train_network.py \
  --pretrained_model_name_or_path="sd_xl_base_1.0.safetensors" \
  --dataset_config="dataset_config.toml" \
  --output_dir="output" \
  --output_name="my_lora" \
  --network_module=networks.lora \
  --network_dim=32 \
  --network_alpha=16 \
  --save_every_n_epochs=1 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW8bit" \
  --mixed_precision="bf16" \
  --logging_dir=logs
```

The validation loss will be calculated once per epoch and saved to the `logs` directory, which you can view with TensorBoard.

<details>
<summary>日本語</summary>

### 実践的な例

検証を有効にしてLoRAの学習を実行する完全な例を次に示します。

**1. `dataset_config.toml`を準備します:**

```toml
[general]
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = "1024,1024"
batch_size = 2

  [[datasets.subsets]]
  image_dir = 'path/to/your_images'
  caption_extension = '.txt'
  num_repeats = 10

  [[datasets.subsets]]
  image_dir = 'path/to/your_validation_images'
  caption_extension = '.txt'
  validation_split = 1.0 # このサブセット全体を検証に使用します
```

**2. 学習コマンドを実行します:**

```bash
accelerate launch sdxl_train_network.py \
  --pretrained_model_name_or_path="sd_xl_base_1.0.safetensors" \
  --dataset_config="dataset_config.toml" \
  --output_dir="output" \
  --output_name="my_lora" \
  --network_module=networks.lora \
  --network_dim=32 \
  --network_alpha=16 \
  --save_every_n_epochs=1 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW8bit" \
  --mixed_precision="bf16" \
  --logging_dir=logs
```

検証損失はエポックごとに1回計算され、`logs`ディレクトリに保存されます。これはTensorBoardで表示できます。

</details>
