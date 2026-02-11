# Advanced Settings: Detailed Guide for SDXL LoRA Training Script `sdxl_train_network.py` / 高度な設定: SDXL LoRA学習スクリプト `sdxl_train_network.py` 詳細ガイド

This document describes the advanced options available when training LoRA models for SDXL (Stable Diffusion XL) with `sdxl_train_network.py` in the `sd-scripts` repository. For the basics, please read [How to Use the LoRA Training Script `train_network.py`](train_network.md) and [How to Use the SDXL LoRA Training Script `sdxl_train_network.py`](sdxl_train_network.md).

This guide targets experienced users who want to fine tune settings in detail.

**Prerequisites:**

* You have cloned the `sd-scripts` repository and prepared a Python environment.
* A training dataset and its `.toml` configuration are ready (see the dataset configuration guide).
* You are familiar with running basic LoRA training commands.

## 1. Command Line Options / コマンドライン引数 詳細解説

`sdxl_train_network.py` inherits the functionality of `train_network.py` and adds SDXL-specific features. Major options are grouped and explained below. For common arguments, see the other guides mentioned above.

### 1.1. Model Loading

* `--pretrained_model_name_or_path=\"<model path>\"` **[Required]**: specify the base SDXL model. Supports a Hugging Face model ID, a local Diffusers directory or a `.safetensors` file.
* `--vae=\"<VAE path>\"`: optionally use a different VAE. Specify when using a VAE other than the one included in the SDXL model. Can specify `.ckpt` or `.safetensors` files.
* `--no_half_vae`: keep the VAE in float32 even with fp16/bf16 training. The VAE for SDXL can become unstable with `float16`, so it is recommended to enable this when `fp16` is specified. Usually unnecessary for `bf16`.
* `--fp8_base` / `--fp8_base_unet`: **Experimental**: load the base model (U-Net, Text Encoder) or just the U-Net in FP8 to reduce VRAM (requires PyTorch 2.1+). For details, refer to the relevant section in TODO add document later (this is an SD3 explanation but also applies to SDXL).

### 1.2. Dataset Settings

* `--dataset_config=\"<path to config>\"`: specify a `.toml` dataset config. High resolution data and aspect ratio buckets (specify `enable_bucket = true` in `.toml`) are common for SDXL. The resolution steps for aspect ratio buckets (`bucket_reso_steps`) must be multiples of 32 for SDXL. For details on writing `.toml` files, refer to the [Dataset Configuration Guide](link/to/dataset/config/doc).

### 1.3. Output and Saving

Options match `train_network.py`:

* `--output_dir`, `--output_name` (both required)
* `--save_model_as` (recommended `safetensors`), `ckpt`, `pt`, `diffusers`, `diffusers_safetensors`
* `--save_precision=\"fp16\"`, `\"bf16\"`, `\"float\"`: Specifies the precision for saving the model. If not specified, the model is saved with the training precision (`fp16`, `bf16`, etc.).
* `--save_every_n_epochs=N`, `--save_every_n_steps=N`: Saves the model every N epochs/steps.
* `--save_last_n_epochs=M`, `--save_last_n_steps=M`: When saving at every epoch/step, only the latest M files are kept, and older ones are deleted.
* `--save_state`, `--save_state_on_train_end`: Saves the training state (`state`), including Optimizer status, etc., when saving the model or at the end of training. Required for resuming training with the `--resume` option.
* `--save_last_n_epochs_state=M`, `--save_last_n_steps_state=M`: Limits the number of saved `state` files to M. Overrides the `--save_last_n_epochs/steps` specification.
* `--no_metadata`: Does not save metadata to the output model.
* `--save_state_to_huggingface` and related options (e.g., `--huggingface_repo_id`): Options related to uploading models and states to Hugging Face Hub. See TODO add document for details.

### 1.4. Network Parameters (LoRA)

* `--network_module=networks.lora` **[Required]**
* `--network_dim=N` **[Required]**: Specifies the rank (dimensionality) of LoRA. For SDXL, values like 32 or 64 are often tried, but adjustment is necessary depending on the dataset and purpose.
* `--network_alpha=M`: LoRA alpha value. Generally around half of `network_dim` or the same value as `network_dim`. Default is 1.
* `--network_dropout=P`: Dropout rate (0.0-1.0) within LoRA modules. Can be effective in suppressing overfitting. Default is None (no dropout).
* `--network_args ...`: Allows advanced settings by specifying additional arguments to the network module in `key=value` format. For LoRA, the following advanced settings are available:
    *   **Block-wise dimensions/alphas:**
        *   Allows specifying different `dim` and `alpha` for each block of the U-Net. This enables adjustments to strengthen or weaken the influence of specific layers.
        *   `block_dims`: Comma-separated dims for Linear and Conv2d 1x1 layers in U-Net (23 values for SDXL).
        *   `block_alphas`: Comma-separated alpha values corresponding to the above.
        *   `conv_block_dims`: Comma-separated dims for Conv2d 3x3 layers in U-Net.
        *   `conv_block_alphas`: Comma-separated alpha values corresponding to the above.
        *   Blocks not specified will use values from `--network_dim`/`--network_alpha` or `--conv_dim`/`--conv_alpha` (if they exist).
        *   For details, refer to [Block-wise learning rate for LoRA](train_network.md#lora-の階層別学習率) (in train_network.md, applicable to SDXL) and the implementation ([lora.py](lora.py)).
    *   **LoRA+:**
        *   `loraplus_lr_ratio=R`: Sets the learning rate of LoRA's upward weights (UP) to R times the learning rate of downward weights (DOWN). Expected to improve learning speed. Paper recommends 16.
        *   `loraplus_unet_lr_ratio=RU`: Specifies the LoRA+ learning rate ratio for the U-Net part individually.
        *   `loraplus_text_encoder_lr_ratio=RT`: Specifies the LoRA+ learning rate ratio for the Text Encoder part individually (multiplied by the learning rates specified with `--text_encoder_lr1`, `--text_encoder_lr2`).
        *   For details, refer to [README](../README.md#jan-17-2025--2025-01-17-version-090) and the implementation ([lora.py](lora.py)).
* `--network_train_unet_only`: Trains only the LoRA modules of the U-Net. Specify this if not training Text Encoders. Required when using `--cache_text_encoder_outputs`.
* `--network_train_text_encoder_only`: Trains only the LoRA modules of the Text Encoders. Specify this if not training the U-Net.
* `--network_weights=\"<weight file>\"`: Starts training by loading pre-trained LoRA weights. Used for fine-tuning or resuming training. The difference from `--resume` is that this option only loads LoRA module weights, while `--resume` also restores Optimizer state, step count, etc.
* `--dim_from_weights`: Automatically reads the LoRA dimension (`dim`) from the weight file specified by `--network_weights`. Specification of `--network_dim` becomes unnecessary.

### 1.5. Training Parameters

*   `--learning_rate=LR`: Sets the overall learning rate. This becomes the default value for each module (`unet_lr`, `text_encoder_lr1`, `text_encoder_lr2`). Values like `1e-3` or `1e-4` are often tried.
*   `--unet_lr=LR_U`: Learning rate for the LoRA module of the U-Net part.
*   `--text_encoder_lr1=LR_TE1`: Learning rate for the LoRA module of Text Encoder 1 (OpenCLIP ViT-G/14). Usually, a smaller value than U-Net (e.g., `1e-5`, `2e-5`) is recommended.
*   `--text_encoder_lr2=LR_TE2`: Learning rate for the LoRA module of Text Encoder 2 (CLIP ViT-L/14). Usually, a smaller value than U-Net (e.g., `1e-5`, `2e-5`) is recommended.
*   `--optimizer_type=\"...\"`: Specifies the optimizer to use. Options include `AdamW8bit` (memory-efficient, common), `Adafactor` (even more memory-efficient, proven in SDXL full model training), `Lion`, `DAdaptation`, `Prodigy`, etc. Each optimizer may require additional arguments (see `--optimizer_args`). `AdamW8bit` or `PagedAdamW8bit` (requires `bitsandbytes`) are common. `Adafactor` is memory-efficient but slightly complex to configure (relative step (`relative_step=True`) recommended, `adafactor` learning rate scheduler recommended). `DAdaptation`, `Prodigy` have automatic learning rate adjustment but cannot be used with LoRA+. Specify a learning rate around `1.0`. For details, see the `get_optimizer` function in [train_util.py](train_util.py).
*   `--optimizer_args ...`: Specifies additional arguments to the optimizer in `key=value` format (e.g., `\"weight_decay=0.01\"` `\"betas=0.9,0.999\"`).
*   `--lr_scheduler=\"...\"`: Specifies the learning rate scheduler. Options include `constant` (no change), `cosine` (cosine curve), `linear` (linear decay), `constant_with_warmup` (constant with warmup), `cosine_with_restarts`, etc. `constant`, `cosine`, and `constant_with_warmup` are commonly used. Some schedulers require additional arguments (see `--lr_scheduler_args`). If using optimizers with auto LR adjustment like `DAdaptation` or `Prodigy`, a scheduler is not needed (`constant` should be specified).
*   `--lr_warmup_steps=N`: Number of warmup steps for the learning rate scheduler. The learning rate gradually increases during this period at the start of training. If N < 1, it's interpreted as a fraction of total steps.
*   `--lr_scheduler_num_cycles=N` / `--lr_scheduler_power=P`: Parameters for specific schedulers (`cosine_with_restarts`, `polynomial`).
*   `--max_train_steps=N` / `--max_train_epochs=N`: Specifies the total number of training steps or epochs. Epoch specification takes precedence.
*   `--mixed_precision=\"bf16\"` / `\"fp16\"` / `\"no\"`: Mixed precision training settings. For SDXL, using `bf16` (if GPU supports it) or `fp16` is strongly recommended. Reduces VRAM usage and improves training speed.
*   `--full_fp16` / `--full_bf16`: Performs gradient calculations entirely in half-precision/bf16. Can further reduce VRAM usage but may affect training stability. Use if VRAM is critically low.
*   `--gradient_accumulation_steps=N`: Accumulates gradients for N steps before updating the optimizer. Effectively increases the batch size to `train_batch_size * N`, achieving the effect of a larger batch size with less VRAM. Default is 1.
*   `--max_grad_norm=N`: Gradient clipping threshold. Clips gradients if their norm exceeds N. Default is 1.0. `0` disables it.
*   `--gradient_checkpointing`: Significantly reduces memory usage but slightly decreases training speed. Recommended for SDXL due to high memory consumption.
*   `--fused_backward_pass`: **Experimental**: Fuses gradient calculation and optimizer steps to reduce VRAM usage. Available for SDXL. Currently only supports `Adafactor` optimizer. Cannot be used with Gradient Accumulation.
*   `--resume=\"<state directory>\"`: Resumes training from a saved state (saved with `--save_state`). Restores optimizer state, step count, etc.

### 1.6. Caching

Caching is effective for SDXL due to its high computational cost.

*   `--cache_latents`: Caches VAE outputs (latents) in memory. Skips VAE computation, reducing VRAM usage and speeding up training. **Note:** Image augmentations (`color_aug`, `flip_aug`, `random_crop`, etc.) will be disabled.
*   `--cache_latents_to_disk`: Used with `--cache_latents` to cache to disk. Particularly effective for large datasets or multiple training runs. Caches are generated on disk during the first run and loaded from there on subsequent runs.
*   `--cache_text_encoder_outputs`: Caches Text Encoder outputs in memory. Skips Text Encoder computation, reducing VRAM usage and speeding up training. **Note:** Caption augmentations (`shuffle_caption`, `caption_dropout_rate`, etc.) will be disabled. **Also, when using this option, Text Encoder LoRA modules cannot be trained (requires `--network_train_unet_only`).**
*   `--cache_text_encoder_outputs_to_disk`: Used with `--cache_text_encoder_outputs` to cache to disk.
*   `--skip_cache_check`: Skips validation of cache file contents. File existence is checked, and if not found, caches are generated. Usually not needed unless intentionally re-caching for debugging, etc.

### 1.7. Sample Image Generation

Basic options are common with `train_network.py`.

*   `--sample_every_n_steps=N` / `--sample_every_n_epochs=N`: Generates sample images every N steps/epochs.
*   `--sample_at_first`: Generates sample images before training starts.
*   `--sample_prompts=\"<prompt file>\"`: Specifies a file (`.txt`, `.toml`, `.json`) containing prompts for sample image generation. 
*   `--sample_sampler=\"...\"`: Specifies the sampler (scheduler) for sample image generation. `euler_a`, `dpm++_2m_karras`, etc., are common. See `--help` for choices.

#### Format of Prompt File

A prompt file can contain multiple prompts with options, for example:

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following can be used.

  * `--n` Negative prompt up to the next option. Ignored when CFG scale is `1.0`.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image. For FLUX.1 models, the default is `1.0`, which means no CFG. For Chroma models, set to around `4.0` to enable CFG.
  * `--g` Specifies the embedded guidance scale for the models with embedded guidance (FLUX.1), the default is `3.5`. Set to `0.0` for Chroma models.
  * `--s` Specifies the number of steps in the generation.

The prompt weighting such as `( )` and `[ ]` are working for SD/SDXL models, not working for other models like FLUX.1.

### 1.8. Logging & Tracking

*   `--logging_dir=\"<log directory>\"`: Specifies the directory for TensorBoard and other logs. If not specified, logs are not output.
*   `--log_with=\"tensorboard\"` / `\"wandb\"` / `\"all\"`: Specifies the logging tool to use. If using `wandb`, `pip install wandb` is required.
*   `--log_prefix=\"<prefix>\"`: Specifies the prefix for subdirectory names created within `logging_dir`.
*   `--wandb_api_key=\"<API key>\"` / `--wandb_run_name=\"<run name>\"`: Options for Weights & Biases (wandb).
*   `--log_tracker_name` / `--log_tracker_config`: Advanced tracker configuration options. Usually not needed.
*   `--log_config`: Logs the training configuration used (excluding some sensitive information) at the start of training. Helps ensure reproducibility.

### 1.9. Regularization and Advanced Techniques

*   `--noise_offset=N`: Enables noise offset and specifies its value. Expected to improve bias in image brightness and contrast. Recommended to enable as SDXL base models are trained with this (e.g., 0.0357). Original technical explanation [here](https://www.crosslabs.org/blog/diffusion-with-offset-noise).
*   `--noise_offset_random_strength`: Randomly varies noise offset strength between 0 and the specified value.
*   `--adaptive_noise_scale=N`: Adjusts noise offset based on the mean absolute value of latents. Used with `--noise_offset`.
*   `--multires_noise_iterations=N` / `--multires_noise_discount=D`: Enables multi-resolution noise. Adding noise of different frequency components is expected to improve detail reproduction. Specify iteration count N (around 6-10) and discount rate D (around 0.3). Technical explanation [here](https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2).
*   `--ip_noise_gamma=G` / `--ip_noise_gamma_random_strength`: Enables Input Perturbation Noise. Adds small noise to input (latents) for regularization. Specify Gamma value (around 0.1). Strength can be randomized with `random_strength`.
*   `--min_snr_gamma=N`: Applies Min-SNR Weighting Strategy. Adjusts loss weights for timesteps with high noise in early training to stabilize learning. `N=5` etc. are used.
*   `--scale_v_pred_loss_like_noise_pred`: In v-prediction models, scales v-prediction loss similarly to noise prediction loss. **Not typically used for SDXL** as it's not a v-prediction model.
*   `--v_pred_like_loss=N`: Adds v-prediction-like loss to noise prediction models. `N` specifies its weight. **Not typically used for SDXL**.
*   `--debiased_estimation_loss`: Calculates loss using Debiased Estimation. Similar purpose to Min-SNR but a different approach.
*   `--loss_type=\"l1\"` / `\"l2\"` / `\"huber\"` / `\"smooth_l1\"`: Specifies the loss function. Default is `l2` (MSE). `huber` and `smooth_l1` are robust to outliers.
*   `--huber_schedule=\"constant\"` / `\"exponential\"` / `\"snr\"`: Scheduling method when using `huber` or `smooth_l1` loss. `snr` is recommended.
*   `--huber_c=C` / `--huber_scale=S`: Parameters for `huber` or `smooth_l1` loss.
*   `--masked_loss`: Limits loss calculation area based on a mask image. Requires specifying mask images (black and white) in `conditioning_data_dir` in dataset settings. See [About Masked Loss](masked_loss_README.md) for details.

### 1.10. Distributed Training and Other Training Related Options

*   `--seed=N`: Specifies the random seed. Set this to ensure training reproducibility.
*   `--max_token_length=N` (`75`, `150`, `225`): Maximum token length processed by Text Encoders. For SDXL, typically `75` (default), `150`, or `225`. Longer lengths can handle more complex prompts but increase VRAM usage.
*   `--clip_skip=N`: Uses the output from N layers skipped from the final layer of Text Encoders. **Not typically used for SDXL**.
*   `--lowram` / `--highvram`: Options for memory usage optimization. `--lowram` is for environments like Colab where RAM < VRAM, `--highvram` is for environments with ample VRAM.
*   `--persistent_data_loader_workers` / `--max_data_loader_n_workers=N`: Settings for DataLoader worker processes. Affects wait time between epochs and memory usage.
*   `--config_file="<config file>"` / `--output_config`: Options to use/output a `.toml` file instead of command line arguments.
*   **Accelerate/DeepSpeed related:** (`--ddp_timeout`, `--ddp_gradient_as_bucket_view`, `--ddp_static_graph`): Detailed settings for distributed training. Accelerate settings (`accelerate config`) are usually sufficient. DeepSpeed requires separate configuration.
* `--initial_epoch=<integer>` – Sets the initial epoch number. `1` means first epoch (same as not specifying). Note: `initial_epoch`/`initial_step` doesn't affect the lr scheduler, which means lr scheduler will start from 0 without `--resume`.
* `--initial_step=<integer>` – Sets the initial step number including all epochs. `0` means first step (same as not specifying). Overwrites `initial_epoch`.
* `--skip_until_initial_step` – Skips training until `initial_step` is reached.

### 1.11. Console and Logging / コンソールとログ

* `--console_log_level`: Sets the logging level for the console output. Choose from `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
* `--console_log_file`: Redirects console logs to a specified file.
* `--console_log_simple`: Enables a simpler log format.

### 1.12. Hugging Face Hub Integration / Hugging Face Hub 連携

* `--huggingface_repo_id`: The repository name on Hugging Face Hub to upload the model to (e.g., `your-username/your-model`).
* `--huggingface_repo_type`: The type of repository on Hugging Face Hub. Usually `model`.
* `--huggingface_path_in_repo`: The path within the repository to upload files to.
* `--huggingface_token`: Your Hugging Face Hub authentication token.
* `--huggingface_repo_visibility`: Sets the visibility of the repository (`public` or `private`).
* `--resume_from_huggingface`: Resumes training from a state saved on Hugging Face Hub.
* `--async_upload`: Enables asynchronous uploading of models to the Hub, preventing it from blocking the training process.
* `--save_n_epoch_ratio`: Saves the model at a certain ratio of total epochs. For example, `5` will save at least 5 checkpoints throughout the training.

### 1.13. Advanced Attention Settings / 高度なAttention設定

* `--mem_eff_attn`: Use memory-efficient attention mechanism. This is an older implementation and `sdpa` or `xformers` are generally recommended.
* `--xformers`: Use xformers library for memory-efficient attention. Requires `pip install xformers`.

### 1.14. Advanced LR Scheduler Settings / 高度な学習率スケジューラ設定

* `--lr_scheduler_type`: Specifies a custom scheduler module.
* `--lr_scheduler_args`: Provides additional arguments to the custom scheduler (e.g., `"T_max=100"`).
* `--lr_decay_steps`: Sets the number of steps for the learning rate to decay.
* `--lr_scheduler_timescale`: The timescale for the inverse square root scheduler.
* `--lr_scheduler_min_lr_ratio`: Sets the minimum learning rate as a ratio of the initial learning rate for certain schedulers.

### 1.15. Differential Learning with LoRA / LoRAの差分学習

This technique involves merging a pre-trained LoRA into the base model before starting a new training session. This is useful for fine-tuning an existing LoRA or for learning the 'difference' from it.

* `--base_weights`: Path to one or more LoRA weight files to be merged into the base model before training begins.
* `--base_weights_multiplier`: A multiplier for the weights of the LoRA specified by `--base_weights`. You can specify multiple values if you provide multiple weights.

### 1.16. Other Miscellaneous Options / その他のオプション

* `--tokenizer_cache_dir`: Specifies a directory to cache the tokenizer, which is useful for offline training.
* `--scale_weight_norms`: Scales the weight norms of the LoRA modules. This can help prevent overfitting by controlling the magnitude of the weights. A value of `1.0` is a good starting point.
* `--disable_mmap_load_safetensors`: Disables memory-mapped loading for `.safetensors` files. This can speed up model loading in some environments like WSL.

## 2. Other Tips / その他のTips

*   **VRAM Usage:** SDXL LoRA training requires a lot of VRAM. Even with 24GB VRAM, you might run out of memory depending on settings. Reduce VRAM usage with these settings:
    *   `--mixed_precision=\"bf16\"` or `\"fp16\"` (essential)
    *   `--gradient_checkpointing` (strongly recommended)
    *   `--cache_latents` / `--cache_text_encoder_outputs` (highly effective, with limitations)
    *   `--optimizer_type=\"AdamW8bit\"` or `\"Adafactor\"`
    *   Increase `--gradient_accumulation_steps` (reduce batch size)
    *   `--full_fp16` / `--full_bf16` (be mindful of stability)
    *   `--fp8_base` / `--fp8_base_unet` (experimental)
    *   `--fused_backward_pass` (Adafactor only, experimental)
*   **Learning Rate:** Appropriate learning rates for SDXL LoRA depend on the dataset and `network_dim`/`alpha`. Starting around `1e-4` ~ `4e-5` (U-Net), `1e-5` ~ `2e-5` (Text Encoders) is common.
*   **Training Time:** Training takes time due to high-resolution data and the size of the SDXL model. Using caching features and appropriate hardware is important.
*   **Troubleshooting:**
    *   **NaN Loss:** Learning rate might be too high, mixed precision settings incorrect (e.g., `--no_half_vae` not specified with `fp16`), or dataset issues.
    *   **Out of Memory (OOM):** Try the VRAM reduction measures listed above.
    *   **Training not progressing:** Learning rate might be too low, optimizer/scheduler settings incorrect, or dataset issues.

## 3. Conclusion / おわりに

`sdxl_train_network.py` offers many options to customize SDXL LoRA training. Refer to `--help`, other documents and the source code for further details.

<details>
<summary>日本語</summary>

# 高度な設定: SDXL LoRA学習スクリプト `sdxl_train_network.py` 詳細ガイド

このドキュメントでは、`sd-scripts` リポジトリに含まれる `sdxl_train_network.py` を使用した、SDXL (Stable Diffusion XL) モデルに対する LoRA (Low-Rank Adaptation) モデル学習の高度な設定オプションについて解説します。

基本的な使い方については、以下のドキュメントを参照してください。

*   [LoRA学習スクリプト `train_network.py` の使い方](train_network.md)
*   [SDXL LoRA学習スクリプト `sdxl_train_network.py` の使い方](sdxl_train_network.md)

このガイドは、基本的なLoRA学習の経験があり、より詳細な設定や高度な機能を試したい熟練した利用者を対象としています。

**前提条件:**

*   `sd-scripts` リポジトリのクローンと Python 環境のセットアップが完了していること。
*   学習用データセットの準備と設定（`.toml`ファイル）が完了していること。（[データセット設定ガイド](link/to/dataset/config/doc)参照）
*   基本的なLoRA学習のコマンドライン実行経験があること。

## 1. コマンドライン引数 詳細解説

`sdxl_train_network.py` は `train_network.py` の機能を継承しつつ、SDXL特有の機能を追加しています。ここでは、SDXL LoRA学習に関連する主要なコマンドライン引数について、機能別に分類して詳細に解説します。

基本的な引数については、[LoRA学習スクリプト `train_network.py` の使い方](train_network.md#31-主要なコマンドライン引数) および [SDXL LoRA学習スクリプト `sdxl_train_network.py` の使い方](sdxl_train_network.md#31-主要なコマンドライン引数（差分）) を参照してください。

### 1.1. モデル読み込み関連

*   `--pretrained_model_name_or_path="<モデルパス>"` **[必須]**
    *   学習のベースとなる **SDXLモデル** を指定します。Hugging Face HubのモデルID、ローカルのDiffusers形式モデルディレクトリ、または`.safetensors`ファイルを指定できます。
    *   詳細は[基本ガイド](sdxl_train_network.md#モデル関連)を参照してください。
*   `--vae="<VAEパス>"`
    *   オプションで、学習に使用するVAEを指定します。SDXLモデルに含まれるVAE以外を使用する場合に指定します。`.ckpt`または`.safetensors`ファイルを指定できます。
*   `--no_half_vae`
    *   混合精度(`fp16`/`bf16`)使用時でもVAEを`float32`で動作させます。SDXLのVAEは`float16`で不安定になることがあるため、`fp16`指定時には有効にすることが推奨されます。`bf16`では通常不要です。
*   `--fp8_base` / `--fp8_base_unet`
    *   **実験的機能:** ベースモデル（U-Net, Text Encoder）またはU-NetのみをFP8で読み込み、VRAM使用量を削減します。PyTorch 2.1以上が必要です。詳細は TODO 後でドキュメントを追加 の関連セクションを参照してください (SD3の説明ですがSDXLにも適用されます)。

### 1.2. データセット設定関連

*   `--dataset_config="<設定ファイルのパス>"` 
    *   データセットの設定を記述した`.toml`ファイルを指定します。SDXLでは高解像度データとバケツ機能（`.toml` で `enable_bucket = true` を指定）の利用が一般的です。
    *   `.toml`ファイルの書き方の詳細は[データセット設定ガイド](link/to/dataset/config/doc)を参照してください。
    *   アスペクト比バケツの解像度ステップ(`bucket_reso_steps`)は、SDXLでは32の倍数とする必要があります。

### 1.3. 出力・保存関連

基本的なオプションは `train_network.py` と共通です。

*   `--output_dir="<出力先ディレクトリ>"` **[必須]**
*   `--output_name="<出力ファイル名>"` **[必須]**
*   `--save_model_as="safetensors"` (推奨), `ckpt`, `pt`, `diffusers`, `diffusers_safetensors`
*   `--save_precision="fp16"`, `"bf16"`, `"float"`
    *   モデルの保存精度を指定します。未指定時は学習時の精度(`fp16`, `bf16`等)で保存されます。
*   `--save_every_n_epochs=N` / `--save_every_n_steps=N`
    *   Nエポック/ステップごとにモデルを保存します。
*   `--save_last_n_epochs=M` / `--save_last_n_steps=M`
    *   エポック/ステップごとに保存する際、最新のM個のみを保持し、古いものは削除します。
*   `--save_state` / `--save_state_on_train_end`
    *   モデル保存時/学習終了時に、Optimizerの状態などを含む学習状態(`state`)を保存します。`--resume`オプションでの学習再開に必要です。
*   `--save_last_n_epochs_state=M` / `--save_last_n_steps_state=M`
    *   `state`の保存数をM個に制限します。`--save_last_n_epochs/steps`の指定を上書きします。
*   `--no_metadata`
    *   出力モデルにメタデータを保存しません。
*   `--save_state_to_huggingface` / `--huggingface_repo_id` など
    *   Hugging Face Hubへのモデルやstateのアップロード関連オプション。詳細は TODO ドキュメントを追加 を参照してください。

### 1.4. ネットワークパラメータ (LoRA)

基本的なオプションは `train_network.py` と共通です。

*   `--network_module=networks.lora` **[必須]**
*   `--network_dim=N` **[必須]**
    *   LoRAのランク (次元数) を指定します。SDXLでは32や64などが試されることが多いですが、データセットや目的に応じて調整が必要です。
*   `--network_alpha=M`
    *   LoRAのアルファ値。`network_dim`の半分程度、または`network_dim`と同じ値などが一般的です。デフォルトは1。
*   `--network_dropout=P`
    *   LoRAモジュール内のドロップアウト率 (0.0~1.0)。過学習抑制の効果が期待できます。デフォルトはNone (ドロップアウトなし)。
*   `--network_args ...`
    *   ネットワークモジュールへの追加引数を `key=value` 形式で指定します。LoRAでは以下の高度な設定が可能です。
        *   **階層別 (Block-wise) 次元数/アルファ:**
            *   U-Netの各ブロックごとに異なる`dim`と`alpha`を指定できます。これにより、特定の層の影響を強めたり弱めたりする調整が可能です。
            *   `block_dims`: U-NetのLinear層およびConv2d 1x1層に対するブロックごとのdimをカンマ区切りで指定します (SDXLでは23個の数値)。
            *   `block_alphas`: 上記に対応するalpha値をカンマ区切りで指定します。
            *   `conv_block_dims`: U-NetのConv2d 3x3層に対するブロックごとのdimをカンマ区切りで指定します。
            *   `conv_block_alphas`: 上記に対応するalpha値をカンマ区切りで指定します。
            *   指定しないブロックは `--network_dim`/`--network_alpha` または `--conv_dim`/`--conv_alpha` (存在する場合) の値が使用されます。
            *   詳細は[LoRA の階層別学習率](train_network.md#lora-の階層別学習率) (train\_network.md内、SDXLでも同様に適用可能) や実装 ([lora.py](lora.py)) を参照してください。
        *   **LoRA+:**
            *   `loraplus_lr_ratio=R`: LoRAの上向き重み(UP)の学習率を、下向き重み(DOWN)の学習率のR倍にします。学習速度の向上が期待できます。論文推奨は16。
            *   `loraplus_unet_lr_ratio=RU`: U-Net部分のLoRA+学習率比を個別に指定します。
            *   `loraplus_text_encoder_lr_ratio=RT`: Text Encoder部分のLoRA+学習率比を個別に指定します。(`--text_encoder_lr1`, `--text_encoder_lr2`で指定した学習率に乗算されます)
            *   詳細は[README](../README.md#jan-17-2025--2025-01-17-version-090)や実装 ([lora.py](lora.py)) を参照してください。
*   `--network_train_unet_only`
    *   U-NetのLoRAモジュールのみを学習します。Text Encoderの学習を行わない場合に指定します。`--cache_text_encoder_outputs` を使用する場合は必須です。
*   `--network_train_text_encoder_only`
    *   Text EncoderのLoRAモジュールのみを学習します。U-Netの学習を行わない場合に指定します。
*   `--network_weights="<重みファイル>"`
    *   学習済みのLoRA重みを読み込んで学習を開始します。ファインチューニングや学習再開に使用します。`--resume` との違いは、このオプションはLoRAモジュールの重みのみを読み込み、`--resume` はOptimizerの状態や学習ステップ数なども復元します。
*   `--dim_from_weights`
    *   `--network_weights` で指定した重みファイルからLoRAの次元数 (`dim`) を自動的に読み込みます。`--network_dim` の指定は不要になります。

### 1.5. 学習パラメータ

*   `--learning_rate=LR`
    *   全体の学習率。各モジュール(`unet_lr`, `text_encoder_lr1`, `text_encoder_lr2`)のデフォルト値となります。`1e-3` や `1e-4` などが試されることが多いです。
*   `--unet_lr=LR_U`
    *   U-Net部分のLoRAモジュールの学習率。
*   `--text_encoder_lr1=LR_TE1`
    *   Text Encoder 1 (OpenCLIP ViT-G/14) のLoRAモジュールの学習率。通常、U-Netより小さい値 (例: `1e-5`, `2e-5`) が推奨されます。
*   `--text_encoder_lr2=LR_TE2`
    *   Text Encoder 2 (CLIP ViT-L/14) のLoRAモジュールの学習率。通常、U-Netより小さい値 (例: `1e-5`, `2e-5`) が推奨されます。
*   `--optimizer_type="..."`
    *   使用するOptimizerを指定します。`AdamW8bit` (省メモリ、一般的), `Adafactor` (さらに省メモリ、SDXLフルモデル学習で実績あり), `Lion`, `DAdaptation`, `Prodigy`などが選択可能です。各Optimizerには追加の引数が必要な場合があります (`--optimizer_args`参照)。
    *   `AdamW8bit` や `PagedAdamW8bit` (要 `bitsandbytes`) が一般的です。
    *   `Adafactor` はメモリ効率が良いですが、設定がやや複雑です (相対ステップ(`relative_step=True`)推奨、学習率スケジューラは`adafactor`推奨)。
    *   `DAdaptation`, `Prodigy` は学習率の自動調整機能がありますが、LoRA+との併用はできません。学習率は`1.0`程度を指定します。
    *   詳細は[train\_util.py](train_util.py)の`get_optimizer`関数を参照してください。
*   `--optimizer_args ...`
    *   Optimizerへの追加引数を `key=value` 形式で指定します (例: `"weight_decay=0.01"` `"betas=0.9,0.999"`).
*   `--lr_scheduler="..."`
    *   学習率スケジューラを指定します。`constant` (変化なし), `cosine` (コサインカーブ), `linear` (線形減衰), `constant_with_warmup` (ウォームアップ付き定数), `cosine_with_restarts` など。`constant` や `cosine` 、 `constant_with_warmup` がよく使われます。
    *   スケジューラによっては追加の引数が必要です (`--lr_scheduler_args`参照)。
    *   `DAdaptation` や `Prodigy` などの自己学習率調整機能付きOptimizerを使用する場合、スケジューラは不要です (`constant` を指定)。
*   `--lr_warmup_steps=N`
    *   学習率スケジューラのウォームアップステップ数。学習開始時に学習率を徐々に上げていく期間です。N < 1 の場合は全ステップ数に対する割合と解釈されます。
*   `--lr_scheduler_num_cycles=N` / `--lr_scheduler_power=P`
    *   特定のスケジューラ (`cosine_with_restarts`, `polynomial`) のためのパラメータ。
*   `--max_train_steps=N` / `--max_train_epochs=N`
    *   学習の総ステップ数またはエポック数を指定します。エポック指定が優先されます。
*   `--mixed_precision="bf16"` / `"fp16"` / `"no"`
    *   混合精度学習の設定。SDXLでは `bf16` (対応GPUの場合) または `fp16` の使用が強く推奨されます。VRAM使用量を削減し、学習速度を向上させます。
*   `--full_fp16` / `--full_bf16`
    *   勾配計算も含めて完全に半精度/bf16で行います。VRAM使用量をさらに削減できますが、学習の安定性に影響する可能性があります。VRAMがどうしても足りない場合に使用します。
*   `--gradient_accumulation_steps=N`
    *   勾配をNステップ分蓄積してからOptimizerを更新します。実質的なバッチサイズを `train_batch_size * N` に増やし、少ないVRAMで大きなバッチサイズ相当の効果を得られます。デフォルトは1。
*   `--max_grad_norm=N`
    *   勾配クリッピングの閾値。勾配のノルムがNを超える場合にクリッピングします。デフォルトは1.0。`0`で無効。
*   `--gradient_checkpointing`
    *   メモリ使用量を大幅に削減しますが、学習速度は若干低下します。SDXLではメモリ消費が大きいため、有効にすることが推奨されます。
*   `--fused_backward_pass`
    *   **実験的機能:** 勾配計算とOptimizerのステップを融合し、VRAM使用量を削減します。SDXLで利用可能です。現在 `Adafactor` Optimizerのみ対応。Gradient Accumulationとは併用できません。
*   `--resume="<stateディレクトリ>"`
    *   `--save_state`で保存された学習状態から学習を再開します。Optimizerの状態や学習ステップ数などが復元されます。

### 1.6. キャッシュ機能関連

SDXLは計算コストが高いため、キャッシュ機能が効果的です。

*   `--cache_latents`
    *   VAEの出力(Latent)をメモリにキャッシュします。VAEの計算を省略でき、VRAM使用量を削減し、学習を高速化します。**注意:** 画像に対するAugmentation (`color_aug`, `flip_aug`, `random_crop` 等) は無効になります。
*   `--cache_latents_to_disk`
    *   `--cache_latents` と併用し、キャッシュ先をディスクにします。大量のデータセットや複数回の学習で特に有効です。初回実行時にディスクにキャッシュが生成され、2回目以降はそれを読み込みます。
*   `--cache_text_encoder_outputs`
    *   Text Encoderの出力をメモリにキャッシュします。Text Encoderの計算を省略でき、VRAM使用量を削減し、学習を高速化します。**注意:** キャプションに対するAugmentation (`shuffle_caption`, `caption_dropout_rate` 等) は無効になります。**また、このオプションを使用する場合、Text EncoderのLoRAモジュールは学習できません (`--network_train_unet_only` の指定が必須です)。**
*   `--cache_text_encoder_outputs_to_disk`
    *   `--cache_text_encoder_outputs` と併用し、キャッシュ先をディスクにします。
*   `--skip_cache_check`
    *   キャッシュファイルの内容の検証をスキップします。ファイルの存在確認は行われ、存在しない場合はキャッシュが生成されます。デバッグ等で意図的に再キャッシュしたい場合を除き、通常は指定不要です。

### 1.7. サンプル画像生成関連

基本的なオプションは `train_network.py` と共通です。

*   `--sample_every_n_steps=N` / `--sample_every_n_epochs=N`
    *   Nステップ/エポックごとにサンプル画像を生成します。
*   `--sample_at_first`
    *   学習開始前にサンプル画像を生成します。
*   `--sample_prompts="<プロンプトファイル>"`
    *   サンプル画像生成に使用するプロンプトを記述したファイル (`.txt`, `.toml`, `.json`) を指定します。
*   `--sample_sampler="..."`
    *   サンプル画像生成時のサンプラー（スケジューラ）を指定します。`euler_a`, `dpm++_2m_karras` などが一般的です。選択肢は `--help` を参照してください。

#### プロンプトファイルの書式
プロンプトファイルは複数のプロンプトとオプションを含めることができます。例えば：

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

`#`で始まる行はコメントです。生成画像のオプションはプロンプトの後に `--n` のように指定できます。以下のオプションが使用可能です。

  * `--n` 次のオプションまでがネガティブプロンプトです。CFGスケールが `1.0` の場合は無視されます。
  * `--w` 生成画像の幅を指定します。
  * `--h` 生成画像の高さを指定します。
  * `--d` 生成画像のシード値を指定します。
  * `--l` 生成画像のCFGスケールを指定します。FLUX.1モデルでは、デフォルトは `1.0` でCFGなしを意味します。Chromaモデルでは、CFGを有効にするために `4.0` 程度に設定してください。
  * `--g` 埋め込みガイダンス付きモデル（FLUX.1）の埋め込みガイダンススケールを指定、デフォルトは `3.5`。Chromaモデルでは `0.0` に設定してください。
  * `--s` 生成時のステップ数を指定します。

プロンプトの重み付け `( )` や `[ ]` はSD/SDXLモデルで動作し、FLUX.1など他のモデルでは動作しません。

### 1.8. Logging & Tracking 関連

*   `--logging_dir="<ログディレクトリ>"`
    *   TensorBoardなどのログを出力するディレクトリを指定します。指定しない場合、ログは出力されません。
*   `--log_with="tensorboard"` / `"wandb"` / `"all"`
    *   使用するログツールを指定します。`wandb`を使用する場合、`pip install wandb`が必要です。
*   `--log_prefix="<プレフィックス>"`
    *   `logging_dir` 内に作成されるサブディレクトリ名の接頭辞を指定します。
*   `--wandb_api_key="<APIキー>"` / `--wandb_run_name="<実行名>"`
    *   Weights & Biases (wandb) 使用時のオプション。
*   `--log_tracker_name` / `--log_tracker_config`
    *   高度なトラッカー設定用オプション。通常は指定不要。
*   `--log_config`
    *   学習開始時に、使用された学習設定（一部の機密情報を除く）をログに出力します。再現性の確保に役立ちます。

### 1.9. 正則化・高度な学習テクニック関連

*   `--noise_offset=N`
    *   ノイズオフセットを有効にし、その値を指定します。画像の明るさやコントラストの偏りを改善する効果が期待できます。SDXLのベースモデルはこの値で学習されているため、有効にすることが推奨されます (例: 0.0357)。元々の技術解説は[こちら](https://www.crosslabs.org/blog/diffusion-with-offset-noise)。
*   `--noise_offset_random_strength`
    *   ノイズオフセットの強度を0から指定値の間でランダムに変動させます。
*   `--adaptive_noise_scale=N`
    *   Latentの平均絶対値に応じてノイズオフセットを調整します。`--noise_offset`と併用します。
*   `--multires_noise_iterations=N` / `--multires_noise_discount=D`
    *   複数解像度ノイズを有効にします。異なる周波数成分のノイズを加えることで、ディテールの再現性を向上させる効果が期待できます。イテレーション回数N (6-10程度) と割引率D (0.3程度) を指定します。技術解説は[こちら](https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2)。
*   `--ip_noise_gamma=G` / `--ip_noise_gamma_random_strength`
    *   Input Perturbation Noiseを有効にします。入力(Latent)に微小なノイズを加えて正則化を行います。Gamma値 (0.1程度) を指定します。`random_strength`で強度をランダム化できます。
*   `--min_snr_gamma=N`
    *   Min-SNR Weighting Strategy を適用します。学習初期のノイズが大きいタイムステップでのLossの重みを調整し、学習を安定させます。`N=5` などが使用されます。
*   `--scale_v_pred_loss_like_noise_pred`
    *   v-predictionモデルにおいて、vの予測ロスをノイズ予測ロスと同様のスケールに調整します。SDXLはv-predictionではないため、**通常は使用しません**。
*   `--v_pred_like_loss=N`
    *   ノイズ予測モデルにv予測ライクなロスを追加します。`N`でその重みを指定します。SDXLでは**通常は使用しません**。
*   `--debiased_estimation_loss`
    *   Debiased EstimationによるLoss計算を行います。Min-SNRと類似の目的を持ちますが、異なるアプローチです。
*   `--loss_type="l1"` / `"l2"` / `"huber"` / `"smooth_l1"`
    *   損失関数を指定します。デフォルトは`l2` (MSE)。`huber`や`smooth_l1`は外れ値に頑健な損失関数です。
*   `--huber_schedule="constant"` / `"exponential"` / `"snr"`
    *   `huber`または`smooth_l1`損失使用時のスケジューリング方法。`snr`が推奨されています。
*   `--huber_c=C` / `--huber_scale=S`
    *   `huber`または`smooth_l1`損失のパラメータ。
*   `--masked_loss`
    *   マスク画像に基づいてLoss計算領域を限定します。データセット設定で`conditioning_data_dir`にマスク画像（白黒）を指定する必要があります。詳細は[マスクロスについて](masked_loss_README.md)を参照してください。

### 1.10. 分散学習、その他学習関連

*   `--seed=N`
    *   乱数シードを指定します。学習の再現性を確保したい場合に設定します。
*   `--max_token_length=N` (`75`, `150`, `225`)
    *   Text Encoderが処理するトークンの最大長。SDXLでは通常`75` (デフォルト) または `150`, `225`。長くするとより複雑なプロンプトを扱えますが、VRAM使用量が増加します。
*   `--clip_skip=N`
    *   Text Encoderの最終層からN層スキップした層の出力を使用します。SDXLでは**通常使用しません**。
*   `--lowram` / `--highvram`
    *   メモリ使用量の最適化に関するオプション。`--lowram`はColabなどRAM < VRAM環境向け、`--highvram`はVRAM潤沢な環境向け。
*   `--persistent_data_loader_workers` / `--max_data_loader_n_workers=N`
    *   DataLoaderのワーカプロセスに関する設定。エポック間の待ち時間やメモリ使用量に影響します。
*   `--config_file="<設定ファイル>"` / `--output_config`
    *   コマンドライン引数の代わりに`.toml`ファイルを使用/出力するオプション。
*   **Accelerate/DeepSpeed関連:** (`--ddp_timeout`, `--ddp_gradient_as_bucket_view`, `--ddp_static_graph`)
    *   分散学習時の詳細設定。通常はAccelerateの設定 (`accelerate config`) で十分です。DeepSpeedを使用する場合は、別途設定が必要です。
*   `--initial_epoch=<integer>` – 開始エポック番号を設定します。`1`で最初のエポック（未指定時と同じ）。注意：`initial_epoch`/`initial_step`はlr schedulerに影響しないため、`--resume`しない場合はlr schedulerは0から始まります。
*   `--initial_step=<integer>` – 全エポックを含む開始ステップ番号を設定します。`0`で最初のステップ（未指定時と同じ）。`initial_epoch`を上書きします。
*   `--skip_until_initial_step` – `initial_step`に到達するまで学習をスキップします。

### 1.11. コンソールとログ

* `--console_log_level`: コンソール出力のログレベルを設定します。`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`から選択します。
* `--console_log_file`: コンソールのログを指定されたファイルに出力します。
* `--console_log_simple`: よりシンプルなログフォーマットを有効にします。

### 1.12. Hugging Face Hub 連携

* `--huggingface_repo_id`: モデルをアップロードするHugging Face Hubのリポジトリ名 (例: `your-username/your-model`)。
* `--huggingface_repo_type`: Hugging Face Hubのリポジトリの種類。通常は`model`です。
* `--huggingface_path_in_repo`: リポジトリ内でファイルをアップロードするパス。
* `--huggingface_token`: Hugging Face Hubの認証トークン。
* `--huggingface_repo_visibility`: リポジトリの公開設定 (`public`または`private`)。
* `--resume_from_huggingface`: Hugging Face Hubに保存された状態から学習を再開します。
* `--async_upload`: Hubへのモデルの非同期アップロードを有効にし、学習プロセスをブロックしないようにします。
* `--save_n_epoch_ratio`: 総エポック数に対する特定の比率でモデルを保存します。例えば`5`を指定すると、学習全体で少なくとも5つのチェックポイントが保存されます。

### 1.13. 高度なAttention設定

* `--mem_eff_attn`: メモリ効率の良いAttentionメカニズムを使用します。これは古い実装であり、一般的には`sdpa`や`xformers`の使用が推奨されます。
* `--xformers`: メモリ効率の良いAttentionのためにxformersライブラリを使用します。`pip install xformers`が必要です。

### 1.14. 高度な学習率スケジューラ設定

* `--lr_scheduler_type`: カスタムスケジューラモジュールを指定します。
* `--lr_scheduler_args`: カスタムスケジューラに追加の引数を渡します (例: `"T_max=100"`)。
* `--lr_decay_steps`: 学習率が減衰するステップ数を設定します。
* `--lr_scheduler_timescale`: 逆平方根スケジューラのタイムスケール。
* `--lr_scheduler_min_lr_ratio`: 特定のスケジューラについて、初期学習率に対する最小学習率の比率を設定します。

### 1.15. LoRAの差分学習

既存の学習済みLoRAをベースモデルにマージしてから、新たな学習を開始する手法です。既存LoRAのファインチューニングや、差分を学習させたい場合に有効です。

* `--base_weights`: 学習開始前にベースモデルにマージするLoRAの重みファイルを1つ以上指定します。
* `--base_weights_multiplier`: `--base_weights`で指定したLoRAの重みの倍率。複数指定も可能です。

### 1.16. その他のオプション

* `--tokenizer_cache_dir`: オフラインでの学習に便利なように、tokenizerをキャッシュするディレクトリを指定します。
* `--scale_weight_norms`: LoRAモジュールの重みのノルムをスケーリングします。重みの大きさを制御することで過学習を防ぐ助けになります。`1.0`が良い出発点です。
* `--disable_mmap_load_safetensors`: `.safetensors`ファイルのメモリマップドローディングを無効にします。WSLなどの一部環境でモデルの読み込みを高速化できます。

## 2. その他のTips


*   **VRAM使用量:** SDXL LoRA学習は多くのVRAMを必要とします。24GB VRAMでも設定によってはメモリ不足になることがあります。以下の設定でVRAM使用量を削減できます。
    *   `--mixed_precision="bf16"` または `"fp16"` (必須級)
    *   `--gradient_checkpointing` (強く推奨)
    *   `--cache_latents` / `--cache_text_encoder_outputs` (効果大、制約あり)
    *   `--optimizer_type="AdamW8bit"` または `"Adafactor"`
    *   `--gradient_accumulation_steps` の値を増やす (バッチサイズを小さくする)
    *   `--full_fp16` / `--full_bf16` (安定性に注意)
    *   `--fp8_base` / `--fp8_base_unet` (実験的)
    *   `--fused_backward_pass` (Adafactor限定、実験的)
*   **学習率:** SDXL LoRAの適切な学習率はデータセットや`network_dim`/`alpha`に依存します。`1e-4` ~ `4e-5` (U-Net), `1e-5` ~ `2e-5` (Text Encoders) あたりから試すのが一般的です。
*   **学習時間:** 高解像度データとSDXLモデルのサイズのため、学習には時間がかかります。キャッシュ機能や適切なハードウェアの利用が重要です。
*   **トラブルシューティング:**
    *   **NaN Loss:** 学習率が高すぎる、混合精度の設定が不適切 (`fp16`時の`--no_half_vae`未指定など)、データセットの問題などが考えられます。
    *   **VRAM不足 (OOM):** 上記のVRAM削減策を試してください。
    *   **学習が進まない:** 学習率が低すぎる、Optimizer/Schedulerの設定が不適切、データセットの問題などが考えられます。

## 3. おわりに

`sdxl_train_network.py` は非常に多くのオプションを提供しており、SDXL LoRA学習の様々な側面をカスタマイズできます。このドキュメントが、より高度な設定やチューニングを行う際の助けとなれば幸いです。

不明な点や詳細については、各スクリプトの `--help` オプションや、リポジトリ内の他のドキュメント、実装コード自体を参照してください。

</details>
