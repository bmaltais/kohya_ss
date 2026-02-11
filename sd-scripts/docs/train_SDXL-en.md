## SDXL training

The documentation will be moved to the training documentation in the future. The following is a brief explanation of the training scripts for SDXL.

### Training scripts for SDXL

- `sdxl_train.py` is a script for SDXL fine-tuning. The usage is almost the same as `fine_tune.py`, but it also supports DreamBooth dataset.
  - `--full_bf16` option is added. Thanks to KohakuBlueleaf!
    - This option enables the full bfloat16 training (includes gradients). This option is useful to reduce the GPU memory usage. 
    - The full bfloat16 training might be unstable. Please use it at your own risk.
  - The different learning rates for each U-Net block are now supported in sdxl_train.py. Specify with `--block_lr` option. Specify 23 values separated by commas like `--block_lr 1e-3,1e-3 ... 1e-3`.
    - 23 values correspond to `0: time/label embed, 1-9: input blocks 0-8, 10-12: mid blocks 0-2, 13-21: output blocks 0-8, 22: out`.
- `prepare_buckets_latents.py` now supports SDXL fine-tuning.

- `sdxl_train_network.py` is a script for LoRA training for SDXL. The usage is almost the same as `train_network.py`.

- Both scripts has following additional options:
  - `--cache_text_encoder_outputs` and `--cache_text_encoder_outputs_to_disk`: Cache the outputs of the text encoders. This option is useful to reduce the GPU memory usage. This option cannot be used with options for shuffling or dropping the captions.
  - `--no_half_vae`: Disable the half-precision (mixed-precision) VAE. VAE for SDXL seems to produce NaNs in some cases. This option is useful to avoid the NaNs.

- `--weighted_captions` option is not supported yet for both scripts.

- `sdxl_train_textual_inversion.py` is a script for Textual Inversion training for SDXL. The usage is almost the same as `train_textual_inversion.py`.
  - `--cache_text_encoder_outputs` is not supported.
  - There are two options for captions:
    1. Training with captions. All captions must include the token string. The token string is replaced with multiple tokens.
    2. Use `--use_object_template` or `--use_style_template` option. The captions are generated from the template. The existing captions are ignored.
  - See below for the format of the embeddings.

- `--min_timestep` and `--max_timestep` options are added to each training script. These options can be used to train U-Net with different timesteps. The default values are 0 and 1000.

### Utility scripts for SDXL

- `tools/cache_latents.py` is added. This script can be used to cache the latents to disk in advance. 
  - The options are almost the same as `sdxl_train.py'. See the help message for the usage.
  - Please launch the script as follows:
    `accelerate launch  --num_cpu_threads_per_process 1 tools/cache_latents.py ...`
  - This script should work with multi-GPU, but it is not tested in my environment.

- `tools/cache_text_encoder_outputs.py` is added. This script can be used to cache the text encoder outputs to disk in advance. 
  - The options are almost the same as `cache_latents.py` and `sdxl_train.py`. See the help message for the usage.

- `sdxl_gen_img.py` is added. This script can be used to generate images with SDXL, including LoRA, Textual Inversion and ControlNet-LLLite. See the help message for the usage.

### Tips for SDXL training

- The default resolution of SDXL is 1024x1024.
- The fine-tuning can be done with 24GB GPU memory with the batch size of 1. For 24GB GPU, the following options are recommended __for the fine-tuning with 24GB GPU memory__:
  - Train U-Net only.
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` option and caching latents.
  - Use Adafactor optimizer. RMSprop 8bit or Adagrad 8bit may work. AdamW 8bit doesn't seem to work.
- The LoRA training can be done with 8GB GPU memory (10GB recommended). For reducing the GPU memory usage, the following options are recommended:
  - Train U-Net only.
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` option and caching latents.
  - Use one of 8bit optimizers or Adafactor optimizer.
  - Use lower dim (4 to 8 for 8GB GPU).
- `--network_train_unet_only` option is highly recommended for SDXL LoRA. Because SDXL has two text encoders, the result of the training will be unexpected.
- PyTorch 2 seems to use slightly less GPU memory than PyTorch 1.
- `--bucket_reso_steps` can be set to 32 instead of the default value 64. Smaller values than 32 will not work for SDXL training.

Example of the optimizer settings for Adafactor with the fixed learning rate:
```toml
optimizer_type = "adafactor"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False" ]
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100
learning_rate = 4e-7 # SDXL original learning rate
```

### Format of Textual Inversion embeddings for SDXL

```python
from safetensors.torch import save_file

state_dict = {"clip_g": embs_for_text_encoder_1280, "clip_l": embs_for_text_encoder_768}
save_file(state_dict, file)
```

### ControlNet-LLLite

ControlNet-LLLite, a novel method for ControlNet with SDXL, is added. See [documentation](./docs/train_lllite_README.md) for details.

