# Finetuning Resource Guide

This guide is a resource compilation to facilitate the development of robust LoRA models.

-Need to add resources here

## Guidelines for SDXL Finetuning 

- Set the `Max resolution` to at least 1024x1024, as this is the standard resolution for SDXL.
- The fine-tuning can be done with 24GB GPU memory with the batch size of 1.
  - Train U-Net only.
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` option and caching latents.
  - Use Adafactor optimizer. RMSprop 8bit or Adagrad 8bit may work. AdamW 8bit doesn't seem to work.
- PyTorch 2 seems to use slightly less GPU memory than PyTorch 1.

Example of the optimizer settings for Adafactor with the fixed learning rate:
```
optimizer_type = "adafactor"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False" ]
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100
learning_rate = 4e-7 # SDXL original learning rate
```

## Resource Contributions

If you have valuable resources to add, kindly create a PR on Github.