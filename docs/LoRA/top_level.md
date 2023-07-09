# LoRA Resource Guide

This guide is a resource compilation to facilitate the development of robust LoRA models.

Access EDG's tutorials here: https://ko-fi.com/post/EDGs-tutorials-P5P6KT5MT

## Guidelines for SDXL LoRA Training 

- Set the `Max resolution` to at least 1024x1024, as this is the standard resolution for SDXL.
- Use a GPU that has at least 12GB memory for the LoRA training process.
- We strongly recommend using the `--train_unet_only` option for SDXL LoRA to avoid unforeseen training results caused by dual text encoders in SDXL.
- PyTorch 2 tends to use less GPU memory than PyTorch 1.

Here's an example configuration for the Adafactor optimizer with a fixed learning rate:

```
optimizer_type = "adafactor"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False" ]
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100
learning_rate = 4e-7 # This is the standard learning rate for SDXL
```

## Resource Contributions

If you have valuable resources to add, kindly create a PR on Github.