# About ConrtolNet-LLLite

## Overview

ConrtolNet-LLLite is a lightweight version of [ConrtolNet](https://github.com/lllyasviel/ControlNet). It is a "LoRA Like Lite" that is inspired by LoRA and has a lightweight structure. Currently, only SDXL is supported.

## Sample weight file and inference

Sample weight file is available here: https://huggingface.co/kohya-ss/controlnet-lllite

A custom node for ComfyUI is available: https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI

Sample images are at the end of this page.

## Model structure

A single LLLite module consists of a conditioning image embedding that maps a conditioning image to a latent space and a small network with a structure similar to LoRA. The LLLite module is added to U-Net's Linear and Conv in the same way as LoRA. Please refer to the source code for details.

Due to the limitations of the inference environment, only CrossAttention (attn1 q/k/v, attn2 q) is currently added.

## Model training

### Preparing the dataset

In addition to the normal dataset, please store the conditioning image in the directory specified by `conditioning_data_dir`. The conditioning image must have the same basename as the training image. The conditioning image will be automatically resized to the same size as the training image.

```toml
[[datasets.subsets]]
image_dir = "path/to/image/dir"
caption_extension = ".txt"
conditioning_data_dir = "path/to/conditioning/image/dir"
```

At the moment, random_crop cannot be used.

### Training

Run `sdxl_train_control_net_lllite.py`. You can specify the dimension of the conditioning image embedding with `--cond_emb_dim`. You can specify the rank of the LoRA-like module with `--network_dim`. Other options are the same as `sdxl_train_network.py`, but `--network_module` is not required.

For the sample Canny, the dimension of the conditioning image embedding is 32. The rank of the LoRA-like module is also 64. Adjust according to the features of the conditioning image you are targeting.

(The sample Canny is probably quite difficult. It may be better to reduce it to about half for depth, etc.)

### Inference

If you want to generate images with a script, run `sdxl_gen_img.py`. You can specify the LLLite model file with `--control_net_lllite_models`. The dimension is automatically obtained from the model file.

Specify the conditioning image to be used for inference with `--guide_image_path`. Since preprocess is not performed, if it is Canny, specify an image processed with Canny (white line on black background). `--control_net_preps`, `--control_net_weights`, and `--control_net_ratios` are not supported.

## Sample

Canny
