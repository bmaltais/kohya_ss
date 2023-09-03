# About ControlNet-LLLite

__This is an extremely experimental implementation and may change significantly in the future.__

日本語版は[こちら](./train_lllite_README-ja.md)

## Overview

ControlNet-LLLite is a lightweight version of [ControlNet](https://github.com/lllyasviel/ControlNet). It is a "LoRA Like Lite" that is inspired by LoRA and has a lightweight structure. Currently, only SDXL is supported.

## Sample weight file and inference

Sample weight file is available here: https://huggingface.co/kohya-ss/controlnet-lllite

A custom node for ComfyUI is available: https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI

Sample images are at the end of this page.

## Model structure

A single LLLite module consists of a conditioning image embedding that maps a conditioning image to a latent space and a small network with a structure similar to LoRA. The LLLite module is added to U-Net's Linear and Conv in the same way as LoRA. Please refer to the source code for details.

Due to the limitations of the inference environment, only CrossAttention (attn1 q/k/v, attn2 q) is currently added.

## Model training

### Preparing the dataset

In addition to the normal dataset, please store the conditioning image in the directory specified by `conditioning_data_dir`. The conditioning image must have the same basename as the training image. The conditioning image will be automatically resized to the same size as the training image. The conditioning image does not require a caption file.

```toml
[[datasets.subsets]]
image_dir = "path/to/image/dir"
caption_extension = ".txt"
conditioning_data_dir = "path/to/conditioning/image/dir"
```

At the moment, random_crop cannot be used.

For training data, it is easiest to use a synthetic dataset with the original model-generated images as training images and processed images as conditioning images (the quality of the dataset may be problematic). See below for specific methods of synthesizing datasets.

Note that if you use an image with a different art style than the original model as a training image, the model will have to learn not only the control but also the art style. ControlNet-LLLite has a small capacity, so it is not suitable for learning art styles. In such cases, increase the number of dimensions as described below.

### Training

Run `sdxl_train_control_net_lllite.py`. You can specify the dimension of the conditioning image embedding with `--cond_emb_dim`. You can specify the rank of the LoRA-like module with `--network_dim`. Other options are the same as `sdxl_train_network.py`, but `--network_module` is not required.

Since a large amount of memory is used during training, please enable memory-saving options such as cache and gradient checkpointing. It is also effective to use BFloat16 with the `--full_bf16` option (requires RTX 30 series or later GPU). It has been confirmed to work with 24GB VRAM.

For the sample Canny, the dimension of the conditioning image embedding is 32. The rank of the LoRA-like module is also 64. Adjust according to the features of the conditioning image you are targeting.

(The sample Canny is probably quite difficult. It may be better to reduce it to about half for depth, etc.)

The following is an example of a .toml configuration.

```toml
pretrained_model_name_or_path = "/path/to/model_trained_on.safetensors"
max_train_epochs = 12
max_data_loader_n_workers = 4
persistent_data_loader_workers = true
seed = 42
gradient_checkpointing = true
mixed_precision = "bf16"
save_precision = "bf16"
full_bf16 = true
optimizer_type = "adamw8bit"
learning_rate = 2e-4
xformers = true
output_dir = "/path/to/output/dir"
output_name = "output_name"
save_every_n_epochs = 1
save_model_as = "safetensors"
vae_batch_size = 4
cache_latents = true
cache_latents_to_disk = true
cache_text_encoder_outputs = true
cache_text_encoder_outputs_to_disk = true
network_dim = 64
cond_emb_dim = 32
dataset_config = "/path/to/dataset.toml"
```

### Inference

If you want to generate images with a script, run `sdxl_gen_img.py`. You can specify the LLLite model file with `--control_net_lllite_models`. The dimension is automatically obtained from the model file.

Specify the conditioning image to be used for inference with `--guide_image_path`. Since preprocess is not performed, if it is Canny, specify an image processed with Canny (white line on black background). `--control_net_preps`, `--control_net_weights`, and `--control_net_ratios` are not supported.

## How to synthesize a dataset

### Generating training images

Generate images with the base model for training. Please generate them with Web UI or ComfyUI etc. The image size should be the default size of the model (1024x1024, etc.). You can also use bucketing. In that case, please generate it at an arbitrary resolution.

The captions and other settings when generating the images should be the same as when generating the images with the trained ControlNet-LLLite model.

Save the generated images in an arbitrary directory. Specify this directory in the dataset configuration file.


You can also generate them with `sdxl_gen_img.py` in this repository. For example, run as follows:

```dos
python sdxl_gen_img.py --ckpt path/to/model.safetensors --n_iter 1 --scale 10 --steps 36 --outdir path/to/output/dir --xformers --W 1024 --H 1024 --original_width 2048 --original_height 2048 --bf16 --sampler ddim --batch_size 4 --vae_batch_size 2 --images_per_prompt 512 --max_embeddings_multiples 1 --prompt "{portrait|digital art|anime screen cap|detailed illustration} of 1girl, {standing|sitting|walking|running|dancing} on {classroom|street|town|beach|indoors|outdoors}, {looking at viewer|looking away|looking at another}, {in|wearing} {shirt and skirt|school uniform|casual wear} { |, dynamic pose}, (solo), teen age, {0-1$$smile,|blush,|kind smile,|expression less,|happy,|sadness,} {0-1$$upper body,|full body,|cowboy shot,|face focus,} trending on pixiv, {0-2$$depth of fields,|8k wallpaper,|highly detailed,|pov,} {0-1$$summer, |winter, |spring, |autumn, } beautiful face { |, from below|, from above|, from side|, from behind|, from back} --n nsfw, bad face, lowres, low quality, worst quality, low effort, watermark, signature, ugly, poorly drawn"
```

This is a setting for VRAM 24GB. Adjust `--batch_size` and `--vae_batch_size` according to the VRAM size.

The images are generated randomly using wildcards in `--prompt`. Adjust as necessary.

### Processing images

Use an external program to process the generated images. Save the processed images in an arbitrary directory. These will be the conditioning images.

For example, you can use the following script to process the images with Canny.

```python
import glob
import os
import random
import cv2
import numpy as np

IMAGES_DIR = "path/to/generated/images"
CANNY_DIR = "path/to/canny/images"

os.makedirs(CANNY_DIR, exist_ok=True)
img_files = glob.glob(IMAGES_DIR + "/*.png")
for img_file in img_files:
    can_file = CANNY_DIR + "/" + os.path.basename(img_file)
    if os.path.exists(can_file):
        print("Skip: " + img_file)
        continue

    print(img_file)

    img = cv2.imread(img_file)

    # random threshold
    # while True:
    #     threshold1 = random.randint(0, 127)
    #     threshold2 = random.randint(128, 255)
    #     if threshold2 - threshold1 > 80:
    #         break

    # fixed threshold
    threshold1 = 100
    threshold2 = 200

    img = cv2.Canny(img, threshold1, threshold2)

    cv2.imwrite(can_file, img)
```

### Creating caption files

Create a caption file for each image with the same basename as the training image. It is fine to use the same caption as the one used when generating the image. 

If you generated the images with `sdxl_gen_img.py`, you can use the following script to create the caption files (`*.txt`) from the metadata in the generated images.

```python
import glob
import os
from PIL import Image

IMAGES_DIR = "path/to/generated/images"

img_files = glob.glob(IMAGES_DIR + "/*.png")
for img_file in img_files:
    cap_file = img_file.replace(".png", ".txt")
    if os.path.exists(cap_file):
        print(f"Skip: {img_file}")
        continue
    print(img_file)

    img = Image.open(img_file)
    prompt = img.text["prompt"] if "prompt" in img.text else ""
    if prompt == "":
        print(f"Prompt not found in {img_file}")

    with open(cap_file, "w") as f:
        f.write(prompt + "\n")
```

### Creating a dataset configuration file

You can use the command line arguments of `sdxl_train_control_net_lllite.py` to specify the conditioning image directory. However, if you want to use a `.toml` file, specify the conditioning image directory in `conditioning_data_dir`.

```toml
[general]
flip_aug = false
color_aug = false
resolution = [1024,1024]

[[datasets]]
batch_size = 8
enable_bucket = false

    [[datasets.subsets]]
    image_dir = "path/to/generated/image/dir"
    caption_extension = ".txt"
    conditioning_data_dir = "path/to/canny/image/dir"
```

## Credit

I would like to thank lllyasviel, the author of ControlNet, furusu, who provided me with advice on implementation and helped me solve problems, and ddPn08, who implemented the ControlNet dataset.

## Sample

Canny
![kohya_ss_girl_standing_at_classroom_smiling_to_the_viewer_class_78976b3e-0d4d-4ea0-b8e3-053ae493abbc](https://github.com/kohya-ss/sd-scripts/assets/52813779/37e9a736-649b-4c0f-ab26-880a1bf319b5)

![im_20230820104253_000_1](https://github.com/kohya-ss/sd-scripts/assets/52813779/c8896900-ab86-4120-932f-6e2ae17b77c0)

![im_20230820104302_000_1](https://github.com/kohya-ss/sd-scripts/assets/52813779/b12457a0-ee3c-450e-ba9a-b712d0fe86bb)

![im_20230820104310_000_1](https://github.com/kohya-ss/sd-scripts/assets/52813779/8845b8d9-804a-44ac-9618-113a28eac8a1)
