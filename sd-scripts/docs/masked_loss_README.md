## Masked Loss

Masked loss is a feature that allows you to train only part of an image by calculating the loss only for the part specified by the mask of the input image. For example, if you want to train a character, you can train only the character part by masking it, ignoring the background.

There are two ways to specify the mask for masked loss.

- Using a mask image
- Using transparency (alpha channel) of the image

The sample uses the "AI image model training data" from [ZunZunPJ Illustration/3D Data](https://zunko.jp/con_illust.html).

### Using a mask image

This is a method of preparing a mask image corresponding to each training image. Prepare a mask image with the same file name as the training image and save it in a different directory from the training image.

- Training image
  ![image](https://github.com/kohya-ss/sd-scripts/assets/52813779/607c5116-5f62-47de-8b66-9c4a597f0441)
- Mask image
  ![image](https://github.com/kohya-ss/sd-scripts/assets/52813779/53e9b0f8-a4bf-49ed-882d-4026f84e8450)

```.toml
[[datasets.subsets]]
image_dir = "/path/to/a_zundamon"
caption_extension = ".txt"
conditioning_data_dir = "/path/to/a_zundamon_mask"
num_repeats = 8
```

The mask image is the same size as the training image, with the part to be trained drawn in white and the part to be ignored in black. It also supports grayscale (127 gives a loss weight of 0.5). The R channel of the mask image is used currently.

Use the dataset in the DreamBooth method, and save the mask image in the directory specified by `conditioning_data_dir`. It is the same as the ControlNet dataset, so please refer to [ControlNet-LLLite](train_lllite_README.md#Preparing-the-dataset) for details.

### Using transparency (alpha channel) of the image

The transparency (alpha channel) of the training image is used as a mask. The part with transparency 0 is ignored, the part with transparency 255 is trained. For semi-transparent parts, the loss weight changes according to the transparency (127 gives a weight of about 0.5).

![image](https://github.com/kohya-ss/sd-scripts/assets/52813779/0baa129b-446a-4aac-b98c-7208efb0e75e)

※Each image is a transparent PNG

Specify `--alpha_mask` in the training script options or specify `alpha_mask` in the subset of the dataset configuration file. For example, it will look like this.

```toml
[[datasets.subsets]]
image_dir = "/path/to/image/dir"
caption_extension = ".txt"
num_repeats = 8
alpha_mask = true
```

## Notes on training

- At the moment, only the dataset in the DreamBooth method is supported.
- The mask is applied after the size is reduced to 1/8, which is the size of the latents. Therefore, fine details (such as ahoge or earrings) may not be learned well. Some dilations of the mask may be necessary.
- If using masked loss, it may not be necessary to include parts that are not to be trained in the caption. (To be verified)
- In the case of `alpha_mask`, the latents cache is automatically regenerated when the enable/disable state of the mask is switched.
