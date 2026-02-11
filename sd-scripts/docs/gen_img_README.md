<!-- filepath: d:\\Work\\SD\\dev\\sd-scripts\\docs\\gen_img_README-en.md -->
This is an inference (image generation) script that supports SD 1.x and 2.x models, LoRA trained with this repository, ControlNet (only v1.0 has been confirmed to work), etc. It is used from the command line.

# Overview

* Inference (image generation) script.
* Supports SD 1.x, 2.x (base/v-parameterization), and SDXL models.
* Supports txt2img, img2img, and inpainting.
* Supports interactive mode, prompt reading from files, and continuous generation.
* The number of images generated per prompt line can be specified.
* The total number of repetitions can be specified.
* Supports not only `fp16` but also `bf16`.
* Supports xformers and SDPA (Scaled Dot-Product Attention).
* Extension of prompts to 225 tokens. Supports negative prompts and weighting.
* Supports various samplers from Diffusers.
* Supports clip skip (uses the output of the nth layer from the end) of Text Encoder.
* Separate loading of VAE, supports VAE batch processing and slicing for memory saving.
* Highres. fix (original implementation and Gradual Latent), upscale support.
* LoRA, DyLoRA support. Supports application rate specification, simultaneous use of multiple LoRAs, and weight merging.
* Supports Attention Couple, Regional LoRA.
* Supports ControlNet (v1.0/v1.1), ControlNet-LLLite.
* It is not possible to switch models midway, but it can be handled by creating a batch file.

# Basic Usage

## Image Generation in Interactive Mode

Enter as follows:

```batchfile
python gen_img.py --ckpt <model_name> --outdir <image_output_destination> --xformers --fp16 --interactive
```

Specify the model (Stable Diffusion checkpoint file or Diffusers model folder) in the `--ckpt` option and the image output destination folder in the `--outdir` option.

Specify the use of xformers with the `--xformers` option (remove it if you do not use xformers). The `--fp16` option performs inference in fp16 (single precision). For RTX 30 series GPUs, you can also perform inference in bf16 (bfloat16) with the `--bf16` option.

The `--interactive` option specifies interactive mode.

If you are using Stable Diffusion 2.0 (or a model with additional training from it), add the `--v2` option. If you are using a model that uses v-parameterization (`768-v-ema.ckpt` and models with additional training from it), add `--v_parameterization` as well.

If the `--v2` specification is incorrect, an error will occur when loading the model. If the `--v_parameterization` specification is incorrect, a brown image will be displayed.

When `Type prompt:` is displayed, enter the prompt.

![image](https://user-images.githubusercontent.com/52813779/235343115-f3b8ac82-456d-4aab-9724-0cc73c4534aa.png)

*If the image is not displayed and an error occurs, headless (no screen display function) OpenCV may be installed. Install normal OpenCV with `pip install opencv-python`. Alternatively, stop image display with the `--no_preview` option.

Select the image window and press any key to close the window and enter the next prompt. Press Ctrl+Z and then Enter in the prompt to close the script.

## Batch Generation of Images with a Single Prompt

Enter as follows (actually entered on one line):

```batchfile
python gen_img.py --ckpt <model_name> --outdir <image_output_destination> \
    --xformers --fp16 --images_per_prompt <number_of_images_to_generate> --prompt "<prompt>"
```

Specify the number of images to generate per prompt with the `--images_per_prompt` option. Specify the prompt with the `--prompt` option. If it contains spaces, enclose it in double quotes.

You can specify the batch size with the `--batch_size` option (described later).

## Batch Generation by Reading Prompts from a File

Enter as follows:

```batchfile
python gen_img.py --ckpt <model_name> --outdir <image_output_destination> \
    --xformers --fp16 --from_file <prompt_file_name>
```

Specify the file containing the prompts with the `--from_file` option. Write one prompt per line. You can specify the number of images to generate per line with the `--images_per_prompt` option.

## Using Negative Prompts and Weighting

If you write `--n` in the prompt options (specified like `--x` in the prompt, described later), the following will be a negative prompt.

Also, weighting with `()` and `[]`, `(xxx:1.3)`, etc., similar to AUTOMATIC1111's Web UI, is possible (the implementation is copied from Diffusers' [Long Prompt Weighting Stable Diffusion](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#long-prompt-weighting-stable-diffusion)).

It can be specified similarly for prompt specification from the command line and prompt reading from files.

![image](https://user-images.githubusercontent.com/52813779/235343128-e79cd768-ec59-46f5-8395-fce9bdc46208.png)

# Main Options

Specify from the command line.

## Model Specification

- `--ckpt <model_name>`: Specifies the model name. The `--ckpt` option is mandatory. You can specify a Stable Diffusion checkpoint file, a Diffusers model folder, or a Hugging Face model ID.

- `--v1`: Specify when using Stable Diffusion 1.x series models. This is the default behavior.

- `--v2`: Specify when using Stable Diffusion 2.x series models. Not required for 1.x series.

- `--sdxl`: Specify when using Stable Diffusion XL models.

- `--v_parameterization`: Specify when using models that use v-parameterization (`768-v-ema.ckpt` and models with additional training from it, Waifu Diffusion v1.5, etc.).

    If the `--v2` or `--sdxl` specification is incorrect, an error will occur when loading the model. If the `--v_parameterization` specification is incorrect, a brown image will be displayed.

- `--zero_terminal_snr`: Modifies the noise scheduler betas to enforce zero terminal SNR.

- `--pyramid_noise_prob`: Specifies the probability of applying pyramid noise.

- `--pyramid_noise_discount_range`: Specifies the discount range for pyramid noise.

- `--noise_offset_prob`: Specifies the probability of applying noise offset.

- `--noise_offset_range`: Specifies the range of noise offset.

- `--vae`: Specifies the VAE to use. If not specified, the VAE in the model will be used.

- `--tokenizer_cache_dir`: Specifies the cache directory for the tokenizer (for offline usage).

## Image Generation and Output

- `--interactive`: Operates in interactive mode. Images are generated when prompts are entered.

- `--prompt <prompt>`: Specifies the prompt. If it contains spaces, enclose it in double quotes.

- `--from_file <prompt_file_name>`: Specifies the file containing the prompts. Write one prompt per line. Image size and guidance scale can be specified with prompt options (described later).

- `--from_module <module_file>`: Loads prompts from a Python module. The module should implement a `get_prompter(args, pipe, networks)` function.

- `--prompter_module_args`: Specifies additional arguments to pass to the prompter module.

- `--W <image_width>`: Specifies the width of the image. The default is `512`.

- `--H <image_height>`: Specifies the height of the image. The default is `512`.

- `--steps <number_of_steps>`: Specifies the number of sampling steps. The default is `50`.

- `--scale <guidance_scale>`: Specifies the unconditional guidance scale. The default is `7.5`.

- `--sampler <sampler_name>`: Specifies the sampler. The default is `ddim`.
    `ddim`, `pndm`, `lms`, `euler`, `euler_a`, `heun`, `dpm_2`, `dpm_2_a`, `dpmsolver`, `dpmsolver++`, `dpmsingle`, `k_lms`, `k_euler`, `k_euler_a`, `k_dpm_2`, `k_dpm_2_a` can be specified.

- `--outdir <image_output_destination_folder>`: Specifies the output destination for images.

- `--images_per_prompt <number_of_images_to_generate>`: Specifies the number of images to generate per prompt. The default is `1`.

- `--clip_skip <number_of_skips>`: Specifies which layer from the end of CLIP to use. Default is 1 for SD1/2, 2 for SDXL.

- `--max_embeddings_multiples <multiplier>`: Specifies how many times the CLIP input/output length should be multiplied by the default (75). If not specified, it remains 75. For example, specifying 3 makes the input/output length 225.

- `--negative_scale`: Specifies the guidance scale for unconditioning individually. Implemented with reference to [this article by gcem156](https://note.com/gcem156/n/ne9a53e4a6f43).

- `--emb_normalize_mode`: Specifies the embedding normalization mode. Options are "original" (default), "abs", and "none". This affects how prompt weights are normalized.

- `--force_scheduler_zero_steps_offset`: Forces the scheduler step offset to zero regardless of the `steps_offset` value in the scheduler configuration.

## SDXL-Specific Options

When using SDXL models (with `--sdxl` flag), additional conditioning options are available:

- `--original_height`: Specifies the original height for SDXL conditioning. This affects the model's understanding of the target resolution.

- `--original_width`: Specifies the original width for SDXL conditioning. This affects the model's understanding of the target resolution.

- `--original_height_negative`: Specifies the original height for SDXL negative conditioning.

- `--original_width_negative`: Specifies the original width for SDXL negative conditioning.

- `--crop_top`: Specifies the crop top offset for SDXL conditioning.

- `--crop_left`: Specifies the crop left offset for SDXL conditioning.

## Adjusting Memory Usage and Generation Speed

- `--batch_size <batch_size>`: Specifies the batch size. The default is `1`. A larger batch size consumes more memory but speeds up generation.

- `--vae_batch_size <VAE_batch_size>`: Specifies the VAE batch size. The default is the same as the batch size.
    Since VAE consumes more memory, memory shortages may occur after denoising (after the step reaches 100%). In such cases, reduce the VAE batch size.

- `--vae_slices <number_of_slices>`: Splits the image into slices for VAE processing to reduce VRAM usage. None (default) for no splitting. Values like 16 or 32 are recommended. Enabling this is slower but uses less VRAM.

- `--no_half_vae`: Prevents using fp16/bf16 precision for VAE processing. Uses fp32 instead. Use this if you encounter VAE-related issues or artifacts.

- `--xformers`: Specify when using xformers.

- `--sdpa`: Use scaled dot-product attention in PyTorch 2 for optimization.

- `--diffusers_xformers`: Use xformers via Diffusers (note: incompatible with Hypernetworks).

- `--fp16`: Performs inference in fp16 (single precision). If neither `fp16` nor `bf16` is specified, inference is performed in fp32 (single precision).

- `--bf16`: Performs inference in bf16 (bfloat16). Can only be specified for RTX 30 series GPUs. The `--bf16` option will cause an error on GPUs other than the RTX 30 series. It seems that `bf16` is less likely to result in NaN (black image) inference results than `fp16`.

## Using Additional Networks (LoRA, etc.)

- `--network_module`: Specifies the additional network to use. For LoRA, specify `--network_module networks.lora`. To use multiple LoRAs, specify like `--network_module networks.lora networks.lora networks.lora`.

- `--network_weights`: Specifies the weight file of the additional network to use. Specify like `--network_weights model.safetensors`. To use multiple LoRAs, specify like `--network_weights model1.safetensors model2.safetensors model3.safetensors`. The number of arguments should be the same as the number specified in `--network_module`.

- `--network_mul`: Specifies how many times to multiply the weight of the additional network to use. The default is `1`. Specify like `--network_mul 0.8`. To use multiple LoRAs, specify like `--network_mul 0.4 0.5 0.7`. The number of arguments should be the same as the number specified in `--network_module`.

- `--network_merge`: Merges the weights of the additional networks to be used in advance with the weights specified in `--network_mul`. Cannot be used simultaneously with `--network_pre_calc`. The prompt option `--am` and Regional LoRA can no longer be used, but generation will be accelerated to the same extent as when LoRA is not used.

- `--network_pre_calc`: Calculates the weights of the additional network to be used in advance for each generation. The prompt option `--am` can be used. Generation is accelerated to the same extent as when LoRA is not used, but time is required to calculate the weights before generation, and memory usage also increases slightly. It is disabled when Regional LoRA is used.

- `--network_regional_mask_max_color_codes`: Specifies the maximum number of color codes to use for regional masks. If not specified, masks are applied by channel. Used with Regional LoRA to control the number of regions that can be defined by colors in the mask.

- `--network_args`: Specifies additional arguments to pass to the network module in key=value format. For example: `--network_args "alpha=1.0,dropout=0.1"`.

- `--network_merge_n_models`: When using network merging, specifies the number of models to merge (instead of merging all loaded networks).

# Examples of Main Option Specifications

The following is an example of batch generating 64 images with the same prompt and a batch size of 4.

```batchfile
python gen_img.py --ckpt model.ckpt --outdir outputs \
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a \
    --steps 32 --batch_size 4 --images_per_prompt 64 \
    --prompt "beautiful flowers --n monochrome"
```

The following is an example of batch generating 10 images each for prompts written in a file, with a batch size of 4.

```batchfile
python gen_img.py --ckpt model.ckpt --outdir outputs \
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a \
    --steps 32 --batch_size 4 --images_per_prompt 10 \
    --from_file prompts.txt
```

Example of using Textual Inversion (described later) and LoRA.

```batchfile
python gen_img.py --ckpt model.safetensors \
    --scale 8 --steps 48 --outdir txt2img --xformers \
    --W 512 --H 768 --fp16 --sampler k_euler_a \
    --textual_inversion_embeddings goodembed.safetensors negprompt.pt \
    --network_module networks.lora networks.lora \
    --network_weights model1.safetensors model2.safetensors \
    --network_mul 0.4 0.8 \
    --clip_skip 2 --max_embeddings_multiples 1 \
    --batch_size 8 --images_per_prompt 1 --interactive
```

# Prompt Options

In the prompt, you can specify various options from the prompt with "two hyphens + n alphabetic characters" like `--n`. It is valid whether specifying the prompt from interactive mode, command line, or file.

Please put spaces before and after the prompt option specification `--n`.

- `--n`: Specifies a negative prompt.

- `--w`: Specifies the image width. Overrides the command line specification.

- `--h`: Specifies the image height. Overrides the command line specification.

- `--s`: Specifies the number of steps. Overrides the command line specification.

- `--d`: Specifies the random seed for this image. If `--images_per_prompt` is specified, specify multiple seeds separated by commas, like "--d 1,2,3,4".
    *For various reasons, the generated image may differ from the Web UI even with the same random seed.

- `--l`: Specifies the guidance scale. Overrides the command line specification.

- `--t`: Specifies the strength of img2img (described later). Overrides the command line specification.

- `--nl`: Specifies the guidance scale for negative prompts (described later). Overrides the command line specification.

- `--am`: Specifies the weight of the additional network. Overrides the command line specification. If using multiple additional networks, specify them separated by __commas__, like `--am 0.8,0.5,0.3`.

- `--ow`: Specifies original_width for SDXL.

- `--oh`: Specifies original_height for SDXL.

- `--nw`: Specifies original_width_negative for SDXL.

- `--nh`: Specifies original_height_negative for SDXL.

- `--ct`: Specifies crop_top for SDXL.

- `--cl`: Specifies crop_left for SDXL.

- `--c`: Specifies the CLIP prompt.

- `--f`: Specifies the generated file name.

- `--glt`: Specifies the timestep to start increasing the size of the latent for Gradual Latent. Overrides the command line specification.

- `--glr`: Specifies the initial size of the latent for Gradual Latent as a ratio. Overrides the command line specification.

- `--gls`: Specifies the ratio to increase the size of the latent for Gradual Latent. Overrides the command line specification.

- `--gle`: Specifies the interval to increase the size of the latent for Gradual Latent. Overrides the command line specification.

*Specifying these options may cause the batch to be executed with a size smaller than the batch size (because they cannot be generated collectively if these values are different). (You don't have to worry too much, but when reading prompts from a file and generating, arranging prompts with the same values for these options will improve efficiency.)

Example:
```
(masterpiece, best quality), 1girl, in shirt and plated skirt, standing at street under cherry blossoms, upper body, [from below], kind smile, looking at another, [goodembed] --n realistic, real life, (negprompt), (lowres:1.1), (worst quality:1.2), (low quality:1.1), bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry --w 960 --h 640 --s 28 --d 1
```

![image](https://user-images.githubusercontent.com/52813779/235343446-25654172-fff4-4aaf-977a-20d262b51676.png)

# Wildcards in Prompts (Dynamic Prompts)

Dynamic Prompts (Wildcard) notation is supported. While not exactly the same as the Web UI extension, the following features are available.

- `{A|B|C}` : Randomly selects one from A, B, or C.
- `{e$$A|B|C}` : Uses all of A, B, and C in order (enumeration). If there are multiple `{e$$...}` in the prompt, all combinations will be generated.
  - Example: `{e$$red|blue} flower, {e$$1girl|2girls}` -> Generates 4 images: `red flower, 1girl`, `red flower, 2girls`, `blue flower, 1girl`, `blue flower, 2girls`.
- `{n$$A|B|C}` : Randomly selects n items from A, B, C and combines them.
  - Example: `{2$$A|B|C}` -> `A, B` or `B, C`, etc.
- `{n-m$$A|B|C}` : Randomly selects between n and m items from A, B, C and combines them.
- `{$$sep$$A|B|C}` : Combines selected items with `sep` (default is `, `).
  - Example: `{2$$ and $$A|B|C}` -> `A and B`, etc.

These can be used in combination.

# img2img

## Options

- `--image_path`: Specifies the image to use for img2img. Specify like `--image_path template.png`. If a folder is specified, images in that folder will be used sequentially.

- `--strength`: Specifies the strength of img2img. Specify like `--strength 0.8`. The default is `0.8`.

- `--sequential_file_name`: Specifies whether to make file names sequential. If specified, the generated file names will be sequential starting from `im_000001.png`.

- `--use_original_file_name`: If specified, the generated file name will be prepended with the original file name (for img2img mode).

- `--clip_vision_strength`: Enables CLIP Vision Conditioning for img2img with the specified strength. Uses the CLIP Vision model to enhance conditioning from the input image.

## Command Line Execution Example

```batchfile
python gen_img.py --ckpt trinart_characters_it4_v1_vae_merged.ckpt \
    --outdir outputs --xformers --fp16 --scale 12.5 --sampler k_euler --steps 32 \
    --image_path template.png --strength 0.8 \
    --prompt "1girl, cowboy shot, brown hair, pony tail, brown eyes, \
          sailor school uniform, outdoors \
          --n lowres, bad anatomy, bad hands, error, missing fingers, cropped, \
          worst quality, low quality, normal quality, jpeg artifacts, (blurry), \
          hair ornament, glasses" \
    --batch_size 8 --images_per_prompt 32
```

If a folder is specified in the `--image_path` option, images in that folder will be read sequentially. The number of images generated will be the number of prompts, not the number of images, so please match the number of images to img2img and the number of prompts by specifying the `--images_per_prompt` option.

Files are read sorted by file name. Note that the sort order is string order (not `1.jpg -> 2.jpg -> 10.jpg` but `1.jpg -> 10.jpg -> 2.jpg`), so please pad the beginning with zeros (e.g., `01.jpg -> 02.jpg -> 10.jpg`).

## Upscale using img2img

If you specify the generated image size with the `--W` and `--H` command line options during img2img, the original image will be resized to that size before img2img.

Also, if the original image for img2img was generated by this script, omitting the prompt will retrieve the prompt from the original image's metadata and use it as is. This allows you to perform only the 2nd stage operation of Highres. fix.

## Inpainting during img2img

You can specify an image and a mask image for inpainting (inpainting models are not supported, it simply performs img2img on the mask area).

The options are as follows:

- `--mask_image`: Specifies the mask image. Similar to `--img_path`, if a folder is specified, images in that folder will be used sequentially.

The mask image is a grayscale image, and the white parts will be inpainted. It is recommended to gradient the boundaries to make it somewhat smooth.

![image](https://user-images.githubusercontent.com/52813779/235343795-9eaa6d98-02ff-4f32-b089-80d1fc482453.png)

# Other Features

## Textual Inversion

Specify the embeddings to use with the `--textual_inversion_embeddings` option (multiple specifications possible). By using the file name without the extension in the prompt, that embedding will be used (same usage as Web UI). It can also be used in negative prompts.

As models, you can use Textual Inversion models trained with this repository and Textual Inversion models trained with Web UI (image embedding is not supported).

## Highres. fix

This is a similar feature to the one in AUTOMATIC1111's Web UI (it may differ in various ways as it is an original implementation). It first generates a smaller image and then uses that image as a base for img2img to generate a large resolution image while preventing the entire image from collapsing.

The number of steps for the 2nd stage is calculated from the values of the `--steps` and `--strength` options (`steps*strength`).

Cannot be used with img2img.

The following options are available:

- `--highres_fix_scale`: Enables Highres. fix and specifies the size of the image generated in the 1st stage as a magnification. If the final output is 1024x1024 and you want to generate a 512x512 image first, specify like `--highres_fix_scale 0.5`. Please note that this is the reciprocal of the specification in Web UI.

- `--highres_fix_steps`: Specifies the number of steps for the 1st stage image. The default is `28`.

- `--highres_fix_save_1st`: Specifies whether to save the 1st stage image.

- `--highres_fix_latents_upscaling`: If specified, the 1st stage image will be upscaled on a latent basis during 2nd stage image generation (only bilinear is supported). If not specified, the image will be upscaled with LANCZOS4.

- `--highres_fix_upscaler`: Uses an arbitrary upscaler for the 2nd stage. Currently, only `--highres_fix_upscaler tools.latent_upscaler` is supported.

- `--highres_fix_upscaler_args`: Specifies the arguments to pass to the upscaler specified with `--highres_fix_upscaler`.
    For `tools.latent_upscaler`, specify the weight file like `--highres_fix_upscaler_args "weights=D:\\Work\\SD\\Models\\others\\etc\\upscaler-v1-e100-220.safetensors"`.

- `--highres_fix_disable_control_net`: Disables ControlNet for the 2nd stage of Highres fix. By default, ControlNet is used in both stages.

Command line example:

```batchfile
python gen_img.py  --ckpt trinart_characters_it4_v1_vae_merged.ckpt\
    --n_iter 1 --scale 7.5 --W 1024 --H 1024 --batch_size 1 --outdir ../txt2img \
    --steps 48 --sampler ddim --fp16 \
    --xformers \
    --images_per_prompt 1  --interactive \
    --highres_fix_scale 0.5 --highres_fix_steps 28 --strength 0.5
```

## Deep Shrink

Deep Shrink is a technique that optimizes the generation process by using different depths of the UNet at different timesteps. It can improve generation quality and efficiency.

The following options are available:

- `--ds_depth_1`: Enables Deep Shrink with this depth for the first phase. Valid values are 0 to 8.

- `--ds_timesteps_1`: Applies Deep Shrink depth 1 until this timestep. Default is 650.

- `--ds_depth_2`: Specifies the depth for the second phase of Deep Shrink.

- `--ds_timesteps_2`: Applies Deep Shrink depth 2 until this timestep. Default is 650.

- `--ds_ratio`: Specifies the ratio for downsampling in Deep Shrink. Default is 0.5.

These parameters can also be specified through prompt options:

- `--dsd1`: Specifies Deep Shrink depth 1 from the prompt.
  
- `--dst1`: Specifies Deep Shrink timestep 1 from the prompt.
  
- `--dsd2`: Specifies Deep Shrink depth 2 from the prompt.
  
- `--dst2`: Specifies Deep Shrink timestep 2 from the prompt.
  
- `--dsr`: Specifies Deep Shrink ratio from the prompt.

*Additional prompt options for Gradual Latent (requires `euler_a` sampler):*

- `--glt`: Specifies the timestep to start increasing the size of the latent for Gradual Latent. Overrides the command line specification.

- `--glr`: Specifies the initial size of the latent for Gradual Latent as a ratio. Overrides the command line specification.

- `--gls`: Specifies the ratio to increase the size of the latent for Gradual Latent. Overrides the command line specification.

- `--gle`: Specifies the interval to increase the size of the latent for Gradual Latent. Overrides the command line specification.

## ControlNet

Currently, only ControlNet 1.0 has been confirmed to work. Only Canny is supported for preprocessing.

The following options are available:

- `--control_net_models`: Specifies the ControlNet model file.
    If multiple are specified, they will be switched and used for each step (differs from the implementation of the ControlNet extension in Web UI). Supports both diff and normal.

- `--guide_image_path`: Specifies the hint image to use for ControlNet. Similar to `--img_path`, if a folder is specified, images in that folder will be used sequentially. For models other than Canny, please perform preprocessing beforehand.

- `--control_net_preps`: Specifies the preprocessing for ControlNet. Multiple specifications are possible, similar to `--control_net_models`. Currently, only canny is supported. If preprocessing is not used for the target model, specify `none`.
   For canny, you can specify thresholds 1 and 2 separated by `_`, like `--control_net_preps canny_63_191`.

- `--control_net_weights`: Specifies the weight when applying ControlNet (`1.0` for normal, `0.5` for half influence). Multiple specifications are possible, similar to `--control_net_models`.

- `--control_net_ratios`: Specifies the range of steps to apply ControlNet. If `0.5`, ControlNet is applied up to half the number of steps. Multiple specifications are possible, similar to `--control_net_models`.

Command line example:

```batchfile
python gen_img.py --ckpt model_ckpt --scale 8 --steps 48 --outdir txt2img --xformers \
    --W 512 --H 768 --bf16 --sampler k_euler_a \
    --control_net_models diff_control_sd15_canny.safetensors --control_net_weights 1.0 \
    --guide_image_path guide.png --control_net_ratios 1.0 --interactive
```

## ControlNet-LLLite

ControlNet-LLLite is a lightweight alternative to ControlNet that can be used for similar guidance purposes.

The following options are available:

- `--control_net_lllite_models`: Specifies the ControlNet-LLLite model files.

- `--control_net_multipliers`: Specifies the multiplier for ControlNet-LLLite (similar to weights).

- `--control_net_ratios`: Specifies the ratio of steps to apply ControlNet-LLLite.

Note that ControlNet and ControlNet-LLLite cannot be used at the same time.

## Attention Couple + Regional LoRA

This is a feature that allows you to divide the prompt into several parts and specify which region in the image each prompt should be applied to. There are no individual options, but it is specified with `mask_path` and the prompt.

First, define multiple parts using ` AND ` in the prompt. Region specification can be done for the first three parts, and subsequent parts are applied to the entire image. Negative prompts are applied to the entire image.

In the following, three parts are defined with AND.

```
shs 2girls, looking at viewer, smile AND bsb 2girls, looking back AND 2girls --n bad quality, worst quality
```

Next, prepare a mask image. The mask image is a color image, and each RGB channel corresponds to the part separated by AND in the prompt. Also, if the value of a certain channel is all 0, it is applied to the entire image.

In the example above, the R channel corresponds to `shs 2girls, looking at viewer, smile`, the G channel to `bsb 2girls, looking back`, and the B channel to `2girls`. If you use a mask image like the following, since there is no specification for the B channel, `2girls` will be applied to the entire image.

![image](https://user-images.githubusercontent.com/52813779/235343061-b4dc9392-3dae-4831-8347-1e9ae5054251.png)

The mask image is specified with `--mask_path`. Currently, only one image is supported. It is automatically resized and applied to the specified image size.

It can also be combined with ControlNet (combination with ControlNet is recommended for detailed position specification).

If LoRA is specified, multiple LoRAs specified with `--network_weights` will correspond to each part of AND. As a current constraint, the number of LoRAs must be the same as the number of AND parts.

# Other Options

- `--no_preview`: Does not display preview images in interactive mode. Specify this if OpenCV is not installed or if you want to check the output files directly.

- `--n_iter`: Specifies the number of times to repeat generation. The default is 1. Specify this when you want to perform generation multiple times when reading prompts from a file.

- `--tokenizer_cache_dir`: Specifies the cache directory for the tokenizer. (Work in progress)

- `--seed`: Specifies the random seed. When generating one image, it is the seed for that image. When generating multiple images, it is the seed for the random numbers used to generate the seeds for each image (when generating multiple images with `--from_file`, specifying the `--seed` option will make each image have the same seed when executed multiple times).

- `--iter_same_seed`: When there is no random seed specification in the prompt, the same seed is used for all repetitions of `--n_iter`. Used to unify and compare seeds between multiple prompts specified with `--from_file`.

- `--shuffle_prompts`: Shuffles the order of prompts in iteration. Useful when using `--from_file` with multiple prompts.

- `--diffusers_xformers`: Uses Diffuser's xformers.

- `--opt_channels_last`: Arranges tensor channels last during inference. May speed up in some cases.

- `--network_show_meta`: Displays the metadata of the additional network.


---

# About Gradual Latent

Gradual Latent is a Hires fix that gradually increases the size of the latent.  `gen_img.py`, `sdxl_gen_img.py`, and `gen_img.py` have the following options.

- `--gradual_latent_timesteps`: Specifies the timestep to start increasing the size of the latent. The default is None, which means Gradual Latent is not used. Please try around 750 at first.
- `--gradual_latent_ratio`: Specifies the initial size of the latent. The default is 0.5, which means it starts with half the default latent size.
- `--gradual_latent_ratio_step`: Specifies the ratio to increase the size of the latent. The default is 0.125, which means the latent size is gradually increased to 0.625, 0.75, 0.875, 1.0.
- `--gradual_latent_ratio_every_n_steps`: Specifies the interval to increase the size of the latent. The default is 3, which means the latent size is increased every 3 steps.
- `--gradual_latent_s_noise`: Specifies the s_noise parameter for Gradual Latent. Default is 1.0.
- `--gradual_latent_unsharp_params`: Specifies unsharp mask parameters for Gradual Latent in the format: ksize,sigma,strength,target-x (target-x: 1=True, 0=False). Recommended values: `3,0.5,0.5,1` or `3,1.0,1.0,0`.

Each option can also be specified with prompt options, `--glt`, `--glr`, `--gls`, `--gle`.

__Please specify `euler_a` for the sampler.__ Because the source code of the sampler is modified. It will not work with other samplers.

It is more effective with SD 1.5. It is quite subtle with SDXL.
