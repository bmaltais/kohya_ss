# How to use

##Diffusers to Stable Diffusion .ckpt conversion

Specify the folder of the source model and the destination .ckpt file as follows (actually written on one line). The v1/v2 version is automatically determined.

```
python convert_diffusers20_original_sd.py ..\models\diffusers_model 
    ..\models\sd.ckpt
```

Note that v2 Diffusers' Text Encoder has only 22 layers, and if you convert it to Stable Diffusion as it is, the weights will be insufficient, so the weights of the 22 layers will be copied as the 23rd layer. The weight of the 23rd layer is not used during image generation, so it has no effect. Similarly, text_projection logit_scale also adds dummy weights (it doesn't seem to be used for image generation).

## Stable Diffusion .ckpt to Diffusers

Enter the following:

```
python convert_diffusers20_original_sd.py ..\models\sd.ckpt 
    ..\models\diffusers_model 
    --v2 --reference_model stabilityai/stable-diffusion-2
```

Specify the .ckpt file and the destination folder as arguments.
Model judgment is not possible, so please use the `--v1` option or the `--v2` option depending on the model.

Also, since `.ckpt` does not contain scheduler and tokenizer information, you need to copy them from some existing Diffusers model. Please specify with `--reference_model`. You can specify the HuggingFace id or a local model directory.

If you don't have a local model, you can specify "stabilityai/stable-diffusion-2" or "stabilityai/stable-diffusion-2-base" for v2.
For v1.4/1.5, "CompVis/stable-diffusion-v1-4" is fine (v1.4 and v1.5 seem to be the same).

## What can you do?

`--fp16 / --bf16 / --float`

You can specify the data format when saving checkpoint. --fp16 only, also valid when loading Diffusers models.

`--epoch / --global_step`

When saving checkpoint, write epoch and global_step with the specified values. If not specified, both will be 0.

## Conclusion

Some people may be troubled by the Diffusers model due to the poor inference environment. I hope it helps a little.

(Note that converting the data format from checkpoint to checkpoint is also possible, although it has not been tested.) ）