This repository contains training, generation and utility scripts for Stable Diffusion.

[日本語版README](./README-ja.md)

For easier use (GUI and PowerShell scripts etc...), please visit [the repository maintained by bmaltais](https://github.com/bmaltais/kohya_ss). Thanks to @bmaltais!

This repository contains the scripts for:

* DreamBooth training, including U-Net and Text Encoder
* fine-tuning (native training), including U-Net and Text Encoder
* image generation
* model conversion (supports 1.x and 2.x, Stable Diffision ckpt/safetensors and Diffusers)

## About requirements_*.txt

These files do not contain requirements for PyTorch and Diffusers. Because the versions of them depend on your environment. Please install PyTorch at first, then Diffusers. 

The scripts are tested with PyTorch 1.12.1 and 1.13.0, Diffusers 0.10.2.

## Links to how-to-use documents

All documents are in Japanese currently, and CUI based.

* [Environment setup and DreamBooth training guide](https://note.com/kohya_ss/n/nee3ed1649fb6)
* [Fine-tuning step-by-step guide](https://note.com/kohya_ss/n/nbf7ce8d80f29):
Including BLIP captioning and tagging by DeepDanbooru or WD14 tagger
* [Image generation](https://note.com/kohya_ss/n/n2693183a798e)
* [Model conversion](https://note.com/kohya_ss/n/n374f316fe4ad)
