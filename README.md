This repository contains training, generation and utility scripts for Stable Diffusion.

[__Change History__](#change-history) is moved to the bottom of the page.
更新履歴は[ページ末尾](#change-history)に移しました。

[日本語版README](./README-ja.md)

For easier use (GUI and PowerShell scripts etc...), please visit [the repository maintained by bmaltais](https://github.com/bmaltais/kohya_ss). Thanks to @bmaltais!

This repository contains the scripts for:

* DreamBooth training, including U-Net and Text Encoder
* Fine-tuning (native training), including U-Net and Text Encoder
* LoRA training
* Texutl Inversion training
* Image generation
* Model conversion (supports 1.x and 2.x, Stable Diffision ckpt/safetensors and Diffusers)

__Stable Diffusion web UI now seems to support LoRA trained by ``sd-scripts``.__ (SD 1.x based only) Thank you for great work!!! 

## About requirements.txt

These files do not contain requirements for PyTorch. Because the versions of them depend on your environment. Please install PyTorch at first (see installation guide below.) 

The scripts are tested with PyTorch 1.12.1 and 1.13.0, Diffusers 0.10.2.

## Links to how-to-use documents

All documents are in Japanese currently.

* [Training guide - common](./train_README-ja.md) : data preparation, options etc...
    * [Dataset config](./config_README-ja.md)
* [DreamBooth training guide](./train_db_README-ja.md)
* [Step by Step fine-tuning guide](./fine_tune_README_ja.md):
* [training LoRA](./train_network_README-ja.md)
* [training Textual Inversion](./train_ti_README-ja.md)
* note.com [Image generation](https://note.com/kohya_ss/n/n2693183a798e)
* note.com [Model conversion](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windows Required Dependencies

Python 3.10.6 and Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Windows Installation

Open a regular Powershell terminal and type the following inside:

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config
```

update: ``python -m venv venv`` is seemed to be safer than ``python -m venv --system-site-packages venv`` (some user have packages in global python).

Answers to accelerate config:

```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

note: Some user reports ``ValueError: fp16 mixed precision requires a GPU`` is occurred in training. In this case, answer `0` for the 6th question: 
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`` 

(Single GPU with id `0` will be used.)

### about PyTorch and xformers

Other versions of PyTorch and xformers seem to have problems with training.
If there is no other reason, please install the specified version.

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

## Credits

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for great work!

The LoRA expansion to Conv2d 3x3 was initially released by cloneofsimo and its effectiveness was demonstrated at [LoCon](https://github.com/KohakuBlueleaf/LoCon) by KohakuBlueleaf. Thank you so much KohakuBlueleaf!

## License

The majority of scripts is licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's and LoCon), however portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause

## Change History

- 4 Apr. 2023, 2023/4/4:
  - Add options to `train_network.py` to specify block weights for learning rates. Thanks to u-haru for the great contribution!
    - Specify the weights of 25 blocks for the full model.
      - No LoRA corresponds to the first block, but 25 blocks are specified for compatibility with 'LoRA block weight' etc. Also, if you do not expand to conv2d3x3, some blocks do not have LoRA, but please specify 25 values ​​for the argument for consistency.
    - Specify the following arguments with `--network_args`.
    - `down_lr_weight` : Specify the learning rate weight of the down blocks of U-Net. The following can be specified.
      - The weight for each block: Specify 12 numbers such as `"down_lr_weight=0,0,0,0,0,0,1,1,1,1,1,1"`.
      - Specify from preset: Specify such as `"down_lr_weight=sine"` (the weights by sine curve). sine, cosine, linear, reverse_linear, zeros can be specified. Also, if you add `+number` such as `"down_lr_weight=cosine+.25"`, the specified number is added (such as 0.25~1.25).
    - `mid_lr_weight` : Specify the learning rate weight of the mid block of U-Net. Specify one number such as `"down_lr_weight=0.5"`.
    - `up_lr_weight` : Specify the learning rate weight of the up blocks of U-Net. The same as down_lr_weight.
    - If you omit the some arguments, the 1.0 is used. Also, if you set the weight to 0, the LoRA modules of that block are not created.
    - `block_lr_zero_threshold` : If the weight is not more than this value, the LoRA module is not created. The default is 0.

  - Add options to `train_network.py` to specify block dims (ranks) for variable rank.
    - Specify 25 values ​​for the full model of 25 blocks. Some blocks do not have LoRA, but specify 25 values ​​always.
    - Specify the following arguments with `--network_args`.
    - `block_dims` : Specify the dim (rank) of each block. Specify 25 numbers such as `"block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"`.
    - `block_alphas` : Specify the alpha of each block. Specify 25 numbers as with block_dims. If omitted, the value of network_alpha is used.
    - `conv_block_dims` : Expand LoRA to Conv2d 3x3 and specify the dim (rank) of each block.
    - `conv_block_alphas` : Specify the alpha of each block when expanding LoRA to Conv2d 3x3. If omitted, the value of conv_alpha is used.

  - 階層別学習率を `train_network.py` で指定できるようになりました。u-haru 氏の多大な貢献に感謝します。
    - フルモデルの25個のブロックの重みを指定できます。
      - 最初のブロックに該当するLoRAは存在しませんが、階層別LoRA適用等との互換性のために25個としています。またconv2d3x3に拡張しない場合も一部のブロックにはLoRAが存在しませんが、記述を統一するため常に25個の値を指定してください。
    -`--network_args` で以下の引数を指定してください。
    - `down_lr_weight` : U-Netのdown blocksの学習率の重みを指定します。以下が指定可能です。
      - ブロックごとの重み : `"down_lr_weight=0,0,0,0,0,0,1,1,1,1,1,1"` のように12個の数値を指定します。
      - プリセットからの指定 : `"down_lr_weight=sine"` のように指定します（サインカーブで重みを指定します）。sine, cosine, linear, reverse_linear, zeros が指定可能です。また `"down_lr_weight=cosine+.25"` のように `+数値` を追加すると、指定した数値を加算します（0.25~1.25になります）。
    - `mid_lr_weight` : U-Netのmid blockの学習率の重みを指定します。`"down_lr_weight=0.5"` のように数値を一つだけ指定します。
    - `up_lr_weight` : U-Netのup blocksの学習率の重みを指定します。down_lr_weightと同様です。
    - 指定を省略した部分は1.0として扱われます。また重みを0にするとそのブロックのLoRAモジュールは作成されません。
    - `block_lr_zero_threshold` : 重みがこの値以下の場合、LoRAモジュールを作成しません。デフォルトは0です。

  - 階層別dim (rank)を `train_network.py` で指定できるようになりました。
    - フルモデルの25個のブロックのdim (rank)を指定できます。階層別学習率と同様に一部のブロックにはLoRAが存在しない場合がありますが、常に25個の値を指定してください。
    - `--network_args` で以下の引数を指定してください。
    - `block_dims` : 各ブロックのdim (rank)を指定します。`"block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"` のように25個の数値を指定します。
    - `block_alphas` : 各ブロックのalphaを指定します。block_dimsと同様に25個の数値を指定します。省略時はnetwork_alphaの値が使用されます。
    - `conv_block_dims` : LoRAをConv2d 3x3に拡張し、各ブロックのdim (rank)を指定します。
    - `conv_block_alphas` : LoRAをConv2d 3x3に拡張したときの各ブロックのalphaを指定します。省略時はconv_alphaの値が使用されます。

  - 階層別学習率コマンドライン指定例 / Examples of block learning rate command line specification:

    ` --network_args "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5" "mid_lr_weight=2.0" "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5"`
  
    ` --network_args "block_lr_zero_threshold=0.1" "down_lr_weight=sine+.5" "mid_lr_weight=1.5" "up_lr_weight=cosine+.5"`

  - 階層別dim (rank)コマンドライン指定例 / Examples of block dim (rank) command line specification:

    ` --network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2"`
  
    ` --network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "conv_block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"`

    ` --network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "block_alphas=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"`


## Sample image generation during training
  A prompt file might look like this, for example

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following can be used.

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

  The prompt weighting such as `( )` and `[ ]` are working.

## サンプル画像生成
プロンプトファイルは例えば以下のようになります。

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  `#` で始まる行はコメントになります。`--n` のように「ハイフン二個＋英小文字」の形でオプションを指定できます。以下が使用可能できます。

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

  `( )` や `[ ]` などの重みづけも動作します。

Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.
最近の更新情報は [Release](https://github.com/kohya-ss/sd-scripts/releases) をご覧ください。
