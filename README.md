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

### 8 Apr. 2021, 2021/4/8:

- Added support for training with weighted captions. Thanks to AI-Casanova for the great contribution! 
  - Please refer to the PR for details: [PR #336](https://github.com/kohya-ss/sd-scripts/pull/336)
  - Specify the `--weighted_captions` option. It is available for all training scripts except Textual Inversion and XTI.
  - This option is also applicable to token strings of the DreamBooth method.
  - The syntax for weighted captions is almost the same as the Web UI, and you can use things like `(abc)`, `[abc]`, and `(abc:1.23)`. Nesting is also possible.
  - If you include a comma in the parentheses, the parentheses will not be properly matched in the prompt shuffle/dropout, so do not include a comma in the parentheses.

- 重みづけキャプションによる学習に対応しました。 AI-Casanova 氏の素晴らしい貢献に感謝します。
  - 詳細はこちらをご確認ください。[PR #336](https://github.com/kohya-ss/sd-scripts/pull/336)
  - `--weighted_captions` オプションを指定してください。Textual InversionおよびXTIを除く学習スクリプトで使用可能です。
  - キャプションだけでなく DreamBooth 手法の token string でも有効です。
  - 重みづけキャプションの記法はWeb UIとほぼ同じで、`(abc)`や`[abc]`、`(abc:1.23)`などが使用できます。入れ子も可能です。
  - 括弧内にカンマを含めるとプロンプトのshuffle/dropoutで括弧の対応付けがおかしくなるため、括弧内にはカンマを含めないでください。
  
### 6 Apr. 2023, 2023/4/6:
- There may be bugs because I changed a lot. If you cannot revert the script to the previous version when a problem occurs, please wait for the update for a while.

- Added a feature to upload model and state to HuggingFace. Thanks to ddPn08 for the contribution! [PR #348](https://github.com/kohya-ss/sd-scripts/pull/348)
  - When `--huggingface_repo_id` is specified, the model is uploaded to HuggingFace at the same time as saving the model.
  - Please note that the access token is handled with caution. Please refer to the [HuggingFace documentation](https://huggingface.co/docs/hub/security-tokens).
  - For example, specify other arguments as follows.
    - `--huggingface_repo_id "your-hf-name/your-model" --huggingface_path_in_repo "path" --huggingface_repo_type model --huggingface_repo_visibility private --huggingface_token hf_YourAccessTokenHere`
  - If `public` is specified for `--huggingface_repo_visibility`, the repository will be public. If the option is omitted or `private` (or anything other than `public`) is specified, it will be private.
  - If you specify `--save_state` and `--save_state_to_huggingface`, the state will also be uploaded.
  - If you specify `--resume` and `--resume_from_huggingface`, the state will be downloaded from HuggingFace and resumed.
    - In this case, the `--resume` option is `--resume {repo_id}/{path_in_repo}:{revision}:{repo_type}`. For example: `--resume_from_huggingface --resume your-hf-name/your-model/path/test-000002-state:main:model`
  - If you specify `--async_upload`, the upload will be done asynchronously.
- Added the documentation for applying LoRA to generate with the standard pipeline of Diffusers.   [training LoRA](./train_network_README-ja.md#diffusersのpipelineで生成する) (Japanese only)
- Support for Attention Couple and regional LoRA in `gen_img_diffusers.py`.
  - If you use ` AND ` to separate the prompts, each sub-prompt is sequentially applied to LoRA. `--mask_path` is treated as a mask image. The number of sub-prompts and the number of LoRA must match.


- 大きく変更したため不具合があるかもしれません。問題が起きた時にスクリプトを前のバージョンに戻せない場合は、しばらく更新を控えてください。

- モデルおよびstateをHuggingFaceにアップロードする機能を各スクリプトに追加しました。 [PR #348](https://github.com/kohya-ss/sd-scripts/pull/348) ddPn08 氏の貢献に感謝します。
  - `--huggingface_repo_id`が指定されているとモデル保存時に同時にHuggingFaceにアップロードします。
  - アクセストークンの取り扱いに注意してください。[HuggingFaceのドキュメント](https://huggingface.co/docs/hub/security-tokens)を参照してください。
  - 他の引数をたとえば以下のように指定してください。
    - `--huggingface_repo_id "your-hf-name/your-model" --huggingface_path_in_repo "path" --huggingface_repo_type model --huggingface_repo_visibility private --huggingface_token hf_YourAccessTokenHere`
  - `--huggingface_repo_visibility`に`public`を指定するとリポジトリが公開されます。省略時または`private`（など`public`以外）を指定すると非公開になります。
  - `--save_state`オプション指定時に`--save_state_to_huggingface`を指定するとstateもアップロードします。
  - `--resume`オプション指定時に`--resume_from_huggingface`を指定するとHuggingFaceからstateをダウンロードして再開します。
    - その時の `--resume`オプションは `--resume {repo_id}/{path_in_repo}:{revision}:{repo_type}`になります。例: `--resume_from_huggingface --resume your-hf-name/your-model/path/test-000002-state:main:model`
  - `--async_upload`オプションを指定するとアップロードを非同期で行います。
- [LoRAの文書](./train_network_README-ja.md#diffusersのpipelineで生成する)に、LoRAを適用してDiffusersの標準的なパイプラインで生成する方法を追記しました。
- `gen_img_diffusers.py` で Attention Couple および領域別LoRAに対応しました。
  - プロンプトを` AND `で区切ると各サブプロンプトが順にLoRAに適用されます。`--mask_path` がマスク画像として扱われます。サブプロンプトの数とLoRAの数は一致している必要があります。


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
