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

- 10 Mar. 2023, 2023/3/10: release v0.5.1
  - Fix to LoRA modules in the model are same to the previous (before 0.5.0) if Conv2d-3x3 is disabled (no `conv_dim` arg, default).
    - Conv2D with kernel size 1x1 in ResNet modules were accidentally included in v0.5.0.
    - Trained models with v0.5.0 will work with Web UI's built-in LoRA and Additional Networks extension.
  - Fix an issue that dim (rank) of LoRA module is limited to the in/out dimensions of the target Linear/Conv2d (in case of the dim > 320).
  - `resize_lora.py` now have a feature to `dynamic resizing` which means each LoRA module can have different ranks (dims). Thanks to mgz-dev for this great work!
    - The appropriate rank is selected based on the complexity of each module with an algorithm specified in the command line arguments. For details: https://github.com/kohya-ss/sd-scripts/pull/243
  - Multiple GPUs training is finally supported in `train_network.py`. Thanks to ddPn08 to solve this long running issue!
  - Dataset with fine-tuning method (with metadata json) now works without images if `.npz` files exist. Thanks to rvhfxb!
  - `train_network.py` can work if the current directory is not the directory where the script is in. Thanks to mio2333!
  - Fix `extract_lora_from_models.py` and `svd_merge_lora.py` doesn't work with higher rank (>320).

  - LoRAのConv2d-3x3拡張を行わない場合（`conv_dim` を指定しない場合）、以前（v0.5.0）と同じ構成になるよう修正しました。
    - ResNetのカーネルサイズ1x1のConv2dが誤って対象になっていました。
    - ただv0.5.0で学習したモデルは Additional Networks 拡張、およびWeb UIのLoRA機能で問題なく使えると思われます。
  - LoRAモジュールの dim (rank) が、対象モジュールの次元数以下に制限される不具合を修正しました（320より大きい dim を指定した場合）。
  - `resize_lora.py` に `dynamic resizing` （リサイズ後の各LoRAモジュールが異なるrank (dim) を持てる機能）を追加しました。mgz-dev 氏の貢献に感謝します。
    - 適切なランクがコマンドライン引数で指定したアルゴリズムにより自動的に選択されます。詳細はこちらをご覧ください: https://github.com/kohya-ss/sd-scripts/pull/243
  - `train_network.py` でマルチGPU学習をサポートしました。長年の懸案を解決された ddPn08 氏に感謝します。
  - fine-tuning方式のデータセット（メタデータ.jsonファイルを使うデータセット）で `.npz` が存在するときには画像がなくても動作するようになりました。rvhfxb 氏に感謝します。
  - 他のディレクトリから `train_network.py` を呼び出しても動作するよう変更しました。 mio2333 氏に感謝します。
  - `extract_lora_from_models.py` および `svd_merge_lora.py` が320より大きいrankを指定すると動かない不具合を修正しました。
  
- 9 Mar. 2023, 2023/3/9: release v0.5.0
  - There may be problems due to major changes. If you cannot revert back to the previous version when problems occur, please do not update for a while.
  - Minimum metadata (module name, dim, alpha and network_args) is recorded even with `--no_metadata`, issue https://github.com/kohya-ss/sd-scripts/issues/254
  - `train_network.py` supports LoRA for Conv2d-3x3 (extended to conv2d with a kernel size not 1x1).
    - Same as a current version of [LoCon](https://github.com/KohakuBlueleaf/LoCon). __Thank you very much KohakuBlueleaf for your help!__
      - LoCon will be enhanced in the future. Compatibility for future versions is not guaranteed.
    - Specify `--network_args` option like: `--network_args "conv_dim=4" "conv_alpha=1"`
    - [Additional Networks extension](https://github.com/kohya-ss/sd-webui-additional-networks) version 0.5.0 or later is required to use 'LoRA for Conv2d-3x3' in Stable Diffusion web UI.
    - __Stable Diffusion web UI built-in LoRA does not support 'LoRA for Conv2d-3x3' now. Consider carefully whether or not to use it.__
  - Merging/extracting scripts also support LoRA for Conv2d-3x3.
  - Free CUDA memory after sample generation to reduce VRAM usage, issue https://github.com/kohya-ss/sd-scripts/issues/260 
  - Empty caption doesn't cause error now, issue https://github.com/kohya-ss/sd-scripts/issues/258
  - Fix sample generation is crashing in Textual Inversion training when using templates, or if height/width is not divisible by 8.
  - Update documents (Japanese only).

  - 大きく変更したため不具合があるかもしれません。問題が起きた時にスクリプトを前のバージョンに戻せない場合は、しばらく更新を控えてください。
  - 最低限のメタデータ（module name, dim, alpha および network_args）が `--no_metadata` オプション指定時にも記録されます。issue https://github.com/kohya-ss/sd-scripts/issues/254
  - `train_network.py` で LoRAの Conv2d-3x3 拡張に対応しました（カーネルサイズ1x1以外のConv2dにも対象範囲を拡大します）。
    - 現在のバージョンの [LoCon](https://github.com/KohakuBlueleaf/LoCon) と同一の仕様です。__KohakuBlueleaf氏のご支援に深く感謝します。__
      - LoCon が将来的に拡張された場合、それらのバージョンでの互換性は保証できません。
    - `--network_args` オプションを `--network_args "conv_dim=4" "conv_alpha=1"` のように指定してください。
    - Stable Diffusion web UI での使用には [Additional Networks extension](https://github.com/kohya-ss/sd-webui-additional-networks) のversion 0.5.0 以降が必要です。
    - __Stable Diffusion web UI の LoRA 機能は LoRAの Conv2d-3x3 拡張に対応していないようです。使用するか否か慎重にご検討ください。__
  - マージ、抽出のスクリプトについても LoRA の Conv2d-3x3 拡張に対応しました.
  - サンプル画像生成後にCUDAメモリを解放しVRAM使用量を削減しました。 issue https://github.com/kohya-ss/sd-scripts/issues/260 
  - 空のキャプションが使えるようになりました。 issue https://github.com/kohya-ss/sd-scripts/issues/258
  - Textual Inversion 学習でテンプレートを使ったとき、height/width が 8 で割り切れなかったときにサンプル画像生成がクラッシュするのを修正しました。
  - ドキュメント類を更新しました。

  - Sample image generation:
    A prompt file might look like this, for example

    ```
    # prompt 1
    masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

    # prompt 2
    masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
    ```

    Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following can be used.

    * `--n` Negative prompt up to the next option.
    * `--w` Specifies the width of the generated image.
    * `--h` Specifies the height of the generated image.
    * `--d` Specifies the seed of the generated image.
    * `--l` Specifies the CFG scale of the generated image.
    * `--s` Specifies the number of steps in the generation.

    The prompt weighting such as `( )` and `[ ]` are not working.

  - サンプル画像生成：
    プロンプトファイルは例えば以下のようになります。

    ```
    # prompt 1
    masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

    # prompt 2
    masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
    ```

    `#` で始まる行はコメントになります。`--n` のように「ハイフン二個＋英小文字」の形でオプションを指定できます。以下が使用可能できます。

    * `--n` Negative prompt up to the next option.
    * `--w` Specifies the width of the generated image.
    * `--h` Specifies the height of the generated image.
    * `--d` Specifies the seed of the generated image.
    * `--l` Specifies the CFG scale of the generated image.
    * `--s` Specifies the number of steps in the generation.

    `( )` や `[ ]` などの重みづけは動作しません。

Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.
最近の更新情報は [Release](https://github.com/kohya-ss/sd-scripts/releases) をご覧ください。
