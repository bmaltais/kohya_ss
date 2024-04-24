# ControlNet-LLLite について

__きわめて実験的な実装のため、将来的に大きく変更される可能性があります。__

## 概要
ControlNet-LLLite は、[ControlNet](https://github.com/lllyasviel/ControlNet) の軽量版です。LoRA Like Lite という意味で、LoRAからインスピレーションを得た構造を持つ、軽量なControlNetです。現在はSDXLにのみ対応しています。

## サンプルの重みファイルと推論

こちらにあります: https://huggingface.co/kohya-ss/controlnet-lllite

ComfyUIのカスタムノードを用意しています。: https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI

生成サンプルはこのページの末尾にあります。

## モデル構造
ひとつのLLLiteモジュールは、制御用画像（以下conditioning image）を潜在空間に写像するconditioning image embeddingと、LoRAにちょっと似た構造を持つ小型のネットワークからなります。LLLiteモジュールを、LoRAと同様にU-NetのLinearやConvに追加します。詳しくはソースコードを参照してください。

推論環境の制限で、現在はCrossAttentionのみ（attn1のq/k/v、attn2のq）に追加されます。

## モデルの学習

### データセットの準備
DreamBooth 方式の dataset で、`conditioning_data_dir` で指定したディレクトリにconditioning imageを格納してください。

（finetuning 方式の dataset はサポートしていません。）

conditioning imageは学習用画像と同じbasenameを持つ必要があります。また、conditioning imageは学習用画像と同じサイズに自動的にリサイズされます。conditioning imageにはキャプションファイルは不要です。

たとえば、キャプションにフォルダ名ではなくキャプションファイルを用いる場合の設定ファイルは以下のようになります。

```toml
[[datasets.subsets]]
image_dir = "path/to/image/dir"
caption_extension = ".txt"
conditioning_data_dir = "path/to/conditioning/image/dir"
```

現時点の制約として、random_cropは使用できません。

学習データとしては、元のモデルで生成した画像を学習用画像として、そこから加工した画像をconditioning imageとした、合成によるデータセットを用いるのがもっとも簡単です（データセットの品質的には問題があるかもしれません）。具体的なデータセットの合成方法については後述します。

なお、元モデルと異なる画風の画像を学習用画像とすると、制御に加えて、その画風についても学ぶ必要が生じます。ControlNet-LLLiteは容量が少ないため、画風学習には不向きです。このような場合には、後述の次元数を多めにしてください。

### 学習
スクリプトで生成する場合は、`sdxl_train_control_net_lllite.py` を実行してください。`--cond_emb_dim` でconditioning image embeddingの次元数を指定できます。`--network_dim` でLoRA的モジュールのrankを指定できます。その他のオプションは`sdxl_train_network.py`に準じますが、`--network_module`の指定は不要です。

学習時にはメモリを大量に使用しますので、キャッシュやgradient checkpointingなどの省メモリ化のオプションを有効にしてください。また`--full_bf16` オプションで、BFloat16を使用するのも有効です（RTX 30シリーズ以降のGPUが必要です）。24GB VRAMで動作確認しています。

conditioning image embeddingの次元数は、サンプルのCannyでは32を指定しています。LoRA的モジュールのrankは同じく64です。対象とするconditioning imageの特徴に合わせて調整してください。

（サンプルのCannyは恐らくかなり難しいと思われます。depthなどでは半分程度にしてもいいかもしれません。）

以下は .toml の設定例です。

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

### 推論

スクリプトで生成する場合は、`sdxl_gen_img.py` を実行してください。`--control_net_lllite_models` でLLLiteのモデルファイルを指定できます。次元数はモデルファイルから自動取得します。

`--guide_image_path`で推論に用いるconditioning imageを指定してください。なおpreprocessは行われないため、たとえばCannyならCanny処理を行った画像を指定してください（背景黒に白線）。`--control_net_preps`, `--control_net_weights`, `--control_net_ratios` には未対応です。

## データセットの合成方法

### 学習用画像の生成

学習のベースとなるモデルで画像生成を行います。Web UIやComfyUIなどで生成してください。画像サイズはモデルのデフォルトサイズで良いと思われます（1024x1024など）。bucketingを用いることもできます。その場合は適宜適切な解像度で生成してください。

生成時のキャプション等は、ControlNet-LLLiteの利用時に生成したい画像にあわせるのが良いと思われます。

生成した画像を任意のディレクトリに保存してください。このディレクトリをデータセットの設定ファイルで指定します。

当リポジトリ内の `sdxl_gen_img.py` でも生成できます。例えば以下のように実行します。

```dos
python sdxl_gen_img.py --ckpt path/to/model.safetensors --n_iter 1 --scale 10 --steps 36 --outdir path/to/output/dir --xformers --W 1024 --H 1024 --original_width 2048 --original_height 2048 --bf16 --sampler ddim --batch_size 4 --vae_batch_size 2 --images_per_prompt 512 --max_embeddings_multiples 1 --prompt "{portrait|digital art|anime screen cap|detailed illustration} of 1girl, {standing|sitting|walking|running|dancing} on {classroom|street|town|beach|indoors|outdoors}, {looking at viewer|looking away|looking at another}, {in|wearing} {shirt and skirt|school uniform|casual wear} { |, dynamic pose}, (solo), teen age, {0-1$$smile,|blush,|kind smile,|expression less,|happy,|sadness,} {0-1$$upper body,|full body,|cowboy shot,|face focus,} trending on pixiv, {0-2$$depth of fields,|8k wallpaper,|highly detailed,|pov,} {0-1$$summer, |winter, |spring, |autumn, } beautiful face { |, from below|, from above|, from side|, from behind|, from back} --n nsfw, bad face, lowres, low quality, worst quality, low effort, watermark, signature, ugly, poorly drawn"
```

VRAM 24GBの設定です。VRAMサイズにより`--batch_size` `--vae_batch_size`を調整してください。

`--prompt`でワイルドカードを利用してランダムに生成しています。適宜調整してください。

### 画像の加工

外部のプログラムを用いて、生成した画像を加工します。加工した画像を任意のディレクトリに保存してください。これらがconditioning imageになります。

加工にはたとえばCannyなら以下のようなスクリプトが使えます。

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

### キャプションファイルの作成

学習用画像のbasenameと同じ名前で、それぞれの画像に対応したキャプションファイルを作成してください。生成時のプロンプトをそのまま利用すれば良いと思われます。

`sdxl_gen_img.py` で生成した場合は、画像内のメタデータに生成時のプロンプトが記録されていますので、以下のようなスクリプトで学習用画像と同じディレクトリにキャプションファイルを作成できます（拡張子 `.txt`）。

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

### データセットの設定ファイルの作成

コマンドラインオプションからの指定も可能ですが、`.toml`ファイルを作成する場合は `conditioning_data_dir` に加工した画像を保存したディレクトリを指定します。

以下は設定ファイルの例です。

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

## 謝辞

ControlNetの作者である lllyasviel 氏、実装上のアドバイスとトラブル解決へのご尽力をいただいた furusu 氏、ControlNetデータセットを実装していただいた ddPn08 氏に感謝いたします。

## サンプル
Canny
![kohya_ss_girl_standing_at_classroom_smiling_to_the_viewer_class_78976b3e-0d4d-4ea0-b8e3-053ae493abbc](https://github.com/kohya-ss/sd-scripts/assets/52813779/37e9a736-649b-4c0f-ab26-880a1bf319b5)

![im_20230820104253_000_1](https://github.com/kohya-ss/sd-scripts/assets/52813779/c8896900-ab86-4120-932f-6e2ae17b77c0)

![im_20230820104302_000_1](https://github.com/kohya-ss/sd-scripts/assets/52813779/b12457a0-ee3c-450e-ba9a-b712d0fe86bb)

![im_20230820104310_000_1](https://github.com/kohya-ss/sd-scripts/assets/52813779/8845b8d9-804a-44ac-9618-113a28eac8a1)

