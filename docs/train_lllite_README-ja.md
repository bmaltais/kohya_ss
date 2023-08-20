# ConrtolNet-LLLite について

__きわめて実験的な実装のため、将来的に大きく変更される可能性があります。__

## 概要
ConrtolNet-LLLite は、[ConrtolNet](https://github.com/lllyasviel/ControlNet) の軽量版です。LoRA Like Lite という意味で、LoRAからインスピレーションを得た構造を持つ、軽量なControlNetです。現在はSDXLにのみ対応しています。

## サンプルの重みファイルと推論

こちらにあります: https://huggingface.co/kohya-ss/controlnet-lllite

ComfyUIのカスタムノードを用意しています。: https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI

生成サンプルはこのページの末尾にあります。

## モデル構造
ひとつのLLLiteモジュールは、制御用画像（以下conditioning image）を潜在空間に写像するconditioning image embeddingと、LoRAにちょっと似た構造を持つ小型のネットワークからなります。LLLiteモジュールを、LoRAと同様にU-NetのLinearやConvに追加します。詳しくはソースコードを参照してください。

推論環境の制限で、現在はCrossAttentionのみ（attn1のq/k/v、attn2のq）に追加されます。

## モデルの学習

### データセットの準備
通常のdatasetに加え、`conditioning_data_dir` で指定したディレクトリにconditioning imageを格納してください。conditioning imageは学習用画像と同じbasenameを持つ必要があります。また、conditioning imageは学習用画像と同じサイズに自動的にリサイズされます。

```toml
[[datasets.subsets]]
image_dir = "path/to/image/dir"
caption_extension = ".txt"
conditioning_data_dir = "path/to/conditioning/image/dir"
```

現時点の制約として、random_cropは使用できません。

### 学習
スクリプトで生成する場合は、`sdxl_train_control_net_lllite.py` を実行してください。`--cond_emb_dim` でconditioning image embeddingの次元数を指定できます。`--network_dim` でLoRA的モジュールのrankを指定できます。その他のオプションは`sdxl_train_network.py`に準じますが、`--network_module`の指定は不要です。

conditioning image embeddingの次元数は、サンプルのCannyでは32を指定しています。LoRA的モジュールのrankは同じく64です。対象とするconditioning imageの特徴に合わせて調整してください。

（サンプルのCannyは恐らくかなり難しいと思われます。depthなどでは半分程度にしてもいいかもしれません。）

### 推論


スクリプトで生成する場合は、`sdxl_gen_img.py` を実行してください。`--control_net_lllite_models` でLLLiteのモデルファイルを指定できます。次元数はモデルファイルから自動取得します。

`--guide_image_path`で推論に用いるconditioning imageを指定してください。なおpreprocessは行われないため、たとえばCannyならCanny処理を行った画像を指定してください（背景黒に白線）。`--control_net_preps`, `--control_net_weights`, `--control_net_ratios` には未対応です。

## 謝辞

ControlNetの作者である lllyasviel 氏、実装上のアドバイスとトラブル解決へのご尽力をいただいた furusu 氏、ControlNetデータセットを実装していただいた ddPn08 氏に感謝いたします。

## サンプル
Canny
![kohya_ss_girl_standing_at_classroom_smiling_to_the_viewer_class_78976b3e-0d4d-4ea0-b8e3-053ae493abbc](https://github.com/kohya-ss/sd-scripts/assets/52813779/37e9a736-649b-4c0f-ab26-880a1bf319b5)

![im_20230820104253_000_1](https://github.com/kohya-ss/sd-scripts/assets/52813779/c8896900-ab86-4120-932f-6e2ae17b77c0)

![im_20230820104302_000_1](https://github.com/kohya-ss/sd-scripts/assets/52813779/b12457a0-ee3c-450e-ba9a-b712d0fe86bb)

![im_20230820104310_000_1](https://github.com/kohya-ss/sd-scripts/assets/52813779/8845b8d9-804a-44ac-9618-113a28eac8a1)

