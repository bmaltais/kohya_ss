# ConrtolNet-LLLite について

## 概要
ConrtolNet-LLLite は、[ConrtolNet](https://github.com/lllyasviel/ControlNet) の軽量版です。LoRA Like Lite という意味で、LoRAからインスピレーションを得た構造を持つ、軽量なControlNetです。現在はSDXLにのみ対応しています。

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
`sdxl_train_control_net_lllite.py` を実行してください。`--cond_emb_dim` でconditioning image embeddingの次元数を指定できます。`--network_dim` でLoRA的モジュールのrankを指定できます。その他のオプションは`sdxl_train_network.py`に準じますが、`--network_module`の指定は不要です。

conditioning image embeddingの次元数は、サンプルのCannyでは32を指定しています。LoRA的モジュールのrankは同じく64です。対象とするconditioning imageの特徴に合わせて調整してください。

（サンプルのCannyは恐らくかなり難しいと思われます。depthなどでは半分程度にしてもいいかもしれません。）

### 推論
ComfyUIのカスタムノードを用意しています。: https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI

スクリプトで生成する場合は、`sdxl_gen_img.py` を実行してください。`--control_net_lllite_models` でLLLiteのモデルファイルを指定できます。次元数はモデルファイルから自動取得します。

`--guide_image_path`で推論に用いるconditioning imageを指定してください。なおpreprocessは行われないため、たとえばCannyならCanny処理を行った画像を指定してください（背景黒に白線）。`--control_net_preps`, `--control_net_weights`, `--control_net_ratios` には未対応です。

## サンプル
Canny
