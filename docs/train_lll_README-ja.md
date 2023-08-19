# ConrtolNet-LLLite について

## 概要
ConrtolNet-LLLite は、[ConrtolNet](https://github.com/lllyasviel/ControlNet) の軽量版です。LoRA Like Lite という意味で、LoRAに似た構造の軽量なControlNetです。現在はSDXLにのみ対応しています。

## モデル構造
制御用画像（以下conditioning image）を潜在空間に写像するconditioning image embeddingと、U-Netの各モジュールに付与されるLoRAにちょっと似た構造を持つモジュールを組み合わせたモデルです。詳しくはソースコードを参照してください。

## モデルの学習

### データセットの準備
通常のdatasetに加え、`conditioning_data_dir` で指定したディレクトリにconditioning imageを格納してください。conditioning imageは学習用画像と同じbasenameを持つ必要があります。また、conditioning imageは学習用画像と同じサイズに自動的にリサイズされます。

```toml
[[datasets.subsets]]
image_dir = "path/to/image/dir"
caption_extension = ".txt"
conditioning_data_dir = "path/to/conditioning/image/dir"
```

### 学習
`sdxl_train_control_net_lllite.py` を実行してください。`--cond_emb_dim` でconditioning image embeddingの次元数を指定できます。`--network_dim` でLoRA的モジュールのrankを指定できます。その他のオプションは`sdxl_train_network.py`に準じますが、`--network_module`の指定は不要です。


### 推論
`sdxl_gen_img.py` を実行してください。`--control_net_lllite_models` でLLLiteのモデルファイルを指定できます。次元数はモデルファイルから自動取得します。

`--guide_image_path`で推論に用いるconditioning imageを指定してください。なおpreprocessは行われないため、たとえばCannyならCanny処理を行った画像を指定してください（背景黒に白線）。`--control_net_preps`, `--control_net_weights`, `--control_net_ratios` には未対応です。

### サンプル
Canny
![kohya_ss_girl_standing_at_classroom_smiling_to_the_viewer_class_78976b3e-0d4d-4ea0-b8e3-053ae493abbc](https://github.com/kohya-ss/sd-scripts/assets/52813779/7e883352-0fea-4f5a-b820-94e17ec3f3f2)

![im_20230819212806_000_1](https://github.com/kohya-ss/sd-scripts/assets/52813779/c28196f9-b2c3-40ad-b000-21a77e657968)

![im_20230819212815_000_1](https://github.com/kohya-ss/sd-scripts/assets/52813779/b8506354-feb8-4d58-86a8-738a9ba03911)

![im_20230819212822_000_1](https://github.com/kohya-ss/sd-scripts/assets/52813779/1612c221-8df5-420c-b907-75758d89aca7)

