## マスクロスについて

マスクロスは、入力画像のマスクで指定された部分だけ損失計算することで、画像の一部分だけを学習することができる機能です。
たとえばキャラクタを学習したい場合、キャラクタ部分だけをマスクして学習することで、背景を無視して学習することができます。

マスクロスのマスクには、二種類の指定方法があります。

- マスク画像を用いる方法
- 透明度（アルファチャネル）を使用する方法

なお、サンプルは [ずんずんPJイラスト/3Dデータ](https://zunko.jp/con_illust.html) の「AI画像モデル用学習データ」を使用しています。

### マスク画像を用いる方法

学習画像それぞれに対応するマスク画像を用意する方法です。学習画像と同じファイル名のマスク画像を用意し、それを学習画像と別のディレクトリに保存します。

- 学習画像
  ![image](https://github.com/kohya-ss/sd-scripts/assets/52813779/607c5116-5f62-47de-8b66-9c4a597f0441)
- マスク画像
  ![image](https://github.com/kohya-ss/sd-scripts/assets/52813779/53e9b0f8-a4bf-49ed-882d-4026f84e8450)

```.toml
[[datasets.subsets]]
image_dir = "/path/to/a_zundamon"
caption_extension = ".txt"
conditioning_data_dir = "/path/to/a_zundamon_mask"
num_repeats = 8
```

マスク画像は、学習画像と同じサイズで、学習する部分を白、無視する部分を黒で描画します。グレースケールにも対応しています（127 ならロス重みが 0.5 になります）。なお、正確にはマスク画像の R チャネルが用いられます。

DreamBooth 方式の dataset で、`conditioning_data_dir` で指定したディレクトリにマスク画像を保存してください。ControlNet のデータセットと同じですので、詳細は [ControlNet-LLLite](train_lllite_README-ja.md#データセットの準備) を参照してください。

### 透明度（アルファチャネル）を使用する方法

学習画像の透明度（アルファチャネル）がマスクとして使用されます。透明度が 0 の部分は無視され、255 の部分は学習されます。半透明の場合は、その透明度に応じてロス重みが変化します（127 ならおおむね 0.5）。

![image](https://github.com/kohya-ss/sd-scripts/assets/52813779/0baa129b-446a-4aac-b98c-7208efb0e75e)

※それぞれの画像は透過PNG

学習時のスクリプトのオプションに `--alpha_mask` を指定するか、dataset の設定ファイルの subset で、`alpha_mask` を指定してください。たとえば、以下のようになります。

```toml
[[datasets.subsets]]
image_dir = "/path/to/image/dir"
caption_extension = ".txt"
num_repeats = 8
alpha_mask = true
```

## 学習時の注意事項

- 現時点では DreamBooth 方式の dataset のみ対応しています。
- マスクは latents のサイズ、つまり 1/8 に縮小されてから適用されます。そのため、細かい部分（たとえばアホ毛やイヤリングなど）はうまく学習できない可能性があります。マスクをわずかに拡張するなどの工夫が必要かもしれません。
- マスクロスを用いる場合、学習対象外の部分をキャプションに含める必要はないかもしれません。（要検証）
- `alpha_mask` の場合、マスクの有無を切り替えると latents キャッシュが自動的に再生成されます。
