## Textual Inversionの学習について

[Textual Inversion](https://textual-inversion.github.io/)です。実装に当たっては https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion を大いに参考にしました。

学習したモデルはWeb UIでもそのまま使えます。

なお恐らくSD2.xにも対応していますが現時点では未テストです。

## 学習方法

``train_textual_inversion.py`` を用います。

データの準備については ``train_network.py`` と全く同じですので、[そちらのドキュメント](./train_network_README-ja.md)を参照してください。

## オプション

以下はコマンドラインの例です（DreamBooth手法）。

```
accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py 
    --pretrained_model_name_or_path=..\models\model.ckpt 
    --train_data_dir=..\data\db\char1 --output_dir=..\ti_train1 
    --resolution=448,640 --train_batch_size=1 --learning_rate=1e-4 
    --max_train_steps=400 --use_8bit_adam --xformers --mixed_precision=fp16 
    --save_every_n_epochs=1 --save_model_as=safetensors --clip_skip=2 --seed=42 --color_aug 
    --token_string=mychar4 --init_word=cute --num_vectors_per_token=4
```

``--token_string`` に学習時のトークン文字列を指定します。__学習時のプロンプトは、この文字列を含むようにしてください（token_stringがmychar4なら、``mychar4 1girl`` など）__。プロンプトのこの文字列の部分が、Textual Inversionの新しいtokenに置換されて学習されます。

プロンプトにトークン文字列が含まれているかどうかは、``--debug_dataset`` で置換後のtoken idが表示されますので、以下のように ``49408`` 以降のtokenが存在するかどうかで確認できます。

```
input ids: tensor([[49406, 49408, 49409, 49410, 49411, 49412, 49413, 49414, 49415, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]])
```

tokenizerがすでに持っている単語（一般的な単語）は使用できません。

``--init_word`` にembeddingsを初期化するときのコピー元トークンの文字列を指定します。学ばせたい概念が近いものを選ぶとよいようです。二つ以上のトークンになる文字列は指定できません。

``--num_vectors_per_token`` にいくつのトークンをこの学習で使うかを指定します。多いほうが表現力が増しますが、その分多くのトークンを消費します。たとえばnum_vectors_per_token=8の場合、指定したトークン文字列は（一般的なプロンプトの77トークン制限のうち）8トークンを消費します。


その他、以下のオプションが指定できます。

* --weights
  * 学習前に学習済みのembeddingsを読み込み、そこから追加で学習します。
* --use_object_template
  * キャプションではなく既定の物体用テンプレート文字列（``a photo of a {}``など）で学習します。公式実装と同じになります。キャプションは無視されます。
* --use_style_template
  * キャプションではなく既定のスタイル用テンプレート文字列で学習します（``a painting in the style of {}``など）。公式実装と同じになります。キャプションは無視されます。

## 当リポジトリ内の画像生成スクリプトで生成する

gen_img_diffusers.pyに、``--textual_inversion_embeddings`` オプションで学習したembeddingsファイルを指定してください（複数可）。プロンプトでembeddingsファイルのファイル名（拡張子を除く）を使うと、そのembeddingsが適用されます。

