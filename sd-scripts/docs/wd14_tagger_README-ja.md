# WD14Taggerによるタグ付け

こちらのgithubページ（https://github.com/toriato/stable-diffusion-webui-wd14-tagger#mrsmilingwolfs-model-aka-waifu-diffusion-14-tagger ）の情報を参考にさせていただきました。

onnx を用いた推論を推奨します。以下のコマンドで onnx をインストールしてください。

```powershell
pip install onnx==1.15.0 onnxruntime-gpu==1.17.1
```

モデルの重みはHugging Faceから自動的にダウンロードしてきます。

# 使い方

スクリプトを実行してタグ付けを行います。
```
python fintune/tag_images_by_wd14_tagger.py --onnx --repo_id <モデルのrepo id> --batch_size <バッチサイズ> <教師データフォルダ>
```

レポジトリに `SmilingWolf/wd-swinv2-tagger-v3` を使用し、バッチサイズを4にして、教師データを親フォルダの `train_data`に置いた場合、以下のようになります。

```
python tag_images_by_wd14_tagger.py --onnx --repo_id SmilingWolf/wd-swinv2-tagger-v3 --batch_size 4 ..\train_data
```

初回起動時にはモデルファイルが `wd14_tagger_model` フォルダに自動的にダウンロードされます（フォルダはオプションで変えられます）。

タグファイルが教師データ画像と同じディレクトリに、同じファイル名、拡張子.txtで作成されます。

![生成されたタグファイル](https://user-images.githubusercontent.com/52813779/208910534-ea514373-1185-4b7d-9ae3-61eb50bc294e.png)

![タグと画像](https://user-images.githubusercontent.com/52813779/208910599-29070c15-7639-474f-b3e4-06bd5a3df29e.png)

## 記述例

Animagine XL 3.1 方式で出力する場合、以下のようになります（実際には 1 行で入力してください）。

```
python tag_images_by_wd14_tagger.py --onnx --repo_id SmilingWolf/wd-swinv2-tagger-v3 
    --batch_size 4  --remove_underscore --undesired_tags "PUT,YOUR,UNDESIRED,TAGS" --recursive 
    --use_rating_tags_as_last_tag --character_tags_first --character_tag_expand 
    --always_first_tags "1girl,1boy"  ..\train_data
```

## 使用可能なリポジトリID

[SmilingWolf 氏の V2、V3 のモデル](https://huggingface.co/SmilingWolf)が使用可能です。`SmilingWolf/wd-vit-tagger-v3` のように指定してください。省略時のデフォルトは `SmilingWolf/wd-v1-4-convnext-tagger-v2` です。

# オプション

## 一般オプション

- `--onnx` : ONNX を使用して推論します。指定しない場合は TensorFlow を使用します。TensorFlow 使用時は別途 TensorFlow をインストールしてください。
- `--batch_size` : 一度に処理する画像の数。デフォルトは1です。VRAMの容量に応じて増減してください。
- `--caption_extension` : キャプションファイルの拡張子。デフォルトは `.txt` です。
- `--max_data_loader_n_workers` : DataLoader の最大ワーカー数です。このオプションに 1 以上の数値を指定すると、DataLoader を用いて画像読み込みを高速化します。未指定時は DataLoader を用いません。
- `--thresh` : 出力するタグの信頼度の閾値。デフォルトは0.35です。値を下げるとより多くのタグが付与されますが、精度は下がります。
- `--general_threshold` : 一般タグの信頼度の閾値。省略時は `--thresh` と同じです。
- `--character_threshold` : キャラクタータグの信頼度の閾値。省略時は `--thresh` と同じです。
- `--recursive` : 指定すると、指定したフォルダ内のサブフォルダも再帰的に処理します。
- `--append_tags` : 既存のタグファイルにタグを追加します。
- `--frequency_tags` : タグの頻度を出力します。
- `--debug` : デバッグモード。指定するとデバッグ情報を出力します。

## モデルのダウンロード

- `--model_dir` : モデルファイルの保存先フォルダ。デフォルトは `wd14_tagger_model` です。
- `--force_download` : 指定するとモデルファイルを再ダウンロードします。

## タグ編集関連

- `--remove_underscore` : 出力するタグからアンダースコアを削除します。
- `--undesired_tags` : 出力しないタグを指定します。カンマ区切りで複数指定できます。たとえば `black eyes,black hair` のように指定します。
- `--use_rating_tags` : タグの最初にレーティングタグを出力します。
- `--use_rating_tags_as_last_tag` : タグの最後にレーティングタグを追加します。
- `--character_tags_first` : キャラクタータグを最初に出力します。
- `--character_tag_expand` : キャラクタータグのシリーズ名を展開します。たとえば `chara_name_(series)` のタグを `chara_name, series` に分割します。
- `--always_first_tags` : あるタグが画像に出力されたとき、そのタグを最初に出力するタグを指定します。カンマ区切りで複数指定できます。たとえば `1girl,1boy` のように指定します。
- `--caption_separator` : 出力するファイルでタグをこの文字列で区切ります。デフォルトは `, ` です。
- `--tag_replacement` : タグの置換を行います。`tag1,tag2;tag3,tag4` のように指定します。`,` および `;` を使う場合は `\` でエスケープしてください。\
    たとえば `aira tsubase,aira tsubase (uniform)` （特定の衣装を学習させたいとき）、`aira tsubase,aira tsubase\, heir of shadows` （シリーズ名がタグに含まれないとき）のように指定します。

`tag_replacement` は `character_tag_expand` の後に適用されます。

`remove_underscore` 指定時は、`undesired_tags`、`always_first_tags`、`tag_replacement` はアンダースコアを含めずに指定してください。

`caption_separator` 指定時は、`undesired_tags`、`always_first_tags` は `caption_separator`  で区切ってください。`tag_replacement` は必ず `,` で区切ってください。

