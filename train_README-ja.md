当リポジトリではモデルのfine tuning、DreamBooth、およびLoRAとTextual Inversionの学習をサポートします。この文書ではそれらに共通する、学習データの準備方法やスクリプトオプションについて説明します。

# 概要

あらかじめこのリポジトリのREADMEを参照し、環境整備を行ってください。


以下について説明します。

1. 学習データの準備について（設定ファイルを用いる新形式）
1. Aspect Ratio Bucketingについて
1. 以前の指定形式（設定ファイルを用いずコマンドラインから指定）
1. fine tuning 方式のメタデータ準備：キャプションニングなど

1.だけ実行すればとりあえず学習は可能です（学習については各スクリプトのドキュメントを参照）。2.以降は必要に応じて参照してください。

<!--
1. 各スクリプトで共通のオプション
-->

# 学習データの準備について

任意のフォルダ（複数でも可）に学習データの画像ファイルを用意しておきます。`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp` をサポートします。リサイズなどの前処理は基本的に必要ありません。

ただし学習解像度（後述）よりも極端に小さい画像は使わないか、あらかじめ超解像AIなどで拡大しておくことをお勧めします。また極端に大きな画像（3000x3000ピクセル程度？）よりも大きな画像はエラーになる場合があるようですので事前に縮小してください。

学習時には、モデルに学ばせる画像データを整理し、スクリプトに対して指定する必要があります。学習データの数、学習対象、キャプション（画像の説明）が用意できるか否かなどにより、いくつかの方法で学習データを指定できます。以下の方式があります（それぞれの名前は一般的なものではなく、当リポジトリ独自の定義です）。正則化画像については後述します。

1. DreamBooth、class+identifier方式（正則化画像使用可）

    特定の単語 (identifier) に学習対象を紐づけるように学習します。キャプションを用意する必要はありません。たとえば特定のキャラを学ばせる場合に使うとキャプションを用意する必要がない分、手軽ですが、髪型や服装、背景など学習データの全要素が identifier に紐づけられて学習されるため、生成時のプロンプトで服が変えられない、といった事態も起こりえます。

1. DreamBooth、キャプション方式（正則化画像使用可）

    画像ごとにキャプションが記録されたテキストファイルを用意して学習します。たとえば特定のキャラを学ばせると、画像の詳細をキャプションに記述することで（白い服を着たキャラA、赤い服を着たキャラA、など）キャラとそれ以外の要素が分離され、より厳密にモデルがキャラだけを学ぶことが期待できます。

1. fine tuning方式（正則化画像使用不可）

    あらかじめキャプションをメタデータファイルにまとめます。タグとキャプションを分けて管理したり、学習を高速化するためlatentsを事前キャッシュしたりなどの機能をサポートします（いずれも別文書で説明しています）。

学習したいものと使用できる指定方法の組み合わせは以下の通りです。

| 学習対象または方法 | スクリプト | DB / class+identifier | DB / キャプション | fine tuning |
| ----- | ----- | ----- | ----- | ----- |
| モデルをfine tuning | `fine_tune.py`| x | x | o |
| モデルをDreamBooth | `train_db.py`| o | o | x |
| LoRA | `train_network.py`| o | o | o |
| Textual Invesion | `train_textual_inversion.py`| o | o | o |

## どれを選ぶか

LoRA、Textual Inversionについては、手軽にキャプションファイルを用意せずに学習したい場合はDreamBooth class+identifier、用意できるならDreamBooth キャプション方式がよいでしょう。学習データの枚数が多く、かつ正則化画像を使用しない場合はfine tuning方式も検討してください。

DreamBoothについても同様ですが、fine tuning方式は使えません。fine tuningの場合はfine tuning方式のみです。

# 各方式の指定方法について

ここではそれぞれの指定方法で典型的なパターンについてだけ説明します。より詳細な指定方法については [データセット設定](./config_README-ja.md) をご覧ください。

# DreamBooth、class+identifier方式（正則化画像使用可）

この方式では、各画像は `class identifier` というキャプションで学習されたのと同じことになります（`shs dog` など）。

## step 1. identifierとclassを決める

学ばせたい対象を結びつける単語identifierと、対象の属するclassを決めます。

（instanceなどいろいろな呼び方がありますが、とりあえず元の論文に合わせます。）

以下ごく簡単に説明します（詳しくは調べてください）。

classは学習対象の一般的な種別です。たとえば特定の犬種を学ばせる場合には、classはdogになります。アニメキャラならモデルによりboyやgirl、1boyや1girlになるでしょう。

identifierは学習対象を識別して学習するためのものです。任意の単語で構いませんが、元論文によると「tokinizerで1トークンになる3文字以下でレアな単語」が良いとのことです。

identifierとclassを使い、たとえば「shs dog」などでモデルを学習することで、学習させたい対象をclassから識別して学習できます。

画像生成時には「shs dog」とすれば学ばせた犬種の画像が生成されます。

（identifierとして私が最近使っているものを参考までに挙げると、``shs sts scs cpc coc cic msm usu ici lvl cic dii muk ori hru rik koo yos wny`` などです。本当は Danbooru Tag に含まれないやつがより望ましいです。）

## step 2. 正則化画像を使うか否かを決め、使う場合には正則化画像を生成する

正則化画像とは、前述のclass全体が、学習対象に引っ張られることを防ぐための画像です（language drift）。正則化画像を使わないと、たとえば `shs 1girl` で特定のキャラクタを学ばせると、単なる `1girl` というプロンプトで生成してもそのキャラに似てきます。これは `1girl` が学習時のキャプションに含まれているためです。

学習対象の画像と正則化画像を同時に学ばせることで、class は class のままで留まり、identifier をプロンプトにつけた時だけ学習対象が生成されるようになります。

LoRAやDreamBoothで特定のキャラだけ出てくればよい場合は、正則化画像を用いなくても良いといえます。

Textual Inversionでは用いなくてよいでしょう（学ばせる token string がキャプションに含まれない場合はなにも学習されないため）。

正則化画像としては、学習対象のモデルで、class 名だけで生成した画像を用いるのが一般的です（たとえば `1girl`）。ただし生成画像の品質が悪い場合には、プロンプトを工夫したり、ネットから別途ダウンロードした画像を用いることもできます。

（正則化画像も学習されるため、その品質はモデルに影響します。）

一般的には数百枚程度、用意するのが望ましいようです（枚数が少ないと class 画像が一般化されずそれらの特徴を学んでしまいます）。

生成画像を使う場合、通常、生成画像のサイズは学習解像度（より正確にはbucketの解像度、後述）にあわせてください。

## step 2. 設定ファイルの記述

テキストファイルを作成し、拡張子を `.toml` にします。たとえば以下のように記述します。

（`#` で始まっている部分はコメントですので、このままコピペしてそのままでもよいですし、削除しても問題ありません。）

```toml
[general]
enable_bucket = true                        # Aspect Ratio Bucketingを使うか否か

[[datasets]]
resolution = 512                            # 学習解像度
batch_size = 4                              # バッチサイズ

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # 学習用画像を入れたフォルダを指定
  class_tokens = 'hoge girl'                # identifier class を指定
  num_repeats = 10                          # 学習用画像の繰り返し回数

  # 以下は正則化画像を用いる場合のみ記述する。用いない場合は削除する
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # 正則化画像を入れたフォルダを指定
  class_tokens = 'girl'                     # class を指定
  num_repeats = 1                           # 正則化画像の繰り返し回数、基本的には1でよい
```

基本的には以下を場所のみ書き換えれば学習できます。

1. 学習解像度

    数値1つを指定すると正方形（`512`なら512x512）、鍵カッコカンマ区切りで2つ指定すると横×縦（`[512,768]`なら512x768）になります。SD1.x系ではもともとの学習解像度は512です。`[512,768]` 等の大きめの解像度を指定すると縦長、横長画像生成時の破綻を小さくできるかもしれません。SD2.x 768系では `768` です。

1. バッチサイズ

    同時に何件のデータを学習するかを指定します。GPUのVRAMサイズ、学習解像度によって変わってきます。またfine tuning/DreamBooth/LoRA等でも変わってきますので、詳しくは各スクリプトの説明をご覧ください。

1. フォルダ指定

    学習用画像、正則化画像（使用する場合のみ）のフォルダを指定します。画像データが含まれているフォルダそのものを指定します。

1. identifier と class の指定

    前述のサンプルの通りです。

1. 繰り返し回数

    後述します。

### 繰り返し回数について

繰り返し回数は、正則化画像の枚数と学習用画像の枚数を調整するために用いられます。正則化画像の枚数は学習用画像よりも多いため、学習用画像を繰り返して枚数を合わせ、1対1の比率で学習できるようにします。

繰り返し回数は「 __学習用画像の繰り返し回数×学習用画像の枚数≧正則化画像の繰り返し回数×正則化画像の枚数__ 」となるように指定してください。

（1 epoch（データが一周すると1 epoch）のデータ数が「学習用画像の繰り返し回数×学習用画像の枚数」となります。正則化画像の枚数がそれより多いと、余った部分の正則化画像は使用されません。）

## step 3. 学習

それぞれのドキュメントを参考に学習を行ってください。

# DreamBooth、キャプション方式（正則化画像使用可）

この方式では各画像はキャプションで学習されます。

## step 1. キャプションファイルを準備する

学習用画像のフォルダに、画像と同じファイル名で、拡張子 `.caption`（設定で変えられます）のファイルを置いてください。それぞれのファイルは1行のみとしてください。エンコーディングは `UTF-8` です。

## step 2. 正則化画像を使うか否かを決め、使う場合には正則化画像を生成する

class+identifier形式と同様です。なお正則化画像にもキャプションを付けることができますが、通常は不要でしょう。

## step 2. 設定ファイルの記述

テキストファイルを作成し、拡張子を `.toml` にします。たとえば以下のように記述します。

```toml
[general]
enable_bucket = true                        # Aspect Ratio Bucketingを使うか否か

[[datasets]]
resolution = 512                            # 学習解像度
batch_size = 4                              # バッチサイズ

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # 学習用画像を入れたフォルダを指定
  caption_extension = '.caption'            # キャプションファイルの拡張子　.txt を使う場合には書き換える
  num_repeats = 10                          # 学習用画像の繰り返し回数

  # 以下は正則化画像を用いる場合のみ記述する。用いない場合は削除する
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # 正則化画像を入れたフォルダを指定
  class_tokens = 'girl'                     # class を指定
  num_repeats = 1                           # 正則化画像の繰り返し回数、基本的には1でよい
```

基本的には以下を場所のみ書き換えれば学習できます。特に記述がない部分は class+identifier 方式と同じです。

1. 学習解像度
1. バッチサイズ
1. フォルダ指定
1. キャプションファイルの拡張子

    任意の拡張子を指定できます。
1. 繰り返し回数

## step 3. 学習

それぞれのドキュメントを参考に学習を行ってください。

# fine tuning 方式

## step 1. メタデータを準備する

キャプションやタグをまとめた管理用ファイルをメタデータと呼びます。json形式で拡張子は `.json`
 です。作成方法は長くなりますのでこの文書の末尾に書きました。

## step 2. 設定ファイルの記述

テキストファイルを作成し、拡張子を `.toml` にします。たとえば以下のように記述します。

```toml
[general]
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = 512                                    # 学習解像度
batch_size = 4                                      # バッチサイズ

  [[datasets.subsets]]
  image_dir = 'C:\piyo'                             # 学習用画像を入れたフォルダを指定
  metadata_file = 'C:\piyo\piyo_md.json'            # メタデータファイル名
```

基本的には以下を場所のみ書き換えれば学習できます。特に記述がない部分は DreamBooth, class+identifier 方式と同じです。

1. 学習解像度
1. バッチサイズ
1. フォルダ指定
1. メタデータファイル名

    後述の方法で作成したメタデータファイルを指定します。


## step 3. 学習

それぞれのドキュメントを参考に学習を行ってください。

# Aspect Ratio Bucketing について

Stable Diffusion のv1は512\*512で学習されていますが、それに加えて256\*1024や384\*640といった解像度でも学習します。これによりトリミングされる部分が減り、より正しくキャプションと画像の関係が学習されることが期待されます。

また任意の解像度で学習するため、事前に画像データの縦横比を統一しておく必要がなくなります。

設定で有効、向こうが切り替えられますが、ここまでの設定ファイルの記述例では有効になっています（`true` が設定されています）。

学習解像度はパラメータとして与えられた解像度の面積（＝メモリ使用量）を超えない範囲で、64ピクセル単位（デフォルト、変更可）で縦横に調整、作成されます。

機械学習では入力サイズをすべて統一するのが一般的ですが、特に制約があるわけではなく、実際は同一のバッチ内で統一されていれば大丈夫です。NovelAIの言うbucketingは、あらかじめ教師データを、アスペクト比に応じた学習解像度ごとに分類しておくことを指しているようです。そしてバッチを各bucket内の画像で作成することで、バッチの画像サイズを統一します。

# 以前のデータ指定方法

フォルダ名で繰り返し回数を指定する方法です。

## step 1. 学習用画像の準備

学習用画像を格納するフォルダを作成します。 __さらにその中に__ 、以下の名前でディレクトリを作成します。

```
<繰り返し回数>_<identifier> <class>
```

間の``_``を忘れないでください。

たとえば「sls frog」というプロンプトで、データを20回繰り返す場合、「20_sls frog」となります。以下のようになります。

![image](https://user-images.githubusercontent.com/52813779/210770636-1c851377-5936-4c15-90b7-8ac8ad6c2074.png)

### 複数class、複数対象（identifier）の学習

方法は単純で、学習用画像のフォルダ内に ``繰り返し回数_<identifier> <class>`` のフォルダを複数、正則化画像フォルダにも同様に ``繰り返し回数_<class>`` のフォルダを複数、用意してください。

たとえば「sls frog」と「cpc rabbit」を同時に学習する場合、以下のようになります。

![image](https://user-images.githubusercontent.com/52813779/210777933-a22229db-b219-4cd8-83ca-e87320fc4192.png)

classがひとつで対象が複数の場合、正則化画像フォルダはひとつで構いません。たとえば1girlにキャラAとキャラBがいる場合は次のようにします。

- train_girls
  - 10_sls 1girl
  - 10_cpc 1girl
- reg_girls
  - 1_1girl

### DreamBoothでキャプションを使う

学習用画像、正則化画像のフォルダに、画像と同じファイル名で、拡張子.caption（オプションで変えられます）のファイルを置くと、そのファイルからキャプションを読み込みプロンプトとして学習します。

※それらの画像の学習に、フォルダ名（identifier class）は使用されなくなります。

キャプションファイルの拡張子はデフォルトで.captionです。学習スクリプトの `--caption_extension` オプションで変更できます。`--shuffle_caption` オプションで学習時のキャプションについて、カンマ区切りの各部分をシャッフルしながら学習します。

## step 2. 正則化画像の準備

正則化画像を使う場合の手順です。

正則化画像を格納するフォルダを作成します。 __さらにその中に__  ``<繰り返し回数>_<class>`` という名前でディレクトリを作成します。

たとえば「frog」というプロンプトで、データを繰り返さない（1回だけ）場合、以下のようになります。

![image](https://user-images.githubusercontent.com/52813779/210770897-329758e5-3675-49f1-b345-c135f1725832.png)


## step 3. 学習の実行

各学習スクリプトを実行します。 `--train_data_dir` オプションで前述の学習用データのフォルダを（__画像を含むフォルダではなく、その親フォルダ__）、`--reg_data_dir` オプションで正則化画像のフォルダ（__画像を含むフォルダではなく、その親フォルダ__）を指定してください。

<!-- 
# 学習スクリプト共通のオプション

スクリプトの更新後、ドキュメントの更新が追い付いていない場合があります。その場合は `--help` オプションで使用できるオプションを確認してください。

## TODO 書きます
-->

# メタデータファイルの作成

## 教師データの用意

前述のように学習させたい画像データを用意し、任意のフォルダに入れてください。

たとえば以下のように画像を格納します。

![教師データフォルダのスクショ](https://user-images.githubusercontent.com/52813779/208907739-8e89d5fa-6ca8-4b60-8927-f484d2a9ae04.png)

## 自動キャプショニング

キャプションを使わずタグだけで学習する場合はスキップしてください。

また手動でキャプションを用意する場合、キャプションは教師データ画像と同じディレクトリに、同じファイル名、拡張子.caption等で用意してください。各ファイルは1行のみのテキストファイルとします。

### BLIPによるキャプショニング

最新版ではBLIPのダウンロード、重みのダウンロード、仮想環境の追加は不要になりました。そのままで動作します。

finetuneフォルダ内のmake_captions.pyを実行します。

```
python finetune\make_captions.py --batch_size <バッチサイズ> <教師データフォルダ>
```

バッチサイズ8、教師データを親フォルダのtrain_dataに置いた場合、以下のようになります。

```
python finetune\make_captions.py --batch_size 8 ..\train_data
```

キャプションファイルが教師データ画像と同じディレクトリに、同じファイル名、拡張子.captionで作成されます。

batch_sizeはGPUのVRAM容量に応じて増減してください。大きいほうが速くなります（VRAM 12GBでももう少し増やせると思います）。
max_lengthオプションでキャプションの最大長を指定できます。デフォルトは75です。モデルをトークン長225で学習する場合には長くしても良いかもしれません。
caption_extensionオプションでキャプションの拡張子を変更できます。デフォルトは.captionです（.txtにすると後述のDeepDanbooruと競合します）。

複数の教師データフォルダがある場合には、それぞれのフォルダに対して実行してください。

なお、推論にランダム性があるため、実行するたびに結果が変わります。固定する場合には--seedオプションで `--seed 42` のように乱数seedを指定してください。

その他のオプションは `--help` でヘルプをご参照ください（パラメータの意味についてはドキュメントがまとまっていないようで、ソースを見るしかないようです）。

デフォルトでは拡張子.captionでキャプションファイルが生成されます。

![captionが生成されたフォルダ](https://user-images.githubusercontent.com/52813779/208908845-48a9d36c-f6ee-4dae-af71-9ab462d1459e.png)

たとえば以下のようなキャプションが付きます。

![キャプションと画像](https://user-images.githubusercontent.com/52813779/208908947-af936957-5d73-4339-b6c8-945a52857373.png)

## DeepDanbooruによるタグ付け

danbooruタグのタグ付け自体を行わない場合は「キャプションとタグ情報の前処理」に進んでください。

タグ付けはDeepDanbooruまたはWD14Taggerで行います。WD14Taggerのほうが精度が良いようです。WD14Taggerでタグ付けする場合は、次の章へ進んでください。

### 環境整備

DeepDanbooru https://github.com/KichangKim/DeepDanbooru  を作業フォルダにcloneしてくるか、zipをダウンロードして展開します。私はzipで展開しました。
またDeepDanbooruのReleasesのページ https://github.com/KichangKim/DeepDanbooru/releases  の「DeepDanbooru Pretrained Model v3-20211112-sgd-e28」のAssetsから、deepdanbooru-v3-20211112-sgd-e28.zipをダウンロードしてきてDeepDanbooruのフォルダに展開します。

以下からダウンロードします。Assetsをクリックして開き、そこからダウンロードします。

![DeepDanbooruダウンロードページ](https://user-images.githubusercontent.com/52813779/208909417-10e597df-7085-41ee-bd06-3e856a1339df.png)

以下のようなこういうディレクトリ構造にしてください

![DeepDanbooruのディレクトリ構造](https://user-images.githubusercontent.com/52813779/208909486-38935d8b-8dc6-43f1-84d3-fef99bc471aa.png)

Diffusersの環境に必要なライブラリをインストールします。DeepDanbooruのフォルダに移動してインストールします（実質的にはtensorflow-ioが追加されるだけだと思います）。

```
pip install -r requirements.txt
```

続いてDeepDanbooru自体をインストールします。

```
pip install .
```

以上でタグ付けの環境整備は完了です。

### タグ付けの実施
DeepDanbooruのフォルダに移動し、deepdanbooruを実行してタグ付けを行います。

```
deepdanbooru evaluate <教師データフォルダ> --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

教師データを親フォルダのtrain_dataに置いた場合、以下のようになります。

```
deepdanbooru evaluate ../train_data --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

タグファイルが教師データ画像と同じディレクトリに、同じファイル名、拡張子.txtで作成されます。1件ずつ処理されるためわりと遅いです。

複数の教師データフォルダがある場合には、それぞれのフォルダに対して実行してください。

以下のように生成されます。

![DeepDanbooruの生成ファイル](https://user-images.githubusercontent.com/52813779/208909855-d21b9c98-f2d3-4283-8238-5b0e5aad6691.png)

こんな感じにタグが付きます（すごい情報量……）。

![DeepDanbooruタグと画像](https://user-images.githubusercontent.com/52813779/208909908-a7920174-266e-48d5-aaef-940aba709519.png)

## WD14Taggerによるタグ付け

DeepDanbooruの代わりにWD14Taggerを用いる手順です。

Automatic1111氏のWebUIで使用しているtaggerを利用します。こちらのgithubページ（https://github.com/toriato/stable-diffusion-webui-wd14-tagger#mrsmilingwolfs-model-aka-waifu-diffusion-14-tagger ）の情報を参考にさせていただきました。

最初の環境整備で必要なモジュールはインストール済みです。また重みはHugging Faceから自動的にダウンロードしてきます。

### タグ付けの実施

スクリプトを実行してタグ付けを行います。
```
python tag_images_by_wd14_tagger.py --batch_size <バッチサイズ> <教師データフォルダ>
```

教師データを親フォルダのtrain_dataに置いた場合、以下のようになります。
```
python tag_images_by_wd14_tagger.py --batch_size 4 ..\train_data
```

初回起動時にはモデルファイルがwd14_tagger_modelフォルダに自動的にダウンロードされます（フォルダはオプションで変えられます）。以下のようになります。

![ダウンロードされたファイル](https://user-images.githubusercontent.com/52813779/208910447-f7eb0582-90d6-49d3-a666-2b508c7d1842.png)

タグファイルが教師データ画像と同じディレクトリに、同じファイル名、拡張子.txtで作成されます。

![生成されたタグファイル](https://user-images.githubusercontent.com/52813779/208910534-ea514373-1185-4b7d-9ae3-61eb50bc294e.png)

![タグと画像](https://user-images.githubusercontent.com/52813779/208910599-29070c15-7639-474f-b3e4-06bd5a3df29e.png)

threshオプションで、判定されたタグのconfidence（確信度）がいくつ以上でタグをつけるかが指定できます。デフォルトはWD14Taggerのサンプルと同じ0.35です。値を下げるとより多くのタグが付与されますが、精度は下がります。

batch_sizeはGPUのVRAM容量に応じて増減してください。大きいほうが速くなります（VRAM 12GBでももう少し増やせると思います）。caption_extensionオプションでタグファイルの拡張子を変更できます。デフォルトは.txtです。

model_dirオプションでモデルの保存先フォルダを指定できます。

またforce_downloadオプションを指定すると保存先フォルダがあってもモデルを再ダウンロードします。

複数の教師データフォルダがある場合には、それぞれのフォルダに対して実行してください。

## キャプションとタグ情報の前処理

スクリプトから処理しやすいようにキャプションとタグをメタデータとしてひとつのファイルにまとめます。

### キャプションの前処理

キャプションをメタデータに入れるには、作業フォルダ内で以下を実行してください（キャプションを学習に使わない場合は実行不要です）（実際は1行で記述します、以下同様）。`--full_path` オプションを指定してメタデータに画像ファイルの場所をフルパスで格納します。このオプションを省略すると相対パスで記録されますが、フォルダ指定が `.toml` ファイル内で別途必要になります。

```
python merge_captions_to_metadata.py --full_apth <教師データフォルダ>
　  --in_json <読み込むメタデータファイル名> <メタデータファイル名>
```

メタデータファイル名は任意の名前です。
教師データがtrain_data、読み込むメタデータファイルなし、メタデータファイルがmeta_cap.jsonの場合、以下のようになります。

```
python merge_captions_to_metadata.py --full_path train_data meta_cap.json
```

caption_extensionオプションでキャプションの拡張子を指定できます。

複数の教師データフォルダがある場合には、full_path引数を指定しつつ、それぞれのフォルダに対して実行してください。

```
python merge_captions_to_metadata.py --full_path 
    train_data1 meta_cap1.json
python merge_captions_to_metadata.py --full_path --in_json meta_cap1.json 
    train_data2 meta_cap2.json
```

in_jsonを省略すると書き込み先メタデータファイルがあるとそこから読み込み、そこに上書きします。

__※in_jsonオプションと書き込み先を都度書き換えて、別のメタデータファイルへ書き出すようにすると安全です。__

### タグの前処理

同様にタグもメタデータにまとめます（タグを学習に使わない場合は実行不要です）。
```
python merge_dd_tags_to_metadata.py --full_path <教師データフォルダ> 
    --in_json <読み込むメタデータファイル名> <書き込むメタデータファイル名>
```

先と同じディレクトリ構成で、meta_cap.jsonを読み、meta_cap_dd.jsonに書きだす場合、以下となります。
```
python merge_dd_tags_to_metadata.py --full_path train_data --in_json meta_cap.json meta_cap_dd.json
```

複数の教師データフォルダがある場合には、full_path引数を指定しつつ、それぞれのフォルダに対して実行してください。

```
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap2.json
    train_data1 meta_cap_dd1.json
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap_dd1.json 
    train_data2 meta_cap_dd2.json
```

in_jsonを省略すると書き込み先メタデータファイルがあるとそこから読み込み、そこに上書きします。

__※in_jsonオプションと書き込み先を都度書き換えて、別のメタデータファイルへ書き出すようにすると安全です。__

### キャプションとタグのクリーニング

ここまででメタデータファイルにキャプションとDeepDanbooruのタグがまとめられています。ただ自動キャプショニングにしたキャプションは表記ゆれなどがあり微妙（※）ですし、タグにはアンダースコアが含まれていたりratingが付いていたりしますので（DeepDanbooruの場合）、エディタの置換機能などを用いてキャプションとタグのクリーニングをしたほうがいいでしょう。

※たとえばアニメ絵の少女を学習する場合、キャプションにはgirl/girls/woman/womenなどのばらつきがあります。また「anime girl」なども単に「girl」としたほうが適切かもしれません。

クリーニング用のスクリプトが用意してありますので、スクリプトの内容を状況に応じて編集してお使いください。

（教師データフォルダの指定は不要になりました。メタデータ内の全データをクリーニングします。）

```
python clean_captions_and_tags.py <読み込むメタデータファイル名> <書き込むメタデータファイル名>
```

--in_jsonは付きませんのでご注意ください。たとえば次のようになります。

```
python clean_captions_and_tags.py meta_cap_dd.json meta_clean.json
```

以上でキャプションとタグの前処理は完了です。

## latentsの事前取得

※ このステップは必須ではありません。省略しても学習時にlatentsを取得しながら学習できます。
また学習時に `random_crop` や `color_aug` などを行う場合にはlatentsの事前取得はできません（画像を毎回変えながら学習するため）。事前取得をしない場合、ここまでのメタデータで学習できます。

あらかじめ画像の潜在表現を取得しディスクに保存しておきます。それにより、学習を高速に進めることができます。あわせてbucketing（教師データをアスペクト比に応じて分類する）を行います。

作業フォルダで以下のように入力してください。
```
python prepare_buckets_latents.py --full_path <教師データフォルダ>  
    <読み込むメタデータファイル名> <書き込むメタデータファイル名> 
    <fine tuningするモデル名またはcheckpoint> 
    --batch_size <バッチサイズ> 
    --max_resolution <解像度 幅,高さ> 
    --mixed_precision <精度>
```

モデルがmodel.ckpt、バッチサイズ4、学習解像度は512\*512、精度no（float32）で、meta_clean.jsonからメタデータを読み込み、meta_lat.jsonに書き込む場合、以下のようになります。

```
python prepare_buckets_latents.py --full_path 
    train_data meta_clean.json meta_lat.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no
```

教師データフォルダにnumpyのnpz形式でlatentsが保存されます。

解像度の最小サイズを--min_bucket_resoオプションで、最大サイズを--max_bucket_resoで指定できます。デフォルトはそれぞれ256、1024です。たとえば最小サイズに384を指定すると、256\*1024や320\*768などの解像度は使わなくなります。
解像度を768\*768のように大きくした場合、最大サイズに1280などを指定すると良いでしょう。

--flip_augオプションを指定すると左右反転のaugmentation（データ拡張）を行います。疑似的にデータ量を二倍に増やすことができますが、データが左右対称でない場合に指定すると（例えばキャラクタの外見、髪型など）学習がうまく行かなくなります。


（反転した画像についてもlatentsを取得し、\*\_flip.npzファイルを保存する単純な実装です。fline_tune.pyには特にオプション指定は必要ありません。\_flip付きのファイルがある場合、flip付き・なしのファイルを、ランダムに読み込みます。）

バッチサイズはVRAM 12GBでももう少し増やせるかもしれません。
解像度は64で割り切れる数字で、"幅,高さ"で指定します。解像度はfine tuning時のメモリサイズに直結します。VRAM 12GBでは512,512が限界と思われます（※）。16GBなら512,704や512,768まで上げられるかもしれません。なお256,256等にしてもVRAM 8GBでは厳しいようです（パラメータやoptimizerなどは解像度に関係せず一定のメモリが必要なため）。

※batch size 1の学習で12GB VRAM、640,640で動いたとの報告もありました。

以下のようにbucketingの結果が表示されます。

![bucketingの結果](https://user-images.githubusercontent.com/52813779/208911419-71c00fbb-2ce6-49d5-89b5-b78d7715e441.png)

複数の教師データフォルダがある場合には、full_path引数を指定しつつ、それぞれのフォルダに対して実行してください。
```
python prepare_buckets_latents.py --full_path  
    train_data1 meta_clean.json meta_lat1.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

python prepare_buckets_latents.py --full_path 
    train_data2 meta_lat1.json meta_lat2.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

```
読み込み元と書き込み先を同じにすることも可能ですが別々の方が安全です。

__※引数を都度書き換えて、別のメタデータファイルに書き込むと安全です。__

