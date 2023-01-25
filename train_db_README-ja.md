DreamBoothのガイドです。LoRA等の追加ネットワークの学習にも同じ手順を使います。

# 概要

スクリプトの主な機能は以下の通りです。

- 8bit Adam optimizerおよびlatentのキャッシュによる省メモリ化（ShivamShrirao氏版と同様）。
- xformersによる省メモリ化。
- 512x512だけではなく任意サイズでの学習。
- augmentationによる品質の向上。
- DreamBoothだけではなくText Encoder+U-Netのfine tuningに対応。
- StableDiffusion形式でのモデルの読み書き。
- Aspect Ratio Bucketing。
- Stable Diffusion v2.0対応。

# 学習の手順

## step 1. 環境整備

このリポジトリのREADMEを参照してください。


## step 2. identifierとclassを決める

学ばせたい対象を結びつける単語identifierと、対象の属するclassを決めます。

（instanceなどいろいろな呼び方がありますが、とりあえず元の論文に合わせます。）

以下ごく簡単に説明します（詳しくは調べてください）。

classは学習対象の一般的な種別です。たとえば特定の犬種を学ばせる場合には、classはdogになります。アニメキャラならモデルによりboyやgirl、1boyや1girlになるでしょう。

identifierは学習対象を識別して学習するためのものです。任意の単語で構いませんが、元論文によると「tokinizerで1トークンになる3文字以下でレアな単語」が良いとのことです。

identifierとclassを使い、たとえば「shs dog」などでモデルを学習することで、学習させたい対象をclassから識別して学習できます。

画像生成時には「shs dog」とすれば学ばせた犬種の画像が生成されます。

（identifierとして私が最近使っているものを参考までに挙げると、``shs sts scs cpc coc cic msm usu ici lvl cic dii muk ori hru rik koo yos wny`` などです。）

## step 3. 学習用画像の準備
学習用画像を格納するフォルダを作成します。 __さらにその中に__ 、以下の名前でディレクトリを作成します。

```
<繰り返し回数>_<identifier> <class>
```

間の``_``を忘れないでください。

繰り返し回数は、正則化画像と枚数を合わせるために指定します（後述します）。

たとえば「sls frog」というプロンプトで、データを20回繰り返す場合、「20_sls frog」となります。以下のようになります。

![image](https://user-images.githubusercontent.com/52813779/210770636-1c851377-5936-4c15-90b7-8ac8ad6c2074.png)

## step 4. 正則化画像の準備
正則化画像を使う場合の手順です。使わずに学習することもできます（正則化画像を使わないと区別ができなくなるので対象class全体が影響を受けます）。

正則化画像を格納するフォルダを作成します。 __さらにその中に__  ``<繰り返し回数>_<class>`` という名前でディレクトリを作成します。

たとえば「frog」というプロンプトで、データを繰り返さない（1回だけ）場合、以下のようになります。

![image](https://user-images.githubusercontent.com/52813779/210770897-329758e5-3675-49f1-b345-c135f1725832.png)

繰り返し回数は「 __学習用画像の繰り返し回数×学習用画像の枚数≧正則化画像の繰り返し回数×正則化画像の枚数__ 」となるように指定してください。

（1 epochのデータ数が「学習用画像の繰り返し回数×学習用画像の枚数」となります。正則化画像の枚数がそれより多いと、余った部分の正則化画像は使用されません。）

## step 5. 学習の実行
スクリプトを実行します。最大限、メモリを節約したコマンドは以下のようになります（実際には1行で入力します）。

※LoRA等の追加ネットワークを学習する場合のコマンドは ``train_db.py`` ではなく ``train_network.py`` となります。また追加でnetwork_\*オプションが必要となりますので、LoRAのガイドを参照してください。

```
accelerate launch --num_cpu_threads_per_process 1 train_db.py 
    --pretrained_model_name_or_path=<.ckptまたは.safetensordまたはDiffusers版モデルのディレクトリ> 
    --train_data_dir=<学習用データのディレクトリ> 
    --reg_data_dir=<正則化画像のディレクトリ> 
    --output_dir=<学習したモデルの出力先ディレクトリ> 
    --prior_loss_weight=1.0 
    --resolution=512 
    --train_batch_size=1 
    --learning_rate=1e-6 
    --max_train_steps=1600 
    --use_8bit_adam 
    --xformers 
    --mixed_precision="bf16" 
    --cache_latents
    --gradient_checkpointing
```

num_cpu_threads_per_processには通常は1を指定するとよいようです。

pretrained_model_name_or_pathに追加学習を行う元となるモデルを指定します。Stable Diffusionのcheckpointファイル（.ckptまたは.safetensors）、Diffusersのローカルディスクにあるモデルディレクトリ、DiffusersのモデルID（"stabilityai/stable-diffusion-2"など）が指定できます。学習後のモデルの保存形式はデフォルトでは元のモデルと同じになります（save_model_asオプションで変更できます）。

prior_loss_weightは正則化画像のlossの重みです。通常は1.0を指定します。

resolutionは画像のサイズ（解像度、幅と高さ）になります。bucketing（後述）を用いない場合、学習用画像、正則化画像はこのサイズとしてください。

train_batch_sizeは学習時のバッチサイズです。max_train_stepsを1600とします。学習率learning_rateは、diffusers版では5e-6ですがStableDiffusion版は1e-6ですのでここでは1e-6を指定しています。

省メモリ化のためmixed_precision="bf16"（または"fp16"）、およびgradient_checkpointing を指定します。

xformersオプションを指定し、xformersのCrossAttentionを用います。xformersをインストールしていない場合、エラーとなる場合（mixed_precisionなしの場合、私の環境ではエラーとなりました）、代わりにmem_eff_attnオプションを指定すると省メモリ版CrossAttentionを使用します（速度は遅くなります）。

省メモリ化のためcache_latentsオプションを指定してVAEの出力をキャッシュします。

ある程度メモリがある場合はたとえば以下のように指定します。

```
accelerate launch --num_cpu_threads_per_process 8 train_db.py 
    --pretrained_model_name_or_path=<.ckptまたは.safetensordまたはDiffusers版モデルのディレクトリ> 
    --train_data_dir=<学習用データのディレクトリ> 
    --reg_data_dir=<正則化画像のディレクトリ> 
    --output_dir=<学習したモデルの出力先ディレクトリ> 
    --prior_loss_weight=1.0 
    --resolution=512 
    --train_batch_size=4 
    --learning_rate=1e-6 
    --max_train_steps=400 
    --use_8bit_adam 
    --xformers 
    --mixed_precision="bf16" 
    --cache_latents
```

gradient_checkpointingを外し高速化します（メモリ使用量は増えます）。バッチサイズを増やし、高速化と精度向上を図ります。

bucketing（後述）を利用しかつaugmentation（後述）を使う場合の例は以下のようになります。

```
accelerate launch --num_cpu_threads_per_process 8 train_db.py 
    --pretrained_model_name_or_path=<.ckptまたは.safetensordまたはDiffusers版モデルのディレクトリ> 
    --train_data_dir=<学習用データのディレクトリ> 
    --reg_data_dir=<正則化画像のディレクトリ> 
    --output_dir=<学習したモデルの出力先ディレクトリ> 
    --resolution=768,512 
    --train_batch_size=20 --learning_rate=5e-6 --max_train_steps=800 
    --use_8bit_adam --xformers --mixed_precision="bf16" 
    --save_every_n_epochs=1 --save_state --save_precision="bf16" 
    --logging_dir=logs 
    --enable_bucket --min_bucket_reso=384 --max_bucket_reso=1280 
    --color_aug --flip_aug --gradient_checkpointing --seed 42
```

### ステップ数について
省メモリ化のため、ステップ当たりの学習回数がtrain_dreambooth.pyの半分になっています（対象の画像と正則化画像を同一のバッチではなく別のバッチに分割して学習するため）。
元のDiffusers版やXavierXiao氏のStableDiffusion版とほぼ同じ学習を行うには、ステップ数を倍にしてください。

（shuffle=Trueのため厳密にはデータの順番が変わってしまいますが、学習には大きな影響はないと思います。）

## 学習したモデルで画像生成する

学習が終わると指定したフォルダにlast.ckptという名前でcheckpointが出力されます（DiffUsers版モデルを学習した場合はlastフォルダになります）。

v1.4/1.5およびその他の派生モデルの場合、このモデルでAutomatic1111氏のWebUIなどで推論できます。models\Stable-diffusionフォルダに置いてください。

v2.xモデルでWebUIで画像生成する場合、モデルの仕様が記述された.yamlファイルが別途必要になります。v2.x baseの場合はv2-inference.yamlを、768/vの場合はv2-inference-v.yamlを、同じフォルダに置き、拡張子の前の部分をモデルと同じ名前にしてください。

![image](https://user-images.githubusercontent.com/52813779/210776915-061d79c3-6582-42c2-8884-8b91d2f07313.png)

各yamlファイルは[Stability AIのSD2.0のリポジトリ](https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion)にあります。

# その他の学習オプション

## Stable Diffusion 2.0対応 --v2 / --v_parameterization
Hugging Faceのstable-diffusion-2-baseを使う場合はv2オプションを、stable-diffusion-2または768-v-ema.ckptを使う場合はv2とv_parameterizationの両方のオプションを指定してください。

なおSD 2.0の学習はText Encoderが大きくなっているためVRAM 12GBでは厳しいようです。

Stable Diffusion 2.0では大きく以下の点が変わっています。

1. 使用するTokenizer
2. 使用するText Encoderおよび使用する出力層（2.0は最後から二番目の層を使う）
3. Text Encoderの出力次元数（768->1024）
4. U-Netの構造（CrossAttentionのhead数など）
5. v-parameterization（サンプリング方法が変更されているらしい）

このうちbaseでは1～4が、baseのつかない方（768-v）では1～5が採用されています。1～4を有効にするのがv2オプション、5を有効にするのがv_parameterizationオプションです。

## 学習データの確認 --debug_dataset
このオプションを付けることで学習を行う前に事前にどのような画像データ、キャプションで学習されるかを確認できます。Escキーを押すと終了してコマンドラインに戻ります。

※Colabなど画面が存在しない環境で実行するとハングするようですのでご注意ください。

## Text Encoderの学習を途中から行わない --stop_text_encoder_training
stop_text_encoder_trainingオプションに数値を指定すると、そのステップ数以降はText Encoderの学習を行わずU-Netだけ学習します。場合によっては精度の向上が期待できるかもしれません。

（恐らくText Encoderだけ先に過学習することがあり、それを防げるのではないかと推測していますが、詳細な影響は不明です。）

## VAEを別途読み込んで学習する --vae
vaeオプションにStable Diffusionのcheckpoint、VAEのcheckpointファイル、DiffusesのモデルまたはVAE（ともにローカルまたはHugging FaceのモデルIDが指定できます）のいずれかを指定すると、そのVAEを使って学習します（latentsのキャッシュ時または学習中のlatents取得時）。
保存されるモデルはこのVAEを組み込んだものになります。

## 学習途中での保存 --save_every_n_epochs / --save_state / --resume
save_every_n_epochsオプションに数値を指定すると、そのエポックごとに学習途中のモデルを保存します。

save_stateオプションを同時に指定すると、optimizer等の状態も含めた学習状態を合わせて保存します（checkpointから学習再開するのに比べて、精度の向上、学習時間の短縮が期待できます）。学習状態は保存先フォルダに"epoch-??????-state"（??????はエポック数）という名前のフォルダで出力されます。長時間にわたる学習時にご利用ください。

保存された学習状態から学習を再開するにはresumeオプションを使います。学習状態のフォルダを指定してください。

なおAcceleratorの仕様により(?)、エポック数、global stepは保存されておらず、resumeしたときにも1からになりますがご容赦ください。

## Tokenizerのパディングをしない --no_token_padding
no_token_paddingオプションを指定するとTokenizerの出力をpaddingしません（Diffusers版の旧DreamBoothと同じ動きになります）。

## 任意サイズの画像での学習 --resolution
正方形以外で学習できます。resolutionに「448,640」のように「幅,高さ」で指定してください。幅と高さは64で割り切れる必要があります。学習用画像、正則化画像のサイズを合わせてください。

個人的には縦長の画像を生成することが多いため「448,640」などで学習することもあります。

## Aspect Ratio Bucketing --enable_bucket / --min_bucket_reso / --max_bucket_reso
enable_bucketオプションを指定すると有効になります。Stable Diffusionは512x512で学習されていますが、それに加えて256x768や384x640といった解像度でも学習します。

このオプションを指定した場合は、学習用画像、正則化画像を特定の解像度に統一する必要はありません。いくつかの解像度（アスペクト比）から最適なものを選び、その解像度で学習します。
解像度は64ピクセル単位のため、元画像とアスペクト比が完全に一致しない場合がありますが、その場合は、はみ出した部分がわずかにトリミングされます。

解像度の最小サイズをmin_bucket_resoオプションで、最大サイズをmax_bucket_resoで指定できます。デフォルトはそれぞれ256、1024です。
たとえば最小サイズに384を指定すると、256x1024や320x768などの解像度は使わなくなります。
解像度を768x768のように大きくした場合、最大サイズに1280などを指定しても良いかもしれません。

なおAspect Ratio Bucketingを有効にするときには、正則化画像についても、学習用画像と似た傾向の様々な解像度を用意した方がいいかもしれません。

（ひとつのバッチ内の画像が学習用画像、正則化画像に偏らなくなるため。そこまで大きな影響はないと思いますが……。）

## augmentation --color_aug / --flip_aug
augmentationは学習時に動的にデータを変化させることで、モデルの性能を上げる手法です。color_augで色合いを微妙に変えつつ、flip_augで左右反転をしつつ、学習します。

動的にデータを変化させるため、cache_latentsオプションと同時に指定できません。

## 保存時のデータ精度の指定 --save_precision
save_precisionオプションにfloat、fp16、bf16のいずれかを指定すると、その形式でcheckpointを保存します（Stable Diffusion形式で保存する場合のみ）。checkpointのサイズを削減したい場合などにお使いください。

## 任意の形式で保存する --save_model_as
モデルの保存形式を指定します。ckpt、safetensors、diffusers、diffusers_safetensorsのいずれかを指定してください。

Stable Diffusion形式（ckptまたはsafetensors）を読み込み、Diffusers形式で保存する場合、不足する情報はHugging Faceからv1.5またはv2.1の情報を落としてきて補完します。

## 学習ログの保存 --logging_dir / --log_prefix
logging_dirオプションにログ保存先フォルダを指定してください。TensorBoard形式のログが保存されます。

たとえば--logging_dir=logsと指定すると、作業フォルダにlogsフォルダが作成され、その中の日時フォルダにログが保存されます。
また--log_prefixオプションを指定すると、日時の前に指定した文字列が追加されます。「--logging_dir=logs --log_prefix=db_style1_」などとして識別用にお使いください。

TensorBoardでログを確認するには、別のコマンドプロンプトを開き、作業フォルダで以下のように入力します（tensorboardはDiffusersのインストール時にあわせてインストールされると思いますが、もし入っていないならpip install tensorboardで入れてください）。

```
tensorboard --logdir=logs
```

その後ブラウザを開き、http://localhost:6006/ へアクセスすると表示されます。

## 学習率のスケジューラ関連の指定 --lr_scheduler / --lr_warmup_steps
lr_schedulerオプションで学習率のスケジューラをlinear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmupから選べます。デフォルトはconstantです。lr_warmup_stepsでスケジューラのウォームアップ（だんだん学習率を変えていく）ステップ数を指定できます。詳細については各自お調べください。

## 勾配をfp16とした学習（実験的機能） --full_fp16
full_fp16オプションを指定すると勾配を通常のfloat32からfloat16（fp16）に変更して学習します（mixed precisionではなく完全なfp16学習になるようです）。
これによりSD1.xの512x512サイズでは8GB未満、SD2.xの512x512サイズで12GB未満のVRAM使用量で学習できるようです。

あらかじめaccelerate configでfp16を指定し、オプションで ``mixed_precision="fp16"`` としてください（bf16では動作しません）。

メモリ使用量を最小化するためには、xformers、use_8bit_adam、cache_latents、gradient_checkpointingの各オプションを指定し、train_batch_sizeを1としてください。

（余裕があるようならtrain_batch_sizeを段階的に増やすと若干精度が上がるはずです。）

PyTorchのソースにパッチを当てて無理やり実現しています（PyTorch 1.12.1と1.13.0で確認）。精度はかなり落ちますし、途中で学習失敗する確率も高くなります。
学習率やステップ数の設定もシビアなようです。それらを認識したうえで自己責任でお使いください。

# その他の学習方法

## 複数class、複数対象（identifier）の学習
方法は単純で、学習用画像のフォルダ内に ``繰り返し回数_<identifier> <class>`` のフォルダを複数、正則化画像フォルダにも同様に ``繰り返し回数_<class>`` のフォルダを複数、用意してください。

たとえば「sls frog」と「cpc rabbit」を同時に学習する場合、以下のようになります。

![image](https://user-images.githubusercontent.com/52813779/210777933-a22229db-b219-4cd8-83ca-e87320fc4192.png)

classがひとつで対象が複数の場合、正則化画像フォルダはひとつで構いません。たとえば1girlにキャラAとキャラBがいる場合は次のようにします。

- train_girls
  - 10_sls 1girl
  - 10_cpc 1girl
- reg_girls
  - 1_1girl

データ数にばらつきがある場合、繰り返し回数を調整してclass、identifierごとの枚数を統一すると良い結果が得られることがあるようです。

## DreamBoothでキャプションを使う
学習用画像、正則化画像のフォルダに、画像と同じファイル名で、拡張子.caption（オプションで変えられます）のファイルを置くと、そのファイルからキャプションを読み込みプロンプトとして学習します。

※それらの画像の学習に、フォルダ名（identifier class）は使用されなくなります。

各画像にキャプションを付けることで（BLIP等を使っても良いでしょう）、学習したい属性をより明確にできるかもしれません。

キャプションファイルの拡張子はデフォルトで.captionです。--caption_extensionで変更できます。--shuffle_captionオプションで学習時のキャプションについて、カンマ区切りの各部分をシャッフルしながら学習します。

