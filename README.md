# HOWTO

This repo provide all the required config to run the Dreambooth version found in this note: https://note.com/kohya_ss/n/nee3ed1649fb6
The setup of bitsandbytes with Adam8bit support for windows: https://note.com/kohya_ss/n/n47f654dc161e

## Required Dependencies

Python 3.10.6 and Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Installation

Open a regular Powershell terminal and type the following inside:

```powershell
# Clone the Kohya_ss repository
git clone https://github.com/bmaltais/kohya_ss.git

# Navigate to the newly cloned directory
cd kohya_ss

# Create a virtual environment using the system-site-packages option
python -m venv --system-site-packages venv

# Activate the virtual environment
.\venv\Scripts\activate

# Install the required packages
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

# Copy the necessary files to the virtual environment's site-packages directory
cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

# Configure the accelerate utility
accelerate config

```

Answers to accelerate config:

```txt
- 0
- 0
- NO
- NO
- All
- fp16
```

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd kohya_ss
git pull
.\venv\Scripts\activate
pip install --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

## Folders configuration

Refer to the note to understand how to create the folde structure. In short it should look like:

```
<arbitrary folder name>
|- <arbitrary class folder name>
    |- <repeat count>_<class>
|- <arbitrary training folder name>
   |- <repeat count>_<token> <class>
```

Example for `asd dog` where `asd` is the token word and `dog` is the class. In this example the regularization `dog` class images contained in the folder will be repeated only 1 time and the `asd dog` images will be repeated 20 times:

```
my_asd_dog_dreambooth
|- reg_dog
    |- 1_dog
       `- reg_image_1.png
       `- reg_image_2.png
       ...
       `- reg_image_256.png
|- train_dog
    |- 20_asd dog
       `- dog1.png
       ...
       `- dog8.png
```

## Execution

### SD1.5 example

Edit and paste the following in a Powershell terminal:

```powershell
accelerate launch --num_cpu_threads_per_process 6 train_db_fixed.py `
    --pretrained_model_name_or_path="D:\models\last.ckpt" `
    --train_data_dir="D:\dreambooth\train_bernard\train_man" `
    --reg_data_dir="D:\dreambooth\train_bernard\reg_man" `
    --output_dir="D:\dreambooth\train_bernard" `
    --prior_loss_weight=1.0 `
    --resolution=512 `
    --train_batch_size=1 `
    --learning_rate=1e-6 `
    --max_train_steps=2100 `
    --use_8bit_adam `
    --xformers `
    --mixed_precision="fp16" `
    --cache_latents `
    --gradient_checkpointing `
    --save_every_n_epochs=1
```

### SD2.0 512 Base example

```powershell
# variable values
$pretrained_model_name_or_path = "D:\models\512-base-ema.ckpt"
$data_dir = "D:\models\dariusz_zawadzki\kohya_reg\data"
$reg_data_dir = "D:\models\dariusz_zawadzki\kohya_reg\reg"
$logging_dir = "D:\models\dariusz_zawadzki\logs"
$output_dir = "D:\models\dariusz_zawadzki\train_db_fixed_model_reg_v2"
$resolution = "512,512"
$lr_scheduler="polynomial"
$cache_latents = 1 # 1 = true, 0 = false

$image_num = Get-ChildItem $data_dir -Recurse -File -Include *.png, *.jpg, *.webp | Measure-Object | %{$_.Count}

Write-Output "image_num: $image_num"

$dataset_repeats = 200
$learning_rate = 2e-6
$train_batch_size = 4
$epoch = 1
$save_every_n_epochs=1
$mixed_precision="bf16"
$num_cpu_threads_per_process=6

# You should not have to change values past this point
if ($cache_latents -eq 1) {
    $cache_latents_value="--cache_latents"
}
else {
    $cache_latents_value=""
}

$repeats = $image_num * $dataset_repeats
$mts = [Math]::Ceiling($repeats / $train_batch_size * $epoch)

Write-Output "Repeats: $repeats"

cd D:\kohya_ss
.\venv\Scripts\activate

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process train_db_fixed.py `
    --v2 `
    --pretrained_model_name_or_path=$pretrained_model_name_or_path `
    --train_data_dir=$data_dir `
    --output_dir=$output_dir `
    --resolution=$resolution `
    --train_batch_size=$train_batch_size `
    --learning_rate=$learning_rate `
    --max_train_steps=$mts `
    --use_8bit_adam `
    --xformers `
    --mixed_precision=$mixed_precision `
    $cache_latents_value `
    --save_every_n_epochs=$save_every_n_epochs `
    --logging_dir=$logging_dir `
    --save_precision="fp16" `
    --reg_data_dir=$reg_data_dir `
    --seed=494481440 `
    --lr_scheduler=$lr_scheduler

# Add the inference yaml file along with the model for proper loading. Need to have the same name as model... Most likelly "last.yaml" in our case.
cp v2_inference\v2-inference.yaml $output_dir"\last.yaml"
```

### SD2.0 768v Base example

```powershell
# variable values
$pretrained_model_name_or_path = "C:\Users\berna\Downloads\768-v-ema.ckpt"
$data_dir = "D:\dreambooth\train_paper_artwork\kohya\data"
$logging_dir = "D:\dreambooth\train_paper_artwork"
$output_dir = "D:\models\paper_artwork\train_db_fixed_model_v2_768v"
$resolution = "768,768"
$lr_scheduler="polynomial"
$cache_latents = 1 # 1 = true, 0 = false

$image_num = Get-ChildItem $data_dir -Recurse -File -Include *.png, *.jpg, *.webp | Measure-Object | %{$_.Count}

Write-Output "image_num: $image_num"

$dataset_repeats = 200
$learning_rate = 2e-6
$train_batch_size = 4
$epoch = 1
$save_every_n_epochs=1
$mixed_precision="bf16"
$num_cpu_threads_per_process=6

# You should not have to change values past this point
if ($cache_latents -eq 1) {
    $cache_latents_value="--cache_latents"
}
else {
    $cache_latents_value=""
}

$repeats = $image_num * $dataset_repeats
$mts = [Math]::Ceiling($repeats / $train_batch_size * $epoch)

Write-Output "Repeats: $repeats"

cd D:\kohya_ss
.\venv\Scripts\activate

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process train_db_fixed.py `
    --v2 `
    --v_parameterization `
    --pretrained_model_name_or_path=$pretrained_model_name_or_path `
    --train_data_dir=$data_dir `
    --output_dir=$output_dir `
    --resolution=$resolution `
    --train_batch_size=$train_batch_size `
    --learning_rate=$learning_rate `
    --max_train_steps=$mts `
    --use_8bit_adam `
    --xformers `
    --mixed_precision=$mixed_precision `
    $cache_latents_value `
    --save_every_n_epochs=$save_every_n_epochs `
    --logging_dir=$logging_dir `
    --save_precision="fp16" `
    --seed=494481440 `
    --lr_scheduler=$lr_scheduler

# Add the inference 768v yaml file along with the model for proper loading. Need to have the same name as model... Most likelly "last.yaml" in our case.
cp v2_inference\v2-inference-v.yaml $output_dir"\last.yaml"
```

## Finetuning

If you would rather use model finetuning rather than the dreambooth method you can use a command similat to the following. The advantage of fine tuning is that you do not need to worry about regularization images... but you need to provide captions for every images. The caption will be used to train the model. You can use auto1111 to preprocess your training images and add either BLIP or danbooru captions to them. You then need to edit those to add the name of the model and correct any wrong description.

```
accelerate launch --num_cpu_threads_per_process 6 train_db_fixed-ber.py `
    --pretrained_model_name_or_path="D:\models\alexandrine_teissier_and_bernard_maltais-400-kohya-sd15-v1.ckpt" `
    --train_data_dir="D:\dreambooth\source\alet_et_bernard\landscape-pp" `
    --output_dir="D:\dreambooth\train_alex_and_bernard" `
    --resolution="640,448" `
    --train_batch_size=1 `
    --learning_rate=1e-6 `
    --max_train_steps=550 `
    --use_8bit_adam `
    --xformers `
    --mixed_precision="fp16" `
    --cache_latents `
    --save_every_n_epochs=1 `
    --fine_tuning `
    --enable_bucket `
    --dataset_repeats=200 `
    --seed=23 `
    ---save_precision="fp16"
```

Refer to this url for more details about finetuning: https://note.com/kohya_ss/n/n1269f1e1a54e

## Options list

```txt
usage: train_db_fixed.py [-h] [--v2] [--v_parameterization] [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                         [--fine_tuning] [--shuffle_caption] [--caption_extention CAPTION_EXTENTION]
                         [--caption_extension CAPTION_EXTENSION] [--train_data_dir TRAIN_DATA_DIR]
                         [--reg_data_dir REG_DATA_DIR] [--dataset_repeats DATASET_REPEATS] [--output_dir OUTPUT_DIR]
                         [--save_every_n_epochs SAVE_EVERY_N_EPOCHS] [--save_state] [--resume RESUME]
                         [--prior_loss_weight PRIOR_LOSS_WEIGHT] [--no_token_padding]
                         [--stop_text_encoder_training STOP_TEXT_ENCODER_TRAINING] [--color_aug] [--flip_aug]
                         [--face_crop_aug_range FACE_CROP_AUG_RANGE] [--random_crop] [--debug_dataset]
                         [--resolution RESOLUTION] [--train_batch_size TRAIN_BATCH_SIZE] [--use_8bit_adam] [--mem_eff_attn]    
                         [--xformers] [--cache_latents] [--enable_bucket] [--min_bucket_reso MIN_BUCKET_RESO]
                         [--max_bucket_reso MAX_BUCKET_RESO] [--learning_rate LEARNING_RATE]
                         [--max_train_steps MAX_TRAIN_STEPS] [--seed SEED] [--gradient_checkpointing]
                         [--mixed_precision {no,fp16,bf16}] [--save_precision {None,float,fp16,bf16}] [--clip_skip CLIP_SKIP]  
                         [--logging_dir LOGGING_DIR] [--lr_scheduler LR_SCHEDULER] [--lr_warmup_steps LR_WARMUP_STEPS]

options:
  -h, --help            show this help message and exit
  --v2                  load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む
  --v_parameterization  enable v-parameterization training / v-parameterization学習を有効にする
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint /
                        学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル
  --fine_tuning         fine tune the model instead of DreamBooth / DreamBoothではなくfine tuningする
  --shuffle_caption     shuffle comma-separated caption / コンマで区切られたcaptionの各要素をshuffleする
  --caption_extention CAPTION_EXTENTION
                        extension of caption files (backward compatiblity) / 読み込むcaptionファイルの拡張子（スペルミスを残し てあります）
  --caption_extension CAPTION_EXTENSION
                        extension of caption files / 読み込むcaptionファイルの拡張子
  --train_data_dir TRAIN_DATA_DIR
                        directory for train images / 学習画像データのディレクトリ
  --reg_data_dir REG_DATA_DIR
                        directory for regularization images / 正則化画像データのディレクトリ
  --dataset_repeats DATASET_REPEATS
                        repeat dataset in fine tuning / fine tuning時にデータセットを繰り返す回数
  --output_dir OUTPUT_DIR
                        directory to output trained model (default format is same to input) /
                        学習後のモデル出力先ディレクトリ（デフォルトの保存形式は読み込んだ形式と同じ）
  --save_every_n_epochs SAVE_EVERY_N_EPOCHS
                        save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存します
  --save_state          save training state additionally (including optimizer states etc.) / optimizerなど学習状態も含めたstateを追加で保存する
  --resume RESUME       saved state to resume training / 学習再開するモデルのstate
  --prior_loss_weight PRIOR_LOSS_WEIGHT
                        loss weight for regularization images / 正則化画像のlossの重み
  --no_token_padding    disable token padding (same as Diffuser's DreamBooth) / トークンのpaddingを無効にする（Diffusers版DreamBoothと同じ動作）
  --stop_text_encoder_training STOP_TEXT_ENCODER_TRAINING
                        steps to stop text encoder training / Text Encoderの学習を止めるステップ数
  --color_aug           enable weak color augmentation / 学習時に色合いのaugmentationを有効にする
  --flip_aug            enable horizontal flip augmentation / 学習時に左右反転のaugmentationを有効にする
  --face_crop_aug_range FACE_CROP_AUG_RANGE
                        enable face-centered crop augmentation and its range (e.g. 2.0,4.0) /
                        学習時に顔を中心とした切り出しaugmentationを有効にするときは倍率を指定する（例：2.0,4.0）
  --random_crop         enable random crop (for style training in face-centered crop augmentation) /
                        ランダムな切り出しを有効にする（顔を中心としたaugmentationを行うときに画風の学習用に指定する）
  --debug_dataset       show images for debugging (do not train) / デバッグ用に学習データを画面表示する（学習は行わない）      
  --resolution RESOLUTION
                        resolution in training ('size' or 'width,height') / 学習時の画像解像度（'サイズ'指定、または'幅,高さ'指定）
  --train_batch_size TRAIN_BATCH_SIZE
                        batch size for training (1 means one train or reg data, not train/reg pair) /
                        学習時のバッチサイズ（1でtrain/regをそれぞれ1件ずつ学習）
  --use_8bit_adam       use 8bit Adam optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）
  --mem_eff_attn        use memory efficient attention for CrossAttention / CrossAttentionに省メモリ版attentionを使う
  --xformers            use xformers for CrossAttention / CrossAttentionにxformersを使う
  --cache_latents       cache latents to reduce memory (augmentations must be disabled) /
                        メモリ削減のためにlatentをcacheする（augmentationは使用不可）
  --enable_bucket       enable buckets for multi aspect ratio training / 複数解像度学習のためのbucketを有効にする
  --min_bucket_reso MIN_BUCKET_RESO
                        minimum resolution for buckets / bucketの最小解像度
  --max_bucket_reso MAX_BUCKET_RESO
                        maximum resolution for buckets / bucketの最小解像度
  --learning_rate LEARNING_RATE
                        learning rate / 学習率
  --max_train_steps MAX_TRAIN_STEPS
                        training steps / 学習ステップ数
  --seed SEED           random seed for training / 学習時の乱数のseed
  --gradient_checkpointing
                        enable gradient checkpointing / grandient checkpointingを有効にする
  --mixed_precision {no,fp16,bf16}
                        use mixed precision / 混合精度を使う場合、その精度
  --save_precision {None,float,fp16,bf16}
                        precision in saving (available in StableDiffusion checkpoint) /
  --logging_dir LOGGING_DIR
                        enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する
  --lr_scheduler LR_SCHEDULER
                        scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial,
                        constant (default), constant_with_warmup
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps for the warmup in the lr scheduler (default is 0) / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）
```

## Change history

* 11/30 (v13) update:
    - fix training text encoder at specified step (`--stop_text_encoder_training=<step #>`) that was causing both Unet and text encoder training to stop completely at the specified step rather than continue without text encoding training.
* 11/29 (v12) update:
    - stop training text encoder at specified step (`--stop_text_encoder_training=<step #>`)
    - tqdm smoothing
    - updated fine tuning script to support SD2.0 768/v
* 11/27 (v11) update:
    - DiffUsers 0.9.0 is required. Update with `pip install --upgrade -r requirements.txt` in the virtual environment.
    - The way captions are handled in DreamBooth has changed. When a caption file existed, the file's caption was added to the folder caption until v10, but from v11 it is only the file's caption. Please be careful.
    - Fixed a bug where prior_loss_weight was applied to learning images. Sorry for the inconvenience.
    - Compatible with Stable Diffusion v2.0. Add the `--v2` option. If you are using `768-v-ema.ckpt` or `stable-diffusion-2` instead of `stable-diffusion-v2-base`, add `--v_parameterization` as well. Learn more about other options.
    - Added options related to the learning rate scheduler.
    - You can download and use DiffUsers models directly from Hugging Face. In addition, DiffUsers models can be saved during training.
* 11/21 (v10):
    - Added minimum/maximum resolution specification when using Aspect Ratio Bucketing (min_bucket_reso/max_bucket_reso option).
    - Added extension specification for caption files (caption_extention).
    - Added support for images with .webp extension.
    - Added a function that allows captions to learning images and regularized images.
* 11/18 (v9):
    - Added support for Aspect Ratio Bucketing (enable_bucket option). (--enable_bucket)
    - Added support for selecting data format (fp16/bf16/float) when saving checkpoint (--save_precision)
    - Added support for saving learning state (--save_state, --resume)
    - Added support for logging (--logging_dir)
* 11/14 (diffusers_fine_tuning v2):
    - script name is now fine_tune.py.
    - Added option to learn Text Encoder --train_text_encoder.
    - The data format of checkpoint at the time of saving can be specified with the --save_precision option. You can choose float, fp16, and bf16.
    - Added a --save_state option to save the learning state (optimizer, etc.) in the middle. It can be resumed with the --resume option.
* 11/9 (v8): supports Diffusers 0.7.2. To upgrade diffusers run `pip install --upgrade diffusers[torch]`
* 11/7 (v7): Text Encoder supports checkpoint files in different storage formats (it is converted at the time of import, so export will be in normal format). Changed the average value of EPOCH loss to output to the screen. Added a function to save epoch and global step in checkpoint in SD format (add values if there is existing data). The reg_data_dir option is enabled during fine tuning (fine tuning while mixing regularized images). Added dataset_repeats option that is valid for fine tuning (specified when the number of teacher images is small and the epoch is extremely short).
