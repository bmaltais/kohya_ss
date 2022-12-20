# Sylvia Ritter. AKA: by silvery trait

# variable values
$pretrained_model_name_or_path = "D:\models\v1-5-pruned-mse-vae.ckpt"
$train_dir = "D:\dreambooth\train_sylvia_ritter\raw_data"
$training_folder = "all-images-v3"

$learning_rate = 5e-6
$dataset_repeats = 40
$train_batch_size = 6
$epoch = 4
$save_every_n_epochs=1
$mixed_precision="bf16"
$num_cpu_threads_per_process=6

$max_resolution = "768,576"

# You should not have to change values past this point

# stop script on error
$ErrorActionPreference = "Stop"

# activate venv
.\venv\Scripts\activate

# create caption json file
python D:\kohya_ss\finetune\merge_captions_to_metadata.py `
--caption_extention ".txt" $train_dir"\"$training_folder $train_dir"\meta_cap.json"

# create images buckets
python D:\kohya_ss\finetune\prepare_buckets_latents.py `
    $train_dir"\"$training_folder `
    $train_dir"\meta_cap.json" `
    $train_dir"\meta_lat.json" `
    $pretrained_model_name_or_path `
    --batch_size 4 --max_resolution $max_resolution --mixed_precision fp16

# Get number of valid images
$image_num = Get-ChildItem "$train_dir\$training_folder" -Recurse -File -Include *.npz | Measure-Object | %{$_.Count}
$repeats = $image_num * $dataset_repeats

# calculate max_train_set
$max_train_set = [Math]::Ceiling($repeats / $train_batch_size * $epoch)


accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process D:\kohya_ss\finetune\fine_tune.py `
    --pretrained_model_name_or_path=$pretrained_model_name_or_path `
    --in_json $train_dir"\meta_lat.json" `
    --train_data_dir=$train_dir"\"$training_folder `
    --output_dir=$train_dir"\fine_tuned2" `
    --train_batch_size=$train_batch_size `
    --dataset_repeats=$dataset_repeats `
    --learning_rate=$learning_rate `
    --max_train_steps=$max_train_set `
    --use_8bit_adam --xformers `
    --mixed_precision=$mixed_precision `
    --save_every_n_epochs=$save_every_n_epochs `
    --train_text_encoder `
    --save_precision="fp16"

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process D:\kohya_ss\finetune\fine_tune.py `
    --pretrained_model_name_or_path=$train_dir"\fine_tuned\last.ckpt" `
    --in_json $train_dir"\meta_lat.json" `
    --train_data_dir=$train_dir"\"$training_folder `
    --output_dir=$train_dir"\fine_tuned2" `
    --train_batch_size=$train_batch_size `
    --dataset_repeats=$([Math]::Ceiling($dataset_repeats / 2)) `
    --learning_rate=$learning_rate `
    --max_train_steps=$([Math]::Ceiling($max_train_set / 2)) `
    --use_8bit_adam --xformers `
    --mixed_precision=$mixed_precision `
    --save_every_n_epochs=$save_every_n_epochs `
    --save_precision="fp16"

# Hypernetwork

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process D:\kohya_ss\finetune\fine_tune.py `
    --pretrained_model_name_or_path=$pretrained_model_name_or_path `
    --in_json $train_dir"\meta_lat.json" `
    --train_data_dir=$train_dir"\"$training_folder `
    --output_dir=$train_dir"\fine_tuned" `
    --train_batch_size=$train_batch_size `
    --dataset_repeats=$dataset_repeats `
    --learning_rate=$learning_rate `
    --max_train_steps=$max_train_set `
    --use_8bit_adam --xformers `
    --mixed_precision=$mixed_precision `
    --save_every_n_epochs=$save_every_n_epochs `
    --save_precision="fp16" `
    --hypernetwork_module="hypernetwork_nai"