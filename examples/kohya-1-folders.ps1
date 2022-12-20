# This powershell script will create a model using the fine tuning dreambooth method. It will require landscape,
# portrait and square images.
#
# Adjust the script to your own needs

# Sylvia Ritter
# variable values
$pretrained_model_name_or_path = "D:\models\v1-5-pruned-mse-vae.ckpt"
$data_dir = "D:\test\squat"
$train_dir = "D:\test\"
$resolution = "512,512"

$image_num = Get-ChildItem $data_dir -Recurse -File -Include *.png | Measure-Object | %{$_.Count}

Write-Output "image_num: $image_num"

$learning_rate = 1e-6
$dataset_repeats = 40
$train_batch_size = 8
$epoch = 1
$save_every_n_epochs=1
$mixed_precision="fp16"
$num_cpu_threads_per_process=6

# You should not have to change values past this point

$output_dir = $train_dir + "\model"
$repeats = $image_num * $dataset_repeats
$mts = [Math]::Ceiling($repeats / $train_batch_size * $epoch)

Write-Output "Repeats: $repeats"

.\venv\Scripts\activate

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process train_db.py `
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
    --cache_latents `
    --save_every_n_epochs=$save_every_n_epochs `
    --fine_tuning `
    --dataset_repeats=$dataset_repeats `
    --save_precision="fp16"
    
# 2nd pass at half the dataset repeat value

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process train_db.py `
    --pretrained_model_name_or_path=$output_dir"\last.ckpt" `
    --train_data_dir=$data_dir `
    --output_dir=$output_dir"2" `
    --resolution=$resolution `
    --train_batch_size=$train_batch_size `
    --learning_rate=$learning_rate `
    --max_train_steps=$([Math]::Ceiling($mts/2)) `
    --use_8bit_adam `
    --xformers `
    --mixed_precision=$mixed_precision `
    --cache_latents `
    --save_every_n_epochs=$save_every_n_epochs `
    --fine_tuning `
    --dataset_repeats=$([Math]::Ceiling($dataset_repeats/2)) `
    --save_precision="fp16"
    
    accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process train_db.py `
    --pretrained_model_name_or_path=$output_dir"\last.ckpt" `
    --train_data_dir=$data_dir `
    --output_dir=$output_dir"2" `
    --resolution=$resolution `
    --train_batch_size=$train_batch_size `
    --learning_rate=$learning_rate `
    --max_train_steps=$mts `
    --use_8bit_adam `
    --xformers `
    --mixed_precision=$mixed_precision `
    --cache_latents `
    --save_every_n_epochs=$save_every_n_epochs `
    --fine_tuning `
    --dataset_repeats=$dataset_repeats `
    --save_precision="fp16"
    