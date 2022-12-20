# This powershell script will create a model using the fine tuning dreambooth method. It will require landscape,
# portrait and square images.
#
# Adjust the script to your own needs

# Sylvia Ritter
# variable values
$pretrained_model_name_or_path = "D:\models\v1-5-pruned-mse-vae.ckpt"
$train_dir = "D:\dreambooth\train_sylvia_ritter\raw_data"

$landscape_image_num = 4
$portrait_image_num = 25
$square_image_num = 2

$learning_rate = 1e-6
$dataset_repeats = 120
$train_batch_size = 4
$epoch = 1
$save_every_n_epochs=1
$mixed_precision="fp16"
$num_cpu_threads_per_process=6

$landscape_folder_name = "landscape-pp"
$landscape_resolution = "832,512"
$portrait_folder_name = "portrait-pp"
$portrait_resolution = "448,896"
$square_folder_name = "square-pp"
$square_resolution = "512,512"

# You should not have to change values past this point

$landscape_data_dir = $train_dir + "\" + $landscape_folder_name
$portrait_data_dir = $train_dir + "\" + $portrait_folder_name
$square_data_dir = $train_dir + "\" + $square_folder_name
$landscape_output_dir = $train_dir + "\model-l"
$portrait_output_dir = $train_dir + "\model-lp"
$square_output_dir = $train_dir + "\model-lps"

$landscape_repeats = $landscape_image_num * $dataset_repeats
$portrait_repeats = $portrait_image_num * $dataset_repeats
$square_repeats = $square_image_num * $dataset_repeats

$landscape_mts = [Math]::Ceiling($landscape_repeats / $train_batch_size * $epoch)
$portrait_mts = [Math]::Ceiling($portrait_repeats / $train_batch_size * $epoch)
$square_mts = [Math]::Ceiling($square_repeats / $train_batch_size * $epoch)

# Write-Output $landscape_repeats

.\venv\Scripts\activate

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process train_db.py `
    --pretrained_model_name_or_path=$pretrained_model_name_or_path `
    --train_data_dir=$landscape_data_dir `
    --output_dir=$landscape_output_dir `
    --resolution=$landscape_resolution `
    --train_batch_size=$train_batch_size `
    --learning_rate=$learning_rate `
    --max_train_steps=$landscape_mts `
    --use_8bit_adam `
    --xformers `
    --mixed_precision=$mixed_precision `
    --cache_latents `
    --save_every_n_epochs=$save_every_n_epochs `
    --fine_tuning `
    --dataset_repeats=$dataset_repeats `
    --save_precision="fp16"
    
accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process train_db.py `
    --pretrained_model_name_or_path=$landscape_output_dir"\last.ckpt" `
    --train_data_dir=$portrait_data_dir `
    --output_dir=$portrait_output_dir `
    --resolution=$portrait_resolution `
    --train_batch_size=$train_batch_size `
    --learning_rate=$learning_rate `
    --max_train_steps=$portrait_mts `
    --use_8bit_adam `
    --xformers `
    --mixed_precision=$mixed_precision `
    --cache_latents `
    --save_every_n_epochs=$save_every_n_epochs `
    --fine_tuning `
    --dataset_repeats=$dataset_repeats `
    --save_precision="fp16"
    
accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process train_db.py `
    --pretrained_model_name_or_path=$portrait_output_dir"\last.ckpt" `
    --train_data_dir=$square_data_dir `
    --output_dir=$square_output_dir `
    --resolution=$square_resolution `
    --train_batch_size=$train_batch_size `
    --learning_rate=$learning_rate `
    --max_train_steps=$square_mts `
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
    --pretrained_model_name_or_path=$square_output_dir"\last.ckpt" `
    --train_data_dir=$landscape_data_dir `
    --output_dir=$landscape_output_dir"2" `
    --resolution=$landscape_resolution `
    --train_batch_size=$train_batch_size `
    --learning_rate=$learning_rate `
    --max_train_steps=$([Math]::Ceiling($landscape_mts/2)) `
    --use_8bit_adam `
    --xformers `
    --mixed_precision=$mixed_precision `
    --cache_latents `
    --save_every_n_epochs=$save_every_n_epochs `
    --fine_tuning `
    --dataset_repeats=$([Math]::Ceiling($dataset_repeats/2)) `
    --save_precision="fp16"
    
accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process train_db.py `
    --pretrained_model_name_or_path=$landscape_output_dir"2\last.ckpt" `
    --train_data_dir=$portrait_data_dir `
    --output_dir=$portrait_output_dir"2" `
    --resolution=$portrait_resolution `
    --train_batch_size=$train_batch_size `
    --learning_rate=$learning_rate `
    --max_train_steps=$([Math]::Ceiling($portrait_mts/2)) `
    --use_8bit_adam `
    --xformers `
    --mixed_precision=$mixed_precision `
    --cache_latents `
    --save_every_n_epochs=$save_every_n_epochs `
    --fine_tuning `
    --dataset_repeats=$([Math]::Ceiling($dataset_repeats/2)) `
    --save_precision="fp16"
    
accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process train_db.py `
    --pretrained_model_name_or_path=$portrait_output_dir"2\last.ckpt" `
    --train_data_dir=$square_data_dir `
    --output_dir=$square_output_dir"2" `
    --resolution=$square_resolution `
    --train_batch_size=$train_batch_size `
    --learning_rate=$learning_rate `
    --max_train_steps=$([Math]::Ceiling($square_mts/2)) `
    --use_8bit_adam `
    --xformers `
    --mixed_precision=$mixed_precision `
    --cache_latents `
    --save_every_n_epochs=$save_every_n_epochs `
    --fine_tuning `
    --dataset_repeats=$([Math]::Ceiling($dataset_repeats/2)) `
    --save_precision="fp16"
    