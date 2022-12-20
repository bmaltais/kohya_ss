# This powershell script will create a model using the fine tuning dreambooth method. It will require landscape,
# portrait and square images.
#
# Adjust the script to your own needs

# Sylvia Ritter
# variable values
$pretrained_model_name_or_path = "D:\models\v1-5-pruned-mse-vae.ckpt"
$train_dir = "D:\dreambooth\train_bernard\v3"
$folder_name = "dataset"

$learning_rate = 1e-6
$dataset_repeats = 80
$train_batch_size = 6
$epoch = 1
$save_every_n_epochs=1
$mixed_precision="fp16"
$num_cpu_threads_per_process=6


# You should not have to change values past this point

$data_dir = $train_dir + "\" + $folder_name
$output_dir = $train_dir + "\model"

# stop script on error
$ErrorActionPreference = "Stop"

.\venv\Scripts\activate

$data_dir_buckets = $data_dir + "-buckets"

python .\diffusers_fine_tuning\create_buckets.py $data_dir $data_dir_buckets --max_resolution "768,512"

foreach($directory in Get-ChildItem -path $data_dir_buckets -Directory)

{
    if (Test-Path -Path $output_dir-$directory)
    {
        Write-Host "The folder $output_dir-$directory already exists, skipping bucket."
    }
    else
    {
        Write-Host $directory
        $dir_img_num = Get-ChildItem "$data_dir_buckets\$directory" -Recurse -File -Include *.jpg | Measure-Object | %{$_.Count}
        $repeats = $dir_img_num * $dataset_repeats
        $mts = [Math]::Ceiling($repeats / $train_batch_size * $epoch)

        Write-Host 

        accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process train_db_fixed-ber.py `
        --pretrained_model_name_or_path=$pretrained_model_name_or_path `
        --train_data_dir=$data_dir_buckets\$directory `
        --output_dir=$output_dir-$directory `
        --resolution=$directory `
        --train_batch_size=$train_batch_size `
        --learning_rate=$learning_rate `
        --max_train_steps=$mts `
        --use_8bit_adam `
        --xformers `
        --mixed_precision=$mixed_precision `
        --save_every_n_epochs=$save_every_n_epochs `
        --fine_tuning `
        --dataset_repeats=$dataset_repeats `
        --save_precision="fp16"
    }
    
    $pretrained_model_name_or_path = "$output_dir-$directory\last.ckpt"
}