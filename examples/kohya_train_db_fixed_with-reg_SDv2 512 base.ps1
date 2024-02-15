# This powershell script will create a model using the fine tuning dreambooth method. It will require landscape,
# portrait and square images.
#
# Adjust the script to your own needs

# variable values
$pretrained_model_name_or_path = "D:\models\512-base-ema.ckpt"
$data_dir = "D:\models\dariusz_zawadzki\kohya_reg\data"
$reg_data_dir = "D:\models\dariusz_zawadzki\kohya_reg\reg"
$logging_dir = "D:\models\dariusz_zawadzki\logs"
$output_dir = "D:\models\dariusz_zawadzki\train_db_model_reg_v2"
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

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process train_db.py `
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

# Add the inference yaml file along with the model for proper loading. Need to have the same name as model... Most likely "last.yaml" in our case.
