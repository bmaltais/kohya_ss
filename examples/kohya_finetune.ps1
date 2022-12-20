# variables related to the pretrained model
$pretrained_model_name_or_path = "D:\models\test\samdoesart2\model\last"
$v2 = 1 # set to 1 for true or 0 for false
$v_model = 0 # set to 1 for true or 0 for false

# variables related to the training dataset and output directory
$train_dir = "D:\models\test\samdoesart2"
$image_folder = "D:\dataset\samdoesart2\raw"
$output_dir = "D:\models\test\samdoesart2\model_e2\"
$max_resolution = "512,512"

# variables related to the training process
$learning_rate = 1e-6
$lr_scheduler = "constant" # Default is constant
$lr_warmup = 0 # % of steps to warmup for 0 - 100. Default is 0.
$dataset_repeats = 40
$train_batch_size = 8
$epoch = 1
$save_every_n_epochs = 1
$mixed_precision = "bf16"
$save_precision = "fp16" # use fp16 for better compatibility with auto1111 and other repo
$seed = "494481440"
$num_cpu_threads_per_process = 6
$train_text_encoder = 0 # set to 1 to train text encoder otherwise set to 0

# variables related to the resulting diffuser model. If input is ckpt or tensors then it is not applicable
$convert_to_safetensors = 1 # set to 1 to convert resulting diffuser to ckpt
$convert_to_ckpt = 1 # set to 1 to convert resulting diffuser to ckpt

# other variables
$kohya_finetune_repo_path = "D:\kohya_ss"

### You should not need to change things below

# Set variables to useful values using ternary operator
$v_model = ($v_model -eq 0) ? $null : "--v_parameterization"
$v2 = ($v2 -eq 0) ? $null : "--v2"
$train_text_encoder = ($train_text_encoder -eq 0) ? $null : "--train_text_encoder"

# stop script on error
$ErrorActionPreference = "Stop"

# define a list of substrings to search for
$substrings_v2 = "stable-diffusion-2-1-base", "stable-diffusion-2-base"

# check if $v2 and $v_model are empty and if $pretrained_model_name_or_path contains any of the substrings in the v2 list
if ($v2 -eq $null -and $v_model -eq $null -and ($substrings_v2 | Where-Object { $pretrained_model_name_or_path -match $_ }).Count -gt 0) {
    Write-Host("SD v2 model detected. Setting --v2 parameter")
    $v2 = "--v2"
    $v_model = $null
}

# define a list of substrings to search for v-objective
$substrings_v_model = "stable-diffusion-2-1", "stable-diffusion-2"

# check if $v2 and $v_model are empty and if $pretrained_model_name_or_path contains any of the substrings in the v_model list
elseif ($v2 -eq $null -and $v_model -eq $null -and ($substrings_v_model | Where-Object { $pretrained_model_name_or_path -match $_ }).Count -gt 0) {
    Write-Host("SD v2 v_model detected. Setting --v2 parameter and --v_parameterization")
    $v2 = "--v2"
    $v_model = "--v_parameterization"
}

# activate venv
cd $kohya_finetune_repo_path
.\venv\Scripts\activate

# create caption json file
if (!(Test-Path -Path $train_dir)) {
    New-Item -Path $train_dir -ItemType "directory"
}

python $kohya_finetune_repo_path\script\merge_captions_to_metadata.py `
    --caption_extention ".txt" $image_folder $train_dir"\meta_cap.json"

# create images buckets
python $kohya_finetune_repo_path\script\prepare_buckets_latents.py `
    $image_folder `
    $train_dir"\meta_cap.json" `
    $train_dir"\meta_lat.json" `
    $pretrained_model_name_or_path `
    --batch_size 4 --max_resolution $max_resolution --mixed_precision $mixed_precision

# Get number of valid images
$image_num = Get-ChildItem "$image_folder" -Recurse -File -Include *.npz | Measure-Object | % { $_.Count }

$repeats = $image_num * $dataset_repeats
Write-Host("Repeats = $repeats")

# calculate max_train_set
$max_train_set = [Math]::Ceiling($repeats / $train_batch_size * $epoch)
Write-Host("max_train_set = $max_train_set")

$lr_warmup_steps = [Math]::Round($lr_warmup * $max_train_set / 100)
Write-Host("lr_warmup_steps = $lr_warmup_steps")

Write-Host("$v2 $v_model")

accelerate launch --num_cpu_threads_per_process $num_cpu_threads_per_process $kohya_finetune_repo_path\script\fine_tune.py `
    $v2 `
    $v_model `
    --pretrained_model_name_or_path=$pretrained_model_name_or_path `
    --in_json $train_dir\meta_lat.json `
    --train_data_dir="$image_folder" `
    --output_dir=$output_dir `
    --train_batch_size=$train_batch_size `
    --dataset_repeats=$dataset_repeats `
    --learning_rate=$learning_rate `
    --lr_scheduler=$lr_scheduler `
    --lr_warmup_steps=$lr_warmup_steps `
    --max_train_steps=$max_train_set `
    --use_8bit_adam `
    --xformers `
    --mixed_precision=$mixed_precision `
    --save_every_n_epochs=$save_every_n_epochs `
    --seed=$seed `
    $train_text_encoder `
    --save_precision=$save_precision

# check if $output_dir\last is a directory... therefore it is a diffuser model
if (Test-Path "$output_dir\last" -PathType Container) {
    if ($convert_to_ckpt) {
        Write-Host("Converting diffuser model $output_dir\last to $output_dir\last.ckpt")
        python "$kohya_finetune_repo_path\tools\convert_diffusers20_original_sd.py" `
            $output_dir\last `
            $output_dir\last.ckpt `
            --$save_precision
    }
    if ($convert_to_safetensors) {
        Write-Host("Converting diffuser model $output_dir\last to $output_dir\last.safetensors")
        python "$kohya_finetune_repo_path\tools\convert_diffusers20_original_sd.py" `
            $output_dir\last `
            $output_dir\last.safetensors `
            --$save_precision
    }
}

# define a list of substrings to search for inference file
$substrings_sd_model = ".ckpt", ".safetensors"
$matching_extension = foreach ($ext in $substrings_sd_model) {
    Get-ChildItem $output_dir -File | Where-Object { $_.Extension -contains $ext }
}

if ($matching_extension.Count -gt 0) {
    # copy the file named "v2-inference.yaml" from the "v2_inference" folder to $output_dir as last.yaml
    if ( $v2 -ne $null -and $v_model -ne $null) {
        Write-Host("Saving v2-inference-v.yaml as $output_dir\last.yaml")
        Copy-Item -Path "$kohya_finetune_repo_path\v2_inference\v2-inference-v.yaml" -Destination "$output_dir\last.yaml"
    }
    elseif ( $v2 -ne $null ) {
        Write-Host("Saving v2-inference.yaml as $output_dir\last.yaml")
        Copy-Item -Path "$kohya_finetune_repo_path\v2_inference\v2-inference.yaml" -Destination "$output_dir\last.yaml"
    }
}