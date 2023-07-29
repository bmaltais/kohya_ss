# Command 1: merge_captions_to_metadata.py
$captionExtension = "--caption_extension=.txt"
$sourceDir1 = "d:\test\1_1960-1969"
$targetFile1 = "d:\test\1_1960-1969/meta_cap.json"

# Command 2: prepare_buckets_latents.py
$targetLatentFile = "d:\test\1_1960-1969/meta_lat.json"
$modelFile = "E:\models\sdxl\sd_xl_base_0.9.safetensors"

./venv/Scripts/python.exe finetune/merge_captions_to_metadata.py $captionExtension $sourceDir1 $targetFile1 --full_path
./venv/Scripts/python.exe finetune/prepare_buckets_latents.py $sourceDir1 $targetFile1 $targetLatentFile $modelFile --batch_size=4 --max_resolution=1024,1024 --min_bucket_reso=64 --max_bucket_reso=2048 --mixed_precision=bf16 --full_path
