@echo off

set VENV_DIR=.\venv
set PYTHON=python

call %VENV_DIR%\Scripts\activate.bat

accelerate launch --num_cpu_threads_per_process=2 "train_network.py" --enable_bucket --pretrained_model_name_or_path="D:\NovelAI\stable-diffusion-webui\models\Stable-diffusion\anime/animefull-final-pruned.safetensors" --train_data_dir="D:\NovelAI\additinal pt\Train\¥Ÿ¿Ã≈∏ƒÌ «Ô∏Æø¿Ω∫ v2/img" --resolution=768,768 --output_dir="D:\NovelAI\additinal pt\Train\¥Ÿ¿Ã≈∏ƒÌ «Ô∏Æø¿Ω∫ v2/model" --logging_dir="D:\NovelAI\additinal pt\Train\¥Ÿ¿Ã≈∏ƒÌ «Ô∏Æø¿Ω∫ v2/logs" --network_alpha="16" --training_comment="trigger word : daitaku helios \(umamusume\)" --save_model_as=safetensors --network_module=networks.lora --text_encoder_lr=0.5 --unet_lr=1.0 --network_dim=16 --output_name="daitaku_helios_lora" --lr_scheduler_num_cycles="10" --learning_rate="1.0" --lr_scheduler="constant_with_warmup" --lr_warmup_steps="156" --train_batch_size="4" --max_train_steps="1560" --save_every_n_epochs="1" --mixed_precision="fp16" --save_precision="fp16" --seed="1234" --caption_extension=".txt" --max_token_length=150 --bucket_reso_steps=64 --shuffle_caption --gradient_checkpointing --xformers --use_dadaptation --persistent_data_loader_workers --bucket_no_upscale --random_crop

pause