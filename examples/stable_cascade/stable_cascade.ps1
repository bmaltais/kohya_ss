& accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 2 stable_cascade_train_stage_c.py `
  --mixed_precision bf16 --save_precision bf16 --persistent_data_loader_workers `
  --gradient_checkpointing --learning_rate 1e-4 `
  --optimizer_type adafactor --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False" `
  --max_train_epochs 10 --save_every_n_epochs 1 --save_precision bf16 `
  --output_dir "e:\model\test" --output_name "sc_test" `
  --stage_c_checkpoint_path "E:\models\stable_cascade\stage_c_bf16.safetensors" `
  --effnet_checkpoint_path "E:\models\stable_cascade\effnet_encoder.safetensors" `
  --previewer_checkpoint_path "E:\models\stable_cascade\previewer.safetensors" `
  --dataset_config "D:\kohya_ss\examples\stable_cascade\test_dataset.toml" `
  --sample_every_n_epochs 1 --sample_prompts "D:\kohya_ss\examples\stable_cascade\prompt.txt" `
  --adaptive_loss_weight --max_data_loader_n_workers 0

  & accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 2 stable_cascade_train_stage_c.py `
  --mixed_precision bf16 --save_precision bf16 --persistent_data_loader_workers `
  --gradient_checkpointing --learning_rate 1e-4 `
  --optimizer_type adafactor --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False" `
  --max_train_epochs 10 --save_every_n_epochs 1 --save_precision bf16 `
  --output_dir "e:\model\test2" --output_name "sc_test2" `
  --stage_c_checkpoint_path "E:\models\stable_cascade\stage_c_bf16.safetensors" `
  --effnet_checkpoint_path "E:\models\stable_cascade\effnet_encoder.safetensors" `
  --previewer_checkpoint_path "E:\models\stable_cascade\previewer.safetensors" `
  --dataset_config "D:\kohya_ss\examples\stable_cascade\test_dataset2.toml" `
  --sample_every_n_epochs 1 --sample_prompts "D:\kohya_ss\examples\stable_cascade\prompt2.txt" `
  --adaptive_loss_weight --max_data_loader_n_workers 0

  accelerate launch --num_cpu_threads_per_process=2 --mixed_precision bf16  "./stable_cascade_train_stage_c.py" `
  --stage_c_checkpoint_path "E:\models\stable_cascade\stage_c_bf16.safetensors" --effnet_checkpoint_path `
  "E:\models\stable_cascade\effnet_encoder.safetensors" --previewer_checkpoint_path "E:\models\stable_cascade\previewer.safetensors"   `
  --dataset_config "D:\kohya_ss\examples\stable_cascade\test_dataset2.toml" --adaptive_loss_weight --bucket_no_upscale `
  --bucket_reso_steps=32 --dataset_repeats="1" --gradient_checkpointing --learning_rate="0.0001" --lr_scheduler="constant" `
  --max_data_loader_n_workers="0" --max_train_epochs=10 --mixed_precision="bf16" --optimizer_args "scale_parameter=False" `
  "relative_step=False" "warmup_init=False" --optimizer_type="Adafactor" --output_dir="E:/model/grandizer_sc" `
  --output_name="grandizer_sc" --save_every_n_epochs="1" --save_precision="bf16" --seed="1234" --train_batch_size="4" `
  --sample_sampler=euler --sample_prompts="E:/model/grandizer_sc\sample\prompt2.txt" --sample_every_n_epochs="1"