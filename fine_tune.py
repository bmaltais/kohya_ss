# training with captions
# XXX dropped option: hypernetwork training

import argparse
import gc
import math
import os

from tqdm import tqdm
import torch
from accelerate.utils import set_seed
import diffusers
from diffusers import DDPMScheduler

import library.train_util as train_util


def collate_fn(examples):
  return examples[0]


def train(args):
  train_util.verify_training_args(args)
  train_util.prepare_dataset_args(args, True)

  cache_latents = args.cache_latents

  if args.seed is not None:
    set_seed(args.seed)                           # 乱数系列を初期化する

  tokenizer = train_util.load_tokenizer(args)

  train_dataset = train_util.FineTuningDataset(args.in_json, args.train_batch_size, args.train_data_dir,
                                               tokenizer, args.max_token_length, args.shuffle_caption, args.keep_tokens,
                                               args.resolution, args.enable_bucket, args.min_bucket_reso, args.max_bucket_reso,
                                               args.bucket_reso_steps, args.bucket_no_upscale,
                                               args.flip_aug, args.color_aug, args.face_crop_aug_range, args.random_crop,
                                               args.dataset_repeats, args.debug_dataset)

  # 学習データのdropout率を設定する
  train_dataset.set_caption_dropout(args.caption_dropout_rate, args.caption_dropout_every_n_epochs, args.caption_tag_dropout_rate)

  train_dataset.make_buckets()

  if args.debug_dataset:
    train_util.debug_dataset(train_dataset)
    return
  if len(train_dataset) == 0:
    print("No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。")
    return

  # acceleratorを準備する
  print("prepare accelerator")
  accelerator, unwrap_model = train_util.prepare_accelerator(args)

  # mixed precisionに対応した型を用意しておき適宜castする
  weight_dtype, save_dtype = train_util.prepare_dtype(args)

  # モデルを読み込む
  text_encoder, vae, unet, load_stable_diffusion_format = train_util.load_target_model(args, weight_dtype)

  # verify load/save model formats
  if load_stable_diffusion_format:
    src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
    src_diffusers_model_path = None
  else:
    src_stable_diffusion_ckpt = None
    src_diffusers_model_path = args.pretrained_model_name_or_path

  if args.save_model_as is None:
    save_stable_diffusion_format = load_stable_diffusion_format
    use_safetensors = args.use_safetensors
  else:
    save_stable_diffusion_format = args.save_model_as.lower() == 'ckpt' or args.save_model_as.lower() == 'safetensors'
    use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())

  # Diffusers版のxformers使用フラグを設定する関数
  def set_diffusers_xformers_flag(model, valid):
    #   model.set_use_memory_efficient_attention_xformers(valid)            # 次のリリースでなくなりそう
    # pipeが自動で再帰的にset_use_memory_efficient_attention_xformersを探すんだって(;´Д｀)
    # U-Netだけ使う時にはどうすればいいのか……仕方ないからコピって使うか
    # 0.10.2でなんか巻き戻って個別に指定するようになった(;^ω^)

    # Recursively walk through all the children.
    # Any children which exposes the set_use_memory_efficient_attention_xformers method
    # gets the message
    def fn_recursive_set_mem_eff(module: torch.nn.Module):
      if hasattr(module, "set_use_memory_efficient_attention_xformers"):
        module.set_use_memory_efficient_attention_xformers(valid)

      for child in module.children():
        fn_recursive_set_mem_eff(child)

    fn_recursive_set_mem_eff(model)

  # モデルに xformers とか memory efficient attention を組み込む
  if args.diffusers_xformers:
    print("Use xformers by Diffusers")
    set_diffusers_xformers_flag(unet, True)
  else:
    # Windows版のxformersはfloatで学習できないのでxformersを使わない設定も可能にしておく必要がある
    print("Disable Diffusers' xformers")
    set_diffusers_xformers_flag(unet, False)
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers)

  # 学習を準備する
  if cache_latents:
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.eval()
    with torch.no_grad():
      train_dataset.cache_latents(vae)
    vae.to("cpu")
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    gc.collect()

  # 学習を準備する：モデルを適切な状態にする
  training_models = []
  if args.gradient_checkpointing:
    unet.enable_gradient_checkpointing()
  training_models.append(unet)

  if args.train_text_encoder:
    print("enable text encoder training")
    if args.gradient_checkpointing:
      text_encoder.gradient_checkpointing_enable()
    training_models.append(text_encoder)
  else:
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)             # text encoderは学習しない
    if args.gradient_checkpointing:
      text_encoder.gradient_checkpointing_enable()
      text_encoder.train()                # required for gradient_checkpointing
    else:
      text_encoder.eval()

  if not cache_latents:
    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device, dtype=weight_dtype)

  for m in training_models:
    m.requires_grad_(True)
  params = []
  for m in training_models:
    params.extend(m.parameters())
  params_to_optimize = params

  # 学習に必要なクラスを準備する
  print("prepare optimizer, data loader etc.")
  _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

  # dataloaderを準備する
  # DataLoaderのプロセス数：0はメインプロセスになる
  n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)      # cpu_count-1 ただし最大で指定された数まで
  train_dataloader = torch.utils.data.DataLoader(
      train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=n_workers, persistent_workers=args.persistent_data_loader_workers)

  # 学習ステップ数を計算する
  if args.max_train_epochs is not None:
    args.max_train_steps = args.max_train_epochs * len(train_dataloader)
    print(f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

  # lr schedulerを用意する
  lr_scheduler = train_util.get_scheduler_fix(args.lr_scheduler, optimizer, num_warmup_steps=args.lr_warmup_steps,
                                              num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
                                              num_cycles=args.lr_scheduler_num_cycles, power=args.lr_scheduler_power)

  # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
  if args.full_fp16:
    assert args.mixed_precision == "fp16", "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
    print("enable full fp16 training.")
    unet.to(weight_dtype)
    text_encoder.to(weight_dtype)

  # acceleratorがなんかよろしくやってくれるらしい
  if args.train_text_encoder:
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
  else:
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

  # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
  if args.full_fp16:
    train_util.patch_accelerator_for_fp16_training(accelerator)

  # resumeする
  if args.resume is not None:
    print(f"resume training from state: {args.resume}")
    accelerator.load_state(args.resume)

  # epoch数を計算する
  num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
  num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
  if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
    args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

  # 学習する
  total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
  print("running training / 学習開始")
  print(f"  num examples / サンプル数: {train_dataset.num_train_images}")
  print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
  print(f"  num epochs / epoch数: {num_train_epochs}")
  print(f"  batch size per device / バッチサイズ: {args.train_batch_size}")
  print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
  print(f"  gradient ccumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
  print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

  progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
  global_step = 0

  noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                  num_train_timesteps=1000, clip_sample=False)

  if accelerator.is_main_process:
    accelerator.init_trackers("finetuning")

  for epoch in range(num_train_epochs):
    print(f"epoch {epoch+1}/{num_train_epochs}")
    train_dataset.set_current_epoch(epoch + 1)

    for m in training_models:
      m.train()

    loss_total = 0
    for step, batch in enumerate(train_dataloader):
      with accelerator.accumulate(training_models[0]):  # 複数モデルに対応していない模様だがとりあえずこうしておく
        with torch.no_grad():
          if "latents" in batch and batch["latents"] is not None:
            latents = batch["latents"].to(accelerator.device)
          else:
            # latentに変換
            latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
          latents = latents * 0.18215
        b_size = latents.shape[0]

        with torch.set_grad_enabled(args.train_text_encoder):
          # Get the text embedding for conditioning
          input_ids = batch["input_ids"].to(accelerator.device)
          encoder_hidden_states = train_util.get_hidden_states(
              args, input_ids, tokenizer, text_encoder, None if not args.full_fp16 else weight_dtype)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, device=latents.device)
        if args.noise_offset:
          # https://www.crosslabs.org//blog/diffusion-with-offset-noise
          noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if args.v_parameterization:
          # v-parameterization training
          target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
          target = noise

        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        accelerator.backward(loss)
        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
          params_to_clip = []
          for m in training_models:
            params_to_clip.extend(m.parameters())
          accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

      # Checks if the accelerator has performed an optimization step behind the scenes
      if accelerator.sync_gradients:
        progress_bar.update(1)
        global_step += 1

      current_loss = loss.detach().item()        # 平均なのでbatch sizeは関係ないはず
      if args.logging_dir is not None:
        logs = {"loss": current_loss, "lr": float(lr_scheduler.get_last_lr()[0])}
        if args.optimizer_type.lower() == "DAdaptation".lower():  # tracking d*lr value
          logs["lr/d*lr"] = lr_scheduler.optimizers[0].param_groups[0]['d']*lr_scheduler.optimizers[0].param_groups[0]['lr']
        accelerator.log(logs, step=global_step)

      # TODO moving averageにする
      loss_total += current_loss
      avr_loss = loss_total / (step+1)
      logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
      progress_bar.set_postfix(**logs)

      if global_step >= args.max_train_steps:
        break

    if args.logging_dir is not None:
      logs = {"loss/epoch": loss_total / len(train_dataloader)}
      accelerator.log(logs, step=epoch+1)

    accelerator.wait_for_everyone()

    if args.save_every_n_epochs is not None:
      src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
      train_util.save_sd_model_on_epoch_end(args, accelerator, src_path, save_stable_diffusion_format, use_safetensors,
                                            save_dtype, epoch, num_train_epochs, global_step,  unwrap_model(text_encoder), unwrap_model(unet), vae)

  is_main_process = accelerator.is_main_process
  if is_main_process:
    unet = unwrap_model(unet)
    text_encoder = unwrap_model(text_encoder)

  accelerator.end_training()

  if args.save_state:
    train_util.save_state_on_train_end(args, accelerator)

  del accelerator                         # この後メモリを使うのでこれは消す

  if is_main_process:
    src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
    train_util.save_sd_model_on_train_end(args, src_path, save_stable_diffusion_format, use_safetensors,
                                          save_dtype, epoch, global_step,  text_encoder, unet, vae)
    print("model saved.")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  train_util.add_sd_models_arguments(parser)
  train_util.add_dataset_arguments(parser, False, True, True)
  train_util.add_training_arguments(parser, False)
  train_util.add_sd_saving_arguments(parser)
  train_util.add_optimizer_arguments(parser)

  parser.add_argument("--diffusers_xformers", action='store_true',
                      help='use xformers by diffusers / Diffusersでxformersを使用する')
  parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")

  args = parser.parse_args()
  train(args)
