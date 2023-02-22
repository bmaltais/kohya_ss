# DreamBooth training
# XXX dropped option: fine_tune

import gc
import time
import argparse
import itertools
import math
import os

from tqdm import tqdm
import torch
from accelerate.utils import set_seed
import diffusers
from diffusers import DDPMScheduler

import library.train_util as train_util
from library.train_util import DreamBoothDataset


def collate_fn(examples):
  return examples[0]


def train(args):
  train_util.verify_training_args(args)
  train_util.prepare_dataset_args(args, False)

  cache_latents = args.cache_latents

  if args.seed is not None:
    set_seed(args.seed)                           # 乱数系列を初期化する

  tokenizer = train_util.load_tokenizer(args)

  train_dataset = DreamBoothDataset(args.train_batch_size, args.train_data_dir, args.reg_data_dir,
                                    tokenizer, args.max_token_length, args.caption_extension, args.shuffle_caption, args.keep_tokens,
                                    args.resolution, args.enable_bucket, args.min_bucket_reso, args.max_bucket_reso,
                                    args.bucket_reso_steps, args.bucket_no_upscale,
                                    args.prior_loss_weight, args.flip_aug, args.color_aug, args.face_crop_aug_range, args.random_crop, args.debug_dataset)

  if args.no_token_padding:
    train_dataset.disable_token_padding()

  # 学習データのdropout率を設定する
  train_dataset.set_caption_dropout(args.caption_dropout_rate, args.caption_dropout_every_n_epochs, args.caption_tag_dropout_rate)

  train_dataset.make_buckets()

  if args.debug_dataset:
    train_util.debug_dataset(train_dataset)
    return

  # acceleratorを準備する
  print("prepare accelerator")

  if args.gradient_accumulation_steps > 1:
    print(f"gradient_accumulation_steps is {args.gradient_accumulation_steps}. accelerate does not support gradient_accumulation_steps when training multiple models (U-Net and Text Encoder), so something might be wrong")
    print(
        f"gradient_accumulation_stepsが{args.gradient_accumulation_steps}に設定されています。accelerateは複数モデル（U-NetおよびText Encoder）の学習時にgradient_accumulation_stepsをサポートしていないため結果は未知数です")

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

  # モデルに xformers とか memory efficient attention を組み込む
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
  train_text_encoder = args.stop_text_encoder_training is None or args.stop_text_encoder_training >= 0
  unet.requires_grad_(True)                   # 念のため追加
  text_encoder.requires_grad_(train_text_encoder)
  if not train_text_encoder:
    print("Text Encoder is not trained.")

  if args.gradient_checkpointing:
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()

  if not cache_latents:
    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device, dtype=weight_dtype)

  # 学習に必要なクラスを準備する
  print("prepare optimizer, data loader etc.")
  if train_text_encoder:
    trainable_params = (itertools.chain(unet.parameters(), text_encoder.parameters()))
  else:
    trainable_params = unet.parameters()

  _, _, optimizer = train_util.get_optimizer(args, trainable_params)

  # dataloaderを準備する
  # DataLoaderのプロセス数：0はメインプロセスになる
  n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)      # cpu_count-1 ただし最大で指定された数まで
  train_dataloader = torch.utils.data.DataLoader(
      train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=n_workers, persistent_workers=args.persistent_data_loader_workers)

  # 学習ステップ数を計算する
  if args.max_train_epochs is not None:
    args.max_train_steps = args.max_train_epochs * len(train_dataloader)
    print(f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

  if args.stop_text_encoder_training is None:
    args.stop_text_encoder_training = args.max_train_steps + 1                # do not stop until end

  # lr schedulerを用意する TODO gradient_accumulation_stepsの扱いが何かおかしいかもしれない。後で確認する
  lr_scheduler = train_util.get_scheduler_fix(args.lr_scheduler, optimizer, num_warmup_steps=args.lr_warmup_steps,
                                              num_training_steps=args.max_train_steps,
                                              num_cycles=args.lr_scheduler_num_cycles, power=args.lr_scheduler_power)

  # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
  if args.full_fp16:
    assert args.mixed_precision == "fp16", "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
    print("enable full fp16 training.")
    unet.to(weight_dtype)
    text_encoder.to(weight_dtype)

  # acceleratorがなんかよろしくやってくれるらしい
  if train_text_encoder:
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
  else:
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

  if not train_text_encoder:
    text_encoder.to(accelerator.device, dtype=weight_dtype)             # to avoid 'cpu' vs 'cuda' error

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
  print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset.num_train_images}")
  print(f"  num reg images / 正則化画像の数: {train_dataset.num_reg_images}")
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
    accelerator.init_trackers("dreambooth")

  loss_list = []
  loss_total = 0.0
  for epoch in range(num_train_epochs):
    print(f"epoch {epoch+1}/{num_train_epochs}")
    train_dataset.set_current_epoch(epoch + 1)

    # 指定したステップ数までText Encoderを学習する：epoch最初の状態
    unet.train()
    # train==True is required to enable gradient_checkpointing
    if args.gradient_checkpointing or global_step < args.stop_text_encoder_training:
      text_encoder.train()

    for step, batch in enumerate(train_dataloader):
      # 指定したステップ数でText Encoderの学習を止める
      if global_step == args.stop_text_encoder_training:
        print(f"stop text encoder training at step {global_step}")
        if not args.gradient_checkpointing:
          text_encoder.train(False)
        text_encoder.requires_grad_(False)

      with accelerator.accumulate(unet):
        with torch.no_grad():
          # latentに変換
          if cache_latents:
            latents = batch["latents"].to(accelerator.device)
          else:
            latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
          latents = latents * 0.18215
        b_size = latents.shape[0]

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, device=latents.device)
        if args.noise_offset:
          # https://www.crosslabs.org//blog/diffusion-with-offset-noise
          noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)

        # Get the text embedding for conditioning
        with torch.set_grad_enabled(global_step < args.stop_text_encoder_training):
          input_ids = batch["input_ids"].to(accelerator.device)
          encoder_hidden_states = train_util.get_hidden_states(
              args, input_ids, tokenizer, text_encoder, None if not args.full_fp16 else weight_dtype)

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

        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
        loss = loss.mean([1, 2, 3])

        loss_weights = batch["loss_weights"]                      # 各sampleごとのweight
        loss = loss * loss_weights

        loss = loss.mean()                # 平均なのでbatch_sizeで割る必要なし

        accelerator.backward(loss)
        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
          if train_text_encoder:
            params_to_clip = (itertools.chain(unet.parameters(), text_encoder.parameters()))
          else:
            params_to_clip = unet.parameters()
          accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

      # Checks if the accelerator has performed an optimization step behind the scenes
      if accelerator.sync_gradients:
        progress_bar.update(1)
        global_step += 1

      current_loss = loss.detach().item()
      if args.logging_dir is not None:
        logs = {"loss": current_loss, "lr": float(lr_scheduler.get_last_lr()[0])}
        if args.optimizer_type.lower() == "DAdaptation".lower():  # tracking d*lr value
          logs["lr/d*lr"] = lr_scheduler.optimizers[0].param_groups[0]['d']*lr_scheduler.optimizers[0].param_groups[0]['lr']
        accelerator.log(logs, step=global_step)

      if epoch == 0:
        loss_list.append(current_loss)
      else:
        loss_total -= loss_list[step]
        loss_list[step] = current_loss
      loss_total += current_loss
      avr_loss = loss_total / len(loss_list)
      logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
      progress_bar.set_postfix(**logs)

      if global_step >= args.max_train_steps:
        break

    if args.logging_dir is not None:
      logs = {"loss/epoch": loss_total / len(loss_list)}
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
  train_util.add_dataset_arguments(parser, True, False, True)
  train_util.add_training_arguments(parser, True)
  train_util.add_sd_saving_arguments(parser)
  train_util.add_optimizer_arguments(parser)

  parser.add_argument("--no_token_padding", action="store_true",
                      help="disable token padding (same as Diffuser's DreamBooth) / トークンのpaddingを無効にする（Diffusers版DreamBoothと同じ動作）")
  parser.add_argument("--stop_text_encoder_training", type=int, default=None,
                      help="steps to stop text encoder training, -1 for no training / Text Encoderの学習を止めるステップ数、-1で最初から学習しない")

  args = parser.parse_args()
  train(args)
