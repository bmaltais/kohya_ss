from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import importlib
import argparse
import gc
import math
import os
import random
import time
import json

from tqdm import tqdm
import torch
from accelerate.utils import set_seed
import diffusers
from diffusers import DDPMScheduler

import library.train_util as train_util
from library.train_util import DreamBoothDataset, FineTuningDataset


def collate_fn(examples):
  return examples[0]


# TODO 他のスクリプトと共通化する
def generate_step_logs(args: argparse.Namespace, current_loss, avr_loss, lr_scheduler):
  logs = {"loss/current": current_loss, "loss/average": avr_loss}

  if args.network_train_unet_only:
    logs["lr/unet"] = float(lr_scheduler.get_last_lr()[0])
  elif args.network_train_text_encoder_only:
    logs["lr/textencoder"] = float(lr_scheduler.get_last_lr()[0])
  else:
    logs["lr/textencoder"] = float(lr_scheduler.get_last_lr()[0])
    logs["lr/unet"] = float(lr_scheduler.get_last_lr()[-1])          # may be same to textencoder

  if args.optimizer_type.lower() == "DAdaptation".lower():  # tracking d*lr value of unet.
    logs["lr/d*lr"] = lr_scheduler.optimizers[-1].param_groups[0]['d']*lr_scheduler.optimizers[-1].param_groups[0]['lr']

  return logs


def train(args):
  session_id = random.randint(0, 2**32)
  training_started_at = time.time()
  train_util.verify_training_args(args)
  train_util.prepare_dataset_args(args, True)

  cache_latents = args.cache_latents
  use_dreambooth_method = args.in_json is None

  if args.seed is not None:
    set_seed(args.seed)

  tokenizer = train_util.load_tokenizer(args)

  # データセットを準備する
  if use_dreambooth_method:
    print("Use DreamBooth method.")
    train_dataset = DreamBoothDataset(args.train_batch_size, args.train_data_dir, args.reg_data_dir,
                                      tokenizer, args.max_token_length, args.caption_extension, args.shuffle_caption, args.keep_tokens,
                                      args.resolution, args.enable_bucket, args.min_bucket_reso, args.max_bucket_reso,
                                      args.bucket_reso_steps, args.bucket_no_upscale,
                                      args.prior_loss_weight, args.flip_aug, args.color_aug, args.face_crop_aug_range,
                                      args.random_crop, args.debug_dataset)
  else:
    print("Train with captions.")
    train_dataset = FineTuningDataset(args.in_json, args.train_batch_size, args.train_data_dir,
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
    print("No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）")
    return

  # acceleratorを準備する
  print("prepare accelerator")
  accelerator, unwrap_model = train_util.prepare_accelerator(args)

  # mixed precisionに対応した型を用意しておき適宜castする
  weight_dtype, save_dtype = train_util.prepare_dtype(args)

  # モデルを読み込む
  text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype)

  # work on low-ram device
  if args.lowram:
    text_encoder.to("cuda")
    unet.to("cuda")

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

  # prepare network
  print("import network module:", args.network_module)
  network_module = importlib.import_module(args.network_module)

  net_kwargs = {}
  if args.network_args is not None:
    for net_arg in args.network_args:
      key, value = net_arg.split('=')
      net_kwargs[key] = value

  # if a new network is added in future, add if ~ then blocks for each network (;'∀')
  network = network_module.create_network(1.0, args.network_dim, args.network_alpha, vae, text_encoder, unet, **net_kwargs)
  if network is None:
    return

  if args.network_weights is not None:
    print("load network weights from:", args.network_weights)
    network.load_weights(args.network_weights)

  train_unet = not args.network_train_text_encoder_only
  train_text_encoder = not args.network_train_unet_only
  network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

  if args.gradient_checkpointing:
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()
    network.enable_gradient_checkpointing()                   # may have no effect

  # 学習に必要なクラスを準備する
  print("prepare optimizer, data loader etc.")

  trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)
  optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

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
    network.to(weight_dtype)

  # acceleratorがなんかよろしくやってくれるらしい
  if train_unet and train_text_encoder:
    unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler)
  elif train_unet:
    unet, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet,  network, optimizer, train_dataloader, lr_scheduler)
  elif train_text_encoder:
    text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, network, optimizer, train_dataloader, lr_scheduler)
  else:
    network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        network, optimizer, train_dataloader, lr_scheduler)

  unet.requires_grad_(False)
  unet.to(accelerator.device, dtype=weight_dtype)
  text_encoder.requires_grad_(False)
  text_encoder.to(accelerator.device)
  if args.gradient_checkpointing:                       # according to TI example in Diffusers, train is required
    unet.train()
    text_encoder.train()

    # set top parameter requires_grad = True for gradient checkpointing works
    if type(text_encoder) == DDP:
      text_encoder.module.text_model.embeddings.requires_grad_(True)
    else:
      text_encoder.text_model.embeddings.requires_grad_(True)
  else:
    unet.eval()
    text_encoder.eval()

  # support DistributedDataParallel
  if type(text_encoder) == DDP:
    text_encoder = text_encoder.module
    unet = unet.module
    network = network.module

  network.prepare_grad_etc(text_encoder, unet)

  if not cache_latents:
    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device, dtype=weight_dtype)

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
  print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
  print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

  metadata = {
      "ss_session_id": session_id,            # random integer indicating which group of epochs the model came from
      "ss_training_started_at": training_started_at,          # unix timestamp
      "ss_output_name": args.output_name,
      "ss_learning_rate": args.learning_rate,
      "ss_text_encoder_lr": args.text_encoder_lr,
      "ss_unet_lr": args.unet_lr,
      "ss_num_train_images": train_dataset.num_train_images,          # includes repeating
      "ss_num_reg_images": train_dataset.num_reg_images,
      "ss_num_batches_per_epoch": len(train_dataloader),
      "ss_num_epochs": num_train_epochs,
      "ss_batch_size_per_device": args.train_batch_size,
      "ss_total_batch_size": total_batch_size,
      "ss_gradient_checkpointing": args.gradient_checkpointing,
      "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
      "ss_max_train_steps": args.max_train_steps,
      "ss_lr_warmup_steps": args.lr_warmup_steps,
      "ss_lr_scheduler": args.lr_scheduler,
      "ss_network_module": args.network_module,
      "ss_network_dim": args.network_dim,          # None means default because another network than LoRA may have another default dim
      "ss_network_alpha": args.network_alpha,      # some networks may not use this value
      "ss_mixed_precision": args.mixed_precision,
      "ss_full_fp16": bool(args.full_fp16),
      "ss_v2": bool(args.v2),
      "ss_resolution": args.resolution,
      "ss_clip_skip": args.clip_skip,
      "ss_max_token_length": args.max_token_length,
      "ss_color_aug": bool(args.color_aug),
      "ss_flip_aug": bool(args.flip_aug),
      "ss_random_crop": bool(args.random_crop),
      "ss_shuffle_caption": bool(args.shuffle_caption),
      "ss_cache_latents": bool(args.cache_latents),
      "ss_enable_bucket": bool(train_dataset.enable_bucket),
      "ss_bucket_no_upscale": bool(train_dataset.bucket_no_upscale),
      "ss_min_bucket_reso": train_dataset.min_bucket_reso,
      "ss_max_bucket_reso": train_dataset.max_bucket_reso,
      "ss_seed": args.seed,
      "ss_lowram": args.lowram,
      "ss_keep_tokens": args.keep_tokens,
      "ss_noise_offset": args.noise_offset,
      "ss_dataset_dirs": json.dumps(train_dataset.dataset_dirs_info),
      "ss_reg_dataset_dirs": json.dumps(train_dataset.reg_dataset_dirs_info),
      "ss_tag_frequency": json.dumps(train_dataset.tag_frequency),
      "ss_bucket_info": json.dumps(train_dataset.bucket_info),
      "ss_training_comment": args.training_comment,       # will not be updated after training
      "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
      "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
      "ss_max_grad_norm": args.max_grad_norm,
      "ss_caption_dropout_rate": args.caption_dropout_rate,
      "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
      "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
      "ss_face_crop_aug_range": args.face_crop_aug_range,
      "ss_prior_loss_weight": args.prior_loss_weight,
  }

  # uncomment if another network is added
  # for key, value in net_kwargs.items():
  #   metadata["ss_arg_" + key] = value

  if args.pretrained_model_name_or_path is not None:
    sd_model_name = args.pretrained_model_name_or_path
    if os.path.exists(sd_model_name):
      metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
      metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
      sd_model_name = os.path.basename(sd_model_name)
    metadata["ss_sd_model_name"] = sd_model_name

  if args.vae is not None:
    vae_name = args.vae
    if os.path.exists(vae_name):
      metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
      metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
      vae_name = os.path.basename(vae_name)
    metadata["ss_vae_name"] = vae_name

  metadata = {k: str(v) for k, v in metadata.items()}

  progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
  global_step = 0

  noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                  num_train_timesteps=1000, clip_sample=False)

  if accelerator.is_main_process:
    accelerator.init_trackers("network_train")

  loss_list = []
  loss_total = 0.0
  for epoch in range(num_train_epochs):
    print(f"epoch {epoch+1}/{num_train_epochs}")
    train_dataset.set_current_epoch(epoch + 1)

    metadata["ss_epoch"] = str(epoch+1)

    network.on_epoch_start(text_encoder, unet)

    for step, batch in enumerate(train_dataloader):
      with accelerator.accumulate(network):
        with torch.no_grad():
          if "latents" in batch and batch["latents"] is not None:
            latents = batch["latents"].to(accelerator.device)
          else:
            # latentに変換
            latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
          latents = latents * 0.18215
        b_size = latents.shape[0]

        with torch.set_grad_enabled(train_text_encoder):
          # Get the text embedding for conditioning
          input_ids = batch["input_ids"].to(accelerator.device)
          encoder_hidden_states = train_util.get_hidden_states(args, input_ids, tokenizer, text_encoder, weight_dtype)

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
        with accelerator.autocast():
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
          params_to_clip = network.get_trainable_params()
          accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

      # Checks if the accelerator has performed an optimization step behind the scenes
      if accelerator.sync_gradients:
        progress_bar.update(1)
        global_step += 1

      current_loss = loss.detach().item()
      if epoch == 0:
        loss_list.append(current_loss)
      else:
        loss_total -= loss_list[step]
        loss_list[step] = current_loss
      loss_total += current_loss
      avr_loss = loss_total / len(loss_list)
      logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
      progress_bar.set_postfix(**logs)

      if args.logging_dir is not None:
        logs = generate_step_logs(args, current_loss, avr_loss, lr_scheduler)
        accelerator.log(logs, step=global_step)

      if global_step >= args.max_train_steps:
        break

    if args.logging_dir is not None:
      logs = {"loss/epoch": loss_total / len(loss_list)}
      accelerator.log(logs, step=epoch+1)

    accelerator.wait_for_everyone()

    if args.save_every_n_epochs is not None:
      model_name = train_util.DEFAULT_EPOCH_NAME if args.output_name is None else args.output_name

      def save_func():
        ckpt_name = train_util.EPOCH_FILE_NAME.format(model_name, epoch + 1) + '.' + args.save_model_as
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        metadata["ss_training_finished_at"] = str(time.time())
        print(f"saving checkpoint: {ckpt_file}")
        unwrap_model(network).save_weights(ckpt_file, save_dtype, None if args.no_metadata else metadata)

      def remove_old_func(old_epoch_no):
        old_ckpt_name = train_util.EPOCH_FILE_NAME.format(model_name, old_epoch_no) + '.' + args.save_model_as
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
          print(f"removing old checkpoint: {old_ckpt_file}")
          os.remove(old_ckpt_file)

      saving = train_util.save_on_epoch_end(args, save_func, remove_old_func, epoch + 1, num_train_epochs)
      if saving and args.save_state:
        train_util.save_state_on_epoch_end(args, accelerator, model_name, epoch + 1)

    # end of epoch

  metadata["ss_epoch"] = str(num_train_epochs)
  metadata["ss_training_finished_at"] = str(time.time())

  is_main_process = accelerator.is_main_process
  if is_main_process:
    network = unwrap_model(network)

  accelerator.end_training()

  if args.save_state:
    train_util.save_state_on_train_end(args, accelerator)

  del accelerator                         # この後メモリを使うのでこれは消す

  if is_main_process:
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = train_util.DEFAULT_LAST_OUTPUT_NAME if args.output_name is None else args.output_name
    ckpt_name = model_name + '.' + args.save_model_as
    ckpt_file = os.path.join(args.output_dir, ckpt_name)

    print(f"save trained model to {ckpt_file}")
    network.save_weights(ckpt_file, save_dtype, None if args.no_metadata else metadata)
    print("model saved.")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  train_util.add_sd_models_arguments(parser)
  train_util.add_dataset_arguments(parser, True, True, True)
  train_util.add_training_arguments(parser, True)
  train_util.add_optimizer_arguments(parser)

  parser.add_argument("--no_metadata", action='store_true', help="do not save metadata in output model / メタデータを出力先モデルに保存しない")
  parser.add_argument("--save_model_as", type=str, default="safetensors", choices=[None, "ckpt", "pt", "safetensors"],
                      help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）")

  parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
  parser.add_argument("--text_encoder_lr", type=float, default=None, help="learning rate for Text Encoder / Text Encoderの学習率")

  parser.add_argument("--network_weights", type=str, default=None,
                      help="pretrained weights for network / 学習するネットワークの初期重み")
  parser.add_argument("--network_module", type=str, default=None, help='network module to train / 学習対象のネットワークのモジュール')
  parser.add_argument("--network_dim", type=int, default=None,
                      help='network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）')
  parser.add_argument("--network_alpha", type=float, default=1,
                      help='alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）')
  parser.add_argument("--network_args", type=str, default=None, nargs='*',
                      help='additional argmuments for network (key=value) / ネットワークへの追加の引数')
  parser.add_argument("--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する")
  parser.add_argument("--network_train_text_encoder_only", action="store_true",
                      help="only training Text Encoder part / Text Encoder関連部分のみ学習する")
  parser.add_argument("--training_comment", type=str, default=None,
                      help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")

  args = parser.parse_args()
  train(args)
