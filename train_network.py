import gc
import importlib
import json
import time
import argparse
import math
import os

from tqdm import tqdm
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPTokenizer
import diffusers
from diffusers import DDPMScheduler, StableDiffusionPipeline
import numpy as np
import cv2

import library.model_util as model_util
import library.train_util as train_util
from library.train_util import DreamBoothDataset, FineTuningDataset


def collate_fn(examples):
  return examples[0]


def train(args):
  cache_latents = args.cache_latents

  # latentsをキャッシュする場合のオプション設定を確認する
  if cache_latents:
    assert not args.color_aug, "when caching latents, color_aug cannot be used / latentをキャッシュするときはcolor_augは使えません"

  # その他のオプション設定を確認する
  if args.v_parameterization and not args.v2:
    print("v_parameterization should be with v2 / v1でv_parameterizationを使用することは想定されていません")
  if args.v2 and args.clip_skip is not None:
    print("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

  use_dreambooth_method = args.in_json is None

  # モデル形式のオプション設定を確認する：
  load_stable_diffusion_format = os.path.isfile(args.pretrained_model_name_or_path)

  # 乱数系列を初期化する
  if args.seed is not None:
    set_seed(args.seed)

  # tokenizerを読み込む
  print("prepare tokenizer")
  if args.v2:
    tokenizer = CLIPTokenizer.from_pretrained(train_util. V2_STABLE_DIFFUSION_PATH, subfolder="tokenizer")
  else:
    tokenizer = CLIPTokenizer.from_pretrained(train_util. TOKENIZER_PATH)

  if args.max_token_length is not None:
    print(f"update token length: {args.max_token_length}")

  # 学習データを用意する
  assert args.resolution is not None, f"resolution is required / resolution（解像度）を指定してください"
  resolution = tuple([int(r) for r in args.resolution.split(',')])
  if len(resolution) == 1:
    resolution = (resolution[0], resolution[0])
  assert len(resolution) == 2, \
      f"resolution must be 'size' or 'width,height' / resolution（解像度）は'サイズ'または'幅','高さ'で指定してください: {args.resolution}"

  if args.face_crop_aug_range is not None:
    face_crop_aug_range = tuple([float(r) for r in args.face_crop_aug_range.split(',')])
    assert len(
        face_crop_aug_range) == 2, f"face_crop_aug_range must be two floats / face_crop_aug_rangeは'下限,上限'で指定してください: {args.face_crop_aug_range}"
  else:
    face_crop_aug_range = None

  # データセットを準備する
  if use_dreambooth_method:
    print("Use DreamBooth method.")
    train_dataset = DreamBoothDataset(args.train_batch_size, args.train_data_dir, args.reg_data_dir,
                                      tokenizer, args.max_token_length, args.caption_extension, args.shuffle_caption, args.keep_tokens,
                                      resolution, args.prior_loss_weight, args.flip_aug, args.color_aug, face_crop_aug_range, args.random_crop, args.debug_dataset)
  else:
    print("Train with captions.")

    if args.color_aug:
      print(f"latents in npz is ignored when color_aug is True / color_augを有効にした場合、npzファイルのlatentsは無視されます")

    train_dataset = FineTuningDataset(args.in_json, args.train_batch_size, args.train_data_dir,
                                      tokenizer, args.max_token_length, args.shuffle_caption, args.keep_tokens,
                                      resolution, args.flip_aug, args.color_aug, face_crop_aug_range, args.dataset_repeats, args.debug_dataset)

    if train_dataset.min_bucket_reso is not None and (args.enable_bucket or train_dataset.min_bucket_reso != train_dataset.max_bucket_reso):
      print(f"using bucket info in metadata / メタデータ内のbucket情報を使います")
      args.min_bucket_reso = train_dataset.min_bucket_reso
      args.max_bucket_reso = train_dataset.max_bucket_reso
      args.enable_bucket = True
      print(f"min bucket reso: {args.min_bucket_reso}, max bucket reso: {args.max_bucket_reso}")

  if args.enable_bucket:
    assert min(resolution) >= args.min_bucket_reso, f"min_bucket_reso must be equal or less than resolution / min_bucket_resoは最小解像度より大きくできません。解像度を大きくするかmin_bucket_resoを小さくしてください"
    assert max(resolution) <= args.max_bucket_reso, f"max_bucket_reso must be equal or greater than resolution / max_bucket_resoは最大解像度より小さくできません。解像度を小さくするかmin_bucket_resoを大きくしてください"

  train_dataset.make_buckets(args.enable_bucket, args.min_bucket_reso, args.max_bucket_reso)

  if args.debug_dataset:
    print(f"Total dataset length (steps) / データセットの長さ（ステップ数）: {len(train_dataset)}")
    print("Escape for exit. / Escキーで中断、終了します")
    k = 0
    for example in train_dataset:
      if example['latents'] is not None:
        print("sample has latents from npz file")
      for j, (ik, cap, lw) in enumerate(zip(example['image_keys'], example['captions'], example['loss_weights'])):
        print(f'{ik}, size: {train_dataset.image_data[ik].image_size}, caption: "{cap}", loss weight: {lw}')
        if example['images'] is not None:
          im = example['images'][j]
          im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
          im = np.transpose(im, (1, 2, 0))                # c,H,W -> H,W,c
          im = im[:, :, ::-1]                             # RGB -> BGR (OpenCV)
          cv2.imshow("img", im)
          k = cv2.waitKey()
          cv2.destroyAllWindows()
          if k == 27:
            break
      if k == 27 or example['images'] is None:
        break
    return

  if len(train_dataset) == 0:
    print("No data found. Please verify arguments / 画像がありません。引数指定を確認してください")
    return

  # acceleratorを準備する
  print("prepare accelerator")
  if args.logging_dir is None:
    log_with = None
    logging_dir = None
  else:
    log_with = "tensorboard"
    log_prefix = "" if args.log_prefix is None else args.log_prefix
    logging_dir = args.logging_dir + "/" + log_prefix + time.strftime('%Y%m%d%H%M%S', time.localtime())

  accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision,
                            log_with=log_with, logging_dir=logging_dir)

  # accelerateの互換性問題を解決する
  accelerator_0_15 = True
  try:
    accelerator.unwrap_model("dummy", True)
    print("Using accelerator 0.15.0 or above.")
  except TypeError:
    accelerator_0_15 = False

  def unwrap_model(model):
    if accelerator_0_15:
      return accelerator.unwrap_model(model, True)
    return accelerator.unwrap_model(model)

  # mixed precisionに対応した型を用意しておき適宜castする
  weight_dtype = torch.float32
  if args.mixed_precision == "fp16":
    weight_dtype = torch.float16
  elif args.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

  save_dtype = None
  if args.save_precision == "fp16":
    save_dtype = torch.float16
  elif args.save_precision == "bf16":
    save_dtype = torch.bfloat16
  elif args.save_precision == "float":
    save_dtype = torch.float32

  # モデルを読み込む
  if load_stable_diffusion_format:
    print("load StableDiffusion checkpoint")
    text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.pretrained_model_name_or_path)
  else:
    print("load Diffusers pretrained models")
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, tokenizer=None, safety_checker=None)
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    del pipe

  # VAEを読み込む
  if args.vae is not None:
    vae = model_util.load_vae(args.vae, weight_dtype)
    print("additional VAE loaded")

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

  network = network_module.create_network(1.0, args.network_dim, vae, text_encoder, unet, **net_kwargs)
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

  # 8-bit Adamを使う
  if args.use_8bit_adam:
    try:
      import bitsandbytes as bnb
    except ImportError:
      raise ImportError("No bitsand bytes / bitsandbytesがインストールされていないようです")
    print("use 8-bit Adam optimizer")
    optimizer_class = bnb.optim.AdamW8bit
  else:
    optimizer_class = torch.optim.AdamW

  trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)

  # betaやweight decayはdiffusers DreamBoothもDreamBooth SDもデフォルト値のようなのでオプションはとりあえず省略
  optimizer = optimizer_class(trainable_params, lr=args.learning_rate)

  # dataloaderを準備する
  # DataLoaderのプロセス数：0はメインプロセスになる
  n_workers = min(8, os.cpu_count() - 1)      # cpu_count-1 ただし最大8
  train_dataloader = torch.utils.data.DataLoader(
      train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=n_workers)

  # lr schedulerを用意する
  lr_scheduler = diffusers.optimization.get_scheduler(
      args.lr_scheduler, optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps)

  # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
  if args.full_fp16:
    assert args.mixed_precision == "fp16", "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
    print("enable full fp16 training.")
    # unet.to(weight_dtype)
    # text_encoder.to(weight_dtype)
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
  unet.eval()
  text_encoder.requires_grad_(False)
  text_encoder.to(accelerator.device, dtype=weight_dtype)
  text_encoder.eval()

  network.prepare_grad_etc(text_encoder, unet)

  if not cache_latents:
    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device, dtype=weight_dtype)

  # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
  if args.full_fp16:
    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
      return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer

  # resumeする
  if args.resume is not None:
    print(f"resume training from state: {args.resume}")
    accelerator.load_state(args.resume)

  # epoch数を計算する
  num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
  num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

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
    accelerator.init_trackers("network_train")

  for epoch in range(num_train_epochs):
    print(f"epoch {epoch+1}/{num_train_epochs}")

    # 指定したステップ数までText Encoderを学習する：epoch最初の状態
    network.on_epoch_start(text_encoder, unet)

    loss_total = 0
    for step, batch in enumerate(train_dataloader):
      with accelerator.accumulate(network):
        with torch.no_grad():
          # latentに変換
          if batch["latents"] is not None:
            latents = batch["latents"].to(accelerator.device)
          else:
            latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
          latents = latents * 0.18215
        b_size = latents.shape[0]

        with torch.set_grad_enabled(train_text_encoder):
          # Get the text embedding for conditioning
          input_ids = batch["input_ids"].to(accelerator.device)
          input_ids = input_ids.reshape((-1, tokenizer.model_max_length))     # batch_size*3, 77

          if args.clip_skip is None:
            encoder_hidden_states = text_encoder(input_ids)[0]
          else:
            enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
            encoder_hidden_states = enc_out['hidden_states'][-args.clip_skip]
            encoder_hidden_states = encoder_hidden_states.to(weight_dtype)                    # なぜかこれが必要
            encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

          # bs*3, 77, 768 or 1024
          encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

          if args.max_token_length is not None:
            if args.v2:
              # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
              states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]                              # <BOS>
              for i in range(1, args.max_token_length, tokenizer.model_max_length):
                chunk = encoder_hidden_states[:, i:i + tokenizer.model_max_length - 2]              # <BOS> の後から 最後の前まで
                if i > 0:
                  for j in range(len(chunk)):
                    if input_ids[j, 1] == tokenizer.eos_token:                                      # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                      chunk[j, 0] = chunk[j, 1]                                                     # 次の <PAD> の値をコピーする
                states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
              states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))                         # <EOS> か <PAD> のどちらか
              encoder_hidden_states = torch.cat(states_list, dim=1)
            else:
              # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
              states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]                              # <BOS>
              for i in range(1, args.max_token_length, tokenizer.model_max_length):
                states_list.append(encoder_hidden_states[:, i:i + tokenizer.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
              states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))                         # <EOS>
              encoder_hidden_states = torch.cat(states_list, dim=1)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, device=latents.device)

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
          # Diffusers 0.10.0からv_parameterizationの学習に対応したのでそちらを使う
          target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
          target = noise

        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
        loss = loss.mean([1, 2, 3])

        loss_weights = batch["loss_weights"]                      # 各sampleごとのweight
        loss = loss * loss_weights

        loss = loss.mean()                # 平均なのでbatch_sizeで割る必要なし

        accelerator.backward(loss)
        if accelerator.sync_gradients:
          params_to_clip = network.get_trainable_params()
          accelerator.clip_grad_norm_(params_to_clip, 1.0)  # args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

      # Checks if the accelerator has performed an optimization step behind the scenes
      if accelerator.sync_gradients:
        progress_bar.update(1)
        global_step += 1

      current_loss = loss.detach().item()
      if args.logging_dir is not None:
        logs = {"loss": current_loss, "lr": lr_scheduler.get_last_lr()[0]}
        accelerator.log(logs, step=global_step)

      loss_total += current_loss
      avr_loss = loss_total / (step+1)
      logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
      progress_bar.set_postfix(**logs)

      if global_step >= args.max_train_steps:
        break

    if args.logging_dir is not None:
      logs = {"epoch_loss": loss_total / len(train_dataloader)}
      accelerator.log(logs, step=epoch+1)

    accelerator.wait_for_everyone()

    if args.save_every_n_epochs is not None:
      if (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs:
        print("saving checkpoint.")
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, train_util.EPOCH_FILE_NAME.format(epoch + 1) + '.' + args.save_model_as)
        unwrap_model(network).save_weights(ckpt_file, save_dtype)

        if args.save_state:
          print("saving state.")
          accelerator.save_state(os.path.join(args.output_dir, train_util.EPOCH_STATE_NAME.format(epoch + 1)))

  is_main_process = accelerator.is_main_process
  if is_main_process:
    network = unwrap_model(network)

  accelerator.end_training()

  if args.save_state:
    print("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator.save_state(os.path.join(args.output_dir, train_util.LAST_STATE_NAME))

  del accelerator                         # この後メモリを使うのでこれは消す

  if is_main_process:
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_file = os.path.join(args.output_dir, train_util.LAST_FILE_NAME + '.' + args.save_model_as)
    print(f"save trained model to {ckpt_file}")
    network.save_weights(ckpt_file, save_dtype)
    print("model saved.")


if __name__ == '__main__':
  # torch.cuda.set_per_process_memory_fraction(0.48)
  parser = argparse.ArgumentParser()
  parser.add_argument("--v2", action='store_true',
                      help='load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む')
  parser.add_argument("--v_parameterization", action='store_true',
                      help='enable v-parameterization training / v-parameterization学習を有効にする')
  parser.add_argument("--pretrained_model_name_or_path", type=str, default=None,
                      help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint / 学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル")
  parser.add_argument("--network_weights", type=str, default=None,
                      help="pretrained weights for network / 学習するネットワークの初期重み")
  parser.add_argument("--shuffle_caption", action="store_true",
                      help="shuffle comma-separated caption / コンマで区切られたcaptionの各要素をshuffleする")
  parser.add_argument("--keep_tokens", type=int, default=None,
                      help="keep heading N tokens when shuffling caption tokens / captionのシャッフル時に、先頭からこの個数のトークンをシャッフルしないで残す")
  parser.add_argument("--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("--reg_data_dir", type=str, default=None, help="directory for regularization images / 正則化画像データのディレクトリ")
  parser.add_argument("--in_json", type=str, default=None, help="json metadata for dataset / データセットのmetadataのjsonファイル")
  parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption files / 読み込むcaptionファイルの拡張子")
  parser.add_argument("--dataset_repeats", type=int, default=1,
                      help="repeat dataset when training with captions / キャプションでの学習時にデータセットを繰り返す回数")
  parser.add_argument("--output_dir", type=str, default=None,
                      help="directory to output trained model / 学習後のモデル出力先ディレクトリ")
  parser.add_argument("--save_precision", type=str, default=None,
                      choices=[None, "float", "fp16", "bf16"], help="precision in saving / 保存時に精度を変更して保存する")
  parser.add_argument("--save_model_as", type=str, default="pt", choices=[None, "ckpt", "pt", "safetensors"],
                      help="format to save the model (default is .pt) / モデル保存時の形式（デフォルトはpt）")
  parser.add_argument("--save_every_n_epochs", type=int, default=None,
                      help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する")
  parser.add_argument("--save_state", action="store_true",
                      help="save training state additionally (including optimizer states etc.) / optimizerなど学習状態も含めたstateを追加で保存する")
  parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")
  parser.add_argument("--color_aug", action="store_true", help="enable weak color augmentation / 学習時に色合いのaugmentationを有効にする")
  parser.add_argument("--flip_aug", action="store_true", help="enable horizontal flip augmentation / 学習時に左右反転のaugmentationを有効にする")
  parser.add_argument("--face_crop_aug_range", type=str, default=None,
                      help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0) / 学習時に顔を中心とした切り出しaugmentationを有効にするときは倍率を指定する（例：2.0,4.0）")
  parser.add_argument("--random_crop", action="store_true",
                      help="enable random crop (for style training in face-centered crop augmentation) / ランダムな切り出しを有効にする（顔を中心としたaugmentationを行うときに画風の学習用に指定する）")
  parser.add_argument("--debug_dataset", action="store_true",
                      help="show images for debugging (do not train) / デバッグ用に学習データを画面表示する（学習は行わない）")
  parser.add_argument("--resolution", type=str, default=None,
                      help="resolution in training ('size' or 'width,height') / 学習時の画像解像度（'サイズ'指定、または'幅,高さ'指定）")
  parser.add_argument("--train_batch_size", type=int, default=1, help="batch size for training / 学習時のバッチサイズ")
  parser.add_argument("--max_token_length", type=int, default=None, choices=[None, 150, 225],
                      help="max token length of text encoder (default for 75, 150 or 225) / text encoderのトークンの最大長（未指定で75、150または225が指定可）")
  parser.add_argument("--use_8bit_adam", action="store_true",
                      help="use 8bit Adam optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）")
  parser.add_argument("--mem_eff_attn", action="store_true",
                      help="use memory efficient attention for CrossAttention / CrossAttentionに省メモリ版attentionを使う")
  parser.add_argument("--xformers", action="store_true",
                      help="use xformers for CrossAttention / CrossAttentionにxformersを使う")
  parser.add_argument("--vae", type=str, default=None,
                      help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ")
  parser.add_argument("--cache_latents", action="store_true",
                      help="cache latents to reduce memory (augmentations must be disabled) / メモリ削減のためにlatentをcacheする（augmentationは使用不可）")
  parser.add_argument("--enable_bucket", action="store_true",
                      help="enable buckets for multi aspect ratio training / 複数解像度学習のためのbucketを有効にする")
  parser.add_argument("--min_bucket_reso", type=int, default=256, help="minimum resolution for buckets / bucketの最小解像度")
  parser.add_argument("--max_bucket_reso", type=int, default=1024, help="maximum resolution for buckets / bucketの最大解像度")
  parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
  parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
  parser.add_argument("--text_encoder_lr", type=float, default=None, help="learning rate for Text Encoder / Text Encoderの学習率")
  parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
  parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="loss weight for regularization images / 正則化画像のlossの重み")
  # parser.add_argument("--stop_text_encoder_training", type=int, default=None,
  #                     help="steps to stop text encoder training / Text Encoderの学習を止めるステップ数")
  parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
  parser.add_argument("--gradient_checkpointing", action="store_true",
                      help="enable gradient checkpointing / grandient checkpointingを有効にする")
  parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数")
  parser.add_argument("--mixed_precision", type=str, default="no",
                      choices=["no", "fp16", "bf16"], help="use mixed precision / 混合精度を使う場合、その精度")
  parser.add_argument("--full_fp16", action="store_true", help="fp16 training including gradients / 勾配も含めてfp16で学習する")
  parser.add_argument("--clip_skip", type=int, default=None,
                      help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）")
  parser.add_argument("--logging_dir", type=str, default=None,
                      help="enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する")
  parser.add_argument("--log_prefix", type=str, default=None, help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列")
  parser.add_argument("--lr_scheduler", type=str, default="constant",
                      help="scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup")
  parser.add_argument("--lr_warmup_steps", type=int, default=0,
                      help="Number of steps for the warmup in the lr scheduler (default is 0) / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）")
  parser.add_argument("--network_module", type=str, default=None, help='network module to train / 学習対象のネットワークのモジュール')
  parser.add_argument("--network_dim", type=int, default=None,
                      help='network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）')
  parser.add_argument("--network_args", type=str, default=None, nargs='*',
                      help='additional argmuments for network (key=value) / ネットワークへの追加の引数')
  parser.add_argument("--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する")
  parser.add_argument("--network_train_text_encoder_only", action="store_true",
                      help="only training Text Encoder part / Text Encoder関連部分のみ学習する")

  args = parser.parse_args()
  train(args)
