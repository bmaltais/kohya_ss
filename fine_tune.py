# training with captions
# XXX dropped option: fine_tune

import argparse
import gc
import math
import os
import random
import json
import importlib
import time

from tqdm import tqdm
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPTokenizer
import diffusers
from diffusers import DDPMScheduler, StableDiffusionPipeline
import numpy as np
from einops import rearrange
from torch import einsum

import library.model_util as model_util
import library.train_util as train_util


def collate_fn(examples):
  return examples[0]


class FineTuningDataset(torch.utils.data.Dataset):
  def __init__(self, metadata, train_data_dir, batch_size, tokenizer, max_token_length, shuffle_caption, shuffle_keep_tokens, dataset_repeats, debug) -> None:
    super().__init__()

    self.metadata = metadata
    self.train_data_dir = train_data_dir
    self.batch_size = batch_size
    self.tokenizer: CLIPTokenizer = tokenizer
    self.max_token_length = max_token_length
    self.shuffle_caption = shuffle_caption
    self.shuffle_keep_tokens = shuffle_keep_tokens
    self.debug = debug

    self.tokenizer_max_length = self.tokenizer.model_max_length if max_token_length is None else max_token_length + 2

    print("make buckets")

    # 最初に数を数える
    self.bucket_resos = set()
    for img_md in metadata.values():
      if 'train_resolution' in img_md:
        self.bucket_resos.add(tuple(img_md['train_resolution']))
    self.bucket_resos = list(self.bucket_resos)
    self.bucket_resos.sort()
    print(f"number of buckets: {len(self.bucket_resos)}")

    reso_to_index = {}
    for i, reso in enumerate(self.bucket_resos):
      reso_to_index[reso] = i

    # bucketに割り当てていく
    self.buckets = [[] for _ in range(len(self.bucket_resos))]
    n = 1 if dataset_repeats is None else dataset_repeats
    images_count = 0
    for image_key, img_md in metadata.items():
      if 'train_resolution' not in img_md:
        continue
      if not os.path.exists(self.image_key_to_npz_file(image_key)):
        continue

      reso = tuple(img_md['train_resolution'])
      for _ in range(n):
        self.buckets[reso_to_index[reso]].append(image_key)
      images_count += n

    # 参照用indexを作る
    self.buckets_indices = []
    for bucket_index, bucket in enumerate(self.buckets):
      batch_count = int(math.ceil(len(bucket) / self.batch_size))
      for batch_index in range(batch_count):
        self.buckets_indices.append((bucket_index, batch_index))

    self.shuffle_buckets()
    self._length = len(self.buckets_indices)
    self.images_count = images_count

  def show_buckets(self):
    for i, (reso, bucket) in enumerate(zip(self.bucket_resos, self.buckets)):
      print(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

  def shuffle_buckets(self):
    random.shuffle(self.buckets_indices)
    for bucket in self.buckets:
      random.shuffle(bucket)

  def image_key_to_npz_file(self, image_key):
    npz_file_norm = os.path.splitext(image_key)[0] + '.npz'
    if os.path.exists(npz_file_norm):
      if random.random() < .5:
        npz_file_flip = os.path.splitext(image_key)[0] + '_flip.npz'
        if os.path.exists(npz_file_flip):
          return npz_file_flip
      return npz_file_norm

    npz_file_norm = os.path.join(self.train_data_dir, image_key + '.npz')
    if random.random() < .5:
      npz_file_flip = os.path.join(self.train_data_dir, image_key + '_flip.npz')
      if os.path.exists(npz_file_flip):
        return npz_file_flip
    return npz_file_norm

  def load_latent(self, image_key):
    return np.load(self.image_key_to_npz_file(image_key))['arr_0']

  def __len__(self):
    return self._length

  def __getitem__(self, index):
    if index == 0:
      self.shuffle_buckets()

    bucket = self.buckets[self.buckets_indices[index][0]]
    image_index = self.buckets_indices[index][1] * self.batch_size

    input_ids_list = []
    latents_list = []
    captions = []
    for image_key in bucket[image_index:image_index + self.batch_size]:
      img_md = self.metadata[image_key]
      caption = img_md.get('caption')
      tags = img_md.get('tags')

      if caption is None:
        caption = tags
      elif tags is not None and len(tags) > 0:
        caption = caption + ', ' + tags
      assert caption is not None and len(caption) > 0, f"caption or tag is required / キャプションまたはタグは必須です:{image_key}"

      latents = self.load_latent(image_key)

      if self.shuffle_caption:
        tokens = caption.strip().split(",")
        if self.shuffle_keep_tokens is None:
          random.shuffle(tokens)
        else:
          if len(tokens) > self.shuffle_keep_tokens:
            keep_tokens = tokens[:self.shuffle_keep_tokens]
            tokens = tokens[self.shuffle_keep_tokens:]
            random.shuffle(tokens)
            tokens = keep_tokens + tokens
        caption = ",".join(tokens).strip()

      captions.append(caption)

      input_ids = self.tokenizer(caption, padding="max_length", truncation=True,
                                 max_length=self.tokenizer_max_length, return_tensors="pt").input_ids

      if self.tokenizer_max_length > self.tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        iids_list = []
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
          # v1
          # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
          # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
          for i in range(1, self.tokenizer_max_length - self.tokenizer.model_max_length + 2, self.tokenizer.model_max_length - 2):  # (1, 152, 75)
            ids_chunk = (input_ids[0].unsqueeze(0),
                         input_ids[i:i + self.tokenizer.model_max_length - 2],
                         input_ids[-1].unsqueeze(0))
            ids_chunk = torch.cat(ids_chunk)
            iids_list.append(ids_chunk)
        else:
          # v2
          # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
          for i in range(1, self.tokenizer_max_length - self.tokenizer.model_max_length + 2, self.tokenizer.model_max_length - 2):
            ids_chunk = (input_ids[0].unsqueeze(0),       # BOS
                         input_ids[i:i + self.tokenizer.model_max_length - 2],
                         input_ids[-1].unsqueeze(0))      # PAD or EOS
            ids_chunk = torch.cat(ids_chunk)

            # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
            # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
            if ids_chunk[-2] != self.tokenizer.eos_token_id and ids_chunk[-2] != self.tokenizer.pad_token_id:
              ids_chunk[-1] = self.tokenizer.eos_token_id
            # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
            if ids_chunk[1] == self.tokenizer.pad_token_id:
              ids_chunk[1] = self.tokenizer.eos_token_id

            iids_list.append(ids_chunk)

        input_ids = torch.stack(iids_list)      # 3,77

      input_ids_list.append(input_ids)
      latents_list.append(torch.FloatTensor(latents))

    example = {}
    example['input_ids'] = torch.stack(input_ids_list)
    example['latents'] = torch.stack(latents_list)
    if self.debug:
      example['image_keys'] = bucket[image_index:image_index + self.batch_size]
      example['captions'] = captions
    return example


def train(args):
  train_util.verify_training_args(args)
  train_util.prepare_dataset_args(args, True)

  cache_latents = args.cache_latents

  # verify load/save model formats
  load_stable_diffusion_format = os.path.isfile(args.pretrained_model_name_or_path)

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

  if args.seed is not None:
    set_seed(args.seed)                           # 乱数系列を初期化する

  tokenizer = train_util.load_tokenizer(args)

  train_dataset = FineTuningDataset(args.in_json, args.train_batch_size, args.train_data_dir,
                                    tokenizer, args.max_token_length, args.shuffle_caption, args.keep_tokens,
                                    args.resolution, args.enable_bucket, args.min_bucket_reso, args.max_bucket_reso,
                                    args.flip_aug, args.color_aug, args.face_crop_aug_range, args.dataset_repeats, args.debug_dataset)
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

  # betaやweight decayはdiffusers DreamBoothもDreamBooth SDもデフォルト値のようなのでオプションはとりあえず省略
  optimizer = optimizer_class(params_to_optimize, lr=args.learning_rate)

  # dataloaderを準備する
  # DataLoaderのプロセス数：0はメインプロセスになる
  n_workers = min(8, os.cpu_count() - 1)      # cpu_count-1 ただし最大8
  train_dataloader = torch.utils.data.DataLoader(
      train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=n_workers)

  # lr schedulerを用意する
  lr_scheduler = diffusers.optimization.get_scheduler(
      args.lr_scheduler, optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)

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
  print(f"  num examples / サンプル数: {train_dataset.images_count}")
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
    for m in training_models:
      m.train()

    loss_total = 0
    for step, batch in enumerate(train_dataloader):
      with accelerator.accumulate(training_models[0]):  # 複数モデルに対応していない模様だがとりあえずこうしておく
        latents = batch["latents"].to(accelerator.device)
        latents = latents * 0.18215
        b_size = latents.shape[0]

        # with torch.no_grad():
        with torch.set_grad_enabled(args.train_text_encoder):
          # Get the text embedding for conditioning
          input_ids = batch["input_ids"].to(accelerator.device)
          encoder_hidden_states = train_util.get_hidden_states(args, input_ids, tokenizer, text_encoder)

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

        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        accelerator.backward(loss)
        if accelerator.sync_gradients:
          params_to_clip = []
          for m in training_models:
            params_to_clip.extend(m.parameters())
          accelerator.clip_grad_norm_(params_to_clip, 1.0)  # args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

      # Checks if the accelerator has performed an optimization step behind the scenes
      if accelerator.sync_gradients:
        progress_bar.update(1)
        global_step += 1

      current_loss = loss.detach().item()        # 平均なのでbatch sizeは関係ないはず
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
      def save_func(file):
        model_util.save_diffusers_checkpoint(args.v2, out_dir, unwrap_model(text_encoder), unwrap_model(unet),
                                             src_diffusers_model_path, vae=vae, use_safetensors=use_safetensors)
      train_util.save_on_epoch_end(args, accelerator, epoch, num_train_epochs, save_func)
      if (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs:
        print("saving checkpoint.")
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, model_util.get_epoch_ckpt_name(use_safetensors, epoch + 1))

        if save_stable_diffusion_format:
          model_util.save_stable_diffusion_checkpoint(args.v2, ckpt_file, unwrap_model(text_encoder), unwrap_model(unet),
                                                      src_stable_diffusion_ckpt, epoch + 1, global_step, save_dtype, vae)
        else:
          out_dir = os.path.join(args.output_dir, train_util.EPOCH_DIFFUSERS_DIR_NAME.format(epoch + 1))
          os.makedirs(out_dir, exist_ok=True)
          model_util.save_diffusers_checkpoint(args.v2, out_dir, unwrap_model(text_encoder), unwrap_model(unet),
                                                src_diffusers_model_path, vae=vae, use_safetensors=use_safetensors)
        if args.save_state:
          print("saving state.")
          accelerator.save_state(os.path.join(args.output_dir, train_util.EPOCH_STATE_NAME.format(epoch + 1)))

  is_main_process = accelerator.is_main_process
  if is_main_process:
    if fine_tuning:
      unet = unwrap_model(unet)
      text_encoder = unwrap_model(text_encoder)
    else:
      hypernetwork = unwrap_model(hypernetwork)

  accelerator.end_training()

  if args.save_state:
    print("saving last state.")
    accelerator.save_state(os.path.join(args.output_dir, train_util.LAST_STATE_NAME))

  del accelerator                         # この後メモリを使うのでこれは消す

  if is_main_process:
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_file = os.path.join(args.output_dir, model_util.get_last_ckpt_name(use_safetensors))

    if fine_tuning:
      if save_stable_diffusion_format:
        print(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        model_util.save_stable_diffusion_checkpoint(args.v2, ckpt_file, text_encoder, unet,
                                                    src_stable_diffusion_ckpt, epoch, global_step, save_dtype, vae)
      else:
        # Create the pipeline using using the trained modules and save it.
        print(f"save trained model as Diffusers to {args.output_dir}")
        out_dir = os.path.join(args.output_dir, train_util.LAST_DIFFUSERS_DIR_NAME)
        os.makedirs(out_dir, exist_ok=True)
        model_util.save_diffusers_checkpoint(args.v2, out_dir, text_encoder, unet,
                                             src_diffusers_model_path, vae=vae, use_safetensors=use_safetensors)
    else:
      print(f"save trained model to {ckpt_file}")
      save_hypernetwork(ckpt_file, hypernetwork)

    print("model saved.")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  train_util.add_sd_models_arguments(parser)
  train_util.add_dataset_arguments(parser, False, True)
  train_util.add_training_arguments(parser, False)

  parser.add_argument("--use_safetensors", action='store_true',
                      help="use safetensors format to save (if save_model_as is not specified) / checkpoint、モデルをsafetensors形式で保存する（save_model_as未指定時）")
  parser.add_argument("--diffusers_xformers", action='store_true',
                      help='use xformers by diffusers / Diffusersでxformersを使用する')

  args = parser.parse_args()
  train(args)
