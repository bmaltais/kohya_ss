# このスクリプトのライセンスは、train_dreambooth.pyと同じくApache License 2.0とします
# (c) 2022 Kohya S. @kohya_ss

import argparse
import math
import os
import random
import json
import importlib

from tqdm import tqdm
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
import numpy as np
from einops import rearrange
from torch import einsum

import fine_tuning_utils

# Tokenizer: checkpointから読み込むのではなくあらかじめ提供されているものを使う
TOKENIZER_PATH = "openai/clip-vit-large-patch14"

# checkpointファイル名
LAST_CHECKPOINT_NAME = "last.ckpt"
EPOCH_CHECKPOINT_NAME = "epoch-{:06d}.ckpt"


def collate_fn(examples):
  return examples[0]


class FineTuningDataset(torch.utils.data.Dataset):
  def __init__(self, metadata, train_data_dir, batch_size, tokenizer, max_token_length, shuffle_caption, dataset_repeats, debug) -> None:
    super().__init__()

    self.metadata = metadata
    self.train_data_dir = train_data_dir
    self.batch_size = batch_size
    self.tokenizer = tokenizer
    self.max_token_length = max_token_length
    self.shuffle_caption = shuffle_caption
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
      if not os.path.exists(os.path.join(self.train_data_dir, image_key + '.npz')):
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

  def load_latent(self, image_key):
    return np.load(os.path.join(self.train_data_dir, image_key + '.npz'))['arr_0']

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
        random.shuffle(tokens)
        caption = ",".join(tokens).strip()

      captions.append(caption)

      input_ids = self.tokenizer(caption, padding="max_length", truncation=True,
                                 max_length=self.tokenizer_max_length, return_tensors="pt").input_ids

      # 77以上の時は "<CLS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<CLS>...<EOS>"の三連に変換する
      # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
      if self.tokenizer_max_length > self.tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        iids_list = []
        for i in range(1, self.tokenizer_max_length - self.tokenizer.model_max_length + 2, self.tokenizer.model_max_length - 2):
          iid = (input_ids[0].unsqueeze(0),
                 input_ids[i:i + self.tokenizer.model_max_length - 2],
                 input_ids[-1].unsqueeze(0))
          iid = torch.cat(iid)
          iids_list.append(iid)
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


def save_hypernetwork(output_file, hypernetwork):
  state_dict = hypernetwork.get_state_dict()
  torch.save(state_dict, output_file)


def train(args):
  fine_tuning = args.hypernetwork_module is None            # fine tuning or hypernetwork training

  # モデル形式のオプション設定を確認する
  use_stable_diffusion_format = os.path.isfile(args.pretrained_model_name_or_path)
  if not use_stable_diffusion_format:
    assert os.path.exists(
        args.pretrained_model_name_or_path), f"no pretrained model / 学習元モデルがありません : {args.pretrained_model_name_or_path}"

  assert not fine_tuning or (
      args.save_every_n_epochs is None or use_stable_diffusion_format), "when loading Diffusers model, save_every_n_epochs does not work / Diffusersのモデルを読み込むときにはsave_every_n_epochsオプションは無効になります"

  if args.seed is not None:
    set_seed(args.seed)

  # メタデータを読み込む
  if os.path.exists(args.in_json):
    print(f"loading existing metadata: {args.in_json}")
    with open(args.in_json, "rt", encoding='utf-8') as f:
      metadata = json.load(f)
  else:
    print(f"no metadata / メタデータファイルがありません: {args.in_json}")
    return

  # tokenizerを読み込む
  print("prepare tokenizer")
  tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)
  if args.max_token_length is not None:
    print(f"update token length in tokenizer: {args.max_token_length}")

  # datasetを用意する
  print("prepare dataset")
  train_dataset = FineTuningDataset(metadata, args.train_data_dir, args.train_batch_size,
                                    tokenizer, args.max_token_length, args.shuffle_caption, args.dataset_repeats, args.debug_dataset)

  if args.debug_dataset:
    print(f"Total dataset length / データセットの長さ: {len(train_dataset)}")
    print(f"Total images / 画像数: {train_dataset.images_count}")
    train_dataset.show_buckets()
    i = 0
    for example in train_dataset:
      print(f"image: {example['image_keys']}")
      print(f"captions: {example['captions']}")
      print(f"latents: {example['latents'].shape}")
      print(f"input_ids: {example['input_ids'].shape}")
      print(example['input_ids'])
      i += 1
      if i >= 8:
        break
    return

  # acceleratorを準備する
  print("prepare accelerator")
  accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)

  # モデルを読み込む
  if use_stable_diffusion_format:
    print("load StableDiffusion checkpoint")
    text_encoder, _, unet = fine_tuning_utils.load_models_from_stable_diffusion_checkpoint(args.pretrained_model_name_or_path)
  else:
    print("load Diffusers pretrained models")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

  # モデルに xformers とか memory efficient attention を組み込む
  replace_unet_modules(unet, args.mem_eff_attn, args.xformers)

  if not fine_tuning:
    # Hypernetwork
    print("import hypernetwork module:", args.hypernetwork_module)
    hyp_module = importlib.import_module(args.hypernetwork_module)

    hypernetwork = hyp_module.Hypernetwork()

    if args.hypernetwork_weights is not None:
      print("load hypernetwork weights from:", args.hypernetwork_weights)
      hyp_sd = torch.load(args.hypernetwork_weights, map_location='cpu')
      success = hypernetwork.load_from_state_dict(hyp_sd)
      assert success, "hypernetwork weights loading failed."

    print("apply hypernetwork")
    hypernetwork.apply_to_diffusers(None, text_encoder, unet)

  # mixed precisionに対応した型を用意しておき適宜castする
  weight_dtype = torch.float32
  if args.mixed_precision == "fp16":
    weight_dtype = torch.float16
  elif args.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

  # 学習を準備する
  if fine_tuning:
    if args.gradient_checkpointing:
      unet.enable_gradient_checkpointing()
    unet.requires_grad_(True)             # unetは学習しない
    net = unet
  else:
    unet.requires_grad_(False)             # unetは学習しない
    unet.eval()

    hypernetwork.requires_grad_(True)
    net = hypernetwork

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
  optimizer = optimizer_class(net.parameters(), lr=args.learning_rate)

  # dataloaderを準備する
  # DataLoaderのプロセス数：0はメインプロセスになる
  n_workers = min(8, os.cpu_count() - 1)      # cpu_count-1 ただし最大8
  train_dataloader = torch.utils.data.DataLoader(
      train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=n_workers)

  # lr schedulerを用意する
  lr_scheduler = diffusers.optimization.get_scheduler(
      "constant", optimizer, num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)

  # acceleratorがなんかよろしくやってくれるらしい
  if fine_tuning:
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
    net = unet
  else:
    unet, hypernetwork, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, hypernetwork, optimizer, train_dataloader, lr_scheduler)
    net = hypernetwork

  text_encoder.to(accelerator.device, dtype=weight_dtype)
  text_encoder.requires_grad_(False)             # text encoderは学習しない

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
  print(f"  total train batch size (with parallel & distributed) / 総バッチサイズ（並列学習含む）: {total_batch_size}")
  print(f"  gradient ccumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
  print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

  progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="steps")
  global_step = 0

  noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

  if accelerator.is_main_process:
    accelerator.init_trackers("finetuning" if fine_tuning else "hypernetwork")

  # 以下 train_dreambooth.py からほぼコピペ
  for epoch in range(num_train_epochs):
    print(f"epoch {epoch+1}/{num_train_epochs}")
    net.train()

    loss_total = 0
    for step, batch in enumerate(train_dataloader):
      with accelerator.accumulate(unet):
        latents = batch["latents"].to(accelerator.device)
        latents = latents * 0.18215
        b_size = latents.shape[0]

        with torch.no_grad():
          # Get the text embedding for conditioning
          input_ids = batch["input_ids"].to(accelerator.device)
          input_ids = input_ids.reshape((-1, tokenizer.model_max_length))     # batch_size*3, 77

          if args.clip_skip is None:
            encoder_hidden_states = text_encoder(input_ids)[0]
          else:
            enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
            encoder_hidden_states = enc_out['hidden_states'][-args.clip_skip]
            encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

          encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

          if args.max_token_length is not None:
            # <CLS>...<EOS> の三連を <CLS>...<EOS> へ戻す
            sts_list = [encoder_hidden_states[:, 0].unsqueeze(1)]
            for i in range(1, args.max_token_length, tokenizer.model_max_length):
              sts_list.append(encoder_hidden_states[:, i:i + tokenizer.model_max_length - 2])
            sts_list.append(encoder_hidden_states[:, -1].unsqueeze(1))
            encoder_hidden_states = torch.cat(sts_list, dim=1)

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

        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        accelerator.backward(loss)
        if accelerator.sync_gradients:
          accelerator.clip_grad_norm_(net.parameters(), 1.0)  # args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

      # Checks if the accelerator has performed an optimization step behind the scenes
      if accelerator.sync_gradients:
        progress_bar.update(1)
        global_step += 1

      current_loss = loss.detach().item() * b_size
      loss_total += current_loss
      avr_loss = loss_total / (step+1)
      logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
      progress_bar.set_postfix(**logs)
      # accelerator.log(logs, step=global_step)

      if global_step >= args.max_train_steps:
        break

    accelerator.wait_for_everyone()

    if args.save_every_n_epochs is not None:
      if (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs:
        print("saving check point.")
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, EPOCH_CHECKPOINT_NAME.format(epoch + 1))

        if fine_tuning:
          fine_tuning_utils.save_stable_diffusion_checkpoint(
              ckpt_file, text_encoder, accelerator.unwrap_model(net), args.pretrained_model_name_or_path, epoch + 1, global_step)
        else:
          save_hypernetwork(ckpt_file, accelerator.unwrap_model(net))

  is_main_process = accelerator.is_main_process
  if is_main_process:
    net = accelerator.unwrap_model(net)

  accelerator.end_training()
  del accelerator                         # この後メモリを使うのでこれは消す

  if is_main_process:
    os.makedirs(args.output_dir, exist_ok=True)
    if fine_tuning:
      if use_stable_diffusion_format:
        ckpt_file = os.path.join(args.output_dir, LAST_CHECKPOINT_NAME)
        print(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        fine_tuning_utils.save_stable_diffusion_checkpoint(
            ckpt_file, text_encoder, unet, args.pretrained_model_name_or_path, epoch, global_step)
      else:
        # Create the pipeline using using the trained modules and save it.
        print(f"save trained model as Diffusers to {args.output_dir}")
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            text_encoder=text_encoder,
        )
        pipeline.save_pretrained(args.output_dir)
    else:
      ckpt_file = os.path.join(args.output_dir, LAST_CHECKPOINT_NAME)
      print(f"save trained model to {ckpt_file}")
      save_hypernetwork(ckpt_file, net)
    print("model saved.")


# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
  return val is not None


def default(val, d):
  return val if exists(val) else d

# flash attention forwards and backwards

# https://arxiv.org/abs/2205.14135


class FlashAttentionFunction(torch.autograd.function.Function):
  @ staticmethod
  @ torch.no_grad()
  def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
    """ Algorithm 2 in the paper """

    device = q.device
    dtype = q.dtype
    max_neg_value = -torch.finfo(q.dtype).max
    qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

    o = torch.zeros_like(q)
    all_row_sums = torch.zeros((*q.shape[:-1], 1), dtype=dtype, device=device)
    all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, dtype=dtype, device=device)

    scale = (q.shape[-1] ** -0.5)

    if not exists(mask):
      mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
    else:
      mask = rearrange(mask, 'b n -> b 1 1 n')
      mask = mask.split(q_bucket_size, dim=-1)

    row_splits = zip(
        q.split(q_bucket_size, dim=-2),
        o.split(q_bucket_size, dim=-2),
        mask,
        all_row_sums.split(q_bucket_size, dim=-2),
        all_row_maxes.split(q_bucket_size, dim=-2),
    )

    for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
      q_start_index = ind * q_bucket_size - qk_len_diff

      col_splits = zip(
          k.split(k_bucket_size, dim=-2),
          v.split(k_bucket_size, dim=-2),
      )

      for k_ind, (kc, vc) in enumerate(col_splits):
        k_start_index = k_ind * k_bucket_size

        attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

        if exists(row_mask):
          attn_weights.masked_fill_(~row_mask, max_neg_value)

        if causal and q_start_index < (k_start_index + k_bucket_size - 1):
          causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                   device=device).triu(q_start_index - k_start_index + 1)
          attn_weights.masked_fill_(causal_mask, max_neg_value)

        block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
        attn_weights -= block_row_maxes
        exp_weights = torch.exp(attn_weights)

        if exists(row_mask):
          exp_weights.masked_fill_(~row_mask, 0.)

        block_row_sums = exp_weights.sum(dim=-1, keepdims=True).clamp(min=EPSILON)

        new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

        exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

        exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
        exp_block_row_max_diff = torch.exp(block_row_maxes - new_row_maxes)

        new_row_sums = exp_row_max_diff * row_sums + exp_block_row_max_diff * block_row_sums

        oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_((exp_block_row_max_diff / new_row_sums) * exp_values)

        row_maxes.copy_(new_row_maxes)
        row_sums.copy_(new_row_sums)

    ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
    ctx.save_for_backward(q, k, v, o, all_row_sums, all_row_maxes)

    return o

  @ staticmethod
  @ torch.no_grad()
  def backward(ctx, do):
    """ Algorithm 4 in the paper """

    causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
    q, k, v, o, l, m = ctx.saved_tensors

    device = q.device

    max_neg_value = -torch.finfo(q.dtype).max
    qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    row_splits = zip(
        q.split(q_bucket_size, dim=-2),
        o.split(q_bucket_size, dim=-2),
        do.split(q_bucket_size, dim=-2),
        mask,
        l.split(q_bucket_size, dim=-2),
        m.split(q_bucket_size, dim=-2),
        dq.split(q_bucket_size, dim=-2)
    )

    for ind, (qc, oc, doc, row_mask, lc, mc, dqc) in enumerate(row_splits):
      q_start_index = ind * q_bucket_size - qk_len_diff

      col_splits = zip(
          k.split(k_bucket_size, dim=-2),
          v.split(k_bucket_size, dim=-2),
          dk.split(k_bucket_size, dim=-2),
          dv.split(k_bucket_size, dim=-2),
      )

      for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
        k_start_index = k_ind * k_bucket_size

        attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

        if causal and q_start_index < (k_start_index + k_bucket_size - 1):
          causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                   device=device).triu(q_start_index - k_start_index + 1)
          attn_weights.masked_fill_(causal_mask, max_neg_value)

        exp_attn_weights = torch.exp(attn_weights - mc)

        if exists(row_mask):
          exp_attn_weights.masked_fill_(~row_mask, 0.)

        p = exp_attn_weights / lc

        dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
        dp = einsum('... i d, ... j d -> ... i j', doc, vc)

        D = (doc * oc).sum(dim=-1, keepdims=True)
        ds = p * scale * (dp - D)

        dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
        dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

        dqc.add_(dq_chunk)
        dkc.add_(dk_chunk)
        dvc.add_(dv_chunk)

    return dq, dk, dv, None, None, None, None


def replace_unet_modules(unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers):
  if mem_eff_attn:
    replace_unet_cross_attn_to_memory_efficient()
  elif xformers:
    replace_unet_cross_attn_to_xformers()


def replace_unet_cross_attn_to_memory_efficient():
  print("Replace CrossAttention.forward to use FlashAttention")
  flash_func = FlashAttentionFunction

  def forward_flash_attn(self, x, context=None, mask=None):
    q_bucket_size = 512
    k_bucket_size = 1024

    h = self.heads
    q = self.to_q(x)

    context = context if context is not None else x
    context = context.to(x.dtype)

    if hasattr(self, 'hypernetwork') and self.hypernetwork is not None:
      context_k, context_v = self.hypernetwork.forward(x, context)
      context_k = context_k.to(x.dtype)
      context_v = context_v.to(x.dtype)
    else:
      context_k = context
      context_v = context

    k = self.to_k(context_k)
    v = self.to_v(context_v)
    del context, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

    out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

    out = rearrange(out, 'b h n d -> b n (h d)')

    # diffusers 0.6.0
    if type(self.to_out) is torch.nn.Sequential:
      return self.to_out(out)

    # diffusers 0.7.0~  わざわざ変えるなよ (;´Д｀)
    out = self.to_out[0](out)
    out = self.to_out[1](out)
    return out

  diffusers.models.attention.CrossAttention.forward = forward_flash_attn


def replace_unet_cross_attn_to_xformers():
  print("Replace CrossAttention.forward to use xformers")
  try:
    import xformers.ops
  except ImportError:
    raise ImportError("No xformers / xformersがインストールされていないようです")

  def forward_xformers(self, x, context=None, mask=None):
    h = self.heads
    q_in = self.to_q(x)

    context = default(context, x)
    context = context.to(x.dtype)

    if hasattr(self, 'hypernetwork') and self.hypernetwork is not None:
      context_k, context_v = self.hypernetwork.forward(x, context)
      context_k = context_k.to(x.dtype)
      context_v = context_v.to(x.dtype)
    else:
      context_k = context
      context_v = context

    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)        # 最適なのを選んでくれる

    out = rearrange(out, 'b n h d -> b n (h d)', h=h)

    # diffusers 0.6.0
    if type(self.to_out) is torch.nn.Sequential:
      return self.to_out(out)

    # diffusers 0.7.0~
    out = self.to_out[0](out)
    out = self.to_out[1](out)
    return out

  diffusers.models.attention.CrossAttention.forward = forward_xformers
# endregion


if __name__ == '__main__':
  # torch.cuda.set_per_process_memory_fraction(0.48)
  parser = argparse.ArgumentParser()
  parser.add_argument("--pretrained_model_name_or_path", type=str, default=None,
                      help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint / 学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル")
  parser.add_argument("--in_json", type=str, default=None, help="metadata file to input / 読みこむメタデータファイル")
  parser.add_argument("--shuffle_caption", action="store_true",
                      help="shuffle comma-separated caption when fine tuning / fine tuning時にコンマで区切られたcaptionの各要素をshuffleする")
  parser.add_argument("--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("--dataset_repeats", type=int, default=None, help="num times to repeat dataset / 学習にデータセットを繰り返す回数")
  parser.add_argument("--output_dir", type=str, default=None,
                      help="directory to output trained model, save as same format as input / 学習後のモデル出力先ディレクトリ（入力と同じ形式で保存）")
  parser.add_argument("--hypernetwork_module", type=str, default=None,
                      help='train hypernetwork instead of fine tuning, module to use / fine tuningの代わりにHypernetworkの学習をする場合、そのモジュール')
  parser.add_argument("--hypernetwork_weights", type=str, default=None,
                      help='hypernetwork weights to initialize for additional training / Hypernetworkの学習時に読み込む重み（Hypernetworkの追加学習）')
  parser.add_argument("--save_every_n_epochs", type=int, default=None,
                      help="save checkpoint every N epochs (only supports in StableDiffusion checkpoint) / 学習中のモデルを指定エポックごとに保存する（StableDiffusion形式のモデルを読み込んだ場合のみ有効）")
  parser.add_argument("--max_token_length", type=int, default=None, choices=[None, 150, 225],
                      help="max token length of text encoder (default for 75, 150 or 225) / text encoderのトークンの最大長（未指定で75、150または225が指定可）")
  parser.add_argument("--train_batch_size", type=int, default=1,
                      help="batch size for training / 学習時のバッチサイズ")
  parser.add_argument("--use_8bit_adam", action="store_true",
                      help="use 8bit Adam optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）")
  parser.add_argument("--mem_eff_attn", action="store_true",
                      help="use memory efficient attention for CrossAttention / CrossAttentionに省メモリ版attentionを使う")
  parser.add_argument("--xformers", action="store_true",
                      help="use xformers for CrossAttention / CrossAttentionにxformersを使う")
  parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
  parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
  parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
  parser.add_argument("--gradient_checkpointing", action="store_true",
                      help="enable gradient checkpointing / grandient checkpointingを有効にする")
  parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数")
  parser.add_argument("--mixed_precision", type=str, default="no",
                      choices=["no", "fp16", "bf16"], help="use mixed precision / 混合精度を使う場合、その精度")
  parser.add_argument("--clip_skip", type=int, default=None,
                      help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）")
  parser.add_argument("--debug_dataset", action="store_true",
                      help="show images for debugging (do not train) / デバッグ用に学習データを画面表示する（学習は行わない）")

  args = parser.parse_args()
  train(args)
