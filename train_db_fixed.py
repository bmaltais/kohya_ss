# このスクリプトのライセンスは、train_dreambooth.pyと同じくApache License 2.0とします
# (c) 2022 Kohya S. @kohya_ss

# v7: another text encoder ckpt format, average loss, save epochs/global steps, show num of train/reg images,
#     enable reg images in fine-tuning, add dataset_repeats option
# v8: supports Diffusers 0.7.2
# v9: add bucketing option
# v10: add min_bucket_reso/max_bucket_reso options, read captions for train/reg images in DreamBooth
# v11: Diffusers 0.9.0 is required. support for Stable Diffusion 2.0/v-parameterization
#      add lr scheduler options, change handling folder/file caption, support loading DiffUser model from Huggingface
#      support save_ever_n_epochs/save_state in DiffUsers model
#      fix the issue that prior_loss_weight is applyed to train images
# v12: stop train text encode, tqdm smoothing

import time
from torch.autograd.function import Function
import argparse
import glob
import itertools
import math
import os
import random

from tqdm import tqdm
import torch
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
import albumentations as albu
import numpy as np
from PIL import Image
import cv2
from einops import rearrange
from torch import einsum

# Tokenizer: checkpointから読み込むのではなくあらかじめ提供されているものを使う
TOKENIZER_PATH = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2"     # ここからtokenizerだけ使う

# checkpointファイル名
LAST_CHECKPOINT_NAME = "last.ckpt"
LAST_STATE_NAME = "last-state"
LAST_DIFFUSERS_DIR_NAME = "last"
EPOCH_CHECKPOINT_NAME = "epoch-{:06d}.ckpt"
EPOCH_STATE_NAME = "epoch-{:06d}-state"
EPOCH_DIFFUSERS_DIR_NAME = "epoch-{:06d}"


# region dataset


def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
  max_width, max_height = max_reso
  max_area = (max_width // divisible) * (max_height // divisible)

  resos = set()

  size = int(math.sqrt(max_area)) * divisible
  resos.add((size, size))

  size = min_size
  while size <= max_size:
    width = size
    height = min(max_size, (max_area // (width // divisible)) * divisible)
    resos.add((width, height))
    resos.add((height, width))
    size += divisible

  resos = list(resos)
  resos.sort()

  aspect_ratios = [w / h for w, h in resos]
  return resos, aspect_ratios


class DreamBoothOrFineTuningDataset(torch.utils.data.Dataset):
  def __init__(self, batch_size, fine_tuning, train_img_path_captions, reg_img_path_captions, tokenizer, resolution, prior_loss_weight, flip_aug, color_aug, face_crop_aug_range, random_crop, shuffle_caption, disable_padding, debug_dataset) -> None:
    super().__init__()

    self.batch_size = batch_size
    self.fine_tuning = fine_tuning
    self.train_img_path_captions = train_img_path_captions
    self.reg_img_path_captions = reg_img_path_captions
    self.tokenizer = tokenizer
    self.width, self.height = resolution
    self.size = min(self.width, self.height)                  # 短いほう
    self.prior_loss_weight = prior_loss_weight
    self.face_crop_aug_range = face_crop_aug_range
    self.random_crop = random_crop
    self.debug_dataset = debug_dataset
    self.shuffle_caption = shuffle_caption
    self.disable_padding = disable_padding
    self.latents_cache = None
    self.enable_bucket = False

    # augmentation
    flip_p = 0.5 if flip_aug else 0.0
    if color_aug:
      # わりと弱めの色合いaugmentation：brightness/contrastあたりは画像のpixel valueの最大値・最小値を変えてしまうのでよくないのではという想定でgamma/hue/saturationあたりを触る
      self.aug = albu.Compose([
          albu.OneOf([
              # albu.RandomBrightnessContrast(0.05, 0.05, p=.2),
              albu.HueSaturationValue(5, 8, 0, p=.2),
              # albu.RGBShift(5, 5, 5, p=.1),
              albu.RandomGamma((95, 105), p=.5),
          ], p=.33),
          albu.HorizontalFlip(p=flip_p)
      ], p=1.)
    elif flip_aug:
      self.aug = albu.Compose([
          albu.HorizontalFlip(p=flip_p)
      ], p=1.)
    else:
      self.aug = None

    self.num_train_images = len(self.train_img_path_captions)
    self.num_reg_images = len(self.reg_img_path_captions)

    self.enable_reg_images = self.num_reg_images > 0

    if self.enable_reg_images and self.num_train_images < self.num_reg_images:
      print("some of reg images are not used / 正則化画像の数が多いので、一部使用されない正則化画像があります")

    self.image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

  # bucketingを行わない場合も呼び出し必須（ひとつだけbucketを作る）
  def make_buckets_with_caching(self, enable_bucket, vae, min_size, max_size):
    self.enable_bucket = enable_bucket

    cache_latents = vae is not None
    if cache_latents:
      if enable_bucket:
        print("cache latents with bucketing")
      else:
        print("cache latents")
    else:
      if enable_bucket:
        print("make buckets")
      else:
        print("prepare dataset")

    # bucketingを用意する
    if enable_bucket:
      bucket_resos, bucket_aspect_ratios = make_bucket_resolutions((self.width, self.height), min_size, max_size)
    else:
      # bucketはひとつだけ、すべての画像は同じ解像度
      bucket_resos = [(self.width, self.height)]
      bucket_aspect_ratios = [self.width / self.height]
    bucket_aspect_ratios = np.array(bucket_aspect_ratios)

    # 画像の解像度、latentをあらかじめ取得する
    img_ar_errors = []
    self.size_lat_cache = {}
    for image_path, _ in tqdm(self.train_img_path_captions + self.reg_img_path_captions):
      if image_path in self.size_lat_cache:
        continue

      image = self.load_image(image_path)[0]
      image_height, image_width = image.shape[0:2]

      if not enable_bucket:
        # assert image_width == self.width and image_height == self.height, \
        #     f"all images must have specific resolution when bucketing is disabled / bucketを使わない場合、すべての画像のサイズを統一してください: {image_path}"
        reso = (self.width, self.height)
      else:
        # bucketを決める
        aspect_ratio = image_width / image_height
        ar_errors = bucket_aspect_ratios - aspect_ratio
        bucket_id = np.abs(ar_errors).argmin()
        reso = bucket_resos[bucket_id]
        ar_error = ar_errors[bucket_id]
        img_ar_errors.append(ar_error)

        if cache_latents:
          image = self.resize_and_trim(image, reso)

      # latentを取得する
      if cache_latents:
        img_tensor = self.image_transforms(image)
        img_tensor = img_tensor.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(img_tensor).latent_dist.sample().squeeze(0).to("cpu")
      else:
        latents = None

      self.size_lat_cache[image_path] = (reso, latents)

    # 画像をbucketに分割する
    self.buckets = [[] for _ in range(len(bucket_resos))]
    reso_to_index = {}
    for i, reso in enumerate(bucket_resos):
      reso_to_index[reso] = i

    def split_to_buckets(is_reg, img_path_captions):
      for image_path, caption in img_path_captions:
        reso, _ = self.size_lat_cache[image_path]
        bucket_index = reso_to_index[reso]
        self.buckets[bucket_index].append((is_reg, image_path, caption))

    split_to_buckets(False, self.train_img_path_captions)

    if self.enable_reg_images:
      l = []
      while len(l) < len(self.train_img_path_captions):
        l += self.reg_img_path_captions
      l = l[:len(self.train_img_path_captions)]
      split_to_buckets(True, l)

    if enable_bucket:
      print("number of images with repeats / 繰り返し回数込みの各bucketの画像枚数")
      for i, (reso, imgs) in enumerate(zip(bucket_resos, self.buckets)):
        print(f"bucket {i}: resolution {reso}, count: {len(imgs)}")
      img_ar_errors = np.array(img_ar_errors)
      print(f"mean ar error: {np.mean(np.abs(img_ar_errors))}")

    # 参照用indexを作る
    self.buckets_indices = []
    for bucket_index, bucket in enumerate(self.buckets):
      batch_count = int(math.ceil(len(bucket) / self.batch_size))
      for batch_index in range(batch_count):
        self.buckets_indices.append((bucket_index, batch_index))

    self.shuffle_buckets()
    self._length = len(self.buckets_indices)

  # どのサイズにリサイズするか→トリミングする方向で
  def resize_and_trim(self, image, reso):
    image_height, image_width = image.shape[0:2]
    ar_img = image_width / image_height
    ar_reso = reso[0] / reso[1]
    if ar_img > ar_reso:                   # 横が長い→縦を合わせる
      scale = reso[1] / image_height
    else:
      scale = reso[0] / image_width
    resized_size = (int(image_width * scale + .5), int(image_height * scale + .5))

    image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)       # INTER_AREAでやりたいのでcv2でリサイズ
    if resized_size[0] > reso[0]:
      trim_size = resized_size[0] - reso[0]
      image = image[:, trim_size//2:trim_size//2 + reso[0]]
    elif resized_size[1] > reso[1]:
      trim_size = resized_size[1] - reso[1]
      image = image[trim_size//2:trim_size//2 + reso[1]]
    assert image.shape[0] == reso[1] and image.shape[1] == reso[0],  \
        f"internal error, illegal trimmed size: {image.shape}, {reso}"
    return image

  def shuffle_buckets(self):
    random.shuffle(self.buckets_indices)
    for bucket in self.buckets:
      random.shuffle(bucket)

  def load_image(self, image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
      image = image.convert("RGB")
    img = np.array(image, np.uint8)

    face_cx = face_cy = face_w = face_h = 0
    if self.face_crop_aug_range is not None:
      tokens = os.path.splitext(os.path.basename(image_path))[0].split('_')
      if len(tokens) >= 5:
        face_cx = int(tokens[-4])
        face_cy = int(tokens[-3])
        face_w = int(tokens[-2])
        face_h = int(tokens[-1])

    return img, face_cx, face_cy, face_w, face_h

  # いい感じに切り出す
  def crop_target(self, image, face_cx, face_cy, face_w, face_h):
    height, width = image.shape[0:2]
    if height == self.height and width == self.width:
      return image

    # 画像サイズはsizeより大きいのでリサイズする
    face_size = max(face_w, face_h)
    min_scale = max(self.height / height, self.width / width)        # 画像がモデル入力サイズぴったりになる倍率（最小の倍率）
    min_scale = min(1.0, max(min_scale, self.size / (face_size * self.face_crop_aug_range[1])))             # 指定した顔最小サイズ
    max_scale = min(1.0, max(min_scale, self.size / (face_size * self.face_crop_aug_range[0])))             # 指定した顔最大サイズ
    if min_scale >= max_scale:          # range指定がmin==max
      scale = min_scale
    else:
      scale = random.uniform(min_scale, max_scale)

    nh = int(height * scale + .5)
    nw = int(width * scale + .5)
    assert nh >= self.height and nw >= self.width, f"internal error. small scale {scale}, {width}*{height}"
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    face_cx = int(face_cx * scale + .5)
    face_cy = int(face_cy * scale + .5)
    height, width = nh, nw

    # 顔を中心として448*640とかへを切り出す
    for axis, (target_size, length, face_p) in enumerate(zip((self.height, self.width), (height, width), (face_cy, face_cx))):
      p1 = face_p - target_size // 2                # 顔を中心に持ってくるための切り出し位置

      if self.random_crop:
        # 背景も含めるために顔を中心に置く確率を高めつつずらす
        range = max(length - face_p, face_p)        # 画像の端から顔中心までの距離の長いほう
        p1 = p1 + (random.randint(0, range) + random.randint(0, range)) - range     # -range ~ +range までのいい感じの乱数
      else:
        # range指定があるときのみ、すこしだけランダムに（わりと適当）
        if self.face_crop_aug_range[0] != self.face_crop_aug_range[1]:
          if face_size > self.size // 10 and face_size >= 40:
            p1 = p1 + random.randint(-face_size // 20, +face_size // 20)

      p1 = max(0, min(p1, length - target_size))

      if axis == 0:
        image = image[p1:p1 + target_size, :]
      else:
        image = image[:, p1:p1 + target_size]

    return image

  def __len__(self):
    return self._length

  def __getitem__(self, index):
    if index == 0:
      self.shuffle_buckets()

    bucket = self.buckets[self.buckets_indices[index][0]]
    image_index = self.buckets_indices[index][1] * self.batch_size

    latents_list = []
    images = []
    captions = []
    loss_weights = []

    for is_reg, image_path, caption in bucket[image_index:image_index + self.batch_size]:
      loss_weights.append(self.prior_loss_weight if is_reg else 1.0)

      # image/latentsを処理する
      reso, latents = self.size_lat_cache[image_path]

      if latents is None:
        # 画像を読み込み必要ならcropする
        img, face_cx, face_cy, face_w, face_h = self.load_image(image_path)
        im_h, im_w = img.shape[0:2]

        if self.enable_bucket:
          img = self.resize_and_trim(img, reso)
        else:
          if face_cx > 0:                   # 顔位置情報あり
            img = self.crop_target(img, face_cx, face_cy, face_w, face_h)
          elif im_h > self.height or im_w > self.width:
            assert self.random_crop, f"image too large, and face_crop_aug_range and random_crop are disabled / 画像サイズが大きいのでface_crop_aug_rangeかrandom_cropを有効にしてください"
            if im_h > self.height:
              p = random.randint(0, im_h - self.height)
              img = img[p:p + self.height]
            if im_w > self.width:
              p = random.randint(0, im_w - self.width)
              img = img[:, p:p + self.width]

          im_h, im_w = img.shape[0:2]
          assert im_h == self.height and im_w == self.width, f"image size is small / 画像サイズが小さいようです: {image_path}"

        # augmentation
        if self.aug is not None:
          img = self.aug(image=img)['image']

        image = self.image_transforms(img)      # -1.0~1.0のtorch.Tensorになる
      else:
        image = None

      images.append(image)
      latents_list.append(latents)

      # captionを処理する
      if self.shuffle_caption:         # captionのshuffleをする
        tokens = caption.strip().split(",")
        random.shuffle(tokens)
        caption = ",".join(tokens).strip()
      captions.append(caption)

    # input_idsをpadしてTensor変換
    if self.disable_padding:
      # paddingしない：padding==Trueはバッチの中の最大長に合わせるだけ（やはりバグでは……？）
      input_ids = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").input_ids
    else:
      # paddingする
      input_ids = self.tokenizer(captions, padding='max_length', truncation=True, return_tensors='pt').input_ids

    example = {}
    example['loss_weights'] = torch.FloatTensor(loss_weights)
    example['input_ids'] = input_ids
    if images[0] is not None:
      images = torch.stack(images)
      images = images.to(memory_format=torch.contiguous_format).float()
    else:
      images = None
    example['images'] = images
    example['latents'] = torch.stack(latents_list) if latents_list[0] is not None else None
    if self.debug_dataset:
      example['image_paths'] = [image_path for _, image_path, _ in bucket[image_index:image_index + self.batch_size]]
      example['captions'] = captions
    return example
# endregion


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


class FlashAttentionFunction(Function):
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
    k = self.to_k(context)
    v = self.to_v(context)
    del context, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

    out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

    out = rearrange(out, 'b h n d -> b n (h d)')

    # diffusers 0.6.0
    if type(self.to_out) is torch.nn.Sequential:
      return self.to_out(out)

    # diffusers 0.7.0~
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

    k_in = self.to_k(context)
    v_in = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q_in, k_in, v_in))          # new format
    # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))      # legacy format
    del q_in, k_in, v_in
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)        # 最適なのを選んでくれる

    out = rearrange(out, 'b n h d -> b n (h d)', h=h)
    # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

    # diffusers 0.6.0
    if type(self.to_out) is torch.nn.Sequential:
      return self.to_out(out)

    # diffusers 0.7.0~
    out = self.to_out[0](out)
    out = self.to_out[1](out)
    return out

  diffusers.models.attention.CrossAttention.forward = forward_xformers
# endregion


# region checkpoint変換、読み込み、書き込み ###############################

# DiffUsers版StableDiffusionのモデルパラメータ
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.00085
BETA_END = 0.0120

UNET_PARAMS_MODEL_CHANNELS = 320
UNET_PARAMS_CHANNEL_MULT = [1, 2, 4, 4]
UNET_PARAMS_ATTENTION_RESOLUTIONS = [4, 2, 1]
UNET_PARAMS_IMAGE_SIZE = 32  # unused
UNET_PARAMS_IN_CHANNELS = 4
UNET_PARAMS_OUT_CHANNELS = 4
UNET_PARAMS_NUM_RES_BLOCKS = 2
UNET_PARAMS_CONTEXT_DIM = 768
UNET_PARAMS_NUM_HEADS = 8

VAE_PARAMS_Z_CHANNELS = 4
VAE_PARAMS_RESOLUTION = 256
VAE_PARAMS_IN_CHANNELS = 3
VAE_PARAMS_OUT_CH = 3
VAE_PARAMS_CH = 128
VAE_PARAMS_CH_MULT = [1, 2, 4, 4]
VAE_PARAMS_NUM_RES_BLOCKS = 2

# V2
V2_UNET_PARAMS_ATTENTION_HEAD_DIM = [5, 10, 20, 20]
V2_UNET_PARAMS_CONTEXT_DIM = 1024


# region StableDiffusion->Diffusersの変換コード
# convert_original_stable_diffusion_to_diffusers をコピーしている（ASL 2.0）


def shave_segments(path, n_shave_prefix_segments=1):
  """
  Removes segments. Positive values shave the first segments, negative shave the last segments.
  """
  if n_shave_prefix_segments >= 0:
    return ".".join(path.split(".")[n_shave_prefix_segments:])
  else:
    return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside resnets to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item.replace("in_layers.0", "norm1")
    new_item = new_item.replace("in_layers.2", "conv1")

    new_item = new_item.replace("out_layers.0", "norm2")
    new_item = new_item.replace("out_layers.3", "conv2")

    new_item = new_item.replace("emb_layers.1", "time_emb_proj")
    new_item = new_item.replace("skip_connection", "conv_shortcut")

    new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside resnets to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item

    new_item = new_item.replace("nin_shortcut", "conv_shortcut")
    new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside attentions to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item

    #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
    #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

    #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
    #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

    #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside attentions to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item

    new_item = new_item.replace("norm.weight", "group_norm.weight")
    new_item = new_item.replace("norm.bias", "group_norm.bias")

    new_item = new_item.replace("q.weight", "query.weight")
    new_item = new_item.replace("q.bias", "query.bias")

    new_item = new_item.replace("k.weight", "key.weight")
    new_item = new_item.replace("k.bias", "key.bias")

    new_item = new_item.replace("v.weight", "value.weight")
    new_item = new_item.replace("v.bias", "value.bias")

    new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
    new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

    new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
  """
  This does the final conversion step: take locally converted weights and apply a global renaming
  to them. It splits attention layers, and takes into account additional replacements
  that may arise.

  Assigns the weights to the new checkpoint.
  """
  assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

  # Splits the attention layers into three variables.
  if attention_paths_to_split is not None:
    for path, path_map in attention_paths_to_split.items():
      old_tensor = old_checkpoint[path]
      channels = old_tensor.shape[0] // 3

      target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

      num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

      old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
      query, key, value = old_tensor.split(channels // num_heads, dim=1)

      checkpoint[path_map["query"]] = query.reshape(target_shape)
      checkpoint[path_map["key"]] = key.reshape(target_shape)
      checkpoint[path_map["value"]] = value.reshape(target_shape)

  for path in paths:
    new_path = path["new"]

    # These have already been assigned
    if attention_paths_to_split is not None and new_path in attention_paths_to_split:
      continue

    # Global renaming happens here
    new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
    new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
    new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

    if additional_replacements is not None:
      for replacement in additional_replacements:
        new_path = new_path.replace(replacement["old"], replacement["new"])

    # proj_attn.weight has to be converted from conv 1D to linear
    if "proj_attn.weight" in new_path:
      checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
    else:
      checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
  keys = list(checkpoint.keys())
  attn_keys = ["query.weight", "key.weight", "value.weight"]
  for key in keys:
    if ".".join(key.split(".")[-2:]) in attn_keys:
      if checkpoint[key].ndim > 2:
        checkpoint[key] = checkpoint[key][:, :, 0, 0]
    elif "proj_attn.weight" in key:
      if checkpoint[key].ndim > 2:
        checkpoint[key] = checkpoint[key][:, :, 0]


def linear_transformer_to_conv(checkpoint):
  keys = list(checkpoint.keys())
  tf_keys = ["proj_in.weight", "proj_out.weight"]
  for key in keys:
    if ".".join(key.split(".")[-2:]) in tf_keys:
      if checkpoint[key].ndim == 2:
        checkpoint[key] = checkpoint[key].unsqueeze(2).unsqueeze(2)


def convert_ldm_unet_checkpoint(v2, checkpoint, config):
  """
  Takes a state dict and a config, and returns a converted checkpoint.
  """

  # extract state_dict for UNet
  unet_state_dict = {}
  unet_key = "model.diffusion_model."
  keys = list(checkpoint.keys())
  for key in keys:
    if key.startswith(unet_key):
      unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

  new_checkpoint = {}

  new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
  new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
  new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
  new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

  new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
  new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

  new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
  new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
  new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
  new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

  # Retrieves the keys for the input blocks only
  num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
  input_blocks = {
      layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
      for layer_id in range(num_input_blocks)
  }

  # Retrieves the keys for the middle blocks only
  num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
  middle_blocks = {
      layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
      for layer_id in range(num_middle_blocks)
  }

  # Retrieves the keys for the output blocks only
  num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
  output_blocks = {
      layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
      for layer_id in range(num_output_blocks)
  }

  for i in range(1, num_input_blocks):
    block_id = (i - 1) // (config["layers_per_block"] + 1)
    layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

    resnets = [
        key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
    ]
    attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

    if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
      new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
          f"input_blocks.{i}.0.op.weight"
      )
      new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
          f"input_blocks.{i}.0.op.bias"
      )

    paths = renew_resnet_paths(resnets)
    meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
    assign_to_checkpoint(
        paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    if len(attentions):
      paths = renew_attention_paths(attentions)
      meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
      assign_to_checkpoint(
          paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
      )

  resnet_0 = middle_blocks[0]
  attentions = middle_blocks[1]
  resnet_1 = middle_blocks[2]

  resnet_0_paths = renew_resnet_paths(resnet_0)
  assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

  resnet_1_paths = renew_resnet_paths(resnet_1)
  assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

  attentions_paths = renew_attention_paths(attentions)
  meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
  assign_to_checkpoint(
      attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
  )

  for i in range(num_output_blocks):
    block_id = i // (config["layers_per_block"] + 1)
    layer_in_block_id = i % (config["layers_per_block"] + 1)
    output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
    output_block_list = {}

    for layer in output_block_layers:
      layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
      if layer_id in output_block_list:
        output_block_list[layer_id].append(layer_name)
      else:
        output_block_list[layer_id] = [layer_name]

    if len(output_block_list) > 1:
      resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
      attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

      resnet_0_paths = renew_resnet_paths(resnets)
      paths = renew_resnet_paths(resnets)

      meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
      assign_to_checkpoint(
          paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
      )

      if ["conv.weight", "conv.bias"] in output_block_list.values():
        index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])
        new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
            f"output_blocks.{i}.{index}.conv.weight"
        ]
        new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
            f"output_blocks.{i}.{index}.conv.bias"
        ]

        # Clear attentions as they have been attributed above.
        if len(attentions) == 2:
          attentions = []

      if len(attentions):
        paths = renew_attention_paths(attentions)
        meta_path = {
            "old": f"output_blocks.{i}.1",
            "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
        }
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )
    else:
      resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
      for path in resnet_0_paths:
        old_path = ".".join(["output_blocks", str(i), path["old"]])
        new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

        new_checkpoint[new_path] = unet_state_dict[old_path]

  # SDのv2では1*1のconv2dがlinearに変わっているので、linear->convに変換する
  if v2:
    linear_transformer_to_conv(new_checkpoint)

  return new_checkpoint


def convert_ldm_vae_checkpoint(checkpoint, config):
  # extract state dict for VAE
  vae_state_dict = {}
  vae_key = "first_stage_model."
  keys = list(checkpoint.keys())
  for key in keys:
    if key.startswith(vae_key):
      vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

  new_checkpoint = {}

  new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
  new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
  new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
  new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
  new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
  new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

  new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
  new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
  new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
  new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
  new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
  new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

  new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
  new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
  new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
  new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

  # Retrieves the keys for the encoder down blocks only
  num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
  down_blocks = {
      layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
  }

  # Retrieves the keys for the decoder up blocks only
  num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
  up_blocks = {
      layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
  }

  for i in range(num_down_blocks):
    resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

    if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
      new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
          f"encoder.down.{i}.downsample.conv.weight"
      )
      new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
          f"encoder.down.{i}.downsample.conv.bias"
      )

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
  num_mid_res_blocks = 2
  for i in range(1, num_mid_res_blocks + 1):
    resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
  paths = renew_vae_attention_paths(mid_attentions)
  meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
  assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
  conv_attn_to_linear(new_checkpoint)

  for i in range(num_up_blocks):
    block_id = num_up_blocks - 1 - i
    resnets = [
        key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
    ]

    if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
      new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
          f"decoder.up.{block_id}.upsample.conv.weight"
      ]
      new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
          f"decoder.up.{block_id}.upsample.conv.bias"
      ]

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
  num_mid_res_blocks = 2
  for i in range(1, num_mid_res_blocks + 1):
    resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
  paths = renew_vae_attention_paths(mid_attentions)
  meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
  assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
  conv_attn_to_linear(new_checkpoint)
  return new_checkpoint


def create_unet_diffusers_config(v2):
  """
  Creates a config for the diffusers based on the config of the LDM model.
  """
  # unet_params = original_config.model.params.unet_config.params

  block_out_channels = [UNET_PARAMS_MODEL_CHANNELS * mult for mult in UNET_PARAMS_CHANNEL_MULT]

  down_block_types = []
  resolution = 1
  for i in range(len(block_out_channels)):
    block_type = "CrossAttnDownBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "DownBlock2D"
    down_block_types.append(block_type)
    if i != len(block_out_channels) - 1:
      resolution *= 2

  up_block_types = []
  for i in range(len(block_out_channels)):
    block_type = "CrossAttnUpBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "UpBlock2D"
    up_block_types.append(block_type)
    resolution //= 2

  config = dict(
      sample_size=UNET_PARAMS_IMAGE_SIZE,
      in_channels=UNET_PARAMS_IN_CHANNELS,
      out_channels=UNET_PARAMS_OUT_CHANNELS,
      down_block_types=tuple(down_block_types),
      up_block_types=tuple(up_block_types),
      block_out_channels=tuple(block_out_channels),
      layers_per_block=UNET_PARAMS_NUM_RES_BLOCKS,
      cross_attention_dim=UNET_PARAMS_CONTEXT_DIM if not v2 else V2_UNET_PARAMS_CONTEXT_DIM,
      attention_head_dim=UNET_PARAMS_NUM_HEADS if not v2 else V2_UNET_PARAMS_ATTENTION_HEAD_DIM,
  )

  return config


def create_vae_diffusers_config():
  """
  Creates a config for the diffusers based on the config of the LDM model.
  """
  # vae_params = original_config.model.params.first_stage_config.params.ddconfig
  # _ = original_config.model.params.first_stage_config.params.embed_dim
  block_out_channels = [VAE_PARAMS_CH * mult for mult in VAE_PARAMS_CH_MULT]
  down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
  up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

  config = dict(
      sample_size=VAE_PARAMS_RESOLUTION,
      in_channels=VAE_PARAMS_IN_CHANNELS,
      out_channels=VAE_PARAMS_OUT_CH,
      down_block_types=tuple(down_block_types),
      up_block_types=tuple(up_block_types),
      block_out_channels=tuple(block_out_channels),
      latent_channels=VAE_PARAMS_Z_CHANNELS,
      layers_per_block=VAE_PARAMS_NUM_RES_BLOCKS,
  )
  return config


def convert_ldm_clip_checkpoint_v1(checkpoint):
  keys = list(checkpoint.keys())
  text_model_dict = {}
  for key in keys:
    if key.startswith("cond_stage_model.transformer"):
      text_model_dict[key[len("cond_stage_model.transformer."):]] = checkpoint[key]
  return text_model_dict


def convert_ldm_clip_checkpoint_v2(checkpoint, max_length):
  # 嫌になるくらい違うぞ！
  def convert_key(key):
    if not key.startswith("cond_stage_model"):
      return None

    # common conversion
    key = key.replace("cond_stage_model.model.transformer.", "text_model.encoder.")
    key = key.replace("cond_stage_model.model.", "text_model.")

    if "resblocks" in key:
      # resblocks conversion
      key = key.replace(".resblocks.", ".layers.")
      if ".ln_" in key:
        key = key.replace(".ln_", ".layer_norm")
      elif ".mlp." in key:
        key = key.replace(".c_fc.", ".fc1.")
        key = key.replace(".c_proj.", ".fc2.")
      elif '.attn.out_proj' in key:
        key = key.replace(".attn.out_proj.", ".self_attn.out_proj.")
      elif '.attn.in_proj' in key:
        key = None                  # 特殊なので後で処理する
      else:
        raise ValueError(f"unexpected key in SD: {key}")
    elif '.positional_embedding' in key:
      key = key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
    elif '.text_projection' in key:
      key = None    # 使われない???
    elif '.logit_scale' in key:
      key = None    # 使われない???
    elif '.token_embedding' in key:
      key = key.replace(".token_embedding.weight", ".embeddings.token_embedding.weight")
    elif '.ln_final' in key:
      key = key.replace(".ln_final", ".final_layer_norm")
    return key

  keys = list(checkpoint.keys())
  new_sd = {}
  for key in keys:
    # remove resblocks 23
    if '.resblocks.23.' in key:
      continue
    new_key = convert_key(key)
    if new_key is None:
      continue
    new_sd[new_key] = checkpoint[key]

  # attnの変換
  for key in keys:
    if '.resblocks.23.' in key:
      continue
    if '.resblocks' in key and '.attn.in_proj_' in key:
      # 三つに分割
      values = torch.chunk(checkpoint[key], 3)

      key_suffix = ".weight" if "weight" in key else ".bias"
      key_pfx = key.replace("cond_stage_model.model.transformer.resblocks.", "text_model.encoder.layers.")
      key_pfx = key_pfx.replace("_weight", "")
      key_pfx = key_pfx.replace("_bias", "")
      key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
      new_sd[key_pfx + "q_proj" + key_suffix] = values[0]
      new_sd[key_pfx + "k_proj" + key_suffix] = values[1]
      new_sd[key_pfx + "v_proj" + key_suffix] = values[2]

  # position_idsの追加
  new_sd["text_model.embeddings.position_ids"] = torch.Tensor([list(range(max_length))]).to(torch.int64)
  return new_sd

# endregion


# region Diffusers->StableDiffusion の変換コード
# convert_diffusers_to_original_stable_diffusion をコピーしている（ASL 2.0）

def conv_transformer_to_linear(checkpoint):
  keys = list(checkpoint.keys())
  tf_keys = ["proj_in.weight", "proj_out.weight"]
  for key in keys:
    if ".".join(key.split(".")[-2:]) in tf_keys:
      if checkpoint[key].ndim > 2:
        checkpoint[key] = checkpoint[key][:, :, 0, 0]


def convert_unet_state_dict_to_sd(v2, unet_state_dict):
  unet_conversion_map = [
      # (stable-diffusion, HF Diffusers)
      ("time_embed.0.weight", "time_embedding.linear_1.weight"),
      ("time_embed.0.bias", "time_embedding.linear_1.bias"),
      ("time_embed.2.weight", "time_embedding.linear_2.weight"),
      ("time_embed.2.bias", "time_embedding.linear_2.bias"),
      ("input_blocks.0.0.weight", "conv_in.weight"),
      ("input_blocks.0.0.bias", "conv_in.bias"),
      ("out.0.weight", "conv_norm_out.weight"),
      ("out.0.bias", "conv_norm_out.bias"),
      ("out.2.weight", "conv_out.weight"),
      ("out.2.bias", "conv_out.bias"),
  ]

  unet_conversion_map_resnet = [
      # (stable-diffusion, HF Diffusers)
      ("in_layers.0", "norm1"),
      ("in_layers.2", "conv1"),
      ("out_layers.0", "norm2"),
      ("out_layers.3", "conv2"),
      ("emb_layers.1", "time_emb_proj"),
      ("skip_connection", "conv_shortcut"),
  ]

  unet_conversion_map_layer = []
  for i in range(4):
      # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
      hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
      sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
      unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

      if i < 3:
        # no attention layers in down_blocks.3
        hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
        sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
        unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(3):
      # loop over resnets/attentions for upblocks
      hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
      sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
      unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

      if i > 0:
        # no attention layers in up_blocks.0
        hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
        sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
        unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
      # no downsample in down_blocks.3
      hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
      sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
      unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

      # no upsample in up_blocks.3
      hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
      sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
      unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

  hf_mid_atn_prefix = "mid_block.attentions.0."
  sd_mid_atn_prefix = "middle_block.1."
  unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

  for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

  # buyer beware: this is a *brittle* function,
  # and correct output requires that all of these pieces interact in
  # the exact order in which I have arranged them.
  mapping = {k: k for k in unet_state_dict.keys()}
  for sd_name, hf_name in unet_conversion_map:
    mapping[hf_name] = sd_name
  for k, v in mapping.items():
    if "resnets" in k:
      for sd_part, hf_part in unet_conversion_map_resnet:
        v = v.replace(hf_part, sd_part)
      mapping[k] = v
  for k, v in mapping.items():
    for sd_part, hf_part in unet_conversion_map_layer:
      v = v.replace(hf_part, sd_part)
    mapping[k] = v
  new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}

  if v2:
    conv_transformer_to_linear(new_state_dict)

  return new_state_dict

# endregion


def load_checkpoint_with_text_encoder_conversion(ckpt_path):
  # text encoderの格納形式が違うモデルに対応する ('text_model'がない)
  TEXT_ENCODER_KEY_REPLACEMENTS = [
      ('cond_stage_model.transformer.embeddings.', 'cond_stage_model.transformer.text_model.embeddings.'),
      ('cond_stage_model.transformer.encoder.', 'cond_stage_model.transformer.text_model.encoder.'),
      ('cond_stage_model.transformer.final_layer_norm.', 'cond_stage_model.transformer.text_model.final_layer_norm.')
  ]

  checkpoint = torch.load(ckpt_path, map_location="cpu")
  state_dict = checkpoint["state_dict"]

  key_reps = []
  for rep_from, rep_to in TEXT_ENCODER_KEY_REPLACEMENTS:
    for key in state_dict.keys():
      if key.startswith(rep_from):
        new_key = rep_to + key[len(rep_from):]
        key_reps.append((key, new_key))

  for key, new_key in key_reps:
    state_dict[new_key] = state_dict[key]
    del state_dict[key]

  return checkpoint


def load_models_from_stable_diffusion_checkpoint(v2, ckpt_path, dtype=None):
  checkpoint = load_checkpoint_with_text_encoder_conversion(ckpt_path)
  state_dict = checkpoint["state_dict"]
  if dtype is not None:
    for k, v in state_dict.items():
      if type(v) is torch.Tensor:
        state_dict[k] = v.to(dtype)

  # Convert the UNet2DConditionModel model.
  unet_config = create_unet_diffusers_config(v2)
  converted_unet_checkpoint = convert_ldm_unet_checkpoint(v2, state_dict, unet_config)

  unet = UNet2DConditionModel(**unet_config)
  info = unet.load_state_dict(converted_unet_checkpoint)
  print("loading u-net:", info)

  # Convert the VAE model.
  vae_config = create_vae_diffusers_config()
  converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)

  vae = AutoencoderKL(**vae_config)
  info = vae.load_state_dict(converted_vae_checkpoint)
  print("loadint vae:", info)

  # convert text_model
  if v2:
    converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v2(state_dict, 77)
    cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=23,
        num_attention_heads=16,
        max_position_embeddings=77,
        hidden_act="gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=512,
        torch_dtype="float32",
        transformers_version="4.25.0.dev0",
    )
    text_model = CLIPTextModel._from_config(cfg)
    info = text_model.load_state_dict(converted_text_encoder_checkpoint)
  else:
    converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v1(state_dict)
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    info = text_model.load_state_dict(converted_text_encoder_checkpoint)
  print("loading text encoder:", info)

  return text_model, vae, unet


def convert_text_encoder_state_dict_to_sd_v2(checkpoint):
  def convert_key(key):
    # position_idsの除去
    if ".position_ids" in key:
      return None

    # common
    key = key.replace("text_model.encoder.", "transformer.")
    key = key.replace("text_model.", "")
    if "layers" in key:
      # resblocks conversion
      key = key.replace(".layers.", ".resblocks.")
      if ".layer_norm" in key:
        key = key.replace(".layer_norm", ".ln_")
      elif ".mlp." in key:
        key = key.replace(".fc1.", ".c_fc.")
        key = key.replace(".fc2.", ".c_proj.")
      elif '.self_attn.out_proj' in key:
        key = key.replace(".self_attn.out_proj.", ".attn.out_proj.")
      elif '.self_attn.' in key:
        key = None                  # 特殊なので後で処理する
      else:
        raise ValueError(f"unexpected key in DiffUsers model: {key}")
    elif '.position_embedding' in key:
      key = key.replace("embeddings.position_embedding.weight", "positional_embedding")
    elif '.token_embedding' in key:
      key = key.replace("embeddings.token_embedding.weight", "token_embedding.weight")
    elif 'final_layer_norm' in key:
      key = key.replace("final_layer_norm", "ln_final")
    return key

  keys = list(checkpoint.keys())
  new_sd = {}
  for key in keys:
    new_key = convert_key(key)
    if new_key is None:
      continue
    new_sd[new_key] = checkpoint[key]

  # attnの変換
  for key in keys:
    if 'layers' in key and 'q_proj' in key:
      # 三つを結合
      key_q = key
      key_k = key.replace("q_proj", "k_proj")
      key_v = key.replace("q_proj", "v_proj")

      value_q = checkpoint[key_q]
      value_k = checkpoint[key_k]
      value_v = checkpoint[key_v]
      value = torch.cat([value_q, value_k, value_v])

      new_key = key.replace("text_model.encoder.layers.", "transformer.resblocks.")
      new_key = new_key.replace(".self_attn.q_proj.", ".attn.in_proj_")
      new_sd[new_key] = value

  return new_sd


def save_stable_diffusion_checkpoint(v2, output_file, text_encoder, unet, ckpt_path, epochs, steps, save_dtype=None):
  # VAEがメモリ上にないので、もう一度VAEを含めて読み込む
  checkpoint = load_checkpoint_with_text_encoder_conversion(ckpt_path)
  state_dict = checkpoint["state_dict"]

  def assign_new_sd(prefix, sd):
    for k, v in sd.items():
      key = prefix + k
      assert key in state_dict, f"Illegal key in save SD: {key}"
      if save_dtype is not None:
        v = v.detach().clone().to("cpu").to(save_dtype)
      state_dict[key] = v

  # Convert the UNet model
  unet_state_dict = convert_unet_state_dict_to_sd(v2, unet.state_dict())
  assign_new_sd("model.diffusion_model.", unet_state_dict)

  # Convert the text encoder model
  if v2:
    text_enc_dict = convert_text_encoder_state_dict_to_sd_v2(text_encoder.state_dict())
    assign_new_sd("cond_stage_model.model.", text_enc_dict)
  else:
    text_enc_dict = text_encoder.state_dict()
    assign_new_sd("cond_stage_model.transformer.", text_enc_dict)

  # Put together new checkpoint
  new_ckpt = {'state_dict': state_dict}

  if 'epoch' in checkpoint:
    epochs += checkpoint['epoch']
  if 'global_step' in checkpoint:
    steps += checkpoint['global_step']

  new_ckpt['epoch'] = epochs
  new_ckpt['global_step'] = steps

  torch.save(new_ckpt, output_file)


def save_diffusers_checkpoint(v2, output_dir, text_encoder, unet, pretrained_model_name_or_path, save_dtype):
  pipeline = StableDiffusionPipeline(
      unet=unet,
      text_encoder=text_encoder,
      vae=AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae"),
      scheduler=DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler"),
      tokenizer=CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer"),
      safety_checker=None,
      feature_extractor=None,
      requires_safety_checker=None,
  )
  pipeline.save_pretrained(output_dir)

# endregion


def collate_fn(examples):
  return examples[0]


def train(args):
  if args.caption_extention is not None:
    args.caption_extension = args.caption_extention
    args.caption_extention = None

  fine_tuning = args.fine_tuning
  cache_latents = args.cache_latents

  # latentsをキャッシュする場合のオプション設定を確認する
  if cache_latents:
    assert not args.flip_aug and not args.color_aug, "when caching latents, augmentation cannot be used / latentをキャッシュするときはaugmentationは使えません"

  # その他のオプション設定を確認する
  if args.v_parameterization and not args.v2:
    print("v_parameterization should be with v2 / v1でv_parameterizationを使用することは想定されていません")
  if args.v2 and args.clip_skip is not None:
    print("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

  # モデル形式のオプション設定を確認する：
  # v11からDiffUsersから直接落としてくるのもOK（ただし認証がいるやつは未対応）、またv11からDiffUsersも途中保存に対応した
  use_stable_diffusion_format = os.path.isfile(args.pretrained_model_name_or_path)

  # 乱数系列を初期化する
  if args.seed is not None:
    set_seed(args.seed)

  # 学習データを用意する
  def read_caption(img_path):
    # captionの候補ファイル名を作る
    base_name = os.path.splitext(img_path)[0]
    base_name_face_det = base_name
    tokens = base_name.split("_")
    if len(tokens) >= 5:
      base_name_face_det = "_".join(tokens[:-4])
    cap_paths = [base_name + args.caption_extension, base_name_face_det + args.caption_extension]

    caption = None
    for cap_path in cap_paths:
      if os.path.isfile(cap_path):
        with open(cap_path, "rt", encoding='utf-8') as f:
          caption = f.readlines()[0].strip()
        break
    return caption

  def load_dreambooth_dir(dir):
    tokens = os.path.basename(dir).split('_')
    try:
      n_repeats = int(tokens[0])
    except ValueError as e:
      return 0, []

    caption_by_folder = '_'.join(tokens[1:])

    print(f"found directory {n_repeats}_{caption_by_folder}")

    img_paths = glob.glob(os.path.join(dir, "*.png")) + glob.glob(os.path.join(dir, "*.jpg")) + \
        glob.glob(os.path.join(dir, "*.webp"))

    # 画像ファイルごとにプロンプトを読み込み、もしあればそちらを使う（v11から仕様変更した）
    captions = []
    for img_path in img_paths:
      cap_for_img = read_caption(img_path)
      captions.append(caption_by_folder if cap_for_img is None else cap_for_img)

    return n_repeats, list(zip(img_paths, captions))

  print("prepare train images.")
  train_img_path_captions = []

  if fine_tuning:
    img_paths = glob.glob(os.path.join(args.train_data_dir, "*.png")) + \
        glob.glob(os.path.join(args.train_data_dir, "*.jpg")) + glob.glob(os.path.join(args.train_data_dir, "*.webp"))
    for img_path in tqdm(img_paths):
      caption = read_caption(img_path)
      assert caption is not None and len(
          caption) > 0, f"no caption for image. check caption_extension option / キャプションファイルが見つからないかcaptionが空です。caption_extensionオプションを確認してください: {img_path}"

      train_img_path_captions.append((img_path, caption))

    if args.dataset_repeats is not None:
      l = []
      for _ in range(args.dataset_repeats):
        l.extend(train_img_path_captions)
      train_img_path_captions = l
  else:
    train_dirs = os.listdir(args.train_data_dir)
    for dir in train_dirs:
      n_repeats, img_caps = load_dreambooth_dir(os.path.join(args.train_data_dir, dir))
      for _ in range(n_repeats):
        train_img_path_captions.extend(img_caps)
  print(f"{len(train_img_path_captions)} train images with repeating.")

  reg_img_path_captions = []
  if args.reg_data_dir:
    print("prepare reg images.")
    reg_dirs = os.listdir(args.reg_data_dir)
    for dir in reg_dirs:
      n_repeats, img_caps = load_dreambooth_dir(os.path.join(args.reg_data_dir, dir))
      for _ in range(n_repeats):
        reg_img_path_captions.extend(img_caps)
    print(f"{len(reg_img_path_captions)} reg images.")

  # データセットを準備する
  resolution = tuple([int(r) for r in args.resolution.split(',')])
  if len(resolution) == 1:
    resolution = (resolution[0], resolution[0])
  assert len(resolution) == 2, \
      f"resolution must be 'size' or 'width,height' / resolutionは'サイズ'または'幅','高さ'で指定してください: {args.resolution}"

  if args.enable_bucket:
    assert min(resolution) >= args.min_bucket_reso, f"min_bucket_reso must be equal or greater than resolution / min_bucket_resoは解像度の数値以上で指定してください"
    assert max(resolution) <= args.max_bucket_reso, f"max_bucket_reso must be equal or less than resolution / max_bucket_resoは解像度の数値以下で指定してください"

  if args.face_crop_aug_range is not None:
    face_crop_aug_range = tuple([float(r) for r in args.face_crop_aug_range.split(',')])
    assert len(
        face_crop_aug_range) == 2, f"face_crop_aug_range must be two floats / face_crop_aug_rangeは'下限,上限'で指定してください: {args.face_crop_aug_range}"
  else:
    face_crop_aug_range = None

  # tokenizerを読み込む
  print("prepare tokenizer")
  if args.v2:
    tokenizer = CLIPTokenizer.from_pretrained(V2_STABLE_DIFFUSION_PATH, subfolder="tokenizer")
  else:
    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)

  print("prepare dataset")
  train_dataset = DreamBoothOrFineTuningDataset(args.train_batch_size, fine_tuning, train_img_path_captions, reg_img_path_captions, tokenizer, resolution,
                                                args.prior_loss_weight, args.flip_aug, args.color_aug, face_crop_aug_range, args.random_crop,
                                                args.shuffle_caption, args.no_token_padding, args.debug_dataset)

  if args.debug_dataset:
    train_dataset.make_buckets_with_caching(args.enable_bucket, None, args.min_bucket_reso,
                                            args.max_bucket_reso)  # デバッグ用にcacheなしで作る
    print(f"Total dataset length (steps) / データセットの長さ（ステップ数）: {len(train_dataset)}")
    print("Escape for exit. / Escキーで中断、終了します")
    for example in train_dataset:
      for im, cap, lw in zip(example['images'], example['captions'], example['loss_weights']):
        im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))                # c,H,W -> H,W,c
        im = im[:, :, ::-1]                             # RGB -> BGR (OpenCV)
        print(f'size: {im.shape[1]}*{im.shape[0]}, caption: "{cap}", loss weight: {lw}')
        cv2.imshow("img", im)
        k = cv2.waitKey()
        cv2.destroyAllWindows()
        if k == 27:
          break
      if k == 27:
        break
    return

  # acceleratorを準備する
  # gradient accumulationは複数モデルを学習する場合には対応していないとのことなので、1固定にする
  print("prepare accelerator")
  if args.logging_dir is None:
    log_with = None
    logging_dir = None
  else:
    log_with = "tensorboard"
    logging_dir = args.logging_dir + "/" + time.strftime('%Y%m%d%H%M%S', time.localtime())
  accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision=args.mixed_precision,
                            log_with=log_with, logging_dir=logging_dir)

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
  if use_stable_diffusion_format:
    print("load StableDiffusion checkpoint")
    text_encoder, vae, unet = load_models_from_stable_diffusion_checkpoint(args.v2, args.pretrained_model_name_or_path)
  else:
    print("load Diffusers pretrained models")
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, tokenizer=None, safety_checker=None)
    # , torch_dtype=weight_dtype) ここでtorch_dtypeを指定すると学習時にエラーになる
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    del pipe

  # モデルに xformers とか memory efficient attention を組み込む
  replace_unet_modules(unet, args.mem_eff_attn, args.xformers)

  # 学習を準備する
  if cache_latents:
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.eval()
    with torch.no_grad():
      train_dataset.make_buckets_with_caching(args.enable_bucket, vae, args.min_bucket_reso, args.max_bucket_reso)
    del vae
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  else:
    train_dataset.make_buckets_with_caching(args.enable_bucket, None, args.min_bucket_reso, args.max_bucket_reso)
    vae.requires_grad_(False)
    vae.eval()

  unet.requires_grad_(True)                   # 念のため追加
  text_encoder.requires_grad_(True)

  if args.gradient_checkpointing:
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()

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

  trainable_params = (itertools.chain(unet.parameters(), text_encoder.parameters()))

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

  # acceleratorがなんかよろしくやってくれるらしい
  unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
      unet, text_encoder, optimizer, train_dataloader, lr_scheduler)

  if not cache_latents:
    vae.to(accelerator.device, dtype=weight_dtype)

  # resumeする
  if args.resume is not None:
    print(f"resume training from state: {args.resume}")
    accelerator.load_state(args.resume)

  # epoch数を計算する
  num_train_epochs = math.ceil(args.max_train_steps / len(train_dataloader))

  # 学習する
  total_batch_size = args.train_batch_size  # * accelerator.num_processes
  print("running training / 学習開始")
  print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset.num_train_images}")
  print(f"  num reg images / 正則化画像の数: {train_dataset.num_reg_images}")
  print(f"  num examples / サンプル数: {train_dataset.num_train_images * (2 if train_dataset.enable_reg_images else 1)}")
  print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
  print(f"  num epochs / epoch数: {num_train_epochs}")
  print(f"  batch size per device / バッチサイズ: {args.train_batch_size}")
  print(f"  total train batch size (with parallel & distributed) / 総バッチサイズ（並列学習含む）: {total_batch_size}")
  print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

  progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
  global_step = 0

  # v12で更新：clip_sample=Falseに
  # Diffusersのtrain_dreambooth.pyがconfigから持ってくるように変更されたので、clip_sample=Falseになるため、それに合わせる
  # 既存の1.4/1.5/2.0はすべてschdulerのconfigは（クラス名を除いて）同じ
  # よくソースを見たら学習時は関係ないや(;'∀')　
  noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                  num_train_timesteps=1000, clip_sample=False)

  if accelerator.is_main_process:
    accelerator.init_trackers("dreambooth")

  # 以下 train_dreambooth.py からほぼコピペ
  for epoch in range(num_train_epochs):
    print(f"epoch {epoch+1}/{num_train_epochs}")

    # 指定したステップ数までText Encoderを学習する：epoch最初の状態
    train_text_encoder = args.stop_text_encoder_training is None or global_step < args.stop_text_encoder_training 
    unet.train()
    if train_text_encoder:
      text_encoder.train()

    loss_total = 0
    for step, batch in enumerate(train_dataloader):
      # 指定したステップ数でText Encoderの学習を止める
      stop_text_encoder_training = args.stop_text_encoder_training is not None and global_step == args.stop_text_encoder_training 
      if stop_text_encoder_training:
        print(f"stop text encoder training at step {global_step}")
        text_encoder.train(False)

      with accelerator.accumulate(unet):
        with torch.no_grad():
          # latentに変換
          if cache_latents:
            latents = batch["latents"].to(accelerator.device)
          else:
            latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
          latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, device=latents.device)
        b_size = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        if args.clip_skip is None:
          encoder_hidden_states = text_encoder(batch["input_ids"])[0]
        else:
          enc_out = text_encoder(batch["input_ids"], output_hidden_states=True, return_dict=True)
          encoder_hidden_states = enc_out['hidden_states'][-args.clip_skip]
          encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

        # Predict the noise residual
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if args.v_parameterization:
          # v-parameterization training
          # こうしたい：
          # target = noise_scheduler.get_v(latents, noise, timesteps)

          # StabilityAiのddpm.pyのコード：
          # elif self.parameterization == "v":
          #     target = self.get_v(x_start, noise, t)
          # ...
          # def get_v(self, x, noise, t):
          #   return (
          #           extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
          #           extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
          #   )

          # scheduling_ddim.pyのコード：
          # elif self.config.prediction_type == "v_prediction":
          #     pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
          #     # predict V
          #     model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample

          # これでいいかな？：
          alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps]
          beta_prod_t = 1 - alpha_prod_t
          alpha_prod_t = torch.reshape(alpha_prod_t, (len(alpha_prod_t), 1, 1, 1))    # broadcastされないらしいのでreshape
          beta_prod_t = torch.reshape(beta_prod_t, (len(beta_prod_t), 1, 1, 1))
          target = (alpha_prod_t ** 0.5) * noise - (beta_prod_t ** 0.5) * latents
        else:
          target = noise

        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
        loss = loss.mean([1, 2, 3])

        loss_weights = batch["loss_weights"]                      # 各sampleごとのweight
        loss = loss * loss_weights

        loss = loss.mean()                # 平均なのでbatch_sizeで割る必要なし

        accelerator.backward(loss)
        if accelerator.sync_gradients:
          params_to_clip = (itertools.chain(unet.parameters(), text_encoder.parameters()))
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
        if use_stable_diffusion_format:
          os.makedirs(args.output_dir, exist_ok=True)
          ckpt_file = os.path.join(args.output_dir, EPOCH_CHECKPOINT_NAME.format(epoch + 1))
          save_stable_diffusion_checkpoint(args.v2, ckpt_file, accelerator.unwrap_model(text_encoder), accelerator.unwrap_model(unet),
                                           args.pretrained_model_name_or_path, epoch + 1, global_step, save_dtype)
        else:
          out_dir = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(epoch + 1))
          os.makedirs(out_dir, exist_ok=True)
          save_diffusers_checkpoint(args.v2, out_dir, accelerator.unwrap_model(text_encoder),
                                    accelerator.unwrap_model(unet), args.pretrained_model_name_or_path, save_dtype)

        if args.save_state:
          print("saving state.")
          accelerator.save_state(os.path.join(args.output_dir, EPOCH_STATE_NAME.format(epoch + 1)))

  is_main_process = accelerator.is_main_process
  if is_main_process:
    unet = accelerator.unwrap_model(unet)
    text_encoder = accelerator.unwrap_model(text_encoder)

  accelerator.end_training()

  if args.save_state:
    print("saving last state.")
    accelerator.save_state(os.path.join(args.output_dir, LAST_STATE_NAME))

  del accelerator                         # この後メモリを使うのでこれは消す

  if is_main_process:
    os.makedirs(args.output_dir, exist_ok=True)
    if use_stable_diffusion_format:
      ckpt_file = os.path.join(args.output_dir, LAST_CHECKPOINT_NAME)
      print(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
      save_stable_diffusion_checkpoint(args.v2, ckpt_file, text_encoder, unet,
                                       args.pretrained_model_name_or_path, epoch, global_step, save_dtype)
    else:
      # Create the pipeline using using the trained modules and save it.
      print(f"save trained model as Diffusers to {args.output_dir}")
      out_dir = os.path.join(args.output_dir, LAST_DIFFUSERS_DIR_NAME)
      os.makedirs(out_dir, exist_ok=True)
      save_diffusers_checkpoint(args.v2, out_dir, text_encoder, unet, args.pretrained_model_name_or_path, save_dtype)
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
  parser.add_argument("--fine_tuning", action="store_true",
                      help="fine tune the model instead of DreamBooth / DreamBoothではなくfine tuningする")
  parser.add_argument("--shuffle_caption", action="store_true",
                      help="shuffle comma-separated caption / コンマで区切られたcaptionの各要素をshuffleする")
  parser.add_argument("--caption_extention", type=str, default=None,
                      help="extension of caption files (backward compatiblity) / 読み込むcaptionファイルの拡張子（スペルミスを残してあります）")
  parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption files / 読み込むcaptionファイルの拡張子")
  parser.add_argument("--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("--reg_data_dir", type=str, default=None, help="directory for regularization images / 正則化画像データのディレクトリ")
  parser.add_argument("--dataset_repeats", type=int, default=None,
                      help="repeat dataset in fine tuning / fine tuning時にデータセットを繰り返す回数")
  parser.add_argument("--output_dir", type=str, default=None,
                      help="directory to output trained model (default format is same to input) / 学習後のモデル出力先ディレクトリ（デフォルトの保存形式は読み込んだ形式と同じ）")
  # parser.add_argument("--save_as_sd", action='store_true',
  #                     help="save the model as StableDiffusion checkpoint / 学習後のモデルをStableDiffusionのcheckpointとして保存する")
  parser.add_argument("--save_every_n_epochs", type=int, default=None,
                      help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存します")
  parser.add_argument("--save_state", action="store_true",
                      help="save training state additionally (including optimizer states etc.) / optimizerなど学習状態も含めたstateを追加で保存する")
  parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")
  parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="loss weight for regularization images / 正則化画像のlossの重み")
  parser.add_argument("--no_token_padding", action="store_true",
                      help="disable token padding (same as Diffuser's DreamBooth) / トークンのpaddingを無効にする（Diffusers版DreamBoothと同じ動作）")
  parser.add_argument("--stop_text_encoder_training", type=int, default=None, help="steps to stop text encoder training / Text Encoderの学習を止めるステップ数")
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
  parser.add_argument("--train_batch_size", type=int, default=1,
                      help="batch size for training (1 means one train or reg data, not train/reg pair) / 学習時のバッチサイズ（1でtrain/regをそれぞれ1件ずつ学習）")
  parser.add_argument("--use_8bit_adam", action="store_true",
                      help="use 8bit Adam optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）")
  parser.add_argument("--mem_eff_attn", action="store_true",
                      help="use memory efficient attention for CrossAttention / CrossAttentionに省メモリ版attentionを使う")
  parser.add_argument("--xformers", action="store_true",
                      help="use xformers for CrossAttention / CrossAttentionにxformersを使う")
  parser.add_argument("--cache_latents", action="store_true",
                      help="cache latents to reduce memory (augmentations must be disabled) / メモリ削減のためにlatentをcacheする（augmentationは使用不可）")
  parser.add_argument("--enable_bucket", action="store_true",
                      help="enable buckets for multi aspect ratio training / 複数解像度学習のためのbucketを有効にする")
  parser.add_argument("--min_bucket_reso", type=int, default=256, help="minimum resolution for buckets / bucketの最小解像度")
  parser.add_argument("--max_bucket_reso", type=int, default=1024, help="maximum resolution for buckets / bucketの最小解像度")
  parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
  parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
  parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
  parser.add_argument("--gradient_checkpointing", action="store_true",
                      help="enable gradient checkpointing / grandient checkpointingを有効にする")
  parser.add_argument("--mixed_precision", type=str, default="no",
                      choices=["no", "fp16", "bf16"], help="use mixed precision / 混合精度を使う場合、その精度")
  parser.add_argument("--save_precision", type=str, default=None,
                      choices=[None, "float", "fp16", "bf16"], help="precision in saving (available in StableDiffusion checkpoint) / 保存時に精度を変更して保存する（StableDiffusion形式での保存時のみ有効）")
  parser.add_argument("--clip_skip", type=int, default=None,
                      help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）")
  parser.add_argument("--logging_dir", type=str, default=None,
                      help="enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する")
  parser.add_argument("--lr_scheduler", type=str, default="constant",
                      help="scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup")
  parser.add_argument("--lr_warmup_steps", type=int, default=0,
                      help="Number of steps for the warmup in the lr scheduler (default is 0) / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）")

  args = parser.parse_args()
  train(args)
