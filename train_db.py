# DreamBooth training

import gc
import time
from torch.autograd.function import Function
import argparse
import glob
import itertools
import math
import os
import random
import shutil

from tqdm import tqdm
import torch
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPTokenizer
import diffusers
from diffusers import DDPMScheduler, StableDiffusionPipeline
import albumentations as albu
import numpy as np
from PIL import Image
import cv2
from einops import rearrange
from torch import einsum

import library.model_util as model_util
import library.train_util as train_util
from library.train_util import DreamBoothDataset, FineTuningDataset


# region dataset

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
      bucket_resos, bucket_aspect_ratios = model_util.make_bucket_resolutions((self.width, self.height), min_size, max_size)
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
          lines = f.readlines()
          assert len(lines) > 0, f"caption file is empty / キャプションファイルが空です: {cap_path}"
          caption = lines[0].strip()
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
    tokenizer = CLIPTokenizer.from_pretrained(train_util.V2_STABLE_DIFFUSION_PATH, subfolder="tokenizer")
  else:
    tokenizer = CLIPTokenizer.from_pretrained(train_util.TOKENIZER_PATH)

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
    log_prefix = "" if args.log_prefix is None else args.log_prefix
    logging_dir = args.logging_dir + "/" + log_prefix + time.strftime('%Y%m%d%H%M%S', time.localtime())
  accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision=args.mixed_precision,
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
    # , torch_dtype=weight_dtype) ここでtorch_dtypeを指定すると学習時にエラーになる
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    del pipe

  # # 置換するCLIPを読み込む
  # if args.replace_clip_l14_336:
  #   text_encoder = load_clip_l14_336(weight_dtype)
  #   print(f"large clip {CLIP_ID_L14_336} is loaded")

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
      train_dataset.make_buckets_with_caching(args.enable_bucket, vae, args.min_bucket_reso, args.max_bucket_reso)
    vae.to("cpu")
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    gc.collect()
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

  # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
  if args.full_fp16:
    assert args.mixed_precision == "fp16", "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
    print("enable full fp16 training.")
    unet.to(weight_dtype)
    text_encoder.to(weight_dtype)

  # acceleratorがなんかよろしくやってくれるらしい
  unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
      unet, text_encoder, optimizer, train_dataloader, lr_scheduler)

  if not cache_latents:
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
  # 既存の1.4/1.5/2.0/2.1はすべてschedulerのconfigは（クラス名を除いて）同じ
  # よくソースを見たら学習時はclip_sampleは関係ないや(;'∀')　
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
        text_encoder.requires_grad_(False)

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
        if save_stable_diffusion_format:
          os.makedirs(args.output_dir, exist_ok=True)
          ckpt_file = os.path.join(args.output_dir, model_util.get_epoch_ckpt_name(use_safetensors, epoch + 1))
          model_util.save_stable_diffusion_checkpoint(args.v2, ckpt_file, unwrap_model(text_encoder), unwrap_model(unet),
                                                      src_stable_diffusion_ckpt, epoch + 1, global_step, save_dtype, vae)
          if args.save_last_n_epochs is not None:
            old_ckpt_file = os.path.join(args.output_dir, model_util.get_epoch_ckpt_name(use_safetensors, epoch + 1 - args.save_every_n_epochs * args.save_last_n_epochs))
            if os.path.exists(old_ckpt_file):
              os.remove(old_ckpt_file)
        else:
          out_dir = os.path.join(args.output_dir, train_util.EPOCH_DIFFUSERS_DIR_NAME.format(epoch + 1))
          os.makedirs(out_dir, exist_ok=True)
          model_util.save_diffusers_checkpoint(args.v2, out_dir, unwrap_model(text_encoder),
                                               unwrap_model(unet), src_diffusers_model_path,
                                               use_safetensors=use_safetensors)
          if args.save_last_n_epochs is not None:
            out_dir_old = os.path.join(args.output_dir, train_util.EPOCH_DIFFUSERS_DIR_NAME.format(epoch + 1 - args.save_every_n_epochs * args.save_last_n_epochs))
            if os.path.exists(out_dir_old):
              shutil.rmtree(out_dir_old)

        if args.save_state:
          print("saving state.")
          accelerator.save_state(os.path.join(args.output_dir, train_util.EPOCH_STATE_NAME.format(epoch + 1)))
          if args.save_last_n_epochs is not None:
            state_dir_old = os.path.join(args.output_dir, train_util.EPOCH_STATE_NAME.format(epoch + 1 - args.save_every_n_epochs * args.save_last_n_epochs))
            if os.path.exists(state_dir_old):
              shutil.rmtree(state_dir_old)

  is_main_process = accelerator.is_main_process
  if is_main_process:
    unet = unwrap_model(unet)
    text_encoder = unwrap_model(text_encoder)

  accelerator.end_training()

  if args.save_state:
    print("saving last state.")
    accelerator.save_state(os.path.join(args.output_dir, train_util.LAST_STATE_NAME))

  del accelerator                         # この後メモリを使うのでこれは消す

  if is_main_process:
    os.makedirs(args.output_dir, exist_ok=True)
    if save_stable_diffusion_format:
      ckpt_file = os.path.join(args.output_dir, model_util.get_last_ckpt_name(use_safetensors))
      print(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
      model_util.save_stable_diffusion_checkpoint(args.v2, ckpt_file, text_encoder, unet,
                                                  src_stable_diffusion_ckpt, epoch, global_step, save_dtype, vae)
    else:
      print(f"save trained model as Diffusers to {args.output_dir}")
      out_dir = os.path.join(args.output_dir, train_util.LAST_DIFFUSERS_DIR_NAME)
      os.makedirs(out_dir, exist_ok=True)
      model_util.save_diffusers_checkpoint(args.v2, out_dir, text_encoder, unet, src_diffusers_model_path,
                                           use_safetensors=use_safetensors)
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
  # parser.add_argument("--replace_clip_l14_336", action='store_true',
  #                     help="Replace CLIP (Text Encoder) to l/14@336 / CLIP(Text Encoder)をl/14@336に入れ替える")
  parser.add_argument("--fine_tuning", action="store_true",
                      help="fine tune the model instead of DreamBooth / DreamBoothではなくfine tuningする")
  parser.add_argument("--shuffle_caption", action="store_true",
                      help="shuffle comma-separated caption / コンマで区切られたcaptionの各要素をshuffleする")
  parser.add_argument("--caption_extention", type=str, default=None,
                      help="extension of caption files (backward compatibility) / 読み込むcaptionファイルの拡張子（スペルミスを残してあります）")
  parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption files / 読み込むcaptionファイルの拡張子")
  parser.add_argument("--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("--reg_data_dir", type=str, default=None, help="directory for regularization images / 正則化画像データのディレクトリ")
  parser.add_argument("--dataset_repeats", type=int, default=None,
                      help="repeat dataset in fine tuning / fine tuning時にデータセットを繰り返す回数")
  parser.add_argument("--output_dir", type=str, default=None,
                      help="directory to output trained model / 学習後のモデル出力先ディレクトリ")
  parser.add_argument("--save_precision", type=str, default=None,
                      choices=[None, "float", "fp16", "bf16"], help="precision in saving (available in StableDiffusion checkpoint) / 保存時に精度を変更して保存する（StableDiffusion形式での保存時のみ有効）")
  parser.add_argument("--save_model_as", type=str, default=None, choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
                      help="format to save the model (default is same to original) / モデル保存時の形式（未指定時は元モデルと同じ）")
  parser.add_argument("--use_safetensors", action='store_true',
                      help="use safetensors format to save (if save_model_as is not specified) / checkpoint、モデルをsafetensors形式で保存する（save_model_as未指定時）")
  parser.add_argument("--save_every_n_epochs", type=int, default=None,
                      help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する")
  parser.add_argument("--save_last_n_epochs", type=int, default=None,
                      help="save last N checkpoints / 最大Nエポック保存する")
  parser.add_argument("--save_state", action="store_true",
                      help="save training state additionally (including optimizer states etc.) / optimizerなど学習状態も含めたstateを追加で保存する")
  parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")
  parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="loss weight for regularization images / 正則化画像のlossの重み")
  parser.add_argument("--no_token_padding", action="store_true",
                      help="disable token padding (same as Diffuser's DreamBooth) / トークンのpaddingを無効にする（Diffusers版DreamBoothと同じ動作）")
  parser.add_argument("--stop_text_encoder_training", type=int, default=None,
                      help="steps to stop text encoder training / Text Encoderの学習を止めるステップ数")
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
  parser.add_argument("--vae", type=str, default=None,
                      help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ")
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

  args = parser.parse_args()
  train(args)

