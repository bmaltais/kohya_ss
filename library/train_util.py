# common functions for training

import argparse
import importlib
import json
import shutil
import time
from typing import Dict, List, NamedTuple, Tuple
from typing import Optional, Union
from accelerate import Accelerator
from torch.autograd.function import Function
import glob
import math
import os
import random
import hashlib
import subprocess
from io import BytesIO

from tqdm import tqdm
import torch
from torch.optim import Optimizer
from torchvision import transforms
from transformers import CLIPTokenizer
import transformers
import diffusers
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from diffusers import DDPMScheduler, StableDiffusionPipeline
import albumentations as albu
import numpy as np
from PIL import Image
import cv2
from einops import rearrange
from torch import einsum
import safetensors.torch

import library.model_util as model_util

# Tokenizer: checkpointから読み込むのではなくあらかじめ提供されているものを使う
TOKENIZER_PATH = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2"     # ここからtokenizerだけ使う v2とv2.1はtokenizer仕様は同じ

# checkpointファイル名
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

# region dataset

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
# , ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]         # Linux?


class ImageInfo():
  def __init__(self, image_key: str, num_repeats: int, caption: str, is_reg: bool, absolute_path: str) -> None:
    self.image_key: str = image_key
    self.num_repeats: int = num_repeats
    self.caption: str = caption
    self.is_reg: bool = is_reg
    self.absolute_path: str = absolute_path
    self.image_size: Tuple[int, int] = None
    self.resized_size: Tuple[int, int] = None
    self.bucket_reso: Tuple[int, int] = None
    self.latents: torch.Tensor = None
    self.latents_flipped: torch.Tensor = None
    self.latents_npz: str = None
    self.latents_npz_flipped: str = None


class BucketManager():
  def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
    self.no_upscale = no_upscale
    if max_reso is None:
      self.max_reso = None
      self.max_area = None
    else:
      self.max_reso = max_reso
      self.max_area = max_reso[0] * max_reso[1]
    self.min_size = min_size
    self.max_size = max_size
    self.reso_steps = reso_steps

    self.resos = []
    self.reso_to_id = {}
    self.buckets = []                     # 前処理時は (image_key, image)、学習時は image_key

  def add_image(self, reso, image):
    bucket_id = self.reso_to_id[reso]
    self.buckets[bucket_id].append(image)

  def shuffle(self):
    for bucket in self.buckets:
      random.shuffle(bucket)

  def sort(self):
    # 解像度順にソートする（表示時、メタデータ格納時の見栄えをよくするためだけ）。bucketsも入れ替えてreso_to_idも振り直す
    sorted_resos = self.resos.copy()
    sorted_resos.sort()

    sorted_buckets = []
    sorted_reso_to_id = {}
    for i, reso in enumerate(sorted_resos):
      bucket_id = self.reso_to_id[reso]
      sorted_buckets.append(self.buckets[bucket_id])
      sorted_reso_to_id[reso] = i

    self.resos = sorted_resos
    self.buckets = sorted_buckets
    self.reso_to_id = sorted_reso_to_id

  def make_buckets(self):
    resos = model_util.make_bucket_resolutions(self.max_reso, self.min_size, self.max_size, self.reso_steps)
    self.set_predefined_resos(resos)

  def set_predefined_resos(self, resos):
    # 規定サイズから選ぶ場合の解像度、aspect ratioの情報を格納しておく
    self.predefined_resos = resos.copy()
    self.predefined_resos_set = set(resos)
    self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

  def add_if_new_reso(self, reso):
    if reso not in self.reso_to_id:
      bucket_id = len(self.resos)
      self.reso_to_id[reso] = bucket_id
      self.resos.append(reso)
      self.buckets.append([])
      # print(reso, bucket_id, len(self.buckets))

  def round_to_steps(self, x):
    x = int(x + .5)
    return x - x % self.reso_steps

  def select_bucket(self, image_width, image_height):
    aspect_ratio = image_width / image_height
    if not self.no_upscale:
      # 同じaspect ratioがあるかもしれないので（fine tuningで、no_upscale=Trueで前処理した場合）、解像度が同じものを優先する
      reso = (image_width, image_height)
      if reso in self.predefined_resos_set:
        pass
      else:
        ar_errors = self.predefined_aspect_ratios - aspect_ratio
        predefined_bucket_id = np.abs(ar_errors).argmin()          # 当該解像度以外でaspect ratio errorが最も少ないもの
        reso = self.predefined_resos[predefined_bucket_id]

      ar_reso = reso[0] / reso[1]
      if aspect_ratio > ar_reso:                   # 横が長い→縦を合わせる
        scale = reso[1] / image_height
      else:
        scale = reso[0] / image_width

      resized_size = (int(image_width * scale + .5), int(image_height * scale + .5))
      # print("use predef", image_width, image_height, reso, resized_size)
    else:
      if image_width * image_height > self.max_area:
        # 画像が大きすぎるのでアスペクト比を保ったまま縮小することを前提にbucketを決める
        resized_width = math.sqrt(self.max_area * aspect_ratio)
        resized_height = self.max_area / resized_width
        assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

        # リサイズ後の短辺または長辺をreso_steps単位にする：aspect ratioの差が少ないほうを選ぶ
        # 元のbucketingと同じロジック
        b_width_rounded = self.round_to_steps(resized_width)
        b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
        ar_width_rounded = b_width_rounded / b_height_in_wr

        b_height_rounded = self.round_to_steps(resized_height)
        b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
        ar_height_rounded = b_width_in_hr / b_height_rounded

        # print(b_width_rounded, b_height_in_wr, ar_width_rounded)
        # print(b_width_in_hr, b_height_rounded, ar_height_rounded)

        if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
          resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + .5))
        else:
          resized_size = (int(b_height_rounded * aspect_ratio + .5), b_height_rounded)
        # print(resized_size)
      else:
        resized_size = (image_width, image_height)              # リサイズは不要

      # 画像のサイズ未満をbucketのサイズとする（paddingせずにcroppingする）
      bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
      bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
      # print("use arbitrary", image_width, image_height, resized_size, bucket_width, bucket_height)

      reso = (bucket_width, bucket_height)

    self.add_if_new_reso(reso)

    ar_error = (reso[0] / reso[1]) - aspect_ratio
    return reso, resized_size, ar_error


class BucketBatchIndex(NamedTuple):
  bucket_index: int
  bucket_batch_size: int
  batch_index: int


class BaseDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer, max_token_length, shuffle_caption, shuffle_keep_tokens, resolution, flip_aug: bool, color_aug: bool, face_crop_aug_range, random_crop, debug_dataset: bool) -> None:
    super().__init__()
    self.tokenizer: CLIPTokenizer = tokenizer
    self.max_token_length = max_token_length
    self.shuffle_caption = shuffle_caption
    self.shuffle_keep_tokens = shuffle_keep_tokens
    # width/height is used when enable_bucket==False
    self.width, self.height = (None, None) if resolution is None else resolution
    self.face_crop_aug_range = face_crop_aug_range
    self.flip_aug = flip_aug
    self.color_aug = color_aug
    self.debug_dataset = debug_dataset
    self.random_crop = random_crop
    self.token_padding_disabled = False
    self.dataset_dirs_info = {}
    self.reg_dataset_dirs_info = {}
    self.tag_frequency = {}

    self.enable_bucket = False
    self.bucket_manager: BucketManager = None                         # not initialized
    self.min_bucket_reso = None
    self.max_bucket_reso = None
    self.bucket_reso_steps = None
    self.bucket_no_upscale = None
    self.bucket_info = None                                           # for metadata

    self.tokenizer_max_length = self.tokenizer.model_max_length if max_token_length is None else max_token_length + 2

    self.current_epoch: int = 0            # インスタンスがepochごとに新しく作られるようなので外側から渡さないとダメ
    self.dropout_rate: float = 0
    self.dropout_every_n_epochs: int = None
    self.tag_dropout_rate: float = 0

    # augmentation
    flip_p = 0.5 if flip_aug else 0.0
    if color_aug:
      # わりと弱めの色合いaugmentation：brightness/contrastあたりは画像のpixel valueの最大値・最小値を変えてしまうのでよくないのではという想定でgamma/hueあたりを触る
      self.aug = albu.Compose([
          albu.OneOf([
              albu.HueSaturationValue(8, 0, 0, p=.5),
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

    self.image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), ])

    self.image_data: Dict[str, ImageInfo] = {}

    self.replacements = {}

  def set_current_epoch(self, epoch):
    self.current_epoch = epoch

  def set_caption_dropout(self, dropout_rate, dropout_every_n_epochs, tag_dropout_rate):
    # コンストラクタで渡さないのはTextual Inversionで意識したくないから（ということにしておく）
    self.dropout_rate = dropout_rate
    self.dropout_every_n_epochs = dropout_every_n_epochs
    self.tag_dropout_rate = tag_dropout_rate

  def set_tag_frequency(self, dir_name, captions):
    frequency_for_dir = self.tag_frequency.get(dir_name, {})
    self.tag_frequency[dir_name] = frequency_for_dir
    for caption in captions:
      for tag in caption.split(","):
        if tag and not tag.isspace():
          tag = tag.lower()
          frequency = frequency_for_dir.get(tag, 0)
          frequency_for_dir[tag] = frequency + 1

  def disable_token_padding(self):
    self.token_padding_disabled = True

  def add_replacement(self, str_from, str_to):
    self.replacements[str_from] = str_to

  def process_caption(self, caption):
    # dropoutの決定：tag dropがこのメソッド内にあるのでここで行うのが良い
    is_drop_out = self.dropout_rate > 0 and random.random() < self.dropout_rate
    is_drop_out = is_drop_out or self.dropout_every_n_epochs and self.current_epoch % self.dropout_every_n_epochs == 0

    if is_drop_out:
      caption = ""
    else:
      if self.shuffle_caption or self.tag_dropout_rate > 0:
        def dropout_tags(tokens):
          if self.tag_dropout_rate <= 0:
            return tokens
          l = []
          for token in tokens:
            if random.random() >= self.tag_dropout_rate:
              l.append(token)
          return l

        tokens = [t.strip() for t in caption.strip().split(",")]
        if self.shuffle_keep_tokens is None:
          if self.shuffle_caption:
            random.shuffle(tokens)

          tokens = dropout_tags(tokens)
        else:
          if len(tokens) > self.shuffle_keep_tokens:
            keep_tokens = tokens[:self.shuffle_keep_tokens]
            tokens = tokens[self.shuffle_keep_tokens:]

            if self.shuffle_caption:
              random.shuffle(tokens)

            tokens = dropout_tags(tokens)

            tokens = keep_tokens + tokens
        caption = ", ".join(tokens)

      # textual inversion対応
      for str_from, str_to in self.replacements.items():
        if str_from == "":
          # replace all
          if type(str_to) == list:
            caption = random.choice(str_to)
          else:
            caption = str_to
        else:
          caption = caption.replace(str_from, str_to)

    return caption

  def get_input_ids(self, caption):
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
    return input_ids

  def register_image(self, info: ImageInfo):
    self.image_data[info.image_key] = info

  def make_buckets(self):
    '''
    bucketingを行わない場合も呼び出し必須（ひとつだけbucketを作る）
    min_size and max_size are ignored when enable_bucket is False
    '''
    print("loading image sizes.")
    for info in tqdm(self.image_data.values()):
      if info.image_size is None:
        info.image_size = self.get_image_size(info.absolute_path)

    if self.enable_bucket:
      print("make buckets")
    else:
      print("prepare dataset")

    # bucketを作成し、画像をbucketに振り分ける
    if self.enable_bucket:
      if self.bucket_manager is None:                         # fine tuningの場合でmetadataに定義がある場合は、すでに初期化済み
        self.bucket_manager = BucketManager(self.bucket_no_upscale, (self.width, self.height),
                                            self.min_bucket_reso, self.max_bucket_reso, self.bucket_reso_steps)
        if not self.bucket_no_upscale:
          self.bucket_manager.make_buckets()
        else:
          print("min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます")

      img_ar_errors = []
      for image_info in self.image_data.values():
        image_width, image_height = image_info.image_size
        image_info.bucket_reso, image_info.resized_size, ar_error = self.bucket_manager.select_bucket(image_width, image_height)

        # print(image_info.image_key, image_info.bucket_reso)
        img_ar_errors.append(abs(ar_error))

      self.bucket_manager.sort()
    else:
      self.bucket_manager = BucketManager(False, (self.width, self.height), None, None, None)
      self.bucket_manager.set_predefined_resos([(self.width, self.height)])  # ひとつの固定サイズbucketのみ
      for image_info in self.image_data.values():
        image_width, image_height = image_info.image_size
        image_info.bucket_reso, image_info.resized_size, _ = self.bucket_manager.select_bucket(image_width, image_height)

    for image_info in self.image_data.values():
      for _ in range(image_info.num_repeats):
        self.bucket_manager.add_image(image_info.bucket_reso, image_info.image_key)

    # bucket情報を表示、格納する
    if self.enable_bucket:
      self.bucket_info = {"buckets": {}}
      print("number of images (including repeats) / 各bucketの画像枚数（繰り返し回数を含む）")
      for i, (reso, bucket) in enumerate(zip(self.bucket_manager.resos, self.bucket_manager.buckets)):
        count = len(bucket)
        if count > 0:
          self.bucket_info["buckets"][i] = {"resolution": reso, "count": len(bucket)}
          print(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

      img_ar_errors = np.array(img_ar_errors)
      mean_img_ar_error = np.mean(np.abs(img_ar_errors))
      self.bucket_info["mean_img_ar_error"] = mean_img_ar_error
      print(f"mean ar error (without repeats): {mean_img_ar_error}")

    # データ参照用indexを作る。このindexはdatasetのshuffleに用いられる
    self.buckets_indices: List(BucketBatchIndex) = []
    for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
      batch_count = int(math.ceil(len(bucket) / self.batch_size))
      for batch_index in range(batch_count):
        self.buckets_indices.append(BucketBatchIndex(bucket_index, self.batch_size, batch_index))

      # ↓以下はbucketごとのbatch件数があまりにも増えて混乱を招くので元に戻す
      # 　学習時はステップ数がランダムなので、同一画像が同一batch内にあってもそれほど悪影響はないであろう、と考えられる
      #
      # # bucketが細分化されることにより、ひとつのbucketに一種類の画像のみというケースが増え、つまりそれは
      # # ひとつのbatchが同じ画像で占められることになるので、さすがに良くないであろう
      # # そのためバッチサイズを画像種類までに制限する
      # # ただそれでも同一画像が同一バッチに含まれる可能性はあるので、繰り返し回数が少ないほうがshuffleの品質は良くなることは間違いない？
      # # TO DO 正則化画像をepochまたがりで利用する仕組み
      # num_of_image_types = len(set(bucket))
      # bucket_batch_size = min(self.batch_size, num_of_image_types)
      # batch_count = int(math.ceil(len(bucket) / bucket_batch_size))
      # # print(bucket_index, num_of_image_types, bucket_batch_size, batch_count)
      # for batch_index in range(batch_count):
      #   self.buckets_indices.append(BucketBatchIndex(bucket_index, bucket_batch_size, batch_index))
      # ↑ここまで

    self.shuffle_buckets()
    self._length = len(self.buckets_indices)

  def shuffle_buckets(self):
    random.shuffle(self.buckets_indices)
    self.bucket_manager.shuffle()

  def load_image(self, image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
      image = image.convert("RGB")
    img = np.array(image, np.uint8)
    return img

  def trim_and_resize_if_required(self, image, reso, resized_size):
    image_height, image_width = image.shape[0:2]

    if image_width != resized_size[0] or image_height != resized_size[1]:
      # リサイズする
      image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)       # INTER_AREAでやりたいのでcv2でリサイズ

    image_height, image_width = image.shape[0:2]
    if image_width > reso[0]:
      trim_size = image_width - reso[0]
      p = trim_size // 2 if not self.random_crop else random.randint(0, trim_size)
      # print("w", trim_size, p)
      image = image[:, p:p + reso[0]]
    if image_height > reso[1]:
      trim_size = image_height - reso[1]
      p = trim_size // 2 if not self.random_crop else random.randint(0, trim_size)
      # print("h", trim_size, p)
      image = image[p:p + reso[1]]

    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}"
    return image

  def cache_latents(self, vae):
    # TODO ここを高速化したい
    print("caching latents.")
    for info in tqdm(self.image_data.values()):
      if info.latents_npz is not None:
        info.latents = self.load_latents_from_npz(info, False)
        info.latents = torch.FloatTensor(info.latents)
        info.latents_flipped = self.load_latents_from_npz(info, True)             # might be None
        if info.latents_flipped is not None:
          info.latents_flipped = torch.FloatTensor(info.latents_flipped)
        continue

      image = self.load_image(info.absolute_path)
      image = self.trim_and_resize_if_required(image, info.bucket_reso, info.resized_size)

      img_tensor = self.image_transforms(image)
      img_tensor = img_tensor.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
      info.latents = vae.encode(img_tensor).latent_dist.sample().squeeze(0).to("cpu")

      if self.flip_aug:
        image = image[:, ::-1].copy()     # cannot convert to Tensor without copy
        img_tensor = self.image_transforms(image)
        img_tensor = img_tensor.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
        info.latents_flipped = vae.encode(img_tensor).latent_dist.sample().squeeze(0).to("cpu")

  def get_image_size(self, image_path):
    image = Image.open(image_path)
    return image.size

  def load_image_with_face_info(self, image_path: str):
    img = self.load_image(image_path)

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

    # 顔を中心として448*640とかへ切り出す
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

  def load_latents_from_npz(self, image_info: ImageInfo, flipped):
    npz_file = image_info.latents_npz_flipped if flipped else image_info.latents_npz
    if npz_file is None:
      return None
    return np.load(npz_file)['arr_0']

  def __len__(self):
    return self._length

  def __getitem__(self, index):
    if index == 0:
      self.shuffle_buckets()

    bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
    bucket_batch_size = self.buckets_indices[index].bucket_batch_size
    image_index = self.buckets_indices[index].batch_index * bucket_batch_size

    loss_weights = []
    captions = []
    input_ids_list = []
    latents_list = []
    images = []

    for image_key in bucket[image_index:image_index + bucket_batch_size]:
      image_info = self.image_data[image_key]
      loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)

      # image/latentsを処理する
      if image_info.latents is not None:
        latents = image_info.latents if not self.flip_aug or random.random() < .5 else image_info.latents_flipped
        image = None
      elif image_info.latents_npz is not None:
        latents = self.load_latents_from_npz(image_info, self.flip_aug and random.random() >= .5)
        latents = torch.FloatTensor(latents)
        image = None
      else:
        # 画像を読み込み、必要ならcropする
        img, face_cx, face_cy, face_w, face_h = self.load_image_with_face_info(image_info.absolute_path)
        im_h, im_w = img.shape[0:2]

        if self.enable_bucket:
          img = self.trim_and_resize_if_required(img, image_info.bucket_reso, image_info.resized_size)
        else:
          if face_cx > 0:                   # 顔位置情報あり
            img = self.crop_target(img, face_cx, face_cy, face_w, face_h)
          elif im_h > self.height or im_w > self.width:
            assert self.random_crop, f"image too large, but cropping and bucketing are disabled / 画像サイズが大きいのでface_crop_aug_rangeかrandom_crop、またはbucketを有効にしてください: {image_info.absolute_path}"
            if im_h > self.height:
              p = random.randint(0, im_h - self.height)
              img = img[p:p + self.height]
            if im_w > self.width:
              p = random.randint(0, im_w - self.width)
              img = img[:, p:p + self.width]

          im_h, im_w = img.shape[0:2]
          assert im_h == self.height and im_w == self.width, f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}"

        # augmentation
        if self.aug is not None:
          img = self.aug(image=img)['image']

        latents = None
        image = self.image_transforms(img)      # -1.0~1.0のtorch.Tensorになる

      images.append(image)
      latents_list.append(latents)

      caption = self.process_caption(image_info.caption)
      captions.append(caption)
      if not self.token_padding_disabled:                     # this option might be omitted in future
        input_ids_list.append(self.get_input_ids(caption))

    example = {}
    example['loss_weights'] = torch.FloatTensor(loss_weights)

    if self.token_padding_disabled:
      # padding=True means pad in the batch
      example['input_ids'] = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").input_ids
    else:
      # batch processing seems to be good
      example['input_ids'] = torch.stack(input_ids_list)

    if images[0] is not None:
      images = torch.stack(images)
      images = images.to(memory_format=torch.contiguous_format).float()
    else:
      images = None
    example['images'] = images

    example['latents'] = torch.stack(latents_list) if latents_list[0] is not None else None

    if self.debug_dataset:
      example['image_keys'] = bucket[image_index:image_index + self.batch_size]
      example['captions'] = captions
    return example


class DreamBoothDataset(BaseDataset):
  def __init__(self, batch_size, train_data_dir, reg_data_dir, tokenizer, max_token_length, caption_extension, shuffle_caption, shuffle_keep_tokens, resolution, enable_bucket, min_bucket_reso, max_bucket_reso, bucket_reso_steps, bucket_no_upscale, prior_loss_weight, flip_aug, color_aug, face_crop_aug_range, random_crop, debug_dataset) -> None:
    super().__init__(tokenizer, max_token_length, shuffle_caption, shuffle_keep_tokens,
                     resolution, flip_aug, color_aug, face_crop_aug_range, random_crop, debug_dataset)

    assert resolution is not None, f"resolution is required / resolution（解像度）指定は必須です"

    self.batch_size = batch_size
    self.size = min(self.width, self.height)                  # 短いほう
    self.prior_loss_weight = prior_loss_weight
    self.latents_cache = None

    self.enable_bucket = enable_bucket
    if self.enable_bucket:
      assert min(resolution) >= min_bucket_reso, f"min_bucket_reso must be equal or less than resolution / min_bucket_resoは最小解像度より大きくできません。解像度を大きくするかmin_bucket_resoを小さくしてください"
      assert max(resolution) <= max_bucket_reso, f"max_bucket_reso must be equal or greater than resolution / max_bucket_resoは最大解像度より小さくできません。解像度を小さくするかmin_bucket_resoを大きくしてください"
      self.min_bucket_reso = min_bucket_reso
      self.max_bucket_reso = max_bucket_reso
      self.bucket_reso_steps = bucket_reso_steps
      self.bucket_no_upscale = bucket_no_upscale
    else:
      self.min_bucket_reso = None
      self.max_bucket_reso = None
      self.bucket_reso_steps = None                              # この情報は使われない
      self.bucket_no_upscale = False

    def read_caption(img_path):
      # captionの候補ファイル名を作る
      base_name = os.path.splitext(img_path)[0]
      base_name_face_det = base_name
      tokens = base_name.split("_")
      if len(tokens) >= 5:
        base_name_face_det = "_".join(tokens[:-4])
      cap_paths = [base_name + caption_extension, base_name_face_det + caption_extension]

      caption = None
      for cap_path in cap_paths:
        if os.path.isfile(cap_path):
          with open(cap_path, "rt", encoding='utf-8') as f:
            try:
              lines = f.readlines()
            except UnicodeDecodeError as e:
              print(f"illegal char in file (not UTF-8) / ファイルにUTF-8以外の文字があります: {cap_path}")
              raise e
            assert len(lines) > 0, f"caption file is empty / キャプションファイルが空です: {cap_path}"
            caption = lines[0].strip()
          break
      return caption

    def load_dreambooth_dir(dir):
      if not os.path.isdir(dir):
        # print(f"ignore file: {dir}")
        return 0, [], []

      tokens = os.path.basename(dir).split('_')
      try:
        n_repeats = int(tokens[0])
      except ValueError as e:
        print(f"ignore directory without repeats / 繰り返し回数のないディレクトリを無視します: {dir}")
        return 0, [], []

      caption_by_folder = '_'.join(tokens[1:])
      img_paths = glob_images(dir, "*")
      print(f"found directory {n_repeats}_{caption_by_folder} contains {len(img_paths)} image files")

      # 画像ファイルごとにプロンプトを読み込み、もしあればそちらを使う
      captions = []
      for img_path in img_paths:
        cap_for_img = read_caption(img_path)
        captions.append(caption_by_folder if cap_for_img is None else cap_for_img)

      self.set_tag_frequency(os.path.basename(dir), captions)         # タグ頻度を記録

      return n_repeats, img_paths, captions

    print("prepare train images.")
    train_dirs = os.listdir(train_data_dir)
    num_train_images = 0
    for dir in train_dirs:
      n_repeats, img_paths, captions = load_dreambooth_dir(os.path.join(train_data_dir, dir))
      num_train_images += n_repeats * len(img_paths)

      for img_path, caption in zip(img_paths, captions):
        info = ImageInfo(img_path, n_repeats, caption, False, img_path)
        self.register_image(info)

      self.dataset_dirs_info[os.path.basename(dir)] = {"n_repeats": n_repeats, "img_count": len(img_paths)}

    print(f"{num_train_images} train images with repeating.")
    self.num_train_images = num_train_images

    # reg imageは数を数えて学習画像と同じ枚数にする
    num_reg_images = 0
    if reg_data_dir:
      print("prepare reg images.")
      reg_infos: List[ImageInfo] = []

      reg_dirs = os.listdir(reg_data_dir)
      for dir in reg_dirs:
        n_repeats, img_paths, captions = load_dreambooth_dir(os.path.join(reg_data_dir, dir))
        num_reg_images += n_repeats * len(img_paths)

        for img_path, caption in zip(img_paths, captions):
          info = ImageInfo(img_path, n_repeats, caption, True, img_path)
          reg_infos.append(info)

        self.reg_dataset_dirs_info[os.path.basename(dir)] = {"n_repeats": n_repeats, "img_count": len(img_paths)}

      print(f"{num_reg_images} reg images.")
      if num_train_images < num_reg_images:
        print("some of reg images are not used / 正則化画像の数が多いので、一部使用されない正則化画像があります")

      if num_reg_images == 0:
        print("no regularization images / 正則化画像が見つかりませんでした")
      else:
        # num_repeatsを計算する：どうせ大した数ではないのでループで処理する
        n = 0
        first_loop = True
        while n < num_train_images:
          for info in reg_infos:
            if first_loop:
              self.register_image(info)
              n += info.num_repeats
            else:
              info.num_repeats += 1
              n += 1
            if n >= num_train_images:
              break
          first_loop = False

    self.num_reg_images = num_reg_images


class FineTuningDataset(BaseDataset):
  def __init__(self, json_file_name, batch_size, train_data_dir, tokenizer, max_token_length, shuffle_caption, shuffle_keep_tokens, resolution, enable_bucket, min_bucket_reso, max_bucket_reso, bucket_reso_steps, bucket_no_upscale, flip_aug, color_aug, face_crop_aug_range, random_crop, dataset_repeats, debug_dataset) -> None:
    super().__init__(tokenizer, max_token_length, shuffle_caption, shuffle_keep_tokens,
                     resolution, flip_aug, color_aug, face_crop_aug_range, random_crop, debug_dataset)

    # メタデータを読み込む
    if os.path.exists(json_file_name):
      print(f"loading existing metadata: {json_file_name}")
      with open(json_file_name, "rt", encoding='utf-8') as f:
        metadata = json.load(f)
    else:
      raise ValueError(f"no metadata / メタデータファイルがありません: {json_file_name}")

    self.metadata = metadata
    self.train_data_dir = train_data_dir
    self.batch_size = batch_size

    tags_list = []
    for image_key, img_md in metadata.items():
      # path情報を作る
      if os.path.exists(image_key):
        abs_path = image_key
      else:
        # わりといい加減だがいい方法が思いつかん
        abs_path = glob_images(train_data_dir, image_key)
        assert len(abs_path) >= 1, f"no image / 画像がありません: {image_key}"
        abs_path = abs_path[0]

      caption = img_md.get('caption')
      tags = img_md.get('tags')
      if caption is None:
        caption = tags
      elif tags is not None and len(tags) > 0:
        caption = caption + ', ' + tags
        tags_list.append(tags)
      assert caption is not None and len(caption) > 0, f"caption or tag is required / キャプションまたはタグは必須です:{abs_path}"

      image_info = ImageInfo(image_key, dataset_repeats, caption, False, abs_path)
      image_info.image_size = img_md.get('train_resolution')

      if not self.color_aug and not self.random_crop:
        # if npz exists, use them
        image_info.latents_npz, image_info.latents_npz_flipped = self.image_key_to_npz_file(image_key)

      self.register_image(image_info)
    self.num_train_images = len(metadata) * dataset_repeats
    self.num_reg_images = 0

    # TODO do not record tag freq when no tag
    self.set_tag_frequency(os.path.basename(json_file_name), tags_list)
    self.dataset_dirs_info[os.path.basename(json_file_name)] = {"n_repeats": dataset_repeats, "img_count": len(metadata)}

    # check existence of all npz files
    use_npz_latents = not (self.color_aug or self.random_crop)
    if use_npz_latents:
      npz_any = False
      npz_all = True
      for image_info in self.image_data.values():
        has_npz = image_info.latents_npz is not None
        npz_any = npz_any or has_npz

        if self.flip_aug:
          has_npz = has_npz and image_info.latents_npz_flipped is not None
        npz_all = npz_all and has_npz

        if npz_any and not npz_all:
          break

      if not npz_any:
        use_npz_latents = False
        print(f"npz file does not exist. ignore npz files / npzファイルが見つからないためnpzファイルを無視します")
      elif not npz_all:
        use_npz_latents = False
        print(f"some of npz file does not exist. ignore npz files / いくつかのnpzファイルが見つからないためnpzファイルを無視します")
        if self.flip_aug:
          print("maybe no flipped files / 反転されたnpzファイルがないのかもしれません")
    # else:
    #   print("npz files are not used with color_aug and/or random_crop / color_augまたはrandom_cropが指定されているためnpzファイルは使用されません")

    # check min/max bucket size
    sizes = set()
    resos = set()
    for image_info in self.image_data.values():
      if image_info.image_size is None:
        sizes = None                  # not calculated
        break
      sizes.add(image_info.image_size[0])
      sizes.add(image_info.image_size[1])
      resos.add(tuple(image_info.image_size))

    if sizes is None:
      if use_npz_latents:
        use_npz_latents = False
        print(f"npz files exist, but no bucket info in metadata. ignore npz files / メタデータにbucket情報がないためnpzファイルを無視します")

      assert resolution is not None, "if metadata doesn't have bucket info, resolution is required / メタデータにbucket情報がない場合はresolutionを指定してください"

      self.enable_bucket = enable_bucket
      if self.enable_bucket:
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso
        self.bucket_reso_steps = bucket_reso_steps
        self.bucket_no_upscale = bucket_no_upscale
    else:
      if not enable_bucket:
        print("metadata has bucket info, enable bucketing / メタデータにbucket情報があるためbucketを有効にします")
      print("using bucket info in metadata / メタデータ内のbucket情報を使います")
      self.enable_bucket = True

      assert not bucket_no_upscale, "if metadata has bucket info, bucket reso is precalculated, so bucket_no_upscale cannot be used / メタデータ内にbucket情報がある場合はbucketの解像度は計算済みのため、bucket_no_upscaleは使えません"

      # bucket情報を初期化しておく、make_bucketsで再作成しない
      self.bucket_manager = BucketManager(False, None, None, None, None)
      self.bucket_manager.set_predefined_resos(resos)

    # npz情報をきれいにしておく
    if not use_npz_latents:
      for image_info in self.image_data.values():
        image_info.latents_npz = image_info.latents_npz_flipped = None

  def image_key_to_npz_file(self, image_key):
    base_name = os.path.splitext(image_key)[0]
    npz_file_norm = base_name + '.npz'

    if os.path.exists(npz_file_norm):
      # image_key is full path
      npz_file_flip = base_name + '_flip.npz'
      if not os.path.exists(npz_file_flip):
        npz_file_flip = None
      return npz_file_norm, npz_file_flip

    # image_key is relative path
    npz_file_norm = os.path.join(self.train_data_dir, image_key + '.npz')
    npz_file_flip = os.path.join(self.train_data_dir, image_key + '_flip.npz')

    if not os.path.exists(npz_file_norm):
      npz_file_norm = None
      npz_file_flip = None
    elif not os.path.exists(npz_file_flip):
      npz_file_flip = None

    return npz_file_norm, npz_file_flip


def debug_dataset(train_dataset, show_input_ids=False):
  print(f"Total dataset length (steps) / データセットの長さ（ステップ数）: {len(train_dataset)}")
  print("Escape for exit. / Escキーで中断、終了します")

  train_dataset.set_current_epoch(1)
  k = 0
  for i, example in enumerate(train_dataset):
    if example['latents'] is not None:
      print(f"sample has latents from npz file: {example['latents'].size()}")
    for j, (ik, cap, lw, iid) in enumerate(zip(example['image_keys'], example['captions'], example['loss_weights'], example['input_ids'])):
      print(f'{ik}, size: {train_dataset.image_data[ik].image_size}, loss weight: {lw}, caption: "{cap}"')
      if show_input_ids:
        print(f"input ids: {iid}")
      if example['images'] is not None:
        im = example['images'][j]
        print(f"image size: {im.size()}")
        im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))                # c,H,W -> H,W,c
        im = im[:, :, ::-1]                             # RGB -> BGR (OpenCV)
        if os.name == 'nt':                             # only windows
          cv2.imshow("img", im)
        k = cv2.waitKey()
        cv2.destroyAllWindows()
        if k == 27:
          break
    if k == 27 or (example['images'] is None and i >= 8):
      break


def glob_images(directory, base="*"):
  img_paths = []
  for ext in IMAGE_EXTENSIONS:
    if base == '*':
      img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
    else:
      img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
  # img_paths = list(set(img_paths))                    # 重複を排除
  # img_paths.sort()
  return img_paths


def glob_images_pathlib(dir_path, recursive):
  image_paths = []
  if recursive:
    for ext in IMAGE_EXTENSIONS:
      image_paths += list(dir_path.rglob('*' + ext))
  else:
    for ext in IMAGE_EXTENSIONS:
      image_paths += list(dir_path.glob('*' + ext))
  # image_paths = list(set(image_paths))        # 重複を排除
  # image_paths.sort()
  return image_paths

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


def model_hash(filename):
  """Old model hash used by stable-diffusion-webui"""
  try:
    with open(filename, "rb") as file:
      m = hashlib.sha256()

      file.seek(0x100000)
      m.update(file.read(0x10000))
      return m.hexdigest()[0:8]
  except FileNotFoundError:
    return 'NOFILE'


def calculate_sha256(filename):
  """New model hash used by stable-diffusion-webui"""
  hash_sha256 = hashlib.sha256()
  blksize = 1024 * 1024

  with open(filename, "rb") as f:
    for chunk in iter(lambda: f.read(blksize), b""):
      hash_sha256.update(chunk)

  return hash_sha256.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata):
  """Precalculate the model hashes needed by sd-webui-additional-networks to
  save time on indexing the model later."""

  # Because writing user metadata to the file can change the result of
  # sd_models.model_hash(), only retain the training metadata for purposes of
  # calculating the hash, as they are meant to be immutable
  metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

  bytes = safetensors.torch.save(tensors, metadata)
  b = BytesIO(bytes)

  model_hash = addnet_hash_safetensors(b)
  legacy_hash = addnet_hash_legacy(b)
  return model_hash, legacy_hash


def addnet_hash_legacy(b):
  """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
  m = hashlib.sha256()

  b.seek(0x100000)
  m.update(b.read(0x10000))
  return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
  """New model hash used by sd-webui-additional-networks for .safetensors format files"""
  hash_sha256 = hashlib.sha256()
  blksize = 1024 * 1024

  b.seek(0)
  header = b.read(8)
  n = int.from_bytes(header, "little")

  offset = n + 8
  b.seek(offset)
  for chunk in iter(lambda: b.read(blksize), b""):
    hash_sha256.update(chunk)

  return hash_sha256.hexdigest()


def get_git_revision_hash() -> str:
  try:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__)).decode('ascii').strip()
  except:
    return "(unknown)"


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
  print("Replace CrossAttention.forward to use FlashAttention (not xformers)")
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

    # diffusers 0.7.0~
    out = self.to_out[0](out)
    out = self.to_out[1](out)
    return out

  diffusers.models.attention.CrossAttention.forward = forward_xformers
# endregion


# region arguments

def add_sd_models_arguments(parser: argparse.ArgumentParser):
  # for pretrained models
  parser.add_argument("--v2", action='store_true',
                      help='load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む')
  parser.add_argument("--v_parameterization", action='store_true',
                      help='enable v-parameterization training / v-parameterization学習を有効にする')
  parser.add_argument("--pretrained_model_name_or_path", type=str, default=None,
                      help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint / 学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル")


def add_optimizer_arguments(parser: argparse.ArgumentParser):
  parser.add_argument("--optimizer_type", type=str, default="",
                      help="Optimizer to use / オプティマイザの種類: AdamW (default), AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdaptation, AdaFactor")

  # backward compatibility
  parser.add_argument("--use_8bit_adam", action="store_true",
                      help="use 8bit AdamW optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）")
  parser.add_argument("--use_lion_optimizer", action="store_true",
                      help="use Lion optimizer (requires lion-pytorch) / Lionオプティマイザを使う（ lion-pytorch のインストールが必要）")

  parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
  parser.add_argument("--max_grad_norm", default=1.0, type=float,
                      help="Max gradient norm, 0 for no clipping / 勾配正規化の最大norm、0でclippingを行わない")

  parser.add_argument("--optimizer_args", type=str, default=None, nargs='*',
                      help="additional arguments for optimizer (like \"weight_decay=0.01 betas=0.9,0.999 ...\") / オプティマイザの追加引数（例： \"weight_decay=0.01 betas=0.9,0.999 ...\"）")

  parser.add_argument("--lr_scheduler", type=str, default="constant",
                      help="scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor")
  parser.add_argument("--lr_warmup_steps", type=int, default=0,
                      help="Number of steps for the warmup in the lr scheduler (default is 0) / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）")
  parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1,
                      help="Number of restarts for cosine scheduler with restarts / cosine with restartsスケジューラでのリスタート回数")
  parser.add_argument("--lr_scheduler_power", type=float, default=1,
                      help="Polynomial power for polynomial scheduler / polynomialスケジューラでのpolynomial power")


def add_training_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool):
  parser.add_argument("--output_dir", type=str, default=None,
                      help="directory to output trained model / 学習後のモデル出力先ディレクトリ")
  parser.add_argument("--output_name", type=str, default=None,
                      help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名")
  parser.add_argument("--save_precision", type=str, default=None,
                      choices=[None, "float", "fp16", "bf16"], help="precision in saving / 保存時に精度を変更して保存する")
  parser.add_argument("--save_every_n_epochs", type=int, default=None,
                      help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する")
  parser.add_argument("--save_n_epoch_ratio", type=int, default=None,
                      help="save checkpoint N epoch ratio (for example 5 means save at least 5 files total) / 学習中のモデルを指定のエポック割合で保存する（たとえば5を指定すると最低5個のファイルが保存される）")
  parser.add_argument("--save_last_n_epochs", type=int, default=None, help="save last N checkpoints / 最大Nエポック保存する")
  parser.add_argument("--save_last_n_epochs_state", type=int, default=None,
                      help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)/ 最大Nエポックstateを保存する(--save_last_n_epochsの指定を上書きします)")
  parser.add_argument("--save_state", action="store_true",
                      help="save training state additionally (including optimizer states etc.) / optimizerなど学習状態も含めたstateを追加で保存する")
  parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")

  parser.add_argument("--train_batch_size", type=int, default=1, help="batch size for training / 学習時のバッチサイズ")
  parser.add_argument("--max_token_length", type=int, default=None, choices=[None, 150, 225],
                      help="max token length of text encoder (default for 75, 150 or 225) / text encoderのトークンの最大長（未指定で75、150または225が指定可）")
  parser.add_argument("--mem_eff_attn", action="store_true",
                      help="use memory efficient attention for CrossAttention / CrossAttentionに省メモリ版attentionを使う")
  parser.add_argument("--xformers", action="store_true",
                      help="use xformers for CrossAttention / CrossAttentionにxformersを使う")
  parser.add_argument("--vae", type=str, default=None,
                      help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ")

  parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
  parser.add_argument("--max_train_epochs", type=int, default=None,
                      help="training epochs (overrides max_train_steps) / 学習エポック数（max_train_stepsを上書きします）")
  parser.add_argument("--max_data_loader_n_workers", type=int, default=8,
                      help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading) / DataLoaderの最大プロセス数（小さい値ではメインメモリの使用量が減りエポック間の待ち時間が減りますが、データ読み込みは遅くなります）")
  parser.add_argument("--persistent_data_loader_workers", action="store_true",
                      help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory) / DataLoader のワーカーを持続させる (エポック間の時間差を少なくするのに有効だが、より多くのメモリを消費する可能性がある)")
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
  parser.add_argument("--noise_offset", type=float, default=None,
                      help="enable noise offset with this value (if enabled, around 0.1 is recommended) / Noise offsetを有効にしてこの値を設定する（有効にする場合は0.1程度を推奨）")
  parser.add_argument("--lowram", action="store_true",
                      help="enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle) / メインメモリが少ない環境向け最適化を有効にする。たとえばVRAMにモデルを読み込むなど（ColabやKaggleなどRAMに比べてVRAMが多い環境向け）")

  if support_dreambooth:
    # DreamBooth training
    parser.add_argument("--prior_loss_weight", type=float, default=1.0,
                        help="loss weight for regularization images / 正則化画像のlossの重み")


def verify_training_args(args: argparse.Namespace):
  if args.v_parameterization and not args.v2:
    print("v_parameterization should be with v2 / v1でv_parameterizationを使用することは想定されていません")
  if args.v2 and args.clip_skip is not None:
    print("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")


def add_dataset_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool, support_caption: bool, support_caption_dropout: bool):
  # dataset common
  parser.add_argument("--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("--shuffle_caption", action="store_true",
                      help="shuffle comma-separated caption / コンマで区切られたcaptionの各要素をshuffleする")
  parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption files / 読み込むcaptionファイルの拡張子")
  parser.add_argument("--caption_extention", type=str, default=None,
                      help="extension of caption files (backward compatibility) / 読み込むcaptionファイルの拡張子（スペルミスを残してあります）")
  parser.add_argument("--keep_tokens", type=int, default=None,
                      help="keep heading N tokens when shuffling caption tokens / captionのシャッフル時に、先頭からこの個数のトークンをシャッフルしないで残す")
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
  parser.add_argument("--cache_latents", action="store_true",
                      help="cache latents to reduce memory (augmentations must be disabled) / メモリ削減のためにlatentをcacheする（augmentationは使用不可）")
  parser.add_argument("--enable_bucket", action="store_true",
                      help="enable buckets for multi aspect ratio training / 複数解像度学習のためのbucketを有効にする")
  parser.add_argument("--min_bucket_reso", type=int, default=256, help="minimum resolution for buckets / bucketの最小解像度")
  parser.add_argument("--max_bucket_reso", type=int, default=1024, help="maximum resolution for buckets / bucketの最大解像度")
  parser.add_argument("--bucket_reso_steps", type=int, default=64,
                      help="steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します")
  parser.add_argument("--bucket_no_upscale", action="store_true",
                      help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します")

  if support_caption_dropout:
    # Textual Inversion はcaptionのdropoutをsupportしない
    # いわゆるtensorのDropoutと紛らわしいのでprefixにcaptionを付けておく　every_n_epochsは他と平仄を合わせてdefault Noneに
    parser.add_argument("--caption_dropout_rate", type=float, default=0,
                        help="Rate out dropout caption(0.0~1.0) / captionをdropoutする割合")
    parser.add_argument("--caption_dropout_every_n_epochs", type=int, default=None,
                        help="Dropout all captions every N epochs / captionを指定エポックごとにdropoutする")
    parser.add_argument("--caption_tag_dropout_rate", type=float, default=0,
                        help="Rate out dropout comma separated tokens(0.0~1.0) / カンマ区切りのタグをdropoutする割合")

  if support_dreambooth:
    # DreamBooth dataset
    parser.add_argument("--reg_data_dir", type=str, default=None, help="directory for regularization images / 正則化画像データのディレクトリ")

  if support_caption:
    # caption dataset
    parser.add_argument("--in_json", type=str, default=None, help="json metadata for dataset / データセットのmetadataのjsonファイル")
    parser.add_argument("--dataset_repeats", type=int, default=1,
                        help="repeat dataset when training with captions / キャプションでの学習時にデータセットを繰り返す回数")


def add_sd_saving_arguments(parser: argparse.ArgumentParser):
  parser.add_argument("--save_model_as", type=str, default=None, choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
                      help="format to save the model (default is same to original) / モデル保存時の形式（未指定時は元モデルと同じ）")
  parser.add_argument("--use_safetensors", action='store_true',
                      help="use safetensors format to save (if save_model_as is not specified) / checkpoint、モデルをsafetensors形式で保存する（save_model_as未指定時）")

# endregion

# region utils


def get_optimizer(args, trainable_params):
  # "Optimizer to use: AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdaptation, Adafactor"

  optimizer_type = args.optimizer_type
  if args.use_8bit_adam:
    assert not args.use_lion_optimizer, "both option use_8bit_adam and use_lion_optimizer are specified / use_8bit_adamとuse_lion_optimizerの両方のオプションが指定されています"
    assert optimizer_type is None or optimizer_type == "", "both option use_8bit_adam and optimizer_type are specified / use_8bit_adamとoptimizer_typeの両方のオプションが指定されています"
    optimizer_type = "AdamW8bit"

  elif args.use_lion_optimizer:
    assert optimizer_type is None or optimizer_type == "", "both option use_lion_optimizer and optimizer_type are specified / use_lion_optimizerとoptimizer_typeの両方のオプションが指定されています"
    optimizer_type = "Lion"

  if optimizer_type is None or optimizer_type == "":
    optimizer_type = "AdamW"
  optimizer_type = optimizer_type.lower()

  # 引数を分解する：boolとfloat、tupleのみ対応
  optimizer_kwargs = {}
  if args.optimizer_args is not None and len(args.optimizer_args) > 0:
    for arg in args.optimizer_args:
      key, value = arg.split('=')

      value = value.split(",")
      for i in range(len(value)):
        if value[i].lower() == "true" or value[i].lower() == "false":
          value[i] = (value[i].lower() == "true")
        else:
          value[i] = float(value[i])
      if len(value) == 1:
        value = value[0]
      else:
        value = tuple(value)

      optimizer_kwargs[key] = value
  # print("optkwargs:", optimizer_kwargs)

  lr = args.learning_rate

  if optimizer_type == "AdamW8bit".lower():
    try:
      import bitsandbytes as bnb
    except ImportError:
      raise ImportError("No bitsand bytes / bitsandbytesがインストールされていないようです")
    print(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
    optimizer_class = bnb.optim.AdamW8bit
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

  elif optimizer_type == "SGDNesterov8bit".lower():
    try:
      import bitsandbytes as bnb
    except ImportError:
      raise ImportError("No bitsand bytes / bitsandbytesがインストールされていないようです")
    print(f"use 8-bit SGD with Nesterov optimizer | {optimizer_kwargs}")
    if "momentum" not in optimizer_kwargs:
      print(f"8-bit SGD with Nesterov must be with momentum, set momentum to 0.9 / 8-bit SGD with Nesterovはmomentum指定が必須のため0.9に設定します")
      optimizer_kwargs["momentum"] = 0.9

    optimizer_class = bnb.optim.SGD8bit
    optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

  elif optimizer_type == "Lion".lower():
    try:
      import lion_pytorch
    except ImportError:
      raise ImportError("No lion_pytorch / lion_pytorch がインストールされていないようです")
    print(f"use Lion optimizer | {optimizer_kwargs}")
    optimizer_class = lion_pytorch.Lion
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

  elif optimizer_type == "SGDNesterov".lower():
    print(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
    if "momentum" not in optimizer_kwargs:
      print(f"SGD with Nesterov must be with momentum, set momentum to 0.9 / SGD with Nesterovはmomentum指定が必須のため0.9に設定します")
      optimizer_kwargs["momentum"] = 0.9

    optimizer_class = torch.optim.SGD
    optimizer = optimizer_class(trainable_params, lr=lr,  nesterov=True, **optimizer_kwargs)

  elif optimizer_type == "DAdaptation".lower():
    try:
      import dadaptation
    except ImportError:
      raise ImportError("No dadaptation / dadaptation がインストールされていないようです")
    print(f"use D-Adaptation Adam optimizer | {optimizer_kwargs}")

    min_lr = lr
    if type(trainable_params) == list and type(trainable_params[0]) == dict:
      for group in trainable_params:
        min_lr = min(min_lr, group.get("lr", lr))

    if min_lr <= 0.1:
      print(
          f'learning rate is too low. If using dadaptation, set learning rate around 1.0 / 学習率が低すぎるようです。1.0前後の値を指定してください: {min_lr}')
      print('recommend option: lr=1.0 / 推奨は1.0です')

    optimizer_class = dadaptation.DAdaptAdam
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

  elif optimizer_type == "Adafactor".lower():
    # 引数を確認して適宜補正する
    if "relative_step" not in optimizer_kwargs:
      optimizer_kwargs["relative_step"] = True                  # default
    if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
      print(f"set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします")
      optimizer_kwargs["relative_step"] = True
    print(f"use Adafactor optimizer | {optimizer_kwargs}")

    if optimizer_kwargs["relative_step"]:
      print(f"relative_step is true / relative_stepがtrueです")
      if lr != 0.0:
        print(f"learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
      args.learning_rate = None

      # trainable_paramsがgroupだった時の処理：lrを削除する
      if type(trainable_params) == list and type(trainable_params[0]) == dict:
        has_group_lr = False
        for group in trainable_params:
          p = group.pop("lr", None)
          has_group_lr = has_group_lr or (p is not None)

        if has_group_lr:
          # 一応argsを無効にしておく TODO 依存関係が逆転してるのであまり望ましくない
          print(f"unet_lr and text_encoder_lr are ignored / unet_lrとtext_encoder_lrは無視されます")
          args.unet_lr = None
          args.text_encoder_lr = None

      if args.lr_scheduler != "adafactor":
        print(f"use adafactor_scheduler / スケジューラにadafactor_schedulerを使用します")
      args.lr_scheduler = f"adafactor:{lr}"                               # ちょっと微妙だけど

      lr = None
    else:
      if args.max_grad_norm != 0.0:
        print(f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません")
      if args.lr_scheduler != "constant_with_warmup":
        print(f"constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
      if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
        print(f"clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")

    optimizer_class = transformers.optimization.Adafactor
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

  elif optimizer_type == "AdamW".lower():
    print(f"use AdamW optimizer | {optimizer_kwargs}")
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

  else:
    # 任意のoptimizerを使う
    optimizer_type = args.optimizer_type   # lowerでないやつ（微妙）
    print(f"use {optimizer_type} | {optimizer_kwargs}")
    if "." not in optimizer_type:
      optimizer_module = torch.optim
    else:
      values = optimizer_type.split(".")
      optimizer_module = importlib.import_module(".".join(values[:-1]))
      optimizer_type = values[-1]

    optimizer_class = getattr(optimizer_module, optimizer_type)
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

  optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
  optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

  return optimizer_name, optimizer_args, optimizer


# Monkeypatch newer get_scheduler() function overridng current version of diffusers.optimizer.get_scheduler
# code is taken from https://github.com/huggingface/diffusers diffusers.optimizer, commit d87cc15977b87160c30abaace3894e802ad9e1e6
# Which is a newer release of diffusers than currently packaged with sd-scripts
# This code can be removed when newer diffusers version (v0.12.1 or greater) is tested and implemented to sd-scripts


def get_scheduler_fix(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: int = 1,
    power: float = 1.0,
):
  """
  Unified API to get any scheduler from its name.
  Args:
      name (`str` or `SchedulerType`):
          The name of the scheduler to use.
      optimizer (`torch.optim.Optimizer`):
          The optimizer that will be used during training.
      num_warmup_steps (`int`, *optional*):
          The number of warmup steps to do. This is not required by all schedulers (hence the argument being
          optional), the function will raise an error if it's unset and the scheduler type requires it.
      num_training_steps (`int``, *optional*):
          The number of training steps to do. This is not required by all schedulers (hence the argument being
          optional), the function will raise an error if it's unset and the scheduler type requires it.
      num_cycles (`int`, *optional*):
          The number of hard restarts used in `COSINE_WITH_RESTARTS` scheduler.
      power (`float`, *optional*, defaults to 1.0):
          Power factor. See `POLYNOMIAL` scheduler
      last_epoch (`int`, *optional*, defaults to -1):
          The index of the last epoch when resuming training.
  """
  if name.startswith("adafactor"):
    assert type(optimizer) == transformers.optimization.Adafactor, f"adafactor scheduler must be used with Adafactor optimizer / adafactor schedulerはAdafactorオプティマイザと同時に使ってください"
    initial_lr = float(name.split(':')[1])
    # print("adafactor scheduler init lr", initial_lr)
    return transformers.optimization.AdafactorSchedule(optimizer, initial_lr)

  name = SchedulerType(name)
  schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
  if name == SchedulerType.CONSTANT:
    return schedule_func(optimizer)

  # All other schedulers require `num_warmup_steps`
  if num_warmup_steps is None:
    raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

  if name == SchedulerType.CONSTANT_WITH_WARMUP:
    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

  # All other schedulers require `num_training_steps`
  if num_training_steps is None:
    raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

  if name == SchedulerType.COSINE_WITH_RESTARTS:
    return schedule_func(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=num_cycles
    )

  if name == SchedulerType.POLYNOMIAL:
    return schedule_func(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, power=power
    )

  return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


def prepare_dataset_args(args: argparse.Namespace, support_metadata: bool):
  # backward compatibility
  if args.caption_extention is not None:
    args.caption_extension = args.caption_extention
    args.caption_extention = None

  if args.cache_latents:
    assert not args.color_aug, "when caching latents, color_aug cannot be used / latentをキャッシュするときはcolor_augは使えません"
    assert not args.random_crop, "when caching latents, random_crop cannot be used / latentをキャッシュするときはrandom_cropは使えません"

  # assert args.resolution is not None, f"resolution is required / resolution（解像度）を指定してください"
  if args.resolution is not None:
    args.resolution = tuple([int(r) for r in args.resolution.split(',')])
    if len(args.resolution) == 1:
      args.resolution = (args.resolution[0], args.resolution[0])
    assert len(args.resolution) == 2, \
        f"resolution must be 'size' or 'width,height' / resolution（解像度）は'サイズ'または'幅','高さ'で指定してください: {args.resolution}"

  if args.face_crop_aug_range is not None:
    args.face_crop_aug_range = tuple([float(r) for r in args.face_crop_aug_range.split(',')])
    assert len(args.face_crop_aug_range) == 2 and args.face_crop_aug_range[0] <= args.face_crop_aug_range[1], \
        f"face_crop_aug_range must be two floats / face_crop_aug_rangeは'下限,上限'で指定してください: {args.face_crop_aug_range}"
  else:
    args.face_crop_aug_range = None

  if support_metadata:
    if args.in_json is not None and (args.color_aug or args.random_crop):
      print(f"latents in npz is ignored when color_aug or random_crop is True / color_augまたはrandom_cropを有効にした場合、npzファイルのlatentsは無視されます")


def load_tokenizer(args: argparse.Namespace):
  print("prepare tokenizer")
  if args.v2:
    tokenizer = CLIPTokenizer.from_pretrained(V2_STABLE_DIFFUSION_PATH, subfolder="tokenizer")
  else:
    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)
  if args.max_token_length is not None:
    print(f"update token length: {args.max_token_length}")
  return tokenizer


def prepare_accelerator(args: argparse.Namespace):
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

  return accelerator, unwrap_model


def prepare_dtype(args: argparse.Namespace):
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

  return weight_dtype, save_dtype


def load_target_model(args: argparse.Namespace, weight_dtype):
  name_or_path = args.pretrained_model_name_or_path
  name_or_path = os.readlink(name_or_path) if os.path.islink(name_or_path) else name_or_path
  load_stable_diffusion_format = os.path.isfile(name_or_path)           # determine SD or Diffusers
  if load_stable_diffusion_format:
    print("load StableDiffusion checkpoint")
    text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, name_or_path)
  else:
    print("load Diffusers pretrained models")
    try:
      pipe = StableDiffusionPipeline.from_pretrained(name_or_path, tokenizer=None, safety_checker=None)
    except EnvironmentError as ex:
      print(
          f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}")
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    del pipe

  # VAEを読み込む
  if args.vae is not None:
    vae = model_util.load_vae(args.vae, weight_dtype)
    print("additional VAE loaded")

  return text_encoder, vae, unet, load_stable_diffusion_format


def patch_accelerator_for_fp16_training(accelerator):
  org_unscale_grads = accelerator.scaler._unscale_grads_

  def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
    return org_unscale_grads(optimizer, inv_scale, found_inf, True)

  accelerator.scaler._unscale_grads_ = _unscale_grads_replacer


def get_hidden_states(args: argparse.Namespace, input_ids, tokenizer, text_encoder, weight_dtype=None):
  # with no_token_padding, the length is not max length, return result immediately
  if input_ids.size()[-1] != tokenizer.model_max_length:
    return text_encoder(input_ids)[0]

  b_size = input_ids.size()[0]
  input_ids = input_ids.reshape((-1, tokenizer.model_max_length))     # batch_size*3, 77

  if args.clip_skip is None:
    encoder_hidden_states = text_encoder(input_ids)[0]
  else:
    enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
    encoder_hidden_states = enc_out['hidden_states'][-args.clip_skip]
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

  if weight_dtype is not None:
    # this is required for additional network training
    encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

  return encoder_hidden_states


def get_epoch_ckpt_name(args: argparse.Namespace, use_safetensors, epoch):
  model_name = DEFAULT_EPOCH_NAME if args.output_name is None else args.output_name
  ckpt_name = EPOCH_FILE_NAME.format(model_name, epoch) + (".safetensors" if use_safetensors else ".ckpt")
  return model_name, ckpt_name


def save_on_epoch_end(args: argparse.Namespace, save_func, remove_old_func, epoch_no: int, num_train_epochs: int):
  saving = epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
  if saving:
    os.makedirs(args.output_dir, exist_ok=True)
    save_func()

    if args.save_last_n_epochs is not None:
      remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
      remove_old_func(remove_epoch_no)
  return saving


def save_sd_model_on_epoch_end(args: argparse.Namespace, accelerator, src_path: str, save_stable_diffusion_format: bool, use_safetensors: bool, save_dtype: torch.dtype, epoch: int, num_train_epochs: int, global_step: int, text_encoder, unet, vae):
  epoch_no = epoch + 1
  model_name, ckpt_name = get_epoch_ckpt_name(args, use_safetensors, epoch_no)

  if save_stable_diffusion_format:
    def save_sd():
      ckpt_file = os.path.join(args.output_dir, ckpt_name)
      print(f"saving checkpoint: {ckpt_file}")
      model_util.save_stable_diffusion_checkpoint(args.v2, ckpt_file, text_encoder, unet,
                                                  src_path, epoch_no, global_step, save_dtype, vae)

    def remove_sd(old_epoch_no):
      _, old_ckpt_name = get_epoch_ckpt_name(args,  use_safetensors, old_epoch_no)
      old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
      if os.path.exists(old_ckpt_file):
        print(f"removing old checkpoint: {old_ckpt_file}")
        os.remove(old_ckpt_file)

    save_func = save_sd
    remove_old_func = remove_sd
  else:
    def save_du():
      out_dir = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, epoch_no))
      print(f"saving model: {out_dir}")
      os.makedirs(out_dir, exist_ok=True)
      model_util.save_diffusers_checkpoint(args.v2, out_dir, text_encoder, unet,
                                           src_path, vae=vae, use_safetensors=use_safetensors)

    def remove_du(old_epoch_no):
      out_dir_old = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, old_epoch_no))
      if os.path.exists(out_dir_old):
        print(f"removing old model: {out_dir_old}")
        shutil.rmtree(out_dir_old)

    save_func = save_du
    remove_old_func = remove_du

  saving = save_on_epoch_end(args, save_func, remove_old_func, epoch_no, num_train_epochs)
  if saving and args.save_state:
    save_state_on_epoch_end(args, accelerator, model_name, epoch_no)


def save_state_on_epoch_end(args: argparse.Namespace, accelerator, model_name, epoch_no):
  print("saving state.")
  accelerator.save_state(os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no)))

  last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
  if last_n_epochs is not None:
    remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
    state_dir_old = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
    if os.path.exists(state_dir_old):
      print(f"removing old state: {state_dir_old}")
      shutil.rmtree(state_dir_old)


def save_sd_model_on_train_end(args: argparse.Namespace, src_path: str, save_stable_diffusion_format: bool, use_safetensors: bool, save_dtype: torch.dtype, epoch: int, global_step: int, text_encoder, unet, vae):
  model_name = DEFAULT_LAST_OUTPUT_NAME if args.output_name is None else args.output_name

  if save_stable_diffusion_format:
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_name = model_name + (".safetensors" if use_safetensors else ".ckpt")
    ckpt_file = os.path.join(args.output_dir, ckpt_name)

    print(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
    model_util.save_stable_diffusion_checkpoint(args.v2, ckpt_file, text_encoder, unet,
                                                src_path, epoch, global_step, save_dtype, vae)
  else:
    out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"save trained model as Diffusers to {out_dir}")
    model_util.save_diffusers_checkpoint(args.v2, out_dir, text_encoder, unet,
                                         src_path, vae=vae, use_safetensors=use_safetensors)


def save_state_on_train_end(args: argparse.Namespace, accelerator):
  print("saving last state.")
  os.makedirs(args.output_dir, exist_ok=True)
  model_name = DEFAULT_LAST_OUTPUT_NAME if args.output_name is None else args.output_name
  accelerator.save_state(os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name)))

# endregion

# region 前処理用


class ImageLoadingDataset(torch.utils.data.Dataset):
  def __init__(self, image_paths):
    self.images = image_paths

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img_path = self.images[idx]

    try:
      image = Image.open(img_path).convert("RGB")
      # convert to tensor temporarily so dataloader will accept it
      tensor_pil = transforms.functional.pil_to_tensor(image)
    except Exception as e:
      print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
      return None

    return (tensor_pil, img_path)


# endregion
