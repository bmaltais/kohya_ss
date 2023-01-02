# common functions for training

import json
from typing import NamedTuple
from torch.autograd.function import Function
import glob
import math
import os
import random

from tqdm import tqdm
import torch
from torchvision import transforms
from transformers import CLIPTokenizer
import diffusers
import albumentations as albu
import numpy as np
from PIL import Image
import cv2
from einops import rearrange
from torch import einsum

import library.model_util as model_util

# Tokenizer: checkpointから読み込むのではなくあらかじめ提供されているものを使う
TOKENIZER_PATH = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2"     # ここからtokenizerだけ使う v2とv2.1はtokenizer仕様は同じ

# checkpointファイル名
EPOCH_STATE_NAME = "epoch-{:06d}-state"
LAST_STATE_NAME = "last-state"

EPOCH_FILE_NAME = "epoch-{:06d}"
LAST_FILE_NAME = "last"


# region dataset

class ImageInfo():
  def __init__(self, image_key: str, num_repeats: int, caption: str, is_reg: bool, absolute_path: str) -> None:
    self.image_key: str = image_key
    self.num_repeats: int = num_repeats
    self.caption: str = caption
    self.is_reg: bool = is_reg
    self.absolute_path: str = absolute_path
    self.image_size: tuple[int, int] = None
    self.bucket_reso: tuple[int, int] = None
    self.latents: torch.Tensor = None
    self.latents_flipped: torch.Tensor = None
    self.latents_npz: str = None
    self.latents_npz_flipped: str = None


class BucketBatchIndex(NamedTuple):
  bucket_index: int
  batch_index: int


class BaseDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer, max_token_length, shuffle_caption, shuffle_keep_tokens, resolution, flip_aug: bool, color_aug: bool, face_crop_aug_range, debug_dataset: bool) -> None:
    super().__init__()
    self.tokenizer: CLIPTokenizer = tokenizer
    self.max_token_length = max_token_length
    self.shuffle_caption = shuffle_caption
    self.shuffle_keep_tokens = shuffle_keep_tokens
    self.width, self.height = resolution
    self.face_crop_aug_range = face_crop_aug_range
    self.flip_aug = flip_aug
    self.color_aug = color_aug
    self.debug_dataset = debug_dataset

    self.tokenizer_max_length = self.tokenizer.model_max_length if max_token_length is None else max_token_length + 2

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

    self.image_data: dict[str, ImageInfo] = {}

  def process_caption(self, caption):
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

  def make_buckets(self, enable_bucket, min_size, max_size):
    '''
    bucketingを行わない場合も呼び出し必須（ひとつだけbucketを作る）
    min_size and max_size are ignored when enable_bucket is False
    '''

    self.enable_bucket = enable_bucket

    print("loading image sizes.")
    for info in tqdm(self.image_data.values()):
      if info.image_size is None:
        info.image_size = self.get_image_size(info.absolute_path)

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

    # bucketを作成する
    if enable_bucket:
      img_ar_errors = []
      for image_info in self.image_data.values():
        # bucketを決める
        image_width, image_height = image_info.image_size
        aspect_ratio = image_width / image_height
        ar_errors = bucket_aspect_ratios - aspect_ratio

        bucket_id = np.abs(ar_errors).argmin()
        image_info.bucket_reso = bucket_resos[bucket_id]

        ar_error = ar_errors[bucket_id]
        img_ar_errors.append(ar_error)
    else:
      reso = (self.width, self.height)
      for image_info in self.image_data.values():
        image_info.bucket_reso = reso

    # 画像をbucketに分割する
    self.buckets: list[str] = [[] for _ in range(len(bucket_resos))]
    reso_to_index = {}
    for i, reso in enumerate(bucket_resos):
      reso_to_index[reso] = i

    for image_info in self.image_data.values():
      bucket_index = reso_to_index[image_info.bucket_reso]
      for _ in range(image_info.num_repeats):
        self.buckets[bucket_index].append(image_info.image_key)

    if enable_bucket:
      print("number of images (including repeats for DreamBooth) / 各bucketの画像枚数（DreamBoothの場合は繰り返し回数を含む）")
      for i, (reso, img_keys) in enumerate(zip(bucket_resos, self.buckets)):
        print(f"bucket {i}: resolution {reso}, count: {len(img_keys)}")
      img_ar_errors = np.array(img_ar_errors)
      print(f"mean ar error (without repeats): {np.mean(np.abs(img_ar_errors))}")

    # 参照用indexを作る
    self.buckets_indices: list(BucketBatchIndex) = []
    for bucket_index, bucket in enumerate(self.buckets):
      batch_count = int(math.ceil(len(bucket) / self.batch_size))
      for batch_index in range(batch_count):
        self.buckets_indices.append(BucketBatchIndex(bucket_index, batch_index))

    self.shuffle_buckets()
    self._length = len(self.buckets_indices)

  def shuffle_buckets(self):
    random.shuffle(self.buckets_indices)
    for bucket in self.buckets:
      random.shuffle(bucket)

  def load_image(self, image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
      image = image.convert("RGB")
    img = np.array(image, np.uint8)
    return img

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

  def cache_latents(self, vae):
    print("caching latents.")
    for info in tqdm(self.image_data.values()):
      if info.latents_npz is not None:
        info.latents = self.load_latents_from_npz(info, False)
        info.latents = torch.FloatTensor(info.latents)
        info.latents_flipped = self.load_latents_from_npz(info, True)
        info.latents_flipped = torch.FloatTensor(info.latents_flipped)
        continue

      image = self.load_image(info.absolute_path)
      image = self.resize_and_trim(image, info.bucket_reso)

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
    return np.load(npz_file)['arr_0']

  def __len__(self):
    return self._length

  def __getitem__(self, index):
    if index == 0:
      self.shuffle_buckets()

    bucket = self.buckets[self.buckets_indices[index].bucket_index]
    image_index = self.buckets_indices[index].batch_index * self.batch_size

    loss_weights = []
    captions = []
    input_ids_list = []
    latents_list = []
    images = []

    for image_key in bucket[image_index:image_index + self.batch_size]:
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
          img = self.resize_and_trim(img, image_info.bucket_reso)
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
      input_ids_list.append(self.get_input_ids(caption))

    example = {}
    example['loss_weights'] = torch.FloatTensor(loss_weights)
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
  def __init__(self, batch_size, train_data_dir, reg_data_dir, tokenizer, max_token_length, caption_extension, shuffle_caption, shuffle_keep_tokens, resolution, prior_loss_weight, flip_aug, color_aug, face_crop_aug_range, random_crop, debug_dataset) -> None:
    super().__init__(tokenizer, max_token_length, shuffle_caption, shuffle_keep_tokens,
                     resolution, flip_aug, color_aug, face_crop_aug_range, debug_dataset)

    self.batch_size = batch_size
    self.size = min(self.width, self.height)                  # 短いほう
    self.prior_loss_weight = prior_loss_weight
    self.random_crop = random_crop
    self.latents_cache = None
    self.enable_bucket = False

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
            lines = f.readlines()
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
      img_paths = glob.glob(os.path.join(dir, "*.png")) + glob.glob(os.path.join(dir, "*.jpg")) + \
          glob.glob(os.path.join(dir, "*.webp"))
      print(f"found directory {n_repeats}_{caption_by_folder} contains {len(img_paths)} image files")

      # 画像ファイルごとにプロンプトを読み込み、もしあればそちらを使う
      captions = []
      for img_path in img_paths:
        cap_for_img = read_caption(img_path)
        captions.append(caption_by_folder if cap_for_img is None else cap_for_img)

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
    print(f"{num_train_images} train images with repeating.")
    self.num_train_images = num_train_images

    # reg imageは数を数えて学習画像と同じ枚数にする
    num_reg_images = 0
    if reg_data_dir:
      print("prepare reg images.")
      reg_infos: list[ImageInfo] = []

      reg_dirs = os.listdir(reg_data_dir)
      for dir in reg_dirs:
        n_repeats, img_paths, captions = load_dreambooth_dir(os.path.join(reg_data_dir, dir))
        num_reg_images += n_repeats * len(img_paths)
        for img_path, caption in zip(img_paths, captions):
          info = ImageInfo(img_path, n_repeats, caption, True, img_path)
          reg_infos.append(info)

      print(f"{num_reg_images} reg images.")
      if num_train_images < num_reg_images:
        print("some of reg images are not used / 正則化画像の数が多いので、一部使用されない正則化画像があります")

      if num_reg_images == 0:
        print("no regularization images / 正則化画像が見つかりませんでした")
      else:
        n = 0
        while n < num_train_images:
          for info in reg_infos:
            self.register_image(info)
            n += info.num_repeats
            if n >= num_train_images:                 # reg画像にnum_repeats>1のときはまずありえないので考慮しない
              break

    self.num_reg_images = num_reg_images


class FineTuningDataset(BaseDataset):
  def __init__(self, json_file_name, batch_size, train_data_dir, tokenizer, max_token_length, shuffle_caption, shuffle_keep_tokens, resolution, flip_aug, color_aug, face_crop_aug_range, dataset_repeats, debug_dataset) -> None:
    super().__init__(tokenizer, max_token_length, shuffle_caption, shuffle_keep_tokens,
                     resolution, flip_aug, color_aug, face_crop_aug_range, debug_dataset)

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

    for image_key, img_md in metadata.items():
      # path情報を作る
      if os.path.exists(image_key):
        abs_path = image_key
      else:
        # わりといい加減だがいい方法が思いつかん
        abs_path = (glob.glob(os.path.join(train_data_dir, f"{image_key}.png")) + glob.glob(os.path.join(train_data_dir, f"{image_key}.jpg")) +
                    glob.glob(os.path.join(train_data_dir, f"{image_key}.webp")))
        assert len(abs_path) >= 1, f"no image / 画像がありません: {abs_path}"
        abs_path = abs_path[0]

      caption = img_md.get('caption')
      tags = img_md.get('tags')
      if caption is None:
        caption = tags
      elif tags is not None and len(tags) > 0:
        caption = caption + ', ' + tags
      assert caption is not None and len(caption) > 0, f"caption or tag is required / キャプションまたはタグは必須です:{abs_path}"

      image_info = ImageInfo(image_key, dataset_repeats, caption, False, abs_path)
      image_info.image_size = img_md.get('train_resolution')

      if not self.color_aug:
        # if npz exists, use them
        image_info.latents_npz, image_info.latents_npz_flipped = self.image_key_to_npz_file(image_key)

      self.register_image(image_info)
    self.num_train_images = len(metadata) * dataset_repeats
    self.num_reg_images = 0

    # check existence of all npz files
    if not self.color_aug:
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
        print(f"npz file does not exist. make latents with VAE / npzファイルが見つからないためVAEを使ってlatentsを取得します")
      elif not npz_all:
        print(f"some of npz file does not exist. ignore npz files / いくつかのnpzファイルが見つからないためnpzファイルを無視します")
        for image_info in self.image_data.values():
          image_info.latents_npz = image_info.latents_npz_flipped = None

    # check min/max bucket size
    sizes = set()
    for image_info in self.image_data.values():
      if image_info.image_size is None:
        sizes = None                  # not calculated
        break
      sizes.add(image_info.image_size[0])
      sizes.add(image_info.image_size[1])

    if sizes is None:
      self.min_bucket_reso = self.max_bucket_reso = None                # set as not calculated
    else:
      self.min_bucket_reso = min(sizes)
      self.max_bucket_reso = max(sizes)

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
