# このスクリプトのライセンスは、train_dreambooth.pyと同じくApache License 2.0とします
# The license of this script, like train_dreambooth.py, is Apache License 2.0
# (c) 2022 Kohya S. @kohya_ss

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
from transformers import CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
import albumentations as albu
import numpy as np
from PIL import Image
import cv2
from einops import rearrange
from torch import einsum

# Tokenizer: checkpointから読み込むのではなくあらかじめ提供されているものを使う
# Tokenizer: use the one provided beforehand instead of reading from checkpoints
TOKENIZER_PATH = "openai/clip-vit-large-patch14"

# StableDiffusionのモデルパラメータ
# StableDiffusion model parameters
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

# checkpointファイル名
# checkpoint filename
LAST_CHECKPOINT_NAME = "last.ckpt"
EPOCH_CHECKPOINT_NAME = "epoch-{:06d}.ckpt"


class DreamBoothOrFineTuningDataset(torch.utils.data.Dataset):
  def __init__(self, fine_tuning, train_img_path_captions, reg_img_path_captions, tokenizer, resolution, prior_loss_weight, flip_aug, color_aug, face_crop_aug_range, random_crop, shuffle_caption, disable_padding, debug_dataset) -> None:
    super().__init__()

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

    # augmentation
    flip_p = 0.5 if flip_aug else 0.0
    if color_aug:
      # わりと弱めの色合いaugmentation：brightness/contrastあたりは画像のpixel valueの最大値・最小値を変えてしまうのでよくないのではという想定でgamma/hue/saturationあたりを触る
      # Weak tint augmentation: touch gamma/hue/saturation on the assumption that brightness/contrast is not good because it changes the maximum and minimum pixel value of the image.
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

    if self.fine_tuning:
      self._length = len(self.train_img_path_captions)
    else:
      # 学習データの倍として、奇数ならtrain
      # train as double the training data, train if odd
      self._length = len(self.train_img_path_captions) * 2
      if self._length // 2 < len(self.reg_img_path_captions):
        print("some of reg images are not used / Due to the large number of regularized images, some regularized images are not used")

    self.image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

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
  # Cutting it out for good
  def crop_target(self, image, face_cx, face_cy, face_w, face_h):
    height, width = image.shape[0:2]
    if height == self.height and width == self.width:
      return image

    # 画像サイズはsizeより大きいのでリサイズする
    # Resize the image size because it is larger than size
    face_size = max(face_w, face_h)
    min_scale = max(self.height / height, self.width / width)        # 画像がモデル入力サイズぴったりになる倍率（最小の倍率）# Magnification at which the image exactly matches the model input size (minimum magnification)
    min_scale = min(1.0, max(min_scale, self.size / (face_size * self.face_crop_aug_range[1])))             # 指定した顔最小サイズ # Minimum size of the specified face
    max_scale = min(1.0, max(min_scale, self.size / (face_size * self.face_crop_aug_range[0])))             # 指定した顔最大サイズ # Minimum size of the specified face
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

    # Cut out 448*640 or so centered on the face.
    for axis, (target_size, length, face_p) in enumerate(zip((self.height, self.width), (height, width), (face_cy, face_cx))):
      p1 = face_p - target_size // 2                # 顔を中心に持ってくるための切り出し位置 # Cutout position to bring the face to the center

      if self.random_crop:
        # 背景も含めるために顔を中心に置く確率を高めつつずらす
        # Shift while increasing the probability of centering the face to include the background
        range = max(length - face_p, face_p)        # 画像の端から顔中心までの距離の長いほう # Longer distance from the edge of the image to the center of the face
        p1 = p1 + (random.randint(0, range) + random.randint(0, range)) - range     # -range ~ +range までのいい感じの乱数 # nice random numbers from -range to +range
      else:
        # range指定があるときのみ、すこしだけランダムに（わりと適当）
        # Only when a range is specified, a little bit random (rather appropriate)
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

  def set_cached_latents(self, image_path, latents):
    if self.latents_cache is None:
      self.latents_cache = {}
    self.latents_cache[image_path] = latents

  def __getitem__(self, index_arg):
    example = {}

    if self.fine_tuning or len(self.reg_img_path_captions) == 0:
      index = index_arg
      img_path_captions = self.train_img_path_captions
      reg = False
    else:
      # 偶数ならtrain、奇数ならregを返す
      # Return train for even numbers, reg for odd numbers
      if index_arg % 2 == 0:
        img_path_captions = self.train_img_path_captions
        reg = False
      else:
        img_path_captions = self.reg_img_path_captions
        reg = True
      index = index_arg // 2
    example['loss_weight'] = 1.0 if (not reg or self.fine_tuning) else self.prior_loss_weight

    index = index % len(img_path_captions)
    image_path, caption = img_path_captions[index]
    example['image_path'] = image_path

    # image/latentsを処理する
    # process images/latents
    if self.latents_cache is not None and image_path in self.latents_cache:
      # latentsはキャッシュ済み
      example['latents'] = self.latents_cache[image_path]
    else:
      # 画像を読み込み必要ならcropする
      # load images and crop if necessary
      img, face_cx, face_cy, face_w, face_h = self.load_image(image_path)
      im_h, im_w = img.shape[0:2]
      if face_cx > 0:                   # 顔位置情報あり # With face location information
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
      assert im_h == self.height and im_w == self.width, f"image too small / 画像サイズが小さいようです: {image_path}"

      # augmentation
      if self.aug is not None:
        img = self.aug(image=img)['image']

      example['image'] = self.image_transforms(img)      # -1.0~1.0のtorch.Tensorになる

    # captionを処理する
    if self.fine_tuning and self.shuffle_caption:         # fine tuning時にcaptionのshuffleをする
      tokens = caption.strip().split(",")
      random.shuffle(tokens)
      caption = ",".join(tokens).strip()

    input_ids = self.tokenizer(caption, padding="do_not_pad", truncation=True,
                               max_length=self.tokenizer.model_max_length).input_ids

    # padしてTensor変換
    if self.disable_padding:
      # paddingしない：padding==Trueはバッチの中の最大長に合わせるだけ（やはりバグでは……？）
      input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
    else:
      # paddingする
      input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding='max_length', max_length=self.tokenizer.model_max_length,
                                     return_tensors='pt').input_ids

    example['input_ids'] = input_ids

    if self.debug_dataset:
      example['caption'] = caption
    return example


# checkpoint変換など ###############################

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


def convert_ldm_unet_checkpoint(checkpoint, config):
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


def create_unet_diffusers_config():
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
      cross_attention_dim=UNET_PARAMS_CONTEXT_DIM,
      attention_head_dim=UNET_PARAMS_NUM_HEADS,
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


def convert_ldm_clip_checkpoint(checkpoint):
  text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

  keys = list(checkpoint.keys())

  text_model_dict = {}

  for key in keys:
    if key.startswith("cond_stage_model.transformer"):
      text_model_dict[key[len("cond_stage_model.transformer."):]] = checkpoint[key]

  text_model.load_state_dict(text_model_dict)

  return text_model

# endregion


# region Diffusers->StableDiffusion の変換コード
# convert_diffusers_to_original_stable_diffusion をコピーしている（ASL 2.0）

def convert_unet_state_dict(unet_state_dict):
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
  return new_state_dict

# endregion


def load_stable_diffusion_checkpoint(ckpt_path):
  checkpoint = torch.load(ckpt_path, map_location="cpu")["state_dict"]

  # Convert the UNet2DConditionModel model.
  unet_config = create_unet_diffusers_config()
  converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config)

  unet = UNet2DConditionModel(**unet_config)
  unet.load_state_dict(converted_unet_checkpoint)

  # Convert the VAE model.
  vae_config = create_vae_diffusers_config()
  converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

  vae = AutoencoderKL(**vae_config)
  vae.load_state_dict(converted_vae_checkpoint)

  # convert text_model
  text_model = convert_ldm_clip_checkpoint(checkpoint)

  return text_model, vae, unet


def save_stable_diffusion_checkpoint(output_file, text_encoder, unet, ckpt_path):
  # VAEがメモリ上にないので、もう一度VAEを含めて読み込む
  state_dict = torch.load(ckpt_path, map_location="cpu")['state_dict']

  # Convert the UNet model
  unet_state_dict = convert_unet_state_dict(unet.state_dict())
  for k, v in unet_state_dict.items():
    key = "model.diffusion_model." + k
    assert key in state_dict, f"Illegal key in save SD: {key}"
    state_dict[key] = v

  # Convert the text encoder model
  text_enc_dict = text_encoder.state_dict()             # 変換不要
  for k, v in text_enc_dict.items():
    key = "cond_stage_model.transformer." + k
    assert key in state_dict, f"Illegal key in save SD: {key}"
    state_dict[key] = v

  # Put together new checkpoint
  state_dict = {"state_dict": state_dict}
  torch.save(state_dict, output_file)


def collate_fn(examples):
  input_ids = [e['input_ids'] for e in examples]
  input_ids = torch.stack(input_ids)

  if 'latents' in examples[0]:
    pixel_values = None
    latents = [e['latents'] for e in examples]
    latents = torch.stack(latents)
  else:
    pixel_values = [e['image'] for e in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    latents = None

  loss_weights = [e['loss_weight'] for e in examples]
  loss_weights = torch.FloatTensor(loss_weights)

  batch = {"input_ids": input_ids, "pixel_values": pixel_values, "latents": latents, "loss_weights": loss_weights}
  return batch


def train(args):
  fine_tuning = args.fine_tuning
  cache_latents = args.cache_latents

  # latentsをキャッシュする場合のオプション設定を確認する
  if cache_latents:
    # assert args.face_crop_aug_range is None and not args.random_crop, "when caching latents, crop aug cannot be used / latentをキャッシュするときは切り出しは使えません"
    # →使えるようにしておく（初期イメージの切り出しになる）
    assert not args.flip_aug and not args.color_aug, "when caching latents, augmentation cannot be used / latentをキャッシュするときはaugmentationは使えません"

  # モデル形式のオプション設定を確認する
  use_stable_diffusion_format = os.path.isfile(args.pretrained_model_name_or_path)
  if not use_stable_diffusion_format:
    assert os.path.exists(
        args.pretrained_model_name_or_path), f"no pretrained model / 学習元モデルがありません : {args.pretrained_model_name_or_path}"

  assert args.save_every_n_epochs is None or use_stable_diffusion_format, "when loading Diffusers model, save_every_n_epochs does not work / Diffusersのモデルを読み込むときにはsave_every_n_epochsオプションは無効になります"

  if args.seed is not None:
    set_seed(args.seed)

  # 学習データを用意する
  def load_dreambooth_dir(dir):
    tokens = os.path.basename(dir).split('_')
    try:
      n_repeats = int(tokens[0])
    except ValueError as e:
      print(f"no 'n_repeats' in directory name / DreamBoothのディレクトリ名に繰り返し回数がないようです: {dir}")
      raise e

    caption = '_'.join(tokens[1:])

    img_paths = glob.glob(os.path.join(dir, "*.png")) + glob.glob(os.path.join(dir, "*.jpg"))
    return n_repeats, [(ip, caption) for ip in img_paths]

  print("prepare train images.")
  train_img_path_captions = []

  if fine_tuning:
    img_paths = glob.glob(os.path.join(args.train_data_dir, "*.png")) + glob.glob(os.path.join(args.train_data_dir, "*.jpg"))
    for img_path in tqdm(img_paths):
      # captionの候補ファイル名を作る
      base_name = os.path.splitext(img_path)[0]
      base_name_face_det = base_name
      tokens = base_name.split("_")
      if len(tokens) >= 5:
        base_name_face_det = "_".join(tokens[:-4])
      cap_paths = [base_name + '.txt', base_name + '.caption', base_name_face_det+'.txt', base_name_face_det+'.caption']

      caption = None
      for cap_path in cap_paths:
        if os.path.isfile(cap_path):
          with open(cap_path, "rt", encoding='utf-8') as f:
            caption = f.readlines()[0].strip()
          break

      assert caption is not None and len(caption) > 0, f"no caption / キャプションファイルが見つからないか、captionが空です: {cap_paths}"

      train_img_path_captions.append((img_path, caption))

  else:
    train_dirs = os.listdir(args.train_data_dir)
    for dir in train_dirs:
      n_repeats, img_caps = load_dreambooth_dir(os.path.join(args.train_data_dir, dir))
      for _ in range(n_repeats):
        train_img_path_captions.extend(img_caps)
  print(f"{len(train_img_path_captions)} train images.")

  if fine_tuning:
    reg_img_path_captions = []
  else:
    print("prepare reg images.")
    reg_img_path_captions = []
    if args.reg_data_dir:
      reg_dirs = os.listdir(args.reg_data_dir)
      for dir in reg_dirs:
        n_repeats, img_caps = load_dreambooth_dir(os.path.join(args.reg_data_dir, dir))
        for _ in range(n_repeats):
          reg_img_path_captions.extend(img_caps)
    print(f"{len(reg_img_path_captions)} reg images.")

  if args.debug_dataset:
    # デバッグ時はshuffleして実際のデータセット使用時に近づける（学習時はdata loaderでshuffleする）
    random.shuffle(train_img_path_captions)
    random.shuffle(reg_img_path_captions)

  # データセットを準備する
  resolution = tuple([int(r) for r in args.resolution.split(',')])
  if len(resolution) == 1:
    resolution = (resolution[0], resolution[0])
  assert len(
      resolution) == 2, f"resolution must be 'size' or 'width,height' / resolutionは'サイズ'または'幅','高さ'で指定してください: {args.resolution}"

  if args.face_crop_aug_range is not None:
    face_crop_aug_range = tuple([float(r) for r in args.face_crop_aug_range.split(',')])
    assert len(
        face_crop_aug_range) == 2, f"face_crop_aug_range must be two floats / face_crop_aug_rangeは'下限,上限'で指定してください: {args.face_crop_aug_range}"
  else:
    face_crop_aug_range = None

  # tokenizerを読み込む
  print("prepare tokenizer")
  tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)

  print("prepare dataset")
  train_dataset = DreamBoothOrFineTuningDataset(fine_tuning, train_img_path_captions,
                                                reg_img_path_captions, tokenizer, resolution, args.prior_loss_weight, args.flip_aug, args.color_aug, face_crop_aug_range, args.random_crop, args.shuffle_caption, args.no_token_padding, args.debug_dataset)

  if args.debug_dataset:
    print(f"Total dataset length / データセットの長さ: {len(train_dataset)}")
    print("Escape for exit. / Escキーで中断、終了します")
    for example in train_dataset:
      im = example['image']
      im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
      im = np.transpose(im, (1, 2, 0))                # c,H,W -> H,W,c
      im = im[:, :, ::-1]                             # RGB -> BGR (OpenCV)
      print(f'caption: "{example["caption"]}", loss weight: {example["loss_weight"]}')
      cv2.imshow("img", im)
      k = cv2.waitKey()
      cv2.destroyAllWindows()
      if k == 27:
        break
    return

  # acceleratorを準備する
  # gradient accumulationは複数モデルを学習する場合には対応していないとのことなので、1固定にする
  print("prepare accelerator")
  accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision=args.mixed_precision)

  # モデルを読み込む
  if use_stable_diffusion_format:
    print("load StableDiffusion checkpoint")
    text_encoder, vae, unet = load_stable_diffusion_checkpoint(args.pretrained_model_name_or_path)
  else:
    print("load Diffusers pretrained models")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

  # モデルに xformers とか memory efficient attention を組み込む
  replace_unet_modules(unet, args.mem_eff_attn, args.xformers)

  # mixed precisionに対応した型を用意しておき適宜castする
  weight_dtype = torch.float32
  if args.mixed_precision == "fp16":
    weight_dtype = torch.float16
  elif args.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

  # 学習を準備する
  if cache_latents:
    # latentをcacheする→新しいDatasetを作るとcaptionのshuffleが効かないので元のDatasetにcacheを持つ（cascadeする手もあるが）
    print("caching latents.")
    vae.to(accelerator.device, dtype=weight_dtype)

    for i in tqdm(range(len(train_dataset))):
      example = train_dataset[i]
      if 'latents' not in example:
        image_path = example['image_path']
        with torch.no_grad():
          pixel_values = example["image"].unsqueeze(0).to(device=accelerator.device, dtype=weight_dtype)
          latents = vae.encode(pixel_values).latent_dist.sample().squeeze(0).to("cpu")
          train_dataset.set_cached_latents(image_path, latents)
    # assertion
    for i in range(len(train_dataset)):
      assert 'latents' in train_dataset[i], "internal error: latents not cached"

    del vae
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  else:
    vae.requires_grad_(False)

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
    print("use 8-bit Adma optimizer")
    optimizer_class = bnb.optim.AdamW8bit
  else:
    optimizer_class = torch.optim.AdamW

  trainable_params = (itertools.chain(unet.parameters(), text_encoder.parameters()))

  # betaやweight decayはdiffusers DreamBoothもDreamBooth SDもデフォルト値のようなのでオプションはとりあえず省略
  optimizer = optimizer_class(trainable_params, lr=args.learning_rate)

  # dataloaderを準備する
  # DataLoaderのプロセス数：0はメインプロセスになる
  n_workers = min(4, os.cpu_count() - 1)      # cpu_count-1 ただし最大4
  train_dataloader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=n_workers)

  # lr schedulerを用意する
  lr_scheduler = diffusers.optimization.get_scheduler("constant", optimizer, num_training_steps=args.max_train_steps)

  # acceleratorがなんかよろしくやってくれるらしい
  unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
      unet, text_encoder, optimizer, train_dataloader, lr_scheduler)

  if not cache_latents:
    vae.to(accelerator.device, dtype=weight_dtype)

  # epoch数を計算する
  num_train_epochs = math.ceil(args.max_train_steps / len(train_dataloader))

  # 学習する
  total_batch_size = args.train_batch_size  # * accelerator.num_processes
  print("running training / 学習開始")
  print(f"  num examples / サンプル数: {len(train_dataset)}")
  print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
  print(f"  num epochs / epoch数: {num_train_epochs}")
  print(f"  batch size per device / バッチサイズ: {args.train_batch_size}")
  print(f"  total train batch size (with parallel & distributed) / 総バッチサイズ（並列学習含む）: {total_batch_size}")
  print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

  progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="steps")
  global_step = 0

  noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

  if accelerator.is_main_process:
    accelerator.init_trackers("dreambooth")

  # 以下 train_dreambooth.py からほぼコピペ
  for epoch in range(num_train_epochs):
    print(f"epoch {epoch+1}/{num_train_epochs}")
    unet.train()
    text_encoder.train()        # なんかunetだけでいいらしい？→最新版で修正されてた(;´Д｀)　いろいろ雑だな

    for step, batch in enumerate(train_dataloader):
      with accelerator.accumulate(unet):
        with torch.no_grad():
          # latentに変換
          if cache_latents:
            latents = batch["latents"].to(accelerator.device)
          else:
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
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

        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="none")
        loss = loss.mean([1, 2, 3])

        loss_weights = batch["loss_weights"]                      # 各sampleごとのweight
        loss = loss * loss_weights

        loss = loss.mean()

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

      logs = {"loss": loss.detach().item()}  # , "lr": lr_scheduler.get_last_lr()[0]}
      progress_bar.set_postfix(**logs)
      # accelerator.log(logs, step=global_step)

      if global_step >= args.max_train_steps:
        break

    accelerator.wait_for_everyone()

    if use_stable_diffusion_format and args.save_every_n_epochs is not None:
      if (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs:
        print("saving check point.")
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, EPOCH_CHECKPOINT_NAME.format(epoch + 1))
        save_stable_diffusion_checkpoint(ckpt_file, accelerator.unwrap_model(
            text_encoder), accelerator.unwrap_model(unet), args.pretrained_model_name_or_path)

  is_main_process = accelerator.is_main_process
  if is_main_process:
    unet = accelerator.unwrap_model(unet)
    text_encoder = accelerator.unwrap_model(text_encoder)

  accelerator.end_training()
  del accelerator                         # この後メモリを使うのでこれは消す

  if is_main_process:
    os.makedirs(args.output_dir, exist_ok=True)
    if use_stable_diffusion_format:
      print(f"save trained model as StableDiffusion checkpoint to {args.output_dir}")
      ckpt_file = os.path.join(args.output_dir, LAST_CHECKPOINT_NAME)
      save_stable_diffusion_checkpoint(ckpt_file, text_encoder, unet, args.pretrained_model_name_or_path)
    else:
      # Create the pipeline using using the trained modules and save it.
      print(f"save trained model as Diffusers to {args.output_dir}")
      pipeline = StableDiffusionPipeline.from_pretrained(
          args.pretrained_model_name_or_path,
          unet=unet,
          text_encoder=text_encoder,
      )
      pipeline.save_pretrained(args.output_dir)
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
    return self.to_out(out)

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

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)        # 最適なのを選んでくれる

    out = rearrange(out, 'b n h d -> b n (h d)', h=h)
    return self.to_out(out)

  diffusers.models.attention.CrossAttention.forward = forward_xformers
# endregion


if __name__ == '__main__':
  # torch.cuda.set_per_process_memory_fraction(0.48)
  parser = argparse.ArgumentParser()
  parser.add_argument("--pretrained_model_name_or_path", type=str, default=None,
                      help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint / 学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル")
  parser.add_argument("--fine_tuning", action="store_true",
                      help="fine tune the model instead of DreamBooth / DreamBoothではなくfine tuningする")
  parser.add_argument("--shuffle_caption", action="store_true",
                      help="shuffle comma-separated caption when fine tuning / fine tuning時にコンマで区切られたcaptionの各要素をshuffleする")
  parser.add_argument("--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("--reg_data_dir", type=str, default=None, help="directory for regularization images / 正則化画像データのディレクトリ")
  parser.add_argument("--output_dir", type=str, default=None,
                      help="directory to output trained model, save as same format as input / 学習後のモデル出力先ディレクトリ（入力と同じ形式で保存）")
  parser.add_argument("--save_every_n_epochs", type=int, default=None,
                      help="save checkpoint every N epochs (only supports in StableDiffusion checkpoint) / 学習中のモデルを指定エポックごとに保存します（StableDiffusion形式のモデルを読み込んだ場合のみ有効）")
  parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="loss weight for regularization images / 正則化画像のlossの重み")
  parser.add_argument("--no_token_padding", action="store_true",
                      help="disable token padding (same as Diffuser's DreamBooth) / トークンのpaddingを無効にする（Diffusers版DreamBoothと同じ動作）")
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
  parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
  parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
  parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
  parser.add_argument("--gradient_checkpointing", action="store_true",
                      help="enable gradient checkpointing / grandient checkpointingを有効にする")
  parser.add_argument("--mixed_precision", type=str, default="no",
                      choices=["no", "fp16", "bf16"], help="use mixed precision / 混合精度を使う場合、その精度")
  parser.add_argument("--clip_skip", type=int, default=None,
                      help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）")

  args = parser.parse_args()
  train(args)
