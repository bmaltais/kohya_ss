# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from typing import List
import numpy as np
import torch

from library import train_util


class LoRAModule(torch.nn.Module):
  """
  replaces forward method of the original Linear, instead of replacing the original Linear module.
  """

  def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_dim=4, alpha=1):
    """ if alpha == 0 or None, alpha is rank (no scaling). """
    super().__init__()
    self.lora_name = lora_name

    if org_module.__class__.__name__ == 'Conv2d':
      in_dim = org_module.in_channels
      out_dim = org_module.out_channels
    else:
      in_dim = org_module.in_features
      out_dim = org_module.out_features

    # if limit_rank:
    #   self.lora_dim = min(lora_dim, in_dim, out_dim)
    #   if self.lora_dim != lora_dim:
    #     print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
    # else:
    self.lora_dim = lora_dim

    if org_module.__class__.__name__ == 'Conv2d':
      kernel_size = org_module.kernel_size
      stride = org_module.stride
      padding = org_module.padding
      self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
      self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
    else:
      self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
      self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

    if type(alpha) == torch.Tensor:
      alpha = alpha.detach().float().numpy()                              # without casting, bf16 causes error
    alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
    self.scale = alpha / self.lora_dim
    self.register_buffer('alpha', torch.tensor(alpha))                    # 定数として扱える

    # same as microsoft's
    torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
    torch.nn.init.zeros_(self.lora_up.weight)

    self.multiplier = multiplier
    self.org_module = org_module                  # remove in applying
    self.region = None
    self.region_mask = None

  def apply_to(self):
    self.org_forward = self.org_module.forward
    self.org_module.forward = self.forward
    del self.org_module

  def set_region(self, region):
    self.region = region
    self.region_mask = None

  def forward(self, x):
    if self.region is None:
      return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

    # regional LoRA   FIXME same as additional-network extension
    if x.size()[1] % 77 == 0:
      # print(f"LoRA for context: {self.lora_name}")
      self.region = None
      return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

    # calculate region mask first time
    if self.region_mask is None:
      if len(x.size()) == 4:
        h, w = x.size()[2:4]
      else:
        seq_len = x.size()[1]
        ratio = math.sqrt((self.region.size()[0] * self.region.size()[1]) / seq_len)
        h = int(self.region.size()[0] / ratio + .5)
        w = seq_len // h

      r = self.region.to(x.device)
      if r.dtype == torch.bfloat16:
        r = r.to(torch.float)
      r = r.unsqueeze(0).unsqueeze(1)
      # print(self.lora_name, self.region.size(), x.size(), r.size(), h, w)
      r = torch.nn.functional.interpolate(r, (h, w), mode='bilinear')
      r = r.to(x.dtype)

      if len(x.size()) == 3:
        r = torch.reshape(r, (1, x.size()[1], -1))

      self.region_mask = r

    return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale * self.region_mask


def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
  if network_dim is None:
    network_dim = 4                     # default

  # extract dim/alpha for conv2d, and block dim
  conv_dim = kwargs.get('conv_dim', None)
  conv_alpha = kwargs.get('conv_alpha', None)
  if conv_dim is not None:
    conv_dim = int(conv_dim)
    if conv_alpha is None:
      conv_alpha = 1.0
    else:
      conv_alpha = float(conv_alpha)

  """
  block_dims = kwargs.get("block_dims")
  block_alphas = None

  if block_dims is not None:
    block_dims = [int(d) for d in block_dims.split(',')]
    assert len(block_dims) == NUM_BLOCKS, f"Number of block dimensions is not same to {NUM_BLOCKS}"
    block_alphas = kwargs.get("block_alphas")
    if block_alphas is None:
      block_alphas = [1] * len(block_dims)
    else:
      block_alphas = [int(a) for a in block_alphas(',')]
    assert len(block_alphas) == NUM_BLOCKS, f"Number of block alphas is not same to {NUM_BLOCKS}"

  conv_block_dims = kwargs.get("conv_block_dims")
  conv_block_alphas = None

  if conv_block_dims is not None:
    conv_block_dims = [int(d) for d in conv_block_dims.split(',')]
    assert len(conv_block_dims) == NUM_BLOCKS, f"Number of block dimensions is not same to {NUM_BLOCKS}"
    conv_block_alphas = kwargs.get("conv_block_alphas")
    if conv_block_alphas is None:
      conv_block_alphas = [1] * len(conv_block_dims)
    else:
      conv_block_alphas = [int(a) for a in conv_block_alphas(',')]
    assert len(conv_block_alphas) == NUM_BLOCKS, f"Number of block alphas is not same to {NUM_BLOCKS}"
  """

  network = LoRANetwork(text_encoder, unet, multiplier=multiplier, lora_dim=network_dim,
                        alpha=network_alpha, conv_lora_dim=conv_dim, conv_alpha=conv_alpha)
  return network


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, **kwargs):
  if weights_sd is None:
    if os.path.splitext(file)[1] == '.safetensors':
      from safetensors.torch import load_file, safe_open
      weights_sd = load_file(file)
    else:
      weights_sd = torch.load(file, map_location='cpu')

  # get dim/alpha mapping
  modules_dim = {}
  modules_alpha = {}
  for key, value in weights_sd.items():
    if '.' not in key:
      continue

    lora_name = key.split('.')[0]
    if 'alpha' in key:
      modules_alpha[lora_name] = value
    elif 'lora_down' in key:
      dim = value.size()[0]
      modules_dim[lora_name] = dim
      # print(lora_name, value.size(), dim)

  # support old LoRA without alpha
  for key in modules_dim.keys():
    if key not in modules_alpha:
      modules_alpha = modules_dim[key]

  network = LoRANetwork(text_encoder, unet, multiplier=multiplier, modules_dim=modules_dim, modules_alpha=modules_alpha)
  network.weights_sd = weights_sd
  return network


class LoRANetwork(torch.nn.Module):
  # is it possible to apply conv_in and conv_out?
  UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
  UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
  TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
  LORA_PREFIX_UNET = 'lora_unet'
  LORA_PREFIX_TEXT_ENCODER = 'lora_te'

  def __init__(self, text_encoder, unet, multiplier=1.0, lora_dim=4, alpha=1, conv_lora_dim=None, conv_alpha=None, modules_dim=None, modules_alpha=None) -> None:
    super().__init__()
    self.multiplier = multiplier

    self.lora_dim = lora_dim
    self.alpha = alpha
    self.conv_lora_dim = conv_lora_dim
    self.conv_alpha = conv_alpha

    if modules_dim is not None:
      print(f"create LoRA network from weights")
    else:
      print(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")

    self.apply_to_conv2d_3x3 = self.conv_lora_dim is not None
    if self.apply_to_conv2d_3x3:
      if self.conv_alpha is None:
        self.conv_alpha = self.alpha
      print(f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}")

    # create module instances
    def create_modules(prefix, root_module: torch.nn.Module, target_replace_modules) -> List[LoRAModule]:
      loras = []
      for name, module in root_module.named_modules():
        if module.__class__.__name__ in target_replace_modules:
          # TODO get block index here
          for child_name, child_module in module.named_modules():
            is_linear = child_module.__class__.__name__ == "Linear"
            is_conv2d = child_module.__class__.__name__ == "Conv2d"
            is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)
            if is_linear or is_conv2d:
              lora_name = prefix + '.' + name + '.' + child_name
              lora_name = lora_name.replace('.', '_')

              if modules_dim is not None:
                if lora_name not in modules_dim:
                  continue                                      # no LoRA module in this weights file
                dim = modules_dim[lora_name]
                alpha = modules_alpha[lora_name]
              else:
                if is_linear or is_conv2d_1x1:
                  dim = self.lora_dim
                  alpha = self.alpha
                elif self.apply_to_conv2d_3x3:
                  dim = self.conv_lora_dim
                  alpha = self.conv_alpha
                else:
                  continue

              lora = LoRAModule(lora_name, child_module, self.multiplier, dim, alpha)
              loras.append(lora)
      return loras

    self.text_encoder_loras = create_modules(LoRANetwork.LORA_PREFIX_TEXT_ENCODER,
                                             text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
    print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

    # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
    target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
    if modules_dim is not None or self.conv_lora_dim is not None:
      target_modules += LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

    self.unet_loras = create_modules(LoRANetwork.LORA_PREFIX_UNET, unet, target_modules)
    print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

    self.weights_sd = None

    # assertion
    names = set()
    for lora in self.text_encoder_loras + self.unet_loras:
      assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
      names.add(lora.lora_name)

  def set_multiplier(self, multiplier):
    self.multiplier = multiplier
    for lora in self.text_encoder_loras + self.unet_loras:
      lora.multiplier = self.multiplier

  def load_weights(self, file):
    if os.path.splitext(file)[1] == '.safetensors':
      from safetensors.torch import load_file, safe_open
      self.weights_sd = load_file(file)
    else:
      self.weights_sd = torch.load(file, map_location='cpu')

  def apply_to(self, text_encoder, unet, apply_text_encoder=None, apply_unet=None):
    if self.weights_sd:
      weights_has_text_encoder = weights_has_unet = False
      for key in self.weights_sd.keys():
        if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
          weights_has_text_encoder = True
        elif key.startswith(LoRANetwork.LORA_PREFIX_UNET):
          weights_has_unet = True

      if apply_text_encoder is None:
        apply_text_encoder = weights_has_text_encoder
      else:
        assert apply_text_encoder == weights_has_text_encoder, f"text encoder weights: {weights_has_text_encoder} but text encoder flag: {apply_text_encoder} / 重みとText Encoderのフラグが矛盾しています"

      if apply_unet is None:
        apply_unet = weights_has_unet
      else:
        assert apply_unet == weights_has_unet, f"u-net weights: {weights_has_unet} but u-net flag: {apply_unet} / 重みとU-Netのフラグが矛盾しています"
    else:
      assert apply_text_encoder is not None and apply_unet is not None, f"internal error: flag not set"

    if apply_text_encoder:
      print("enable LoRA for text encoder")
    else:
      self.text_encoder_loras = []

    if apply_unet:
      print("enable LoRA for U-Net")
    else:
      self.unet_loras = []

    for lora in self.text_encoder_loras + self.unet_loras:
      lora.apply_to()
      self.add_module(lora.lora_name, lora)

    if self.weights_sd:
      # if some weights are not in state dict, it is ok because initial LoRA does nothing (lora_up is initialized by zeros)
      info = self.load_state_dict(self.weights_sd, False)
      print(f"weights are loaded: {info}")

  def enable_gradient_checkpointing(self):
    # not supported
    pass

  def prepare_optimizer_params(self, text_encoder_lr, unet_lr):
    def enumerate_params(loras):
      params = []
      for lora in loras:
        params.extend(lora.parameters())
      return params

    self.requires_grad_(True)
    all_params = []

    if self.text_encoder_loras:
      param_data = {'params': enumerate_params(self.text_encoder_loras)}
      if text_encoder_lr is not None:
        param_data['lr'] = text_encoder_lr
      all_params.append(param_data)

    if self.unet_loras:
      param_data = {'params': enumerate_params(self.unet_loras)}
      if unet_lr is not None:
        param_data['lr'] = unet_lr
      all_params.append(param_data)

    return all_params

  def prepare_grad_etc(self, text_encoder, unet):
    self.requires_grad_(True)

  def on_epoch_start(self, text_encoder, unet):
    self.train()

  def get_trainable_params(self):
    return self.parameters()

  def save_weights(self, file, dtype, metadata):
    if metadata is not None and len(metadata) == 0:
      metadata = None

    state_dict = self.state_dict()

    if dtype is not None:
      for key in list(state_dict.keys()):
        v = state_dict[key]
        v = v.detach().clone().to("cpu").to(dtype)
        state_dict[key] = v

    if os.path.splitext(file)[1] == '.safetensors':
      from safetensors.torch import save_file

      # Precalculate model hashes to save time on indexing
      if metadata is None:
        metadata = {}
      model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
      metadata["sshs_model_hash"] = model_hash
      metadata["sshs_legacy_hash"] = legacy_hash

      save_file(state_dict, file, metadata)
    else:
      torch.save(state_dict, file)

  @ staticmethod
  def set_regions(networks, image):
    image = image.astype(np.float32) / 255.0
    for i, network in enumerate(networks[:3]):
      # NOTE: consider averaging overwrapping area
      region = image[:, :, i]
      if region.max() == 0:
        continue
      region = torch.tensor(region)
      network.set_region(region)

  def set_region(self, region):
    for lora in self.unet_loras:
      lora.set_region(region)
