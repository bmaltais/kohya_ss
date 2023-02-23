# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from typing import List
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
    self.lora_dim = lora_dim

    if org_module.__class__.__name__ == 'Conv2d':
      in_dim = org_module.in_channels
      out_dim = org_module.out_channels
      self.lora_down = torch.nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
      self.lora_up = torch.nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
    else:
      in_dim = org_module.in_features
      out_dim = org_module.out_features
      self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
      self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)

    if type(alpha) == torch.Tensor:
      alpha = alpha.detach().float().numpy()                              # without casting, bf16 causes error
    alpha = lora_dim if alpha is None or alpha == 0 else alpha
    self.scale = alpha / self.lora_dim
    self.register_buffer('alpha', torch.tensor(alpha))                    # 定数として扱える

    # same as microsoft's
    torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
    torch.nn.init.zeros_(self.lora_up.weight)

    self.multiplier = multiplier
    self.org_module = org_module                  # remove in applying

  def apply_to(self):
    self.org_forward = self.org_module.forward
    self.org_module.forward = self.forward
    del self.org_module

  def forward(self, x):
    return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale


def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
  if network_dim is None:
    network_dim = 4                     # default
  network = LoRANetwork(text_encoder, unet, multiplier=multiplier, lora_dim=network_dim, alpha=network_alpha)
  return network


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, **kwargs):
  if os.path.splitext(file)[1] == '.safetensors':
    from safetensors.torch import load_file, safe_open
    weights_sd = load_file(file)
  else:
    weights_sd = torch.load(file, map_location='cpu')

  # get dim (rank)
  network_alpha = None
  network_dim = None
  for key, value in weights_sd.items():
    if network_alpha is None and 'alpha' in key:
      network_alpha = value
    if network_dim is None and 'lora_down' in key and len(value.size()) == 2:
      network_dim = value.size()[0]

  if network_alpha is None:
    network_alpha = network_dim

  network = LoRANetwork(text_encoder, unet, multiplier=multiplier, lora_dim=network_dim, alpha=network_alpha)
  network.weights_sd = weights_sd
  return network


class LoRANetwork(torch.nn.Module):
  UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
  TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
  LORA_PREFIX_UNET = 'lora_unet'
  LORA_PREFIX_TEXT_ENCODER = 'lora_te'

  def __init__(self, text_encoder, unet, multiplier=1.0, lora_dim=4, alpha=1) -> None:
    super().__init__()
    self.multiplier = multiplier
    self.lora_dim = lora_dim
    self.alpha = alpha

    # create module instances
    def create_modules(prefix, root_module: torch.nn.Module, target_replace_modules) -> List[LoRAModule]:
      loras = []
      for name, module in root_module.named_modules():
        if module.__class__.__name__ in target_replace_modules:
          for child_name, child_module in module.named_modules():
            if child_module.__class__.__name__ == "Linear" or (child_module.__class__.__name__ == "Conv2d" and child_module.kernel_size == (1, 1)):
              lora_name = prefix + '.' + name + '.' + child_name
              lora_name = lora_name.replace('.', '_')
              lora = LoRAModule(lora_name, child_module, self.multiplier, self.lora_dim, self.alpha)
              loras.append(lora)
      return loras

    self.text_encoder_loras = create_modules(LoRANetwork.LORA_PREFIX_TEXT_ENCODER,
                                             text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
    print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

    self.unet_loras = create_modules(LoRANetwork.LORA_PREFIX_UNET, unet, LoRANetwork.UNET_TARGET_REPLACE_MODULE)
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
