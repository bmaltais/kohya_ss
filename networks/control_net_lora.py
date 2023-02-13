# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from typing import List
import torch
from diffusers import UNet2DConditionModel

from library import train_util


class ControlLoRAModule(torch.nn.Module):
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

      self.lora_dim = min(self.lora_dim, in_dim, out_dim)
      if self.lora_dim != lora_dim:
        print(f"{lora_name} dim (rank) is changed: {self.lora_dim}")

      kernel_size = org_module.kernel_size
      stride = org_module.stride
      padding = org_module.padding
      self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
      self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
    else:
      in_dim = org_module.in_features
      out_dim = org_module.out_features
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

  def apply_to(self):
    self.org_forward = self.org_module.forward
    self.org_module.forward = self.forward
    del self.org_module

  def set_as_control_path(self, control_path):
    self.is_control_path = control_path

  def forward(self, x):
    if not self.is_control_path:
      return self.org_forward(x)
    return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale


class ControlLoRANetwork(torch.nn.Module):
  # UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
  # TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
  LORA_PREFIX_UNET = 'lora_unet'
  LORA_PREFIX_TEXT_ENCODER = 'lora_te'

  def __init__(self, unet, weights_sd, multiplier=1.0, lora_dim=4, alpha=1) -> None:
    super().__init__()
    self.multiplier = multiplier
    self.lora_dim = lora_dim
    self.alpha = alpha

    # create module instances
    def create_modules(prefix, root_module: torch.nn.Module) -> List[ControlLoRAModule]:  # , target_replace_modules
      loras = []
      for name, module in root_module.named_modules():
        # # if module.__class__.__name__ in target_replace_modules:
        # for child_name, child_module in module.named_modules():
        if module.__class__.__name__ == "Linear" or module.__class__.__name__ == "Conv2d":  # and module.kernel_size == (1, 1)):
          lora_name = prefix + '.' + name  # + '.' + child_name
          lora_name = lora_name.replace('.', '_')

          if weights_sd is None:
            dim, alpha = self.lora_dim, self.alpha
          else:
            down_weight = weights_sd.get(lora_name + ".lora_down.weight", None)
            if down_weight is None:
              continue
            dim = down_weight.size()[0]
            alpha = weights_sd.get(lora_name + ".alpha", dim)

          lora = ControlLoRAModule(lora_name, module, self.multiplier, dim, alpha)
          loras.append(lora)
      return loras

    self.unet_loras = create_modules(ControlLoRANetwork.LORA_PREFIX_UNET, unet)  # , LoRANetwork.UNET_TARGET_REPLACE_MODULE)
    print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

    # make control model
    self.control_model = torch.nn.Module()

    dims = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
    zero_convs = torch.nn.ModuleList()
    for i, dim in enumerate(dims):
      sub_list = torch.nn.ModuleList([torch.nn.Conv2d(dim, dim, 1)])
      zero_convs.append(sub_list)
    self.control_model.add_module("zero_convs", zero_convs)

    middle_block_out = torch.nn.Conv2d(1280, 1280, 1)
    self.control_model.add_module("middle_block_out", torch.nn.ModuleList([middle_block_out]))

    dims = [16, 16, 32, 32, 96, 96, 256, 320]
    strides = [1, 1, 2, 1, 2, 1, 2, 1]
    prev_dim = 3
    input_hint_block = torch.nn.Sequential()
    for i, (dim, stride) in enumerate(zip(dims, strides)):
      input_hint_block.append(torch.nn.Conv2d(prev_dim, dim, 3, stride, 1))
      if i < len(dims) - 1:
        input_hint_block.append(torch.nn.SiLU())
      prev_dim = dim
    self.control_model.add_module("input_hint_block", input_hint_block)


  # def load_weights(self, file):
  #   if os.path.splitext(file)[1] == '.safetensors':
  #     from safetensors.torch import load_file, safe_open
  #     self.weights_sd = load_file(file)
  #   else:
  #     self.weights_sd = torch.load(file, map_location='cpu')

  def apply_to(self):
    for lora in self.unet_loras:
      lora.apply_to()
      self.add_module(lora.lora_name, lora)

  def call_unet(self, unet, hint, sample, timestep, encoder_hidden_states):
    # control path
    hint = hint.to(sample.dtype).to(sample.device)
    guided_hint = self.control_model.input_hint_block(hint)

    for lora_module in self.unet_loras:
      lora_module.set_as_control_path(True)

    outs = self.unet_forward(unet, guided_hint, None, sample, timestep, encoder_hidden_states)

    # U-Net
    for lora_module in self.unet_loras:
      lora_module.set_as_control_path(False)

    sample = self.unet_forward(unet, None, outs, sample, timestep, encoder_hidden_states)

    return sample

  def unet_forward(self, unet: UNet2DConditionModel, guided_hint, ctrl_outs, sample, timestep, encoder_hidden_states):
    # copy from UNet2DConditionModel
    default_overall_up_factor = 2**unet.num_upsamplers

    forward_upsample_size = False
    upsample_size = None

    if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
      print("Forward upsample size to force interpolation output size.")
      forward_upsample_size = True

    # 0. center input if necessary
    if unet.config.center_input_sample:
      sample = 2 * sample - 1.0

    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
      # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
      # This would be a good case for the `match` statement (Python 3.10+)
      is_mps = sample.device.type == "mps"
      if isinstance(timestep, float):
        dtype = torch.float32 if is_mps else torch.float64
      else:
        dtype = torch.int32 if is_mps else torch.int64
      timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
      timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = unet.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=unet.dtype)
    emb = unet.time_embedding(t_emb)

    if ctrl_outs is None:
      outs = []                     # control path

    # 2. pre-process
    sample = unet.conv_in(sample)
    if guided_hint is not None:
      sample += guided_hint
    if ctrl_outs is None:
      outs.append(self.control_model.zero_convs[0][0](sample)) # , emb, encoder_hidden_states))

    # 3. down
    zc_idx = 1
    down_block_res_samples = (sample,)
    for downsample_block in unet.down_blocks:
      if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
        sample, res_samples = downsample_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )
      else:
        sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
      if ctrl_outs is None:
        for rs in res_samples:
          print("zc", zc_idx, rs.size())
          outs.append(self.control_model.zero_convs[zc_idx][0](rs)) # , emb, encoder_hidden_states))
          zc_idx += 1

      down_block_res_samples += res_samples

    # 4. mid
    sample = unet.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
    if ctrl_outs is None:
      outs.append(self.control_model.middle_block_out[0](sample)) 
      return outs
    if ctrl_outs is not None:
      sample += ctrl_outs.pop()

    # 5. up
    for i, upsample_block in enumerate(unet.up_blocks):
      is_final_block = i == len(unet.up_blocks) - 1

      res_samples = down_block_res_samples[-len(upsample_block.resnets):]
      down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

      if ctrl_outs is not None and len(ctrl_outs) > 0:
        res_samples = list(res_samples)
        apply_ctrl_outs = ctrl_outs[-len(res_samples):]
        ctrl_outs = ctrl_outs[:-len(res_samples)]
        for j in range(len(res_samples)):
          print(i, j)
          res_samples[j] = res_samples[j] + apply_ctrl_outs[j]
        res_samples = tuple(res_samples)

      # if we have not reached the final block and need to forward the
      # upsample size, we do it here
      if not is_final_block and forward_upsample_size:
        upsample_size = down_block_res_samples[-1].shape[2:]

      if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
        sample = upsample_block(
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=res_samples,
            encoder_hidden_states=encoder_hidden_states,
            upsample_size=upsample_size,
        )
      else:
        sample = upsample_block(
            hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
        )
    # 6. post-process
    sample = unet.conv_norm_out(sample)
    sample = unet.conv_act(sample)
    sample = unet.conv_out(sample)

    return (sample,)

  """
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.config.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
  """

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
