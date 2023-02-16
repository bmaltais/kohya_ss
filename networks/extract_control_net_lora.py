# extract approximating LoRA by svd from SD 1.5 vs ControlNet
# https://github.com/lllyasviel/ControlNet/blob/main/tool_transfer_control.py
#
# The code is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo!

import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from diffusers import UNet2DConditionModel

import library.model_util as model_util
import control_net_lora


CLAMP_QUANTILE = 0.99
MIN_DIFF = 1e-6


def save_to_file(file_name, state_dict, dtype):
  if dtype is not None:
    for key in list(state_dict.keys()):
      if type(state_dict[key]) == torch.Tensor:
        state_dict[key] = state_dict[key].to(dtype)

  if os.path.splitext(file_name)[1] == '.safetensors':
    save_file(state_dict, file_name)
  else:
    torch.save(state_dict, file_name)


def svd(args):
  def str_to_dtype(p):
    if p == 'float':
      return torch.float
    if p == 'fp16':
      return torch.float16
    if p == 'bf16':
      return torch.bfloat16
    return None

  save_dtype = str_to_dtype(args.save_precision)

  # Diffusersのキーに変換するため、original sdとcontrol sdからU-Netに重みを読み込む ###############

  # original sdをDiffusersのU-Netに読み込む
  print(f"loading original SD model : {args.model_org}")
  _, _, org_unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.model_org)

  org_sd = torch.load(args.model_org, map_location='cpu')
  if 'state_dict' in org_sd:
    org_sd = org_sd['state_dict']

  # control sdからキー変換しつつU-Netに対応する部分のみ取り出し、DiffusersのuU-Netに読み込む
  print(f"loading control SD model : {args.model_tuned}")

  ctrl_sd = torch.load(args.model_tuned, map_location='cpu')
  ctrl_unet_sd = org_sd                      # あらかじめloadしておくことでcontrol sdにない部分はoriginal sdと同じにする
  for key in list(ctrl_sd.keys()):
    if key.startswith("control_"):
      unet_key = "model.diffusion_" + key[len("control_"):]
      if unet_key not in ctrl_unet_sd:               # zero conv
        continue
      ctrl_unet_sd[unet_key] = ctrl_sd[key]

  unet_config = model_util.create_unet_diffusers_config(args.v2)
  ctrl_unet_sd_du = model_util.convert_ldm_unet_checkpoint(args.v2, ctrl_unet_sd, unet_config)

  # load weights to U-Net
  ctrl_unet = UNet2DConditionModel(**unet_config)
  info = ctrl_unet.load_state_dict(ctrl_unet_sd_du)
  print("loading control u-net:", info)

  # LoRAに対応する部分のU-Netの重みを読み込む #################################

  diffs = {}
  for (org_name, org_module), (ctrl_name, ctrl_module) in zip(org_unet.named_modules(), ctrl_unet.named_modules()):
    if org_module.__class__.__name__ != "Linear" and org_module.__class__.__name__ != "Conv2d":
      continue
    assert org_name == ctrl_name

    lora_name = control_net_lora.ControlLoRANetwork.LORA_PREFIX_UNET + '.' + org_name  # + '.' + child_name
    lora_name = lora_name.replace('.', '_')

    diff = ctrl_module.weight - org_module.weight
    diff = diff.float()

    if torch.max(torch.abs(diff)) < 1e-5:
      # print(f"weights are same: {lora_name}")
      continue
    print(lora_name)

    if args.device:
      diff = diff.to(args.device)

    diffs[lora_name] = diff

  # make LoRA with svd
  print("calculating by svd")
  rank = args.dim
  ctrl_lora_sd = {}
  with torch.no_grad():
    for lora_name, mat in tqdm(list(diffs.items())):
      conv2d = (len(mat.size()) == 4)
      kernel_size = None if not conv2d else mat.size()[2:]

      if not conv2d or kernel_size == (1, 1):
        if conv2d:
          mat = mat.squeeze()

        U, S, Vh = torch.linalg.svd(mat)

        U = U[:, :rank]
        S = S[:rank]
        U = U @ torch.diag(S)

        Vh = Vh[:rank, :]

        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, CLAMP_QUANTILE)
        low_val = -hi_val

        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)

        if conv2d:
          U = U.unsqueeze(2).unsqueeze(3)
          Vh = Vh.unsqueeze(2).unsqueeze(3)
      else:
        # conv2d kernel != (1,1)
        in_channels = mat.size()[1]
        current_rank = min(rank, in_channels, mat.size()[0])
        if current_rank != rank:
          print(f"channels of conv2d is too small. rank is changed to {current_rank} @ {lora_name}: {mat.size()}")

        mat = mat.flatten(start_dim=1)

        U, S, Vh = torch.linalg.svd(mat)

        U = U[:, :current_rank]
        S = S[:current_rank]
        U = U @ torch.diag(S)

        Vh = Vh[:current_rank, :]

        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, CLAMP_QUANTILE)
        low_val = -hi_val

        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)

        # U is (out_channels, rank) with 1x1 conv. So,
        U = U.reshape(U.shape[0], U.shape[1], 1, 1)
        # V is (rank, in_channels * kernel_size1 * kernel_size2)
        # now reshape:
        Vh = Vh.reshape(Vh.shape[0], in_channels, *kernel_size)

      ctrl_lora_sd[lora_name + ".lora_up.weight"] = U
      ctrl_lora_sd[lora_name + ".lora_down.weight"] = Vh
      ctrl_lora_sd[lora_name + ".alpha"] = torch.tensor(current_rank)

  # create LoRA from sd
  lora_network = control_net_lora.ControlLoRANetwork(org_unet, ctrl_lora_sd, 1.0)
  lora_network.apply_to()

  for key, value in ctrl_sd.items():
    if 'zero_convs' in key or 'input_hint_block' in key or 'middle_block_out' in key:
      ctrl_lora_sd[key] = value

  # verify state dict by loading it
  info = lora_network.load_state_dict(ctrl_lora_sd)
  print(f"loading control lora sd: {info}")

  dir_name = os.path.dirname(args.save_to)
  if dir_name and not os.path.exists(dir_name):
    os.makedirs(dir_name, exist_ok=True)

  # # minimum metadata
  # metadata = {"ss_network_dim": str(args.dim), "ss_network_alpha": str(args.dim)}

  # lora_network.save_weights(args.save_to, save_dtype, metadata)
  save_to_file(args.save_to, ctrl_lora_sd, save_dtype)
  print(f"LoRA weights are saved to: {args.save_to}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--v2", action='store_true',
                      help='load Stable Diffusion v2.x model / Stable Diffusion 2.xのモデルを読み込む')
  parser.add_argument("--save_precision", type=str, default=None,
                      choices=[None, "float", "fp16", "bf16"], help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はfloat")
  parser.add_argument("--model_org", type=str, default=None,
                      help="Stable Diffusion original model: ckpt or safetensors file / 元モデル、ckptまたはsafetensors")
  parser.add_argument("--model_tuned", type=str, default=None,
                      help="Stable Diffusion tuned model, LoRA is difference of `original to tuned`: ckpt or safetensors file / 派生モデル（生成されるLoRAは元→派生の差分になります）、ckptまたはsafetensors")
  parser.add_argument("--save_to", type=str, default=None,
                      help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors")
  parser.add_argument("--dim", type=int, default=4, help="dimension (rank) of LoRA (default 4) / LoRAの次元数（rank）（デフォルト4）")
  parser.add_argument("--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う")

  args = parser.parse_args()
  svd(args)
