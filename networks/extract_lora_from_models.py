# extract approximating LoRA by svd from two SD models
# The code is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo!

import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import library.model_util as model_util
import lora


CLAMP_QUANTILE = 0.99
MIN_DIFF = 1e-6


def save_to_file(file_name, model, state_dict, dtype):
  if dtype is not None:
    for key in list(state_dict.keys()):
      if type(state_dict[key]) == torch.Tensor:
        state_dict[key] = state_dict[key].to(dtype)

  if os.path.splitext(file_name)[1] == '.safetensors':
    save_file(model, file_name)
  else:
    torch.save(model, file_name)


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

  print(f"loading SD model : {args.model_org}")
  text_encoder_o, _, unet_o = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.model_org)
  print(f"loading SD model : {args.model_tuned}")
  text_encoder_t, _, unet_t = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.model_tuned)

  # create LoRA network to extract weights: Use dim (rank) as alpha
  lora_network_o = lora.create_network(1.0, args.dim, args.dim, None, text_encoder_o, unet_o)
  lora_network_t = lora.create_network(1.0, args.dim, args.dim, None, text_encoder_t, unet_t)
  assert len(lora_network_o.text_encoder_loras) == len(
      lora_network_t.text_encoder_loras), f"model version is different (SD1.x vs SD2.x) / それぞれのモデルのバージョンが違います（SD1.xベースとSD2.xベース） "

  # get diffs
  diffs = {}
  text_encoder_different = False
  for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.text_encoder_loras, lora_network_t.text_encoder_loras)):
    lora_name = lora_o.lora_name
    module_o = lora_o.org_module
    module_t = lora_t.org_module
    diff = module_t.weight - module_o.weight

    # Text Encoder might be same
    if torch.max(torch.abs(diff)) > MIN_DIFF:
      text_encoder_different = True

    diff = diff.float()
    diffs[lora_name] = diff

  if not text_encoder_different:
    print("Text encoder is same. Extract U-Net only.")
    lora_network_o.text_encoder_loras = []
    diffs = {}

  for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.unet_loras, lora_network_t.unet_loras)):
    lora_name = lora_o.lora_name
    module_o = lora_o.org_module
    module_t = lora_t.org_module
    diff = module_t.weight - module_o.weight
    diff = diff.float()

    if args.device:
      diff = diff.to(args.device)

    diffs[lora_name] = diff

  # make LoRA with svd
  print("calculating by svd")
  rank = args.dim
  lora_weights = {}
  with torch.no_grad():
    for lora_name, mat in tqdm(list(diffs.items())):
      conv2d = (len(mat.size()) == 4)
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

      lora_weights[lora_name] = (U, Vh)

  # make state dict for LoRA
  lora_network_o.apply_to(text_encoder_o, unet_o, text_encoder_different, True)   # to make state dict
  lora_sd = lora_network_o.state_dict()
  print(f"LoRA has {len(lora_sd)} weights.")

  for key in list(lora_sd.keys()):
    if "alpha" in key:
      continue

    lora_name = key.split('.')[0]
    i = 0 if "lora_up" in key else 1

    weights = lora_weights[lora_name][i]
    # print(key, i, weights.size(), lora_sd[key].size())
    if len(lora_sd[key].size()) == 4:
      weights = weights.unsqueeze(2).unsqueeze(3)

    assert weights.size() == lora_sd[key].size(), f"size unmatch: {key}"
    lora_sd[key] = weights

  # load state dict to LoRA and save it
  info = lora_network_o.load_state_dict(lora_sd)
  print(f"Loading extracted LoRA weights: {info}")

  dir_name = os.path.dirname(args.save_to)
  if dir_name and not os.path.exists(dir_name):
    os.makedirs(dir_name, exist_ok=True)

  # minimum metadata
  metadata = {"ss_network_dim": str(args.dim), "ss_network_alpha": str(args.dim)}

  lora_network_o.save_weights(args.save_to, save_dtype, metadata)
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
