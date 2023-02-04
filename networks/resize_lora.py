# Convert LoRA to different rank approximation (should only be used to go to lower rank)
# This code is based off the extract_lora_from_models.py file which is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo and kohya

import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def load_state_dict(file_name, dtype):
  if os.path.splitext(file_name)[1] == '.safetensors':
    sd = load_file(file_name)
  else:
    sd = torch.load(file_name, map_location='cpu')
  for key in list(sd.keys()):
    if type(sd[key]) == torch.Tensor:
      sd[key] = sd[key].to(dtype)
  return sd


def save_to_file(file_name, model, state_dict, dtype):
  if dtype is not None:
    for key in list(state_dict.keys()):
      if type(state_dict[key]) == torch.Tensor:
        state_dict[key] = state_dict[key].to(dtype)

  if os.path.splitext(file_name)[1] == '.safetensors':
    save_file(model, file_name)
  else:
    torch.save(model, file_name)
    


def resize_lora_model(model, new_rank, merge_dtype, save_dtype):
    print("Loading Model...")
    lora_sd = load_state_dict(model, merge_dtype)

    network_alpha = None
    network_dim = None

    CLAMP_QUANTILE = 0.99

    # Extract loaded lora dim and alpha
    for key, value in lora_sd.items():
        if network_alpha is None and 'alpha' in key:
            network_alpha = value
        if network_dim is None and 'lora_down' in key and len(value.size()) == 2:
            network_dim = value.size()[0]
        if network_alpha is not None and network_dim is not None:
            break
        if network_alpha is None:
            network_alpha = network_dim

    scale = network_alpha/network_dim
    new_alpha = float(scale*new_rank)  # calculate new alpha from scale

    print(f"dimension: {network_dim}, alpha: {network_alpha}, new alpha: {new_alpha}")

    lora_down_weight = None
    lora_up_weight = None

    o_lora_sd = lora_sd.copy()
    block_down_name = None
    block_up_name = None

    print("resizing lora...")
    with torch.no_grad():
        for key, value in tqdm(lora_sd.items()):
            if 'lora_down' in key:
                block_down_name = key.split(".")[0]
                lora_down_weight = value
            if 'lora_up' in key:
                block_up_name = key.split(".")[0]
                lora_up_weight = value

            weights_loaded = (lora_down_weight is not None and lora_up_weight is not None)

            if (block_down_name == block_up_name) and weights_loaded:

                conv2d = (len(lora_down_weight.size()) == 4)
                
                if conv2d:
                    lora_down_weight = lora_down_weight.squeeze()
                    lora_up_weight = lora_up_weight.squeeze()

                if args.device:
                    org_device = lora_up_weight.device
                    lora_up_weight = lora_up_weight.to(args.device)
                    lora_down_weight = lora_down_weight.to(args.device)

                full_weight_matrix = torch.matmul(lora_up_weight, lora_down_weight)

                U, S, Vh = torch.linalg.svd(full_weight_matrix)

                U = U[:, :new_rank]
                S = S[:new_rank]
                U = U @ torch.diag(S)

                Vh = Vh[:new_rank, :]

                dist = torch.cat([U.flatten(), Vh.flatten()])
                hi_val = torch.quantile(dist, CLAMP_QUANTILE)
                low_val = -hi_val

                U = U.clamp(low_val, hi_val)
                Vh = Vh.clamp(low_val, hi_val)
            
                if conv2d:
                    U = U.unsqueeze(2).unsqueeze(3)
                    Vh = Vh.unsqueeze(2).unsqueeze(3)
                
                if args.device:
                   U = U.to(org_device)
                   Vh = Vh.to(org_device)

                o_lora_sd[block_down_name + "." + "lora_down.weight"] = Vh.to(save_dtype).contiguous()
                o_lora_sd[block_up_name + "." + "lora_up.weight"] =  U.to(save_dtype).contiguous()
                o_lora_sd[block_up_name + "." "alpha"] = torch.tensor(new_alpha).to(save_dtype)

                block_down_name = None
                block_up_name = None
                lora_down_weight = None
                lora_up_weight = None
                weights_loaded = False

    print("resizing complete")
    return o_lora_sd

def resize(args):

    def str_to_dtype(p):
        if p == 'float':
            return torch.float
        if p == 'fp16':
            return torch.float16
        if p == 'bf16':
            return torch.bfloat16
        return None

    merge_dtype = str_to_dtype('float') # matmul method above only seems to work in float32
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    state_dict =  resize_lora_model(args.model, args.new_rank, merge_dtype, save_dtype)

    print(f"saving model to: {args.save_to}")
    save_to_file(args.save_to, state_dict, state_dict, save_dtype)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument("--save_precision", type=str, default=None,
                      choices=[None, "float", "fp16", "bf16"], help="precision in saving, float if omitted / 保存時の精度、未指定時はfloat")
  parser.add_argument("--new_rank", type=int, default=4,
                      help="Specify rank of output LoRA / 出力するLoRAのrank (dim)")
  parser.add_argument("--save_to", type=str, default=None,
                      help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors")
  parser.add_argument("--model", type=str, default=None,
                      help="LoRA model to resize at to new rank: ckpt or safetensors file / 読み込むLoRAモデル、ckptまたはsafetensors")
  parser.add_argument("--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う")

  args = parser.parse_args()
  resize(args)
