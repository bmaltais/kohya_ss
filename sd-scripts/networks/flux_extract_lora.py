# extract approximating LoRA by svd from two FLUX models
# The code is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo!

import argparse
import json
import os
import time
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tqdm import tqdm
from library import flux_utils, sai_model_spec, model_util, sdxl_model_util
import lora
from library.utils import MemoryEfficientSafeOpen
from library.utils import setup_logging
from networks import lora_flux

setup_logging()
import logging

logger = logging.getLogger(__name__)

# CLAMP_QUANTILE = 0.99
# MIN_DIFF = 1e-1


def save_to_file(file_name, state_dict, metadata, dtype):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    save_file(state_dict, file_name, metadata=metadata)


def svd(
    model_org=None,
    model_tuned=None,
    save_to=None,
    dim=4,
    device=None,
    save_precision=None,
    clamp_quantile=0.99,
    min_diff=0.01,
    no_metadata=False,
    mem_eff_safe_open=False,
):
    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    calc_dtype = torch.float
    save_dtype = str_to_dtype(save_precision)
    store_device = "cpu"

    # open models
    lora_weights = {}
    if not mem_eff_safe_open:
        # use original safetensors.safe_open
        open_fn = lambda fn: safe_open(fn, framework="pt")
    else:
        logger.info("Using memory efficient safe_open")
        open_fn = lambda fn: MemoryEfficientSafeOpen(fn)

    with open_fn(model_org) as f_org:
        # filter keys
        keys = []
        for key in f_org.keys():
            if not ("single_block" in key or "double_block" in key):
                continue
            if ".bias" in key:
                continue
            if "norm" in key:
                continue
            keys.append(key)

        with open_fn(model_tuned) as f_tuned:
            for key in tqdm(keys):
                # get tensors and calculate difference
                value_o = f_org.get_tensor(key)
                value_t = f_tuned.get_tensor(key)
                mat = value_t.to(calc_dtype) - value_o.to(calc_dtype)
                del value_o, value_t

                # extract LoRA weights
                if device:
                    mat = mat.to(device)
                out_dim, in_dim = mat.size()[0:2]
                rank = min(dim, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

                mat = mat.squeeze()

                U, S, Vh = torch.linalg.svd(mat)

                U = U[:, :rank]
                S = S[:rank]
                U = U @ torch.diag(S)

                Vh = Vh[:rank, :]

                dist = torch.cat([U.flatten(), Vh.flatten()])
                hi_val = torch.quantile(dist, clamp_quantile)
                low_val = -hi_val

                U = U.clamp(low_val, hi_val)
                Vh = Vh.clamp(low_val, hi_val)

                U = U.to(store_device, dtype=save_dtype).contiguous()
                Vh = Vh.to(store_device, dtype=save_dtype).contiguous()

                # print(f"key: {key}, U: {U.size()}, Vh: {Vh.size()}")
                lora_weights[key] = (U, Vh)
                del mat, U, S, Vh

    # make state dict for LoRA
    lora_sd = {}
    for key, (up_weight, down_weight) in lora_weights.items():
        lora_name = key.replace(".weight", "").replace(".", "_")
        lora_name = lora_flux.LoRANetwork.LORA_PREFIX_FLUX + "_" + lora_name
        lora_sd[lora_name + ".lora_up.weight"] = up_weight
        lora_sd[lora_name + ".lora_down.weight"] = down_weight
        lora_sd[lora_name + ".alpha"] = torch.tensor(down_weight.size()[0])  # same as rank

    # minimum metadata
    net_kwargs = {}
    metadata = {
        "ss_v2": str(False),
        "ss_base_model_version": flux_utils.MODEL_VERSION_FLUX_V1,
        "ss_network_module": "networks.lora_flux",
        "ss_network_dim": str(dim),
        "ss_network_alpha": str(float(dim)),
        "ss_network_args": json.dumps(net_kwargs),
    }

    if not no_metadata:
        title = os.path.splitext(os.path.basename(save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(lora_sd, False, False, False, True, False, time.time(), title, flux="dev")
        metadata.update(sai_metadata)

    save_to_file(save_to, lora_sd, metadata, save_dtype)

    logger.info(f"LoRA weights saved to {save_to}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はfloat",
    )
    parser.add_argument(
        "--model_org",
        type=str,
        default=None,
        required=True,
        help="Original model: safetensors file / 元モデル、safetensors",
    )
    parser.add_argument(
        "--model_tuned",
        type=str,
        default=None,
        required=True,
        help="Tuned model, LoRA is difference of `original to tuned`: safetensors file / 派生モデル（生成されるLoRAは元→派生の差分になります）、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--mem_eff_safe_open",
        action="store_true",
        help="use memory efficient safe_open. This is an experimental feature, use only when memory is not enough."
        " / メモリ効率の良いsafe_openを使用する。実装は実験的なものなので、メモリが足りない場合のみ使用してください。",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        required=True,
        help="destination file name: safetensors file / 保存先のファイル名、safetensors",
    )
    parser.add_argument(
        "--dim", type=int, default=4, help="dimension (rank) of LoRA (default 4) / LoRAの次元数（rank）（デフォルト4）"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う"
    )
    parser.add_argument(
        "--clamp_quantile",
        type=float,
        default=0.99,
        help="Quantile clamping value, float, (0-1). Default = 0.99 / 値をクランプするための分位点、float、(0-1)。デフォルトは0.99",
    )
    # parser.add_argument(
    #     "--min_diff",
    #     type=float,
    #     default=0.01,
    #     help="Minimum difference between finetuned model and base to consider them different enough to extract, float, (0-1). Default = 0.01 /"
    #     + "LoRAを抽出するために元モデルと派生モデルの差分の最小値、float、(0-1)。デフォルトは0.01",
    # )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのメタデータを保存しない（LoRAの最低限のss_metadataは保存される）",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    svd(**vars(args))
