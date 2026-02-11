# This script converts the diffusers of a Flux model to a safetensors file of a Flux.1 model.
# It is based on the implementation by 2kpr. Thanks to 2kpr!
# Major changes:
# - Iterates over three safetensors files to reduce memory usage, not loading all tensors at once.
# - Makes reverse map from diffusers map to avoid loading all tensors.
# - Removes dependency on .json file for weights mapping.
# - Adds support for custom memory efficient load and save functions.
# - Supports saving with different precision.
# - Supports .safetensors file as input.

# Copyright 2024 2kpr. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import os
from pathlib import Path
import safetensors
from safetensors.torch import safe_open
import torch
from tqdm import tqdm

from library import flux_utils
from library.utils import setup_logging, str_to_dtype, MemoryEfficientSafeOpen, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)


def convert(args):
    # if diffusers_path is folder, get safetensors file
    diffusers_path = Path(args.diffusers_path)
    if diffusers_path.is_dir():
        diffusers_path = Path.joinpath(diffusers_path, "transformer", "diffusion_pytorch_model-00001-of-00003.safetensors")

    flux_path = Path(args.save_to)
    if not os.path.exists(flux_path.parent):
        os.makedirs(flux_path.parent)

    if not diffusers_path.exists():
        logger.error(f"Error: Missing transformer safetensors file: {diffusers_path}")
        return

    mem_eff_flag = args.mem_eff_load_save
    save_dtype = str_to_dtype(args.save_precision) if args.save_precision is not None else None

    # make reverse map from diffusers map
    diffusers_to_bfl_map = flux_utils.make_diffusers_to_bfl_map()

    # iterate over three safetensors files to reduce memory usage
    flux_sd = {}
    for i in range(3):
        # replace 00001 with 0000i
        current_diffusers_path = Path(str(diffusers_path).replace("00001", f"0000{i+1}"))
        logger.info(f"Loading diffusers file: {current_diffusers_path}")

        open_func = MemoryEfficientSafeOpen if mem_eff_flag else (lambda x: safe_open(x, framework="pt"))
        with open_func(current_diffusers_path) as f:
            for diffusers_key in tqdm(f.keys()):
                if diffusers_key in diffusers_to_bfl_map:
                    tensor = f.get_tensor(diffusers_key).to("cpu")
                    if save_dtype is not None:
                        tensor = tensor.to(save_dtype)

                    index, bfl_key = diffusers_to_bfl_map[diffusers_key]
                    if bfl_key not in flux_sd:
                        flux_sd[bfl_key] = []
                    flux_sd[bfl_key].append((index, tensor))
                else:
                    logger.error(f"Error: Key not found in diffusers_to_bfl_map: {diffusers_key}")
                    return

    # concat tensors if multiple tensors are mapped to a single key, sort by index
    for key, values in flux_sd.items():
        if len(values) == 1:
            flux_sd[key] = values[0][1]
        else:
            flux_sd[key] = torch.cat([value[1] for value in sorted(values, key=lambda x: x[0])])

    # special case for final_layer.adaLN_modulation.1.weight and final_layer.adaLN_modulation.1.bias
    def swap_scale_shift(weight):
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    if "final_layer.adaLN_modulation.1.weight" in flux_sd:
        flux_sd["final_layer.adaLN_modulation.1.weight"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.weight"])
    if "final_layer.adaLN_modulation.1.bias" in flux_sd:
        flux_sd["final_layer.adaLN_modulation.1.bias"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.bias"])

    # save flux_sd to safetensors file
    logger.info(f"Saving Flux safetensors file: {flux_path}")
    if mem_eff_flag:
        mem_eff_save_file(flux_sd, flux_path)
    else:
        safetensors.torch.save_file(flux_sd, flux_path)

    logger.info("Conversion completed.")


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diffusers_path",
        default=None,
        type=str,
        required=True,
        help="Path to the original Flux diffusers folder or *-00001-of-00003.safetensors file."
        " / 元のFlux diffusersフォルダーまたは*-00001-of-00003.safetensorsファイルへのパス",
    )
    parser.add_argument(
        "--save_to",
        default=None,
        type=str,
        required=True,
        help="Output path for the Flux safetensors file. / Flux safetensorsファイルの出力先",
    )
    parser.add_argument(
        "--mem_eff_load_save",
        action="store_true",
        help="use custom memory efficient load and save functions for FLUX.1 model"
        " / カスタムのメモリ効率の良い読み込みと保存関数をFLUX.1モデルに使用する",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        help="precision in saving, default is same as loading precision"
        "float32, fp16, bf16, fp8 (same as fp8_e4m3fn), fp8_e4m3fn, fp8_e4m3fnuz, fp8_e5m2, fp8_e5m2fnuz"
        " / 保存時に精度を変更して保存する、デフォルトは読み込み時と同じ精度",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    convert(args)
