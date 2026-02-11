import argparse
import math
import os
import time
from typing import Any, Dict, Union

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from library.utils import setup_logging, str_to_dtype, MemoryEfficientSafeOpen, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)

import lora_flux as lora_flux
from library import sai_model_spec, train_util


def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, state_dict: Dict[str, Union[Any, torch.Tensor]], dtype, metadata, mem_eff_save=False):
    if dtype is not None:
        logger.info(f"converting to {dtype}...")
        for key in tqdm(list(state_dict.keys())):
            if type(state_dict[key]) == torch.Tensor and state_dict[key].dtype.is_floating_point:
                state_dict[key] = state_dict[key].to(dtype)

    logger.info(f"saving to: {file_name}")
    if mem_eff_save:
        mem_eff_save_file(state_dict, file_name, metadata=metadata)
    else:
        save_file(state_dict, file_name, metadata=metadata)


def merge_to_flux_model(
    loading_device,
    working_device,
    flux_path: str,
    clip_l_path: str,
    t5xxl_path: str,
    models,
    ratios,
    merge_dtype,
    save_dtype,
    mem_eff_load_save=False,
):
    # create module map without loading state_dict
    lora_name_to_module_key = {}
    if flux_path is not None:
        logger.info(f"loading keys from FLUX.1 model: {flux_path}")
        with safe_open(flux_path, framework="pt", device=loading_device) as flux_file:
            keys = list(flux_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_flux.LoRANetwork.LORA_PREFIX_FLUX + "_" + module_name.replace(".", "_")
                    lora_name_to_module_key[lora_name] = key

    lora_name_to_clip_l_key = {}
    if clip_l_path is not None:
        logger.info(f"loading keys from clip_l model: {clip_l_path}")
        with safe_open(clip_l_path, framework="pt", device=loading_device) as clip_l_file:
            keys = list(clip_l_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_flux.LoRANetwork.LORA_PREFIX_TEXT_ENCODER_CLIP + "_" + module_name.replace(".", "_")
                    lora_name_to_clip_l_key[lora_name] = key

    lora_name_to_t5xxl_key = {}
    if t5xxl_path is not None:
        logger.info(f"loading keys from t5xxl model: {t5xxl_path}")
        with safe_open(t5xxl_path, framework="pt", device=loading_device) as t5xxl_file:
            keys = list(t5xxl_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_flux.LoRANetwork.LORA_PREFIX_TEXT_ENCODER_T5 + "_" + module_name.replace(".", "_")
                    lora_name_to_t5xxl_key[lora_name] = key

    flux_state_dict = {}
    clip_l_state_dict = {}
    t5xxl_state_dict = {}
    if mem_eff_load_save:
        if flux_path is not None:
            with MemoryEfficientSafeOpen(flux_path) as flux_file:
                for key in tqdm(flux_file.keys()):
                    flux_state_dict[key] = flux_file.get_tensor(key).to(loading_device)  # dtype is not changed

        if clip_l_path is not None:
            with MemoryEfficientSafeOpen(clip_l_path) as clip_l_file:
                for key in tqdm(clip_l_file.keys()):
                    clip_l_state_dict[key] = clip_l_file.get_tensor(key).to(loading_device)

        if t5xxl_path is not None:
            with MemoryEfficientSafeOpen(t5xxl_path) as t5xxl_file:
                for key in tqdm(t5xxl_file.keys()):
                    t5xxl_state_dict[key] = t5xxl_file.get_tensor(key).to(loading_device)
    else:
        if flux_path is not None:
            flux_state_dict = load_file(flux_path, device=loading_device)
        if clip_l_path is not None:
            clip_l_state_dict = load_file(clip_l_path, device=loading_device)
        if t5xxl_path is not None:
            t5xxl_state_dict = load_file(t5xxl_path, device=loading_device)

    for model, ratio in zip(models, ratios):
        logger.info(f"loading: {model}")
        lora_sd, _ = load_state_dict(model, merge_dtype)  # loading on CPU

        logger.info(f"merging...")
        for key in tqdm(list(lora_sd.keys())):
            if "lora_down" in key:
                lora_name = key[: key.rfind(".lora_down")]
                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[: key.index("lora_down")] + "alpha"

                if lora_name in lora_name_to_module_key:
                    module_weight_key = lora_name_to_module_key[lora_name]
                    state_dict = flux_state_dict
                elif lora_name in lora_name_to_clip_l_key:
                    module_weight_key = lora_name_to_clip_l_key[lora_name]
                    state_dict = clip_l_state_dict
                elif lora_name in lora_name_to_t5xxl_key:
                    module_weight_key = lora_name_to_t5xxl_key[lora_name]
                    state_dict = t5xxl_state_dict
                else:
                    logger.warning(
                        f"no module found for LoRA weight: {key}. Skipping..."
                        f"LoRAの重みに対応するモジュールが見つかりませんでした。スキップします。"
                    )
                    continue

                down_weight = lora_sd.pop(key)
                up_weight = lora_sd.pop(up_key)

                dim = down_weight.size()[0]
                alpha = lora_sd.pop(alpha_key, dim)
                scale = alpha / dim

                # W <- W + U * D
                weight = state_dict[module_weight_key]

                weight = weight.to(working_device, merge_dtype)
                up_weight = up_weight.to(working_device, merge_dtype)
                down_weight = down_weight.to(working_device, merge_dtype)

                # logger.info(module_name, down_weight.size(), up_weight.size())
                if len(weight.size()) == 2:
                    # linear
                    weight = weight + ratio * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                    weight = weight + ratio * conved * scale

                state_dict[module_weight_key] = weight.to(loading_device, save_dtype)
                del up_weight
                del down_weight
                del weight

        if len(lora_sd) > 0:
            logger.warning(f"Unused keys in LoRA model: {list(lora_sd.keys())}")

    return flux_state_dict, clip_l_state_dict, t5xxl_state_dict


def merge_to_flux_model_diffusers(
    loading_device, working_device, flux_model, models, ratios, merge_dtype, save_dtype, mem_eff_load_save=False
):
    logger.info(f"loading keys from FLUX.1 model: {flux_model}")
    if mem_eff_load_save:
        flux_state_dict = {}
        with MemoryEfficientSafeOpen(flux_model) as flux_file:
            for key in tqdm(flux_file.keys()):
                flux_state_dict[key] = flux_file.get_tensor(key).to(loading_device)  # dtype is not changed
    else:
        flux_state_dict = load_file(flux_model, device=loading_device)

    def create_key_map(n_double_layers, n_single_layers):
        key_map = {}
        for index in range(n_double_layers):
            prefix_from = f"transformer_blocks.{index}"
            prefix_to = f"double_blocks.{index}"

            for end in ("weight", "bias"):
                k = f"{prefix_from}.attn."
                qkv_img = f"{prefix_to}.img_attn.qkv.{end}"
                qkv_txt = f"{prefix_to}.txt_attn.qkv.{end}"

                key_map[f"{k}to_q.{end}"] = qkv_img
                key_map[f"{k}to_k.{end}"] = qkv_img
                key_map[f"{k}to_v.{end}"] = qkv_img
                key_map[f"{k}add_q_proj.{end}"] = qkv_txt
                key_map[f"{k}add_k_proj.{end}"] = qkv_txt
                key_map[f"{k}add_v_proj.{end}"] = qkv_txt

            block_map = {
                "attn.to_out.0.weight": "img_attn.proj.weight",
                "attn.to_out.0.bias": "img_attn.proj.bias",
                "norm1.linear.weight": "img_mod.lin.weight",
                "norm1.linear.bias": "img_mod.lin.bias",
                "norm1_context.linear.weight": "txt_mod.lin.weight",
                "norm1_context.linear.bias": "txt_mod.lin.bias",
                "attn.to_add_out.weight": "txt_attn.proj.weight",
                "attn.to_add_out.bias": "txt_attn.proj.bias",
                "ff.net.0.proj.weight": "img_mlp.0.weight",
                "ff.net.0.proj.bias": "img_mlp.0.bias",
                "ff.net.2.weight": "img_mlp.2.weight",
                "ff.net.2.bias": "img_mlp.2.bias",
                "ff_context.net.0.proj.weight": "txt_mlp.0.weight",
                "ff_context.net.0.proj.bias": "txt_mlp.0.bias",
                "ff_context.net.2.weight": "txt_mlp.2.weight",
                "ff_context.net.2.bias": "txt_mlp.2.bias",
                "attn.norm_q.weight": "img_attn.norm.query_norm.scale",
                "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
                "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
                "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale",
            }

            for k, v in block_map.items():
                key_map[f"{prefix_from}.{k}"] = f"{prefix_to}.{v}"

        for index in range(n_single_layers):
            prefix_from = f"single_transformer_blocks.{index}"
            prefix_to = f"single_blocks.{index}"

            for end in ("weight", "bias"):
                k = f"{prefix_from}.attn."
                qkv = f"{prefix_to}.linear1.{end}"
                key_map[f"{k}to_q.{end}"] = qkv
                key_map[f"{k}to_k.{end}"] = qkv
                key_map[f"{k}to_v.{end}"] = qkv
                key_map[f"{prefix_from}.proj_mlp.{end}"] = qkv

            block_map = {
                "norm.linear.weight": "modulation.lin.weight",
                "norm.linear.bias": "modulation.lin.bias",
                "proj_out.weight": "linear2.weight",
                "proj_out.bias": "linear2.bias",
                "attn.norm_q.weight": "norm.query_norm.scale",
                "attn.norm_k.weight": "norm.key_norm.scale",
            }

            for k, v in block_map.items():
                key_map[f"{prefix_from}.{k}"] = f"{prefix_to}.{v}"

        # add as-is keys
        values = list([(v if isinstance(v, str) else v[0]) for v in set(key_map.values())])
        values.sort()
        key_map.update({v: v for v in values})

        return key_map

    key_map = create_key_map(18, 38)  # 18 double layers, 38 single layers

    def find_matching_key(flux_dict, lora_key):
        lora_key = lora_key.replace("diffusion_model.", "")
        lora_key = lora_key.replace("transformer.", "")
        lora_key = lora_key.replace("lora_A", "lora_down").replace("lora_B", "lora_up")
        lora_key = lora_key.replace("single_transformer_blocks", "single_blocks")
        lora_key = lora_key.replace("transformer_blocks", "double_blocks")

        double_block_map = {
            "attn.to_out.0": "img_attn.proj",
            "norm1.linear": "img_mod.lin",
            "norm1_context.linear": "txt_mod.lin",
            "attn.to_add_out": "txt_attn.proj",
            "ff.net.0.proj": "img_mlp.0",
            "ff.net.2": "img_mlp.2",
            "ff_context.net.0.proj": "txt_mlp.0",
            "ff_context.net.2": "txt_mlp.2",
            "attn.norm_q": "img_attn.norm.query_norm",
            "attn.norm_k": "img_attn.norm.key_norm",
            "attn.norm_added_q": "txt_attn.norm.query_norm",
            "attn.norm_added_k": "txt_attn.norm.key_norm",
            "attn.to_q": "img_attn.qkv",
            "attn.to_k": "img_attn.qkv",
            "attn.to_v": "img_attn.qkv",
            "attn.add_q_proj": "txt_attn.qkv",
            "attn.add_k_proj": "txt_attn.qkv",
            "attn.add_v_proj": "txt_attn.qkv",
        }
        single_block_map = {
            "norm.linear": "modulation.lin",
            "proj_out": "linear2",
            "attn.norm_q": "norm.query_norm",
            "attn.norm_k": "norm.key_norm",
            "attn.to_q": "linear1",
            "attn.to_k": "linear1",
            "attn.to_v": "linear1",
            "proj_mlp": "linear1",
        }

        # same key exists in both single_block_map and double_block_map, so we must care about single/double
        # print("lora_key before double_block_map", lora_key)
        for old, new in double_block_map.items():
            if "double" in lora_key:
                lora_key = lora_key.replace(old, new)
        # print("lora_key before single_block_map", lora_key)
        for old, new in single_block_map.items():
            if "single" in lora_key:
                lora_key = lora_key.replace(old, new)
        # print("lora_key after mapping", lora_key)

        if lora_key in key_map:
            flux_key = key_map[lora_key]
            logger.info(f"Found matching key: {flux_key}")
            return flux_key

        # If not found in key_map, try partial matching
        potential_key = lora_key + ".weight"
        logger.info(f"Searching for key: {potential_key}")
        matches = [k for k in flux_dict.keys() if potential_key in k]
        if matches:
            logger.info(f"Found matching key: {matches[0]}")
            return matches[0]
        return None

    merged_keys = set()
    for model, ratio in zip(models, ratios):
        logger.info(f"loading: {model}")
        lora_sd, _ = load_state_dict(model, merge_dtype)

        logger.info("merging...")
        for key in lora_sd.keys():
            if "lora_down" in key or "lora_A" in key:
                lora_name = key[: key.rfind(".lora_down" if "lora_down" in key else ".lora_A")]
                up_key = key.replace("lora_down", "lora_up").replace("lora_A", "lora_B")
                alpha_key = key[: key.index("lora_down" if "lora_down" in key else "lora_A")] + "alpha"

                logger.info(f"Processing LoRA key: {lora_name}")
                flux_key = find_matching_key(flux_state_dict, lora_name)

                if flux_key is None:
                    logger.warning(f"no module found for LoRA weight: {key}")
                    continue

                logger.info(f"Merging LoRA key {lora_name} into Flux key {flux_key}")

                down_weight = lora_sd[key]
                up_weight = lora_sd[up_key]

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim

                weight = flux_state_dict[flux_key]

                weight = weight.to(working_device, merge_dtype)
                up_weight = up_weight.to(working_device, merge_dtype)
                down_weight = down_weight.to(working_device, merge_dtype)

                # print(up_weight.size(), down_weight.size(), weight.size())

                if lora_name.startswith("transformer."):
                    if "qkv" in flux_key or "linear1" in flux_key:  # combined qkv or qkv+mlp
                        update = ratio * (up_weight @ down_weight) * scale
                        # print(update.shape)

                        if "img_attn" in flux_key or "txt_attn" in flux_key:
                            q, k, v = torch.chunk(weight, 3, dim=0)
                            if "to_q" in lora_name or "add_q_proj" in lora_name:
                                q += update.reshape(q.shape)
                            elif "to_k" in lora_name or "add_k_proj" in lora_name:
                                k += update.reshape(k.shape)
                            elif "to_v" in lora_name or "add_v_proj" in lora_name:
                                v += update.reshape(v.shape)
                            weight = torch.cat([q, k, v], dim=0)
                        elif "linear1" in flux_key:
                            q, k, v = torch.chunk(weight[: int(update.shape[-1] * 3)], 3, dim=0)
                            mlp = weight[int(update.shape[-1] * 3) :]
                            # print(q.shape, k.shape, v.shape, mlp.shape)
                            if "to_q" in lora_name:
                                q += update.reshape(q.shape)
                            elif "to_k" in lora_name:
                                k += update.reshape(k.shape)
                            elif "to_v" in lora_name:
                                v += update.reshape(v.shape)
                            elif "proj_mlp" in lora_name:
                                mlp += update.reshape(mlp.shape)
                            weight = torch.cat([q, k, v, mlp], dim=0)
                    else:
                        if len(weight.size()) == 2:
                            weight = weight + ratio * (up_weight @ down_weight) * scale
                        elif down_weight.size()[2:4] == (1, 1):
                            weight = (
                                weight
                                + ratio
                                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                                * scale
                            )
                        else:
                            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                            weight = weight + ratio * conved * scale
                else:
                    if len(weight.size()) == 2:
                        weight = weight + ratio * (up_weight @ down_weight) * scale
                    elif down_weight.size()[2:4] == (1, 1):
                        weight = (
                            weight
                            + ratio
                            * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                            * scale
                        )
                    else:
                        conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                        weight = weight + ratio * conved * scale

                flux_state_dict[flux_key] = weight.to(loading_device, save_dtype)
                merged_keys.add(flux_key)
                del up_weight
                del down_weight
                del weight

    logger.info(f"Merged keys: {sorted(list(merged_keys))}")
    return flux_state_dict


def merge_lora_models(models, ratios, merge_dtype, concat=False, shuffle=False):
    base_alphas = {}  # alpha for merged model
    base_dims = {}

    merged_sd = {}
    base_model = None
    for model, ratio in zip(models, ratios):
        logger.info(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)

        if lora_metadata is not None:
            if base_model is None:
                base_model = lora_metadata.get(train_util.SS_METADATA_KEY_BASE_MODEL_VERSION, None)

        # get alpha and dim
        alphas = {}  # alpha for current model
        dims = {}  # dims for current model
        for key in lora_sd.keys():
            if "alpha" in key:
                lora_module_name = key[: key.rfind(".alpha")]
                alpha = float(lora_sd[key].detach().numpy())
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha
            elif "lora_down" in key:
                lora_module_name = key[: key.rfind(".lora_down")]
                dim = lora_sd[key].size()[0]
                dims[lora_module_name] = dim
                if lora_module_name not in base_dims:
                    base_dims[lora_module_name] = dim

        for lora_module_name in dims.keys():
            if lora_module_name not in alphas:
                alpha = dims[lora_module_name]
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha

        logger.info(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        # merge
        logger.info("merging...")
        for key in tqdm(lora_sd.keys()):
            if "alpha" in key:
                continue

            if "lora_up" in key and concat:
                concat_dim = 1
            elif "lora_down" in key and concat:
                concat_dim = 0
            else:
                concat_dim = None

            lora_module_name = key[: key.rfind(".lora_")]

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            scale = math.sqrt(alpha / base_alpha) * ratio
            scale = abs(scale) if "lora_up" in key else scale  # マイナスの重みに対応する。

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == lora_sd[key].size() or concat_dim is not None
                ), "weights shape mismatch, different dims? / 重みのサイズが合いません。dimが異なる可能性があります。"
                if concat_dim is not None:
                    merged_sd[key] = torch.cat([merged_sd[key], lora_sd[key] * scale], dim=concat_dim)
                else:
                    merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)
        if shuffle:
            key_down = lora_module_name + ".lora_down.weight"
            key_up = lora_module_name + ".lora_up.weight"
            dim = merged_sd[key_down].shape[0]
            perm = torch.randperm(dim)
            merged_sd[key_down] = merged_sd[key_down][perm]
            merged_sd[key_up] = merged_sd[key_up][:, perm]

    logger.info("merged model")
    logger.info(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    # check all dims are same
    dims_list = list(set(base_dims.values()))
    alphas_list = list(set(base_alphas.values()))
    all_same_dims = True
    all_same_alphas = True
    for dims in dims_list:
        if dims != dims_list[0]:
            all_same_dims = False
            break
    for alphas in alphas_list:
        if alphas != alphas_list[0]:
            all_same_alphas = False
            break

    # build minimum metadata
    dims = f"{dims_list[0]}" if all_same_dims else "Dynamic"
    alphas = f"{alphas_list[0]}" if all_same_alphas else "Dynamic"
    metadata = train_util.build_minimum_network_metadata(str(False), base_model, "networks.lora", dims, alphas, None)

    return merged_sd, metadata


def merge(args):
    if args.models is None:
        args.models = []
    if args.ratios is None:
        args.ratios = []

    assert len(args.models) == len(
        args.ratios
    ), "number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    assert (
        args.save_to or args.clip_l_save_to or args.t5xxl_save_to
    ), "save_to or clip_l_save_to or t5xxl_save_to must be specified / save_toまたはclip_l_save_toまたはt5xxl_save_toを指定してください"
    dest_dir = os.path.dirname(args.save_to or args.clip_l_save_to or args.t5xxl_save_to)
    if not os.path.exists(dest_dir):
        logger.info(f"creating directory: {dest_dir}")
        os.makedirs(dest_dir)

    if args.flux_model is not None or args.clip_l is not None or args.t5xxl is not None:
        if not args.diffusers:
            assert (args.clip_l is None and args.clip_l_save_to is None) or (
                args.clip_l is not None and args.clip_l_save_to is not None
            ), "clip_l_save_to must be specified if clip_l is specified / clip_lが指定されている場合はclip_l_save_toも指定してください"
            assert (args.t5xxl is None and args.t5xxl_save_to is None) or (
                args.t5xxl is not None and args.t5xxl_save_to is not None
            ), "t5xxl_save_to must be specified if t5xxl is specified / t5xxlが指定されている場合はt5xxl_save_toも指定してください"
            flux_state_dict, clip_l_state_dict, t5xxl_state_dict = merge_to_flux_model(
                args.loading_device,
                args.working_device,
                args.flux_model,
                args.clip_l,
                args.t5xxl,
                args.models,
                args.ratios,
                merge_dtype,
                save_dtype,
                args.mem_eff_load_save,
            )
        else:
            assert (
                args.clip_l is None and args.t5xxl is None
            ), "clip_l and t5xxl are not supported with --diffusers / clip_l、t5xxlはDiffusersではサポートされていません"
            flux_state_dict = merge_to_flux_model_diffusers(
                args.loading_device,
                args.working_device,
                args.flux_model,
                args.models,
                args.ratios,
                merge_dtype,
                save_dtype,
                args.mem_eff_load_save,
            )
            clip_l_state_dict = None
            t5xxl_state_dict = None

        if args.no_metadata or (flux_state_dict is None or len(flux_state_dict) == 0):
            sai_metadata = None
        else:
            merged_from = sai_model_spec.build_merged_from([args.flux_model] + args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                None, False, False, False, False, False, time.time(), title=title, merged_from=merged_from, flux="dev"
            )

        if flux_state_dict is not None and len(flux_state_dict) > 0:
            logger.info(f"saving FLUX model to: {args.save_to}")
            save_to_file(args.save_to, flux_state_dict, save_dtype, sai_metadata, args.mem_eff_load_save)

        if clip_l_state_dict is not None and len(clip_l_state_dict) > 0:
            logger.info(f"saving clip_l model to: {args.clip_l_save_to}")
            save_to_file(args.clip_l_save_to, clip_l_state_dict, save_dtype, None, args.mem_eff_load_save)

        if t5xxl_state_dict is not None and len(t5xxl_state_dict) > 0:
            logger.info(f"saving t5xxl model to: {args.t5xxl_save_to}")
            save_to_file(args.t5xxl_save_to, t5xxl_state_dict, save_dtype, None, args.mem_eff_load_save)

    else:
        flux_state_dict, metadata = merge_lora_models(args.models, args.ratios, merge_dtype, args.concat, args.shuffle)

        logger.info("calculating hashes and creating metadata...")

        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(flux_state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        if not args.no_metadata:
            merged_from = sai_model_spec.build_merged_from(args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                flux_state_dict, False, False, False, True, False, time.time(), title=title, merged_from=merged_from, flux="dev"
            )
            metadata.update(sai_metadata)

        logger.info(f"saving model to: {args.save_to}")
        save_to_file(args.save_to, flux_state_dict, save_dtype, metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        help="precision in saving, same to merging if omitted. supported types: "
        "float32, fp16, bf16, fp8 (same as fp8_e4m3fn), fp8_e4m3fn, fp8_e4m3fnuz, fp8_e5m2, fp8_e5m2fnuz"
        " / 保存時に精度を変更して保存する、省略時はマージ時の精度と同じ",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        help="precision in merging (float is recommended) / マージの計算時の精度（floatを推奨）",
    )
    parser.add_argument(
        "--flux_model",
        type=str,
        default=None,
        help="FLUX.1 model to load, merge LoRA models if omitted / 読み込むモデル、指定しない場合はLoRAモデルをマージする",
    )
    parser.add_argument(
        "--clip_l",
        type=str,
        default=None,
        help="path to clip_l (*.sft or *.safetensors), should be float16 / clip_lのパス（*.sftまたは*.safetensors）",
    )
    parser.add_argument(
        "--t5xxl",
        type=str,
        default=None,
        help="path to t5xxl (*.sft or *.safetensors), should be float16 / t5xxlのパス（*.sftまたは*.safetensors）",
    )
    parser.add_argument(
        "--mem_eff_load_save",
        action="store_true",
        help="use custom memory efficient load and save functions for FLUX.1 model"
        " / カスタムのメモリ効率の良い読み込みと保存関数をFLUX.1モデルに使用する",
    )
    parser.add_argument(
        "--loading_device",
        type=str,
        default="cpu",
        help="device to load FLUX.1 model. LoRA models are loaded on CPU / FLUX.1モデルを読み込むデバイス。LoRAモデルはCPUで読み込まれます",
    )
    parser.add_argument(
        "--working_device",
        type=str,
        default="cpu",
        help="device to work (merge). Merging LoRA models are done on CPU."
        + " / 作業（マージ）するデバイス。LoRAモデルのマージはCPUで行われます。",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="destination file name: safetensors file / 保存先のファイル名、safetensorsファイル",
    )
    parser.add_argument(
        "--clip_l_save_to",
        type=str,
        default=None,
        help="destination file name for clip_l: safetensors file / clip_lの保存先のファイル名、safetensorsファイル",
    )
    parser.add_argument(
        "--t5xxl_save_to",
        type=str,
        default=None,
        help="destination file name for t5xxl: safetensors file / t5xxlの保存先のファイル名、safetensorsファイル",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="LoRA models to merge: safetensors file / マージするLoRAモデル、safetensorsファイル",
    )
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each model / それぞれのLoRAモデルの比率")
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのメタデータを保存しない（LoRAの最低限のss_metadataは保存される）",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="concat lora instead of merge (The dim(rank) of the output LoRA is the sum of the input dims) / "
        + "マージの代わりに結合する（LoRAのdim(rank)は入力dimの合計になる）",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle lora weight./ " + "LoRAの重みをシャッフルする",
    )
    parser.add_argument(
        "--diffusers",
        action="store_true",
        help="merge Diffusers (?) LoRA models / Diffusers (?) LoRAモデルをマージする",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    merge(args)
