# Convert LoRA to different rank approximation (should only be used to go to lower rank)
# This code is based off the extract_lora_from_models.py file which is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo

import argparse
import os
import torch
from safetensors.torch import load_file, save_file, safe_open
from tqdm import tqdm
from library import train_util, model_util
import numpy as np


def load_state_dict(file_name):
    if model_util.is_safetensors(file_name):
        sd = load_file(file_name)
        with safe_open(file_name, framework="pt") as f:
            metadata = f.metadata()
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = None

    return sd, metadata


def save_to_file(file_name, model, metadata):
    if model_util.is_safetensors(file_name):
        save_file(model, file_name, metadata)
    else:
        torch.save(model, file_name)


# Indexing functions


def index_sv_cumulative(S, target):
    original_sum = float(torch.sum(S))
    cumulative_sums = torch.cumsum(S, dim=0) / original_sum
    index = int(torch.searchsorted(cumulative_sums, target)) + 1
    index = max(1, min(index, len(S) - 1))

    return index


def index_sv_fro(S, target):
    S_squared = S.pow(2)
    s_fro_sq = float(torch.sum(S_squared))
    sum_S_squared = torch.cumsum(S_squared, dim=0) / s_fro_sq
    index = int(torch.searchsorted(sum_S_squared, target**2)) + 1
    index = max(1, min(index, len(S) - 1))

    return index


def index_sv_ratio(S, target):
    max_sv = S[0]
    min_sv = max_sv / target
    index = int(torch.sum(S > min_sv).item())
    index = max(1, min(index, len(S) - 1))

    return index


# Modified from Kohaku-blueleaf's extract/merge functions
def extract_conv(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1):
    out_size, in_size, kernel_size, _ = weight.size()
    U, S, Vh = torch.linalg.svd(weight.reshape(out_size, -1).to(device))

    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(lora_rank, in_size, kernel_size, kernel_size).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank, 1, 1).cpu()
    del U, S, Vh, weight
    return param_dict


def extract_linear(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1):
    out_size, in_size = weight.size()

    U, S, Vh = torch.linalg.svd(weight.to(device))

    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(lora_rank, in_size).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank).cpu()
    del U, S, Vh, weight
    return param_dict


def merge_conv(lora_down, lora_up, device):
    in_rank, in_size, kernel_size, k_ = lora_down.shape
    out_size, out_rank, _, _ = lora_up.shape
    assert in_rank == out_rank and kernel_size == k_, f"rank {in_rank} {out_rank} or kernel {kernel_size} {k_} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    merged = lora_up.reshape(out_size, -1) @ lora_down.reshape(in_rank, -1)
    weight = merged.reshape(out_size, in_size, kernel_size, kernel_size)
    del lora_up, lora_down
    return weight


def merge_linear(lora_down, lora_up, device):
    in_rank, in_size = lora_down.shape
    out_size, out_rank = lora_up.shape
    assert in_rank == out_rank, f"rank {in_rank} {out_rank} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    weight = lora_up @ lora_down
    del lora_up, lora_down
    return weight


# Calculate new rank


def rank_resize(S, rank, dynamic_method, dynamic_param, scale=1):
    param_dict = {}

    if dynamic_method == "sv_ratio":
        # Calculate new dim and alpha based off ratio
        new_rank = index_sv_ratio(S, dynamic_param) + 1
        new_alpha = float(scale * new_rank)

    elif dynamic_method == "sv_cumulative":
        # Calculate new dim and alpha based off cumulative sum
        new_rank = index_sv_cumulative(S, dynamic_param) + 1
        new_alpha = float(scale * new_rank)

    elif dynamic_method == "sv_fro":
        # Calculate new dim and alpha based off sqrt sum of squares
        new_rank = index_sv_fro(S, dynamic_param) + 1
        new_alpha = float(scale * new_rank)
    else:
        new_rank = rank
        new_alpha = float(scale * new_rank)

    if S[0] <= MIN_SV:  # Zero matrix, set dim to 1
        new_rank = 1
        new_alpha = float(scale * new_rank)
    elif new_rank > rank:  # cap max rank at rank
        new_rank = rank
        new_alpha = float(scale * new_rank)

    # Calculate resize info
    s_sum = torch.sum(torch.abs(S))
    s_rank = torch.sum(torch.abs(S[:new_rank]))

    S_squared = S.pow(2)
    s_fro = torch.sqrt(torch.sum(S_squared))
    s_red_fro = torch.sqrt(torch.sum(S_squared[:new_rank]))
    fro_percent = float(s_red_fro / s_fro)

    param_dict["new_rank"] = new_rank
    param_dict["new_alpha"] = new_alpha
    param_dict["sum_retained"] = (s_rank) / s_sum
    param_dict["fro_retained"] = fro_percent
    param_dict["max_ratio"] = S[0] / S[new_rank - 1]

    return param_dict


def split_lora_model(lora_sd, unit):
    max_rank = 0

    # Extract loaded lora dim and alpha
    for key, value in lora_sd.items():
        if "lora_down" in key:
            rank = value.size()[0]
            if rank > max_rank:
                max_rank = rank
    print(f"Max rank: {max_rank}")

    rank = unit
    splitted_models = []
    while rank < max_rank:
        print(f"Splitting rank {rank}")
        new_sd = {}
        for key, value in lora_sd.items():
            if "lora_down" in key:
                new_sd[key] = value[:rank].contiguous()
            elif "lora_up" in key:
                new_sd[key] = value[:, :rank].contiguous()
            else:
                new_sd[key] = value  # alpha and other parameters

        splitted_models.append((new_sd, rank))
        rank += unit

    return max_rank, splitted_models


def split(args):
    print("loading Model...")
    lora_sd, metadata = load_state_dict(args.model)

    print("Splitting Model...")
    original_rank, splitted_models = split_lora_model(lora_sd, args.unit)

    comment = metadata.get("ss_training_comment", "")
    for state_dict, new_rank in splitted_models:
        # update metadata
        if metadata is None:
            new_metadata = {}
        else:
            new_metadata = metadata.copy()

        new_metadata["ss_training_comment"] = f"split from DyLoRA from {original_rank} to {new_rank}; {comment}"
        new_metadata["ss_network_dim"] = str(new_rank)

        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        filename, ext = os.path.splitext(args.save_to)
        model_file_name = filename + f"-{new_rank:04d}{ext}"

        print(f"saving model to: {model_file_name}")
        save_to_file(model_file_name, state_dict, new_metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--unit", type=int, default=None, help="size of rank to split into / rankを分割するサイズ")
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="destination base file name: ckpt or safetensors file / 保存先のファイル名のbase、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="DyLoRA model to resize at to new rank: ckpt or safetensors file / 読み込むDyLoRAモデル、ckptまたはsafetensors",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    split(args)
