# temporary minimum implementation of LoRA
# FLUX doesn't have Conv2d, so we ignore it
# TODO commonize with the original implementation

# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Type, Union
from diffusers import AutoencoderKL
from transformers import CLIPTextModel
import numpy as np
import torch
from torch import Tensor
import re
from library.utils import setup_logging
from library.sdxl_original_unet import SdxlUNet2DConditionModel

setup_logging()
import logging

logger = logging.getLogger(__name__)


NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        split_dims: Optional[List[int]] = None,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
    ):
        """
        if alpha == 0 or None, alpha is rank (no scaling).

        split_dims is used to mimic the split qkv of FLUX as same as Diffusers
        """
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        self.split_dims = split_dims

        if split_dims is None:
            if org_module.__class__.__name__ == "Conv2d":
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
                self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
            else:
                self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
                self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_up.weight)
        else:
            # conv2d not supported
            assert sum(split_dims) == out_dim, "sum of split_dims must be equal to out_dim"
            assert org_module.__class__.__name__ == "Linear", "split_dims is only supported for Linear"
            # print(f"split_dims: {split_dims}")
            self.lora_down = torch.nn.ModuleList(
                [torch.nn.Linear(in_dim, self.lora_dim, bias=False) for _ in range(len(split_dims))]
            )
            self.lora_up = torch.nn.ModuleList([torch.nn.Linear(self.lora_dim, split_dim, bias=False) for split_dim in split_dims])
            for lora_down in self.lora_down:
                torch.nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
            for lora_up in self.lora_up:
                torch.nn.init.zeros_(lora_up.weight)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.ggpo_sigma = ggpo_sigma
        self.ggpo_beta = ggpo_beta

        if self.ggpo_beta is not None and self.ggpo_sigma is not None:
            self.combined_weight_norms = None
            self.grad_norms = None
            self.perturbation_norm_factor = 1.0 / math.sqrt(org_module.weight.shape[0])
            self.initialize_norm_cache(org_module.weight)
            self.org_module_shape: tuple[int] = org_module.weight.shape

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward

        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        if self.split_dims is None:
            lx = self.lora_down(x)

            # normal dropout
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            # rank dropout
            if self.rank_dropout is not None and self.training:
                mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
                if len(lx.size()) == 3:
                    mask = mask.unsqueeze(1)  # for Text Encoder
                elif len(lx.size()) == 4:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
                lx = lx * mask

                # scaling for rank dropout: treat as if the rank is changed
                # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
            else:
                scale = self.scale

            lx = self.lora_up(lx)

            # LoRA Gradient-Guided Perturbation Optimization
            if (
                self.training
                and self.ggpo_sigma is not None
                and self.ggpo_beta is not None
                and self.combined_weight_norms is not None
                and self.grad_norms is not None
            ):
                with torch.no_grad():
                    perturbation_scale = (self.ggpo_sigma * torch.sqrt(self.combined_weight_norms**2)) + (
                        self.ggpo_beta * (self.grad_norms**2)
                    )
                    perturbation_scale_factor = (perturbation_scale * self.perturbation_norm_factor).to(self.device)
                    perturbation = torch.randn(self.org_module_shape, dtype=self.dtype, device=self.device)
                    perturbation.mul_(perturbation_scale_factor)
                    perturbation_output = x @ perturbation.T  # Result: (batch × n)
                return org_forwarded + (self.multiplier * scale * lx) + perturbation_output
            else:
                return org_forwarded + lx * self.multiplier * scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]

            # normal dropout
            if self.dropout is not None and self.training:
                lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

            # rank dropout
            if self.rank_dropout is not None and self.training:
                masks = [torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout for lx in lxs]
                for i in range(len(lxs)):
                    if len(lx.size()) == 3:
                        masks[i] = masks[i].unsqueeze(1)
                    elif len(lx.size()) == 4:
                        masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                    lxs[i] = lxs[i] * masks[i]

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
            else:
                scale = self.scale

            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]

            return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * scale

    @torch.no_grad()
    def initialize_norm_cache(self, org_module_weight: Tensor):
        # Choose a reasonable sample size
        n_rows = org_module_weight.shape[0]
        sample_size = min(1000, n_rows)  # Cap at 1000 samples or use all if smaller

        # Sample random indices across all rows
        indices = torch.randperm(n_rows)[:sample_size]

        # Convert to a supported data type first, then index
        # Use float32 for indexing operations
        weights_float32 = org_module_weight.to(dtype=torch.float32)
        sampled_weights = weights_float32[indices].to(device=self.device)

        # Calculate sampled norms
        sampled_norms = torch.norm(sampled_weights, dim=1, keepdim=True)

        # Store the mean norm as our estimate
        self.org_weight_norm_estimate = sampled_norms.mean()

        # Optional: store standard deviation for confidence intervals
        self.org_weight_norm_std = sampled_norms.std()

        # Free memory
        del sampled_weights, weights_float32

    @torch.no_grad()
    def validate_norm_approximation(self, org_module_weight: Tensor, verbose=True):
        # Calculate the true norm (this will be slow but it's just for validation)
        true_norms = []
        chunk_size = 1024  # Process in chunks to avoid OOM

        for i in range(0, org_module_weight.shape[0], chunk_size):
            end_idx = min(i + chunk_size, org_module_weight.shape[0])
            chunk = org_module_weight[i:end_idx].to(device=self.device, dtype=self.dtype)
            chunk_norms = torch.norm(chunk, dim=1, keepdim=True)
            true_norms.append(chunk_norms.cpu())
            del chunk

        true_norms = torch.cat(true_norms, dim=0)
        true_mean_norm = true_norms.mean().item()

        # Compare with our estimate
        estimated_norm = self.org_weight_norm_estimate.item()

        # Calculate error metrics
        absolute_error = abs(true_mean_norm - estimated_norm)
        relative_error = absolute_error / true_mean_norm * 100  # as percentage

        if verbose:
            logger.info(f"True mean norm: {true_mean_norm:.6f}")
            logger.info(f"Estimated norm: {estimated_norm:.6f}")
            logger.info(f"Absolute error: {absolute_error:.6f}")
            logger.info(f"Relative error: {relative_error:.2f}%")

        return {
            "true_mean_norm": true_mean_norm,
            "estimated_norm": estimated_norm,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
        }

    @torch.no_grad()
    def update_norms(self):
        # Not running GGPO so not currently running update norms
        if self.ggpo_beta is None or self.ggpo_sigma is None:
            return

        # only update norms when we are training
        if self.training is False:
            return

        module_weights = self.lora_up.weight @ self.lora_down.weight
        module_weights.mul(self.scale)

        self.weight_norms = torch.norm(module_weights, dim=1, keepdim=True)
        self.combined_weight_norms = torch.sqrt(
            (self.org_weight_norm_estimate**2) + torch.sum(module_weights**2, dim=1, keepdim=True)
        )

    @torch.no_grad()
    def update_grad_norms(self):
        if self.training is False:
            print(f"skipping update_grad_norms for {self.lora_name}")
            return

        lora_down_grad = None
        lora_up_grad = None

        for name, param in self.named_parameters():
            if name == "lora_down.weight":
                lora_down_grad = param.grad
            elif name == "lora_up.weight":
                lora_up_grad = param.grad

        # Calculate gradient norms if we have both gradients
        if lora_down_grad is not None and lora_up_grad is not None:
            with torch.autocast(self.device.type):
                approx_grad = self.scale * ((self.lora_up.weight @ lora_down_grad) + (lora_up_grad @ self.lora_down.weight))
                self.grad_norms = torch.norm(approx_grad, dim=1, keepdim=True)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        self.org_module_ref = [org_module]  # 後から参照できるように
        self.enabled = True
        self.network: LoRANetwork = None

    def set_network(self, network):
        self.network = network

    # freezeしてマージする
    def merge_to(self, sd, dtype, device):
        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(torch.float)  # calc in float

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        if self.split_dims is None:
            # get up/down weight
            down_weight = sd["lora_down.weight"].to(torch.float).to(device)
            up_weight = sd["lora_up.weight"].to(torch.float).to(device)

            # merge weight
            if len(weight.size()) == 2:
                # linear
                weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                    weight
                    + self.multiplier
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * self.scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                weight = weight + self.multiplier * conved * self.scale

            # set weight to org_module
            org_sd["weight"] = weight.to(dtype)
            self.org_module.load_state_dict(org_sd)
        else:
            # split_dims
            total_dims = sum(self.split_dims)
            for i in range(len(self.split_dims)):
                # get up/down weight
                down_weight = sd[f"lora_down.{i}.weight"].to(torch.float).to(device)  # (rank, in_dim)
                up_weight = sd[f"lora_up.{i}.weight"].to(torch.float).to(device)  # (split dim, rank)

                # pad up_weight -> (total_dims, rank)
                padded_up_weight = torch.zeros((total_dims, up_weight.size(0)), device=device, dtype=torch.float)
                padded_up_weight[sum(self.split_dims[:i]) : sum(self.split_dims[: i + 1])] = up_weight

                # merge weight
                weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale

            # set weight to org_module
            org_sd["weight"] = weight.to(dtype)
            self.org_module.load_state_dict(org_sd)

    # 復元できるマージのため、このモジュールのweightを返す
    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        # get up/down weight from module
        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # pre-calculated weight
        if len(down_weight.size()) == 2:
            # linear
            weight = self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = self.multiplier * conved * self.scale

        return weight

    def set_region(self, region):
        self.region = region
        self.region_mask = None

    def default_forward(self, x):
        # logger.info(f"default_forward {self.lora_name} {x.size()}")
        if self.split_dims is None:
            lx = self.lora_down(x)
            lx = self.lora_up(lx)
            return self.org_forward(x) + lx * self.multiplier * self.scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]
            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]
            return self.org_forward(x) + torch.cat(lxs, dim=-1) * self.multiplier * self.scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    ae: AutoencoderKL,
    text_encoders: List[CLIPTextModel],
    flux,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # attn dim, mlp dim: only for DoubleStreamBlock. SingleStreamBlock is not supported because of combined qkv
    img_attn_dim = kwargs.get("img_attn_dim", None)
    txt_attn_dim = kwargs.get("txt_attn_dim", None)
    img_mlp_dim = kwargs.get("img_mlp_dim", None)
    txt_mlp_dim = kwargs.get("txt_mlp_dim", None)
    img_mod_dim = kwargs.get("img_mod_dim", None)
    txt_mod_dim = kwargs.get("txt_mod_dim", None)
    single_dim = kwargs.get("single_dim", None)  # SingleStreamBlock
    single_mod_dim = kwargs.get("single_mod_dim", None)  # SingleStreamBlock
    if img_attn_dim is not None:
        img_attn_dim = int(img_attn_dim)
    if txt_attn_dim is not None:
        txt_attn_dim = int(txt_attn_dim)
    if img_mlp_dim is not None:
        img_mlp_dim = int(img_mlp_dim)
    if txt_mlp_dim is not None:
        txt_mlp_dim = int(txt_mlp_dim)
    if img_mod_dim is not None:
        img_mod_dim = int(img_mod_dim)
    if txt_mod_dim is not None:
        txt_mod_dim = int(txt_mod_dim)
    if single_dim is not None:
        single_dim = int(single_dim)
    if single_mod_dim is not None:
        single_mod_dim = int(single_mod_dim)
    type_dims = [img_attn_dim, txt_attn_dim, img_mlp_dim, txt_mlp_dim, img_mod_dim, txt_mod_dim, single_dim, single_mod_dim]
    if all([d is None for d in type_dims]):
        type_dims = None

    # in_dims [img, time, vector, guidance, txt]
    in_dims = kwargs.get("in_dims", None)
    if in_dims is not None:
        in_dims = in_dims.strip()
        if in_dims.startswith("[") and in_dims.endswith("]"):
            in_dims = in_dims[1:-1]
        in_dims = [int(d) for d in in_dims.split(",")]  # is it better to use ast.literal_eval?
        assert len(in_dims) == 5, f"invalid in_dims: {in_dims}, must be 5 dimensions (img, time, vector, guidance, txt)"

    # double/single train blocks
    def parse_block_selection(selection: str, total_blocks: int) -> List[bool]:
        """
        Parse a block selection string and return a list of booleans.

        Args:
        selection (str): A string specifying which blocks to select.
        total_blocks (int): The total number of blocks available.

        Returns:
        List[bool]: A list of booleans indicating which blocks are selected.
        """
        if selection == "all":
            return [True] * total_blocks
        if selection == "none" or selection == "":
            return [False] * total_blocks

        selected = [False] * total_blocks
        ranges = selection.split(",")

        for r in ranges:
            if "-" in r:
                start, end = map(str.strip, r.split("-"))
                start = int(start)
                end = int(end)
                assert 0 <= start < total_blocks, f"invalid start index: {start}"
                assert 0 <= end < total_blocks, f"invalid end index: {end}"
                assert start <= end, f"invalid range: {start}-{end}"
                for i in range(start, end + 1):
                    selected[i] = True
            else:
                index = int(r)
                assert 0 <= index < total_blocks, f"invalid index: {index}"
                selected[index] = True

        return selected

    train_double_block_indices = kwargs.get("train_double_block_indices", None)
    train_single_block_indices = kwargs.get("train_single_block_indices", None)
    if train_double_block_indices is not None:
        train_double_block_indices = parse_block_selection(train_double_block_indices, NUM_DOUBLE_BLOCKS)
    if train_single_block_indices is not None:
        train_single_block_indices = parse_block_selection(train_single_block_indices, NUM_SINGLE_BLOCKS)

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # single or double blocks
    train_blocks = kwargs.get("train_blocks", None)  # None (default), "all" (same as None), "single", "double"
    if train_blocks is not None:
        assert train_blocks in ["all", "single", "double"], f"invalid train_blocks: {train_blocks}"

    # split qkv
    split_qkv = kwargs.get("split_qkv", False)
    if split_qkv is not None:
        split_qkv = True if split_qkv == "True" else False

    ggpo_beta = kwargs.get("ggpo_beta", None)
    ggpo_sigma = kwargs.get("ggpo_sigma", None)

    if ggpo_beta is not None:
        ggpo_beta = float(ggpo_beta)

    if ggpo_sigma is not None:
        ggpo_sigma = float(ggpo_sigma)

    # train T5XXL
    train_t5xxl = kwargs.get("train_t5xxl", False)
    if train_t5xxl is not None:
        train_t5xxl = True if train_t5xxl == "True" else False

    # verbose
    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    # regex-specific learning rates
    def parse_kv_pairs(kv_pair_str: str, is_int: bool) -> Dict[str, float]:
        """
        Parse a string of key-value pairs separated by commas.
        """
        pairs = {}
        for pair in kv_pair_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                logger.warning(f"Invalid format: {pair}, expected 'key=value'")
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                pairs[key] = int(value) if is_int else float(value)
            except ValueError:
                logger.warning(f"Invalid value for {key}: {value}")
        return pairs

    # parse regular expression based learning rates
    network_reg_lrs = kwargs.get("network_reg_lrs", None)
    if network_reg_lrs is not None:
        reg_lrs = parse_kv_pairs(network_reg_lrs, is_int=False)
    else:
        reg_lrs = None

    # regex-specific dimensions (ranks)
    network_reg_dims = kwargs.get("network_reg_dims", None)
    if network_reg_dims is not None:
        reg_dims = parse_kv_pairs(network_reg_dims, is_int=True)
    else:
        reg_dims = None

    # すごく引数が多いな ( ^ω^)･･･
    network = LoRANetwork(
        text_encoders,
        flux,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        train_blocks=train_blocks,
        split_qkv=split_qkv,
        train_t5xxl=train_t5xxl,
        type_dims=type_dims,
        in_dims=in_dims,
        train_double_block_indices=train_double_block_indices,
        train_single_block_indices=train_single_block_indices,
        reg_dims=reg_dims,
        ggpo_beta=ggpo_beta,
        ggpo_sigma=ggpo_sigma,
        reg_lrs=reg_lrs,
        verbose=verbose,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return network


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(multiplier, file, ae, text_encoders, flux, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping, and train t5xxl
    modules_dim = {}
    modules_alpha = {}
    train_t5xxl = None
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
            # logger.info(lora_name, value.size(), dim)

        if train_t5xxl is None or train_t5xxl is False:
            train_t5xxl = "lora_te3" in lora_name

    if train_t5xxl is None:
        train_t5xxl = False

    split_qkv = False  # split_qkv is not needed to care, because state_dict is qkv combined

    module_class = LoRAInfModule if for_inference else LoRAModule

    network = LoRANetwork(
        text_encoders,
        flux,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        split_qkv=split_qkv,
        train_t5xxl=train_t5xxl,
    )
    return network, weights_sd


class LoRANetwork(torch.nn.Module):
    # FLUX_TARGET_REPLACE_MODULE = ["DoubleStreamBlock", "SingleStreamBlock"]
    FLUX_TARGET_REPLACE_MODULE_DOUBLE = ["DoubleStreamBlock"]
    FLUX_TARGET_REPLACE_MODULE_SINGLE = ["SingleStreamBlock"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP", "T5Attention", "T5DenseGatedActDense"]
    LORA_PREFIX_FLUX = "lora_unet"  # make ComfyUI compatible
    LORA_PREFIX_TEXT_ENCODER_CLIP = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER_T5 = "lora_te3"  # make ComfyUI compatible

    @classmethod
    def get_qkv_mlp_split_dims(cls) -> List[int]:
        return [3072] * 3 + [12288]

    def __init__(
        self,
        text_encoders: Union[List[CLIPTextModel], CLIPTextModel],
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        train_blocks: Optional[str] = None,
        split_qkv: bool = False,
        train_t5xxl: bool = False,
        type_dims: Optional[List[int]] = None,
        in_dims: Optional[List[int]] = None,
        train_double_block_indices: Optional[List[bool]] = None,
        train_single_block_indices: Optional[List[bool]] = None,
        reg_dims: Optional[Dict[str, int]] = None,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        reg_lrs: Optional[Dict[str, float]] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.train_blocks = train_blocks if train_blocks is not None else "all"
        self.split_qkv = split_qkv
        self.train_t5xxl = train_t5xxl

        self.type_dims = type_dims
        self.in_dims = in_dims
        self.train_double_block_indices = train_double_block_indices
        self.train_single_block_indices = train_single_block_indices
        self.reg_dims = reg_dims
        self.reg_lrs = reg_lrs

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info(f"create LoRA network from weights")
            self.in_dims = [0] * 5  # create in_dims
            # verbose = True
        else:
            logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )
            # if self.conv_lora_dim is not None:
            #     logger.info(
            #         f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}"
            #     )

        if ggpo_beta is not None and ggpo_sigma is not None:
            logger.info(f"LoRA-GGPO training sigma: {ggpo_sigma} beta: {ggpo_beta}")

        if self.split_qkv:
            logger.info(f"split qkv for LoRA")
        if self.train_blocks is not None:
            logger.info(f"train {self.train_blocks} blocks only")

        if train_t5xxl:
            logger.info(f"train T5XXL as well")

        # create module instances
        def create_modules(
            is_flux: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            filter: Optional[str] = None,
            default_dim: Optional[int] = None,
        ) -> List[LoRAModule]:
            prefix = (
                self.LORA_PREFIX_FLUX
                if is_flux
                else (self.LORA_PREFIX_TEXT_ENCODER_CLIP if text_encoder_idx == 0 else self.LORA_PREFIX_TEXT_ENCODER_T5)
            )

            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if target_replace_modules is None or module.__class__.__name__ in target_replace_modules:
                    if target_replace_modules is None:  # dirty hack for all modules
                        module = root_module  # search all modules

                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + (name + "." if name else "") + child_name
                            lora_name = lora_name.replace(".", "_")

                            if filter is not None and not filter in lora_name:
                                continue

                            dim = None
                            alpha = None

                            if modules_dim is not None:
                                # モジュール指定あり
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            elif self.reg_dims is not None:
                                for reg, d in self.reg_dims.items():
                                    if re.search(reg, lora_name):
                                        dim = d
                                        alpha = self.alpha
                                        logger.info(f"LoRA {lora_name} matched with regex {reg}, using dim: {dim}")
                                        break

                            # if modules_dim is None, we use default lora_dim. if modules_dim is not None, we use the specified dim (no default)
                            if dim is None and modules_dim is None:
                                if is_linear or is_conv2d_1x1:
                                    dim = default_dim if default_dim is not None else self.lora_dim
                                    alpha = self.alpha

                                    if is_flux and type_dims is not None:
                                        identifier = [
                                            ("img_attn",),
                                            ("txt_attn",),
                                            ("img_mlp",),
                                            ("txt_mlp",),
                                            ("img_mod",),
                                            ("txt_mod",),
                                            ("single_blocks", "linear"),
                                            ("modulation",),
                                        ]
                                        for i, d in enumerate(type_dims):
                                            if d is not None and all([id in lora_name for id in identifier[i]]):
                                                dim = d  # may be 0 for skip
                                                break

                                    if (
                                        is_flux
                                        and dim
                                        and (
                                            self.train_double_block_indices is not None
                                            or self.train_single_block_indices is not None
                                        )
                                        and ("double" in lora_name or "single" in lora_name)
                                    ):
                                        # "lora_unet_double_blocks_0_..." or "lora_unet_single_blocks_0_..."
                                        block_index = int(lora_name.split("_")[4])  # bit dirty
                                        if (
                                            "double" in lora_name
                                            and self.train_double_block_indices is not None
                                            and not self.train_double_block_indices[block_index]
                                        ):
                                            dim = 0
                                        elif (
                                            "single" in lora_name
                                            and self.train_single_block_indices is not None
                                            and not self.train_single_block_indices[block_index]
                                        ):
                                            dim = 0

                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                            if dim is None or dim == 0:
                                # skipした情報を出力
                                if is_linear or is_conv2d_1x1 or (self.conv_lora_dim is not None):
                                    skipped.append(lora_name)
                                continue

                            # qkv split
                            split_dims = None
                            if is_flux and split_qkv:
                                if "double" in lora_name and "qkv" in lora_name:
                                    (split_dims,) = self.get_qkv_mlp_split_dims()[:3]  # qkv only
                                elif "single" in lora_name and "linear1" in lora_name:
                                    split_dims = self.get_qkv_mlp_split_dims()  # qkv + mlp

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                split_dims=split_dims,
                                ggpo_beta=ggpo_beta,
                                ggpo_sigma=ggpo_sigma,
                            )
                            loras.append(lora)

                if target_replace_modules is None:
                    break  # all modules are searched
            return loras, skipped

        # create LoRA for text encoder
        # 毎回すべてのモジュールを作るのは無駄なので要検討
        self.text_encoder_loras: List[Union[LoRAModule, LoRAInfModule]] = []
        skipped_te = []
        for i, text_encoder in enumerate(text_encoders):
            index = i
            if text_encoder is None:
                logger.info(f"Text Encoder {index+1} is None, skipping LoRA creation for this encoder.")
                continue
            if not train_t5xxl and index > 0:  # 0: CLIP, 1: T5XXL, so we skip T5XXL if train_t5xxl is False
                break

            logger.info(f"create LoRA for Text Encoder {index+1}:")

            text_encoder_loras, skipped = create_modules(False, index, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
            logger.info(f"create LoRA for Text Encoder {index+1}: {len(text_encoder_loras)} modules.")
            self.text_encoder_loras.extend(text_encoder_loras)
            skipped_te += skipped

        # create LoRA for U-Net
        if self.train_blocks == "all":
            target_replace_modules = LoRANetwork.FLUX_TARGET_REPLACE_MODULE_DOUBLE + LoRANetwork.FLUX_TARGET_REPLACE_MODULE_SINGLE
        elif self.train_blocks == "single":
            target_replace_modules = LoRANetwork.FLUX_TARGET_REPLACE_MODULE_SINGLE
        elif self.train_blocks == "double":
            target_replace_modules = LoRANetwork.FLUX_TARGET_REPLACE_MODULE_DOUBLE

        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_replace_modules)

        # img, time, vector, guidance, txt
        if self.in_dims:
            for filter, in_dim in zip(["_img_in", "_time_in", "_vector_in", "_guidance_in", "_txt_in"], self.in_dims):
                loras, _ = create_modules(True, None, unet, None, filter=filter, default_dim=in_dim)
                self.unet_loras.extend(loras)

        logger.info(f"create LoRA for FLUX {self.train_blocks} blocks: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:50} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_te + skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(
                f"because dim (rank) is 0, {len(skipped)} LoRA modules are skipped / dim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                logger.info(f"\t{name}")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def update_norms(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.update_norms()

    def update_grad_norms(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.update_grad_norms()

    def grad_norms(self) -> Tensor | None:
        grad_norms = []
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "grad_norms") and lora.grad_norms is not None:
                grad_norms.append(lora.grad_norms.mean(dim=0))
        return torch.stack(grad_norms) if len(grad_norms) > 0 else None

    def weight_norms(self) -> Tensor | None:
        weight_norms = []
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "weight_norms") and lora.weight_norms is not None:
                weight_norms.append(lora.weight_norms.mean(dim=0))
        return torch.stack(weight_norms) if len(weight_norms) > 0 else None

    def combined_weight_norms(self) -> Tensor | None:
        combined_weight_norms = []
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "combined_weight_norms") and lora.combined_weight_norms is not None:
                combined_weight_norms.append(lora.combined_weight_norms.mean(dim=0))
        return torch.stack(combined_weight_norms) if len(combined_weight_norms) > 0 else None

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def load_state_dict(self, state_dict, strict=True):
        # override to convert original weight to split qkv
        if not self.split_qkv:
            return super().load_state_dict(state_dict, strict)

        # split qkv
        for key in list(state_dict.keys()):
            if "double" in key and "qkv" in key:
                split_dims = self.get_qkv_mlp_split_dims()[:3]  # qkv only
            elif "single" in key and "linear1" in key:
                split_dims = self.get_qkv_mlp_split_dims()  # qkv + mlp
            else:
                continue

            weight = state_dict[key]
            lora_name = key.split(".")[0]
            if "lora_down" in key and "weight" in key:
                # dense weight (rank*3, in_dim)
                split_weight = torch.chunk(weight, len(split_dims), dim=0)
                for i, split_w in enumerate(split_weight):
                    state_dict[f"{lora_name}.lora_down.{i}.weight"] = split_w

                del state_dict[key]
                # print(f"split {key}: {weight.shape} to {[w.shape for w in split_weight]}")
            elif "lora_up" in key and "weight" in key:
                # sparse weight (out_dim=sum(split_dims), rank*3)
                rank = weight.size(1) // len(split_dims)
                i = 0
                for j in range(len(split_dims)):
                    state_dict[f"{lora_name}.lora_up.{j}.weight"] = weight[i : i + split_dims[j], j * rank : (j + 1) * rank]
                    i += split_dims[j]
                del state_dict[key]

                # # check is sparse
                # i = 0
                # is_zero = True
                # for j in range(len(split_dims)):
                #     for k in range(len(split_dims)):
                #         if j == k:
                #             continue
                #         is_zero = is_zero and torch.all(weight[i : i + split_dims[j], k * rank : (k + 1) * rank] == 0)
                #     i += split_dims[j]
                # if not is_zero:
                #     logger.warning(f"weight is not sparse: {key}")
                # else:
                #     logger.info(f"weight is sparse: {key}")

                # print(
                #     f"split {key}: {weight.shape} to {[state_dict[k].shape for k in [f'{lora_name}.lora_up.{j}.weight' for j in range(len(split_dims))]]}"
                # )

            # alpha is unchanged

        return super().load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if not self.split_qkv:
            return super().state_dict(destination, prefix, keep_vars)

        # merge qkv
        state_dict = super().state_dict(destination, prefix, keep_vars)
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if "double" in key and "qkv" in key:
                split_dims = self.get_qkv_mlp_split_dims()[:3]  # qkv only
            elif "single" in key and "linear1" in key:
                split_dims = self.get_qkv_mlp_split_dims()  # qkv + mlp
            else:
                new_state_dict[key] = state_dict[key]
                continue

            if key not in state_dict:
                continue  # already merged

            lora_name = key.split(".")[0]

            # (rank, in_dim) * 3
            down_weights = [state_dict.pop(f"{lora_name}.lora_down.{i}.weight") for i in range(len(split_dims))]
            # (split dim, rank) * 3
            up_weights = [state_dict.pop(f"{lora_name}.lora_up.{i}.weight") for i in range(len(split_dims))]

            alpha = state_dict.pop(f"{lora_name}.alpha")

            # merge down weight
            down_weight = torch.cat(down_weights, dim=0)  # (rank, split_dim) * 3 -> (rank*3, sum of split_dim)

            # merge up weight (sum of split_dim, rank*3)
            rank = up_weights[0].size(1)
            up_weight = torch.zeros((sum(split_dims), down_weight.size(0)), device=down_weight.device, dtype=down_weight.dtype)
            i = 0
            for j in range(len(split_dims)):
                up_weight[i : i + split_dims[j], j * rank : (j + 1) * rank] = up_weights[j]
                i += split_dims[j]

            new_state_dict[f"{lora_name}.lora_down.weight"] = down_weight
            new_state_dict[f"{lora_name}.lora_up.weight"] = up_weight
            new_state_dict[f"{lora_name}.alpha"] = alpha

            # print(
            #     f"merged {lora_name}: {lora_name}, {[w.shape for w in down_weights]}, {[w.shape for w in up_weights]} to {down_weight.shape}, {up_weight.shape}"
            # )
            print(f"new key: {lora_name}.lora_down.weight, {lora_name}.lora_up.weight, {lora_name}.alpha")

        return new_state_dict

    def apply_to(self, text_encoders, flux, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable LoRA for U-Net: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    # マージできるかどうかを返す
    def is_mergeable(self):
        return True

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoders, flux, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER_CLIP) or key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER_T5):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_FLUX):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        logger.info(f"weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}")
        logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        # make sure text_encoder_lr as list of two elements
        # if float, use the same value for both text encoders
        if text_encoder_lr is None or (isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0):
            text_encoder_lr = [default_lr, default_lr]
        elif isinstance(text_encoder_lr, float) or isinstance(text_encoder_lr, int):
            text_encoder_lr = [float(text_encoder_lr), float(text_encoder_lr)]
        elif len(text_encoder_lr) == 1:
            text_encoder_lr = [text_encoder_lr[0], text_encoder_lr[0]]

        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        reg_lrs_list = list(self.reg_lrs.items()) if self.reg_lrs is not None else []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            # regular expression param groups: {"reg_lr_0": {"lora": {}, "plus": {}}, ...}
            reg_groups = {}

            for lora in loras:
                # check if this lora matches any regex learning rate
                matched_reg_lr = None
                for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
                    try:
                        if re.search(regex_str, lora.lora_name):
                            matched_reg_lr = (i, reg_lr)
                            logger.info(f"Module {lora.lora_name} matched regex '{regex_str}' -> LR {reg_lr}")
                            break
                    except re.error:
                        # regex error should have been caught during parsing, but just in case
                        continue

                for name, param in lora.named_parameters():
                    param_key = f"{lora.lora_name}.{name}"
                    is_plus = loraplus_ratio is not None and "lora_up" in name

                    if matched_reg_lr is not None:
                        # use regex-specific learning rate
                        reg_idx, reg_lr = matched_reg_lr
                        group_key = f"reg_lr_{reg_idx}"
                        if group_key not in reg_groups:
                            reg_groups[group_key] = {"lora": {}, "plus": {}, "lr": reg_lr}

                        if is_plus:
                            reg_groups[group_key]["plus"][param_key] = param
                        else:
                            reg_groups[group_key]["lora"][param_key] = param
                    else:
                        # use default learning rate
                        if is_plus:
                            param_groups["plus"][param_key] = param
                        else:
                            param_groups["lora"][param_key] = param

            params = []
            descriptions = []

            # process regex-specific groups first (higher priority)
            for group_key in sorted(reg_groups.keys()):
                group = reg_groups[group_key]
                reg_lr = group["lr"]

                for param_type in ["lora", "plus"]:
                    if len(group[param_type]) == 0:
                        continue

                    param_data = {"params": group[param_type].values()}

                    if param_type == "plus" and loraplus_ratio is not None:
                        param_data["lr"] = reg_lr * loraplus_ratio
                    else:
                        param_data["lr"] = reg_lr

                    if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                        continue

                    params.append(param_data)
                    desc = f"reg_lr_{group_key.split('_')[-1]}"
                    if param_type == "plus":
                        desc += " plus"
                    descriptions.append(desc)

            # process default groups
            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}

                if len(param_data["params"]) == 0:
                    continue

                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    else:
                        param_data["lr"] = lr

                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue

                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")

            return params, descriptions

        if self.text_encoder_loras:
            loraplus_lr_ratio = self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio

            # split text encoder loras for te1 and te3
            te1_loras = [lora for lora in self.text_encoder_loras if lora.lora_name.startswith(self.LORA_PREFIX_TEXT_ENCODER_CLIP)]
            te3_loras = [lora for lora in self.text_encoder_loras if lora.lora_name.startswith(self.LORA_PREFIX_TEXT_ENCODER_T5)]
            if len(te1_loras) > 0:
                logger.info(f"Text Encoder 1 (CLIP-L): {len(te1_loras)} modules, LR {text_encoder_lr[0]}")
                params, descriptions = assemble_params(te1_loras, text_encoder_lr[0], loraplus_lr_ratio)
                all_params.extend(params)
                lr_descriptions.extend(["textencoder 1 " + (" " + d if d else "") for d in descriptions])
            if len(te3_loras) > 0:
                logger.info(f"Text Encoder 2 (T5XXL): {len(te3_loras)} modules, LR {text_encoder_lr[1]}")
                params, descriptions = assemble_params(te3_loras, text_encoder_lr[1], loraplus_lr_ratio)
                all_params.extend(params)
                lr_descriptions.extend(["textencoder 2 " + (" " + d if d else "") for d in descriptions])

        if self.unet_loras:
            params, descriptions = assemble_params(
                self.unet_loras,
                unet_lr if unet_lr is not None else default_lr,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        # not supported
        pass

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

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def backup_weights(self):
        # 重みのバックアップを行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        # 重みのリストアを行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        # 事前計算を行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False
            lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)
