# temporary minimum implementation of LoRA
# SD3 doesn't have Conv2d, so we ignore it
# TODO commonize with the original/SD3/FLUX implementation

# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union
from transformers import CLIPTextModelWithProjection, T5EncoderModel
import numpy as np
import torch
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from networks.lora_flux import LoRAModule, LoRAInfModule
from library import sd3_models


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: sd3_models.SDVAE,
    text_encoders: List[Union[CLIPTextModelWithProjection, T5EncoderModel]],
    mmdit,
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
    context_attn_dim = kwargs.get("context_attn_dim", None)
    context_mlp_dim = kwargs.get("context_mlp_dim", None)
    context_mod_dim = kwargs.get("context_mod_dim", None)
    x_attn_dim = kwargs.get("x_attn_dim", None)
    x_mlp_dim = kwargs.get("x_mlp_dim", None)
    x_mod_dim = kwargs.get("x_mod_dim", None)
    if context_attn_dim is not None:
        context_attn_dim = int(context_attn_dim)
    if context_mlp_dim is not None:
        context_mlp_dim = int(context_mlp_dim)
    if context_mod_dim is not None:
        context_mod_dim = int(context_mod_dim)
    if x_attn_dim is not None:
        x_attn_dim = int(x_attn_dim)
    if x_mlp_dim is not None:
        x_mlp_dim = int(x_mlp_dim)
    if x_mod_dim is not None:
        x_mod_dim = int(x_mod_dim)
    type_dims = [context_attn_dim, context_mlp_dim, context_mod_dim, x_attn_dim, x_mlp_dim, x_mod_dim]
    if all([d is None for d in type_dims]):
        type_dims = None

    # emb_dims [context_embedder, t_embedder, x_embedder, y_embedder, final_mod, final_linear]
    emb_dims = kwargs.get("emb_dims", None)
    if emb_dims is not None:
        emb_dims = emb_dims.strip()
        if emb_dims.startswith("[") and emb_dims.endswith("]"):
            emb_dims = emb_dims[1:-1]
        emb_dims = [int(d) for d in emb_dims.split(",")]  # is it better to use ast.literal_eval?
        assert len(emb_dims) == 6, f"invalid emb_dims: {emb_dims}, must be 6 dimensions (context, t, x, y, final_mod, final_linear)"

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

    train_block_indices = kwargs.get("train_block_indices", None)
    if train_block_indices is not None:
        train_block_indices = parse_block_selection(train_block_indices, 999)  # 999 is a dummy number

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # split qkv
    split_qkv = kwargs.get("split_qkv", False)
    if split_qkv is not None:
        split_qkv = True if split_qkv == "True" else False

    # train T5XXL
    train_t5xxl = kwargs.get("train_t5xxl", False)
    if train_t5xxl is not None:
        train_t5xxl = True if train_t5xxl == "True" else False

    # verbose
    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    # すごく引数が多いな ( ^ω^)･･･
    network = LoRANetwork(
        text_encoders,
        mmdit,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        split_qkv=split_qkv,
        train_t5xxl=train_t5xxl,
        type_dims=type_dims,
        emb_dims=emb_dims,
        train_block_indices=train_block_indices,
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
def create_network_from_weights(multiplier, file, ae, text_encoders, mmdit, weights_sd=None, for_inference=False, **kwargs):
    # if unet is an instance of SdxlUNet2DConditionModel or subclass, set is_sdxl to True
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
        mmdit,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        split_qkv=split_qkv,
        train_t5xxl=train_t5xxl,
    )
    return network, weights_sd


class LoRANetwork(torch.nn.Module):
    SD3_TARGET_REPLACE_MODULE = ["SingleDiTBlock"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP", "T5Attention", "T5DenseGatedActDense"]
    LORA_PREFIX_SD3 = "lora_unet"  # make ComfyUI compatible
    LORA_PREFIX_TEXT_ENCODER_CLIP_L = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER_CLIP_G = "lora_te2"
    LORA_PREFIX_TEXT_ENCODER_T5 = "lora_te3"  # make ComfyUI compatible

    def __init__(
        self,
        text_encoders: List[Union[CLIPTextModelWithProjection, T5EncoderModel]],
        unet: sd3_models.MMDiT,
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
        split_qkv: bool = False,
        train_t5xxl: bool = False,
        type_dims: Optional[List[int]] = None,
        emb_dims: Optional[List[int]] = None,
        train_block_indices: Optional[List[bool]] = None,
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
        self.split_qkv = split_qkv
        self.train_t5xxl = train_t5xxl

        self.type_dims = type_dims
        self.emb_dims = emb_dims
        self.train_block_indices = train_block_indices

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info(f"create LoRA network from weights")
            self.emb_dims = [0] * 6  # create emb_dims
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

        qkv_dim = 0
        if self.split_qkv:
            logger.info(f"split qkv for LoRA")
            qkv_dim = unet.joint_blocks[0].context_block.attn.qkv.weight.size(0)
        if train_t5xxl:
            logger.info(f"train T5XXL as well")

        # create module instances
        def create_modules(
            is_mmdit: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            filter: Optional[str] = None,
            default_dim: Optional[int] = None,
            include_conv2d_if_filter: bool = False,
        ) -> List[LoRAModule]:
            prefix = (
                self.LORA_PREFIX_SD3
                if is_mmdit
                else [self.LORA_PREFIX_TEXT_ENCODER_CLIP_L, self.LORA_PREFIX_TEXT_ENCODER_CLIP_G, self.LORA_PREFIX_TEXT_ENCODER_T5][
                    text_encoder_idx
                ]
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

                            force_incl_conv2d = False
                            if filter is not None:
                                if not filter in lora_name:
                                    continue
                                force_incl_conv2d = include_conv2d_if_filter

                            dim = None
                            alpha = None

                            if modules_dim is not None:
                                # モジュール指定あり
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            else:
                                # 通常、すべて対象とする
                                if is_linear or is_conv2d_1x1:
                                    dim = default_dim if default_dim is not None else self.lora_dim
                                    alpha = self.alpha

                                    if is_mmdit and type_dims is not None:
                                        #     type_dims = [context_attn_dim, context_mlp_dim, context_mod_dim, x_attn_dim, x_mlp_dim, x_mod_dim]
                                        identifier = [
                                            ("context_block", "attn"),
                                            ("context_block", "mlp"),
                                            ("context_block", "adaLN_modulation"),
                                            ("x_block", "attn"),
                                            ("x_block", "mlp"),
                                            ("x_block", "adaLN_modulation"),
                                        ]
                                        for i, d in enumerate(type_dims):
                                            if d is not None and all([id in lora_name for id in identifier[i]]):
                                                dim = d  # may be 0 for skip
                                                break

                                    if is_mmdit and dim and self.train_block_indices is not None and "joint_blocks" in lora_name:
                                        # "lora_unet_joint_blocks_0_x_block_attn_proj..."
                                        block_index = int(lora_name.split("_")[4])  # bit dirty
                                        if self.train_block_indices is not None and not self.train_block_indices[block_index]:
                                            dim = 0

                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha
                                elif force_incl_conv2d:
                                    # x_embedder
                                    dim = default_dim if default_dim is not None else self.lora_dim
                                    alpha = self.alpha

                            if dim is None or dim == 0:
                                # skipした情報を出力
                                if is_linear or is_conv2d_1x1 or (self.conv_lora_dim is not None):
                                    skipped.append(lora_name)
                                continue

                            # qkv split
                            split_dims = None
                            if is_mmdit and split_qkv:
                                if "joint_blocks" in lora_name and "qkv" in lora_name:
                                    split_dims = [qkv_dim // 3] * 3

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
            if not train_t5xxl and index >= 2:  # 0: CLIP-L, 1: CLIP-G, 2: T5XXL, so we skip T5XXL if train_t5xxl is False
                break

            logger.info(f"create LoRA for Text Encoder {index+1}:")

            text_encoder_loras, skipped = create_modules(False, index, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
            logger.info(f"create LoRA for Text Encoder {index+1}: {len(text_encoder_loras)} modules.")
            self.text_encoder_loras.extend(text_encoder_loras)
            skipped_te += skipped

        # create LoRA for U-Net
        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]]
        self.unet_loras, skipped_un = create_modules(True, None, unet, LoRANetwork.SD3_TARGET_REPLACE_MODULE)

        # emb_dims [context_embedder, t_embedder, x_embedder, y_embedder, final_mod, final_linear]
        if self.emb_dims:
            for filter, in_dim in zip(
                [
                    "context_embedder",
                    "_t_embedder",  # don't use "t_embedder" because it's used in "context_embedder"
                    "x_embedder",
                    "y_embedder",
                    "final_layer_adaLN_modulation",
                    "final_layer_linear",
                ],
                self.emb_dims,
            ):
                # x_embedder is conv2d, so we need to include it
                loras, _ = create_modules(
                    True, None, unet, None, filter=filter, default_dim=in_dim, include_conv2d_if_filter=filter == "x_embedder"
                )
                # if len(loras) > 0:
                #     logger.info(f"create LoRA for {filter}: {len(loras)} modules.")
                self.unet_loras.extend(loras)

        logger.info(f"create LoRA for SD3 MMDiT: {len(self.unet_loras)} modules.")
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
            if not ("joint_blocks" in key and "qkv" in key):
                continue

            weight = state_dict[key]
            lora_name = key.split(".")[0]
            if "lora_down" in key and "weight" in key:
                # dense weight (rank*3, in_dim)
                split_weight = torch.chunk(weight, 3, dim=0)
                for i, split_w in enumerate(split_weight):
                    state_dict[f"{lora_name}.lora_down.{i}.weight"] = split_w

                del state_dict[key]
                # print(f"split {key}: {weight.shape} to {[w.shape for w in split_weight]}")
            elif "lora_up" in key and "weight" in key:
                # sparse weight (out_dim=sum(split_dims), rank*3)
                rank = weight.size(1) // 3
                i = 0
                split_dim = weight.shape[0] // 3
                for j in range(3):
                    state_dict[f"{lora_name}.lora_up.{j}.weight"] = weight[i : i + split_dim, j * rank : (j + 1) * rank]
                    i += split_dim
                del state_dict[key]

            # alpha is unchanged

        return super().load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if not self.split_qkv:
            return super().state_dict(destination, prefix, keep_vars)

        # merge qkv
        state_dict = super().state_dict(destination, prefix, keep_vars)
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if not ("joint_blocks" in key and "qkv" in key):
                new_state_dict[key] = state_dict[key]
                continue

            if key not in state_dict:
                continue  # already merged

            lora_name = key.split(".")[0]

            # (rank, in_dim) * 3
            down_weights = [state_dict.pop(f"{lora_name}.lora_down.{i}.weight") for i in range(3)]
            # (split dim, rank) * 3
            up_weights = [state_dict.pop(f"{lora_name}.lora_up.{i}.weight") for i in range(3)]

            alpha = state_dict.pop(f"{lora_name}.alpha")

            # merge down weight
            down_weight = torch.cat(down_weights, dim=0)  # (rank, split_dim) * 3 -> (rank*3, sum of split_dim)

            # merge up weight (sum of split_dim, rank*3)
            split_dim, rank = up_weights[0].size()
            qkv_dim = split_dim * 3
            up_weight = torch.zeros((qkv_dim, down_weight.size(0)), device=down_weight.device, dtype=down_weight.dtype)
            i = 0
            for j in range(3):
                up_weight[i : i + split_dim, j * rank : (j + 1) * rank] = up_weights[j]
                i += split_dim

            new_state_dict[f"{lora_name}.lora_down.weight"] = down_weight
            new_state_dict[f"{lora_name}.lora_up.weight"] = up_weight
            new_state_dict[f"{lora_name}.alpha"] = alpha

            # print(
            #     f"merged {lora_name}: {lora_name}, {[w.shape for w in down_weights]}, {[w.shape for w in up_weights]} to {down_weight.shape}, {up_weight.shape}"
            # )
            print(f"new key: {lora_name}.lora_down.weight, {lora_name}.lora_up.weight, {lora_name}.alpha")

        return new_state_dict

    def apply_to(self, text_encoders, mmdit, apply_text_encoder=True, apply_unet=True):
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
    def merge_to(self, text_encoders, mmdit, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if (
                key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER_CLIP_L)
                or key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER_CLIP_G)
                or key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER_T5)
            ):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_SD3):
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
        # make sure text_encoder_lr as list of three elements
        # if float, use the same value for all three
        if text_encoder_lr is None or (isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0):
            text_encoder_lr = [default_lr, default_lr, default_lr]
        elif isinstance(text_encoder_lr, float) or isinstance(text_encoder_lr, int):
            text_encoder_lr = [float(text_encoder_lr), float(text_encoder_lr), float(text_encoder_lr)]
        elif len(text_encoder_lr) == 1:
            text_encoder_lr = [text_encoder_lr[0], text_encoder_lr[0], text_encoder_lr[0]]
        elif len(text_encoder_lr) == 2:
            text_encoder_lr = [text_encoder_lr[0], text_encoder_lr[1], text_encoder_lr[1]]

        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            for lora in loras:
                for name, param in lora.named_parameters():
                    if loraplus_ratio is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
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
            te1_loras = [
                lora for lora in self.text_encoder_loras if lora.lora_name.startswith(self.LORA_PREFIX_TEXT_ENCODER_CLIP_L)
            ]
            te2_loras = [
                lora for lora in self.text_encoder_loras if lora.lora_name.startswith(self.LORA_PREFIX_TEXT_ENCODER_CLIP_G)
            ]
            te3_loras = [lora for lora in self.text_encoder_loras if lora.lora_name.startswith(self.LORA_PREFIX_TEXT_ENCODER_T5)]
            if len(te1_loras) > 0:
                logger.info(f"Text Encoder 1 (CLIP-L): {len(te1_loras)} modules, LR {text_encoder_lr[0]}")
                params, descriptions = assemble_params(te1_loras, text_encoder_lr[0], loraplus_lr_ratio)
                all_params.extend(params)
                lr_descriptions.extend(["textencoder 1 " + (" " + d if d else "") for d in descriptions])
            if len(te2_loras) > 0:
                logger.info(f"Text Encoder 2 (CLIP-G): {len(te2_loras)} modules, LR {text_encoder_lr[1]}")
                params, descriptions = assemble_params(te2_loras, text_encoder_lr[1], loraplus_lr_ratio)
                all_params.extend(params)
                lr_descriptions.extend(["textencoder 1 " + (" " + d if d else "") for d in descriptions])
            if len(te3_loras) > 0:
                logger.info(f"Text Encoder 3 (T5XXL): {len(te3_loras)} modules, LR {text_encoder_lr[2]}")
                params, descriptions = assemble_params(te3_loras, text_encoder_lr[2], loraplus_lr_ratio)
                all_params.extend(params)
                lr_descriptions.extend(["textencoder 3 " + (" " + d if d else "") for d in descriptions])

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
