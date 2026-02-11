# temporary minimum implementation of LoRA
# FLUX doesn't have Conv2d, so we ignore it
# TODO commonize with the original implementation

# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import os
from typing import Dict, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
import re

from networks import lora_flux
from library.hunyuan_image_vae import HunyuanVAE2D

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


NUM_DOUBLE_BLOCKS = 20
NUM_SINGLE_BLOCKS = 40


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: HunyuanVAE2D,
    text_encoders: List[nn.Module],
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

    ggpo_beta = kwargs.get("ggpo_beta", None)
    ggpo_sigma = kwargs.get("ggpo_sigma", None)

    if ggpo_beta is not None:
        ggpo_beta = float(ggpo_beta)

    if ggpo_sigma is not None:
        ggpo_sigma = float(ggpo_sigma)

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

    # Too many arguments ( ^ω^)･･･
    network = HunyuanImageLoRANetwork(
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
        split_qkv=split_qkv,
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

    split_qkv = False  # split_qkv is not needed to care, because state_dict is qkv combined

    module_class = lora_flux.LoRAInfModule if for_inference else lora_flux.LoRAModule

    network = HunyuanImageLoRANetwork(
        text_encoders,
        flux,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        split_qkv=split_qkv,
    )
    return network, weights_sd


class HunyuanImageLoRANetwork(lora_flux.LoRANetwork):
    TARGET_REPLACE_MODULE_DOUBLE = ["MMDoubleStreamBlock"]
    TARGET_REPLACE_MODULE_SINGLE = ["MMSingleStreamBlock"]
    LORA_PREFIX_HUNYUAN_IMAGE_DIT = "lora_unet"  # make ComfyUI compatible

    @classmethod
    def get_qkv_mlp_split_dims(cls) -> List[int]:
        return [3584] * 3 + [14336]

    def __init__(
        self,
        text_encoders: list[nn.Module],
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[object] = lora_flux.LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        split_qkv: bool = False,
        reg_dims: Optional[Dict[str, int]] = None,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        reg_lrs: Optional[Dict[str, float]] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        nn.Module.__init__(self)
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.split_qkv = split_qkv
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

        # create module instances
        def create_modules(
            is_dit: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            filter: Optional[str] = None,
            default_dim: Optional[int] = None,
        ) -> List[lora_flux.LoRAModule]:
            assert is_dit, "only DIT is supported now"

            prefix = self.LORA_PREFIX_HUNYUAN_IMAGE_DIT

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
                            if is_dit and split_qkv:
                                if "double" in lora_name and "qkv" in lora_name:
                                    split_dims = self.get_qkv_mlp_split_dims()[:3]  # qkv only
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

        # create LoRA for U-Net
        target_replace_modules = (
            HunyuanImageLoRANetwork.TARGET_REPLACE_MODULE_DOUBLE + HunyuanImageLoRANetwork.TARGET_REPLACE_MODULE_SINGLE
        )

        self.unet_loras: List[Union[lora_flux.LoRAModule, lora_flux.LoRAInfModule]]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_replace_modules)
        self.text_encoder_loras = []

        logger.info(f"create LoRA for HunyuanImage-2.1: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:50} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_un
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
