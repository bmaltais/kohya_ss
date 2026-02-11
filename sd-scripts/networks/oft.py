# OFT network module

import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union
from diffusers import AutoencoderKL
import einops
from transformers import CLIPTextModel
import numpy as np
import torch
import torch.nn.functional as F
import re
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")


class OFTModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        oft_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
    ):
        """
        dim -> num blocks
        alpha -> constraint
        """
        super().__init__()
        self.oft_name = oft_name

        self.num_blocks = dim

        if "Linear" in org_module.__class__.__name__:
            out_dim = org_module.out_features
        elif "Conv" in org_module.__class__.__name__:
            out_dim = org_module.out_channels

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        
        # constraint in original paper is alpha * out_dim * out_dim, but we use alpha * out_dim for backward compatibility
        # original alpha is 1e-5, so we use 1e-2 or 1e-4 for alpha
        self.constraint = alpha * out_dim 
        
        self.register_buffer("alpha", torch.tensor(alpha))

        self.block_size = out_dim // self.num_blocks
        self.oft_blocks = torch.nn.Parameter(torch.zeros(self.num_blocks, self.block_size, self.block_size))
        self.I = torch.eye(self.block_size).unsqueeze(0).repeat(self.num_blocks, 1, 1)  # cpu

        self.out_dim = out_dim
        self.shape = org_module.weight.shape

        self.multiplier = multiplier
        self.org_module = [org_module]  # moduleにならないようにlistに入れる

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        block_Q = self.oft_blocks - self.oft_blocks.transpose(1, 2)
        norm_Q = torch.norm(block_Q.flatten())
        new_norm_Q = torch.clamp(norm_Q, max=self.constraint)
        block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))

        if self.I.device != block_Q.device:
            self.I = self.I.to(block_Q.device)
        I = self.I
        block_R = torch.matmul(I + block_Q, (I - block_Q).float().inverse())
        block_R_weighted = self.multiplier * (block_R - I) + I
        return block_R_weighted

    def forward(self, x, scale=None):
        if self.multiplier == 0.0:
            return self.org_forward(x)
        org_module = self.org_module[0]
        org_dtype = x.dtype

        R = self.get_weight().to(torch.float32)
        W = org_module.weight.to(torch.float32)

        if len(W.shape) == 4:  # Conv2d
            W_reshaped = einops.rearrange(W, "(k n) ... -> k n ...", k=self.num_blocks, n=self.block_size)
            RW = torch.einsum("k n m, k n ... -> k m ...", R, W_reshaped)
            RW = einops.rearrange(RW, "k m ... -> (k m) ...")
            result = F.conv2d(
                x, RW.to(org_dtype), org_module.bias, org_module.stride, org_module.padding, org_module.dilation, org_module.groups
            )
        else:  # Linear
            W_reshaped = einops.rearrange(W, "(k n) m -> k n m", k=self.num_blocks, n=self.block_size)
            RW = torch.einsum("k n m, k n p -> k m p", R, W_reshaped)
            RW = einops.rearrange(RW, "k m p -> (k m) p")
            result = F.linear(x, RW.to(org_dtype), org_module.bias)
        return result


class OFTInfModule(OFTModule):
    def __init__(
        self,
        oft_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(oft_name, org_module, multiplier, dim, alpha)
        self.enabled = True
        self.network: OFTNetwork = None

    def set_network(self, network):
        self.network = network

    def forward(self, x, scale=None):
        if not self.enabled:
            return self.org_forward(x)
        return super().forward(x, scale)

    def merge_to(self, multiplier=None):
        # get org weight
        org_sd = self.org_module[0].state_dict()
        org_weight = org_sd["weight"].to(torch.float32)

        R = self.get_weight(multiplier).to(torch.float32)

        weight = org_weight.reshape(self.num_blocks, self.block_size, -1)
        weight = torch.einsum("k n m, k n ... -> k m ...", R, weight)
        weight = weight.reshape(org_weight.shape)

        # convert back to original dtype
        weight = weight.to(org_sd["weight"].dtype)

        # set weight to org_module
        org_sd["weight"] = weight
        self.org_module[0].load_state_dict(org_sd)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: AutoencoderKL,
    text_encoder: Union[CLIPTextModel, List[CLIPTextModel]],
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:  # should be set
        logger.info(
            "network_alpha is not set, use default value 1e-3 / network_alphaが設定されていないのでデフォルト値 1e-3 を使用します"
        )
        network_alpha = 1e-3
    elif network_alpha >= 1:
        logger.warning(
            "network_alpha is too large (>=1, maybe default value is too large), please consider to set smaller value like 1e-3"
            " / network_alphaが大きすぎるようです(>=1, デフォルト値が大きすぎる可能性があります)。1e-3のような小さな値を推奨"
        )

    enable_all_linear = kwargs.get("enable_all_linear", None)
    enable_conv = kwargs.get("enable_conv", None)
    if enable_all_linear is not None:
        enable_all_linear = bool(enable_all_linear)
    if enable_conv is not None:
        enable_conv = bool(enable_conv)

    network = OFTNetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        dim=network_dim,
        alpha=network_alpha,
        enable_all_linear=enable_all_linear,
        enable_conv=enable_conv,
        varbose=True,
    )
    return network


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # check dim, alpha and if weights have for conv2d
    dim = None
    alpha = None
    has_conv2d = None
    all_linear = None
    for name, param in weights_sd.items():
        if name.endswith(".alpha"):
            if alpha is None:
                alpha = param.item()
        else:
            if dim is None:
                dim = param.size()[0]
            if has_conv2d is None and "in_layers_2" in name:
                has_conv2d = True
            if all_linear is None and "_ff_" in name:
                all_linear = True
        if dim is not None and alpha is not None and has_conv2d is not None and all_linear is not None:
            break
    if has_conv2d is None:
        has_conv2d = False
    if all_linear is None:
        all_linear = False

    module_class = OFTInfModule if for_inference else OFTModule
    network = OFTNetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        dim=dim,
        alpha=alpha,
        enable_all_linear=all_linear,
        enable_conv=has_conv2d,
        module_class=module_class,
    )
    return network, weights_sd


class OFTNetwork(torch.nn.Module):
    UNET_TARGET_REPLACE_MODULE_ATTN_ONLY = ["CrossAttention"]
    UNET_TARGET_REPLACE_MODULE_ALL_LINEAR = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    OFT_PREFIX_UNET = "oft_unet"  # これ変えないほうがいいかな

    def __init__(
        self,
        text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
        unet,
        multiplier: float = 1.0,
        dim: int = 4,
        alpha: float = 1,
        enable_all_linear: Optional[bool] = False,
        enable_conv: Optional[bool] = False,
        module_class: Type[object] = OFTModule,
        varbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.dim = dim
        self.alpha = alpha

        logger.info(
            f"create OFT network. num blocks: {self.dim}, constraint: {self.alpha}, multiplier: {self.multiplier}, enable_conv: {enable_conv}, enable_all_linear: {enable_all_linear}"
        )

        # create module instances
        def create_modules(
            root_module: torch.nn.Module,
            target_replace_modules: List[torch.nn.Module],
        ) -> List[OFTModule]:
            prefix = self.OFT_PREFIX_UNET
            ofts = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = "Linear" in child_module.__class__.__name__
                        is_conv2d = "Conv2d" in child_module.__class__.__name__
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d_1x1 or (is_conv2d and enable_conv):
                            oft_name = prefix + "." + name + "." + child_name
                            oft_name = oft_name.replace(".", "_")
                            # logger.info(oft_name)

                            oft = module_class(
                                oft_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                            )
                            ofts.append(oft)
            return ofts

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        if enable_all_linear:
            target_modules = OFTNetwork.UNET_TARGET_REPLACE_MODULE_ALL_LINEAR
        else:
            target_modules = OFTNetwork.UNET_TARGET_REPLACE_MODULE_ATTN_ONLY
        if enable_conv:
            target_modules += OFTNetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_ofts: List[OFTModule] = create_modules(unet, target_modules)
        logger.info(f"create OFT for U-Net: {len(self.unet_ofts)} modules.")

        # assertion
        names = set()
        for oft in self.unet_ofts:
            assert oft.oft_name not in names, f"duplicated oft name: {oft.oft_name}"
            names.add(oft.oft_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for oft in self.unet_ofts:
            oft.multiplier = self.multiplier

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        assert apply_unet, "apply_unet must be True"

        for oft in self.unet_ofts:
            oft.apply_to()
            self.add_module(oft.oft_name, oft)

    # マージできるかどうかを返す
    def is_mergeable(self):
        return True

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        logger.info("enable OFT for U-Net")

        for oft in self.unet_ofts:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(oft.oft_name):
                    sd_for_lora[key[len(oft.oft_name) + 1 :]] = weights_sd[key]
            oft.load_state_dict(sd_for_lora, False)
            oft.merge_to()

        logger.info(f"weights are merged")

    # 二つのText Encoderに別々の学習率を設定できるようにするといいかも
    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)
        all_params = []

        def enumerate_params(ofts):
            params = []
            for oft in ofts:
                params.extend(oft.parameters())

            # logger.info num of params
            num_params = 0
            for p in params:
                num_params += p.numel()
            logger.info(f"OFT params: {num_params}")
            return params

        param_data = {"params": enumerate_params(self.unet_ofts)}
        if unet_lr is not None:
            param_data["lr"] = unet_lr
        all_params.append(param_data)

        return all_params

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
        ofts: List[OFTInfModule] = self.unet_ofts
        for oft in ofts:
            org_module = oft.org_module[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        # 重みのリストアを行う
        ofts: List[OFTInfModule] = self.unet_ofts
        for oft in ofts:
            org_module = oft.org_module[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        # 事前計算を行う
        ofts: List[OFTInfModule] = self.unet_ofts
        for oft in ofts:
            org_module = oft.org_module[0]
            oft.merge_to()
            # sd = org_module.state_dict()
            # org_weight = sd["weight"]
            # lora_weight = oft.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            # sd["weight"] = org_weight + lora_weight
            # assert sd["weight"].shape == org_weight.shape
            # org_module.load_state_dict(sd)

            org_module._lora_restored = False
            oft.enabled = False
