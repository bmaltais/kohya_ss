# temporary minimum implementation of LoRA
# Lumina 2 does not have Conv2d, so ignore
# TODO commonize with the original implementation

# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers import CLIPTextModel
import torch
from torch import Tensor, nn
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float =1.0,
        lora_dim: int = 4,
        alpha: Optional[float | int | Tensor] = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        split_dims: Optional[List[int]] = None,
    ):
        """
        if alpha == 0 or None, alpha is rank (no scaling).

        split_dims is used to mimic the split qkv of lumina as same as Diffusers
        """
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        assert isinstance(in_dim, int)
        assert isinstance(out_dim, int)

        self.lora_dim = lora_dim
        self.split_dims = split_dims

        if split_dims is None:
            if org_module.__class__.__name__ == "Conv2d":
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                self.lora_down = nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
                self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
            else:
                self.lora_down = nn.Linear(in_dim, self.lora_dim, bias=False)
                self.lora_up = nn.Linear(self.lora_dim, out_dim, bias=False)

            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)
        else:
            # conv2d not supported
            assert sum(split_dims) == out_dim, "sum of split_dims must be equal to out_dim"
            assert org_module.__class__.__name__ == "Linear", "split_dims is only supported for Linear"
            # print(f"split_dims: {split_dims}")
            self.lora_down = nn.ModuleList(
                [nn.Linear(in_dim, self.lora_dim, bias=False) for _ in range(len(split_dims))]
            )
            self.lora_up = nn.ModuleList([torch.nn.Linear(self.lora_dim, split_dim, bias=False) for split_dim in split_dims])

            for lora_down in self.lora_down:
                nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
            for lora_up in self.lora_up:
                nn.init.zeros_(lora_up.weight)

        if isinstance(alpha, Tensor):
            alpha = alpha.detach().cpu().float().item()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

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
                    if len(lxs[i].size()) == 3:
                        masks[i] = masks[i].unsqueeze(1)
                    elif len(lxs[i].size()) == 4:
                        masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                    lxs[i] = lxs[i] * masks[i]

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
            else:
                scale = self.scale

            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]

            return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * scale


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
    lumina,
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

    # attn dim, mlp dim for JointTransformerBlock
    attn_dim = kwargs.get("attn_dim", None)  # attention dimension
    mlp_dim = kwargs.get("mlp_dim", None)   # MLP dimension
    mod_dim = kwargs.get("mod_dim", None)   # modulation dimension
    refiner_dim = kwargs.get("refiner_dim", None)  # refiner blocks dimension

    if attn_dim is not None:
        attn_dim = int(attn_dim)
    if mlp_dim is not None:
        mlp_dim = int(mlp_dim)
    if mod_dim is not None:
        mod_dim = int(mod_dim)
    if refiner_dim is not None:
        refiner_dim = int(refiner_dim)

    type_dims = [attn_dim, mlp_dim, mod_dim, refiner_dim]
    if all([d is None for d in type_dims]):
        type_dims = None

    # embedder_dims for embedders
    embedder_dims = kwargs.get("embedder_dims", None)
    if embedder_dims is not None:
        embedder_dims = embedder_dims.strip()
        if embedder_dims.startswith("[") and embedder_dims.endswith("]"):
            embedder_dims = embedder_dims[1:-1]
        embedder_dims = [int(d) for d in embedder_dims.split(",")]
        assert len(embedder_dims) == 3, f"invalid embedder_dims: {embedder_dims}, must be 3 dimensions (x_embedder, t_embedder, cap_embedder)"

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # single or double blocks
    train_blocks = kwargs.get("train_blocks", None)  # None (default), "all" (same as None), "transformer", "refiners", "noise_refiner", "context_refiner"
    if train_blocks is not None:
        assert train_blocks in ["all", "transformer", "refiners", "noise_refiner", "context_refiner"], f"invalid train_blocks: {train_blocks}"

    # split qkv
    split_qkv = kwargs.get("split_qkv", False)
    if split_qkv is not None:
        split_qkv = True if split_qkv == "True" else False

    # verbose
    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    # すごく引数が多いな ( ^ω^)･･･
    network = LoRANetwork(
        text_encoders,
        lumina,
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
        type_dims=type_dims,
        embedder_dims=embedder_dims,
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
def create_network_from_weights(multiplier, file, ae, text_encoders, lumina, weights_sd=None, for_inference=False, **kwargs):
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

    # # split qkv
    # double_qkv_rank = None
    # single_qkv_rank = None
    # rank = None
    # for lora_name, dim in modules_dim.items():
    #     if "double" in lora_name and "qkv" in lora_name:
    #         double_qkv_rank = dim
    #     elif "single" in lora_name and "linear1" in lora_name:
    #         single_qkv_rank = dim
    #     elif rank is None:
    #         rank = dim
    #     if double_qkv_rank is not None and single_qkv_rank is not None and rank is not None:
    #         break
    # split_qkv = (double_qkv_rank is not None and double_qkv_rank != rank) or (
    #     single_qkv_rank is not None and single_qkv_rank != rank
    # )
    split_qkv = False  # split_qkv is not needed to care, because state_dict is qkv combined

    module_class = LoRAInfModule if for_inference else LoRAModule

    network = LoRANetwork(
        text_encoders,
        lumina,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        split_qkv=split_qkv,
    )
    return network, weights_sd


class LoRANetwork(torch.nn.Module):
    LUMINA_TARGET_REPLACE_MODULE = ["JointTransformerBlock", "FinalLayer"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["Gemma2Attention", "Gemma2FlashAttention2", "Gemma2SdpaAttention", "Gemma2MLP"]
    LORA_PREFIX_LUMINA = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"  # Simplified prefix since we only have one text encoder

    def __init__(
        self,
        text_encoders,  # Now this will be a single Gemma2 model
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[LoRAModule] = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        train_blocks: Optional[str] = None,
        split_qkv: bool = False,
        type_dims: Optional[List[int]] = None,
        embedder_dims: Optional[List[int]] = None,
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
        self.train_blocks = train_blocks if train_blocks is not None else "all"
        self.split_qkv = split_qkv

        self.type_dims = type_dims
        self.embedder_dims = embedder_dims

        self.train_block_indices = train_block_indices

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info(f"create LoRA network from weights")
            self.embedder_dims = [0] * 5  # create embedder_dims
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
        if self.split_qkv:
            logger.info(f"split qkv for LoRA")
        if self.train_blocks is not None:
            logger.info(f"train {self.train_blocks} blocks only")

        # create module instances
        def create_modules(
            is_lumina: bool,
            root_module: torch.nn.Module,
            target_replace_modules: Optional[List[str]],
            filter: Optional[str] = None,
            default_dim: Optional[int] = None,
        ) -> List[LoRAModule]:
            prefix = self.LORA_PREFIX_LUMINA if is_lumina else self.LORA_PREFIX_TEXT_ENCODER

            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if target_replace_modules is None or module.__class__.__name__ in target_replace_modules:
                    if target_replace_modules is None:  # for handling embedders
                        module = root_module

                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"

                        lora_name = prefix + "." + (name + "." if name else "") + child_name
                        lora_name = lora_name.replace(".", "_")

                        # Only Linear is supported
                        if not is_linear:
                            skipped.append(lora_name)
                            continue

                        if filter is not None and filter not in lora_name:
                            continue

                        dim = default_dim if default_dim is not None else self.lora_dim
                        alpha = self.alpha

                        # Set dim/alpha to modules dim/alpha
                        if modules_dim is not None and modules_alpha is not None:
                            # network from weights
                            if lora_name in modules_dim:
                                dim = modules_dim[lora_name]
                                alpha = modules_alpha[lora_name]
                            else:
                                dim = 0 # skip if not found

                        else:
                            # Set dims to type_dims
                            if is_lumina and type_dims is not None:
                                identifier = [
                                    ("attention",),  # attention layers
                                    ("mlp",),       # MLP layers
                                    ("modulation",), # modulation layers
                                    ("refiner",),   # refiner blocks
                                ]
                                for i, d in enumerate(type_dims):
                                    if d is not None and all([id in lora_name for id in identifier[i]]):
                                        dim = d  # may be 0 for skip
                                        break

                        # Drop blocks if we are only training some blocks
                        if (
                            is_lumina
                            and dim
                            and (
                                self.train_block_indices is not None
                            )
                            and ("layer" in lora_name)
                        ):
                            # "lora_unet_layers_0_..." or "lora_unet_cap_refiner_0_..." or or "lora_unet_noise_refiner_0_..."
                            block_index = int(lora_name.split("_")[3])  # bit dirty
                            if (
                                "layer" in lora_name
                                and self.train_block_indices is not None
                                and not self.train_block_indices[block_index]
                            ):
                                dim = 0


                        if dim is None or dim == 0:
                            # skipした情報を出力
                            skipped.append(lora_name)
                            continue

                        lora = module_class(
                            lora_name,
                            child_module,
                            self.multiplier,
                            dim,
                            alpha,
                            dropout=dropout,
                            rank_dropout=rank_dropout,
                            module_dropout=module_dropout,
                        )
                        loras.append(lora)

                if target_replace_modules is None:
                    break  # all modules are searched
            return loras, skipped

        # create LoRA for text encoder (Gemma2)
        self.text_encoder_loras: List[Union[LoRAModule, LoRAInfModule]] = []
        skipped_te = []

        logger.info(f"create LoRA for Gemma2 Text Encoder:")
        text_encoder_loras, skipped = create_modules(False, text_encoders[0], LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
        logger.info(f"create LoRA for Gemma2 Text Encoder: {len(text_encoder_loras)} modules.")
        self.text_encoder_loras.extend(text_encoder_loras)
        skipped_te += skipped

        # create LoRA for U-Net
        if self.train_blocks == "all":
            target_replace_modules = LoRANetwork.LUMINA_TARGET_REPLACE_MODULE
        # TODO: limit different blocks
        elif self.train_blocks == "transformer":
            target_replace_modules = LoRANetwork.LUMINA_TARGET_REPLACE_MODULE
        elif self.train_blocks == "refiners":
            target_replace_modules = LoRANetwork.LUMINA_TARGET_REPLACE_MODULE
        elif self.train_blocks == "noise_refiner":
            target_replace_modules = LoRANetwork.LUMINA_TARGET_REPLACE_MODULE
        elif self.train_blocks == "cap_refiner":
            target_replace_modules = LoRANetwork.LUMINA_TARGET_REPLACE_MODULE

        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]]
        self.unet_loras, skipped_un = create_modules(True, unet, target_replace_modules)

        # Handle embedders
        if self.embedder_dims:
            for filter, embedder_dim in zip(["x_embedder", "t_embedder", "cap_embedder"], self.embedder_dims):
                loras, _ = create_modules(True, unet, None, filter=filter, default_dim=embedder_dim)
                self.unet_loras.extend(loras)

        logger.info(f"create LoRA for Lumina blocks: {len(self.unet_loras)} modules.")
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

        # # split qkv
        # for key in list(state_dict.keys()):
        #     if "double" in key and "qkv" in key:
        #         split_dims = [3072] * 3
        #     elif "single" in key and "linear1" in key:
        #         split_dims = [3072] * 3 + [12288]
        #     else:
        #         continue

        #     weight = state_dict[key]
        #     lora_name = key.split(".")[0]

        #     if key not in state_dict:
        #         continue  # already merged

        #     # (rank, in_dim) * 3
        #     down_weights = [state_dict.pop(f"{lora_name}.lora_down.{i}.weight") for i in range(len(split_dims))]
        #     # (split dim, rank) * 3
        #     up_weights = [state_dict.pop(f"{lora_name}.lora_up.{i}.weight") for i in range(len(split_dims))]

        #     alpha = state_dict.pop(f"{lora_name}.alpha")

        #     # merge down weight
        #     down_weight = torch.cat(down_weights, dim=0)  # (rank, split_dim) * 3 -> (rank*3, sum of split_dim)

        #     # merge up weight (sum of split_dim, rank*3)
        #     rank = up_weights[0].size(1)
        #     up_weight = torch.zeros((sum(split_dims), down_weight.size(0)), device=down_weight.device, dtype=down_weight.dtype)
        #     i = 0
        #     for j in range(len(split_dims)):
        #         up_weight[i : i + split_dims[j], j * rank : (j + 1) * rank] = up_weights[j]
        #         i += split_dims[j]

        #     state_dict[f"{lora_name}.lora_down.weight"] = down_weight
        #     state_dict[f"{lora_name}.lora_up.weight"] = up_weight
        #     state_dict[f"{lora_name}.alpha"] = alpha

        #     # print(
        #     #     f"merged {lora_name}: {lora_name}, {[w.shape for w in down_weights]}, {[w.shape for w in up_weights]} to {down_weight.shape}, {up_weight.shape}"
        #     # )
        #     print(f"new key: {lora_name}.lora_down.weight, {lora_name}.lora_up.weight, {lora_name}.alpha")

        return super().load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if not self.split_qkv:
            return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # merge qkv
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if "double" in key and "qkv" in key:
                split_dims = [3072] * 3
            elif "single" in key and "linear1" in key:
                split_dims = [3072] * 3 + [12288]
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
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_LUMINA):
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
            te_loras = [lora for lora in self.text_encoder_loras]
            if len(te_loras) > 0:
                logger.info(f"Text Encoder: {len(te_loras)} modules, LR {text_encoder_lr[0]}")
                params, descriptions = assemble_params(te_loras, text_encoder_lr[0], loraplus_lr_ratio)
                all_params.extend(params)
                lr_descriptions.extend(["textencoder " + (" " + d if d else "") for d in descriptions])

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
