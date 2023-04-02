# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from typing import List, Tuple, Union
import numpy as np
import torch
import re

from library import train_util

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_dim=4, alpha=1):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        # if limit_rank:
        #   self.lora_dim = min(lora_dim, in_dim, out_dim)
        #   if self.lora_dim != lora_dim:
        #     print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
        # else:
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.region = None
        self.region_mask = None

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def merge_to(self, sd, dtype, device):
        # get up/down weight
        up_weight = sd["lora_up.weight"].to(torch.float).to(device)
        down_weight = sd["lora_down.weight"].to(torch.float).to(device)

        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"].to(torch.float)

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
            # print(conved.size(), weight.size(), module.stride, module.padding)
            weight = weight + self.multiplier * conved * self.scale

        # set weight to org_module
        org_sd["weight"] = weight.to(dtype)
        self.org_module.load_state_dict(org_sd)

    def set_region(self, region):
        self.region = region
        self.region_mask = None

    def forward(self, x):
        if self.region is None:
            return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

        # regional LoRA   FIXME same as additional-network extension
        if x.size()[1] % 77 == 0:
            # print(f"LoRA for context: {self.lora_name}")
            self.region = None
            return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

        # calculate region mask first time
        if self.region_mask is None:
            if len(x.size()) == 4:
                h, w = x.size()[2:4]
            else:
                seq_len = x.size()[1]
                ratio = math.sqrt((self.region.size()[0] * self.region.size()[1]) / seq_len)
                h = int(self.region.size()[0] / ratio + 0.5)
                w = seq_len // h

            r = self.region.to(x.device)
            if r.dtype == torch.bfloat16:
                r = r.to(torch.float)
            r = r.unsqueeze(0).unsqueeze(1)
            # print(self.lora_name, self.region.size(), x.size(), r.size(), h, w)
            r = torch.nn.functional.interpolate(r, (h, w), mode="bilinear")
            r = r.to(x.dtype)

            if len(x.size()) == 3:
                r = torch.reshape(r, (1, x.size()[1], -1))

            self.region_mask = r

        return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale * self.region_mask


def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
    if network_dim is None:
        network_dim = 4  # default

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    """
    block_dims = kwargs.get("block_dims")
    block_alphas = None

    if block_dims is not None:
    block_dims = [int(d) for d in block_dims.split(',')]
    assert len(block_dims) == NUM_BLOCKS, f"Number of block dimensions is not same to {NUM_BLOCKS}"
    block_alphas = kwargs.get("block_alphas")
    if block_alphas is None:
        block_alphas = [1] * len(block_dims)
    else:
        block_alphas = [int(a) for a in block_alphas(',')]
    assert len(block_alphas) == NUM_BLOCKS, f"Number of block alphas is not same to {NUM_BLOCKS}"

    conv_block_dims = kwargs.get("conv_block_dims")
    conv_block_alphas = None

    if conv_block_dims is not None:
    conv_block_dims = [int(d) for d in conv_block_dims.split(',')]
    assert len(conv_block_dims) == NUM_BLOCKS, f"Number of block dimensions is not same to {NUM_BLOCKS}"
    conv_block_alphas = kwargs.get("conv_block_alphas")
    if conv_block_alphas is None:
        conv_block_alphas = [1] * len(conv_block_dims)
    else:
        conv_block_alphas = [int(a) for a in conv_block_alphas(',')]
    assert len(conv_block_alphas) == NUM_BLOCKS, f"Number of block alphas is not same to {NUM_BLOCKS}"
    """

    network = LoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
    )

    # if some parameters are not set, use zero
    up_lr_weight = kwargs.get("up_lr_weight", None)
    if up_lr_weight is not None:
        if "," in up_lr_weight:
            up_lr_weight = [(float(s) if s else 0.0) for s in up_lr_weight.split(",")]

    down_lr_weight = kwargs.get("down_lr_weight", None)
    if down_lr_weight is not None:
        if "," in down_lr_weight:
            down_lr_weight = [(float(s) if s else 0.0) for s in down_lr_weight.split(",")]

    mid_lr_weight = kwargs.get("mid_lr_weight", None)
    if mid_lr_weight is not None:
        mid_lr_weight = float(mid_lr_weight)

    network.set_block_lr_weight(up_lr_weight, mid_lr_weight, down_lr_weight, float(kwargs.get("block_lr_zero_threshold", 0.0)))

    return network


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping
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
            # print(lora_name, value.size(), dim)

    # support old LoRA without alpha
    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha = modules_dim[key]

    network = LoRANetwork(text_encoder, unet, multiplier=multiplier, modules_dim=modules_dim, modules_alpha=modules_alpha)
    network.weights_sd = weights_sd
    return network


class LoRANetwork(torch.nn.Module):
    NUM_OF_BLOCKS = 12  # フルモデル相当でのup,downの層の数

    # is it possible to apply conv_in and conv_out? -> yes, newer LoCon supports it (^^;)
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        text_encoder,
        unet,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        conv_lora_dim=None,
        conv_alpha=None,
        modules_dim=None,
        modules_alpha=None,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha

        if modules_dim is not None:
            print(f"create LoRA network from weights")
        else:
            print(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")

        self.apply_to_conv2d_3x3 = self.conv_lora_dim is not None
        if self.apply_to_conv2d_3x3:
            if self.conv_alpha is None:
                self.conv_alpha = self.alpha
            print(f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}")

        # create module instances
        def create_modules(prefix, root_module: torch.nn.Module, target_replace_modules) -> List[LoRAModule]:
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    # TODO get block index here
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)
                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            if modules_dim is not None:
                                if lora_name not in modules_dim:
                                    continue  # no LoRA module in this weights file
                                dim = modules_dim[lora_name]
                                alpha = modules_alpha[lora_name]
                            else:
                                if is_linear or is_conv2d_1x1:
                                    dim = self.lora_dim
                                    alpha = self.alpha
                                elif self.apply_to_conv2d_3x3:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha
                                else:
                                    continue

                            lora = LoRAModule(lora_name, child_module, self.multiplier, dim, alpha)
                            loras.append(lora)
            return loras

        self.text_encoder_loras = create_modules(
            LoRANetwork.LORA_PREFIX_TEXT_ENCODER, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
        )
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
        if modules_dim is not None or self.conv_lora_dim is not None:
            target_modules += LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_loras = create_modules(LoRANetwork.LORA_PREFIX_UNET, unet, target_modules)
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        self.weights_sd = None

        self.up_lr_weight: List[float] = None
        self.down_lr_weight: List[float] = None
        self.mid_lr_weight: float = None
        self.block_lr = False

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            self.weights_sd = load_file(file)
        else:
            self.weights_sd = torch.load(file, map_location="cpu")

    def apply_to(self, text_encoder, unet, apply_text_encoder=None, apply_unet=None):
        if self.weights_sd:
            weights_has_text_encoder = weights_has_unet = False
            for key in self.weights_sd.keys():
                if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                    weights_has_text_encoder = True
                elif key.startswith(LoRANetwork.LORA_PREFIX_UNET):
                    weights_has_unet = True

            if apply_text_encoder is None:
                apply_text_encoder = weights_has_text_encoder
            else:
                assert (
                    apply_text_encoder == weights_has_text_encoder
                ), f"text encoder weights: {weights_has_text_encoder} but text encoder flag: {apply_text_encoder} / 重みとText Encoderのフラグが矛盾しています"

            if apply_unet is None:
                apply_unet = weights_has_unet
            else:
                assert (
                    apply_unet == weights_has_unet
                ), f"u-net weights: {weights_has_unet} but u-net flag: {apply_unet} / 重みとU-Netのフラグが矛盾しています"
        else:
            assert apply_text_encoder is not None and apply_unet is not None, f"internal error: flag not set"

        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        skipped = []
        for lora in self.text_encoder_loras + self.unet_loras:
            if self.block_lr and self.get_lr_weight(lora) == 0:  # no LR weight
                skipped.append(lora.lora_name)
                continue
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        if len(skipped) > 0:
            print(
                f"because block_lr_weight is 0, {len(skipped)} LoRA modules are skipped / block_lr_weightが0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                print(f"\t{name}")

        if self.weights_sd:
            # if some weights are not in state dict, it is ok because initial LoRA does nothing (lora_up is initialized by zeros)
            info = self.load_state_dict(self.weights_sd, False)
            print(f"weights are loaded: {info}")

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoder, unet, dtype, device):
        assert self.weights_sd is not None, "weights are not loaded"

        apply_text_encoder = apply_unet = False
        for key in self.weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_UNET):
                apply_unet = True

        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in self.weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = self.weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        print(f"weights are merged")

    # 層別学習率用に層ごとの学習率に対する倍率を定義する
    def set_block_lr_weight(
        self,
        up_lr_weight: Union[List[float], str] = None,
        mid_lr_weight: float = None,
        down_lr_weight: Union[List[float], str] = None,
        zero_threshold: float = 0.0,
    ):
        # パラメータ未指定時は何もせず、今までと同じ動作とする
        if up_lr_weight is None and mid_lr_weight is None and down_lr_weight is None:
            return

        max_len = LoRANetwork.NUM_OF_BLOCKS  # フルモデル相当でのup,downの層の数

        def get_list(name_with_suffix) -> List[float]:
            import math

            tokens = name_with_suffix.split("+")
            name = tokens[0]
            base_lr = float(tokens[1]) if len(tokens) > 1 else 0.0

            if name == "cosine":
                return [math.sin(math.pi * (i / (max_len - 1)) / 2) + base_lr for i in reversed(range(max_len))]
            elif name == "sine":
                return [math.sin(math.pi * (i / (max_len - 1)) / 2) + base_lr for i in range(max_len)]
            elif name == "linear":
                return [i / (max_len - 1) + base_lr for i in range(max_len)]
            elif name == "reverse_linear":
                return [i / (max_len - 1) + base_lr for i in reversed(range(max_len))]
            elif name == "zeros":
                return [0.0 + base_lr] * max_len
            else:
                print(
                    "Unknown lr_weight argument %s is used. Valid arguments:  / 不明なlr_weightの引数 %s が使われました。有効な引数:\n\tcosine, sine, linear, reverse_linear, zeros"
                    % (name)
                )
                return None

        if type(down_lr_weight) == str:
            down_lr_weight = get_list(down_lr_weight)
        if type(up_lr_weight) == str:
            up_lr_weight = get_list(up_lr_weight)

        if (up_lr_weight != None and len(up_lr_weight) > max_len) or (down_lr_weight != None and len(down_lr_weight) > max_len):
            print("down_weight or up_weight is too long. Parameters after %d-th are ignored." % max_len)
            print("down_weightもしくはup_weightが長すぎます。%d個目以降のパラメータは無視されます。" % max_len)
            up_lr_weight = up_lr_weight[:max_len]
            down_lr_weight = down_lr_weight[:max_len]

        if (up_lr_weight != None and len(up_lr_weight) < max_len) or (down_lr_weight != None and len(down_lr_weight) < max_len):
            print("down_weight or up_weight is too short. Parameters after %d-th are filled with 1." % max_len)
            print("down_weightもしくはup_weightが短すぎます。%d個目までの不足したパラメータは1で補われます。" % max_len)

            if down_lr_weight != None and len(down_lr_weight) < max_len:
                down_lr_weight = down_lr_weight + [1.0] * (max_len - len(down_lr_weight))
            if up_lr_weight != None and len(up_lr_weight) < max_len:
                up_lr_weight = up_lr_weight + [1.0] * (max_len - len(up_lr_weight))

        if (up_lr_weight != None) or (mid_lr_weight != None) or (down_lr_weight != None):
            print("apply block learning rate / 階層別学習率を適用します。")
            self.block_lr = True

            if down_lr_weight != None:
                self.down_lr_weight = [w if w > zero_threshold else 0 for w in down_lr_weight]
                print("down_lr_weight (shallower -> deeper, 浅い層->深い層):", self.down_lr_weight)
            else:
                print("down_lr_weight: all 1.0, すべて1.0")

            if mid_lr_weight != None:
                self.mid_lr_weight = mid_lr_weight if mid_lr_weight > zero_threshold else 0
                print("mid_lr_weight:", self.mid_lr_weight)
            else:
                print("mid_lr_weight: 1.0")

            if up_lr_weight != None:
                self.up_lr_weight = [w if w > zero_threshold else 0 for w in up_lr_weight]
                print("up_lr_weight (deeper -> shallower, 深い層->浅い層):", self.up_lr_weight)
            else:
                print("up_lr_weight: all 1.0, すべて1.0")

        return

    def get_block_index(self, lora: LoRAModule) -> int:
        block_idx = -1  # invalid lora name

        m = RE_UPDOWN.search(lora.lora_name)
        if m:
            g = m.groups()
            i = int(g[1])
            j = int(g[3])
            if g[2] == "resnets":
                idx = 3 * i + j
            elif g[2] == "attentions":
                idx = 3 * i + j
            elif g[2] == "upsamplers" or g[2] == "downsamplers":
                idx = 3 * i + 2

            if g[0] == "down":
                block_idx = 1 + idx  # 0に該当するLoRAは存在しない
            elif g[0] == "up":
                block_idx = LoRANetwork.NUM_OF_BLOCKS + 1 + idx

        elif "mid_block_" in lora.lora_name:
            block_idx = LoRANetwork.NUM_OF_BLOCKS  # idx=12

        return block_idx

    def get_lr_weight(self, lora: LoRAModule) -> float:
        lr_weight = 1.0
        block_idx = self.get_block_index(lora)
        if block_idx < 0:
            return lr_weight

        if block_idx < LoRANetwork.NUM_OF_BLOCKS:
            if self.down_lr_weight != None:
                lr_weight = self.down_lr_weight[block_idx]
        elif block_idx == LoRANetwork.NUM_OF_BLOCKS:
            if self.mid_lr_weight != None:
                lr_weight = self.mid_lr_weight
        elif block_idx > LoRANetwork.NUM_OF_BLOCKS:
            if self.up_lr_weight != None:
                lr_weight = self.up_lr_weight[block_idx - LoRANetwork.NUM_OF_BLOCKS - 1]

        return lr_weight

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)
        all_params = []

        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        if self.text_encoder_loras:
            param_data = {"params": enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            if self.block_lr:
                # 学習率のグラフをblockごとにしたいので、blockごとにloraを分類
                block_idx_to_lora = {}
                for lora in self.unet_loras:
                    idx = self.get_block_index(lora)
                    if idx not in block_idx_to_lora:
                        block_idx_to_lora[idx] = []
                    block_idx_to_lora[idx].append(lora)

                # blockごとにパラメータを設定する
                for idx, block_loras in block_idx_to_lora.items():
                    param_data = {"params": enumerate_params(block_loras)}

                    if unet_lr is not None:
                        param_data["lr"] = unet_lr * self.get_lr_weight(block_loras[0])
                    elif default_lr is not None:
                        param_data["lr"] = default_lr * self.get_lr_weight(block_loras[0])
                    if ("lr" in param_data) and (param_data["lr"] == 0):
                        continue
                    all_params.append(param_data)

            else:
                param_data = {"params": enumerate_params(self.unet_loras)}
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

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    @staticmethod
    def set_regions(networks, image):
        image = image.astype(np.float32) / 255.0
        for i, network in enumerate(networks[:3]):
            # NOTE: consider averaging overwrapping area
            region = image[:, :, i]
            if region.max() == 0:
                continue
            region = torch.tensor(region)
            network.set_region(region)

    def set_region(self, region):
        for lora in self.unet_loras:
            lora.set_region(region)
