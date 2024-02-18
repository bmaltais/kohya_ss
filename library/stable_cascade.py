# コードは Stable Cascade からコピーし、一部修正しています。元ライセンスは MIT です。
# The code is copied from Stable Cascade and modified. The original license is MIT.
# https://github.com/Stability-AI/StableCascade

import math
from types import SimpleNamespace
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torchvision


# region VectorQuantize

# from torchtools https://github.com/pabloppp/pytorch-tools
# 依存ライブラリを増やしたくないのでここにコピペ


class vector_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codebook):
        with torch.no_grad():
            codebook_sqr = torch.sum(codebook**2, dim=1)
            x_sqr = torch.sum(x**2, dim=1, keepdim=True)

            dist = torch.addmm(codebook_sqr + x_sqr, x, codebook.t(), alpha=-2.0, beta=1.0)
            _, indices = dist.min(dim=1)

            ctx.save_for_backward(indices, codebook)
            ctx.mark_non_differentiable(indices)

            nn = torch.index_select(codebook, 0, indices)
            return nn, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors

            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output)

        return (grad_inputs, grad_codebook)


class VectorQuantize(nn.Module):
    def __init__(self, embedding_size, k, ema_decay=0.99, ema_loss=False):
        """
        Takes an input of variable size (as long as the last dimension matches the embedding size).
        Returns one tensor containing the nearest neigbour embeddings to each of the inputs,
        with the same size as the input, vq and commitment components for the loss as a touple
        in the second output and the indices of the quantized vectors in the third:
        quantized, (vq_loss, commit_loss), indices
        """
        super(VectorQuantize, self).__init__()

        self.codebook = nn.Embedding(k, embedding_size)
        self.codebook.weight.data.uniform_(-1.0 / k, 1.0 / k)
        self.vq = vector_quantize.apply

        self.ema_decay = ema_decay
        self.ema_loss = ema_loss
        if ema_loss:
            self.register_buffer("ema_element_count", torch.ones(k))
            self.register_buffer("ema_weight_sum", torch.zeros_like(self.codebook.weight))

    def _laplace_smoothing(self, x, epsilon):
        n = torch.sum(x)
        return (x + epsilon) / (n + x.size(0) * epsilon) * n

    def _updateEMA(self, z_e_x, indices):
        mask = nn.functional.one_hot(indices, self.ema_element_count.size(0)).float()
        elem_count = mask.sum(dim=0)
        weight_sum = torch.mm(mask.t(), z_e_x)

        self.ema_element_count = (self.ema_decay * self.ema_element_count) + ((1 - self.ema_decay) * elem_count)
        self.ema_element_count = self._laplace_smoothing(self.ema_element_count, 1e-5)
        self.ema_weight_sum = (self.ema_decay * self.ema_weight_sum) + ((1 - self.ema_decay) * weight_sum)

        self.codebook.weight.data = self.ema_weight_sum / self.ema_element_count.unsqueeze(-1)

    def idx2vq(self, idx, dim=-1):
        q_idx = self.codebook(idx)
        if dim != -1:
            q_idx = q_idx.movedim(-1, dim)
        return q_idx

    def forward(self, x, get_losses=True, dim=-1):
        if dim != -1:
            x = x.movedim(dim, -1)
        z_e_x = x.contiguous().view(-1, x.size(-1)) if len(x.shape) > 2 else x
        z_q_x, indices = self.vq(z_e_x, self.codebook.weight.detach())
        vq_loss, commit_loss = None, None
        if self.ema_loss and self.training:
            self._updateEMA(z_e_x.detach(), indices.detach())
        # pick the graded embeddings after updating the codebook in order to have a more accurate commitment loss
        z_q_x_grd = torch.index_select(self.codebook.weight, dim=0, index=indices)
        if get_losses:
            vq_loss = (z_q_x_grd - z_e_x.detach()).pow(2).mean()
            commit_loss = (z_e_x - z_q_x_grd.detach()).pow(2).mean()

        z_q_x = z_q_x.view(x.shape)
        if dim != -1:
            z_q_x = z_q_x.movedim(-1, dim)
        return z_q_x, (vq_loss, commit_loss), indices.view(x.shape[:-1])


# endregion


class EfficientNetEncoder(nn.Module):
    def __init__(self, c_latent=16):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(weights="DEFAULT").features.eval()
        self.mapper = nn.Sequential(
            nn.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent, affine=False),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x):
        return self.mapper(self.backbone(x))

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, x):
        """
        VAE と同じように使えるようにするためのメソッド。正しくはちゃんと呼び出し側で分けるべきだが、暫定的な対応。
        The method to make it usable like VAE. It should be separated properly, but it is a temporary response.
        """
        # latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")
        x = self(x)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: x))


# なんかわりと乱暴な実装(;'∀')
# 一から学習することもないだろうから、無効化しておく

# class Linear(torch.nn.Linear):
#     def reset_parameters(self):
#         return None

# class Conv2d(torch.nn.Conv2d):
#     def reset_parameters(self):
#         return None

from torch.nn import Conv2d
from torch.nn import Linear


class Attention2D(nn.Module):
    def __init__(self, c, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True)

    def forward(self, x, kv, self_attn=False):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4
        if self_attn:
            kv = torch.cat([x, kv], dim=1)
        x = self.attn(x, kv, kv, need_weights=False)[0]
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GlobalResponseNorm(nn.Module):
    "from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105"

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ResBlock(nn.Module):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):  # , num_heads=4, expansion=2):
        super().__init__()
        self.depthwise = Conv2d(c, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        #         self.depthwise = SAMBlock(c, num_heads, expansion)
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            Linear(c + c_skip, c * 4), nn.GELU(), GlobalResponseNorm(c * 4), nn.Dropout(dropout), Linear(c * 4, c)
        )

        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, value):
        self.gradient_checkpointing = value

    def forward_body(self, x, x_skip=None):
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + x_res

    def forward(self, x, x_skip=None):
        if self.training and self.gradient_checkpointing:
            # logger.info("ResnetBlock2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.forward_body), x, x_skip)
        else:
            x = self.forward_body(x, x_skip)

        return x


class AttnBlock(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention2D(c, nhead, dropout)
        self.kv_mapper = nn.Sequential(nn.SiLU(), Linear(c_cond, c))

        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, value):
        self.gradient_checkpointing = value

    def forward_body(self, x, kv):
        kv = self.kv_mapper(kv)
        x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn)
        return x

    def forward(self, x, kv):
        if self.training and self.gradient_checkpointing:
            # logger.info("AttnBlock: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.forward_body), x, kv)
        else:
            x = self.forward_body(x, kv)

        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, c, dropout=0.0):
        super().__init__()
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            Linear(c, c * 4), nn.GELU(), GlobalResponseNorm(c * 4), nn.Dropout(dropout), Linear(c * 4, c)
        )

        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, value):
        self.gradient_checkpointing = value

    def forward_body(self, x):
        x = x + self.channelwise(self.norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            # logger.info("FeedForwardBlock: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.forward_body), x)
        else:
            x = self.forward_body(x)

        return x


class TimestepBlock(nn.Module):
    def __init__(self, c, c_timestep, conds=["sca"]):
        super().__init__()
        self.mapper = Linear(c_timestep, c * 2)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", Linear(c_timestep, c * 2))

    def forward(self, x, t):
        t = t.chunk(len(self.conds) + 1, dim=1)
        a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        for i, c in enumerate(self.conds):
            ac, bc = getattr(self, f"mapper_{c}")(t[i + 1])[:, :, None, None].chunk(2, dim=1)
            a, b = a + ac, b + bc
        return x * (1 + a) + b


class UpDownBlock2d(nn.Module):
    def __init__(self, c_in, c_out, mode, enabled=True):
        super().__init__()
        assert mode in ["up", "down"]
        interpolation = (
            nn.Upsample(scale_factor=2 if mode == "up" else 0.5, mode="bilinear", align_corners=True) if enabled else nn.Identity()
        )
        mapping = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.blocks = nn.ModuleList([interpolation, mapping] if mode == "up" else [mapping, interpolation])

        self.mode = mode

        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, value):
        self.gradient_checkpointing = value

    def forward_body(self, x):
        org_dtype = x.dtype
        for i, block in enumerate(self.blocks):
            # 公式の実装では、常に float で計算しているが、すこしでもメモリを節約するために bfloat16 + Upsample のみ float に変換する
            # In the official implementation, it always calculates in float, but for the sake of saving memory, it converts to float only for bfloat16 + Upsample
            if x.dtype == torch.bfloat16 and (self.mode == "up" and i == 0 or self.mode != "up" and i == 1):
                x = x.float()
            x = block(x)
            x = x.to(org_dtype)
        return x

    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            # logger.info("UpDownBlock2d: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.forward_body), x)
        else:
            x = self.forward_body(x)

        return x


class StageAResBlock(nn.Module):
    def __init__(self, c, c_hidden):
        super().__init__()
        # depthwise/attention
        self.norm1 = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.depthwise = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(c, c, kernel_size=3, groups=c))

        # channelwise
        self.norm2 = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c),
        )

        self.gammas = nn.Parameter(torch.zeros(6), requires_grad=True)

        # Init weights
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def _norm(self, x, norm):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x):
        mods = self.gammas

        x_temp = self._norm(x, self.norm1) * (1 + mods[0]) + mods[1]
        x = x + self.depthwise(x_temp) * mods[2]

        x_temp = self._norm(x, self.norm2) * (1 + mods[3]) + mods[4]
        x = x + self.channelwise(x_temp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * mods[5]

        return x


class StageA(nn.Module):
    def __init__(self, levels=2, bottleneck_blocks=12, c_hidden=384, c_latent=4, codebook_size=8192, scale_factor=0.43):  # 0.3764
        super().__init__()
        self.c_latent = c_latent
        self.scale_factor = scale_factor
        c_levels = [c_hidden // (2**i) for i in reversed(range(levels))]

        # Encoder blocks
        self.in_block = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(3 * 4, c_levels[0], kernel_size=1))
        down_blocks = []
        for i in range(levels):
            if i > 0:
                down_blocks.append(nn.Conv2d(c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1))
            block = StageAResBlock(c_levels[i], c_levels[i] * 4)
            down_blocks.append(block)
        down_blocks.append(
            nn.Sequential(
                nn.Conv2d(c_levels[-1], c_latent, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_latent),  # then normalize them to have mean 0 and std 1
            )
        )
        self.down_blocks = nn.Sequential(*down_blocks)
        self.down_blocks[0]

        self.codebook_size = codebook_size
        self.vquantizer = VectorQuantize(c_latent, k=codebook_size)

        # Decoder blocks
        up_blocks = [nn.Sequential(nn.Conv2d(c_latent, c_levels[-1], kernel_size=1))]
        for i in range(levels):
            for j in range(bottleneck_blocks if i == 0 else 1):
                block = StageAResBlock(c_levels[levels - 1 - i], c_levels[levels - 1 - i] * 4)
                up_blocks.append(block)
            if i < levels - 1:
                up_blocks.append(
                    nn.ConvTranspose2d(c_levels[levels - 1 - i], c_levels[levels - 2 - i], kernel_size=4, stride=2, padding=1)
                )
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_block = nn.Sequential(
            nn.Conv2d(c_levels[0], 3 * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )

    def encode(self, x, quantize=False):
        x = self.in_block(x)
        x = self.down_blocks(x)
        if quantize:
            qe, (vq_loss, commit_loss), indices = self.vquantizer.forward(x, dim=1)
            return qe / self.scale_factor, x / self.scale_factor, indices, vq_loss + commit_loss * 0.25
        else:
            return x / self.scale_factor, None, None, None

    def decode(self, x):
        x = x * self.scale_factor
        x = self.up_blocks(x)
        x = self.out_block(x)
        return x

    def forward(self, x, quantize=False):
        qe, x, _, vq_loss = self.encode(x, quantize)
        x = self.decode(qe)
        return x, vq_loss


r"""

https://github.com/Stability-AI/StableCascade/blob/master/configs/inference/stage_b_3b.yaml

# GLOBAL STUFF
model_version: 3B
dtype: bfloat16

# For demonstration purposes in reconstruct_images.ipynb
webdataset_path: file:inference/imagenet_1024.tar
batch_size: 4
image_size: 1024
grad_accum_steps: 1

effnet_checkpoint_path: models/effnet_encoder.safetensors
stage_a_checkpoint_path: models/stage_a.safetensors
generator_checkpoint_path: models/stage_b_bf16.safetensors
"""


class StageB(nn.Module):
    def __init__(
        self,
        c_in=4,
        c_out=4,
        c_r=64,
        patch_size=2,
        c_cond=1280,
        c_hidden=[320, 640, 1280, 1280],
        nhead=[-1, -1, 20, 20],
        blocks=[[2, 6, 28, 6], [6, 28, 6, 2]],
        block_repeat=[[1, 1, 1, 1], [3, 3, 2, 2]],
        level_config=["CT", "CT", "CTA", "CTA"],
        c_clip=1280,
        c_clip_seq=4,
        c_effnet=16,
        c_pixels=3,
        kernel_size=3,
        dropout=[0, 0, 0.1, 0.1],
        self_attn=True,
        t_conds=["sca"],
    ):
        super().__init__()
        self.c_r = c_r
        self.t_conds = t_conds
        self.c_clip_seq = c_clip_seq
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)

        # CONDITIONING
        self.effnet_mapper = nn.Sequential(
            nn.Conv2d(c_effnet, c_hidden[0] * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c_hidden[0] * 4, c_hidden[0], kernel_size=1),
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
        )
        self.pixels_mapper = nn.Sequential(
            nn.Conv2d(c_pixels, c_hidden[0] * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c_hidden[0] * 4, c_hidden[0], kernel_size=1),
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
        )
        self.clip_mapper = nn.Linear(c_clip, c_cond * c_clip_seq)
        self.clip_norm = nn.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6)

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(c_in * (patch_size**2), c_hidden[0], kernel_size=1),
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == "C":
                return ResBlock(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == "A":
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout)
            elif block_type == "F":
                return FeedForwardBlock(c_hidden, dropout=dropout)
            elif block_type == "T":
                return TimestepBlock(c_hidden, c_r, conds=t_conds)
            else:
                raise Exception(f"Block type {block_type} not supported")

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(c_hidden)):
            if i > 0:
                self.down_downscalers.append(
                    nn.Sequential(
                        LayerNorm2d(c_hidden[i - 1], elementwise_affine=False, eps=1e-6),
                        nn.Conv2d(c_hidden[i - 1], c_hidden[i], kernel_size=2, stride=2),
                    )
                )
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.ModuleList()
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    block = get_block(block_type, c_hidden[i], nhead[i], dropout=dropout[i], self_attn=self_attn[i])
                    down_block.append(block)
            self.down_blocks.append(down_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[0][i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1))
                self.down_repeat_mappers.append(block_repeat_mappers)

        # -- up blocks
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            if i > 0:
                self.up_upscalers.append(
                    nn.Sequential(
                        LayerNorm2d(c_hidden[i], elementwise_affine=False, eps=1e-6),
                        nn.ConvTranspose2d(c_hidden[i], c_hidden[i - 1], kernel_size=2, stride=2),
                    )
                )
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.ModuleList()
            for j in range(blocks[1][::-1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    block = get_block(block_type, c_hidden[i], nhead[i], c_skip=c_skip, dropout=dropout[i], self_attn=self_attn[i])
                    up_block.append(block)
            self.up_blocks.append(up_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[1][::-1][i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1))
                self.up_repeat_mappers.append(block_repeat_mappers)

        # OUTPUT
        self.clf = nn.Sequential(
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c_hidden[0], c_out * (patch_size**2), kernel_size=1),
            nn.PixelShuffle(patch_size),
        )

        # --- WEIGHT INIT ---
        self.apply(self._init_weights)  # General init
        nn.init.normal_(self.clip_mapper.weight, std=0.02)  # conditionings
        nn.init.normal_(self.effnet_mapper[0].weight, std=0.02)  # conditionings
        nn.init.normal_(self.effnet_mapper[2].weight, std=0.02)  # conditionings
        nn.init.normal_(self.pixels_mapper[0].weight, std=0.02)  # conditionings
        nn.init.normal_(self.pixels_mapper[2].weight, std=0.02)  # conditionings
        torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # inputs
        nn.init.constant_(self.clf[1].weight, 0)  # outputs

        # blocks
        for level_block in self.down_blocks + self.up_blocks:
            for block in level_block:
                if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
                    block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks[0]))
                elif isinstance(block, TimestepBlock):
                    for layer in block.modules():
                        if isinstance(layer, nn.Linear):
                            nn.init.constant_(layer.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode="constant")
        return emb

    def gen_c_embeddings(self, clip):
        if len(clip.shape) == 2:
            clip = clip.unsqueeze(1)
        clip = self.clip_mapper(clip).view(clip.size(0), clip.size(1) * self.c_clip_seq, -1)
        clip = self.clip_norm(clip)
        return clip

    def _down_encode(self, x, r_embed, clip):
        level_outputs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)
        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, ResBlock) or (
                        hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, ResBlock)
                    ):
                        x = block(x)
                    elif isinstance(block, AttnBlock) or (
                        hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, AttnBlock)
                    ):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or (
                        hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, TimestepBlock)
                    ):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, ResBlock) or (
                        hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, ResBlock)
                    ):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                            x = torch.nn.functional.interpolate(x.float(), skip.shape[-2:], mode="bilinear", align_corners=True)
                        x = block(x, skip)
                    elif isinstance(block, AttnBlock) or (
                        hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, AttnBlock)
                    ):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or (
                        hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, TimestepBlock)
                    ):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
        return x

    def forward(self, x, r, effnet, clip, pixels=None, **kwargs):
        if pixels is None:
            pixels = x.new_zeros(x.size(0), 3, 8, 8)

        # Process the conditioning embeddings
        r_embed = self.gen_r_embedding(r)
        for c in self.t_conds:
            t_cond = kwargs.get(c, torch.zeros_like(r))
            r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond)], dim=1)
        clip = self.gen_c_embeddings(clip)

        # Model Blocks
        x = self.embedding(x)
        x = x + self.effnet_mapper(
            nn.functional.interpolate(effnet.float(), size=x.shape[-2:], mode="bilinear", align_corners=True)
        )
        x = x + nn.functional.interpolate(
            self.pixels_mapper(pixels).float(), size=x.shape[-2:], mode="bilinear", align_corners=True
        )
        level_outputs = self._down_encode(x, r_embed, clip)
        x = self._up_decode(level_outputs, r_embed, clip)
        return self.clf(x)

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data.clone().to(self_params.device) * (1 - beta)
        for self_buffers, src_buffers in zip(self.buffers(), src_model.buffers()):
            self_buffers.data = self_buffers.data * beta + src_buffers.data.clone().to(self_buffers.device) * (1 - beta)


r"""

https://github.com/Stability-AI/StableCascade/blob/master/configs/inference/stage_c_3b.yaml

# GLOBAL STUFF
model_version: 3.6B
dtype: bfloat16

effnet_checkpoint_path: models/effnet_encoder.safetensors
previewer_checkpoint_path: models/previewer.safetensors
generator_checkpoint_path: models/stage_c_bf16.safetensors
"""


class StageC(nn.Module):
    def __init__(
        self,
        c_in=16,
        c_out=16,
        c_r=64,
        patch_size=1,
        c_cond=2048,
        c_hidden=[2048, 2048],
        nhead=[32, 32],
        blocks=[[8, 24], [24, 8]],
        block_repeat=[[1, 1], [1, 1]],
        level_config=["CTA", "CTA"],
        c_clip_text=1280,
        c_clip_text_pooled=1280,
        c_clip_img=768,
        c_clip_seq=4,
        kernel_size=3,
        dropout=[0.1, 0.1],
        self_attn=True,
        t_conds=["sca", "crp"],
        switch_level=[False],
    ):
        super().__init__()
        self.c_r = c_r
        self.t_conds = t_conds
        self.c_clip_seq = c_clip_seq
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)

        # CONDITIONING
        self.clip_txt_mapper = nn.Linear(c_clip_text, c_cond)
        self.clip_txt_pooled_mapper = nn.Linear(c_clip_text_pooled, c_cond * c_clip_seq)
        self.clip_img_mapper = nn.Linear(c_clip_img, c_cond * c_clip_seq)
        self.clip_norm = nn.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6)

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(c_in * (patch_size**2), c_hidden[0], kernel_size=1),
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == "C":
                return ResBlock(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == "A":
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout)
            elif block_type == "F":
                return FeedForwardBlock(c_hidden, dropout=dropout)
            elif block_type == "T":
                return TimestepBlock(c_hidden, c_r, conds=t_conds)
            else:
                raise Exception(f"Block type {block_type} not supported")

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(c_hidden)):
            if i > 0:
                self.down_downscalers.append(
                    nn.Sequential(
                        LayerNorm2d(c_hidden[i - 1], elementwise_affine=False, eps=1e-6),
                        UpDownBlock2d(c_hidden[i - 1], c_hidden[i], mode="down", enabled=switch_level[i - 1]),
                    )
                )
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.ModuleList()
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    block = get_block(block_type, c_hidden[i], nhead[i], dropout=dropout[i], self_attn=self_attn[i])
                    down_block.append(block)
            self.down_blocks.append(down_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[0][i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1))
                self.down_repeat_mappers.append(block_repeat_mappers)

        # -- up blocks
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            if i > 0:
                self.up_upscalers.append(
                    nn.Sequential(
                        LayerNorm2d(c_hidden[i], elementwise_affine=False, eps=1e-6),
                        UpDownBlock2d(c_hidden[i], c_hidden[i - 1], mode="up", enabled=switch_level[i - 1]),
                    )
                )
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.ModuleList()
            for j in range(blocks[1][::-1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    block = get_block(block_type, c_hidden[i], nhead[i], c_skip=c_skip, dropout=dropout[i], self_attn=self_attn[i])
                    up_block.append(block)
            self.up_blocks.append(up_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[1][::-1][i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1))
                self.up_repeat_mappers.append(block_repeat_mappers)

        # OUTPUT
        self.clf = nn.Sequential(
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c_hidden[0], c_out * (patch_size**2), kernel_size=1),
            nn.PixelShuffle(patch_size),
        )

        # --- WEIGHT INIT ---
        self.apply(self._init_weights)  # General init
        nn.init.normal_(self.clip_txt_mapper.weight, std=0.02)  # conditionings
        nn.init.normal_(self.clip_txt_pooled_mapper.weight, std=0.02)  # conditionings
        nn.init.normal_(self.clip_img_mapper.weight, std=0.02)  # conditionings
        torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # inputs
        nn.init.constant_(self.clf[1].weight, 0)  # outputs

        # blocks
        for level_block in self.down_blocks + self.up_blocks:
            for block in level_block:
                if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
                    block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks[0]))
                elif isinstance(block, TimestepBlock):
                    for layer in block.modules():
                        if isinstance(layer, nn.Linear):
                            nn.init.constant_(layer.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def set_gradient_checkpointing(self, value):
        for block in self.down_blocks + self.up_blocks:
            for layer in block:
                if hasattr(layer, "set_gradient_checkpointing"):
                    layer.set_gradient_checkpointing(value)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode="constant")
        return emb

    def gen_c_embeddings(self, clip_txt, clip_txt_pooled, clip_img):
        clip_txt = self.clip_txt_mapper(clip_txt)
        if len(clip_txt_pooled.shape) == 2:
            clip_txt_pool = clip_txt_pooled.unsqueeze(1)
        if len(clip_img.shape) == 2:
            clip_img = clip_img.unsqueeze(1)
        clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(
            clip_txt_pooled.size(0), clip_txt_pooled.size(1) * self.c_clip_seq, -1
        )
        clip_img = self.clip_img_mapper(clip_img).view(clip_img.size(0), clip_img.size(1) * self.c_clip_seq, -1)
        clip = torch.cat([clip_txt, clip_txt_pool, clip_img], dim=1)
        clip = self.clip_norm(clip)
        return clip

    def _down_encode(self, x, r_embed, clip, cnet=None):
        level_outputs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)
        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, ResBlock) or (
                        hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, ResBlock)
                    ):
                        if cnet is not None:
                            next_cnet = cnet()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(next_cnet, size=x.shape[-2:], mode="bilinear", align_corners=True)
                        x = block(x)
                    elif isinstance(block, AttnBlock) or (
                        hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, AttnBlock)
                    ):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or (
                        hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, TimestepBlock)
                    ):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip, cnet=None):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, ResBlock) or (
                        hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, ResBlock)
                    ):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                            x = torch.nn.functional.interpolate(x.float(), skip.shape[-2:], mode="bilinear", align_corners=True)
                        if cnet is not None:
                            next_cnet = cnet()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(next_cnet, size=x.shape[-2:], mode="bilinear", align_corners=True)
                        x = block(x, skip)
                    elif isinstance(block, AttnBlock) or (
                        hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, AttnBlock)
                    ):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or (
                        hasattr(block, "_fsdp_wrapped_module") and isinstance(block._fsdp_wrapped_module, TimestepBlock)
                    ):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
        return x

    def forward(self, x, r, clip_text, clip_text_pooled, clip_img, cnet=None, **kwargs):
        # Process the conditioning embeddings
        r_embed = self.gen_r_embedding(r)
        for c in self.t_conds:
            t_cond = kwargs.get(c, torch.zeros_like(r))
            r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond)], dim=1)
        clip = self.gen_c_embeddings(clip_text, clip_text_pooled, clip_img)

        # Model Blocks
        x = self.embedding(x)
        # ControlNet is not supported yet
        # if cnet is not None:
        #     cnet = ControlNetDeliverer(cnet)
        level_outputs = self._down_encode(x, r_embed, clip, cnet)
        x = self._up_decode(level_outputs, r_embed, clip, cnet)
        return self.clf(x)

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data.clone().to(self_params.device) * (1 - beta)
        for self_buffers, src_buffers in zip(self.buffers(), src_model.buffers()):
            self_buffers.data = self_buffers.data * beta + src_buffers.data.clone().to(self_buffers.device) * (1 - beta)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


# Fast Decoder for Stage C latents. E.g. 16 x 24 x 24 -> 3 x 192 x 192
class Previewer(nn.Module):
    def __init__(self, c_in=16, c_hidden=512, c_out=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=1),  # 16 channels to 512 channels
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),
            nn.ConvTranspose2d(c_hidden, c_hidden // 2, kernel_size=2, stride=2),  # 16 -> 32
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 2),
            nn.Conv2d(c_hidden // 2, c_hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 2),
            nn.ConvTranspose2d(c_hidden // 2, c_hidden // 4, kernel_size=2, stride=2),  # 32 -> 64
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),
            nn.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),
            nn.ConvTranspose2d(c_hidden // 4, c_hidden // 4, kernel_size=2, stride=2),  # 64 -> 128
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),
            nn.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),
            nn.Conv2d(c_hidden // 4, c_out, kernel_size=1),
        )

    def forward(self, x):
        return self.blocks(x)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def get_clip_conditions(captions: Optional[List[str]], input_ids, tokenizer, text_model):
    # deprecated

    # self, batch: dict, tokenizer, text_model, is_eval=False, is_unconditional=False, eval_image_embeds=False, return_fields=None
    # is_eval の処理をここでやるのは微妙なので別のところでやる
    # is_unconditional もここでやるのは微妙なので別のところでやる
    # clip_image はとりあえずサポートしない
    if captions is not None:
        clip_tokens_unpooled = tokenizer(
            captions, truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        ).to(text_model.device)
        text_encoder_output = text_model(**clip_tokens_unpooled, output_hidden_states=True)
    else:
        text_encoder_output = text_model(input_ids, output_hidden_states=True)

    text_embeddings = text_encoder_output.hidden_states[-1]
    text_pooled_embeddings = text_encoder_output.text_embeds.unsqueeze(1)

    return text_embeddings, text_pooled_embeddings
    # return {"clip_text": text_embeddings, "clip_text_pooled": text_pooled_embeddings}  # , "clip_img": image_embeddings}


# region gdf


class SimpleSampler:
    def __init__(self, gdf):
        self.gdf = gdf
        self.current_step = -1

    def __call__(self, *args, **kwargs):
        self.current_step += 1
        return self.step(*args, **kwargs)

    def init_x(self, shape):
        return torch.randn(*shape)

    def step(self, x, x0, epsilon, logSNR, logSNR_prev):
        raise NotImplementedError("You should override the 'apply' function.")


class DDIMSampler(SimpleSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev, eta=0):
        a, b = self.gdf.input_scaler(logSNR)
        if len(a.shape) == 1:
            a, b = a.view(-1, *[1] * (len(x0.shape) - 1)), b.view(-1, *[1] * (len(x0.shape) - 1))

        a_prev, b_prev = self.gdf.input_scaler(logSNR_prev)
        if len(a_prev.shape) == 1:
            a_prev, b_prev = a_prev.view(-1, *[1] * (len(x0.shape) - 1)), b_prev.view(-1, *[1] * (len(x0.shape) - 1))

        sigma_tau = eta * (b_prev**2 / b**2).sqrt() * (1 - a**2 / a_prev**2).sqrt() if eta > 0 else 0
        # x = a_prev * x0 + (1 - a_prev**2 - sigma_tau ** 2).sqrt() * epsilon + sigma_tau * torch.randn_like(x0)
        x = a_prev * x0 + (b_prev**2 - sigma_tau**2).sqrt() * epsilon + sigma_tau * torch.randn_like(x0)
        return x


class DDPMSampler(DDIMSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev, eta=1):
        return super().step(x, x0, epsilon, logSNR, logSNR_prev, eta)


class LCMSampler(SimpleSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev):
        a_prev, b_prev = self.gdf.input_scaler(logSNR_prev)
        if len(a_prev.shape) == 1:
            a_prev, b_prev = a_prev.view(-1, *[1] * (len(x0.shape) - 1)), b_prev.view(-1, *[1] * (len(x0.shape) - 1))
        return x0 * a_prev + torch.randn_like(epsilon) * b_prev


class GDF:
    def __init__(self, schedule, input_scaler, target, noise_cond, loss_weight, offset_noise=0):
        self.schedule = schedule
        self.input_scaler = input_scaler
        self.target = target
        self.noise_cond = noise_cond
        self.loss_weight = loss_weight
        self.offset_noise = offset_noise

    def setup_limits(self, stretch_max=True, stretch_min=True, shift=1):
        stretched_limits = self.input_scaler.setup_limits(self.schedule, self.input_scaler, stretch_max, stretch_min, shift)
        return stretched_limits

    def diffuse(self, x0, epsilon=None, t=None, shift=1, loss_shift=1, offset=None):
        if epsilon is None:
            epsilon = torch.randn_like(x0)
        if self.offset_noise > 0:
            if offset is None:
                offset = torch.randn([x0.size(0), x0.size(1)] + [1] * (len(x0.shape) - 2)).to(x0.device)
            epsilon = epsilon + offset * self.offset_noise
        logSNR = self.schedule(x0.size(0) if t is None else t, shift=shift).to(x0.device)
        a, b = self.input_scaler(logSNR)  # B
        if len(a.shape) == 1:
            a, b = a.view(-1, *[1] * (len(x0.shape) - 1)), b.view(-1, *[1] * (len(x0.shape) - 1))  # BxCxHxW
        target = self.target(x0, epsilon, logSNR, a, b)

        # noised, noise, logSNR, t_cond
        return x0 * a + epsilon * b, epsilon, target, logSNR, self.noise_cond(logSNR), self.loss_weight(logSNR, shift=loss_shift)

    def undiffuse(self, x, logSNR, pred):
        a, b = self.input_scaler(logSNR)
        if len(a.shape) == 1:
            a, b = a.view(-1, *[1] * (len(x.shape) - 1)), b.view(-1, *[1] * (len(x.shape) - 1))
        return self.target.x0(x, pred, logSNR, a, b), self.target.epsilon(x, pred, logSNR, a, b)

    def sample(
        self,
        model,
        model_inputs,
        shape,
        unconditional_inputs=None,
        sampler=None,
        schedule=None,
        t_start=1.0,
        t_end=0.0,
        timesteps=20,
        x_init=None,
        cfg=3.0,
        cfg_t_stop=None,
        cfg_t_start=None,
        cfg_rho=0.7,
        sampler_params=None,
        shift=1,
        device="cpu",
    ):
        sampler_params = {} if sampler_params is None else sampler_params
        if sampler is None:
            sampler = DDPMSampler(self)
        r_range = torch.linspace(t_start, t_end, timesteps + 1)
        schedule = self.schedule if schedule is None else schedule
        logSNR_range = schedule(r_range, shift=shift)[:, None].expand(-1, shape[0] if x_init is None else x_init.size(0)).to(device)

        x = sampler.init_x(shape).to(device) if x_init is None else x_init.clone()
        if cfg is not None:
            if unconditional_inputs is None:
                unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
            model_inputs = {
                k: (
                    torch.cat([v, v_u], dim=0)
                    if isinstance(v, torch.Tensor)
                    else (
                        [
                            (
                                torch.cat([vi, vi_u], dim=0)
                                if isinstance(vi, torch.Tensor) and isinstance(vi_u, torch.Tensor)
                                else None
                            )
                            for vi, vi_u in zip(v, v_u)
                        ]
                        if isinstance(v, list)
                        else (
                            {vk: torch.cat([v[vk], v_u.get(vk, torch.zeros_like(v[vk]))], dim=0) for vk in v}
                            if isinstance(v, dict)
                            else None
                        )
                    )
                )
                for (k, v), (k_u, v_u) in zip(model_inputs.items(), unconditional_inputs.items())
            }
        for i in range(0, timesteps):
            noise_cond = self.noise_cond(logSNR_range[i])
            if (
                cfg is not None
                and (cfg_t_stop is None or r_range[i].item() >= cfg_t_stop)
                and (cfg_t_start is None or r_range[i].item() <= cfg_t_start)
            ):
                cfg_val = cfg
                if isinstance(cfg_val, (list, tuple)):
                    assert len(cfg_val) == 2, "cfg must be a float or a list/tuple of length 2"
                    cfg_val = cfg_val[0] * r_range[i].item() + cfg_val[1] * (1 - r_range[i].item())
                pred, pred_unconditional = model(torch.cat([x, x], dim=0), noise_cond.repeat(2), **model_inputs).chunk(2)
                pred_cfg = torch.lerp(pred_unconditional, pred, cfg_val)
                if cfg_rho > 0:
                    std_pos, std_cfg = pred.std(), pred_cfg.std()
                    pred = cfg_rho * (pred_cfg * std_pos / (std_cfg + 1e-9)) + pred_cfg * (1 - cfg_rho)
                else:
                    pred = pred_cfg
            else:
                pred = model(x, noise_cond, **model_inputs)
            x0, epsilon = self.undiffuse(x, logSNR_range[i], pred)
            x = sampler(x, x0, epsilon, logSNR_range[i], logSNR_range[i + 1], **sampler_params)
            altered_vars = yield (x0, x, pred)

            # Update some running variables if the user wants
            if altered_vars is not None:
                cfg = altered_vars.get("cfg", cfg)
                cfg_rho = altered_vars.get("cfg_rho", cfg_rho)
                sampler = altered_vars.get("sampler", sampler)
                model_inputs = altered_vars.get("model_inputs", model_inputs)
                x = altered_vars.get("x", x)
                x_init = altered_vars.get("x_init", x_init)


class BaseSchedule:
    def __init__(self, *args, force_limits=True, discrete_steps=None, shift=1, **kwargs):
        self.setup(*args, **kwargs)
        self.limits = None
        self.discrete_steps = discrete_steps
        self.shift = shift
        if force_limits:
            self.reset_limits()

    def reset_limits(self, shift=1, disable=False):
        try:
            self.limits = None if disable else self(torch.tensor([1.0, 0.0]), shift=shift).tolist()  # min, max
            return self.limits
        except Exception:
            print("WARNING: this schedule doesn't support t and will be unbounded")
            return None

    def setup(self, *args, **kwargs):
        raise NotImplementedError("this method needs to be overriden")

    def schedule(self, *args, **kwargs):
        raise NotImplementedError("this method needs to be overriden")

    def __call__(self, t, *args, shift=1, **kwargs):
        if isinstance(t, torch.Tensor):
            batch_size = None
            if self.discrete_steps is not None:
                if t.dtype != torch.long:
                    t = (t * (self.discrete_steps - 1)).round().long()
                t = t / (self.discrete_steps - 1)
            t = t.clamp(0, 1)
        else:
            batch_size = t
            t = None
        logSNR = self.schedule(t, batch_size, *args, **kwargs)
        if shift * self.shift != 1:
            logSNR += 2 * np.log(1 / (shift * self.shift))
        if self.limits is not None:
            logSNR = logSNR.clamp(*self.limits)
        return logSNR


class CosineSchedule(BaseSchedule):
    def setup(self, s=0.008, clamp_range=[0.0001, 0.9999], norm_instead=False):
        self.s = torch.tensor([s])
        self.clamp_range = clamp_range
        self.norm_instead = norm_instead
        self.min_var = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

    def schedule(self, t, batch_size):
        if t is None:
            t = (1 - torch.rand(batch_size)).add(0.001).clamp(0.001, 1.0)
        s, min_var = self.s.to(t.device), self.min_var.to(t.device)
        var = torch.cos((s + t) / (1 + s) * torch.pi * 0.5).clamp(0, 1) ** 2 / min_var
        if self.norm_instead:
            var = var * (self.clamp_range[1] - self.clamp_range[0]) + self.clamp_range[0]
        else:
            var = var.clamp(*self.clamp_range)
        logSNR = (var / (1 - var)).log()
        return logSNR


class BaseScaler:
    def __init__(self):
        self.stretched_limits = None

    def setup_limits(self, schedule, input_scaler, stretch_max=True, stretch_min=True, shift=1):
        min_logSNR = schedule(torch.ones(1), shift=shift)
        max_logSNR = schedule(torch.zeros(1), shift=shift)

        min_a, max_b = [v.item() for v in input_scaler(min_logSNR)] if stretch_max else [0, 1]
        max_a, min_b = [v.item() for v in input_scaler(max_logSNR)] if stretch_min else [1, 0]
        self.stretched_limits = [min_a, max_a, min_b, max_b]
        return self.stretched_limits

    def stretch_limits(self, a, b):
        min_a, max_a, min_b, max_b = self.stretched_limits
        return (a - min_a) / (max_a - min_a), (b - min_b) / (max_b - min_b)

    def scalers(self, logSNR):
        raise NotImplementedError("this method needs to be overridden")

    def __call__(self, logSNR):
        a, b = self.scalers(logSNR)
        if self.stretched_limits is not None:
            a, b = self.stretch_limits(a, b)
        return a, b


class VPScaler(BaseScaler):
    def scalers(self, logSNR):
        a_squared = logSNR.sigmoid()
        a = a_squared.sqrt()
        b = (1 - a_squared).sqrt()
        return a, b


class EpsilonTarget:
    def __call__(self, x0, epsilon, logSNR, a, b):
        return epsilon

    def x0(self, noised, pred, logSNR, a, b):
        return (noised - pred * b) / a

    def epsilon(self, noised, pred, logSNR, a, b):
        return pred


class BaseNoiseCond:
    def __init__(self, *args, shift=1, clamp_range=None, **kwargs):
        clamp_range = [-1e9, 1e9] if clamp_range is None else clamp_range
        self.shift = shift
        self.clamp_range = clamp_range
        self.setup(*args, **kwargs)

    def setup(self, *args, **kwargs):
        pass  # this method is optional, override it if required

    def cond(self, logSNR):
        raise NotImplementedError("this method needs to be overriden")

    def __call__(self, logSNR):
        if self.shift != 1:
            logSNR = logSNR.clone() + 2 * np.log(self.shift)
        return self.cond(logSNR).clamp(*self.clamp_range)


class CosineTNoiseCond(BaseNoiseCond):
    def setup(self, s=0.008, clamp_range=[0, 1]):  # [0.0001, 0.9999]
        self.s = torch.tensor([s])
        self.clamp_range = clamp_range
        self.min_var = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

    def cond(self, logSNR):
        var = logSNR.sigmoid()
        var = var.clamp(*self.clamp_range)
        s, min_var = self.s.to(var.device), self.min_var.to(var.device)
        t = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
        return t


# --- Loss Weighting
class BaseLossWeight:
    def weight(self, logSNR):
        raise NotImplementedError("this method needs to be overridden")

    def __call__(self, logSNR, *args, shift=1, clamp_range=None, **kwargs):
        clamp_range = [-1e9, 1e9] if clamp_range is None else clamp_range
        if shift != 1:
            logSNR = logSNR.clone() + 2 * np.log(shift)
        return self.weight(logSNR, *args, **kwargs).clamp(*clamp_range)


# class ComposedLossWeight(BaseLossWeight):
#     def __init__(self, div, mul):
#         self.mul = [mul] if isinstance(mul, BaseLossWeight) else mul
#         self.div = [div] if isinstance(div, BaseLossWeight) else div

#     def weight(self, logSNR):
#         prod, div = 1, 1
#         for m in self.mul:
#             prod *= m.weight(logSNR)
#         for d in self.div:
#             div *= d.weight(logSNR)
#         return prod/div

# class ConstantLossWeight(BaseLossWeight):
#     def __init__(self, v=1):
#         self.v = v

#     def weight(self, logSNR):
#         return torch.ones_like(logSNR) * self.v

# class SNRLossWeight(BaseLossWeight):
#     def weight(self, logSNR):
#         return logSNR.exp()


class P2LossWeight(BaseLossWeight):
    def __init__(self, k=1.0, gamma=1.0, s=1.0):
        self.k, self.gamma, self.s = k, gamma, s

    def weight(self, logSNR):
        return (self.k + (logSNR * self.s).exp()) ** -self.gamma


# class SNRPlusOneLossWeight(BaseLossWeight):
#     def weight(self, logSNR):
#         return logSNR.exp() + 1

# class MinSNRLossWeight(BaseLossWeight):
#     def __init__(self, max_snr=5):
#         self.max_snr = max_snr

#     def weight(self, logSNR):
#         return logSNR.exp().clamp(max=self.max_snr)

# class MinSNRPlusOneLossWeight(BaseLossWeight):
#     def __init__(self, max_snr=5):
#         self.max_snr = max_snr

#     def weight(self, logSNR):
#         return (logSNR.exp() + 1).clamp(max=self.max_snr)

# class TruncatedSNRLossWeight(BaseLossWeight):
#     def __init__(self, min_snr=1):
#         self.min_snr = min_snr

#     def weight(self, logSNR):
#         return logSNR.exp().clamp(min=self.min_snr)

# class SechLossWeight(BaseLossWeight):
#     def __init__(self, div=2):
#         self.div = div

#     def weight(self, logSNR):
#         return 1/(logSNR/self.div).cosh()

# class DebiasedLossWeight(BaseLossWeight):
#     def weight(self, logSNR):
#         return 1/logSNR.exp().sqrt()

# class SigmoidLossWeight(BaseLossWeight):
#     def __init__(self, s=1):
#         self.s = s

#     def weight(self, logSNR):
#         return (logSNR * self.s).sigmoid()


class AdaptiveLossWeight(BaseLossWeight):
    def __init__(self, logsnr_range=[-10, 10], buckets=300, weight_range=[1e-7, 1e7]):
        self.bucket_ranges = torch.linspace(logsnr_range[0], logsnr_range[1], buckets - 1)
        self.bucket_losses = torch.ones(buckets)
        self.weight_range = weight_range

    def weight(self, logSNR):
        indices = torch.searchsorted(self.bucket_ranges.to(logSNR.device), logSNR)
        return (1 / self.bucket_losses.to(logSNR.device)[indices]).clamp(*self.weight_range)

    def update_buckets(self, logSNR, loss, beta=0.99):
        indices = torch.searchsorted(self.bucket_ranges.to(logSNR.device), logSNR).cpu()
        self.bucket_losses[indices] = self.bucket_losses[indices] * beta + loss.detach().cpu() * (1 - beta)


# endregion gdf
