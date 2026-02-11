# some modules/classes are copied and modified from https://github.com/mcmonkey4eva/sd3-ref
# the original code is licensed under the MIT License

# and some module/classes are contributed from KohakuBlueleaf. Thanks for the contribution!

from ast import Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import math
from types import SimpleNamespace
from typing import Dict, List, Optional, Union
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import CLIPTokenizer, T5TokenizerFast

from library import custom_offloading_utils
from library.device_utils import clean_memory_on_device

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


memory_efficient_attention = None
try:
    import xformers
except:
    pass

try:
    from xformers.ops import memory_efficient_attention
except:
    memory_efficient_attention = None


# region mmdit


@dataclass
class SD3Params:
    patch_size: int
    depth: int
    num_patches: int
    pos_embed_max_size: int
    adm_in_channels: int
    qk_norm: Optional[str]
    x_block_self_attn_layers: list[int]
    context_embedder_in_features: int
    context_embedder_out_features: int
    model_type: str


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    scaling_factor=None,
    offset=None,
):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    if scaling_factor is not None:
        grid = grid / scaling_factor
    if offset is not None:
        grid = grid - offset

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_scaled_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, sample_size=64, base_size=16):
    """
    This function is contributed by KohakuBlueleaf. Thanks for the contribution!

    Creates scaled 2D sinusoidal positional embeddings that maintain consistent relative positions
    when the resolution differs from the training resolution.

    Args:
        embed_dim (int): Dimension of the positional embedding.
        grid_size (int or tuple): Size of the position grid (H, W). If int, assumes square grid.
        cls_token (bool): Whether to include class token. Defaults to False.
        extra_tokens (int): Number of extra tokens (e.g., cls_token). Defaults to 0.
        sample_size (int): Reference resolution (typically training resolution). Defaults to 64.
        base_size (int): Base grid size used during training. Defaults to 16.

    Returns:
        numpy.ndarray: Positional embeddings of shape (H*W, embed_dim) or
                      (H*W + extra_tokens, embed_dim) if cls_token is True.
    """
    # Convert grid_size to tuple if it's an integer
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    # Create normalized grid coordinates (0 to 1)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / grid_size[0]
    grid_w = np.arange(grid_size[1], dtype=np.float32) / grid_size[1]

    # Calculate scaling factors for height and width
    # This ensures that the central region matches the original resolution's embeddings
    scale_h = base_size * grid_size[0] / (sample_size)
    scale_w = base_size * grid_size[1] / (sample_size)

    # Calculate shift values to center the original resolution's embedding region
    # This ensures that the central sample_size x sample_size region has similar
    # positional embeddings to the original resolution
    shift_h = 1 * scale_h * (grid_size[0] - sample_size) / (2 * grid_size[0])
    shift_w = 1 * scale_w * (grid_size[1] - sample_size) / (2 * grid_size[1])

    # Apply scaling and shifting to create the final grid coordinates
    grid_h = grid_h * scale_h - shift_h
    grid_w = grid_w * scale_w - shift_w

    # Create 2D grid using meshgrid (note: w goes first)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    # # Calculate the starting indices for the central region
    # # This is used for debugging/visualization of the central region
    # st_h = (grid_size[0] - sample_size) // 2
    # st_w = (grid_size[1] - sample_size) // 2
    # print(grid[:, st_h : st_h + sample_size, st_w : st_w + sample_size])

    # Reshape grid for positional embedding calculation
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    # Generate the sinusoidal positional embeddings
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    # Add zeros for extra tokens (e.g., [CLS] token) if required
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)

    return pos_embed


# if __name__ == "__main__":
#     # This is what you get when you load SD3.5 state dict
#     pos_emb = torch.from_numpy(get_scaled_2d_sincos_pos_embed(
#         1536, [384, 384], sample_size=64, base_size=16
#     )).float().unsqueeze(0)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(
    embed_dim,
    pos,
    device=None,
    dtype=torch.float32,
):
    omega = torch.arange(embed_dim // 2, device=device, dtype=dtype)
    omega *= 2.0 / embed_dim
    omega = 1.0 / 10000**omega
    out = torch.outer(pos.reshape(-1), omega)
    emb = torch.cat([out.sin(), out.cos()], dim=1)
    return emb


def get_2d_sincos_pos_embed_torch(
    embed_dim,
    w,
    h,
    val_center=7.5,
    val_magnitude=7.5,
    device=None,
    dtype=torch.float32,
):
    small = min(h, w)
    val_h = (h / small) * val_magnitude
    val_w = (w / small) * val_magnitude
    grid_h, grid_w = torch.meshgrid(
        torch.linspace(-val_h + val_center, val_h + val_center, h, device=device, dtype=dtype),
        torch.linspace(-val_w + val_center, val_w + val_center, w, device=device, dtype=dtype),
        indexing="ij",
    )
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_h, device=device, dtype=dtype)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_w, device=device, dtype=dtype)
    emb = torch.cat([emb_w, emb_h], dim=1)  # (H*W, D)
    return emb


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def default(x, default_value):
    if x is None:
        return default_value
    return x


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    # freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
    #     device=t.device, dtype=t.dtype
    # )
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(dtype=t.dtype)
    return embedding


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_channels=3,
        embed_dim=512,
        norm_layer=None,
        flatten=True,
        bias=True,
        strict_img_size=True,
        dynamic_img_pad=False,
    ):
        # dynamic_img_pad and norm is omitted in SD3.5
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        if img_size is not None:
            self.img_size = img_size
            self.grid_size = img_size // patch_size
            self.num_patches = self.grid_size**2
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size, bias=bias)
        self.norm = nn.Identity() if norm_layer is None else norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.dynamic_img_pad:
            # Pad input so we won't have partial patch
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# FinalLayer in mmdit.py
class UnPatch(nn.Module):
    def __init__(self, hidden_size=512, patch_size=4, out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.c = out_channels

        # eps is default in mmdit.py
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size**2 * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, cmod, H=None, W=None):
        b, n, _ = x.shape
        p = self.patch_size
        c = self.c
        if H is None and W is None:
            w = h = int(n**0.5)
            assert h * w == n
        else:
            h = H // p if H else n // (W // p)
            w = W // p if W else n // h
            assert h * w == n

        shift, scale = self.adaLN_modulation(cmod).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        x = x.view(b, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, h * p, w * p)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=lambda: nn.GELU(),
        norm_layer=None,
        bias=True,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.use_conv = use_conv

        layer = partial(nn.Conv1d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = layer(in_features, hidden_features, bias=bias)
        self.fc2 = layer(hidden_features, out_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size, freq_embed_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size

    def forward(self, t, dtype=None, **kwargs):
        t_freq = timestep_embedding(t, self.freq_embed_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.mlp(x)


def rmsnorm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine: bool = False,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        x = rmsnorm(x, eps=self.eps)
        if self.learnable_scale:
            return x * self.weight.to(device=x.device, dtype=x.dtype)
        else:
            return x


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


# Linears for SelfAttention in mmdit.py
class AttentionLinears(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        pre_only: bool = False,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if not pre_only:
            self.proj = nn.Linear(dim, dim)
        self.pre_only = pre_only

        if qk_norm == "rms":
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        output:
            q, k, v: [B, L, D]
        """
        B, L, C = x.shape
        qkv: torch.Tensor = self.qkv(x)
        q, k, v = qkv.reshape(B, L, -1, self.head_dim).chunk(3, dim=2)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        return x


MEMORY_LAYOUTS = {
    "torch": (
        lambda x, head_dim: x.reshape(x.shape[0], x.shape[1], -1, head_dim).transpose(1, 2),
        lambda x: x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1),
        lambda x: (1, x, 1, 1),
    ),
    "xformers": (
        lambda x, head_dim: x.reshape(x.shape[0], x.shape[1], -1, head_dim),
        lambda x: x.reshape(x.shape[0], x.shape[1], -1),
        lambda x: (1, 1, x, 1),
    ),
    "math": (
        lambda x, head_dim: x.reshape(x.shape[0], x.shape[1], -1, head_dim).transpose(1, 2),
        lambda x: x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1),
        lambda x: (1, x, 1, 1),
    ),
}
# ATTN_FUNCTION = {
#     "torch": F.scaled_dot_product_attention,
#     "xformers": memory_efficient_attention,
# }


def vanilla_attention(q, k, v, mask, scale=None):
    if scale is None:
        scale = math.sqrt(q.size(-1))
    scores = torch.bmm(q, k.transpose(-1, -2)) / scale
    if mask is not None:
        mask = einops.rearrange(mask, "b ... -> b (...)")
        max_neg_value = -torch.finfo(scores.dtype).max
        mask = einops.repeat(mask, "b j -> (b h) j", h=q.size(-3))
        scores = scores.masked_fill(~mask, max_neg_value)
    p_attn = F.softmax(scores, dim=-1)
    return torch.bmm(p_attn, v)


def attention(q, k, v, head_dim, mask=None, scale=None, mode="xformers"):
    """
    q, k, v: [B, L, D]
    """
    pre_attn_layout = MEMORY_LAYOUTS[mode][0]
    post_attn_layout = MEMORY_LAYOUTS[mode][1]
    q = pre_attn_layout(q, head_dim)
    k = pre_attn_layout(k, head_dim)
    v = pre_attn_layout(v, head_dim)

    # scores = ATTN_FUNCTION[mode](q, k.to(q), v.to(q), mask, scale=scale)
    if mode == "torch":
        assert scale is None
        scores = F.scaled_dot_product_attention(q, k.to(q), v.to(q), mask)  # , scale=scale)
    elif mode == "xformers":
        scores = memory_efficient_attention(q, k.to(q), v.to(q), mask, scale=scale)
    else:
        scores = vanilla_attention(q, k.to(q), v.to(q), mask, scale=scale)

    scores = post_attn_layout(scores)
    return scores


# DismantledBlock in mmdit.py
class SingleDiTBlock(nn.Module):
    """
    A DiT block with gated adaptive layer norm (adaLN) conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "xformers",
        qkv_bias: bool = False,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: Optional[str] = None,
        x_block_self_attn: bool = False,
        **block_kwargs,
    ):
        super().__init__()
        assert attn_mode in MEMORY_LAYOUTS
        self.attn_mode = attn_mode
        if not rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionLinears(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, pre_only=pre_only, qk_norm=qk_norm)

        self.x_block_self_attn = x_block_self_attn
        if self.x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            self.attn2 = AttentionLinears(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, pre_only=False, qk_norm=qk_norm)

        if not pre_only:
            if not rmsnorm:
                self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            else:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = MLP(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=lambda: nn.GELU(approximate="tanh"),
                )
            else:
                self.mlp = SwiGLUFeedForward(
                    dim=hidden_size,
                    hidden_dim=mlp_hidden_dim,
                    multiple_of=256,
                )
        self.scale_mod_only = scale_mod_only
        if self.x_block_self_attn:
            n_mods = 9
        elif not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size))
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if not self.pre_only:
            if not self.scale_mod_only:
                (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=-1)
            else:
                shift_msa = None
                shift_mlp = None
                (scale_msa, gate_msa, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(4, dim=-1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                (shift_msa, scale_msa) = self.adaLN_modulation(c).chunk(2, dim=-1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def pre_attention_x(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert self.x_block_self_attn
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2) = self.adaLN_modulation(
            c
        ).chunk(9, dim=1)
        x_norm = self.norm1(x)
        qkv = self.attn.pre_attention(modulate(x_norm, shift_msa, scale_msa))
        qkv2 = self.attn2.pre_attention(modulate(x_norm, shift_msa2, scale_msa2))
        return qkv, qkv2, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2)

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def post_attention_x(self, attn, attn2, x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2, attn1_dropout: float = 0.0):
        assert not self.pre_only
        if attn1_dropout > 0.0:
            # Use torch.bernoulli to implement dropout, only dropout the batch dimension
            attn1_dropout = torch.bernoulli(torch.full((attn.size(0), 1, 1), 1 - attn1_dropout, device=attn.device))
            attn_ = gate_msa.unsqueeze(1) * self.attn.post_attention(attn) * attn1_dropout
        else:
            attn_ = gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + attn_
        attn2_ = gate_msa2.unsqueeze(1) * self.attn2.post_attention(attn2)
        x = x + attn2_
        mlp_ = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + mlp_
        return x


# JointBlock + block_mixing in mmdit.py
class MMDiTBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop("pre_only")
        x_block_self_attn = kwargs.pop("x_block_self_attn")

        self.context_block = SingleDiTBlock(*args, pre_only=pre_only, **kwargs)
        self.x_block = SingleDiTBlock(*args, pre_only=False, x_block_self_attn=x_block_self_attn, **kwargs)

        self.head_dim = self.x_block.attn.head_dim
        self.mode = self.x_block.attn_mode
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def _forward(self, context, x, c):
        ctx_qkv, ctx_intermediate = self.context_block.pre_attention(context, c)

        if self.x_block.x_block_self_attn:
            x_qkv, x_qkv2, x_intermediates = self.x_block.pre_attention_x(x, c)
        else:
            x_qkv, x_intermediates = self.x_block.pre_attention(x, c)

        ctx_len = ctx_qkv[0].size(1)

        q = torch.concat((ctx_qkv[0], x_qkv[0]), dim=1)
        k = torch.concat((ctx_qkv[1], x_qkv[1]), dim=1)
        v = torch.concat((ctx_qkv[2], x_qkv[2]), dim=1)

        attn = attention(q, k, v, head_dim=self.head_dim, mode=self.mode)
        ctx_attn_out = attn[:, :ctx_len]
        x_attn_out = attn[:, ctx_len:]

        if self.x_block.x_block_self_attn:
            x_q2, x_k2, x_v2 = x_qkv2
            attn2 = attention(x_q2, x_k2, x_v2, self.x_block.attn2.num_heads, mode=self.mode)
            x = self.x_block.post_attention_x(x_attn_out, attn2, *x_intermediates)
        else:
            x = self.x_block.post_attention(x_attn_out, *x_intermediates)

        if not self.context_block.pre_only:
            context = self.context_block.post_attention(ctx_attn_out, *ctx_intermediate)
        else:
            context = None

        return context, x

    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        else:
            return self._forward(*args, **kwargs)


class MMDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    # prepare pos_embed for latent size * 2
    POS_EMBED_MAX_RATIO = 1.5

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        depth: int = 28,
        # hidden_size: Optional[int] = None,
        # num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        adm_in_channels: Optional[int] = None,
        context_embedder_in_features: Optional[int] = None,
        context_embedder_out_features: Optional[int] = None,
        use_checkpoint: bool = False,
        register_length: int = 0,
        attn_mode: str = "torch",
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        out_channels: Optional[int] = None,
        pos_embed_scaling_factor: Optional[float] = None,
        pos_embed_offset: Optional[float] = None,
        pos_embed_max_size: Optional[int] = None,
        num_patches=None,
        qk_norm: Optional[str] = None,
        x_block_self_attn_layers: Optional[list[int]] = [],
        qkv_bias: bool = True,
        pos_emb_random_crop_rate: float = 0.0,
        use_scaled_pos_embed: bool = False,
        pos_embed_latent_sizes: Optional[list[int]] = None,
        model_type: str = "sd3m",
    ):
        super().__init__()
        self._model_type = model_type
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = default(out_channels, default_out_channels)
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size
        self.x_block_self_attn_layers = x_block_self_attn_layers
        self.pos_emb_random_crop_rate = pos_emb_random_crop_rate
        self.gradient_checkpointing = use_checkpoint

        # hidden_size = default(hidden_size, 64 * depth)
        # num_heads = default(num_heads, hidden_size // 64)

        # apply magic --> this defines a head_size of 64
        self.hidden_size = 64 * depth
        num_heads = depth

        self.num_heads = num_heads

        self.enable_scaled_pos_embed(use_scaled_pos_embed, pos_embed_latent_sizes)

        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            in_channels,
            self.hidden_size,
            bias=True,
            strict_img_size=self.pos_embed_max_size is None,
        )
        self.t_embedder = TimestepEmbedding(self.hidden_size)

        self.y_embedder = None
        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = Embedder(adm_in_channels, self.hidden_size)

        if context_embedder_in_features is not None:
            self.context_embedder = nn.Linear(context_embedder_in_features, context_embedder_out_features)
        else:
            self.context_embedder = nn.Identity()

        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, self.hidden_size))

        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # just use a buffer already
        if num_patches is not None:
            self.register_buffer(
                "pos_embed",
                torch.empty(1, num_patches, self.hidden_size),
            )
        else:
            self.pos_embed = None

        self.use_checkpoint = use_checkpoint
        self.joint_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    self.hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_mode=attn_mode,
                    qkv_bias=qkv_bias,
                    pre_only=i == depth - 1,
                    rmsnorm=rmsnorm,
                    scale_mod_only=scale_mod_only,
                    swiglu=swiglu,
                    qk_norm=qk_norm,
                    x_block_self_attn=(i in self.x_block_self_attn_layers),
                )
                for i in range(depth)
            ]
        )
        for block in self.joint_blocks:
            block.gradient_checkpointing = use_checkpoint

        self.final_layer = UnPatch(self.hidden_size, patch_size, self.out_channels)
        # self.initialize_weights()

        self.blocks_to_swap = None
        self.offloader = None
        self.num_blocks = len(self.joint_blocks)

    def enable_scaled_pos_embed(self, use_scaled_pos_embed: bool, latent_sizes: Optional[list[int]]):
        self.use_scaled_pos_embed = use_scaled_pos_embed

        if self.use_scaled_pos_embed:
            # # remove pos_embed to free up memory up to 0.4 GB -> this causes error because pos_embed is not saved
            # self.pos_embed = None
            # move pos_embed to CPU to free up memory up to 0.4 GB
            self.pos_embed = self.pos_embed.cpu()

            # remove duplicates and sort latent sizes in ascending order
            latent_sizes = list(set(latent_sizes))
            latent_sizes = sorted(latent_sizes)

            patched_sizes = [latent_size // self.patch_size for latent_size in latent_sizes]

            # calculate value range for each latent area: this is used to determine the pos_emb size from the latent shape
            max_areas = []
            for i in range(1, len(patched_sizes)):
                prev_area = patched_sizes[i - 1] ** 2
                area = patched_sizes[i] ** 2
                max_areas.append((prev_area + area) // 2)

            # area of the last latent size, if the latent size exceeds this, error will be raised
            max_areas.append(int((patched_sizes[-1] * MMDiT.POS_EMBED_MAX_RATIO) ** 2))
            # print("max_areas", max_areas)

            self.resolution_area_to_latent_size = [(area, latent_size) for area, latent_size in zip(max_areas, patched_sizes)]

            self.resolution_pos_embeds = {}
            for patched_size in patched_sizes:
                grid_size = int(patched_size * MMDiT.POS_EMBED_MAX_RATIO)
                pos_embed = get_scaled_2d_sincos_pos_embed(self.hidden_size, grid_size, sample_size=patched_size)
                pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
                self.resolution_pos_embeds[patched_size] = pos_embed
                # print(f"pos_embed for {patched_size}x{patched_size} latent size: {pos_embed.shape}")

        else:
            self.resolution_area_to_latent_size = None
            self.resolution_pos_embeds = None

    @property
    def model_type(self):
        return self._model_type

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        for block in self.joint_blocks:
            block.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        for block in self.joint_blocks:
            block.disable_gradient_checkpointing()

    def initialize_weights(self):
        # TODO: Init context_embedder?
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        if self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.pos_embed.shape[-2] ** 0.5),
                scaling_factor=self.pos_embed_scaling_factor,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if getattr(self, "y_embedder", None) is not None:
            nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.x_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.x_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.context_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.context_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def set_pos_emb_random_crop_rate(self, rate: float):
        self.pos_emb_random_crop_rate = rate

    def cropped_pos_embed(self, h, w, device=None, random_crop: bool = False):
        p = self.x_embedder.patch_size
        # patched size
        h = (h + 1) // p
        w = (w + 1) // p
        if self.pos_embed is None:  # should not happen
            return get_2d_sincos_pos_embed_torch(self.hidden_size, w, h, device=device)
        assert self.pos_embed_max_size is not None
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)

        if not random_crop:
            top = (self.pos_embed_max_size - h) // 2
            left = (self.pos_embed_max_size - w) // 2
        else:
            top = torch.randint(0, self.pos_embed_max_size - h + 1, (1,)).item()
            left = torch.randint(0, self.pos_embed_max_size - w + 1, (1,)).item()

        spatial_pos_embed = self.pos_embed.reshape(
            1,
            self.pos_embed_max_size,
            self.pos_embed_max_size,
            self.pos_embed.shape[-1],
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def cropped_scaled_pos_embed(self, h, w, device=None, dtype=None, random_crop: bool = False):
        p = self.x_embedder.patch_size
        # patched size
        h = (h + 1) // p
        w = (w + 1) // p

        # select pos_embed size based on area
        area = h * w
        patched_size = None
        for area_, patched_size_ in self.resolution_area_to_latent_size:
            if area <= area_:
                patched_size = patched_size_
                break
        if patched_size is None:
            # raise ValueError(f"Area {area} is too large for the given latent sizes {self.resolution_area_to_latent_size}.")
            # use largest latent size
            patched_size = self.resolution_area_to_latent_size[-1][1]

        pos_embed = self.resolution_pos_embeds[patched_size]
        pos_embed_size = round(math.sqrt(pos_embed.shape[1]))  # max size, patched_size * POS_EMBED_MAX_RATIO
        if h > pos_embed_size or w > pos_embed_size:
            # # fallback to normal pos_embed
            # return self.cropped_pos_embed(h * p, w * p, device=device, random_crop=random_crop)
            # extend pos_embed size
            logger.warning(
                f"Add new pos_embed for size {h}x{w} as it exceeds the scaled pos_embed size {pos_embed_size}. Image is too tall or wide."
            )
            patched_size = max(h, w)
            grid_size = int(patched_size * MMDiT.POS_EMBED_MAX_RATIO)
            pos_embed_size = grid_size
            pos_embed = get_scaled_2d_sincos_pos_embed(self.hidden_size, grid_size, sample_size=patched_size)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
            self.resolution_pos_embeds[patched_size] = pos_embed
            logger.info(f"Added pos_embed for size {patched_size}x{patched_size}")

            # print(torch.allclose(pos_embed.to(torch.float32).cpu(), self.pos_embed.to(torch.float32).cpu(), atol=5e-2))
            # diff = pos_embed.to(torch.float32).cpu() - self.pos_embed.to(torch.float32).cpu()
            # print(diff.abs().max(), diff.abs().mean())

            # insert to resolution_area_to_latent_size, by adding and sorting
            area = pos_embed_size**2
            self.resolution_area_to_latent_size.append((area, patched_size))
            self.resolution_area_to_latent_size = sorted(self.resolution_area_to_latent_size)

        if not random_crop:
            top = (pos_embed_size - h) // 2
            left = (pos_embed_size - w) // 2
        else:
            top = torch.randint(0, pos_embed_size - h + 1, (1,)).item()
            left = torch.randint(0, pos_embed_size - w + 1, (1,)).item()

        if pos_embed.device != device:
            pos_embed = pos_embed.to(device)
            # which is better to update device, or transfer every time to device? -> 64x64 emb is 96*96*1536*4=56MB. It's okay to update device.
            self.resolution_pos_embeds[patched_size] = pos_embed  # update device
        if pos_embed.dtype != dtype:
            pos_embed = pos_embed.to(dtype)
            self.resolution_pos_embeds[patched_size] = pos_embed  # update dtype

        spatial_pos_embed = pos_embed.reshape(1, pos_embed_size, pos_embed_size, pos_embed.shape[-1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        # print(
        #     f"patched size: {h}x{w}, pos_embed size: {pos_embed_size}, pos_embed shape: {pos_embed.shape}, top: {top}, left: {left}"
        # )
        return spatial_pos_embed

    def enable_block_swap(self, num_blocks: int, device: torch.device):
        self.blocks_to_swap = num_blocks

        assert (
            self.blocks_to_swap <= self.num_blocks - 2
        ), f"Cannot swap more than {self.num_blocks - 2} blocks. Requested: {self.blocks_to_swap} blocks."

        self.offloader = custom_offloading_utils.ModelOffloader(
            self.joint_blocks, self.num_blocks, self.blocks_to_swap, device  # , debug=True
        )
        print(f"SD3: Block swap enabled. Swapping {num_blocks} blocks, total blocks: {self.num_blocks}, device: {device}.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            save_blocks = self.joint_blocks
            self.joint_blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.joint_blocks = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.joint_blocks)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, D) tensor of class labels
        """
        pos_emb_random_crop = (
            False if self.pos_emb_random_crop_rate == 0.0 else torch.rand(1).item() < self.pos_emb_random_crop_rate
        )

        B, C, H, W = x.shape

        # x = self.x_embedder(x) + self.cropped_pos_embed(H, W, device=x.device, random_crop=pos_emb_random_crop).to(dtype=x.dtype)
        if not self.use_scaled_pos_embed:
            pos_embed = self.cropped_pos_embed(H, W, device=x.device, random_crop=pos_emb_random_crop).to(dtype=x.dtype)
        else:
            # print(f"Using scaled pos_embed for size {H}x{W}")
            pos_embed = self.cropped_scaled_pos_embed(H, W, device=x.device, dtype=x.dtype, random_crop=pos_emb_random_crop)
        x = self.x_embedder(x) + pos_embed
        del pos_embed

        c = self.t_embedder(t, dtype=x.dtype)  # (N, D)
        if y is not None and self.y_embedder is not None:
            y = self.y_embedder(y)  # (N, D)
            c = c + y  # (N, D)

        if context is not None:
            context = self.context_embedder(context)

        if self.register_length > 0:
            context = torch.cat(
                (einops.repeat(self.register, "1 ... -> b ...", b=x.shape[0]), default(context, torch.Tensor([]).type_as(x))), 1
            )

        if not self.blocks_to_swap:
            for block in self.joint_blocks:
                context, x = block(context, x, c)
        else:
            for block_idx, block in enumerate(self.joint_blocks):
                self.offloader.wait_for_block(block_idx)

                context, x = block(context, x, c)

                self.offloader.submit_move_blocks(self.joint_blocks, block_idx)

        x = self.final_layer(x, c, H, W)  # Our final layer combined UnPatchify
        return x[:, :, :H, :W]


def create_sd3_mmdit(params: SD3Params, attn_mode: str = "torch") -> MMDiT:
    mmdit = MMDiT(
        input_size=None,
        pos_embed_max_size=params.pos_embed_max_size,
        patch_size=params.patch_size,
        in_channels=16,
        adm_in_channels=params.adm_in_channels,
        context_embedder_in_features=params.context_embedder_in_features,
        context_embedder_out_features=params.context_embedder_out_features,
        depth=params.depth,
        mlp_ratio=4,
        qk_norm=params.qk_norm,
        x_block_self_attn_layers=params.x_block_self_attn_layers,
        num_patches=params.num_patches,
        attn_mode=attn_mode,
        model_type=params.model_type,
    )
    return mmdit


# endregion

# region VAE

VAE_SCALE_FACTOR = 1.5305
VAE_SHIFT_FACTOR = 0.0609


def Normalize(in_channels, num_groups=32, dtype=torch.float32, device=None):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)


class ResnetBlock(torch.nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dtype=torch.float32, device=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, dtype=dtype, device=device)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.norm2 = Normalize(out_channels, dtype=dtype, device=device)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device
            )
        else:
            self.nin_shortcut = None
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        hidden = x
        hidden = self.norm1(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv2(hidden)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden


class AttnBlock(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)

    def forward(self, x):
        hidden = self.norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        b, c, h, w = q.shape
        q, k, v = map(lambda x: einops.rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v))
        hidden = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        hidden = einops.rearrange(hidden, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        hidden = self.proj_out(hidden)
        return x + hidden


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0, dtype=dtype, device=device)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)

    def forward(self, x):
        org_dtype = x.dtype
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if x.dtype != org_dtype:
            x = x.to(org_dtype)
        x = self.conv(x)
        return x


class VAEEncoder(torch.nn.Module):
    def __init__(
        self, ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2, in_channels=3, z_channels=16, dtype=torch.float32, device=None
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = torch.nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                block_in = block_out
            down = torch.nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, dtype=dtype, device=device)
            self.down.append(down)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = self.swish(h)
        h = self.conv_out(h)
        return h


class VAEDecoder(torch.nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        resolution=256,
        z_channels=16,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        # upsampling
        self.up = torch.nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = torch.nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                block_in = block_out
            up = torch.nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, dtype=dtype, device=device)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, z):
        # z to block_in
        hidden = self.conv_in(z)
        # middle
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hidden = self.up[i_level].block[i_block](hidden)
            if i_level != 0:
                hidden = self.up[i_level].upsample(hidden)
        # end
        hidden = self.norm_out(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv_out(hidden)
        return hidden


class SDVAE(torch.nn.Module):
    def __init__(self, dtype=torch.float32, device=None):
        super().__init__()
        self.encoder = VAEEncoder(dtype=dtype, device=device)
        self.decoder = VAEDecoder(dtype=dtype, device=device)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    # @torch.autocast("cuda", dtype=torch.float16)
    def decode(self, latent):
        return self.decoder(latent)

    # @torch.autocast("cuda", dtype=torch.float16)
    def encode(self, image):
        hidden = self.encoder(image)
        mean, logvar = torch.chunk(hidden, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)

    @staticmethod
    def process_in(latent):
        return (latent - VAE_SHIFT_FACTOR) * VAE_SCALE_FACTOR

    @staticmethod
    def process_out(latent):
        return (latent / VAE_SCALE_FACTOR) + VAE_SHIFT_FACTOR


# endregion
