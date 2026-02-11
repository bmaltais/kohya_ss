from functools import wraps
import torch
import diffusers # pylint: disable=import-error
from diffusers.utils import torch_utils # pylint: disable=import-error, unused-import # noqa: F401

# pylint: disable=protected-access, missing-function-docstring, line-too-long


# Diffusers FreeU
# Diffusers is imported before ipex hijacks so fourier_filter needs hijacking too
original_fourier_filter = diffusers.utils.torch_utils.fourier_filter
@wraps(diffusers.utils.torch_utils.fourier_filter)
def fourier_filter(x_in, threshold, scale):
    return_dtype = x_in.dtype
    return original_fourier_filter(x_in.to(dtype=torch.float32), threshold, scale).to(dtype=return_dtype)


# fp64 error
class FluxPosEmbed(torch.nn.Module):
    def __init__(self, theta: int, axes_dim):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        for i in range(n_axes):
            cos, sin = diffusers.models.embeddings.get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=torch.float32,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


def hidream_rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."
    return_device = pos.device
    pos = pos.to("cpu")

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.to(return_device, dtype=torch.float32)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, output_type="np"):
    if output_type == "np":
        return diffusers.models.embeddings.get_1d_sincos_pos_embed_from_grid_np(embed_dim=embed_dim, pos=pos)
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def apply_rotary_emb(x, freqs_cis, use_real: bool = True, use_real_unbind_dim: int = -1):
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        # used for lumina
        # force cpu with Alchemist
        x_rotated = torch.view_as_complex(x.to("cpu").float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.to("cpu").unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x).to(x.device)


def ipex_diffusers(device_supports_fp64=False):
    diffusers.utils.torch_utils.fourier_filter = fourier_filter
    if not device_supports_fp64:
        # get around lazy imports
        from diffusers.models import embeddings as diffusers_embeddings # pylint: disable=import-error, unused-import # noqa: F401
        from diffusers.models import transformers as diffusers_transformers # pylint: disable=import-error, unused-import # noqa: F401
        from diffusers.models import controlnets as diffusers_controlnets # pylint: disable=import-error, unused-import # noqa: F401
        diffusers.models.embeddings.get_1d_sincos_pos_embed_from_grid = get_1d_sincos_pos_embed_from_grid
        diffusers.models.embeddings.FluxPosEmbed = FluxPosEmbed
        diffusers.models.embeddings.apply_rotary_emb = apply_rotary_emb
        diffusers.models.transformers.transformer_flux.FluxPosEmbed = FluxPosEmbed
        diffusers.models.transformers.transformer_lumina2.apply_rotary_emb = apply_rotary_emb
        diffusers.models.controlnets.controlnet_flux.FluxPosEmbed = FluxPosEmbed
        diffusers.models.transformers.transformer_hidream_image.rope = hidream_rope
