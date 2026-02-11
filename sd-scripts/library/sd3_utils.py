from dataclasses import dataclass
import math
import re
from typing import Dict, List, Optional, Union
import torch
import safetensors
from safetensors.torch import load_file
from accelerate import init_empty_weights
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPConfig, CLIPTextConfig

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import sd3_models

# TODO move some of functions to model_util.py
from library import sdxl_model_util

# region models

# TODO remove dependency on flux_utils
from library.utils import load_safetensors
from library.flux_utils import load_t5xxl as flux_utils_load_t5xxl


def analyze_state_dict_state(state_dict: Dict, prefix: str = ""):
    logger.info(f"Analyzing state dict state...")

    # analyze configs
    patch_size = state_dict[f"{prefix}x_embedder.proj.weight"].shape[2]
    depth = state_dict[f"{prefix}x_embedder.proj.weight"].shape[0] // 64
    num_patches = state_dict[f"{prefix}pos_embed"].shape[1]
    pos_embed_max_size = round(math.sqrt(num_patches))
    adm_in_channels = state_dict[f"{prefix}y_embedder.mlp.0.weight"].shape[1]
    context_shape = state_dict[f"{prefix}context_embedder.weight"].shape
    qk_norm = "rms" if f"{prefix}joint_blocks.0.context_block.attn.ln_k.weight" in state_dict.keys() else None

    #  x_block_self_attn_layers.append(int(key.split(".x_block.attn2.ln_k.weight")[0].split(".")[-1]))
    x_block_self_attn_layers = []
    re_attn = re.compile(r"\.(\d+)\.x_block\.attn2\.ln_k\.weight")
    for key in list(state_dict.keys()):
        m = re_attn.search(key)
        if m:
            x_block_self_attn_layers.append(int(m.group(1)))

    context_embedder_in_features = context_shape[1]
    context_embedder_out_features = context_shape[0]

    # only supports 3-5-large, medium or 3-medium
    if qk_norm is not None:
        if len(x_block_self_attn_layers) == 0:
            model_type = "3-5-large"
        else:
            model_type = "3-5-medium"
    else:
        model_type = "3-medium"

    params = sd3_models.SD3Params(
        patch_size=patch_size,
        depth=depth,
        num_patches=num_patches,
        pos_embed_max_size=pos_embed_max_size,
        adm_in_channels=adm_in_channels,
        qk_norm=qk_norm,
        x_block_self_attn_layers=x_block_self_attn_layers,
        context_embedder_in_features=context_embedder_in_features,
        context_embedder_out_features=context_embedder_out_features,
        model_type=model_type,
    )
    logger.info(f"Analyzed state dict state: {params}")
    return params


def load_mmdit(
    state_dict: Dict, dtype: Optional[Union[str, torch.dtype]], device: Union[str, torch.device], attn_mode: str = "torch"
) -> sd3_models.MMDiT:
    mmdit_sd = {}

    mmdit_prefix = "model.diffusion_model."
    for k in list(state_dict.keys()):
        if k.startswith(mmdit_prefix):
            mmdit_sd[k[len(mmdit_prefix) :]] = state_dict.pop(k)

    # load MMDiT
    logger.info("Building MMDit")
    params = analyze_state_dict_state(mmdit_sd)
    with init_empty_weights():
        mmdit = sd3_models.create_sd3_mmdit(params, attn_mode)

    logger.info("Loading state dict...")
    info = mmdit.load_state_dict(mmdit_sd, strict=False, assign=True)
    logger.info(f"Loaded MMDiT: {info}")
    return mmdit


def load_clip_l(
    clip_l_path: Optional[str],
    dtype: Optional[Union[str, torch.dtype]],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[Dict] = None,
):
    clip_l_sd = None
    if clip_l_path is None:
        if "text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight" in state_dict:
            # found clip_l: remove prefix "text_encoders.clip_l."
            logger.info("clip_l is included in the checkpoint")
            clip_l_sd = {}
            prefix = "text_encoders.clip_l."
            for k in list(state_dict.keys()):
                if k.startswith(prefix):
                    clip_l_sd[k[len(prefix) :]] = state_dict.pop(k)
        elif clip_l_path is None:
            logger.info("clip_l is not included in the checkpoint and clip_l_path is not provided")
            return None

    # load clip_l
    logger.info("Building CLIP-L")
    config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=768,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    with init_empty_weights():
        clip = CLIPTextModelWithProjection(config)

    if clip_l_sd is None:
        logger.info(f"Loading state dict from {clip_l_path}")
        clip_l_sd = load_safetensors(clip_l_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

    if "text_projection.weight" not in clip_l_sd:
        logger.info("Adding text_projection.weight to clip_l_sd")
        clip_l_sd["text_projection.weight"] = torch.eye(768, dtype=dtype, device=device)

    info = clip.load_state_dict(clip_l_sd, strict=False, assign=True)
    logger.info(f"Loaded CLIP-L: {info}")
    return clip


def load_clip_g(
    clip_g_path: Optional[str],
    dtype: Optional[Union[str, torch.dtype]],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[Dict] = None,
):
    clip_g_sd = None
    if state_dict is not None:
        if "text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight" in state_dict:
            # found clip_g: remove prefix "text_encoders.clip_g."
            logger.info("clip_g is included in the checkpoint")
            clip_g_sd = {}
            prefix = "text_encoders.clip_g."
            for k in list(state_dict.keys()):
                if k.startswith(prefix):
                    clip_g_sd[k[len(prefix) :]] = state_dict.pop(k)
        elif clip_g_path is None:
            logger.info("clip_g is not included in the checkpoint and clip_g_path is not provided")
            return None

    # load clip_g
    logger.info("Building CLIP-G")
    config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        max_position_embeddings=77,
        hidden_act="gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=1280,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    with init_empty_weights():
        clip = CLIPTextModelWithProjection(config)

    if clip_g_sd is None:
        logger.info(f"Loading state dict from {clip_g_path}")
        clip_g_sd = load_safetensors(clip_g_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = clip.load_state_dict(clip_g_sd, strict=False, assign=True)
    logger.info(f"Loaded CLIP-G: {info}")
    return clip


def load_t5xxl(
    t5xxl_path: Optional[str],
    dtype: Optional[Union[str, torch.dtype]],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[Dict] = None,
):
    t5xxl_sd = None
    if state_dict is not None:
        if "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.k.weight" in state_dict:
            # found t5xxl: remove prefix "text_encoders.t5xxl."
            logger.info("t5xxl is included in the checkpoint")
            t5xxl_sd = {}
            prefix = "text_encoders.t5xxl."
            for k in list(state_dict.keys()):
                if k.startswith(prefix):
                    t5xxl_sd[k[len(prefix) :]] = state_dict.pop(k)
        elif t5xxl_path is None:
            logger.info("t5xxl is not included in the checkpoint and t5xxl_path is not provided")
            return None

    return flux_utils_load_t5xxl(t5xxl_path, dtype, device, disable_mmap, state_dict=t5xxl_sd)


def load_vae(
    vae_path: Optional[str],
    vae_dtype: Optional[Union[str, torch.dtype]],
    device: Optional[Union[str, torch.device]],
    disable_mmap: bool = False,
    state_dict: Optional[Dict] = None,
):
    vae_sd = {}
    if vae_path:
        logger.info(f"Loading VAE from {vae_path}...")
        vae_sd = load_safetensors(vae_path, device, disable_mmap)
    else:
        # remove prefix "first_stage_model."
        vae_sd = {}
        vae_prefix = "first_stage_model."
        for k in list(state_dict.keys()):
            if k.startswith(vae_prefix):
                vae_sd[k[len(vae_prefix) :]] = state_dict.pop(k)

    logger.info("Building VAE")
    vae = sd3_models.SDVAE(vae_dtype, device)
    logger.info("Loading state dict...")
    info = vae.load_state_dict(vae_sd)
    logger.info(f"Loaded VAE: {info}")
    vae.to(device=device, dtype=vae_dtype)  # make sure it's in the right device and dtype
    return vae


# endregion


class ModelSamplingDiscreteFlow:
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""

    def __init__(self, shift=1.0):
        self.shift = shift
        timesteps = 1000
        self.sigmas = self.sigma(torch.arange(1, timesteps + 1, 1))

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        # assert max_denoise is False, "max_denoise not implemented"
        # max_denoise is always True, I'm not sure why it's there
        return sigma * noise + (1.0 - sigma) * latent_image
