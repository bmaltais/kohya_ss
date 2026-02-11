import itertools
import json
from typing import Any, List, NamedTuple, Optional, Tuple, Union, Callable
import glob
import importlib
import importlib.util
import sys
import inspect
import time
import zipfile
from diffusers.utils import deprecate
from diffusers.configuration_utils import FrozenDict
import argparse
import math
import os
import random
import re

import diffusers
import numpy as np
import torch

from library.device_utils import init_ipex, clean_memory, get_preferred_device

init_ipex()

import torchvision
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    # UNet2DConditionModel,
    StableDiffusionPipeline,
)
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from accelerate import init_empty_weights
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import library.model_util as model_util
import library.train_util as train_util
import library.sdxl_model_util as sdxl_model_util
import library.sdxl_train_util as sdxl_train_util
from networks.lora import LoRANetwork
import tools.original_control_net as original_control_net
from tools.original_control_net import ControlNetInfo
from library.original_unet import UNet2DConditionModel, InferUNet2DConditionModel
from library.sdxl_original_unet import InferSdxlUNet2DConditionModel
from library.sdxl_original_control_net import SdxlControlNet
from library.original_unet import FlashAttentionFunction
from networks.control_net_lllite import ControlNetLLLite
from library.utils import GradualLatent, EulerAncestralDiscreteSchedulerGL
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

# その他の設定
LATENT_CHANNELS = 4
DOWNSAMPLING_FACTOR = 8

CLIP_VISION_MODEL = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""


# def replace_unet_modules(unet: diffusers.models.unets.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers, sdpa):
def replace_unet_modules(unet, mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        logger.info("Enable memory efficient attention for U-Net")

        # これはDiffusersのU-Netではなく自前のU-Netなので置き換えなくても良い
        unet.set_use_memory_efficient_attention(False, True)
    elif xformers:
        logger.info("Enable xformers for U-Net")
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("No xformers / xformersがインストールされていないようです")

        unet.set_use_memory_efficient_attention(True, False)
    elif sdpa:
        logger.info("Enable SDPA for U-Net")
        unet.set_use_memory_efficient_attention(False, False)
        unet.set_use_sdpa(True)


# TODO common train_util.py
def replace_vae_modules(vae: diffusers.models.AutoencoderKL, mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        replace_vae_attn_to_memory_efficient()
    elif xformers:
        # replace_vae_attn_to_xformers() # 解像度によってxformersがエラーを出す？
        vae.set_use_memory_efficient_attention_xformers(True)  # とりあえずこっちを使う
    elif sdpa:
        replace_vae_attn_to_sdpa()


def replace_vae_attn_to_memory_efficient():
    logger.info("VAE Attention.forward has been replaced to FlashAttention (not xformers)")
    flash_func = FlashAttentionFunction

    def forward_flash_attn(self, hidden_states, **kwargs):
        q_bucket_size = 512
        k_bucket_size = 1024

        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.to_q(hidden_states)
        key_proj = self.to_k(hidden_states)
        value_proj = self.to_v(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (query_proj, key_proj, value_proj)
        )

        out = flash_func.apply(query_proj, key_proj, value_proj, None, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, "b h n d -> b n (h d)")

        # compute next hidden_states
        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    def forward_flash_attn_0_14(self, hidden_states, **kwargs):
        if not hasattr(self, "to_q"):
            self.to_q = self.query
            self.to_k = self.key
            self.to_v = self.value
            self.to_out = [self.proj_attn, torch.nn.Identity()]
            self.heads = self.num_heads
        return forward_flash_attn(self, hidden_states, **kwargs)

    if diffusers.__version__ < "0.15.0":
        diffusers.models.attention.AttentionBlock.forward = forward_flash_attn_0_14
    else:
        diffusers.models.attention_processor.Attention.forward = forward_flash_attn


def replace_vae_attn_to_xformers():
    logger.info("VAE: Attention.forward has been replaced to xformers")
    import xformers.ops

    def forward_xformers(self, hidden_states, **kwargs):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.to_q(hidden_states)
        key_proj = self.to_k(hidden_states)
        value_proj = self.to_v(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (query_proj, key_proj, value_proj)
        )

        query_proj = query_proj.contiguous()
        key_proj = key_proj.contiguous()
        value_proj = value_proj.contiguous()
        out = xformers.ops.memory_efficient_attention(query_proj, key_proj, value_proj, attn_bias=None)

        out = rearrange(out, "b h n d -> b n (h d)")

        # compute next hidden_states
        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    def forward_xformers_0_14(self, hidden_states, **kwargs):
        if not hasattr(self, "to_q"):
            self.to_q = self.query
            self.to_k = self.key
            self.to_v = self.value
            self.to_out = [self.proj_attn, torch.nn.Identity()]
            self.heads = self.num_heads
        return forward_xformers(self, hidden_states, **kwargs)

    if diffusers.__version__ < "0.15.0":
        diffusers.models.attention.AttentionBlock.forward = forward_xformers_0_14
    else:
        diffusers.models.attention_processor.Attention.forward = forward_xformers


def replace_vae_attn_to_sdpa():
    logger.info("VAE: Attention.forward has been replaced to sdpa")

    def forward_sdpa(self, hidden_states, **kwargs):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.to_q(hidden_states)
        key_proj = self.to_k(hidden_states)
        value_proj = self.to_v(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.heads), (query_proj, key_proj, value_proj)
        )

        out = torch.nn.functional.scaled_dot_product_attention(
            query_proj, key_proj, value_proj, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        out = rearrange(out, "b n h d -> b n (h d)")

        # compute next hidden_states
        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    def forward_sdpa_0_14(self, hidden_states, **kwargs):
        if not hasattr(self, "to_q"):
            self.to_q = self.query
            self.to_k = self.key
            self.to_v = self.value
            self.to_out = [self.proj_attn, torch.nn.Identity()]
            self.heads = self.num_heads
        return forward_sdpa(self, hidden_states, **kwargs)

    if diffusers.__version__ < "0.15.0":
        diffusers.models.attention.AttentionBlock.forward = forward_sdpa_0_14
    else:
        diffusers.models.attention_processor.Attention.forward = forward_sdpa


# endregion

# region 画像生成の本体：lpw_stable_diffusion.py （ASL）からコピーして修正
# https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion.py
# Pipelineだけ独立して使えないのと機能追加するのとでコピーして修正


class PipelineLike:
    def __init__(
        self,
        is_sdxl,
        device,
        vae: AutoencoderKL,
        text_encoders: List[CLIPTextModel],
        tokenizers: List[CLIPTokenizer],
        unet: InferSdxlUNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        clip_skip: int,
    ):
        super().__init__()
        self.is_sdxl = is_sdxl
        self.device = device
        self.clip_skip = clip_skip

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.vae = vae
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.unet: Union[InferUNet2DConditionModel, InferSdxlUNet2DConditionModel] = unet
        self.scheduler = scheduler
        self.safety_checker = None

        self.clip_vision_model: CLIPVisionModelWithProjection = None
        self.clip_vision_processor: CLIPImageProcessor = None
        self.clip_vision_strength = 0.0

        # Textual Inversion
        self.token_replacements_list = []
        for _ in range(len(self.text_encoders)):
            self.token_replacements_list.append({})

        # ControlNet
        self.control_nets: List[Union[ControlNetInfo, Tuple[SdxlControlNet, float]]] = []
        self.control_net_lllites: List[Tuple[ControlNetLLLite, float]] = []
        self.control_net_enabled = True  # control_netsが空ならTrueでもFalseでもControlNetは動作しない

        self.gradual_latent: GradualLatent = None

    # Textual Inversion
    def add_token_replacement(self, text_encoder_index, target_token_id, rep_token_ids):
        self.token_replacements_list[text_encoder_index][target_token_id] = rep_token_ids

    def set_enable_control_net(self, en: bool):
        self.control_net_enabled = en

    def get_token_replacer(self, tokenizer):
        tokenizer_index = self.tokenizers.index(tokenizer)
        token_replacements = self.token_replacements_list[tokenizer_index]

        def replace_tokens(tokens):
            # print("replace_tokens", tokens, "=>", token_replacements)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()

            new_tokens = []
            for token in tokens:
                if token in token_replacements:
                    replacement = token_replacements[token]
                    new_tokens.extend(replacement)
                else:
                    new_tokens.append(token)
            return new_tokens

        return replace_tokens

    def set_control_nets(self, ctrl_nets):
        self.control_nets = ctrl_nets

    def set_control_net_lllites(self, ctrl_net_lllites):
        self.control_net_lllites = ctrl_net_lllites

    def set_gradual_latent(self, gradual_latent):
        if gradual_latent is None:
            logger.info("gradual_latent is disabled")
            self.gradual_latent = None
        else:
            logger.info(f"gradual_latent is enabled: {gradual_latent}")
            self.gradual_latent = gradual_latent  # (ds_ratio, start_timesteps, every_n_steps, ratio_step)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        init_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        height: int = 1024,
        width: int = 1024,
        original_height: int = None,
        original_width: int = None,
        original_height_negative: int = None,
        original_width_negative: int = None,
        crop_top: int = 0,
        crop_left: int = 0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_scale: float = None,
        strength: float = 0.8,
        # num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        vae_batch_size: float = None,
        return_latents: bool = False,
        # return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: Optional[int] = 1,
        img2img_noise=None,
        clip_guide_images=None,
        emb_normalize_mode: str = "original",
        **kwargs,
    ):
        # TODO support secondary prompt
        num_images_per_prompt = 1  # fixed because already prompt is repeated

        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        regional_network = " AND " in prompt[0]

        vae_batch_size = (
            batch_size
            if vae_batch_size is None
            else (int(vae_batch_size) if vae_batch_size >= 1 else max(1, int(batch_size * vae_batch_size)))
        )

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type" f" {type(callback_steps)}."
            )

        # get prompt text embeddings

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if not do_classifier_free_guidance and negative_scale is not None:
            logger.warning(f"negative_scale is ignored if guidance scalle <= 1.0")
            negative_scale = None

        # get unconditional embeddings for classifier free guidance
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        tes_text_embs = []
        tes_uncond_embs = []
        tes_real_uncond_embs = []

        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            token_replacer = self.get_token_replacer(tokenizer)

            # use last text_pool, because it is from text encoder 2
            text_embeddings, text_pool, uncond_embeddings, uncond_pool, _ = get_weighted_text_embeddings(
                self.is_sdxl,
                tokenizer,
                text_encoder,
                prompt=prompt,
                uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
                max_embeddings_multiples=max_embeddings_multiples,
                clip_skip=self.clip_skip,
                token_replacer=token_replacer,
                device=self.device,
                emb_normalize_mode=emb_normalize_mode,
                **kwargs,
            )
            tes_text_embs.append(text_embeddings)
            tes_uncond_embs.append(uncond_embeddings)

            if negative_scale is not None:
                _, real_uncond_embeddings, _ = get_weighted_text_embeddings(
                    self.is_sdxl,
                    token_replacer,
                    prompt=prompt,  # こちらのトークン長に合わせてuncondを作るので75トークン超で必須
                    uncond_prompt=[""] * batch_size,
                    max_embeddings_multiples=max_embeddings_multiples,
                    clip_skip=self.clip_skip,
                    token_replacer=token_replacer,
                    device=self.device,
                    emb_normalize_mode=emb_normalize_mode,
                    **kwargs,
                )
                tes_real_uncond_embs.append(real_uncond_embeddings)

        # concat text encoder outputs
        text_embeddings = tes_text_embs[0]
        uncond_embeddings = tes_uncond_embs[0]
        for i in range(1, len(tes_text_embs)):
            text_embeddings = torch.cat([text_embeddings, tes_text_embs[i]], dim=2)  # n,77,2048
            if do_classifier_free_guidance:
                uncond_embeddings = torch.cat([uncond_embeddings, tes_uncond_embs[i]], dim=2)  # n,77,2048

        if do_classifier_free_guidance:
            if negative_scale is None:
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            else:
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings, real_uncond_embeddings])

        if self.control_net_lllites or (self.control_nets and self.is_sdxl):
            # ControlNetのhintにguide imageを流用する。ControlNetの場合はControlNet側で行う
            if isinstance(clip_guide_images, PIL.Image.Image):
                clip_guide_images = [clip_guide_images]
            if isinstance(clip_guide_images[0], PIL.Image.Image):
                clip_guide_images = [preprocess_image(im) for im in clip_guide_images]
                clip_guide_images = torch.cat(clip_guide_images)
            if isinstance(clip_guide_images, list):
                clip_guide_images = torch.stack(clip_guide_images)

            clip_guide_images = clip_guide_images.to(self.device, dtype=text_embeddings.dtype)

        # create size embs
        if original_height is None:
            original_height = height
        if original_width is None:
            original_width = width
        if original_height_negative is None:
            original_height_negative = original_height
        if original_width_negative is None:
            original_width_negative = original_width
        if crop_top is None:
            crop_top = 0
        if crop_left is None:
            crop_left = 0
        if self.is_sdxl:
            emb1 = sdxl_train_util.get_timestep_embedding(torch.FloatTensor([original_height, original_width]).unsqueeze(0), 256)
            uc_emb1 = sdxl_train_util.get_timestep_embedding(
                torch.FloatTensor([original_height_negative, original_width_negative]).unsqueeze(0), 256
            )
            emb2 = sdxl_train_util.get_timestep_embedding(torch.FloatTensor([crop_top, crop_left]).unsqueeze(0), 256)
            emb3 = sdxl_train_util.get_timestep_embedding(torch.FloatTensor([height, width]).unsqueeze(0), 256)
            c_vector = torch.cat([emb1, emb2, emb3], dim=1).to(self.device, dtype=text_embeddings.dtype).repeat(batch_size, 1)
            uc_vector = torch.cat([uc_emb1, emb2, emb3], dim=1).to(self.device, dtype=text_embeddings.dtype).repeat(batch_size, 1)

            if regional_network:
                # use last pool for conditioning
                num_sub_prompts = len(text_pool) // batch_size
                text_pool = text_pool[num_sub_prompts - 1 :: num_sub_prompts]  # last subprompt

            if init_image is not None and self.clip_vision_model is not None:
                logger.info(f"encode by clip_vision_model and apply clip_vision_strength={self.clip_vision_strength}")
                vision_input = self.clip_vision_processor(init_image, return_tensors="pt", device=self.device)
                pixel_values = vision_input["pixel_values"].to(self.device, dtype=text_embeddings.dtype)

                clip_vision_embeddings = self.clip_vision_model(
                    pixel_values=pixel_values, output_hidden_states=True, return_dict=True
                )
                clip_vision_embeddings = clip_vision_embeddings.image_embeds

                if len(clip_vision_embeddings) == 1 and batch_size > 1:
                    clip_vision_embeddings = clip_vision_embeddings.repeat((batch_size, 1))

                clip_vision_embeddings = clip_vision_embeddings * self.clip_vision_strength
                assert clip_vision_embeddings.shape == text_pool.shape, f"{clip_vision_embeddings.shape} != {text_pool.shape}"
                text_pool = clip_vision_embeddings  # replace: same as ComfyUI (?)

            c_vector = torch.cat([text_pool, c_vector], dim=1)
            if do_classifier_free_guidance:
                uc_vector = torch.cat([uncond_pool, uc_vector], dim=1)
                vector_embeddings = torch.cat([uc_vector, c_vector])
            else:
                vector_embeddings = c_vector

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, self.device)

        latents_dtype = text_embeddings.dtype
        init_latents_orig = None
        mask = None

        if init_image is None:
            # get the initial random noise unless the user supplied it

            # Unlike in other pipelines, latents need to be generated in the target device
            # for 1-to-1 results reproducibility with the CompVis implementation.
            # However this currently doesn't work in `mps`.
            latents_shape = (
                batch_size * num_images_per_prompt,
                self.unet.in_channels,
                height // 8,
                width // 8,
            )

            if latents is None:
                if self.device.type == "mps":
                    # randn does not exist on mps
                    latents = torch.randn(
                        latents_shape,
                        generator=generator,
                        device="cpu",
                        dtype=latents_dtype,
                    ).to(self.device)
                else:
                    latents = torch.randn(
                        latents_shape,
                        generator=generator,
                        device=self.device,
                        dtype=latents_dtype,
                    )
            else:
                if latents.shape != latents_shape:
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
                latents = latents.to(self.device)

            timesteps = self.scheduler.timesteps.to(self.device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        else:
            # image to tensor
            if isinstance(init_image, PIL.Image.Image):
                init_image = [init_image]
            if isinstance(init_image[0], PIL.Image.Image):
                init_image = [preprocess_image(im) for im in init_image]
                init_image = torch.cat(init_image)
            if isinstance(init_image, list):
                init_image = torch.stack(init_image)

            # mask image to tensor
            if mask_image is not None:
                if isinstance(mask_image, PIL.Image.Image):
                    mask_image = [mask_image]
                if isinstance(mask_image[0], PIL.Image.Image):
                    mask_image = torch.cat([preprocess_mask(im) for im in mask_image])  # H*W, 0 for repaint

            # encode the init image into latents and scale the latents
            init_image = init_image.to(device=self.device, dtype=latents_dtype)
            if init_image.size()[-2:] == (height // 8, width // 8):
                init_latents = init_image
            else:
                if vae_batch_size >= batch_size:
                    init_latent_dist = self.vae.encode(init_image.to(self.vae.dtype)).latent_dist
                    init_latents = init_latent_dist.sample(generator=generator)
                else:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    init_latents = []
                    for i in tqdm(range(0, min(batch_size, len(init_image)), vae_batch_size)):
                        init_latent_dist = self.vae.encode(
                            (init_image[i : i + vae_batch_size] if vae_batch_size > 1 else init_image[i].unsqueeze(0)).to(
                                self.vae.dtype
                            )
                        ).latent_dist
                        init_latents.append(init_latent_dist.sample(generator=generator))
                    init_latents = torch.cat(init_latents)

                init_latents = (sdxl_model_util.VAE_SCALE_FACTOR if self.is_sdxl else 0.18215) * init_latents

            if len(init_latents) == 1:
                init_latents = init_latents.repeat((batch_size, 1, 1, 1))
            init_latents_orig = init_latents

            # preprocess mask
            if mask_image is not None:
                mask = mask_image.to(device=self.device, dtype=latents_dtype)
                if len(mask) == 1:
                    mask = mask.repeat((batch_size, 1, 1, 1))

                # check sizes
                if not mask.shape == init_latents.shape:
                    raise ValueError("The mask and init_image should be the same size!")

            # get the original timestep using init_timestep
            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)

            timesteps = self.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps] * batch_size * num_images_per_prompt, device=self.device)

            # add noise to latents using the timesteps
            latents = self.scheduler.add_noise(init_latents, img2img_noise, timesteps)

            t_start = max(num_inference_steps - init_timestep + offset, 0)
            timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        num_latent_input = (3 if negative_scale is not None else 2) if do_classifier_free_guidance else 1

        if self.control_nets:
            if not self.is_sdxl:
                guided_hints = original_control_net.get_guided_hints(
                    self.control_nets, num_latent_input, batch_size, clip_guide_images
                )
            else:
                clip_guide_images = clip_guide_images * 0.5 + 0.5  # [-1, 1] => [0, 1]
            each_control_net_enabled = [self.control_net_enabled] * len(self.control_nets)

        if self.control_net_lllites:
            # guided_hints = original_control_net.get_guided_hints(self.control_nets, num_latent_input, batch_size, clip_guide_images)
            if self.control_net_enabled:
                for control_net, _ in self.control_net_lllites:
                    with torch.no_grad():
                        control_net.set_cond_image(clip_guide_images)
            else:
                for control_net, _ in self.control_net_lllites:
                    control_net.set_cond_image(None)

            each_control_net_enabled = [self.control_net_enabled] * len(self.control_net_lllites)

        enable_gradual_latent = False
        if self.gradual_latent:
            if not hasattr(self.scheduler, "set_gradual_latent_params"):
                logger.warning("gradual_latent is not supported for this scheduler. Ignoring.")
                logger.warning(f"{self.scheduler.__class__.__name__}")
            else:
                enable_gradual_latent = True
                step_elapsed = 1000
                current_ratio = self.gradual_latent.ratio

                # first, we downscale the latents to the specified ratio / 最初に指定された比率にlatentsをダウンスケールする
                height, width = latents.shape[-2:]
                org_dtype = latents.dtype
                if org_dtype == torch.bfloat16:
                    latents = latents.float()
                latents = torch.nn.functional.interpolate(
                    latents, scale_factor=current_ratio, mode="bicubic", align_corners=False
                ).to(org_dtype)

                # apply unsharp mask / アンシャープマスクを適用する
                if self.gradual_latent.gaussian_blur_ksize:
                    latents = self.gradual_latent.apply_unshark_mask(latents)

        for i, t in enumerate(tqdm(timesteps)):
            resized_size = None
            if enable_gradual_latent:
                # gradually upscale the latents / latentsを徐々にアップスケールする
                if (
                    t < self.gradual_latent.start_timesteps
                    and current_ratio < 1.0
                    and step_elapsed >= self.gradual_latent.every_n_steps
                ):
                    current_ratio = min(current_ratio + self.gradual_latent.ratio_step, 1.0)
                    # make divisible by 8 because size of latents must be divisible at bottom of UNet
                    h = int(height * current_ratio) // 8 * 8
                    w = int(width * current_ratio) // 8 * 8
                    resized_size = (h, w)
                    self.scheduler.set_gradual_latent_params(resized_size, self.gradual_latent)
                    step_elapsed = 0
                else:
                    self.scheduler.set_gradual_latent_params(None, None)
                step_elapsed += 1

            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # disable ControlNet-LLLite or SDXL ControlNet if ratio is set. ControlNet is disabled in ControlNetInfo
            if self.control_net_lllites:
                for j, ((control_net, ratio), enabled) in enumerate(zip(self.control_net_lllites, each_control_net_enabled)):
                    if not enabled or ratio >= 1.0:
                        continue
                    if ratio < i / len(timesteps):
                        logger.info(f"ControlNetLLLite {j} is disabled (ratio={ratio} at {i} / {len(timesteps)})")
                        control_net.set_cond_image(None)
                        each_control_net_enabled[j] = False
            if self.control_nets and self.is_sdxl:
                for j, ((control_net, ratio), enabled) in enumerate(zip(self.control_nets, each_control_net_enabled)):
                    if not enabled or ratio >= 1.0:
                        continue
                    if ratio < i / len(timesteps):
                        logger.info(f"ControlNet {j} is disabled (ratio={ratio} at {i} / {len(timesteps)})")
                        each_control_net_enabled[j] = False

            # predict the noise residual
            if self.control_nets and self.control_net_enabled and not self.is_sdxl:
                if regional_network:
                    num_sub_and_neg_prompts = len(text_embeddings) // batch_size
                    text_emb_last = text_embeddings[num_sub_and_neg_prompts - 2 :: num_sub_and_neg_prompts]  # last subprompt
                else:
                    text_emb_last = text_embeddings

                noise_pred = original_control_net.call_unet_and_control_net(
                    i,
                    num_latent_input,
                    self.unet,
                    self.control_nets,
                    guided_hints,
                    i / len(timesteps),
                    latent_model_input,
                    t,
                    text_embeddings,
                    text_emb_last,
                ).sample
            elif self.control_nets:
                input_resi_add_list = []
                mid_add_list = []
                for (control_net, _), enbld in zip(self.control_nets, each_control_net_enabled):
                    if not enbld:
                        continue
                    input_resi_add, mid_add = control_net(
                        latent_model_input, t, text_embeddings, vector_embeddings, clip_guide_images
                    )
                    input_resi_add_list.append(input_resi_add)
                    mid_add_list.append(mid_add)
                if len(input_resi_add_list) == 0:
                    noise_pred = self.unet(latent_model_input, t, text_embeddings, vector_embeddings)
                else:
                    if len(input_resi_add_list) > 1:
                        # get mean of input_resi_add_list and mid_add_list
                        input_resi_add_mean = []
                        for i in range(len(input_resi_add_list[0])):
                            input_resi_add_mean.append(
                                torch.mean(torch.stack([input_resi_add_list[j][i] for j in range(len(input_resi_add_list))], dim=0))
                            )
                        input_resi_add = input_resi_add_mean
                        mid_add = torch.mean(torch.stack(mid_add_list), dim=0)
                        
                    noise_pred = self.unet(latent_model_input, t, text_embeddings, vector_embeddings, input_resi_add, mid_add)
            elif self.is_sdxl:
                noise_pred = self.unet(latent_model_input, t, text_embeddings, vector_embeddings)
            else:
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                if negative_scale is None:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred_negative, noise_pred_text, noise_pred_uncond = noise_pred.chunk(
                        num_latent_input
                    )  # uncond is real uncond
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        - negative_scale * (noise_pred_negative - noise_pred_uncond)
                    )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if mask is not None:
                # masking
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, img2img_noise, torch.tensor([t]))
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

            # call the callback, if provided
            if i % callback_steps == 0:
                if callback is not None:
                    callback(i, t, latents)
                if is_cancelled_callback is not None and is_cancelled_callback():
                    return None

        if return_latents:
            return latents

        latents = 1 / (sdxl_model_util.VAE_SCALE_FACTOR if self.is_sdxl else 0.18215) * latents
        if vae_batch_size >= batch_size:
            image = self.vae.decode(latents.to(self.vae.dtype)).sample
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            images = []
            for i in tqdm(range(0, batch_size, vae_batch_size)):
                images.append(
                    self.vae.decode(
                        (latents[i : i + vae_batch_size] if vae_batch_size > 1 else latents[i].unsqueeze(0)).to(self.vae.dtype)
                    ).sample
                )
            image = torch.cat(images)

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if output_type == "pil":
            # image = self.numpy_to_pil(image)
            image = (image * 255).round().astype("uint8")
            image = [Image.fromarray(im) for im in image]

        return image

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    # keep break as separate token
    text = text.replace("BREAK", "\\BREAK\\")

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1] and res[i][0].strip() != "BREAK" and res[i + 1][0].strip() != "BREAK":
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(tokenizer: CLIPTokenizer, token_replacer, prompt: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.
    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False

    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            if word.strip() == "BREAK":
                # pad until next multiple of tokenizer's max token length
                pad_len = tokenizer.model_max_length - (len(text_token) % tokenizer.model_max_length)
                logger.info(f"BREAK pad_len: {pad_len}")
                for i in range(pad_len):
                    # v2のときEOSをつけるべきかどうかわからないぜ
                    # if i == 0:
                    #     text_token.append(tokenizer.eos_token_id)
                    # else:
                    text_token.append(tokenizer.pad_token_id)
                    text_weight.append(1.0)
                continue

            # tokenize and discard the starting and the ending token
            token = tokenizer(word).input_ids[1:-1]

            token = token_replacer(token)  # for Textual Inversion

            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("warning: Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] + [pad] * (max_length - 2 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    is_sdxl: bool,
    text_encoder: CLIPTextModel,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        pool = None
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            if pad == eos:  # v1
                text_input_chunk[:, -1] = text_input[0, -1]
            else:  # v2
                for j in range(len(text_input_chunk)):
                    if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:  # 最後に普通の文字がある
                        text_input_chunk[j, -1] = eos
                    if text_input_chunk[j, 1] == pad:  # BOSだけであとはPAD
                        text_input_chunk[j, 1] = eos

            # in sdxl, value of clip_skip is same for Text Encoder 1 and 2
            enc_out = text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
            text_embedding = enc_out["hidden_states"][-clip_skip]
            if not is_sdxl:  # SD 1.5 requires final_layer_norm
                text_embedding = text_encoder.text_model.final_layer_norm(text_embedding)
            if pool is None:
                pool = enc_out.get("text_embeds", None)  # use 1st chunk, if provided
                if pool is not None:
                    pool = train_util.pool_workaround(text_encoder, enc_out["last_hidden_state"], text_input_chunk, eos)

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        enc_out = text_encoder(text_input, output_hidden_states=True, return_dict=True)
        text_embeddings = enc_out["hidden_states"][-clip_skip]
        if not is_sdxl:  # SD 1.5 requires final_layer_norm
            text_embeddings = text_encoder.text_model.final_layer_norm(text_embeddings)
        pool = enc_out.get("text_embeds", None)  # text encoder 1 doesn't return this
        if pool is not None:
            pool = train_util.pool_workaround(text_encoder, enc_out["last_hidden_state"], text_input, eos)
    return text_embeddings, pool


def get_weighted_text_embeddings(
    is_sdxl: bool,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 1,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
    clip_skip: int = 1,
    token_replacer=None,
    device=None,
    emb_normalize_mode: Optional[str] = "original",  # "original", "abs", "none"
    **kwargs,
):
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    # split the prompts with "AND". each prompt must have the same number of splits
    new_prompts = []
    for p in prompt:
        new_prompts.extend(p.split(" AND "))
    prompt = new_prompts

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(tokenizer, token_replacer, prompt, max_length - 2)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(tokenizer, token_replacer, uncond_prompt, max_length - 2)
    else:
        prompt_tokens = [token[1:-1] for token in tokenizer(prompt, max_length=max_length, truncation=True).input_ids]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [token[1:-1] for token in tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=tokenizer.model_max_length,
        )
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

    # get the embeddings
    text_embeddings, text_pool = get_unweighted_text_embeddings(
        is_sdxl,
        text_encoder,
        prompt_tokens,
        tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
    )

    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=device)
    if uncond_prompt is not None:
        uncond_embeddings, uncond_pool = get_unweighted_text_embeddings(
            is_sdxl,
            text_encoder,
            uncond_tokens,
            tokenizer.model_max_length,
            clip_skip,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
        )
        uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=device)

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    # →全体でいいんじゃないかな

    if (not skip_parsing) and (not skip_weighting):
        if emb_normalize_mode == "abs":
            previous_mean = text_embeddings.float().abs().mean(axis=[-2, -1]).to(text_embeddings.dtype)
            text_embeddings *= prompt_weights.unsqueeze(-1)
            current_mean = text_embeddings.float().abs().mean(axis=[-2, -1]).to(text_embeddings.dtype)
            text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
            if uncond_prompt is not None:
                previous_mean = uncond_embeddings.float().abs().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
                uncond_embeddings *= uncond_weights.unsqueeze(-1)
                current_mean = uncond_embeddings.float().abs().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
                uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

        elif emb_normalize_mode == "none":
            text_embeddings *= prompt_weights.unsqueeze(-1)
            if uncond_prompt is not None:
                uncond_embeddings *= uncond_weights.unsqueeze(-1)

        else:  # "original"
            previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
            text_embeddings *= prompt_weights.unsqueeze(-1)
            current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
            text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
            if uncond_prompt is not None:
                previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
                uncond_embeddings *= uncond_weights.unsqueeze(-1)
                current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
                uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, text_pool, uncond_embeddings, uncond_pool, prompt_tokens
    return text_embeddings, text_pool, None, None, prompt_tokens


def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.BILINEAR)  # LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask


# regular expression for dynamic prompt:
# starts and ends with "{" and "}"
# contains at least one variant divided by "|"
# optional framgments divided by "$$" at start
# if the first fragment is "E" or "e", enumerate all variants
# if the second fragment is a number or two numbers, repeat the variants in the range
# if the third fragment is a string, use it as a separator

RE_DYNAMIC_PROMPT = re.compile(r"\{((e|E)\$\$)?(([\d\-]+)\$\$)?(([^\|\}]+?)\$\$)?(.+?((\|).+?)*?)\}")


def handle_dynamic_prompt_variants(prompt, repeat_count):
    founds = list(RE_DYNAMIC_PROMPT.finditer(prompt))
    if not founds:
        return [prompt]

    # make each replacement for each variant
    enumerating = False
    replacers = []
    for found in founds:
        # if "e$$" is found, enumerate all variants
        found_enumerating = found.group(2) is not None
        enumerating = enumerating or found_enumerating

        separator = ", " if found.group(6) is None else found.group(6)
        variants = found.group(7).split("|")

        # parse count range
        count_range = found.group(4)
        if count_range is None:
            count_range = [1, 1]
        else:
            count_range = count_range.split("-")
            if len(count_range) == 1:
                count_range = [int(count_range[0]), int(count_range[0])]
            elif len(count_range) == 2:
                count_range = [int(count_range[0]), int(count_range[1])]
            else:
                logger.warning(f"invalid count range: {count_range}")
                count_range = [1, 1]
            if count_range[0] > count_range[1]:
                count_range = [count_range[1], count_range[0]]
            if count_range[0] < 0:
                count_range[0] = 0
            if count_range[1] > len(variants):
                count_range[1] = len(variants)

        if found_enumerating:
            # make function to enumerate all combinations
            def make_replacer_enum(vari, cr, sep):
                def replacer():
                    values = []
                    for count in range(cr[0], cr[1] + 1):
                        for comb in itertools.combinations(vari, count):
                            values.append(sep.join(comb))
                    return values

                return replacer

            replacers.append(make_replacer_enum(variants, count_range, separator))
        else:
            # make function to choose random combinations
            def make_replacer_single(vari, cr, sep):
                def replacer():
                    count = random.randint(cr[0], cr[1])
                    comb = random.sample(vari, count)
                    return [sep.join(comb)]

                return replacer

            replacers.append(make_replacer_single(variants, count_range, separator))

    # make each prompt
    if not enumerating:
        # if not enumerating, repeat the prompt, replace each variant randomly
        prompts = []
        for _ in range(repeat_count):
            current = prompt
            for found, replacer in zip(founds, replacers):
                current = current.replace(found.group(0), replacer()[0], 1)
            prompts.append(current)
    else:
        # if enumerating, iterate all combinations for previous prompts
        prompts = [prompt]

        for found, replacer in zip(founds, replacers):
            if found.group(2) is not None:
                # make all combinations for existing prompts
                new_prompts = []
                for current in prompts:
                    replecements = replacer()
                    for replecement in replecements:
                        new_prompts.append(current.replace(found.group(0), replecement, 1))
                prompts = new_prompts

        for found, replacer in zip(founds, replacers):
            # make random selection for existing prompts
            if found.group(2) is None:
                for i in range(len(prompts)):
                    prompts[i] = prompts[i].replace(found.group(0), replacer()[0], 1)

    return prompts


# endregion

# def load_clip_l14_336(dtype):
#   print(f"loading CLIP: {CLIP_ID_L14_336}")
#   text_encoder = CLIPTextModel.from_pretrained(CLIP_ID_L14_336, torch_dtype=dtype)
#   return text_encoder


class BatchDataBase(NamedTuple):
    # バッチ分割が必要ないデータ
    step: int
    prompt: str
    negative_prompt: str
    seed: int
    init_image: Any
    mask_image: Any
    clip_prompt: str
    guide_image: Any
    raw_prompt: str
    file_name: Optional[str]


class BatchDataExt(NamedTuple):
    # バッチ分割が必要なデータ
    width: int
    height: int
    original_width: int
    original_height: int
    original_width_negative: int
    original_height_negative: int
    crop_left: int
    crop_top: int
    steps: int
    scale: float
    negative_scale: float
    strength: float
    network_muls: Tuple[float]
    num_sub_prompts: int


class BatchData(NamedTuple):
    return_latents: bool
    base: BatchDataBase
    ext: BatchDataExt


class ListPrompter:
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
        self.index = 0

    def shuffle(self):
        random.shuffle(self.prompts)

    def __len__(self):
        return len(self.prompts)

    def __call__(self, *args, **kwargs):
        if self.index >= len(self.prompts):
            self.index = 0  # reset
            return None

        prompt = self.prompts[self.index]
        self.index += 1
        return prompt


def main(args):
    if args.fp16:
        dtype = torch.float16
    elif args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    highres_fix = args.highres_fix_scale is not None
    # assert not highres_fix or args.image_path is None, f"highres_fix doesn't work with img2img / highres_fixはimg2imgと同時に使えません"

    if args.v2 and args.clip_skip is not None:
        logger.warning("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

    # モデルを読み込む
    if not os.path.exists(args.ckpt):  # ファイルがないならパターンで探し、一つだけ該当すればそれを使う
        files = glob.glob(args.ckpt)
        if len(files) == 1:
            args.ckpt = files[0]

    name_or_path = os.readlink(args.ckpt) if os.path.islink(args.ckpt) else args.ckpt
    use_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers

    # SDXLかどうかを判定する
    is_sdxl = args.sdxl
    if not is_sdxl and not args.v1 and not args.v2:  # どれも指定されていない場合は自動で判定する
        if use_stable_diffusion_format:
            # if file size > 5.5GB, sdxl
            is_sdxl = os.path.getsize(name_or_path) > 5.5 * 1024**3
        else:
            # if `text_encoder_2` subdirectory exists, sdxl
            is_sdxl = os.path.isdir(os.path.join(name_or_path, "text_encoder_2"))
    logger.info(f"SDXL: {is_sdxl}")

    if is_sdxl:
        if args.clip_skip is None:
            args.clip_skip = 2

        (_, text_encoder1, text_encoder2, vae, unet, _, _) = sdxl_train_util._load_target_model(
            args.ckpt, args.vae, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, dtype
        )
        unet: InferSdxlUNet2DConditionModel = InferSdxlUNet2DConditionModel(unet)
        text_encoders = [text_encoder1, text_encoder2]
    else:
        if args.clip_skip is None:
            args.clip_skip = 2 if args.v2 else 1

        if use_stable_diffusion_format:
            logger.info("load StableDiffusion checkpoint")
            text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.ckpt)
        else:
            logger.info("load Diffusers pretrained models")
            loading_pipe = StableDiffusionPipeline.from_pretrained(args.ckpt, safety_checker=None, torch_dtype=dtype)
            text_encoder = loading_pipe.text_encoder
            vae = loading_pipe.vae
            unet = loading_pipe.unet
            tokenizer = loading_pipe.tokenizer
            del loading_pipe

            # Diffusers U-Net to original U-Net
            original_unet = UNet2DConditionModel(
                unet.config.sample_size,
                unet.config.attention_head_dim,
                unet.config.cross_attention_dim,
                unet.config.use_linear_projection,
                unet.config.upcast_attention,
            )
            original_unet.load_state_dict(unet.state_dict())
            unet = original_unet
        unet: InferUNet2DConditionModel = InferUNet2DConditionModel(unet)
        text_encoders = [text_encoder]

        # VAEを読み込む
        if args.vae is not None:
            vae = model_util.load_vae(args.vae, dtype)
            logger.info("additional VAE loaded")

    # xformers、Hypernetwork対応
    if not args.diffusers_xformers:
        mem_eff = not (args.xformers or args.sdpa)
        replace_unet_modules(unet, mem_eff, args.xformers, args.sdpa)
        replace_vae_modules(vae, mem_eff, args.xformers, args.sdpa)

    # tokenizerを読み込む
    logger.info("loading tokenizer")
    if is_sdxl:
        tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)
        tokenizers = [tokenizer1, tokenizer2]
    else:
        if use_stable_diffusion_format:
            tokenizer = train_util.load_tokenizer(args)
        tokenizers = [tokenizer]

    # schedulerを用意する
    sched_init_args = {}
    has_steps_offset = True
    has_clip_sample = True
    scheduler_num_noises_per_step = 1

    if args.sampler == "ddim":
        scheduler_cls = DDIMScheduler
        scheduler_module = diffusers.schedulers.scheduling_ddim
    elif args.sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
        scheduler_module = diffusers.schedulers.scheduling_ddpm
    elif args.sampler == "pndm":
        scheduler_cls = PNDMScheduler
        scheduler_module = diffusers.schedulers.scheduling_pndm
        has_clip_sample = False
    elif args.sampler == "lms" or args.sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_lms_discrete
        has_clip_sample = False
    elif args.sampler == "euler" or args.sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_euler_discrete
        has_clip_sample = False
    elif args.sampler == "euler_a" or args.sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteSchedulerGL
        scheduler_module = diffusers.schedulers.scheduling_euler_ancestral_discrete
        has_clip_sample = False
    elif args.sampler == "dpmsolver" or args.sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = args.sampler
        scheduler_module = diffusers.schedulers.scheduling_dpmsolver_multistep
        has_clip_sample = False
    elif args.sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
        scheduler_module = diffusers.schedulers.scheduling_dpmsolver_singlestep
        has_clip_sample = False
        has_steps_offset = False
    elif args.sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_heun_discrete
        has_clip_sample = False
    elif args.sampler == "dpm_2" or args.sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_discrete
        has_clip_sample = False
    elif args.sampler == "dpm_2_a" or args.sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete
        scheduler_num_noises_per_step = 2
        has_clip_sample = False

    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    # 警告を出さないようにする
    if has_steps_offset:
        sched_init_args["steps_offset"] = 1
    if has_clip_sample:
        sched_init_args["clip_sample"] = False

    # samplerの乱数をあらかじめ指定するための処理

    # replace randn
    class NoiseManager:
        def __init__(self):
            self.sampler_noises = None
            self.sampler_noise_index = 0

        def reset_sampler_noises(self, noises):
            self.sampler_noise_index = 0
            self.sampler_noises = noises

        def randn(self, shape, device=None, dtype=None, layout=None, generator=None):
            # print("replacing", shape, len(self.sampler_noises), self.sampler_noise_index)
            if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises):
                noise = self.sampler_noises[self.sampler_noise_index]
                if shape != noise.shape:
                    noise = None
            else:
                noise = None

            if noise == None:
                logger.warning(f"unexpected noise request: {self.sampler_noise_index}, {shape}")
                noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)

            self.sampler_noise_index += 1
            return noise

    class TorchRandReplacer:
        def __init__(self, noise_manager):
            self.noise_manager = noise_manager

        def __getattr__(self, item):
            if item == "randn":
                return self.noise_manager.randn
            if hasattr(torch, item):
                return getattr(torch, item)
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    noise_manager = NoiseManager()
    if scheduler_module is not None:
        scheduler_module.torch = TorchRandReplacer(noise_manager)

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    # ↓以下は結局PipeでFalseに設定されるので意味がなかった
    # # clip_sample=Trueにする
    # if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
    #     print("set clip_sample to True")
    #     scheduler.config.clip_sample = True

    # deviceを決定する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "mps"を考量してない

    # custom pipelineをコピったやつを生成する
    if args.vae_slices:
        from library.slicing_vae import SlicingAutoencoderKL

        sli_vae = SlicingAutoencoderKL(
            act_fn="silu",
            block_out_channels=(128, 256, 512, 512),
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            in_channels=3,
            latent_channels=4,
            layers_per_block=2,
            norm_num_groups=32,
            out_channels=3,
            sample_size=512,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            num_slices=args.vae_slices,
        )
        sli_vae.load_state_dict(vae.state_dict())  # vaeのパラメータをコピーする
        vae = sli_vae
        del sli_vae

    vae_dtype = dtype
    if args.no_half_vae:
        logger.info("set vae_dtype to float32")
        vae_dtype = torch.float32
    vae.to(vae_dtype).to(device)
    vae.eval()

    for text_encoder in text_encoders:
        text_encoder.to(dtype).to(device)
        text_encoder.eval()
    unet.to(dtype).to(device)
    unet.eval()

    # networkを組み込む
    if args.network_module:
        networks = []
        network_default_muls = []
        network_pre_calc = args.network_pre_calc

        # merge関連の引数を統合する
        if args.network_merge:
            network_merge = len(args.network_module)  # all networks are merged
        elif args.network_merge_n_models:
            network_merge = args.network_merge_n_models
        else:
            network_merge = 0
        logger.info(f"network_merge: {network_merge}")

        for i, network_module in enumerate(args.network_module):
            logger.info("import network module: {network_module}")
            imported_module = importlib.import_module(network_module)

            network_mul = 1.0 if args.network_mul is None or len(args.network_mul) <= i else args.network_mul[i]

            net_kwargs = {}
            if args.network_args and i < len(args.network_args):
                network_args = args.network_args[i]
                # TODO escape special chars
                network_args = network_args.split(";")
                for net_arg in network_args:
                    key, value = net_arg.split("=")
                    net_kwargs[key] = value

            if args.network_weights is None or len(args.network_weights) <= i:
                raise ValueError("No weight. Weight is required.")

            network_weight = args.network_weights[i]
            logger.info(f"load network weights from: {network_weight}")

            if model_util.is_safetensors(network_weight) and args.network_show_meta:
                from safetensors.torch import safe_open

                with safe_open(network_weight, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is not None:
                    logger.info(f"metadata for: {network_weight}: {metadata}")

            network, weights_sd = imported_module.create_network_from_weights(
                network_mul, network_weight, vae, text_encoders, unet, for_inference=True, **net_kwargs
            )
            if network is None:
                return

            mergeable = network.is_mergeable()
            if network_merge and not mergeable:
                logger.warning("network is not mergiable. ignore merge option.")

            if not mergeable or i >= network_merge:
                # not merging
                network.apply_to(text_encoders, unet)
                info = network.load_state_dict(weights_sd, False)  # network.load_weightsを使うようにするとよい
                logger.info(f"weights are loaded: {info}")

                if args.opt_channels_last:
                    network.to(memory_format=torch.channels_last)
                network.to(dtype).to(device)

                if network_pre_calc:
                    logger.info("backup original weights")
                    network.backup_weights()

                networks.append(network)
                network_default_muls.append(network_mul)
            else:
                network.merge_to(text_encoders, unet, weights_sd, dtype, device)

    else:
        networks = []

    # upscalerの指定があれば取得する
    upscaler = None
    if args.highres_fix_upscaler:
        logger.info("import upscaler module: {args.highres_fix_upscaler}")
        imported_module = importlib.import_module(args.highres_fix_upscaler)

        us_kwargs = {}
        if args.highres_fix_upscaler_args:
            for net_arg in args.highres_fix_upscaler_args.split(";"):
                key, value = net_arg.split("=")
                us_kwargs[key] = value

        logger.info("create upscaler")
        upscaler = imported_module.create_upscaler(**us_kwargs)
        upscaler.to(dtype).to(device)

    # ControlNetの処理
    control_nets: List[Union[ControlNetInfo, Tuple[SdxlControlNet, float]]] = []
    if args.control_net_models:
        if not is_sdxl:
            for i, model in enumerate(args.control_net_models):
                prep_type = None if not args.control_net_preps or len(args.control_net_preps) <= i else args.control_net_preps[i]
                weight = 1.0 if not args.control_net_weights or len(args.control_net_weights) <= i else args.control_net_weights[i]
                ratio = 1.0 if not args.control_net_ratios or len(args.control_net_ratios) <= i else args.control_net_ratios[i]

                ctrl_unet, ctrl_net = original_control_net.load_control_net(args.v2, unet, model)
                prep = original_control_net.load_preprocess(prep_type)
                control_nets.append(ControlNetInfo(ctrl_unet, ctrl_net, prep, weight, ratio))
        else:
            for i, model_file in enumerate(args.control_net_models):
                multiplier = (
                    1.0
                    if not args.control_net_multipliers or len(args.control_net_multipliers) <= i
                    else args.control_net_multipliers[i]
                )
                ratio = 1.0 if not args.control_net_ratios or len(args.control_net_ratios) <= i else args.control_net_ratios[i]

                logger.info(f"loading SDXL ControlNet: {model_file}")
                from safetensors.torch import load_file

                state_dict = load_file(model_file)

                logger.info(f"Initializing SDXL ControlNet with multiplier: {multiplier}")
                with init_empty_weights():
                    control_net = SdxlControlNet(multiplier=multiplier)
                control_net.load_state_dict(state_dict)
                control_net.to(dtype).to(device)
                control_nets.append((control_net, ratio))

    control_net_lllites: List[Tuple[ControlNetLLLite, float]] = []
    if args.control_net_lllite_models:
        for i, model_file in enumerate(args.control_net_lllite_models):
            logger.info(f"loading ControlNet-LLLite: {model_file}")

            from safetensors.torch import load_file

            state_dict = load_file(model_file)
            mlp_dim = None
            cond_emb_dim = None
            for key, value in state_dict.items():
                if mlp_dim is None and "down.0.weight" in key:
                    mlp_dim = value.shape[0]
                elif cond_emb_dim is None and "conditioning1.0" in key:
                    cond_emb_dim = value.shape[0] * 2
                if mlp_dim is not None and cond_emb_dim is not None:
                    break
            assert mlp_dim is not None and cond_emb_dim is not None, f"invalid control net: {model_file}"

            multiplier = (
                1.0
                if not args.control_net_multipliers or len(args.control_net_multipliers) <= i
                else args.control_net_multipliers[i]
            )
            ratio = 1.0 if not args.control_net_ratios or len(args.control_net_ratios) <= i else args.control_net_ratios[i]

            control_net_lllite = ControlNetLLLite(unet, cond_emb_dim, mlp_dim, multiplier=multiplier)
            control_net_lllite.apply_to()
            control_net_lllite.load_state_dict(state_dict)
            control_net_lllite.to(dtype).to(device)
            control_net_lllite.set_batch_cond_only(False, False)
            control_net_lllites.append((control_net_lllite, ratio))
    assert (
        len(control_nets) == 0 or len(control_net_lllites) == 0
    ), "ControlNet and ControlNet-LLLite cannot be used at the same time"

    if args.opt_channels_last:
        logger.info(f"set optimizing: channels last")
        for text_encoder in text_encoders:
            text_encoder.to(memory_format=torch.channels_last)
        vae.to(memory_format=torch.channels_last)
        unet.to(memory_format=torch.channels_last)
        if networks:
            for network in networks:
                network.to(memory_format=torch.channels_last)

        for cn in control_nets:
            cn.to(memory_format=torch.channels_last)

        for cn in control_net_lllites:
            cn.to(memory_format=torch.channels_last)

    pipe = PipelineLike(
        is_sdxl,
        device,
        vae,
        text_encoders,
        tokenizers,
        unet,
        scheduler,
        args.clip_skip,
    )
    pipe.set_control_nets(control_nets)
    pipe.set_control_net_lllites(control_net_lllites)
    logger.info("pipeline is ready.")

    if args.diffusers_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    # Deep Shrink
    if args.ds_depth_1 is not None:
        unet.set_deep_shrink(args.ds_depth_1, args.ds_timesteps_1, args.ds_depth_2, args.ds_timesteps_2, args.ds_ratio)

    # Gradual Latent
    if args.gradual_latent_timesteps is not None:
        if args.gradual_latent_unsharp_params:
            us_params = args.gradual_latent_unsharp_params.split(",")
            us_ksize, us_sigma, us_strength = [float(v) for v in us_params[:3]]
            us_target_x = True if len(us_params) <= 3 else bool(int(us_params[3]))
            us_ksize = int(us_ksize)
        else:
            us_ksize, us_sigma, us_strength, us_target_x = None, None, None, None

        gradual_latent = GradualLatent(
            args.gradual_latent_ratio,
            args.gradual_latent_timesteps,
            args.gradual_latent_every_n_steps,
            args.gradual_latent_ratio_step,
            args.gradual_latent_s_noise,
            us_ksize,
            us_sigma,
            us_strength,
            us_target_x,
        )
        pipe.set_gradual_latent(gradual_latent)

    #  Textual Inversionを処理する
    if args.textual_inversion_embeddings:
        token_ids_embeds1 = []
        token_ids_embeds2 = []
        for embeds_file in args.textual_inversion_embeddings:
            if model_util.is_safetensors(embeds_file):
                from safetensors.torch import load_file

                data = load_file(embeds_file)
            else:
                data = torch.load(embeds_file, map_location="cpu")

            if "string_to_param" in data:
                data = data["string_to_param"]
            if is_sdxl:

                embeds1 = data["clip_l"]  # text encoder 1
                embeds2 = data["clip_g"]  # text encoder 2
            else:
                embeds1 = next(iter(data.values()))
                embeds2 = None

            num_vectors_per_token = embeds1.size()[0]
            token_string = os.path.splitext(os.path.basename(embeds_file))[0]

            token_strings = [token_string] + [f"{token_string}{i+1}" for i in range(num_vectors_per_token - 1)]

            # add new word to tokenizer, count is num_vectors_per_token
            num_added_tokens1 = tokenizers[0].add_tokens(token_strings)
            num_added_tokens2 = tokenizers[1].add_tokens(token_strings) if is_sdxl else 0
            assert num_added_tokens1 == num_vectors_per_token and (
                num_added_tokens2 == 0 or num_added_tokens2 == num_vectors_per_token
            ), (
                f"tokenizer has same word to token string (filename): {embeds_file}"
                + f" / 指定した名前（ファイル名）のトークンが既に存在します: {embeds_file}"
            )

            token_ids1 = tokenizers[0].convert_tokens_to_ids(token_strings)
            token_ids2 = tokenizers[1].convert_tokens_to_ids(token_strings) if is_sdxl else None
            logger.info(f"Textual Inversion embeddings `{token_string}` loaded. Tokens are added: {token_ids1} and {token_ids2}")
            assert (
                min(token_ids1) == token_ids1[0] and token_ids1[-1] == token_ids1[0] + len(token_ids1) - 1
            ), f"token ids1 is not ordered"
            assert not is_sdxl or (
                min(token_ids2) == token_ids2[0] and token_ids2[-1] == token_ids2[0] + len(token_ids2) - 1
            ), f"token ids2 is not ordered"
            assert len(tokenizers[0]) - 1 == token_ids1[-1], f"token ids 1 is not end of tokenize: {len(tokenizers[0])}"
            assert (
                not is_sdxl or len(tokenizers[1]) - 1 == token_ids2[-1]
            ), f"token ids 2 is not end of tokenize: {len(tokenizers[1])}"

            if num_vectors_per_token > 1:
                pipe.add_token_replacement(0, token_ids1[0], token_ids1)  # hoge -> hoge, hogea, hogeb, ...
                if is_sdxl:
                    pipe.add_token_replacement(1, token_ids2[0], token_ids2)

            token_ids_embeds1.append((token_ids1, embeds1))
            if is_sdxl:
                token_ids_embeds2.append((token_ids2, embeds2))

        text_encoders[0].resize_token_embeddings(len(tokenizers[0]))
        token_embeds1 = text_encoders[0].get_input_embeddings().weight.data
        for token_ids, embeds in token_ids_embeds1:
            for token_id, embed in zip(token_ids, embeds):
                token_embeds1[token_id] = embed

        if is_sdxl:
            text_encoders[1].resize_token_embeddings(len(tokenizers[1]))
            token_embeds2 = text_encoders[1].get_input_embeddings().weight.data
            for token_ids, embeds in token_ids_embeds2:
                for token_id, embed in zip(token_ids, embeds):
                    token_embeds2[token_id] = embed

    # promptを取得する
    prompt_list = None
    if args.from_file is not None:
        logger.info(f"reading prompts from {args.from_file}")
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompt_list = f.read().splitlines()
            prompt_list = [d for d in prompt_list if len(d.strip()) > 0 and d[0] != "#"]
        prompter = ListPrompter(prompt_list)

    elif args.from_module is not None:

        def load_module_from_path(module_name, file_path):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                raise ImportError(f"Module '{module_name}' cannot be loaded from '{file_path}'")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        logger.info(f"reading prompts from module: {args.from_module}")
        prompt_module = load_module_from_path("prompt_module", args.from_module)

        prompter = prompt_module.get_prompter(args, pipe, networks)

    elif args.prompt is not None:
        prompter = ListPrompter([args.prompt])

    else:
        prompter = None  # interactive mode

    if args.interactive:
        args.n_iter = 1

    # img2imgの前処理、画像の読み込みなど
    def load_images(path):
        if os.path.isfile(path):
            paths = [path]
        else:
            paths = (
                glob.glob(os.path.join(path, "*.png"))
                + glob.glob(os.path.join(path, "*.jpg"))
                + glob.glob(os.path.join(path, "*.jpeg"))
                + glob.glob(os.path.join(path, "*.webp"))
            )
            paths.sort()

        images = []
        for p in paths:
            image = Image.open(p)
            if image.mode != "RGB":
                logger.info(f"convert image to RGB from {image.mode}: {p}")
                image = image.convert("RGB")
            images.append(image)

        return images

    def resize_images(imgs, size):
        resized = []
        for img in imgs:
            r_img = img.resize(size, Image.Resampling.LANCZOS)
            if hasattr(img, "filename"):  # filename属性がない場合があるらしい
                r_img.filename = img.filename
            resized.append(r_img)
        return resized

    if args.image_path is not None:
        logger.info(f"load image for img2img: {args.image_path}")
        init_images = load_images(args.image_path)
        assert len(init_images) > 0, f"No image / 画像がありません: {args.image_path}"
        logger.info(f"loaded {len(init_images)} images for img2img")

        # CLIP Vision
        if args.clip_vision_strength is not None:
            logger.info(f"load CLIP Vision model: {CLIP_VISION_MODEL}")
            vision_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_VISION_MODEL, projection_dim=1280)
            vision_model.to(device, dtype)
            processor = CLIPImageProcessor.from_pretrained(CLIP_VISION_MODEL)

            pipe.clip_vision_model = vision_model
            pipe.clip_vision_processor = processor
            pipe.clip_vision_strength = args.clip_vision_strength
            logger.info(f"CLIP Vision model loaded.")

    else:
        init_images = None

    if args.mask_path is not None:
        logger.info(f"load mask for inpainting: {args.mask_path}")
        mask_images = load_images(args.mask_path)
        assert len(mask_images) > 0, f"No mask image / マスク画像がありません: {args.image_path}"
        logger.info(f"loaded {len(mask_images)} mask images for inpainting")
    else:
        mask_images = None

    # promptがないとき、画像のPngInfoから取得する
    if init_images is not None and prompter is None and not args.interactive:
        logger.info("get prompts from images' metadata")
        prompt_list = []
        for img in init_images:
            if "prompt" in img.text:
                prompt = img.text["prompt"]
                if "negative-prompt" in img.text:
                    prompt += " --n " + img.text["negative-prompt"]
                prompt_list.append(prompt)
        prompter = ListPrompter(prompt_list)

        # プロンプトと画像を一致させるため指定回数だけ繰り返す（画像を増幅する）
        l = []
        for im in init_images:
            l.extend([im] * args.images_per_prompt)
        init_images = l

        if mask_images is not None:
            l = []
            for im in mask_images:
                l.extend([im] * args.images_per_prompt)
            mask_images = l

    # 画像サイズにオプション指定があるときはリサイズする
    if args.W is not None and args.H is not None:
        # highres fix を考慮に入れる
        w, h = args.W, args.H
        if highres_fix:
            w = int(w * args.highres_fix_scale + 0.5)
            h = int(h * args.highres_fix_scale + 0.5)

        if init_images is not None:
            logger.info(f"resize img2img source images to {w}*{h}")
            init_images = resize_images(init_images, (w, h))
        if mask_images is not None:
            logger.info(f"resize img2img mask images to {w}*{h}")
            mask_images = resize_images(mask_images, (w, h))

    regional_network = False
    if networks and mask_images:
        # mask を領域情報として流用する、現在は一回のコマンド呼び出しで1枚だけ対応
        regional_network = True
        logger.info("use mask as region")

        size = None
        for i, network in enumerate(networks):
            if (i < 3 and args.network_regional_mask_max_color_codes is None) or i < args.network_regional_mask_max_color_codes:
                np_mask = np.array(mask_images[0])

                if args.network_regional_mask_max_color_codes:
                    # カラーコードでマスクを指定する
                    ch0 = (i + 1) & 1
                    ch1 = ((i + 1) >> 1) & 1
                    ch2 = ((i + 1) >> 2) & 1
                    np_mask = np.all(np_mask == np.array([ch0, ch1, ch2]) * 255, axis=2)
                    np_mask = np_mask.astype(np.uint8) * 255
                else:
                    np_mask = np_mask[:, :, i]
                size = np_mask.shape
            else:
                np_mask = np.full(size, 255, dtype=np.uint8)
            mask = torch.from_numpy(np_mask.astype(np.float32) / 255.0)
            network.set_region(i, i == len(networks) - 1, mask)
        mask_images = None

    prev_image = None  # for VGG16 guided
    if args.guide_image_path is not None:
        logger.info(f"load image for ControlNet guidance: {args.guide_image_path}")
        guide_images = []
        for p in args.guide_image_path:
            guide_images.extend(load_images(p))

        logger.info(f"loaded {len(guide_images)} guide images for guidance")
        if len(guide_images) == 0:
            logger.warning(
                f"No guide image, use previous generated image. / ガイド画像がありません。直前に生成した画像を使います: {args.image_path}"
            )
            guide_images = None
    else:
        guide_images = None

    # 新しい乱数生成器を作成する
    if args.seed is not None:
        if prompt_list and len(prompt_list) == 1 and args.images_per_prompt == 1:
            # 引数のseedをそのまま使う
            def fixed_seed(*args, **kwargs):
                return args.seed

            seed_random = SimpleNamespace(randint=fixed_seed)
        else:
            seed_random = random.Random(args.seed)
    else:
        seed_random = random.Random()

    # デフォルト画像サイズを設定する：img2imgではこれらの値は無視される（またはW*Hにリサイズ済み）
    if args.W is None:
        args.W = 1024 if is_sdxl else 512
    if args.H is None:
        args.H = 1024 if is_sdxl else 512

    # 画像生成のループ
    os.makedirs(args.outdir, exist_ok=True)
    max_embeddings_multiples = 1 if args.max_embeddings_multiples is None else args.max_embeddings_multiples

    for gen_iter in range(args.n_iter):
        logger.info(f"iteration {gen_iter+1}/{args.n_iter}")
        if args.iter_same_seed:
            iter_seed = seed_random.randint(0, 2**32 - 1)
        else:
            iter_seed = None

        # shuffle prompt list
        if args.shuffle_prompts:
            prompter.shuffle()

        # バッチ処理の関数
        def process_batch(batch: List[BatchData], highres_fix, highres_1st=False):
            batch_size = len(batch)

            # highres_fixの処理
            if highres_fix and not highres_1st:
                # 1st stageのバッチを作成して呼び出す：サイズを小さくして呼び出す
                is_1st_latent = upscaler.support_latents() if upscaler else args.highres_fix_latents_upscaling

                logger.info("process 1st stage")
                batch_1st = []
                for _, base, ext in batch:

                    def scale_and_round(x):
                        if x is None:
                            return None
                        return int(x * args.highres_fix_scale + 0.5)

                    width_1st = scale_and_round(ext.width)
                    height_1st = scale_and_round(ext.height)
                    width_1st = width_1st - width_1st % 32
                    height_1st = height_1st - height_1st % 32

                    original_width_1st = scale_and_round(ext.original_width)
                    original_height_1st = scale_and_round(ext.original_height)
                    original_width_negative_1st = scale_and_round(ext.original_width_negative)
                    original_height_negative_1st = scale_and_round(ext.original_height_negative)
                    crop_left_1st = scale_and_round(ext.crop_left)
                    crop_top_1st = scale_and_round(ext.crop_top)

                    strength_1st = ext.strength if args.highres_fix_strength is None else args.highres_fix_strength

                    ext_1st = BatchDataExt(
                        width_1st,
                        height_1st,
                        original_width_1st,
                        original_height_1st,
                        original_width_negative_1st,
                        original_height_negative_1st,
                        crop_left_1st,
                        crop_top_1st,
                        args.highres_fix_steps,
                        ext.scale,
                        ext.negative_scale,
                        strength_1st,
                        ext.network_muls,
                        ext.num_sub_prompts,
                    )
                    batch_1st.append(BatchData(is_1st_latent, base, ext_1st))

                pipe.set_enable_control_net(True)  # 1st stageではControlNetを有効にする
                images_1st = process_batch(batch_1st, True, True)

                # 2nd stageのバッチを作成して以下処理する
                logger.info("process 2nd stage")
                width_2nd, height_2nd = batch[0].ext.width, batch[0].ext.height

                if upscaler:
                    # upscalerを使って画像を拡大する
                    lowreso_imgs = None if is_1st_latent else images_1st
                    lowreso_latents = None if not is_1st_latent else images_1st

                    # 戻り値はPIL.Image.Imageかtorch.Tensorのlatents
                    batch_size = len(images_1st)
                    vae_batch_size = (
                        batch_size
                        if args.vae_batch_size is None
                        else (max(1, int(batch_size * args.vae_batch_size)) if args.vae_batch_size < 1 else args.vae_batch_size)
                    )
                    vae_batch_size = int(vae_batch_size)
                    images_1st = upscaler.upscale(
                        vae, lowreso_imgs, lowreso_latents, dtype, width_2nd, height_2nd, batch_size, vae_batch_size
                    )

                elif args.highres_fix_latents_upscaling:
                    # latentを拡大する
                    org_dtype = images_1st.dtype
                    if images_1st.dtype == torch.bfloat16:
                        images_1st = images_1st.to(torch.float)  # interpolateがbf16をサポートしていない
                    images_1st = torch.nn.functional.interpolate(
                        images_1st, (batch[0].ext.height // 8, batch[0].ext.width // 8), mode="bilinear"
                    )  # , antialias=True)
                    images_1st = images_1st.to(org_dtype)

                else:
                    # 画像をLANCZOSで拡大する
                    images_1st = [image.resize((width_2nd, height_2nd), resample=PIL.Image.LANCZOS) for image in images_1st]

                batch_2nd = []
                for i, (bd, image) in enumerate(zip(batch, images_1st)):
                    bd_2nd = BatchData(False, BatchDataBase(*bd.base[0:3], bd.base.seed + 1, image, None, *bd.base[6:]), bd.ext)
                    batch_2nd.append(bd_2nd)
                batch = batch_2nd

                if args.highres_fix_disable_control_net:
                    pipe.set_enable_control_net(False)  # オプション指定時、2nd stageではControlNetを無効にする

            # このバッチの情報を取り出す
            (
                return_latents,
                (step_first, _, _, _, init_image, mask_image, _, guide_image, _, _),
                (
                    width,
                    height,
                    original_width,
                    original_height,
                    original_width_negative,
                    original_height_negative,
                    crop_left,
                    crop_top,
                    steps,
                    scale,
                    negative_scale,
                    strength,
                    network_muls,
                    num_sub_prompts,
                ),
            ) = batch[0]
            noise_shape = (LATENT_CHANNELS, height // DOWNSAMPLING_FACTOR, width // DOWNSAMPLING_FACTOR)

            prompts = []
            negative_prompts = []
            raw_prompts = []
            filenames = []
            start_code = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
            noises = [
                torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
                for _ in range(steps * scheduler_num_noises_per_step)
            ]
            seeds = []
            clip_prompts = []

            if init_image is not None:  # img2img?
                i2i_noises = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
                init_images = []

                if mask_image is not None:
                    mask_images = []
                else:
                    mask_images = None
            else:
                i2i_noises = None
                init_images = None
                mask_images = None

            if guide_image is not None:  # CLIP image guided?
                guide_images = []
            else:
                guide_images = None

            # バッチ内の位置に関わらず同じ乱数を使うためにここで乱数を生成しておく。あわせてimage/maskがbatch内で同一かチェックする
            all_images_are_same = True
            all_masks_are_same = True
            all_guide_images_are_same = True
            for i, (
                _,
                (_, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image, raw_prompt, filename),
                _,
            ) in enumerate(batch):
                prompts.append(prompt)
                negative_prompts.append(negative_prompt)
                seeds.append(seed)
                clip_prompts.append(clip_prompt)
                raw_prompts.append(raw_prompt)
                filenames.append(filename)

                if init_image is not None:
                    init_images.append(init_image)
                    if i > 0 and all_images_are_same:
                        all_images_are_same = init_images[-2] is init_image

                if mask_image is not None:
                    mask_images.append(mask_image)
                    if i > 0 and all_masks_are_same:
                        all_masks_are_same = mask_images[-2] is mask_image

                if guide_image is not None:
                    if type(guide_image) is list:
                        guide_images.extend(guide_image)
                        all_guide_images_are_same = False
                    else:
                        guide_images.append(guide_image)
                        if i > 0 and all_guide_images_are_same:
                            all_guide_images_are_same = guide_images[-2] is guide_image

                # make start code
                torch.manual_seed(seed)
                start_code[i] = torch.randn(noise_shape, device=device, dtype=dtype)

                # make each noises
                for j in range(steps * scheduler_num_noises_per_step):
                    noises[j][i] = torch.randn(noise_shape, device=device, dtype=dtype)

                if i2i_noises is not None:  # img2img noise
                    i2i_noises[i] = torch.randn(noise_shape, device=device, dtype=dtype)

            noise_manager.reset_sampler_noises(noises)

            # すべての画像が同じなら1枚だけpipeに渡すことでpipe側で処理を高速化する
            if init_images is not None and all_images_are_same:
                init_images = init_images[0]
            if mask_images is not None and all_masks_are_same:
                mask_images = mask_images[0]
            if guide_images is not None and all_guide_images_are_same:
                guide_images = guide_images[0]

            # ControlNet使用時はguide imageをリサイズする
            if control_nets or control_net_lllites:
                # TODO resampleのメソッド
                guide_images = guide_images if type(guide_images) == list else [guide_images]
                guide_images = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in guide_images]
                if len(guide_images) == 1:
                    guide_images = guide_images[0]

            # generate
            if networks:
                # 追加ネットワークの処理
                shared = {}
                for n, m in zip(networks, network_muls if network_muls else network_default_muls):
                    n.set_multiplier(m)
                    if regional_network:
                        # TODO バッチから ds_ratio を取り出すべき
                        n.set_current_generation(batch_size, num_sub_prompts, width, height, shared, unet.ds_ratio)

                if not regional_network and network_pre_calc:
                    for n in networks:
                        n.restore_weights()
                    for n in networks:
                        n.pre_calculation()
                    logger.info("pre-calculation... done")

            images = pipe(
                prompts,
                negative_prompts,
                init_images,
                mask_images,
                height,
                width,
                original_height,
                original_width,
                original_height_negative,
                original_width_negative,
                crop_top,
                crop_left,
                steps,
                scale,
                negative_scale,
                strength,
                latents=start_code,
                output_type="pil",
                max_embeddings_multiples=max_embeddings_multiples,
                img2img_noise=i2i_noises,
                vae_batch_size=args.vae_batch_size,
                return_latents=return_latents,
                clip_prompts=clip_prompts,
                clip_guide_images=guide_images,
                emb_normalize_mode=args.emb_normalize_mode,
            )
            if highres_1st and not args.highres_fix_save_1st:  # return images or latents
                return images

            # save image
            highres_prefix = ("0" if highres_1st else "1") if highres_fix else ""
            ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
            for i, (image, prompt, negative_prompts, seed, clip_prompt, raw_prompt, filename) in enumerate(
                zip(images, prompts, negative_prompts, seeds, clip_prompts, raw_prompts, filenames)
            ):
                if highres_fix:
                    seed -= 1  # record original seed
                metadata = PngInfo()
                metadata.add_text("prompt", prompt)
                metadata.add_text("seed", str(seed))
                metadata.add_text("sampler", args.sampler)
                metadata.add_text("steps", str(steps))
                metadata.add_text("scale", str(scale))
                if negative_prompt is not None:
                    metadata.add_text("negative-prompt", negative_prompt)
                if negative_scale is not None:
                    metadata.add_text("negative-scale", str(negative_scale))
                if clip_prompt is not None:
                    metadata.add_text("clip-prompt", clip_prompt)
                if raw_prompt is not None:
                    metadata.add_text("raw-prompt", raw_prompt)
                if is_sdxl:
                    metadata.add_text("original-height", str(original_height))
                    metadata.add_text("original-width", str(original_width))
                    metadata.add_text("original-height-negative", str(original_height_negative))
                    metadata.add_text("original-width-negative", str(original_width_negative))
                    metadata.add_text("crop-top", str(crop_top))
                    metadata.add_text("crop-left", str(crop_left))

                if filename is not None:
                    fln = filename
                else:
                    if args.use_original_file_name and init_images is not None:
                        if type(init_images) is list:
                            fln = os.path.splitext(os.path.basename(init_images[i % len(init_images)].filename))[0] + ".png"
                        else:
                            fln = os.path.splitext(os.path.basename(init_images.filename))[0] + ".png"
                    elif args.sequential_file_name:
                        fln = f"im_{highres_prefix}{step_first + i + 1:06d}.png"
                    else:
                        fln = f"im_{ts_str}_{highres_prefix}{i:03d}_{seed}.png"

                if fln.endswith(".webp"):
                    image.save(os.path.join(args.outdir, fln), pnginfo=metadata, quality=100)  # lossy
                else:
                    image.save(os.path.join(args.outdir, fln), pnginfo=metadata)

            if not args.no_preview and not highres_1st and args.interactive:
                try:
                    import cv2

                    for prompt, image in zip(prompts, images):
                        cv2.imshow(prompt[:128], np.array(image)[:, :, ::-1])  # プロンプトが長いと死ぬ
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                except ImportError:
                    logger.warning(
                        "opencv-python is not installed, cannot preview / opencv-pythonがインストールされていないためプレビューできません"
                    )

            return images

        # 画像生成のプロンプトが一周するまでのループ
        prompt_index = 0
        global_step = 0
        batch_data = []
        while True:
            if args.interactive:
                # interactive
                valid = False
                while not valid:
                    logger.info("\nType prompt:")
                    try:
                        raw_prompt = input()
                    except EOFError:
                        break

                    valid = len(raw_prompt.strip().split(" --")[0].strip()) > 0
                if not valid:  # EOF, end app
                    break
            else:
                raw_prompt = prompter(args, pipe, seed_random, iter_seed, prompt_index, global_step)
                if raw_prompt is None:
                    break

            # sd-dynamic-prompts like variants:
            # count is 1 (not dynamic) or images_per_prompt (no enumeration) or arbitrary (enumeration)
            raw_prompts = handle_dynamic_prompt_variants(raw_prompt, args.images_per_prompt)

            # repeat prompt
            for pi in range(args.images_per_prompt if len(raw_prompts) == 1 else len(raw_prompts)):
                raw_prompt = raw_prompts[pi] if len(raw_prompts) > 1 else raw_prompts[0]
                filename = None

                if pi == 0 or len(raw_prompts) > 1:
                    # parse prompt: if prompt is not changed, skip parsing
                    width = args.W
                    height = args.H
                    original_width = args.original_width
                    original_height = args.original_height
                    original_width_negative = args.original_width_negative
                    original_height_negative = args.original_height_negative
                    crop_top = args.crop_top
                    crop_left = args.crop_left
                    scale = args.scale
                    negative_scale = args.negative_scale
                    steps = args.steps
                    seed = None
                    seeds = None
                    strength = 0.8 if args.strength is None else args.strength
                    negative_prompt = ""
                    clip_prompt = None
                    network_muls = None

                    # Deep Shrink
                    ds_depth_1 = None  # means no override
                    ds_timesteps_1 = args.ds_timesteps_1
                    ds_depth_2 = args.ds_depth_2
                    ds_timesteps_2 = args.ds_timesteps_2
                    ds_ratio = args.ds_ratio

                    # Gradual Latent
                    gl_timesteps = None  # means no override
                    gl_ratio = args.gradual_latent_ratio
                    gl_every_n_steps = args.gradual_latent_every_n_steps
                    gl_ratio_step = args.gradual_latent_ratio_step
                    gl_s_noise = args.gradual_latent_s_noise
                    gl_unsharp_params = args.gradual_latent_unsharp_params

                    prompt_args = raw_prompt.strip().split(" --")
                    prompt = prompt_args[0]
                    length = len(prompter) if hasattr(prompter, "__len__") else 0
                    logger.info(f"prompt {prompt_index+1}/{length}: {prompt}")

                    for parg in prompt_args[1:]:
                        try:
                            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
                            if m:
                                width = int(m.group(1))
                                logger.info(f"width: {width}")
                                continue

                            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
                            if m:
                                height = int(m.group(1))
                                logger.info(f"height: {height}")
                                continue

                            m = re.match(r"ow (\d+)", parg, re.IGNORECASE)
                            if m:
                                original_width = int(m.group(1))
                                logger.info(f"original width: {original_width}")
                                continue

                            m = re.match(r"oh (\d+)", parg, re.IGNORECASE)
                            if m:
                                original_height = int(m.group(1))
                                logger.info(f"original height: {original_height}")
                                continue

                            m = re.match(r"nw (\d+)", parg, re.IGNORECASE)
                            if m:
                                original_width_negative = int(m.group(1))
                                logger.info(f"original width negative: {original_width_negative}")
                                continue

                            m = re.match(r"nh (\d+)", parg, re.IGNORECASE)
                            if m:
                                original_height_negative = int(m.group(1))
                                logger.info(f"original height negative: {original_height_negative}")
                                continue

                            m = re.match(r"ct (\d+)", parg, re.IGNORECASE)
                            if m:
                                crop_top = int(m.group(1))
                                logger.info(f"crop top: {crop_top}")
                                continue

                            m = re.match(r"cl (\d+)", parg, re.IGNORECASE)
                            if m:
                                crop_left = int(m.group(1))
                                logger.info(f"crop left: {crop_left}")
                                continue

                            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
                            if m:  # steps
                                steps = max(1, min(1000, int(m.group(1))))
                                logger.info(f"steps: {steps}")
                                continue

                            m = re.match(r"d ([\d,]+)", parg, re.IGNORECASE)
                            if m:  # seed
                                seeds = [int(d) for d in m.group(1).split(",")]
                                logger.info(f"seeds: {seeds}")
                                continue

                            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # scale
                                scale = float(m.group(1))
                                logger.info(f"scale: {scale}")
                                continue

                            m = re.match(r"nl ([\d\.]+|none|None)", parg, re.IGNORECASE)
                            if m:  # negative scale
                                if m.group(1).lower() == "none":
                                    negative_scale = None
                                else:
                                    negative_scale = float(m.group(1))
                                logger.info(f"negative scale: {negative_scale}")
                                continue

                            m = re.match(r"t ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # strength
                                strength = float(m.group(1))
                                logger.info(f"strength: {strength}")
                                continue

                            m = re.match(r"n (.+)", parg, re.IGNORECASE)
                            if m:  # negative prompt
                                negative_prompt = m.group(1)
                                logger.info(f"negative prompt: {negative_prompt}")
                                continue

                            m = re.match(r"c (.+)", parg, re.IGNORECASE)
                            if m:  # clip prompt
                                clip_prompt = m.group(1)
                                logger.info(f"clip prompt: {clip_prompt}")
                                continue

                            m = re.match(r"am ([\d\.\-,]+)", parg, re.IGNORECASE)
                            if m:  # network multiplies
                                network_muls = [float(v) for v in m.group(1).split(",")]
                                while len(network_muls) < len(networks):
                                    network_muls.append(network_muls[-1])
                                logger.info(f"network mul: {network_muls}")
                                continue

                            # Deep Shrink
                            m = re.match(r"dsd1 ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # deep shrink depth 1
                                ds_depth_1 = int(m.group(1))
                                logger.info(f"deep shrink depth 1: {ds_depth_1}")
                                continue

                            m = re.match(r"dst1 ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # deep shrink timesteps 1
                                ds_timesteps_1 = int(m.group(1))
                                ds_depth_1 = ds_depth_1 if ds_depth_1 is not None else -1  # -1 means override
                                logger.info(f"deep shrink timesteps 1: {ds_timesteps_1}")
                                continue

                            m = re.match(r"dsd2 ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # deep shrink depth 2
                                ds_depth_2 = int(m.group(1))
                                ds_depth_1 = ds_depth_1 if ds_depth_1 is not None else -1  # -1 means override
                                logger.info(f"deep shrink depth 2: {ds_depth_2}")
                                continue

                            m = re.match(r"dst2 ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # deep shrink timesteps 2
                                ds_timesteps_2 = int(m.group(1))
                                ds_depth_1 = ds_depth_1 if ds_depth_1 is not None else -1  # -1 means override
                                logger.info(f"deep shrink timesteps 2: {ds_timesteps_2}")
                                continue

                            m = re.match(r"dsr ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # deep shrink ratio
                                ds_ratio = float(m.group(1))
                                ds_depth_1 = ds_depth_1 if ds_depth_1 is not None else -1  # -1 means override
                                logger.info(f"deep shrink ratio: {ds_ratio}")
                                continue

                            # Gradual Latent
                            m = re.match(r"glt ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # gradual latent timesteps
                                gl_timesteps = int(m.group(1))
                                logger.info(f"gradual latent timesteps: {gl_timesteps}")
                                continue

                            m = re.match(r"glr ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # gradual latent ratio
                                gl_ratio = float(m.group(1))
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1  # -1 means override
                                logger.info(f"gradual latent ratio: {ds_ratio}")
                                continue

                            m = re.match(r"gle ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # gradual latent every n steps
                                gl_every_n_steps = int(m.group(1))
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1  # -1 means override
                                logger.info(f"gradual latent every n steps: {gl_every_n_steps}")
                                continue

                            m = re.match(r"gls ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # gradual latent ratio step
                                gl_ratio_step = float(m.group(1))
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1  # -1 means override
                                logger.info(f"gradual latent ratio step: {gl_ratio_step}")
                                continue

                            m = re.match(r"glsn ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # gradual latent s noise
                                gl_s_noise = float(m.group(1))
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1  # -1 means override
                                logger.info(f"gradual latent s noise: {gl_s_noise}")
                                continue

                            m = re.match(r"glus ([\d\.\-,]+)", parg, re.IGNORECASE)
                            if m:  # gradual latent unsharp params
                                gl_unsharp_params = m.group(1)
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1  # -1 means override
                                logger.info(f"gradual latent unsharp params: {gl_unsharp_params}")
                                continue

                            m = re.match(r"f (.+)", parg, re.IGNORECASE)
                            if m:  # filename
                                filename = m.group(1)
                                logger.info(f"filename: {filename}")
                                continue

                        except ValueError as ex:
                            logger.error(f"Exception in parsing / 解析エラー: {parg}")
                            logger.error(f"{ex}")

                # override Deep Shrink
                if ds_depth_1 is not None:
                    if ds_depth_1 < 0:
                        ds_depth_1 = args.ds_depth_1 or 3
                    unet.set_deep_shrink(ds_depth_1, ds_timesteps_1, ds_depth_2, ds_timesteps_2, ds_ratio)

                # override Gradual Latent
                if gl_timesteps is not None:
                    if gl_timesteps < 0:
                        gl_timesteps = args.gradual_latent_timesteps or 650
                    if gl_unsharp_params is not None:
                        unsharp_params = gl_unsharp_params.split(",")
                        us_ksize, us_sigma, us_strength = [float(v) for v in unsharp_params[:3]]
                        us_target_x = True if len(unsharp_params) < 4 else bool(int(unsharp_params[3]))
                        us_ksize = int(us_ksize)
                    else:
                        us_ksize, us_sigma, us_strength, us_target_x = None, None, None, None
                    gradual_latent = GradualLatent(
                        gl_ratio,
                        gl_timesteps,
                        gl_every_n_steps,
                        gl_ratio_step,
                        gl_s_noise,
                        us_ksize,
                        us_sigma,
                        us_strength,
                        us_target_x,
                    )
                    pipe.set_gradual_latent(gradual_latent)

                # prepare seed
                if seeds is not None:  # given in prompt
                    # num_images_per_promptが多い場合は足りなくなるので、足りない分は前のを使う
                    if len(seeds) > 0:
                        seed = seeds.pop(0)
                else:
                    if args.iter_same_seed:
                        seed = iter_seed
                    else:
                        seed = None  # 前のを消す

                if seed is None:
                    seed = seed_random.randint(0, 2**32 - 1)
                if args.interactive:
                    logger.info(f"seed: {seed}")

                # prepare init image, guide image and mask
                init_image = mask_image = guide_image = None

                # 同一イメージを使うとき、本当はlatentに変換しておくと無駄がないが面倒なのでとりあえず毎回処理する
                if init_images is not None:
                    init_image = init_images[global_step % len(init_images)]

                    # img2imgの場合は、基本的に元画像のサイズで生成する。highres fixの場合はargs.W, args.Hとscaleに従いリサイズ済みなので無視する
                    # 32単位に丸めたやつにresizeされるので踏襲する
                    if not highres_fix:
                        width, height = init_image.size
                        width = width - width % 32
                        height = height - height % 32
                        if width != init_image.size[0] or height != init_image.size[1]:
                            logger.warning(
                                f"img2img image size is not divisible by 32 so aspect ratio is changed / img2imgの画像サイズが32で割り切れないためリサイズされます。画像が歪みます"
                            )

                if mask_images is not None:
                    mask_image = mask_images[global_step % len(mask_images)]

                if guide_images is not None:
                    if control_nets or control_net_lllites:  # 複数件の場合あり
                        c = max(len(control_nets), len(control_net_lllites))
                        p = global_step % (len(guide_images) // c)
                        guide_image = guide_images[p * c : p * c + c]
                    else:
                        guide_image = guide_images[global_step % len(guide_images)]

                if regional_network:
                    num_sub_prompts = len(prompt.split(" AND "))
                    assert (
                        len(networks) <= num_sub_prompts
                    ), "Number of networks must be less than or equal to number of sub prompts."
                else:
                    num_sub_prompts = None

                b1 = BatchData(
                    False,
                    BatchDataBase(
                        global_step,
                        prompt,
                        negative_prompt,
                        seed,
                        init_image,
                        mask_image,
                        clip_prompt,
                        guide_image,
                        raw_prompt,
                        filename,
                    ),
                    BatchDataExt(
                        width,
                        height,
                        original_width,
                        original_height,
                        original_width_negative,
                        original_height_negative,
                        crop_left,
                        crop_top,
                        steps,
                        scale,
                        negative_scale,
                        strength,
                        tuple(network_muls) if network_muls else None,
                        num_sub_prompts,
                    ),
                )
                if len(batch_data) > 0 and batch_data[-1].ext != b1.ext:  # バッチ分割必要？
                    process_batch(batch_data, highres_fix)
                    batch_data.clear()

                batch_data.append(b1)
                if len(batch_data) == args.batch_size:
                    prev_image = process_batch(batch_data, highres_fix)[0]
                    batch_data.clear()

                global_step += 1

            prompt_index += 1

        if len(batch_data) > 0:
            process_batch(batch_data, highres_fix)
            batch_data.clear()

    logger.info("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)

    parser.add_argument(
        "--sdxl", action="store_true", help="load Stable Diffusion XL model / Stable Diffusion XLのモデルを読み込む"
    )
    parser.add_argument(
        "--v1", action="store_true", help="load Stable Diffusion v1.x model / Stable Diffusion 1.xのモデルを読み込む"
    )
    parser.add_argument(
        "--v2", action="store_true", help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む"
    )
    parser.add_argument(
        "--v_parameterization", action="store_true", help="enable v-parameterization training / v-parameterization学習を有効にする"
    )

    parser.add_argument("--prompt", type=str, default=None, help="prompt / プロンプト")
    parser.add_argument(
        "--from_file",
        type=str,
        default=None,
        help="if specified, load prompts from this file / 指定時はプロンプトをファイルから読み込む",
    )
    parser.add_argument(
        "--from_module",
        type=str,
        default=None,
        help="if specified, load prompts from this module / 指定時はプロンプトをモジュールから読み込む",
    )
    parser.add_argument(
        "--prompter_module_args", type=str, default=None, help="args for prompter module / prompterモジュールの引数"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="interactive mode (generates one image) / 対話モード（生成される画像は1枚になります）",
    )
    parser.add_argument(
        "--no_preview", action="store_true", help="do not show generated image in interactive mode / 対話モードで画像を表示しない"
    )
    parser.add_argument(
        "--image_path", type=str, default=None, help="image to inpaint or to generate from / img2imgまたはinpaintを行う元画像"
    )
    parser.add_argument("--mask_path", type=str, default=None, help="mask in inpainting / inpaint時のマスク")
    parser.add_argument("--strength", type=float, default=None, help="img2img strength / img2img時のstrength")
    parser.add_argument("--images_per_prompt", type=int, default=1, help="number of images per prompt / プロンプトあたりの出力枚数")
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to / 生成画像の出力先")
    parser.add_argument(
        "--sequential_file_name", action="store_true", help="sequential output file name / 生成画像のファイル名を連番にする"
    )
    parser.add_argument(
        "--use_original_file_name",
        action="store_true",
        help="prepend original file name in img2img / img2imgで元画像のファイル名を生成画像のファイル名の先頭に付ける",
    )
    # parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often / 繰り返し回数")
    parser.add_argument("--H", type=int, default=None, help="image height, in pixel space / 生成画像高さ")
    parser.add_argument("--W", type=int, default=None, help="image width, in pixel space / 生成画像幅")
    parser.add_argument(
        "--original_height",
        type=int,
        default=None,
        help="original height for SDXL conditioning / SDXLの条件付けに用いるoriginal heightの値",
    )
    parser.add_argument(
        "--original_width",
        type=int,
        default=None,
        help="original width for SDXL conditioning / SDXLの条件付けに用いるoriginal widthの値",
    )
    parser.add_argument(
        "--original_height_negative",
        type=int,
        default=None,
        help="original height for SDXL unconditioning / SDXLのネガティブ条件付けに用いるoriginal heightの値",
    )
    parser.add_argument(
        "--original_width_negative",
        type=int,
        default=None,
        help="original width for SDXL unconditioning / SDXLのネガティブ条件付けに用いるoriginal widthの値",
    )
    parser.add_argument(
        "--crop_top", type=int, default=None, help="crop top for SDXL conditioning / SDXLの条件付けに用いるcrop topの値"
    )
    parser.add_argument(
        "--crop_left", type=int, default=None, help="crop left for SDXL conditioning / SDXLの条件付けに用いるcrop leftの値"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size / バッチサイズ")
    parser.add_argument(
        "--vae_batch_size",
        type=float,
        default=None,
        help="batch size for VAE, < 1.0 for ratio / VAE処理時のバッチサイズ、1未満の値の場合は通常バッチサイズの比率",
    )
    parser.add_argument(
        "--vae_slices",
        type=int,
        default=None,
        help="number of slices to split image into for VAE to reduce VRAM usage, None for no splitting (default), slower if specified. 16 or 32 recommended / VAE処理時にVRAM使用量削減のため画像を分割するスライス数、Noneの場合は分割しない（デフォルト）、指定すると遅くなる。16か32程度を推奨",
    )
    parser.add_argument(
        "--no_half_vae", action="store_true", help="do not use fp16/bf16 precision for VAE / VAE処理時にfp16/bf16を使わない"
    )
    parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps / サンプリングステップ数")
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=[
            "ddim",
            "pndm",
            "lms",
            "euler",
            "euler_a",
            "heun",
            "dpm_2",
            "dpm_2_a",
            "dpmsolver",
            "dpmsolver++",
            "dpmsingle",
            "k_lms",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        help=f"sampler (scheduler) type / サンプラー（スケジューラ）の種類",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) / guidance scale",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="path to checkpoint of model / モデルのcheckpointファイルまたはディレクトリ"
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training) / Tokenizerをキャッシュするディレクトリ（ネット接続なしでの学習のため）",
    )
    # parser.add_argument("--replace_clip_l14_336", action='store_true',
    #                     help="Replace CLIP (Text Encoder) to l/14@336 / CLIP(Text Encoder)をl/14@336に入れ替える")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed, or seed of seeds in multiple generation / 1枚生成時のseed、または複数枚生成時の乱数seedを決めるためのseed",
    )
    parser.add_argument(
        "--iter_same_seed",
        action="store_true",
        help="use same seed for all prompts in iteration if no seed specified / 乱数seedの指定がないとき繰り返し内はすべて同じseedを使う（プロンプト間の差異の比較用）",
    )
    parser.add_argument(
        "--shuffle_prompts",
        action="store_true",
        help="shuffle prompts in iteration / 繰り返し内のプロンプトをシャッフルする",
    )
    parser.add_argument("--fp16", action="store_true", help="use fp16 / fp16を指定し省メモリ化する")
    parser.add_argument("--bf16", action="store_true", help="use bfloat16 / bfloat16を指定し省メモリ化する")
    parser.add_argument("--xformers", action="store_true", help="use xformers / xformersを使用し高速化する")
    parser.add_argument("--sdpa", action="store_true", help="use sdpa in PyTorch 2 / sdpa")
    parser.add_argument(
        "--diffusers_xformers",
        action="store_true",
        help="use xformers by diffusers (Hypernetworks doesn't work) / Diffusersでxformersを使用する（Hypernetwork利用不可）",
    )
    parser.add_argument(
        "--opt_channels_last",
        action="store_true",
        help="set channels last option to model / モデルにchannels lastを指定し最適化する",
    )
    parser.add_argument(
        "--network_module",
        type=str,
        default=None,
        nargs="*",
        help="additional network module to use / 追加ネットワークを使う時そのモジュール名",
    )
    parser.add_argument(
        "--network_weights", type=str, default=None, nargs="*", help="additional network weights to load / 追加ネットワークの重み"
    )
    parser.add_argument(
        "--network_mul", type=float, default=None, nargs="*", help="additional network multiplier / 追加ネットワークの効果の倍率"
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_show_meta", action="store_true", help="show metadata of network model / ネットワークモデルのメタデータを表示する"
    )
    parser.add_argument(
        "--network_merge_n_models",
        type=int,
        default=None,
        help="merge this number of networks / この数だけネットワークをマージする",
    )
    parser.add_argument(
        "--network_merge", action="store_true", help="merge network weights to original model / ネットワークの重みをマージする"
    )
    parser.add_argument(
        "--network_pre_calc",
        action="store_true",
        help="pre-calculate network for generation / ネットワークのあらかじめ計算して生成する",
    )
    parser.add_argument(
        "--network_regional_mask_max_color_codes",
        type=int,
        default=None,
        help="max color codes for regional mask (default is None, mask by channel) / regional maskの最大色数（デフォルトはNoneでチャンネルごとのマスク）",
    )
    parser.add_argument(
        "--textual_inversion_embeddings",
        type=str,
        default=None,
        nargs="*",
        help="Embeddings files of Textual Inversion / Textual Inversionのembeddings",
    )
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="layer number from bottom to use in CLIP, default is 1 for SD1/2, 2 for SDXL "
        + "/ CLIPの後ろからn層目の出力を使う（デフォルトはSD1/2の場合1、SDXLの場合2）",
    )
    parser.add_argument(
        "--max_embeddings_multiples",
        type=int,
        default=None,
        help="max embedding multiples, max token length is 75 * multiples / トークン長をデフォルトの何倍とするか 75*この値 がトークン長となる",
    )
    parser.add_argument(
        "--emb_normalize_mode",
        type=str,
        default="original",
        choices=["original", "none", "abs"],
        help="embedding normalization mode / embeddingの正規化モード",
    )
    parser.add_argument(
        "--guide_image_path", type=str, default=None, nargs="*", help="image to ControlNet / ControlNetでガイドに使う画像"
    )
    parser.add_argument(
        "--highres_fix_scale",
        type=float,
        default=None,
        help="enable highres fix, reso scale for 1st stage / highres fixを有効にして最初の解像度をこのscaleにする",
    )
    parser.add_argument(
        "--highres_fix_steps",
        type=int,
        default=28,
        help="1st stage steps for highres fix / highres fixの最初のステージのステップ数",
    )
    parser.add_argument(
        "--highres_fix_strength",
        type=float,
        default=None,
        help="1st stage img2img strength for highres fix / highres fixの最初のステージのimg2img時のstrength、省略時はstrengthと同じ",
    )
    parser.add_argument(
        "--highres_fix_save_1st",
        action="store_true",
        help="save 1st stage images for highres fix / highres fixの最初のステージの画像を保存する",
    )
    parser.add_argument(
        "--highres_fix_latents_upscaling",
        action="store_true",
        help="use latents upscaling for highres fix / highres fixでlatentで拡大する",
    )
    parser.add_argument(
        "--highres_fix_upscaler",
        type=str,
        default=None,
        help="upscaler module for highres fix / highres fixで使うupscalerのモジュール名",
    )
    parser.add_argument(
        "--highres_fix_upscaler_args",
        type=str,
        default=None,
        help="additional arguments for upscaler (key=value) / upscalerへの追加の引数",
    )
    parser.add_argument(
        "--highres_fix_disable_control_net",
        action="store_true",
        help="disable ControlNet for highres fix / highres fixでControlNetを使わない",
    )

    parser.add_argument(
        "--negative_scale",
        type=float,
        default=None,
        help="set another guidance scale for negative prompt / ネガティブプロンプトのscaleを指定する",
    )

    parser.add_argument(
        "--control_net_lllite_models",
        type=str,
        default=None,
        nargs="*",
        help="ControlNet models to use / 使用するControlNetのモデル名",
    )
    parser.add_argument(
        "--control_net_models", type=str, default=None, nargs="*", help="ControlNet models to use / 使用するControlNetのモデル名"
    )
    parser.add_argument(
        "--control_net_preps",
        type=str,
        default=None,
        nargs="*",
        help="ControlNet preprocess to use / 使用するControlNetのプリプロセス名",
    )
    parser.add_argument(
        "--control_net_multipliers", type=float, default=None, nargs="*", help="ControlNet multiplier / ControlNetの適用率"
    )
    parser.add_argument(
        "--control_net_ratios",
        type=float,
        default=None,
        nargs="*",
        help="ControlNet guidance ratio for steps / ControlNetでガイドするステップ比率",
    )
    parser.add_argument(
        "--clip_vision_strength",
        type=float,
        default=None,
        help="enable CLIP Vision Conditioning for img2img with this strength / img2imgでCLIP Vision Conditioningを有効にしてこのstrengthで処理する",
    )

    # Deep Shrink
    parser.add_argument(
        "--ds_depth_1",
        type=int,
        default=None,
        help="Enable Deep Shrink with this depth 1, valid values are 0 to 8 / Deep Shrinkをこのdepthで有効にする",
    )
    parser.add_argument(
        "--ds_timesteps_1",
        type=int,
        default=650,
        help="Apply Deep Shrink depth 1 until this timesteps / Deep Shrink depth 1を適用するtimesteps",
    )
    parser.add_argument("--ds_depth_2", type=int, default=None, help="Deep Shrink depth 2 / Deep Shrinkのdepth 2")
    parser.add_argument(
        "--ds_timesteps_2",
        type=int,
        default=650,
        help="Apply Deep Shrink depth 2 until this timesteps / Deep Shrink depth 2を適用するtimesteps",
    )
    parser.add_argument(
        "--ds_ratio", type=float, default=0.5, help="Deep Shrink ratio for downsampling / Deep Shrinkのdownsampling比率"
    )

    # gradual latent
    parser.add_argument(
        "--gradual_latent_timesteps",
        type=int,
        default=None,
        help="enable Gradual Latent hires fix and apply upscaling from this timesteps / Gradual Latent hires fixをこのtimestepsで有効にし、このtimestepsからアップスケーリングを適用する",
    )
    parser.add_argument(
        "--gradual_latent_ratio",
        type=float,
        default=0.5,
        help=" this size ratio, 0.5 means 1/2 / Gradual Latent hires fixをこのサイズ比率で有効にする、0.5は1/2を意味する",
    )
    parser.add_argument(
        "--gradual_latent_ratio_step",
        type=float,
        default=0.125,
        help="step to increase ratio for Gradual Latent / Gradual Latentのratioをどのくらいずつ上げるか",
    )
    parser.add_argument(
        "--gradual_latent_every_n_steps",
        type=int,
        default=3,
        help="steps to increase size of latents every this steps for Gradual Latent / Gradual Latentでlatentsのサイズをこのステップごとに上げる",
    )
    parser.add_argument(
        "--gradual_latent_s_noise",
        type=float,
        default=1.0,
        help="s_noise for Gradual Latent / Gradual Latentのs_noise",
    )
    parser.add_argument(
        "--gradual_latent_unsharp_params",
        type=str,
        default=None,
        help="unsharp mask parameters for Gradual Latent: ksize, sigma, strength, target-x (1 means True). `3,0.5,0.5,1` or `3,1.0,1.0,0` is recommended /"
        + " Gradual Latentのunsharp maskのパラメータ: ksize, sigma, strength, target-x. `3,0.5,0.5,1` または `3,1.0,1.0,0` が推奨",
    )

    # # parser.add_argument(
    #     "--control_net_image_path", type=str, default=None, nargs="*", help="image for ControlNet guidance / ControlNetでガイドに使う画像"
    # )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
