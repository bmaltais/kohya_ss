import os
import glob
import random
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
from transformers import CLIPTokenizer, T5TokenizerFast, CLIPTextModel, CLIPTextModelWithProjection, T5EncoderModel

from library import sd3_utils, train_util
from library import sd3_models
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


CLIP_L_TOKENIZER_ID = "openai/clip-vit-large-patch14"
CLIP_G_TOKENIZER_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
T5_XXL_TOKENIZER_ID = "google/t5-v1_1-xxl"


class Sd3TokenizeStrategy(TokenizeStrategy):
    def __init__(self, t5xxl_max_length: int = 256, tokenizer_cache_dir: Optional[str] = None) -> None:
        self.t5xxl_max_length = t5xxl_max_length
        self.clip_l = self._load_tokenizer(CLIPTokenizer, CLIP_L_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        self.clip_g = self._load_tokenizer(CLIPTokenizer, CLIP_G_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        self.t5xxl = self._load_tokenizer(T5TokenizerFast, T5_XXL_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        self.clip_g.pad_token_id = 0  # use 0 as pad token for clip_g

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        l_tokens = self.clip_l(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        g_tokens = self.clip_g(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        t5_tokens = self.t5xxl(text, max_length=self.t5xxl_max_length, padding="max_length", truncation=True, return_tensors="pt")

        l_attn_mask = l_tokens["attention_mask"]
        g_attn_mask = g_tokens["attention_mask"]
        t5_attn_mask = t5_tokens["attention_mask"]
        l_tokens = l_tokens["input_ids"]
        g_tokens = g_tokens["input_ids"]
        t5_tokens = t5_tokens["input_ids"]

        return [l_tokens, g_tokens, t5_tokens, l_attn_mask, g_attn_mask, t5_attn_mask]


class Sd3TextEncodingStrategy(TextEncodingStrategy):
    def __init__(
        self,
        apply_lg_attn_mask: Optional[bool] = None,
        apply_t5_attn_mask: Optional[bool] = None,
        l_dropout_rate: float = 0.0,
        g_dropout_rate: float = 0.0,
        t5_dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            apply_t5_attn_mask: Default value for apply_t5_attn_mask.
        """
        self.apply_lg_attn_mask = apply_lg_attn_mask
        self.apply_t5_attn_mask = apply_t5_attn_mask
        self.l_dropout_rate = l_dropout_rate
        self.g_dropout_rate = g_dropout_rate
        self.t5_dropout_rate = t5_dropout_rate

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
        apply_lg_attn_mask: Optional[bool] = False,
        apply_t5_attn_mask: Optional[bool] = False,
        enable_dropout: bool = True,
    ) -> List[torch.Tensor]:
        """
        returned embeddings are not masked
        """
        clip_l, clip_g, t5xxl = models
        clip_l: Optional[CLIPTextModel]
        clip_g: Optional[CLIPTextModelWithProjection]
        t5xxl: Optional[T5EncoderModel]

        if apply_lg_attn_mask is None:
            apply_lg_attn_mask = self.apply_lg_attn_mask
        if apply_t5_attn_mask is None:
            apply_t5_attn_mask = self.apply_t5_attn_mask

        l_tokens, g_tokens, t5_tokens, l_attn_mask, g_attn_mask, t5_attn_mask = tokens

        # dropout: if enable_dropout is False, dropout is not applied. dropout means zeroing out embeddings

        if l_tokens is None or clip_l is None:
            assert g_tokens is None, "g_tokens must be None if l_tokens is None"
            lg_out = None
            lg_pooled = None
            l_attn_mask = None
            g_attn_mask = None
        else:
            assert g_tokens is not None, "g_tokens must not be None if l_tokens is not None"

            # drop some members of the batch: we do not call clip_l and clip_g for dropped members
            batch_size, l_seq_len = l_tokens.shape
            g_seq_len = g_tokens.shape[1]

            non_drop_l_indices = []
            non_drop_g_indices = []
            for i in range(l_tokens.shape[0]):
                drop_l = enable_dropout and (self.l_dropout_rate > 0.0 and random.random() < self.l_dropout_rate)
                drop_g = enable_dropout and (self.g_dropout_rate > 0.0 and random.random() < self.g_dropout_rate)
                if not drop_l:
                    non_drop_l_indices.append(i)
                if not drop_g:
                    non_drop_g_indices.append(i)

            # filter out dropped members
            if len(non_drop_l_indices) > 0 and len(non_drop_l_indices) < batch_size:
                l_tokens = l_tokens[non_drop_l_indices]
                l_attn_mask = l_attn_mask[non_drop_l_indices]
            if len(non_drop_g_indices) > 0 and len(non_drop_g_indices) < batch_size:
                g_tokens = g_tokens[non_drop_g_indices]
                g_attn_mask = g_attn_mask[non_drop_g_indices]

            # call clip_l for non-dropped members
            if len(non_drop_l_indices) > 0:
                nd_l_attn_mask = l_attn_mask.to(clip_l.device)
                prompt_embeds = clip_l(
                    l_tokens.to(clip_l.device), nd_l_attn_mask if apply_lg_attn_mask else None, output_hidden_states=True
                )
                nd_l_pooled = prompt_embeds[0]
                nd_l_out = prompt_embeds.hidden_states[-2]
            if len(non_drop_g_indices) > 0:
                nd_g_attn_mask = g_attn_mask.to(clip_g.device)
                prompt_embeds = clip_g(
                    g_tokens.to(clip_g.device), nd_g_attn_mask if apply_lg_attn_mask else None, output_hidden_states=True
                )
                nd_g_pooled = prompt_embeds[0]
                nd_g_out = prompt_embeds.hidden_states[-2]

            # fill in the dropped members
            if len(non_drop_l_indices) == batch_size:
                l_pooled = nd_l_pooled
                l_out = nd_l_out
            else:
                # model output is always float32 because of the models are wrapped with Accelerator
                l_pooled = torch.zeros((batch_size, 768), device=clip_l.device, dtype=torch.float32)
                l_out = torch.zeros((batch_size, l_seq_len, 768), device=clip_l.device, dtype=torch.float32)
                l_attn_mask = torch.zeros((batch_size, l_seq_len), device=clip_l.device, dtype=l_attn_mask.dtype)
                if len(non_drop_l_indices) > 0:
                    l_pooled[non_drop_l_indices] = nd_l_pooled
                    l_out[non_drop_l_indices] = nd_l_out
                    l_attn_mask[non_drop_l_indices] = nd_l_attn_mask

            if len(non_drop_g_indices) == batch_size:
                g_pooled = nd_g_pooled
                g_out = nd_g_out
            else:
                g_pooled = torch.zeros((batch_size, 1280), device=clip_g.device, dtype=torch.float32)
                g_out = torch.zeros((batch_size, g_seq_len, 1280), device=clip_g.device, dtype=torch.float32)
                g_attn_mask = torch.zeros((batch_size, g_seq_len), device=clip_g.device, dtype=g_attn_mask.dtype)
                if len(non_drop_g_indices) > 0:
                    g_pooled[non_drop_g_indices] = nd_g_pooled
                    g_out[non_drop_g_indices] = nd_g_out
                    g_attn_mask[non_drop_g_indices] = nd_g_attn_mask

            lg_pooled = torch.cat((l_pooled, g_pooled), dim=-1)
            lg_out = torch.cat([l_out, g_out], dim=-1)

        if t5xxl is None or t5_tokens is None:
            t5_out = None
            t5_attn_mask = None
        else:
            # drop some members of the batch: we do not call t5xxl for dropped members
            batch_size, t5_seq_len = t5_tokens.shape
            non_drop_t5_indices = []
            for i in range(t5_tokens.shape[0]):
                drop_t5 = enable_dropout and (self.t5_dropout_rate > 0.0 and random.random() < self.t5_dropout_rate)
                if not drop_t5:
                    non_drop_t5_indices.append(i)

            # filter out dropped members
            if len(non_drop_t5_indices) > 0 and len(non_drop_t5_indices) < batch_size:
                t5_tokens = t5_tokens[non_drop_t5_indices]
                t5_attn_mask = t5_attn_mask[non_drop_t5_indices]

            # call t5xxl for non-dropped members
            if len(non_drop_t5_indices) > 0:
                nd_t5_attn_mask = t5_attn_mask.to(t5xxl.device)
                nd_t5_out, _ = t5xxl(
                    t5_tokens.to(t5xxl.device),
                    nd_t5_attn_mask if apply_t5_attn_mask else None,
                    return_dict=False,
                    output_hidden_states=True,
                )

            # fill in the dropped members
            if len(non_drop_t5_indices) == batch_size:
                t5_out = nd_t5_out
            else:
                t5_out = torch.zeros((batch_size, t5_seq_len, 4096), device=t5xxl.device, dtype=torch.float32)
                t5_attn_mask = torch.zeros((batch_size, t5_seq_len), device=t5xxl.device, dtype=t5_attn_mask.dtype)
                if len(non_drop_t5_indices) > 0:
                    t5_out[non_drop_t5_indices] = nd_t5_out
                    t5_attn_mask[non_drop_t5_indices] = nd_t5_attn_mask

        # masks are used for attention masking in transformer
        return [lg_out, t5_out, lg_pooled, l_attn_mask, g_attn_mask, t5_attn_mask]

    def drop_cached_text_encoder_outputs(
        self,
        lg_out: torch.Tensor,
        t5_out: torch.Tensor,
        lg_pooled: torch.Tensor,
        l_attn_mask: torch.Tensor,
        g_attn_mask: torch.Tensor,
        t5_attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # dropout: if enable_dropout is True, dropout is not applied. dropout means zeroing out embeddings
        if lg_out is not None:
            for i in range(lg_out.shape[0]):
                drop_l = self.l_dropout_rate > 0.0 and random.random() < self.l_dropout_rate
                if drop_l:
                    lg_out[i, :, :768] = torch.zeros_like(lg_out[i, :, :768])
                    lg_pooled[i, :768] = torch.zeros_like(lg_pooled[i, :768])
                    if l_attn_mask is not None:
                        l_attn_mask[i] = torch.zeros_like(l_attn_mask[i])
                drop_g = self.g_dropout_rate > 0.0 and random.random() < self.g_dropout_rate
                if drop_g:
                    lg_out[i, :, 768:] = torch.zeros_like(lg_out[i, :, 768:])
                    lg_pooled[i, 768:] = torch.zeros_like(lg_pooled[i, 768:])
                    if g_attn_mask is not None:
                        g_attn_mask[i] = torch.zeros_like(g_attn_mask[i])

        if t5_out is not None:
            for i in range(t5_out.shape[0]):
                drop_t5 = self.t5_dropout_rate > 0.0 and random.random() < self.t5_dropout_rate
                if drop_t5:
                    t5_out[i] = torch.zeros_like(t5_out[i])
                    if t5_attn_mask is not None:
                        t5_attn_mask[i] = torch.zeros_like(t5_attn_mask[i])

        return [lg_out, t5_out, lg_pooled, l_attn_mask, g_attn_mask, t5_attn_mask]

    def concat_encodings(
        self, lg_out: torch.Tensor, t5_out: Optional[torch.Tensor], lg_pooled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        if t5_out is None:
            t5_out = torch.zeros((lg_out.shape[0], 77, 4096), device=lg_out.device, dtype=lg_out.dtype)
        return torch.cat([lg_out, t5_out], dim=-2), lg_pooled


class Sd3TextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    SD3_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_sd3_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        apply_lg_attn_mask: bool = False,
        apply_t5_attn_mask: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)
        self.apply_lg_attn_mask = apply_lg_attn_mask
        self.apply_t5_attn_mask = apply_t5_attn_mask

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + Sd3TextEncoderOutputsCachingStrategy.SD3_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "lg_out" not in npz:
                return False
            if "lg_pooled" not in npz:
                return False
            if "clip_l_attn_mask" not in npz or "clip_g_attn_mask" not in npz:  # necessary even if not used
                return False
            if "apply_lg_attn_mask" not in npz:
                return False
            if "t5_out" not in npz:
                return False
            if "t5_attn_mask" not in npz:
                return False
            npz_apply_lg_attn_mask = npz["apply_lg_attn_mask"]
            if npz_apply_lg_attn_mask != self.apply_lg_attn_mask:
                return False
            if "apply_t5_attn_mask" not in npz:
                return False
            npz_apply_t5_attn_mask = npz["apply_t5_attn_mask"]
            if npz_apply_t5_attn_mask != self.apply_t5_attn_mask:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        lg_out = data["lg_out"]
        lg_pooled = data["lg_pooled"]
        t5_out = data["t5_out"]

        l_attn_mask = data["clip_l_attn_mask"]
        g_attn_mask = data["clip_g_attn_mask"]
        t5_attn_mask = data["t5_attn_mask"]

        # apply_t5_attn_mask and apply_lg_attn_mask are same as self.apply_t5_attn_mask and self.apply_lg_attn_mask
        return [lg_out, t5_out, lg_pooled, l_attn_mask, g_attn_mask, t5_attn_mask]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        sd3_text_encoding_strategy: Sd3TextEncodingStrategy = text_encoding_strategy
        captions = [info.caption for info in infos]

        tokens_and_masks = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            # always disable dropout during caching
            lg_out, t5_out, lg_pooled, l_attn_mask, g_attn_mask, t5_attn_mask = sd3_text_encoding_strategy.encode_tokens(
                tokenize_strategy,
                models,
                tokens_and_masks,
                apply_lg_attn_mask=self.apply_lg_attn_mask,
                apply_t5_attn_mask=self.apply_t5_attn_mask,
                enable_dropout=False,
            )

        if lg_out.dtype == torch.bfloat16:
            lg_out = lg_out.float()
        if lg_pooled.dtype == torch.bfloat16:
            lg_pooled = lg_pooled.float()
        if t5_out.dtype == torch.bfloat16:
            t5_out = t5_out.float()

        lg_out = lg_out.cpu().numpy()
        lg_pooled = lg_pooled.cpu().numpy()
        t5_out = t5_out.cpu().numpy()

        l_attn_mask = tokens_and_masks[3].cpu().numpy()
        g_attn_mask = tokens_and_masks[4].cpu().numpy()
        t5_attn_mask = tokens_and_masks[5].cpu().numpy()

        for i, info in enumerate(infos):
            lg_out_i = lg_out[i]
            t5_out_i = t5_out[i]
            lg_pooled_i = lg_pooled[i]
            l_attn_mask_i = l_attn_mask[i]
            g_attn_mask_i = g_attn_mask[i]
            t5_attn_mask_i = t5_attn_mask[i]
            apply_lg_attn_mask = self.apply_lg_attn_mask
            apply_t5_attn_mask = self.apply_t5_attn_mask

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    lg_out=lg_out_i,
                    lg_pooled=lg_pooled_i,
                    t5_out=t5_out_i,
                    clip_l_attn_mask=l_attn_mask_i,
                    clip_g_attn_mask=g_attn_mask_i,
                    t5_attn_mask=t5_attn_mask_i,
                    apply_lg_attn_mask=apply_lg_attn_mask,
                    apply_t5_attn_mask=apply_t5_attn_mask,
                )
            else:
                # it's fine that attn mask is not None. it's overwritten before calling the model if necessary
                info.text_encoder_outputs = (lg_out_i, t5_out_i, lg_pooled_i, l_attn_mask_i, g_attn_mask_i, t5_attn_mask_i)


class Sd3LatentsCachingStrategy(LatentsCachingStrategy):
    SD3_LATENTS_NPZ_SUFFIX = "_sd3.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return Sd3LatentsCachingStrategy.SD3_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + Sd3LatentsCachingStrategy.SD3_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)  # support multi-resolution

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).to("cpu")
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)
