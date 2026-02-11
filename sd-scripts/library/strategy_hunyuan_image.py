import os
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
from transformers import AutoTokenizer, Qwen2Tokenizer

from library import hunyuan_image_text_encoder, hunyuan_image_vae, train_util
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class HunyuanImageTokenizeStrategy(TokenizeStrategy):
    def __init__(self, tokenizer_cache_dir: Optional[str] = None) -> None:
        self.vlm_tokenizer = self._load_tokenizer(
            Qwen2Tokenizer, hunyuan_image_text_encoder.QWEN_2_5_VL_IMAGE_ID, tokenizer_cache_dir=tokenizer_cache_dir
        )
        self.byt5_tokenizer = self._load_tokenizer(
            AutoTokenizer, hunyuan_image_text_encoder.BYT5_TOKENIZER_PATH, subfolder="", tokenizer_cache_dir=tokenizer_cache_dir
        )

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        vlm_tokens, vlm_mask = hunyuan_image_text_encoder.get_qwen_tokens(self.vlm_tokenizer, text)

        # byt5_tokens, byt5_mask = hunyuan_image_text_encoder.get_byt5_text_tokens(self.byt5_tokenizer, text)
        byt5_tokens = []
        byt5_mask = []
        for t in text:
            tokens, mask = hunyuan_image_text_encoder.get_byt5_text_tokens(self.byt5_tokenizer, t)
            if tokens is None:
                tokens = torch.zeros((1, 1), dtype=torch.long)
                mask = torch.zeros((1, 1), dtype=torch.long)
            byt5_tokens.append(tokens)
            byt5_mask.append(mask)
        max_len = max([m.shape[1] for m in byt5_mask])
        byt5_tokens = torch.cat([torch.nn.functional.pad(t, (0, max_len - t.shape[1]), value=0) for t in byt5_tokens], dim=0)
        byt5_mask = torch.cat([torch.nn.functional.pad(m, (0, max_len - m.shape[1]), value=0) for m in byt5_mask], dim=0)

        return [vlm_tokens, vlm_mask, byt5_tokens, byt5_mask]


class HunyuanImageTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self) -> None:
        pass

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        vlm_tokens, vlm_mask, byt5_tokens, byt5_mask = tokens

        qwen2vlm, byt5 = models

        # autocast and no_grad are handled in hunyuan_image_text_encoder
        vlm_embed, vlm_mask = hunyuan_image_text_encoder.get_qwen_prompt_embeds_from_tokens(qwen2vlm, vlm_tokens, vlm_mask)

        # ocr_mask, byt5_embed, byt5_mask = hunyuan_image_text_encoder.get_byt5_prompt_embeds_from_tokens(
        #     byt5, byt5_tokens, byt5_mask
        # )
        ocr_mask, byt5_embed, byt5_updated_mask = [], [], []
        for i in range(byt5_tokens.shape[0]):
            ocr_m, byt5_e, byt5_m = hunyuan_image_text_encoder.get_byt5_prompt_embeds_from_tokens(
                byt5, byt5_tokens[i : i + 1], byt5_mask[i : i + 1]
            )
            ocr_mask.append(torch.zeros((1,), dtype=torch.long) + (1 if ocr_m[0] else 0))  # 1 or 0
            byt5_embed.append(byt5_e)
            byt5_updated_mask.append(byt5_m)

        ocr_mask = torch.cat(ocr_mask, dim=0).to(torch.bool)  # [B]
        byt5_embed = torch.cat(byt5_embed, dim=0)
        byt5_updated_mask = torch.cat(byt5_updated_mask, dim=0)

        return [vlm_embed, vlm_mask, byt5_embed, byt5_updated_mask, ocr_mask]


class HunyuanImageTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    HUNYUAN_IMAGE_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_hi_te.npz"

    def __init__(
        self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool, is_partial: bool = False
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return (
            os.path.splitext(image_abs_path)[0]
            + HunyuanImageTextEncoderOutputsCachingStrategy.HUNYUAN_IMAGE_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX
        )

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "vlm_embed" not in npz:
                return False
            if "vlm_mask" not in npz:
                return False
            if "byt5_embed" not in npz:
                return False
            if "byt5_mask" not in npz:
                return False
            if "ocr_mask" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        vln_embed = data["vlm_embed"]
        vlm_mask = data["vlm_mask"]
        byt5_embed = data["byt5_embed"]
        byt5_mask = data["byt5_mask"]
        ocr_mask = data["ocr_mask"]
        return [vln_embed, vlm_mask, byt5_embed, byt5_mask, ocr_mask]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        huyuan_image_text_encoding_strategy: HunyuanImageTextEncodingStrategy = text_encoding_strategy
        captions = [info.caption for info in infos]

        tokens_and_masks = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            vlm_embed, vlm_mask, byt5_embed, byt5_mask, ocr_mask = huyuan_image_text_encoding_strategy.encode_tokens(
                tokenize_strategy, models, tokens_and_masks
            )

        if vlm_embed.dtype == torch.bfloat16:
            vlm_embed = vlm_embed.float()
        if byt5_embed.dtype == torch.bfloat16:
            byt5_embed = byt5_embed.float()

        vlm_embed = vlm_embed.cpu().numpy()
        vlm_mask = vlm_mask.cpu().numpy()
        byt5_embed = byt5_embed.cpu().numpy()
        byt5_mask = byt5_mask.cpu().numpy()
        ocr_mask = ocr_mask.cpu().numpy()

        for i, info in enumerate(infos):
            vlm_embed_i = vlm_embed[i]
            vlm_mask_i = vlm_mask[i]
            byt5_embed_i = byt5_embed[i]
            byt5_mask_i = byt5_mask[i]
            ocr_mask_i = ocr_mask[i]

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    vlm_embed=vlm_embed_i,
                    vlm_mask=vlm_mask_i,
                    byt5_embed=byt5_embed_i,
                    byt5_mask=byt5_mask_i,
                    ocr_mask=ocr_mask_i,
                )
            else:
                info.text_encoder_outputs = (vlm_embed_i, vlm_mask_i, byt5_embed_i, byt5_mask_i, ocr_mask_i)


class HunyuanImageLatentsCachingStrategy(LatentsCachingStrategy):
    HUNYUAN_IMAGE_LATENTS_NPZ_SUFFIX = "_hi.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return HunyuanImageLatentsCachingStrategy.HUNYUAN_IMAGE_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + HunyuanImageLatentsCachingStrategy.HUNYUAN_IMAGE_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(32, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(32, npz_path, bucket_reso)  # support multi-resolution

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(
        self, vae: hunyuan_image_vae.HunyuanVAE2D, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool
    ):
        # encode_by_vae = lambda img_tensor: vae.encode(img_tensor).sample()
        def encode_by_vae(img_tensor):
            # no_grad is handled in _default_cache_batch_latents
            nonlocal vae
            with torch.autocast(device_type=vae.device.type, dtype=vae.dtype):
                return vae.encode(img_tensor).sample()

        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)
