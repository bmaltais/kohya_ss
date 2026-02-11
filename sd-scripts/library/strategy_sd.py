import glob
import os
from typing import Any, List, Optional, Tuple, Union

import torch
from transformers import CLIPTokenizer
from library import train_util
from library.strategy_base import LatentsCachingStrategy, TokenizeStrategy, TextEncodingStrategy
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


TOKENIZER_ID = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_ID = "stabilityai/stable-diffusion-2"  # ここからtokenizerだけ使う v2とv2.1はtokenizer仕様は同じ


class SdTokenizeStrategy(TokenizeStrategy):
    def __init__(self, v2: bool, max_length: Optional[int], tokenizer_cache_dir: Optional[str] = None) -> None:
        """
        max_length does not include <BOS> and <EOS> (None, 75, 150, 225)
        """
        logger.info(f"Using {'v2' if v2 else 'v1'} tokenizer")
        if v2:
            self.tokenizer = self._load_tokenizer(
                CLIPTokenizer, V2_STABLE_DIFFUSION_ID, subfolder="tokenizer", tokenizer_cache_dir=tokenizer_cache_dir
            )
        else:
            self.tokenizer = self._load_tokenizer(CLIPTokenizer, TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)

        if max_length is None:
            self.max_length = self.tokenizer.model_max_length
        else:
            self.max_length = max_length + 2

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        return [torch.stack([self._get_input_ids(self.tokenizer, t, self.max_length) for t in text], dim=0)]

    def tokenize_with_weights(self, text: str | List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        text = [text] if isinstance(text, str) else text
        tokens_list = []
        weights_list = []
        for t in text:
            tokens, weights = self._get_input_ids(self.tokenizer, t, self.max_length, weighted=True)
            tokens_list.append(tokens)
            weights_list.append(weights)
        return [torch.stack(tokens_list, dim=0)], [torch.stack(weights_list, dim=0)]


class SdTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self, clip_skip: Optional[int] = None) -> None:
        self.clip_skip = clip_skip

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        text_encoder = models[0]
        tokens = tokens[0]
        sd_tokenize_strategy = tokenize_strategy  # type: SdTokenizeStrategy

        # tokens: b,n,77
        b_size = tokens.size()[0]
        max_token_length = tokens.size()[1] * tokens.size()[2]
        model_max_length = sd_tokenize_strategy.tokenizer.model_max_length
        tokens = tokens.reshape((-1, model_max_length))  # batch_size*3, 77

        tokens = tokens.to(text_encoder.device)

        if self.clip_skip is None:
            encoder_hidden_states = text_encoder(tokens)[0]
        else:
            enc_out = text_encoder(tokens, output_hidden_states=True, return_dict=True)
            encoder_hidden_states = enc_out["hidden_states"][-self.clip_skip]
            encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

        # bs*3, 77, 768 or 1024
        encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

        if max_token_length != model_max_length:
            v1 = sd_tokenize_strategy.tokenizer.pad_token_id == sd_tokenize_strategy.tokenizer.eos_token_id
            if not v1:
                # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
                states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
                for i in range(1, max_token_length, model_max_length):
                    chunk = encoder_hidden_states[:, i : i + model_max_length - 2]  # <BOS> の後から 最後の前まで
                    if i > 0:
                        for j in range(len(chunk)):
                            if tokens[j, 1] == sd_tokenize_strategy.tokenizer.eos_token:
                                # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                                chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
                    states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
                states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
                encoder_hidden_states = torch.cat(states_list, dim=1)
            else:
                # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
                states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
                for i in range(1, max_token_length, model_max_length):
                    states_list.append(encoder_hidden_states[:, i : i + model_max_length - 2])  # <BOS> の後から <EOS> の前まで
                states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
                encoder_hidden_states = torch.cat(states_list, dim=1)

        return [encoder_hidden_states]

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens_list: List[torch.Tensor],
        weights_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        encoder_hidden_states = self.encode_tokens(tokenize_strategy, models, tokens_list)[0]

        weights = weights_list[0].to(encoder_hidden_states.device)

        # apply weights
        if weights.shape[1] == 1:  # no max_token_length
            # weights: ((b, 1, 77), (b, 1, 77)), hidden_states: (b, 77, 768), (b, 77, 768)
            encoder_hidden_states = encoder_hidden_states * weights.squeeze(1).unsqueeze(2)
        else:
            # weights: ((b, n, 77), (b, n, 77)), hidden_states: (b, n*75+2, 768), (b, n*75+2, 768)
            for i in range(weights.shape[1]):
                encoder_hidden_states[:, i * 75 + 1 : i * 75 + 76] = encoder_hidden_states[:, i * 75 + 1 : i * 75 + 76] * weights[
                    :, i, 1:-1
                ].unsqueeze(-1)

        return [encoder_hidden_states]


class SdSdxlLatentsCachingStrategy(LatentsCachingStrategy):
    # sd and sdxl share the same strategy. we can make them separate, but the difference is only the suffix.
    # and we keep the old npz for the backward compatibility.

    SD_OLD_LATENTS_NPZ_SUFFIX = ".npz"
    SD_LATENTS_NPZ_SUFFIX = "_sd.npz"
    SDXL_LATENTS_NPZ_SUFFIX = "_sdxl.npz"

    def __init__(self, sd: bool, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
        self.sd = sd
        self.suffix = (
            SdSdxlLatentsCachingStrategy.SD_LATENTS_NPZ_SUFFIX if sd else SdSdxlLatentsCachingStrategy.SDXL_LATENTS_NPZ_SUFFIX
        )
    
    @property
    def cache_suffix(self) -> str:
        return self.suffix

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        # support old .npz
        old_npz_file = os.path.splitext(absolute_path)[0] + SdSdxlLatentsCachingStrategy.SD_OLD_LATENTS_NPZ_SUFFIX
        if os.path.exists(old_npz_file):
            return old_npz_file
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.suffix

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask)

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).latent_dist.sample()
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop)

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)
