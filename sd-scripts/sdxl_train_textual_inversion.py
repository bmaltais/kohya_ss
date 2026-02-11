import argparse
import os
from typing import Optional, Union

import regex

import torch
from library.device_utils import init_ipex

init_ipex()

from library import sdxl_model_util, sdxl_train_util, strategy_sd, strategy_sdxl, train_util
import train_textual_inversion


class SdxlTextualInversionTrainer(train_textual_inversion.TextualInversionTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def assert_extra_args(self, args, train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset], val_dataset_group: Optional[train_util.DatasetGroup]):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        sdxl_train_util.verify_sdxl_training_args(args, supportTextEncoderCaching=False)

        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def get_tokenize_strategy(self, args):
        return strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sdxl.SdxlTokenizeStrategy):
        return [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_sdxl.SdxlTextEncodingStrategy()

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def sample_images(
        self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoders, unet, prompt_replacement
    ):
        sdxl_train_util.sample_images(
            accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoders, unet, prompt_replacement
        )

    def save_weights(self, file, updated_embs, save_dtype, metadata):
        state_dict = {"clip_l": updated_embs[0], "clip_g": updated_embs[1]}

        if save_dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(save_dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            data = load_file(file)
        else:
            data = torch.load(file, map_location="cpu")

        emb_l = data.get("clip_l", None)  # ViT-L text encoder 1
        emb_g = data.get("clip_g", None)  # BiG-G text encoder 2

        assert (
            emb_l is not None or emb_g is not None
        ), f"weight file does not contains weights for text encoder 1 or 2 / 重みファイルにテキストエンコーダー1または2の重みが含まれていません: {file}"

        return [emb_l, emb_g]


def setup_parser() -> argparse.ArgumentParser:
    parser = train_textual_inversion.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser, support_text_encoder_caching=False)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = SdxlTextualInversionTrainer()
    trainer.train(args)
