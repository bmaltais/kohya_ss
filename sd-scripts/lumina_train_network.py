import argparse
import copy
from typing import Any, Tuple

import torch

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

from torch import Tensor
from accelerate import Accelerator


import train_network
from library import (
    lumina_models,
    lumina_util,
    lumina_train_util,
    sd3_train_utils,
    strategy_base,
    strategy_lumina,
    train_util,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class LuminaNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_swapping_blocks: bool = False

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("Enabling cache_text_encoder_outputs due to disk caching")
            args.cache_text_encoder_outputs = True

        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

        self.train_gemma2 = not args.network_train_unet_only

    def load_target_model(self, args, weight_dtype, accelerator):
        loading_dtype = None if args.fp8_base else weight_dtype

        model = lumina_util.load_lumina_model(
            args.pretrained_model_name_or_path,
            loading_dtype,
            torch.device("cpu"),
            disable_mmap=args.disable_mmap_load_safetensors,
            use_flash_attn=args.use_flash_attn,
            use_sage_attn=args.use_sage_attn,
        )

        if args.fp8_base:
            # check dtype of model
            if model.dtype == torch.float8_e4m3fnuz or model.dtype == torch.float8_e5m2 or model.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {model.dtype}")
            elif model.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 Lumina 2 model")
            else:
                logger.info(
                    "Cast Lumina 2 model to fp8. This may take a while. You can reduce the time by using fp8 checkpoint."
                    " / Lumina 2モデルをfp8に変換しています。これには時間がかかる場合があります。fp8チェックポイントを使用することで時間を短縮できます。"
                )
                model.to(torch.float8_e4m3fn)

        if args.blocks_to_swap:
            logger.info(f"Lumina 2: Enabling block swap: {args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)
            self.is_swapping_blocks = True

        gemma2 = lumina_util.load_gemma2(args.gemma2, weight_dtype, "cpu")
        gemma2.eval()
        ae = lumina_util.load_ae(args.ae, weight_dtype, "cpu")

        return lumina_util.MODEL_VERSION_LUMINA_V2, [gemma2], ae, model

    def get_tokenize_strategy(self, args):
        return strategy_lumina.LuminaTokenizeStrategy(args.system_prompt, args.gemma2_max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_lumina.LuminaTokenizeStrategy):
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_lumina.LuminaLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, False)

    def get_text_encoding_strategy(self, args):
        return strategy_lumina.LuminaTextEncodingStrategy()

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [self.train_gemma2]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            # if the text encoders is trained, we need tokenization, so is_partial is True
            return strategy_lumina.LuminaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                is_partial=self.train_gemma2,
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator: Accelerator,
        unet,
        vae,
        text_encoders,
        dataset,
        weight_dtype,
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            logger.info("move text encoders to gpu")
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)  # always not fp8

            if text_encoders[0].dtype == torch.float8_e4m3fn:
                # if we load fp8 weights, the model is already fp8, so we use it as is
                self.prepare_text_encoder_fp8(1, text_encoders[1], text_encoders[1].dtype, weight_dtype)
            else:
                # otherwise, we need to convert it to target dtype
                text_encoders[0].to(weight_dtype)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")

                tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

                assert isinstance(tokenize_strategy, strategy_lumina.LuminaTokenizeStrategy)
                assert isinstance(text_encoding_strategy, strategy_lumina.LuminaTextEncodingStrategy)

                sample_prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in sample_prompts:
                        prompts = [
                            prompt_dict.get("prompt", ""),
                            prompt_dict.get("negative_prompt", ""),
                        ]
                        for i, prompt in enumerate(prompts):
                            if prompt in sample_prompts_te_outputs:
                                continue

                            logger.info(f"cache Text Encoder outputs for prompt: {prompt}")
                            tokens_and_masks = tokenize_strategy.tokenize(prompt, i == 1) # i == 1 means negative prompt
                            sample_prompts_te_outputs[prompt] = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                text_encoders,
                                tokens_and_masks,
                            )

                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # move back to cpu
            if not self.is_train_text_encoder(args):
                logger.info("move Gemma 2 back to cpu")
                text_encoders[0].to("cpu")
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)

    def sample_images(
        self,
        accelerator,
        args,
        epoch,
        global_step,
        device,
        vae,
        tokenizer,
        text_encoder,
        lumina,
    ):
        lumina_train_util.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            lumina,
            vae,
            self.get_models_for_text_encoding(args, accelerator, text_encoder),
            self.sample_prompts_te_outputs,
        )

    # Remaining methods maintain similar structure to flux implementation
    # with Lumina-specific model calls and strategies

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        return vae.encode(images)

    # not sure, they use same flux vae
    def shift_scale_latents(self, args, latents):
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator: Accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds: Tuple[Tensor, Tensor, Tensor],  # (hidden_states, input_ids, attention_masks)
        dit: lumina_models.NextDiT,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        assert isinstance(noise_scheduler, sd3_train_utils.FlowMatchEulerDiscreteScheduler)
        noise = torch.randn_like(latents)
        # get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = lumina_train_util.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Unpack Gemma2 outputs
        gemma2_hidden_states, input_ids, gemma2_attn_mask = text_encoder_conds

        def call_dit(img, gemma2_hidden_states, gemma2_attn_mask, timesteps):
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                # NextDiT forward expects (x, t, cap_feats, cap_mask)
                model_pred = dit(
                    x=img,  # image latents (B, C, H, W)
                    t=1 - timesteps / 1000,  # timesteps需要除以1000来匹配模型预期
                    cap_feats=gemma2_hidden_states,  # Gemma2的hidden states作为caption features
                    cap_mask=gemma2_attn_mask.to(dtype=torch.int32),  # Gemma2的attention mask
                )
            return model_pred

        model_pred = call_dit(
            img=noisy_model_input,
            gemma2_hidden_states=gemma2_hidden_states,
            gemma2_attn_mask=gemma2_attn_mask,
            timesteps=timesteps,
        )

        # apply model prediction type
        model_pred, weighting = lumina_train_util.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)

        # flow matching loss
        target = latents - noise

        # differential output preservation
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                with torch.no_grad():
                    model_pred_prior = call_dit(
                        img=noisy_model_input[diff_output_pr_indices],
                        gemma2_hidden_states=gemma2_hidden_states[diff_output_pr_indices],
                        timesteps=timesteps[diff_output_pr_indices],
                        gemma2_attn_mask=(gemma2_attn_mask[diff_output_pr_indices]),
                    )
                network.set_multiplier(1.0)

                # model_pred_prior = lumina_util.unpack_latents(
                #     model_pred_prior, packed_latent_height, packed_latent_width
                # )
                model_pred_prior, _ = lumina_train_util.apply_model_prediction_type(
                    args,
                    model_pred_prior,
                    noisy_model_input[diff_output_pr_indices],
                    sigmas[diff_output_pr_indices] if sigmas is not None else None,
                )
                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)

        return model_pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, False, True, False, lumina="lumina2")

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_model_prediction_type"] = args.model_prediction_type
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        text_encoder.embed_tokens.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        logger.info(f"prepare Gemma2 for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}")
        text_encoder.to(te_weight_dtype)  # fp8
        text_encoder.embed_tokens.to(dtype=weight_dtype)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        # if we doesn't swap blocks, we can move the model to device
        nextdit = unet
        assert isinstance(nextdit, lumina_models.NextDiT)
        nextdit = accelerator.prepare(nextdit, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(nextdit).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
        accelerator.unwrap_model(nextdit).prepare_block_swap_before_forward()

        return nextdit

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    lumina_train_util.add_lumina_train_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = LuminaNetworkTrainer()
    trainer.train(args)
