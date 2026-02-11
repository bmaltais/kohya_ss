import argparse
import copy
import math
import random
from typing import Any, Optional, Union

import torch
from accelerate import Accelerator
from library import sd3_models, strategy_sd3, utils
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import flux_models, flux_train_utils, flux_utils, sd3_train_utils, sd3_utils, strategy_base, strategy_sd3, train_util
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class Sd3NetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        # super().assert_extra_args(args, train_dataset_group)
        # sdxl_train_util.verify_sdxl_training_args(args)

        if args.fp8_base_unet:
            args.fp8_base = True  # if fp8_base_unet is enabled, fp8_base is also enabled for SD3

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning(
                "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled / cache_text_encoder_outputs_to_diskが有効になっているため、cache_text_encoder_outputsも有効になります"
            )
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        # prepare CLIP-L/CLIP-G/T5XXL training flags
        self.train_clip = not args.network_train_unet_only
        self.train_t5xxl = False  # default is False even if args.network_train_unet_only is False

        if args.max_token_length is not None:
            logger.warning("max_token_length is not used in Flux training / max_token_lengthはFluxのトレーニングでは使用されません")

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing / blocks_to_swapはcpu_offload_checkpointingと併用できません"

        train_dataset_group.verify_bucket_reso_steps(32)  # TODO check this
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)  # TODO check this

        # enumerate resolutions from dataset for positional embeddings
        resolutions = train_dataset_group.get_resolutions()
        if val_dataset_group is not None:
            resolutions = resolutions + val_dataset_group.get_resolutions()
        self.resolutions = resolutions

    def load_target_model(self, args, weight_dtype, accelerator):
        # currently offload to cpu for some models

        # if the file is fp8 and we are using fp8_base, we can load it as is (fp8)
        loading_dtype = None if args.fp8_base else weight_dtype

        # if we load to cpu, flux.to(fp8) takes a long time, so we should load to gpu in future
        state_dict = utils.load_safetensors(
            args.pretrained_model_name_or_path, "cpu", disable_mmap=args.disable_mmap_load_safetensors, dtype=loading_dtype
        )
        mmdit = sd3_utils.load_mmdit(state_dict, loading_dtype, "cpu")
        self.model_type = mmdit.model_type
        mmdit.set_pos_emb_random_crop_rate(args.pos_emb_random_crop_rate)

        # set resolutions for positional embeddings
        if args.enable_scaled_pos_embed:
            latent_sizes = [round(math.sqrt(res[0] * res[1])) // 8 for res in self.resolutions]  # 8 is stride for latent
            latent_sizes = list(set(latent_sizes))  # remove duplicates
            logger.info(f"Prepare scaled positional embeddings for resolutions: {self.resolutions}, sizes: {latent_sizes}")
            mmdit.enable_scaled_pos_embed(True, latent_sizes)

        if args.fp8_base:
            # check dtype of model
            if mmdit.dtype == torch.float8_e4m3fnuz or mmdit.dtype == torch.float8_e5m2 or mmdit.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {mmdit.dtype}")
            elif mmdit.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 SD3 model")
            else:
                logger.info(
                    "Cast SD3 model to fp8. This may take a while. You can reduce the time by using fp8 checkpoint."
                    " / SD3モデルをfp8に変換しています。これには時間がかかる場合があります。fp8チェックポイントを使用することで時間を短縮できます。"
                )
                mmdit.to(torch.float8_e4m3fn)
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            mmdit.enable_block_swap(args.blocks_to_swap, accelerator.device)

        clip_l = sd3_utils.load_clip_l(
            args.clip_l, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors, state_dict=state_dict
        )
        clip_l.eval()
        clip_g = sd3_utils.load_clip_g(
            args.clip_g, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors, state_dict=state_dict
        )
        clip_g.eval()

        # if the file is fp8 and we are using fp8_base (not unet), we can load it as is (fp8)
        if args.fp8_base and not args.fp8_base_unet:
            loading_dtype = None  # as is
        else:
            loading_dtype = weight_dtype

        # loading t5xxl to cpu takes a long time, so we should load to gpu in future
        t5xxl = sd3_utils.load_t5xxl(
            args.t5xxl, loading_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors, state_dict=state_dict
        )
        t5xxl.eval()
        if args.fp8_base and not args.fp8_base_unet:
            # check dtype of model
            if t5xxl.dtype == torch.float8_e4m3fnuz or t5xxl.dtype == torch.float8_e5m2 or t5xxl.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {t5xxl.dtype}")
            elif t5xxl.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 T5XXL model")

        vae = sd3_utils.load_vae(
            args.vae, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors, state_dict=state_dict
        )

        return mmdit.model_type, [clip_l, clip_g, t5xxl], vae, mmdit

    def get_tokenize_strategy(self, args):
        logger.info(f"t5xxl_max_token_length: {args.t5xxl_max_token_length}")
        return strategy_sd3.Sd3TokenizeStrategy(args.t5xxl_max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sd3.Sd3TokenizeStrategy):
        return [tokenize_strategy.clip_l, tokenize_strategy.clip_g, tokenize_strategy.t5xxl]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd3.Sd3LatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_sd3.Sd3TextEncodingStrategy(
            args.apply_lg_attn_mask,
            args.apply_t5_attn_mask,
            args.clip_l_dropout_rate,
            args.clip_g_dropout_rate,
            args.t5_dropout_rate,
        )

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        # check t5xxl is trained or not
        self.train_t5xxl = network.train_t5xxl

        if self.train_t5xxl and args.cache_text_encoder_outputs:
            raise ValueError(
                "T5XXL is trained, so cache_text_encoder_outputs cannot be used / T5XXL学習時はcache_text_encoder_outputsは使用できません"
            )

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            if self.train_clip and not self.train_t5xxl:
                return text_encoders[0:2] + [None]  # only CLIP-L/CLIP-G is needed for encoding because T5XXL is cached
            else:
                return None  # no text encoders are needed for encoding because both are cached
        else:
            return text_encoders  # CLIP-L, CLIP-G and T5XXL are needed for encoding

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [self.train_clip, self.train_clip, self.train_t5xxl]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            # if the text encoders is trained, we need tokenization, so is_partial is True
            return strategy_sd3.Sd3TextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                is_partial=self.train_clip or self.train_t5xxl,
                apply_lg_attn_mask=args.apply_lg_attn_mask,
                apply_t5_attn_mask=args.apply_t5_attn_mask,
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
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
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)  # always not fp8
            text_encoders[2].to(accelerator.device)  # may be fp8

            if text_encoders[2].dtype == torch.float8_e4m3fn:
                # if we load fp8 weights, the model is already fp8, so we use it as is
                self.prepare_text_encoder_fp8(2, text_encoders[2], text_encoders[2].dtype, weight_dtype)
            else:
                # otherwise, we need to convert it to target dtype
                text_encoders[2].to(weight_dtype)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")

                tokenize_strategy: strategy_sd3.Sd3TokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy: strategy_sd3.Sd3TextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy()

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"cache Text Encoder outputs for prompt: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy,
                                    text_encoders,
                                    tokens_and_masks,
                                    args.apply_lg_attn_mask,
                                    args.apply_t5_attn_mask,
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # move back to cpu
            if not self.is_train_text_encoder(args):
                logger.info("move CLIP-L back to cpu")
                text_encoders[0].to("cpu")
                logger.info("move CLIP-G back to cpu")
                text_encoders[1].to("cpu")
            logger.info("move t5XXL back to cpu")
            text_encoders[2].to("cpu")
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            text_encoders[2].to(accelerator.device)

    # def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
    #     noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

    #     # get size embeddings
    #     orig_size = batch["original_sizes_hw"]
    #     crop_size = batch["crop_top_lefts"]
    #     target_size = batch["target_sizes_hw"]
    #     embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

    #     # concat embeddings
    #     encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
    #     vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
    #     text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

    #     noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
    #     return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, mmdit):
        text_encoders = text_encoder  # for compatibility
        text_encoders = self.get_models_for_text_encoding(args, accelerator, text_encoders)

        sd3_train_utils.sample_images(
            accelerator, args, epoch, global_step, mmdit, vae, text_encoders, self.sample_prompts_te_outputs
        )

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        # this scheduler is not used in training, but used  to get num_train_timesteps etc.
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.training_shift)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        return vae.encode(images)

    def shift_scale_latents(self, args, latents):
        return sd3_models.SDVAE.process_in(latents)

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet: flux_models.Flux,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        # get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = sd3_train_utils.get_noisy_model_input_and_timesteps(
            args, latents, noise, accelerator.device, weight_dtype
        )

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Predict the noise residual
        lg_out, t5_out, lg_pooled, l_attn_mask, g_attn_mask, t5_attn_mask = text_encoder_conds
        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        context, lg_pooled = text_encoding_strategy.concat_encodings(lg_out, t5_out, lg_pooled)
        if not args.apply_lg_attn_mask:
            l_attn_mask = None
            g_attn_mask = None
        if not args.apply_t5_attn_mask:
            t5_attn_mask = None

        # call model
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            # TODO support attention mask
            model_pred = unet(noisy_model_input, timesteps, context=context, y=lg_pooled)

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_model_input

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = sd3_train_utils.compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        # flow matching loss
        target = latents

        # differential output preservation
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                with torch.no_grad(), accelerator.autocast():
                    model_pred_prior = unet(
                        noisy_model_input[diff_output_pr_indices],
                        timesteps[diff_output_pr_indices],
                        context=context[diff_output_pr_indices],
                        y=lg_pooled[diff_output_pr_indices],
                    )
                network.set_multiplier(1.0)  # may be overwritten by "network_multipliers" in the next step

                model_pred_prior = model_pred_prior * (-sigmas[diff_output_pr_indices]) + noisy_model_input[diff_output_pr_indices]

                # weighting for differential output preservation is not needed because it is already applied

                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)

        return model_pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, False, True, False, sd3=self.model_type)

    def update_metadata(self, metadata, args):
        metadata["ss_apply_lg_attn_mask"] = args.apply_lg_attn_mask
        metadata["ss_apply_t5_attn_mask"] = args.apply_t5_attn_mask
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        if index == 0 or index == 1:  # CLIP-L/CLIP-G
            return super().prepare_text_encoder_grad_ckpt_workaround(index, text_encoder)
        else:  # T5XXL
            text_encoder.encoder.embed_tokens.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        if index == 0 or index == 1:  # CLIP-L/CLIP-G
            clip_type = "CLIP-L" if index == 0 else "CLIP-G"
            logger.info(f"prepare CLIP-{clip_type} for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}")
            text_encoder.to(te_weight_dtype)  # fp8
            text_encoder.text_model.embeddings.to(dtype=weight_dtype)
        else:  # T5XXL

            def prepare_fp8(text_encoder, target_dtype):
                def forward_hook(module):
                    def forward(hidden_states):
                        hidden_gelu = module.act(module.wi_0(hidden_states))
                        hidden_linear = module.wi_1(hidden_states)
                        hidden_states = hidden_gelu * hidden_linear
                        hidden_states = module.dropout(hidden_states)

                        hidden_states = module.wo(hidden_states)
                        return hidden_states

                    return forward

                for module in text_encoder.modules():
                    if module.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                        # print("set", module.__class__.__name__, "to", target_dtype)
                        module.to(target_dtype)
                    if module.__class__.__name__ in ["T5DenseGatedActDense"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.forward = forward_hook(module)

            if flux_utils.get_t5xxl_actual_dtype(text_encoder) == torch.float8_e4m3fn and text_encoder.dtype == weight_dtype:
                logger.info(f"T5XXL already prepared for fp8")
            else:
                logger.info(f"prepare T5XXL for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}, add hooks")
                text_encoder.to(te_weight_dtype)  # fp8
                prepare_fp8(text_encoder, weight_dtype)

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True):
        # drop cached text encoder outputs: in validation, we drop cached outputs deterministically by fixed seed
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encodoing_strategy: strategy_sd3.Sd3TextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy()
            text_encoder_outputs_list = text_encodoing_strategy.drop_cached_text_encoder_outputs(*text_encoder_outputs_list)
            batch["text_encoder_outputs_list"] = text_encoder_outputs_list

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        # if we doesn't swap blocks, we can move the model to device
        mmdit: sd3_models.MMDiT = unet
        mmdit = accelerator.prepare(mmdit, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(mmdit).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
        accelerator.unwrap_model(mmdit).prepare_block_swap_before_forward()

        return mmdit


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    sd3_train_utils.add_sd3_training_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = Sd3NetworkTrainer()
    trainer.train(args)
