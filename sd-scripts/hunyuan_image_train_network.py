import argparse
import copy
import gc
from typing import Any, Optional, Union, cast
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from accelerate import Accelerator, PartialState

from library import flux_utils, hunyuan_image_models, hunyuan_image_vae, strategy_base, train_util
from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import train_network
from library import (
    flux_train_utils,
    hunyuan_image_models,
    hunyuan_image_text_encoder,
    hunyuan_image_utils,
    hunyuan_image_vae,
    sd3_train_utils,
    strategy_base,
    strategy_hunyuan_image,
    train_util,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# region sampling


# TODO commonize with flux_utils
def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    dit: hunyuan_image_models.HYImageDiffusionTransformer,
    vae,
    text_encoders,
    sample_prompts_te_outputs,
    prompt_replacement=None,
):
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            # sample_every_n_steps は無視する
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:  # steps is not divisible or end of epoch
                return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
    if not os.path.isfile(args.sample_prompts) and sample_prompts_te_outputs is None:
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    # unwrap unet and text_encoder(s)
    dit = accelerator.unwrap_model(dit)
    dit = cast(hunyuan_image_models.HYImageDiffusionTransformer, dit)
    dit.switch_block_swap_for_inference()
    if text_encoders is not None:
        text_encoders = [(accelerator.unwrap_model(te) if te is not None else None) for te in text_encoders]
    # print([(te.parameters().__next__().device if te is not None else None) for te in text_encoders])

    prompts = train_util.load_prompts(args.sample_prompts)

    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    # save random state to restore later
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    if distributed_state.num_processes <= 1:
        # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
        with torch.no_grad(), accelerator.autocast():
            for prompt_dict in prompts:
                sample_image_inference(
                    accelerator,
                    args,
                    dit,
                    text_encoders,
                    vae,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    sample_prompts_te_outputs,
                    prompt_replacement,
                )
    else:
        # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
        # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
        per_process_prompts = []  # list of lists
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i :: distributed_state.num_processes])

        with torch.no_grad():
            with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
                for prompt_dict in prompt_dict_lists[0]:
                    sample_image_inference(
                        accelerator,
                        args,
                        dit,
                        text_encoders,
                        vae,
                        save_dir,
                        prompt_dict,
                        epoch,
                        steps,
                        sample_prompts_te_outputs,
                        prompt_replacement,
                    )

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    dit.switch_block_swap_for_training()
    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    dit: hunyuan_image_models.HYImageDiffusionTransformer,
    text_encoders: Optional[list[nn.Module]],
    vae: hunyuan_image_vae.HunyuanVAE2D,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    assert isinstance(prompt_dict, dict)
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 20)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    cfg_scale = prompt_dict.get("scale", 3.5)
    seed = prompt_dict.get("seed")
    prompt: str = prompt_dict.get("prompt", "")
    flow_shift: float = prompt_dict.get("flow_shift", 5.0)
    # sampler_name: str = prompt_dict.get("sample_sampler", args.sample_sampler)

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt is not None:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        torch.cuda.seed()

    if negative_prompt is None:
        negative_prompt = ""
    height = max(64, height - height % 16)  # round to divisible by 16
    width = max(64, width - width % 16)  # round to divisible by 16
    logger.info(f"prompt: {prompt}")
    if cfg_scale != 1.0:
        logger.info(f"negative_prompt: {negative_prompt}")
    elif negative_prompt != "":
        logger.info(f"negative prompt is ignored because scale is 1.0")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    if cfg_scale != 1.0:
        logger.info(f"CFG scale: {cfg_scale}")
    logger.info(f"flow_shift: {flow_shift}")
    # logger.info(f"sample_sampler: {sampler_name}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    # encode prompts
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

    def encode_prompt(prpt):
        text_encoder_conds = []
        if sample_prompts_te_outputs and prpt in sample_prompts_te_outputs:
            text_encoder_conds = sample_prompts_te_outputs[prpt]
            # print(f"Using cached text encoder outputs for prompt: {prpt}")
        if text_encoders is not None:
            # print(f"Encoding prompt: {prpt}")
            tokens_and_masks = tokenize_strategy.tokenize(prpt)
            encoded_text_encoder_conds = encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, tokens_and_masks)

            # if text_encoder_conds is not cached, use encoded_text_encoder_conds
            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                # if encoded_text_encoder_conds is not None, update cached text_encoder_conds
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]
        return text_encoder_conds

    vl_embed, vl_mask, byt5_embed, byt5_mask, ocr_mask = encode_prompt(prompt)
    arg_c = {
        "embed": vl_embed,
        "mask": vl_mask,
        "embed_byt5": byt5_embed,
        "mask_byt5": byt5_mask,
        "ocr_mask": ocr_mask,
        "prompt": prompt,
    }

    # encode negative prompts
    if cfg_scale != 1.0:
        neg_vl_embed, neg_vl_mask, neg_byt5_embed, neg_byt5_mask, neg_ocr_mask = encode_prompt(negative_prompt)
        arg_c_null = {
            "embed": neg_vl_embed,
            "mask": neg_vl_mask,
            "embed_byt5": neg_byt5_embed,
            "mask_byt5": neg_byt5_mask,
            "ocr_mask": neg_ocr_mask,
            "prompt": negative_prompt,
        }
    else:
        arg_c_null = None

    gen_args = SimpleNamespace(
        image_size=(height, width),
        infer_steps=sample_steps,
        flow_shift=flow_shift,
        guidance_scale=cfg_scale,
        fp8=args.fp8_scaled,
        apg_start_step_ocr=38,
        apg_start_step_general=5,
        guidance_rescale=0.0,
        guidance_rescale_apg=0.0,
    )

    from hunyuan_image_minimal_inference import generate_body  # import here to avoid circular import

    dit_is_training = dit.training
    dit.eval()
    x = generate_body(gen_args, dit, arg_c, arg_c_null, accelerator.device, seed)
    if dit_is_training:
        dit.train()
    clean_memory_on_device(accelerator.device)

    # latent to image
    org_vae_device = vae.device  # will be on cpu
    vae.to(accelerator.device)  # distributed_state.device is same as accelerator.device
    with torch.no_grad():
        x = x / vae.scaling_factor
        x = vae.decode(x.to(vae.device, dtype=vae.dtype))
    vae.to(org_vae_device)

    clean_memory_on_device(accelerator.device)

    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    image = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    # adding accelerator.wait_for_everyone() here should sync up and ensure that sample images are saved in the same order as the original prompt list
    # but adding 'enum' to the filename should be enough

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict["enum"]
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    # send images to wandb if enabled
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")

        import wandb

        # not to commit images to avoid inconsistency between training and logging steps
        wandb_tracker.log({f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False)  # positive prompt as a caption


# endregion


class HunyuanImageNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_swapping_blocks: bool = False
        self.rotary_pos_emb_cache = {}

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        # sdxl_train_util.verify_sdxl_training_args(args)

        if args.mixed_precision == "fp16":
            logger.warning(
                "mixed_precision bf16 is recommended for HunyuanImage-2.1 / HunyuanImage-2.1ではmixed_precision bf16が推奨されます"
            )

        if (args.fp8_base or args.fp8_base_unet) and not args.fp8_scaled:
            logger.warning(
                "fp8_base and fp8_base_unet are not supported. Use fp8_scaled instead / fp8_baseとfp8_base_unetはサポートされていません。代わりにfp8_scaledを使用してください"
            )
        if args.fp8_scaled and (args.fp8_base or args.fp8_base_unet):
            logger.info(
                "fp8_scaled is used, so fp8_base and fp8_base_unet are ignored / fp8_scaledが使われているので、fp8_baseとfp8_base_unetは無視されます"
            )
            args.fp8_base = False
            args.fp8_base_unet = False

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning(
                "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled / cache_text_encoder_outputs_to_diskが有効になっているため、cache_text_encoder_outputsも有効になります"
            )
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

        vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
        vl_device = "cpu"  # loading to cpu and move to gpu later in cache_text_encoder_outputs_if_needed
        _, text_encoder_vlm = hunyuan_image_text_encoder.load_qwen2_5_vl(
            args.text_encoder, dtype=vl_dtype, device=vl_device, disable_mmap=args.disable_mmap_load_safetensors
        )
        _, text_encoder_byt5 = hunyuan_image_text_encoder.load_byt5(
            args.byt5, dtype=torch.float16, device=vl_device, disable_mmap=args.disable_mmap_load_safetensors
        )

        vae = hunyuan_image_vae.load_vae(
            args.vae, "cpu", disable_mmap=args.disable_mmap_load_safetensors, chunk_size=args.vae_chunk_size
        )
        vae.to(dtype=torch.float16)  # VAE is always fp16
        vae.eval()

        model_version = hunyuan_image_utils.MODEL_VERSION_2_1
        return model_version, [text_encoder_vlm, text_encoder_byt5], vae, None  # unet will be loaded later

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
        if args.cache_text_encoder_outputs:
            logger.info("Replace text encoders with dummy models to save memory")

            # This doesn't free memory, so we move text encoders to meta device in cache_text_encoder_outputs_if_needed
            text_encoders = [flux_utils.dummy_clip_l() for _ in text_encoders]
            clean_memory_on_device(accelerator.device)
            gc.collect()

        loading_dtype = None if args.fp8_scaled else weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        logger.info(f"Loading DiT model with attn_mode: {attn_mode}, split_attn: {args.split_attn}, fp8_scaled: {args.fp8_scaled}")
        model = hunyuan_image_models.load_hunyuan_image_model(
            accelerator.device,
            args.pretrained_model_name_or_path,
            attn_mode,
            args.split_attn,
            loading_device,
            loading_dtype,
            args.fp8_scaled,
        )

        if self.is_swapping_blocks:
            # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device, supports_backward=True)

        return model, text_encoders

    def get_tokenize_strategy(self, args):
        return strategy_hunyuan_image.HunyuanImageTokenizeStrategy(args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_hunyuan_image.HunyuanImageTokenizeStrategy):
        return [tokenize_strategy.vlm_tokenizer, tokenize_strategy.byt5_tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_hunyuan_image.HunyuanImageLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, False)

    def get_text_encoding_strategy(self, args):
        return strategy_hunyuan_image.HunyuanImageTextEncodingStrategy()

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders are needed for encoding because both are cached
        else:
            return text_encoders

    def get_text_encoders_train_flags(self, args, text_encoders):
        # HunyuanImage-2.1 does not support training VLM or byT5
        return [False, False]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_hunyuan_image.HunyuanImageTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        vlm_device = "cpu" if args.text_encoder_cpu else accelerator.device
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae to cpu to save memory")
                org_vae_device = vae.device
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info(f"move text encoders to {vlm_device} to encode and cache text encoder outputs")
            text_encoders[0].to(vlm_device)
            text_encoders[1].to(vlm_device)

            # VLM (bf16) and byT5 (fp16) are used for encoding, so we cannot use autocast here
            dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")

                tokenize_strategy: strategy_hunyuan_image.HunyuanImageTokenizeStrategy = (
                    strategy_base.TokenizeStrategy.get_strategy()
                )
                text_encoding_strategy: strategy_hunyuan_image.HunyuanImageTextEncodingStrategy = (
                    strategy_base.TextEncodingStrategy.get_strategy()
                )

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"cache Text Encoder outputs for prompt: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, text_encoders, tokens_and_masks
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # text encoders are not needed for training, so we move to meta device
            logger.info("move text encoders to meta device to save memory")
            text_encoders = [te.to("meta") for te in text_encoders]
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae back to original device")
                vae.to(org_vae_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(vlm_device)
            text_encoders[1].to(vlm_device)

    def sample_images(self, accelerator, args, epoch, global_step, device, ae, tokenizer, text_encoder, flux):
        text_encoders = text_encoder  # for compatibility
        text_encoders = self.get_models_for_text_encoding(args, accelerator, text_encoders)

        sample_images(accelerator, args, epoch, global_step, flux, ae, text_encoders, self.sample_prompts_te_outputs)

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae: hunyuan_image_vae.HunyuanVAE2D, images):
        return vae.encode(images).sample()

    def shift_scale_latents(self, args, latents):
        # for encoding, we need to scale the latents
        return latents * hunyuan_image_vae.LATENT_SCALING_FACTOR

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet: hunyuan_image_models.HYImageDiffusionTransformer,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        # get noisy model input and timesteps
        noisy_model_input, _, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )
        # bfloat16 is too low precision for 0-1000 TODO fix get_noisy_model_input_and_timesteps
        timesteps = (sigmas[:, 0, 0, 0] * 1000).to(torch.int64)
        # print(
        #     f"timestep: {timesteps}, noisy_model_input shape: {noisy_model_input.shape}, mean: {noisy_model_input.mean()}, std: {noisy_model_input.std()}"
        # )

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Predict the noise residual
        # ocr_mask is for inference only, so it is not used here
        vlm_embed, vlm_mask, byt5_embed, byt5_mask, ocr_mask = text_encoder_conds

        # print(f"embed shape: {vlm_embed.shape}, mean: {vlm_embed.mean()}, std: {vlm_embed.std()}")
        # print(f"embed_byt5 shape: {byt5_embed.shape}, mean: {byt5_embed.mean()}, std: {byt5_embed.std()}")
        # print(f"latents shape: {latents.shape}, mean: {latents.mean()}, std: {latents.std()}")
        # print(f"mask shape: {vlm_mask.shape}, sum: {vlm_mask.sum()}")
        # print(f"mask_byt5 shape: {byt5_mask.shape}, sum: {byt5_mask.sum()}")
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = unet(
                noisy_model_input, timesteps, vlm_embed, vlm_mask, byt5_embed, byt5_mask  # , self.rotary_pos_emb_cache
            )

        # apply model prediction type
        model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)

        # flow matching loss
        target = noise - latents

        # differential output preservation is not used for HunyuanImage-2.1 currently

        return model_pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec_dataclass(None, args, False, True, False, hunyuan_image="2.1").to_metadata_dict()

    def update_metadata(self, metadata, args):
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
        # do not support text encoder training for HunyuanImage-2.1
        pass

    def cast_text_encoder(self, args):
        return False  # VLM is bf16, byT5 is fp16, so do not cast to other dtype

    def cast_vae(self, args):
        return False  # VAE is fp16, so do not cast to other dtype

    def cast_unet(self, args):
        return not args.fp8_scaled  # if fp8_scaled is used, do not cast to other dtype

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        # fp8 text encoder for HunyuanImage-2.1 is not supported currently
        pass

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
        model: hunyuan_image_models.HYImageDiffusionTransformer = unet
        model = accelerator.prepare(model, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        return model


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)

    parser.add_argument(
        "--text_encoder",
        type=str,
        help="path to Qwen2.5-VL (*.sft or *.safetensors), should be bfloat16 / Qwen2.5-VLのパス（*.sftまたは*.safetensors）、bfloat16が前提",
    )
    parser.add_argument(
        "--byt5",
        type=str,
        help="path to byt5 (*.sft or *.safetensors), should be float16 / byt5のパス（*.sftまたは*.safetensors）、float16が前提",
    )

    parser.add_argument(
        "--timestep_sampling",
        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift"],
        default="sigma",
        help="Method to sample timesteps: sigma-based, uniform random, sigmoid of random normal, shift of sigmoid and FLUX.1 shifting."
        " / タイムステップをサンプリングする方法：sigma、random uniform、random normalのsigmoid、sigmoidのシフト、FLUX.1のシフト。",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help='Scale factor for sigmoid timestep sampling (only used when timestep-sampling is "sigmoid"). / sigmoidタイムステップサンプリングの倍率（timestep-samplingが"sigmoid"の場合のみ有効）。',
    )
    parser.add_argument(
        "--model_prediction_type",
        choices=["raw", "additive", "sigma_scaled"],
        default="raw",
        help="How to interpret and process the model prediction: "
        "raw (use as is), additive (add to noisy input), sigma_scaled (apply sigma scaling). Default is raw unlike FLUX.1."
        " / モデル予測の解釈と処理方法："
        "raw（そのまま使用）、additive（ノイズ入力に加算）、sigma_scaled（シグマスケーリングを適用）。デフォルトはFLUX.1とは異なりrawです。",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=5.0,
        help="Discrete flow shift for the Euler Discrete Scheduler, default is 5.0. / Euler Discrete Schedulerの離散フローシフト、デフォルトは5.0。",
    )
    parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--fp8_vl", action="store_true", help="Use fp8 for VLM text encoder / VLMテキストエンコーダにfp8を使用する")
    parser.add_argument(
        "--text_encoder_cpu", action="store_true", help="Inference on CPU for Text Encoders / テキストエンコーダをCPUで推論する"
    )
    parser.add_argument(
        "--vae_chunk_size",
        type=int,
        default=None,  # default is None (no chunking)
        help="Chunk size for VAE decoding to reduce memory usage. Default is None (no chunking). 16 is recommended if enabled"
        " / メモリ使用量を減らすためのVAEデコードのチャンクサイズ。デフォルトはNone（チャンクなし）。有効にする場合は16程度を推奨。",
    )

    parser.add_argument(
        "--attn_mode",
        choices=["torch", "xformers", "flash", "sageattn", "sdpa"],  # "sdpa" is for backward compatibility
        default=None,
        help="Attention implementation to use. Default is None (torch). xformers requires --split_attn. sageattn does not support training (inference only). This option overrides --xformers or --sdpa."
        " / 使用するAttentionの実装。デフォルトはNone（torch）です。xformersは--split_attnの指定が必要です。sageattnはトレーニングをサポートしていません（推論のみ）。このオプションは--xformersまたは--sdpaを上書きします。",
    )
    parser.add_argument(
        "--split_attn",
        action="store_true",
        help="split attention computation to reduce memory usage / メモリ使用量を減らすためにattention時にバッチを分割する",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    trainer = HunyuanImageNetworkTrainer()
    trainer.train(args)
