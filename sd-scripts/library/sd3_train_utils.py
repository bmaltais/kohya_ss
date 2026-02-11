import argparse
import math
import os
import toml
import json
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from safetensors.torch import save_file
from accelerate import Accelerator, PartialState
from tqdm import tqdm
from PIL import Image
from transformers import CLIPTextModelWithProjection, T5EncoderModel

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

# from transformers import CLIPTokenizer
# from library import model_util
# , sdxl_model_util, train_util, sdxl_original_unet
# from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import sd3_models, sd3_utils, strategy_base, train_util


def save_models(
    ckpt_path: str,
    mmdit: Optional[sd3_models.MMDiT],
    vae: Optional[sd3_models.SDVAE],
    clip_l: Optional[CLIPTextModelWithProjection],
    clip_g: Optional[CLIPTextModelWithProjection],
    t5xxl: Optional[T5EncoderModel],
    sai_metadata: Optional[dict],
    save_dtype: Optional[torch.dtype] = None,
):
    r"""
    Save models to checkpoint file. Only supports unified checkpoint format.
    """

    state_dict = {}

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            if save_dtype is not None:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    update_sd("model.diffusion_model.", mmdit.state_dict())
    update_sd("first_stage_model.", vae.state_dict())

    # do not support unified checkpoint format for now
    # if clip_l is not None:
    #     update_sd("text_encoders.clip_l.", clip_l.state_dict())
    # if clip_g is not None:
    #     update_sd("text_encoders.clip_g.", clip_g.state_dict())
    # if t5xxl is not None:
    #     update_sd("text_encoders.t5xxl.", t5xxl.state_dict())

    save_file(state_dict, ckpt_path, metadata=sai_metadata)

    if clip_l is not None:
        clip_l_path = ckpt_path.replace(".safetensors", "_clip_l.safetensors")
        save_file(clip_l.state_dict(), clip_l_path)
    if clip_g is not None:
        clip_g_path = ckpt_path.replace(".safetensors", "_clip_g.safetensors")
        save_file(clip_g.state_dict(), clip_g_path)
    if t5xxl is not None:
        t5xxl_path = ckpt_path.replace(".safetensors", "_t5xxl.safetensors")
        t5xxl_state_dict = t5xxl.state_dict()

        # replace "shared.weight" with copy of it to avoid annoying shared tensor error on safetensors.save_file
        shared_weight = t5xxl_state_dict["shared.weight"]
        shared_weight_copy = shared_weight.detach().clone()
        t5xxl_state_dict["shared.weight"] = shared_weight_copy

        save_file(t5xxl_state_dict, t5xxl_path)


def save_sd3_model_on_train_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    clip_l: Optional[CLIPTextModelWithProjection],
    clip_g: Optional[CLIPTextModelWithProjection],
    t5xxl: Optional[T5EncoderModel],
    mmdit: sd3_models.MMDiT,
    vae: sd3_models.SDVAE,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, sd3=mmdit.model_type
        )
        save_models(ckpt_file, mmdit, vae, clip_l, clip_g, t5xxl, sai_metadata, save_dtype)

    train_util.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd3_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    clip_l: Optional[CLIPTextModelWithProjection],
    clip_g: Optional[CLIPTextModelWithProjection],
    t5xxl: Optional[T5EncoderModel],
    mmdit: sd3_models.MMDiT,
    vae: sd3_models.SDVAE,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, sd3=mmdit.model_type
        )
        save_models(ckpt_file, mmdit, vae, clip_l, clip_g, t5xxl, sai_metadata, save_dtype)

    train_util.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        True,
        True,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        None,
    )


def add_sd3_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--clip_l",
        type=str,
        required=False,
        help="CLIP-L model path. if not specified, use ckpt's state_dict / CLIP-Lモデルのパス。指定しない場合はckptのstate_dictを使用",
    )
    parser.add_argument(
        "--clip_g",
        type=str,
        required=False,
        help="CLIP-G model path. if not specified, use ckpt's state_dict / CLIP-Gモデルのパス。指定しない場合はckptのstate_dictを使用",
    )
    parser.add_argument(
        "--t5xxl",
        type=str,
        required=False,
        help="T5-XXL model path. if not specified, use ckpt's state_dict / T5-XXLモデルのパス。指定しない場合はckptのstate_dictを使用",
    )
    parser.add_argument(
        "--save_clip",
        action="store_true",
        help="[DOES NOT WORK] unified checkpoint is not supported / 統合チェックポイントはまだサポートされていません",
    )
    parser.add_argument(
        "--save_t5xxl",
        action="store_true",
        help="[DOES NOT WORK] unified checkpoint is not supported / 統合チェックポイントはまだサポートされていません",
    )

    parser.add_argument(
        "--t5xxl_device",
        type=str,
        default=None,
        help="[DOES NOT WORK] not supported yet. T5-XXL device. if not specified, use accelerator's device / T5-XXLデバイス。指定しない場合はacceleratorのデバイスを使用",
    )
    parser.add_argument(
        "--t5xxl_dtype",
        type=str,
        default=None,
        help="[DOES NOT WORK] not supported yet. T5-XXL dtype. if not specified, use default dtype (from mixed precision) / T5-XXL dtype。指定しない場合はデフォルトのdtype（mixed precisionから）を使用",
    )

    parser.add_argument(
        "--t5xxl_max_token_length",
        type=int,
        default=256,
        help="maximum token length for T5-XXL. 256 is the default value / T5-XXLの最大トークン長。デフォルトは256",
    )
    parser.add_argument(
        "--apply_lg_attn_mask",
        action="store_true",
        help="apply attention mask (zero embs) to CLIP-L and G / CLIP-LとGにアテンションマスク（ゼロ埋め）を適用する",
    )
    parser.add_argument(
        "--apply_t5_attn_mask",
        action="store_true",
        help="apply attention mask (zero embs) to T5-XXL / T5-XXLにアテンションマスク（ゼロ埋め）を適用する",
    )
    parser.add_argument(
        "--clip_l_dropout_rate",
        type=float,
        default=0.0,
        help="Dropout rate for CLIP-L encoder, default is 0.0 / CLIP-Lエンコーダのドロップアウト率、デフォルトは0.0",
    )
    parser.add_argument(
        "--clip_g_dropout_rate",
        type=float,
        default=0.0,
        help="Dropout rate for CLIP-G encoder, default is 0.0 / CLIP-Gエンコーダのドロップアウト率、デフォルトは0.0",
    )
    parser.add_argument(
        "--t5_dropout_rate",
        type=float,
        default=0.0,
        help="Dropout rate for T5 encoder, default is 0.0 / T5エンコーダのドロップアウト率、デフォルトは0.0",
    )
    parser.add_argument(
        "--pos_emb_random_crop_rate",
        type=float,
        default=0.0,
        help="Random crop rate for positional embeddings, default is 0.0. Only for SD3.5M"
        " / 位置埋め込みのランダムクロップ率、デフォルトは0.0。SD3.5M以外では予期しない動作になります",
    )
    parser.add_argument(
        "--enable_scaled_pos_embed",
        action="store_true",
        help="Scale position embeddings for each resolution during multi-resolution training. Only for SD3.5M"
        " / 複数解像度学習時に解像度ごとに位置埋め込みをスケーリングする。SD3.5M以外では予期しない動作になります",
    )

    # Dependencies of Diffusers noise sampler has been removed for clarity in training

    parser.add_argument(
        "--training_shift",
        type=float,
        default=1.0,
        help="Discrete flow shift for training timestep distribution adjustment, applied in addition to the weighting scheme, default is 1.0. /タイムステップ分布のための離散フローシフト、重み付けスキームの上に適用される、デフォルトは1.0。",
    )


def verify_sdxl_training_args(args: argparse.Namespace, supportTextEncoderCaching: bool = True):
    assert not args.v2, "v2 cannot be enabled in SDXL training / SDXL学習ではv2を有効にすることはできません"
    if args.v_parameterization:
        logger.warning("v_parameterization will be unexpected / SDXL学習ではv_parameterizationは想定外の動作になります")

    if args.clip_skip is not None:
        logger.warning("clip_skip will be unexpected / SDXL学習ではclip_skipは動作しません")

    # if args.multires_noise_iterations:
    #     logger.info(
    #         f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET}, but noise_offset is disabled due to multires_noise_iterations / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されていますが、multires_noise_iterationsが有効になっているためnoise_offsetは無効になります"
    #     )
    # else:
    #     if args.noise_offset is None:
    #         args.noise_offset = DEFAULT_NOISE_OFFSET
    #     elif args.noise_offset != DEFAULT_NOISE_OFFSET:
    #         logger.info(
    #             f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET} / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されています"
    #         )
    #     logger.info(f"noise_offset is set to {args.noise_offset} / noise_offsetが{args.noise_offset}に設定されました")

    assert (
        not hasattr(args, "weighted_captions") or not args.weighted_captions
    ), "weighted_captions cannot be enabled in SDXL training currently / SDXL学習では今のところweighted_captionsを有効にすることはできません"

    if supportTextEncoderCaching:
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True
            logger.warning(
                "cache_text_encoder_outputs is enabled because cache_text_encoder_outputs_to_disk is enabled / "
                + "cache_text_encoder_outputs_to_diskが有効になっているためcache_text_encoder_outputsが有効になりました"
            )


# temporary copied from sd3_minimal_inferece.py


def get_all_sigmas(sampling: sd3_utils.ModelSamplingDiscreteFlow, steps):
    start = sampling.timestep(sampling.sigma_max)
    end = sampling.timestep(sampling.sigma_min)
    timesteps = torch.linspace(start, end, steps)
    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(sampling.sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def max_denoise(model_sampling, sigmas):
    max_sigma = float(model_sampling.sigma_max)
    sigma = float(sigmas[0])
    return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma


def do_sample(
    height: int,
    width: int,
    seed: int,
    cond: Tuple[torch.Tensor, torch.Tensor],
    neg_cond: Tuple[torch.Tensor, torch.Tensor],
    mmdit: sd3_models.MMDiT,
    steps: int,
    guidance_scale: float,
    dtype: torch.dtype,
    device: str,
):
    latent = torch.zeros(1, 16, height // 8, width // 8, device=device)
    latent = latent.to(dtype).to(device)

    # noise = get_noise(seed, latent).to(device)
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    noise = (
        torch.randn(latent.size(), dtype=torch.float32, layout=latent.layout, generator=generator, device="cpu")
        .to(latent.dtype)
        .to(device)
    )

    model_sampling = sd3_utils.ModelSamplingDiscreteFlow(shift=3.0)  # 3.0 is for SD3

    sigmas = get_all_sigmas(model_sampling, steps).to(device)

    noise_scaled = model_sampling.noise_scaling(sigmas[0], noise, latent, max_denoise(model_sampling, sigmas))

    c_crossattn = torch.cat([cond[0], neg_cond[0]]).to(device).to(dtype)
    y = torch.cat([cond[1], neg_cond[1]]).to(device).to(dtype)

    x = noise_scaled.to(device).to(dtype)
    # print(x.shape)

    # with torch.no_grad():
    for i in tqdm(range(len(sigmas) - 1)):
        sigma_hat = sigmas[i]

        timestep = model_sampling.timestep(sigma_hat).float()
        timestep = torch.FloatTensor([timestep, timestep]).to(device)

        x_c_nc = torch.cat([x, x], dim=0)
        # print(x_c_nc.shape, timestep.shape, c_crossattn.shape, y.shape)

        mmdit.prepare_block_swap_before_forward()
        model_output = mmdit(x_c_nc, timestep, context=c_crossattn, y=y)
        model_output = model_output.float()
        batched = model_sampling.calculate_denoised(sigma_hat, model_output, x)

        pos_out, neg_out = batched.chunk(2)
        denoised = neg_out + (pos_out - neg_out) * guidance_scale
        # print(denoised.shape)

        # d = to_d(x, sigma_hat, denoised)
        dims_to_append = x.ndim - sigma_hat.ndim
        sigma_hat_dims = sigma_hat[(...,) + (None,) * dims_to_append]
        # print(dims_to_append, x.shape, sigma_hat.shape, denoised.shape, sigma_hat_dims.shape)
        """Converts a denoiser output to a Karras ODE derivative."""
        d = (x - denoised) / sigma_hat_dims

        dt = sigmas[i + 1] - sigma_hat

        # Euler method
        x = x + d * dt
        x = x.to(dtype)

    mmdit.prepare_block_swap_before_forward()
    return x


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    mmdit,
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
    mmdit = accelerator.unwrap_model(mmdit)
    text_encoders = None if text_encoders is None else [accelerator.unwrap_model(te) for te in text_encoders]
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
                    mmdit,
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
                        mmdit,
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

    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    mmdit: sd3_models.MMDiT,
    text_encoders: List[Union[CLIPTextModelWithProjection, T5EncoderModel]],
    vae: sd3_models.SDVAE,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    assert isinstance(prompt_dict, dict)
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    scale = prompt_dict.get("scale", 7.5)
    seed = prompt_dict.get("seed")
    # controlnet_image = prompt_dict.get("controlnet_image")
    prompt: str = prompt_dict.get("prompt", "")
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

    height = max(64, height - height % 8)  # round to divisible by 8
    width = max(64, width - width % 8)  # round to divisible by 8
    logger.info(f"prompt: {prompt}")
    logger.info(f"negative_prompt: {negative_prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"scale: {scale}")
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
            print(f"Using cached text encoder outputs for prompt: {prpt}")
        if text_encoders is not None:
            print(f"Encoding prompt: {prpt}")
            tokens_and_masks = tokenize_strategy.tokenize(prpt)
            # strategy has apply_t5_attn_mask option
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

    lg_out, t5_out, pooled, l_attn_mask, g_attn_mask, t5_attn_mask = encode_prompt(prompt)
    cond = encoding_strategy.concat_encodings(lg_out, t5_out, pooled)

    # encode negative prompts
    lg_out, t5_out, pooled, l_attn_mask, g_attn_mask, t5_attn_mask = encode_prompt(negative_prompt)
    neg_cond = encoding_strategy.concat_encodings(lg_out, t5_out, pooled)

    # sample image
    clean_memory_on_device(accelerator.device)
    with accelerator.autocast(), torch.no_grad():
        # mmdit may be fp8, so we need weight_dtype here. vae is always in that dtype.
        latents = do_sample(height, width, seed, cond, neg_cond, mmdit, sample_steps, scale, vae.dtype, accelerator.device)

    # latent to image
    clean_memory_on_device(accelerator.device)
    org_vae_device = vae.device  # will be on cpu
    vae.to(accelerator.device)
    latents = vae.process_out(latents.to(vae.device, dtype=vae.dtype))
    image = vae.decode(latents)
    vae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)

    image = image.float()
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
    decoded_np = decoded_np.astype(np.uint8)

    image = Image.fromarray(decoded_np)
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


# region Diffusers


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps)

        sigmas = timesteps / self.config.num_train_timesteps
        sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        timesteps = sigmas * self.config.num_train_timesteps
        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator)

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility

        # if self.config.prediction_type == "vector_field":

        denoised = sample - model_output * sigma
        # 2. Convert to an ODE derivative
        derivative = (sample - denoised) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


# endregion


def get_noisy_model_input_and_timesteps(args, latents, noise, device, dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = latents.shape[0]

    # Sample a random timestep for each image
    # for weighting schemes where we sample timesteps non-uniformly
    u = compute_density_for_timestep_sampling(
        weighting_scheme=args.weighting_scheme,
        batch_size=bsz,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
    )
    t_min = args.min_timestep if args.min_timestep is not None else 0
    t_max = args.max_timestep if args.max_timestep is not None else 1000
    shift = args.training_shift

    # weighting shift, value >1 will shift distribution to noisy side (focus more on overall structure), value <1 will shift towards less-noisy side (focus more on details)
    u = (u * shift) / (1 + (shift - 1) * u)

    indices = (u * (t_max - t_min) + t_min).long()
    timesteps = indices.to(device=device, dtype=dtype)

    # sigmas according to flowmatching
    sigmas = timesteps / 1000
    sigmas = sigmas.view(-1, 1, 1, 1)
    noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

    return noisy_model_input, timesteps, sigmas
