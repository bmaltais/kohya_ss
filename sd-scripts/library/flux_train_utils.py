import argparse
import math
import os
import numpy as np
import toml
import json
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, PartialState
from transformers import CLIPTextModel
from tqdm import tqdm
from PIL import Image
from safetensors.torch import save_file

from library import flux_models, flux_utils, strategy_base, train_util
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from .utils import setup_logging, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)


# region sample images


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    flux,
    ae,
    text_encoders,
    sample_prompts_te_outputs,
    prompt_replacement=None,
    controlnet=None,
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
    flux = accelerator.unwrap_model(flux)
    if text_encoders is not None:
        text_encoders = [(accelerator.unwrap_model(te) if te is not None else None) for te in text_encoders]
    if controlnet is not None:
        controlnet = accelerator.unwrap_model(controlnet)
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
                    flux,
                    text_encoders,
                    ae,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    sample_prompts_te_outputs,
                    prompt_replacement,
                    controlnet,
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
                        flux,
                        text_encoders,
                        ae,
                        save_dir,
                        prompt_dict,
                        epoch,
                        steps,
                        sample_prompts_te_outputs,
                        prompt_replacement,
                        controlnet,
                    )

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    flux: flux_models.Flux,
    text_encoders: Optional[List[CLIPTextModel]],
    ae: flux_models.AutoEncoder,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
    controlnet,
):
    assert isinstance(prompt_dict, dict)
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 20)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    # TODO refactor variable names
    cfg_scale = prompt_dict.get("guidance_scale", 1.0)
    emb_guidance_scale = prompt_dict.get("scale", 3.5)
    seed = prompt_dict.get("seed")
    controlnet_image = prompt_dict.get("controlnet_image")
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
    logger.info(f"embedded guidance scale: {emb_guidance_scale}")
    if cfg_scale != 1.0:
        logger.info(f"CFG scale: {cfg_scale}")
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

    l_pooled, t5_out, txt_ids, t5_attn_mask = encode_prompt(prompt)
    # encode negative prompts
    if cfg_scale != 1.0:
        neg_l_pooled, neg_t5_out, _, neg_t5_attn_mask = encode_prompt(negative_prompt)
        neg_t5_attn_mask = (
            neg_t5_attn_mask.to(accelerator.device) if args.apply_t5_attn_mask and neg_t5_attn_mask is not None else None
        )
        neg_cond = (cfg_scale, neg_l_pooled, neg_t5_out, neg_t5_attn_mask)
    else:
        neg_cond = None

    # sample image
    weight_dtype = ae.dtype  # TOFO give dtype as argument
    packed_latent_height = height // 16
    packed_latent_width = width // 16
    noise = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=accelerator.device,
        dtype=weight_dtype,
        generator=torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None,
    )
    timesteps = get_schedule(sample_steps, noise.shape[1], shift=True)  # FLUX.1 dev -> shift=True
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(accelerator.device, weight_dtype)
    t5_attn_mask = t5_attn_mask.to(accelerator.device) if args.apply_t5_attn_mask else None

    if controlnet_image is not None:
        controlnet_image = Image.open(controlnet_image).convert("RGB")
        controlnet_image = controlnet_image.resize((width, height), Image.LANCZOS)
        controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
        controlnet_image = controlnet_image.permute(2, 0, 1).unsqueeze(0).to(weight_dtype).to(accelerator.device)

    with accelerator.autocast(), torch.no_grad():
        x = denoise(
            flux,
            noise,
            img_ids,
            t5_out,
            txt_ids,
            l_pooled,
            timesteps=timesteps,
            guidance=emb_guidance_scale,
            t5_attn_mask=t5_attn_mask,
            controlnet=controlnet,
            controlnet_img=controlnet_image,
            neg_cond=neg_cond,
        )

    x = flux_utils.unpack_latents(x, packed_latent_height, packed_latent_width)

    # latent to image
    clean_memory_on_device(accelerator.device)
    org_vae_device = ae.device  # will be on cpu
    ae.to(accelerator.device)  # distributed_state.device is same as accelerator.device
    with accelerator.autocast(), torch.no_grad():
        x = ae.decode(x)
    ae.to(org_vae_device)
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


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,  # t5_out
    txt_ids: torch.Tensor,
    vec: torch.Tensor,  # l_pooled
    timesteps: list[float],
    guidance: float = 4.0,
    t5_attn_mask: Optional[torch.Tensor] = None,
    controlnet: Optional[flux_models.ControlNetFlux] = None,
    controlnet_img: Optional[torch.Tensor] = None,
    neg_cond: Optional[Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    do_cfg = neg_cond is not None

    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        model.prepare_block_swap_before_forward()

        if controlnet is not None:
            block_samples, block_single_samples = controlnet(
                img=img,
                img_ids=img_ids,
                controlnet_cond=controlnet_img,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                txt_attention_mask=t5_attn_mask,
            )
        else:
            block_samples = None
            block_single_samples = None

        if not do_cfg:
            pred = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                block_controlnet_hidden_states=block_samples,
                block_controlnet_single_hidden_states=block_single_samples,
                timesteps=t_vec,
                guidance=guidance_vec,
                txt_attention_mask=t5_attn_mask,
            )

            img = img + (t_prev - t_curr) * pred
        else:
            cfg_scale, neg_l_pooled, neg_t5_out, neg_t5_attn_mask = neg_cond
            nc_c_t5_attn_mask = None if t5_attn_mask is None else torch.cat([neg_t5_attn_mask, t5_attn_mask], dim=0)

            # TODO is it ok to use the same block samples for both cond and uncond?
            block_samples = None if block_samples is None else torch.cat([block_samples, block_samples], dim=0)
            block_single_samples = (
                None if block_single_samples is None else torch.cat([block_single_samples, block_single_samples], dim=0)
            )

            nc_c_pred = model(
                img=torch.cat([img, img], dim=0),
                img_ids=torch.cat([img_ids, img_ids], dim=0),
                txt=torch.cat([neg_t5_out, txt], dim=0),
                txt_ids=torch.cat([txt_ids, txt_ids], dim=0),
                y=torch.cat([neg_l_pooled, vec], dim=0),
                block_controlnet_hidden_states=block_samples,
                block_controlnet_single_hidden_states=block_single_samples,
                timesteps=t_vec,
                guidance=guidance_vec,
                txt_attention_mask=nc_c_t5_attn_mask,
            )
            neg_pred, pred = torch.chunk(nc_c_pred, 2, dim=0)
            pred = neg_pred + (pred - neg_pred) * cfg_scale

            img = img + (t_prev - t_curr) * pred

    model.prepare_block_swap_before_forward()
    return img


# endregion


# region train
def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
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


def get_noisy_model_input_and_timesteps(
    args, noise_scheduler, latents: torch.Tensor, noise: torch.Tensor, device, dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, _, h, w = latents.shape
    assert bsz > 0, "Batch size not large enough"
    num_timesteps = noise_scheduler.config.num_train_timesteps
    if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
        # Simple random sigma-based noise sampling
        if args.timestep_sampling == "sigmoid":
            # https://github.com/XLabs-AI/x-flux/tree/main
            sigmas = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=device))
        else:
            sigmas = torch.rand((bsz,), device=device)

        timesteps = sigmas * num_timesteps
    elif args.timestep_sampling == "shift":
        shift = args.discrete_flow_shift
        sigmas = torch.randn(bsz, device=device)
        sigmas = sigmas * args.sigmoid_scale  # larger scale for more uniform sampling
        sigmas = sigmas.sigmoid()
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_timesteps
    elif args.timestep_sampling == "flux_shift":
        sigmas = torch.randn(bsz, device=device)
        sigmas = sigmas * args.sigmoid_scale  # larger scale for more uniform sampling
        sigmas = sigmas.sigmoid()
        mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))  # we are pre-packed so must adjust for packed size
        sigmas = time_shift(mu, 1.0, sigmas)
        timesteps = sigmas * num_timesteps
    else:
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * num_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)

    # Broadcast sigmas to latent shape
    sigmas = sigmas.view(-1, 1, 1, 1)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    if args.ip_noise_gamma:
        xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
        if args.ip_noise_gamma_random_strength:
            ip_noise_gamma = torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma
        else:
            ip_noise_gamma = args.ip_noise_gamma
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * (noise + ip_noise_gamma * xi)
    else:
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

    return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas


def apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas):
    weighting = None
    if args.model_prediction_type == "raw":
        pass
    elif args.model_prediction_type == "additive":
        # add the model_pred to the noisy_model_input
        model_pred = model_pred + noisy_model_input
    elif args.model_prediction_type == "sigma_scaled":
        # apply sigma scaling
        model_pred = model_pred * (-sigmas) + noisy_model_input

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

    return model_pred, weighting


def save_models(
    ckpt_path: str,
    flux: flux_models.Flux,
    sai_metadata: Optional[dict],
    save_dtype: Optional[torch.dtype] = None,
    use_mem_eff_save: bool = False,
):
    state_dict = {}

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            if save_dtype is not None and v.dtype != save_dtype:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    update_sd("", flux.state_dict())

    if not use_mem_eff_save:
        save_file(state_dict, ckpt_path, metadata=sai_metadata)
    else:
        mem_eff_save_file(state_dict, ckpt_path, metadata=sai_metadata)


def save_flux_model_on_train_end(
    args: argparse.Namespace, save_dtype: torch.dtype, epoch: int, global_step: int, flux: flux_models.Flux
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(None, args, False, False, False, is_stable_diffusion_ckpt=True, flux="dev")
        save_models(ckpt_file, flux, sai_metadata, save_dtype, args.mem_eff_save)

    train_util.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_flux_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    flux: flux_models.Flux,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(None, args, False, False, False, is_stable_diffusion_ckpt=True, flux="dev")
        save_models(ckpt_file, flux, sai_metadata, save_dtype, args.mem_eff_save)

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


# endregion


def add_flux_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--clip_l",
        type=str,
        help="path to clip_l (*.sft or *.safetensors), should be float16 / clip_lのパス（*.sftまたは*.safetensors）、float16が前提",
    )
    parser.add_argument(
        "--t5xxl",
        type=str,
        help="path to t5xxl (*.sft or *.safetensors), should be float16 / t5xxlのパス（*.sftまたは*.safetensors）、float16が前提",
    )
    parser.add_argument("--ae", type=str, help="path to ae (*.sft or *.safetensors) / aeのパス（*.sftまたは*.safetensors）")
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="path to controlnet (*.sft or *.safetensors) / controlnetのパス（*.sftまたは*.safetensors）",
    )
    parser.add_argument(
        "--t5xxl_max_token_length",
        type=int,
        default=None,
        help="maximum token length for T5-XXL. if omitted, 256 for schnell and 512 for dev"
        " / T5-XXLの最大トークン長。省略された場合、schnellの場合は256、devの場合は512",
    )
    parser.add_argument(
        "--apply_t5_attn_mask",
        action="store_true",
        help="apply attention mask to T5-XXL encode and FLUX double blocks / T5-XXLエンコードとFLUXダブルブロックにアテンションマスクを適用する",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
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
        default="sigma_scaled",
        help="How to interpret and process the model prediction: "
        "raw (use as is), additive (add to noisy input), sigma_scaled (apply sigma scaling)."
        " / モデル予測の解釈と処理方法："
        "raw（そのまま使用）、additive（ノイズ入力に加算）、sigma_scaled（シグマスケーリングを適用）。",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=3.0,
        help="Discrete flow shift for the Euler Discrete Scheduler, default is 3.0. / Euler Discrete Schedulerの離散フローシフト、デフォルトは3.0。",
    )
