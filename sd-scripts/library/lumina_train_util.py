import inspect
import argparse
import math
import os
import numpy as np
import time
from typing import Callable, Dict, List, Optional, Tuple, Any, Union, Generator

import torch
from torch import Tensor
from accelerate import Accelerator, PartialState
from transformers import Gemma2Model
from tqdm import tqdm
from PIL import Image
from safetensors.torch import save_file

from library import lumina_models, strategy_base, strategy_lumina, train_util
from library.flux_models import AutoEncoder
from library.device_utils import init_ipex, clean_memory_on_device
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler
from library.safetensors_utils import mem_eff_save_file

init_ipex()

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# region sample images


def batchify(
    prompt_dicts, batch_size=None
) -> Generator[list[dict[str, str]], None, None]:
    """
    Group prompt dictionaries into batches with configurable batch size.

    Args:
        prompt_dicts (list): List of dictionaries containing prompt parameters.
        batch_size (int, optional): Number of prompts per batch. Defaults to None.

    Yields:
        list[dict[str, str]]: Batch of prompts.
    """
    # Validate batch_size
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None")

    # Group prompts by their parameters
    batches = {}
    for prompt_dict in prompt_dicts:
        # Extract parameters
        width = int(prompt_dict.get("width", 1024))
        height = int(prompt_dict.get("height", 1024))
        height = max(64, height - height % 8)  # round to divisible by 8
        width = max(64, width - width % 8)  # round to divisible by 8
        guidance_scale = float(prompt_dict.get("scale", 3.5))
        sample_steps = int(prompt_dict.get("sample_steps", 38))
        cfg_trunc_ratio = float(prompt_dict.get("cfg_trunc_ratio", 0.25))
        renorm_cfg = float(prompt_dict.get("renorm_cfg", 1.0))
        seed = prompt_dict.get("seed", None)
        seed = int(seed) if seed is not None else None

        # Create a key based on the parameters
        key = (
            width,
            height,
            guidance_scale,
            seed,
            sample_steps,
            cfg_trunc_ratio,
            renorm_cfg,
        )

        # Add the prompt_dict to the corresponding batch
        if key not in batches:
            batches[key] = []
        batches[key].append(prompt_dict)

    # Yield each batch with its parameters
    for key in batches:
        prompts = batches[key]
        if batch_size is None:
            # Yield the entire group as a single batch
            yield prompts
        else:
            # Split the group into batches of size `batch_size`
            start = 0
            while start < len(prompts):
                end = start + batch_size
                batch = prompts[start:end]
                yield batch
                start = end


@torch.no_grad()
def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    nextdit: lumina_models.NextDiT,
    vae: AutoEncoder,
    gemma2_model: Gemma2Model,
    sample_prompts_gemma2_outputs: dict[str, Tuple[Tensor, Tensor, Tensor]],
    prompt_replacement: Optional[Tuple[str, str]] = None,
    controlnet=None,
):
    """
    Generate sample images using the NextDiT model.

    Args:
        accelerator (Accelerator): Accelerator instance.
        args (argparse.Namespace): Command-line arguments.
        epoch (int): Current epoch number.
        global_step (int): Current global step number.
        nextdit (lumina_models.NextDiT): The NextDiT model instance.
        vae (AutoEncoder): The VAE module.
        gemma2_model (Gemma2Model): The Gemma2 model instance.
        sample_prompts_gemma2_outputs (dict[str, Tuple[Tensor, Tensor, Tensor]]):
            Dictionary of tuples containing the encoded prompts, text masks, and timestep for each sample.
        prompt_replacement (Optional[Tuple[str, str]], optional):
            Tuple containing the prompt and negative prompt replacements. Defaults to None.
        controlnet (): ControlNet model, not yet supported

    Returns:
        None
    """
    if global_step == 0:
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
            if (
                global_step % args.sample_every_n_steps != 0 or epoch is not None
            ):  # steps is not divisible or end of epoch
                return

    assert (
        args.sample_prompts is not None
    ), "No sample prompts found. Provide `--sample_prompts` / サンプルプロンプトが見つかりません。`--sample_prompts` を指定してください"

    logger.info("")
    logger.info(
        f"generating sample images at step / サンプル画像生成 ステップ: {global_step}"
    )
    if (
        not os.path.isfile(args.sample_prompts)
        and sample_prompts_gemma2_outputs is None
    ):
        logger.error(
            f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}"
        )
        return

    distributed_state = (
        PartialState()
    )  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    # unwrap nextdit and gemma2_model
    nextdit = accelerator.unwrap_model(nextdit)
    if gemma2_model is not None:
        gemma2_model = accelerator.unwrap_model(gemma2_model)
    # if controlnet is not None:
    #     controlnet = accelerator.unwrap_model(controlnet)
    # print([(te.parameters().__next__().device if te is not None else None) for te in text_encoders])

    prompts = train_util.load_prompts(args.sample_prompts)

    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    # save random state to restore later
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        )
    except Exception:
        pass

    batch_size = args.sample_batch_size or args.train_batch_size or 1

    if distributed_state.num_processes <= 1:
        # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
        # TODO: batch prompts together with buckets of image sizes
        for prompt_dicts in batchify(prompts, batch_size):
            sample_image_inference(
                accelerator,
                args,
                nextdit,
                gemma2_model,
                vae,
                save_dir,
                prompt_dicts,
                epoch,
                global_step,
                sample_prompts_gemma2_outputs,
                prompt_replacement,
                controlnet,
            )
    else:
        # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
        # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
        per_process_prompts = []  # list of lists
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i :: distributed_state.num_processes])

        with distributed_state.split_between_processes(
            per_process_prompts
        ) as prompt_dict_lists:
            # TODO: batch prompts together with buckets of image sizes
            for prompt_dicts in batchify(prompt_dict_lists[0], batch_size):
                sample_image_inference(
                    accelerator,
                    args,
                    nextdit,
                    gemma2_model,
                    vae,
                    save_dir,
                    prompt_dicts,
                    epoch,
                    global_step,
                    sample_prompts_gemma2_outputs,
                    prompt_replacement,
                    controlnet,
                )

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    clean_memory_on_device(accelerator.device)


@torch.no_grad()
def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    nextdit: lumina_models.NextDiT,
    gemma2_model: list[Gemma2Model],
    vae: AutoEncoder,
    save_dir: str,
    prompt_dicts: list[Dict[str, str]],
    epoch: int,
    global_step: int,
    sample_prompts_gemma2_outputs: dict[str, Tuple[Tensor, Tensor, Tensor]],
    prompt_replacement: Optional[Tuple[str, str]] = None,
    controlnet=None,
):
    """
    Generates sample images

    Args:
        accelerator (Accelerator): Accelerator object
        args (argparse.Namespace): Arguments object
        nextdit (lumina_models.NextDiT): NextDiT model
        gemma2_model (list[Gemma2Model]): Gemma2 model
        vae (AutoEncoder): VAE model
        save_dir (str): Directory to save images
        prompt_dict (Dict[str, str]): Prompt dictionary
        epoch (int): Epoch number
        steps (int): Number of steps to run
        sample_prompts_gemma2_outputs (List[Tuple[Tensor, Tensor, Tensor]]): List of tuples containing Gemma 2 outputs
        prompt_replacement (Optional[Tuple[str, str]], optional): Replacement for positive and negative prompt. Defaults to None.

    Returns:
        None
    """

    # encode prompts
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

    assert isinstance(tokenize_strategy, strategy_lumina.LuminaTokenizeStrategy)
    assert isinstance(encoding_strategy, strategy_lumina.LuminaTextEncodingStrategy)

    text_conds = []

    # assuming seed, width, height, sample steps, guidance are the same
    width = int(prompt_dicts[0].get("width", 1024))
    height = int(prompt_dicts[0].get("height", 1024))
    height = max(64, height - height % 8)  # round to divisible by 8
    width = max(64, width - width % 8)  # round to divisible by 8

    guidance_scale = float(prompt_dicts[0].get("scale", 3.5))
    cfg_trunc_ratio = float(prompt_dicts[0].get("cfg_trunc_ratio", 0.25))
    renorm_cfg = float(prompt_dicts[0].get("renorm_cfg", 1.0))
    sample_steps = int(prompt_dicts[0].get("sample_steps", 36))
    seed = prompt_dicts[0].get("seed", None)
    seed = int(seed) if seed is not None else None
    assert seed is None or seed > 0, f"Invalid seed {seed}"
    generator = torch.Generator(device=accelerator.device)
    if seed is not None:
        generator.manual_seed(seed)

    for prompt_dict in prompt_dicts:
        controlnet_image = prompt_dict.get("controlnet_image")
        prompt: str = prompt_dict.get("prompt", "")
        negative_prompt = prompt_dict.get("negative_prompt", "")
        # sampler_name: str = prompt_dict.get("sample_sampler", args.sample_sampler)

        if prompt_replacement is not None:
            prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
            if negative_prompt is not None:
                negative_prompt = negative_prompt.replace(
                    prompt_replacement[0], prompt_replacement[1]
                )

        if negative_prompt is None:
            negative_prompt = ""
        logger.info(f"prompt: {prompt}")
        logger.info(f"negative_prompt: {negative_prompt}")
        logger.info(f"height: {height}")
        logger.info(f"width: {width}")
        logger.info(f"sample_steps: {sample_steps}")
        logger.info(f"scale: {guidance_scale}")
        logger.info(f"trunc: {cfg_trunc_ratio}")
        logger.info(f"renorm: {renorm_cfg}")
        # logger.info(f"sample_sampler: {sampler_name}")


        # No need to add system prompt here, as it has been handled in the tokenize_strategy

        # Get sample prompts from cache
        if sample_prompts_gemma2_outputs and prompt in sample_prompts_gemma2_outputs:
            gemma2_conds = sample_prompts_gemma2_outputs[prompt]
            logger.info(f"Using cached Gemma2 outputs for prompt: {prompt}")

        if (
            sample_prompts_gemma2_outputs
            and negative_prompt in sample_prompts_gemma2_outputs
        ):
            neg_gemma2_conds = sample_prompts_gemma2_outputs[negative_prompt]
            logger.info(
                f"Using cached Gemma2 outputs for negative prompt: {negative_prompt}"
            )

        # Load sample prompts from Gemma 2
        if gemma2_model is not None:
            tokens_and_masks = tokenize_strategy.tokenize(prompt)
            gemma2_conds = encoding_strategy.encode_tokens(
                tokenize_strategy, gemma2_model, tokens_and_masks
            )

            tokens_and_masks = tokenize_strategy.tokenize(negative_prompt, is_negative=True)
            neg_gemma2_conds = encoding_strategy.encode_tokens(
                tokenize_strategy, gemma2_model, tokens_and_masks
            )

        # Unpack Gemma2 outputs
        gemma2_hidden_states, _, gemma2_attn_mask = gemma2_conds
        neg_gemma2_hidden_states, _, neg_gemma2_attn_mask = neg_gemma2_conds

        text_conds.append(
            (
                gemma2_hidden_states.squeeze(0),
                gemma2_attn_mask.squeeze(0),
                neg_gemma2_hidden_states.squeeze(0),
                neg_gemma2_attn_mask.squeeze(0),
            )
        )

    # Stack conditioning
    cond_hidden_states = torch.stack([text_cond[0] for text_cond in text_conds]).to(
        accelerator.device
    )
    cond_attn_masks = torch.stack([text_cond[1] for text_cond in text_conds]).to(
        accelerator.device
    )
    uncond_hidden_states = torch.stack([text_cond[2] for text_cond in text_conds]).to(
        accelerator.device
    )
    uncond_attn_masks = torch.stack([text_cond[3] for text_cond in text_conds]).to(
        accelerator.device
    )

    # sample image
    weight_dtype = vae.dtype  # TOFO give dtype as argument
    latent_height = height // 8
    latent_width = width // 8
    latent_channels = 16
    noise = torch.randn(
        1,
        latent_channels,
        latent_height,
        latent_width,
        device=accelerator.device,
        dtype=weight_dtype,
        generator=generator,
    )
    noise = noise.repeat(cond_hidden_states.shape[0], 1, 1, 1)

    scheduler = FlowMatchEulerDiscreteScheduler(shift=6.0)
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps=sample_steps
    )

    # if controlnet_image is not None:
    #     controlnet_image = Image.open(controlnet_image).convert("RGB")
    #     controlnet_image = controlnet_image.resize((width, height), Image.LANCZOS)
    #     controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
    #     controlnet_image = controlnet_image.permute(2, 0, 1).unsqueeze(0).to(weight_dtype).to(accelerator.device)

    with accelerator.autocast():
        x = denoise(
            scheduler,
            nextdit,
            noise,
            cond_hidden_states,
            cond_attn_masks,
            uncond_hidden_states,
            uncond_attn_masks,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            cfg_trunc_ratio=cfg_trunc_ratio,
            renorm_cfg=renorm_cfg,
        )

    # Latent to image
    clean_memory_on_device(accelerator.device)
    org_vae_device = vae.device  # will be on cpu
    vae.to(accelerator.device)  # distributed_state.device is same as accelerator.device
    for img, prompt_dict in zip(x, prompt_dicts):

        img = (img / vae.scale_factor) + vae.shift_factor

        with accelerator.autocast():
            # Add a single batch image for the VAE to decode
            img = vae.decode(img.unsqueeze(0))

        img = img.clamp(-1, 1)
        img = img.permute(0, 2, 3, 1)  # B, H, W, C
        # Scale images back to 0 to 255
        img = (127.5 * (img + 1.0)).float().cpu().numpy().astype(np.uint8)

        # Get single image
        image = Image.fromarray(img[0])

        # adding accelerator.wait_for_everyone() here should sync up and ensure that sample images are saved in the same order as the original prompt list
        # but adding 'enum' to the filename should be enough

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{global_step:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        i: int = int(prompt_dict.get("enum", 0))
        img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
        image.save(os.path.join(save_dir, img_filename))

        # send images to wandb if enabled
        if "wandb" in [tracker.name for tracker in accelerator.trackers]:
            wandb_tracker = accelerator.get_tracker("wandb")

            import wandb

            # not to commit images to avoid inconsistency between training and logging steps
            wandb_tracker.log(
                {f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False
            )  # positive prompt as a caption

    vae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    return t


def get_lin_function(
    x1: float = 256, x2: float = 4096, y1: float = 0.5, y2: float = 1.15
) -> Callable[[float], float]:
    """
    Get linear function

    Args:
        image_seq_len,
        x1 base_seq_len: int = 256,
        y2 max_seq_len: int = 4096,
        y1 base_shift: float = 0.5,
        y2 max_shift: float = 1.15,

    Return:
        Callable[[float], float]: linear function
    """
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
    """
    Get timesteps schedule

    Args:
        num_steps (int): Number of steps in the schedule.
        image_seq_len (int): Sequence length of the image.
        base_shift (float, optional): Base shift value. Defaults to 0.5.
        max_shift (float, optional): Maximum shift value. Defaults to 1.15.
        shift (bool, optional): Whether to shift the schedule. Defaults to True.

    Return:
        List[float]: timesteps schedule
    """
    timesteps = torch.linspace(1, 1 / num_steps, num_steps)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift, x1=256, x2=4096)(
            image_seq_len
        )
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def denoise(
    scheduler,
    model: lumina_models.NextDiT,
    img: Tensor,
    txt: Tensor,
    txt_mask: Tensor,
    neg_txt: Tensor,
    neg_txt_mask: Tensor,
    timesteps: Union[List[float], torch.Tensor],
    guidance_scale: float = 4.0,
    cfg_trunc_ratio: float = 0.25,
    renorm_cfg: float = 1.0,
):
    """
    Denoise an image using the NextDiT model.

    Args:
        scheduler ():
            Noise scheduler
        model (lumina_models.NextDiT): The NextDiT model instance.
        img (Tensor):
            The input image latent tensor.
        txt (Tensor):
            The input text tensor.
        txt_mask (Tensor):
            The input text mask tensor.
        neg_txt (Tensor):
            The negative input txt tensor
        neg_txt_mask (Tensor):
            The negative input text mask tensor.
        timesteps (List[Union[float, torch.FloatTensor]]):
            A list of timesteps for the denoising process.
        guidance_scale (float, optional):
            The guidance scale for the denoising process. Defaults to 4.0.
        cfg_trunc_ratio (float, optional):
            The ratio of the timestep interval to apply normalization-based guidance scale.
        renorm_cfg (float, optional):
            The factor to limit the maximum norm after guidance. Default: 1.0
    Returns:
        img (Tensor): Denoised latent tensor
    """

    for i, t in enumerate(tqdm(timesteps)):
        model.prepare_block_swap_before_forward()

        # reverse the timestep since Lumina uses t=0 as the noise and t=1 as the image
        current_timestep = 1 - t / scheduler.config.num_train_timesteps
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        current_timestep = current_timestep * torch.ones(
            img.shape[0], device=img.device
        )

        noise_pred_cond = model(
            img,
            current_timestep,
            cap_feats=txt,  # Gemma2的hidden states作为caption features
            cap_mask=txt_mask.to(dtype=torch.int32),  # Gemma2的attention mask
        )

        # compute whether to apply classifier-free guidance based on current timestep
        if current_timestep[0] < cfg_trunc_ratio:
            model.prepare_block_swap_before_forward()
            noise_pred_uncond = model(
                img,
                current_timestep,
                cap_feats=neg_txt,  # Gemma2的hidden states作为caption features
                cap_mask=neg_txt_mask.to(dtype=torch.int32),  # Gemma2的attention mask
            )
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            # apply normalization after classifier-free guidance
            if float(renorm_cfg) > 0.0:
                cond_norm = torch.linalg.vector_norm(
                    noise_pred_cond,
                    dim=tuple(range(1, len(noise_pred_cond.shape))),
                    keepdim=True,
                )
                max_new_norms = cond_norm * float(renorm_cfg)
                noise_norms = torch.linalg.vector_norm(
                    noise_pred, dim=tuple(range(1, len(noise_pred.shape))), keepdim=True
                )
                # Iterate through batch
                for i, (noise_norm, max_new_norm) in enumerate(zip(noise_norms, max_new_norms)):
                    if noise_norm >= max_new_norm:
                        noise_pred[i] = noise_pred[i] * (max_new_norm / noise_norm)
        else:
            noise_pred = noise_pred_cond

        img_dtype = img.dtype

        if img.dtype != img_dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                img = img.to(img_dtype)

        # compute the previous noisy sample x_t -> x_t-1
        noise_pred = -noise_pred
        img = scheduler.step(noise_pred, t, img, return_dict=False)[0]

    model.prepare_block_swap_before_forward()
    return img


# endregion


# region train
def get_sigmas(
    noise_scheduler: FlowMatchEulerDiscreteScheduler,
    timesteps: Tensor,
    device: torch.device,
    n_dim=4,
    dtype=torch.float32,
) -> Tensor:
    """
    Get sigmas for timesteps

    Args:
        noise_scheduler (FlowMatchEulerDiscreteScheduler): The noise scheduler instance.
        timesteps (Tensor): A tensor of timesteps for the denoising process.
        device (torch.device): The device on which the tensors are stored.
        n_dim (int, optional): The number of dimensions for the output tensor. Defaults to 4.
        dtype (torch.dtype, optional): The data type for the output tensor. Defaults to torch.float32.

    Returns:
        sigmas (Tensor): The sigmas tensor.
    """
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.

    Args:
        weighting_scheme (str): The weighting scheme to use.
        batch_size (int): The batch size for the sampling process.
        logit_mean (float, optional): The mean of the logit distribution. Defaults to None.
        logit_std (float, optional): The standard deviation of the logit distribution. Defaults to None.
        mode_scale (float, optional): The mode scale for the mode weighting scheme. Defaults to None.

    Returns:
        u (Tensor): The sampled timesteps.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu"
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None) -> Tensor:
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.

    Args:
        weighting_scheme (str): The weighting scheme to use.
        sigmas (Tensor, optional): The sigmas tensor. Defaults to None.

    Returns:
        u (Tensor): The sampled timesteps.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

# mainly copied from flux_train_utils.get_noisy_model_input_and_timesteps
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
    elif args.timestep_sampling == "nextdit_shift":
        sigmas = torch.rand((bsz,), device=device)
        mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
        sigmas = time_shift(mu, 1.0, sigmas)

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


def apply_model_prediction_type(
    args, model_pred: Tensor, noisy_model_input: Tensor, sigmas: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Apply model prediction type to the model prediction and the sigmas.

    Args:
        args (argparse.Namespace): Arguments.
        model_pred (Tensor): Model prediction.
        noisy_model_input (Tensor): Noisy model input.
        sigmas (Tensor): Sigmas.

    Return:
        Tuple[Tensor, Optional[Tensor]]:
    """
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
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=args.weighting_scheme, sigmas=sigmas
        )

    return model_pred, weighting


def save_models(
    ckpt_path: str,
    lumina: lumina_models.NextDiT,
    sai_metadata: Dict[str, Any],
    save_dtype: Optional[torch.dtype] = None,
    use_mem_eff_save: bool = False,
):
    """
    Save the model to the checkpoint path.

    Args:
        ckpt_path (str): Path to the checkpoint.
        lumina (lumina_models.NextDiT): NextDIT model.
        sai_metadata (Optional[dict]): Metadata for the SAI model.
        save_dtype (Optional[torch.dtype]): Data

    Return:
        None
    """
    state_dict = {}

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            if save_dtype is not None and v.dtype != save_dtype:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    update_sd("", lumina.state_dict())

    if not use_mem_eff_save:
        save_file(state_dict, ckpt_path, metadata=sai_metadata)
    else:
        mem_eff_save_file(state_dict, ckpt_path, metadata=sai_metadata)


def save_lumina_model_on_train_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    lumina: lumina_models.NextDiT,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(
            None,
            args,
            False,
            False,
            False,
            is_stable_diffusion_ckpt=True,
            lumina="lumina2",
        )
        save_models(ckpt_file, lumina, sai_metadata, save_dtype, args.mem_eff_save)

    train_util.save_sd_model_on_train_end_common(
        args, True, True, epoch, global_step, sd_saver, None
    )


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合してている
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_lumina_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator: Accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    lumina: lumina_models.NextDiT,
):
    """
    Save the model to the checkpoint path.

    Args:
        args (argparse.Namespace): Arguments.
        save_dtype (torch.dtype): Data type.
        epoch (int): Epoch.
        global_step (int): Global step.
        lumina (lumina_models.NextDiT): NextDIT model.

    Return:
        None
    """

    def sd_saver(ckpt_file: str, epoch_no: int, global_step: int):
        sai_metadata = train_util.get_sai_model_spec(
            {},
            args,
            False,
            False,
            False,
            is_stable_diffusion_ckpt=True,
            lumina="lumina2",
        )
        save_models(ckpt_file, lumina, sai_metadata, save_dtype, args.mem_eff_save)

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


def add_lumina_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--gemma2",
        type=str,
        help="path to gemma2 model (*.sft or *.safetensors), should be float16 / gemma2のパス（*.sftまたは*.safetensors）、float16が前提",
    )
    parser.add_argument(
        "--ae",
        type=str,
        help="path to ae (*.sft or *.safetensors) / aeのパス（*.sftまたは*.safetensors）",
    )
    parser.add_argument(
        "--gemma2_max_token_length",
        type=int,
        default=None,
        help="maximum token length for Gemma2. if omitted, 256"
        " / Gemma2の最大トークン長。省略された場合、256になります",
    )

    parser.add_argument(
        "--timestep_sampling",
        choices=["sigma", "uniform", "sigmoid", "shift", "nextdit_shift", "flux_shift"],
        default="shift",
        help="Method to sample timesteps: sigma-based, uniform random, sigmoid of random normal, shift of sigmoid, Flux.1 and NextDIT.1 shifting. Default is 'shift'."
        " / タイムステップをサンプリングする方法：sigma、random uniform、random normalのsigmoid、sigmoidのシフト、Flux.1、NextDIT.1のシフト。デフォルトは'shift'です。",
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
        "raw (use as is), additive (add to noisy input), sigma_scaled (apply sigma scaling)."
        " / モデル予測の解釈と処理方法："
        "raw（そのまま使用）、additive（ノイズ入力に加算）、sigma_scaled（シグマスケーリングを適用）。",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=6.0,
        help="Discrete flow shift for the Euler Discrete Scheduler, default is 6.0 / Euler Discrete Schedulerの離散フローシフト、デフォルトは6.0",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use Flash Attention for the model / モデルにFlash Attentionを使用する",
    )
    parser.add_argument(
        "--use_sage_attn",
        action="store_true",
        help="Use Sage Attention for the model / モデルにSage Attentionを使用する",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="System prompt to add to the prompt / プロンプトに追加するシステムプロンプト",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=None,
        help="Batch size to use for sampling, defaults to --training_batch_size value. Sample batches are bucketed by width, height, guidance scale, and seed / サンプリングに使用するバッチサイズ。デフォルトは --training_batch_size の値です。サンプルバッチは、幅、高さ、ガイダンススケール、シードによってバケット化されます",
    )
