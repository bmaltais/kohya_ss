# Minimum Inference Code for SD3

import argparse
import datetime
import math
import os
import random
from typing import Optional, Tuple
import numpy as np

import torch
from safetensors.torch import safe_open, load_file
import torch.amp
from tqdm import tqdm
from PIL import Image
from transformers import CLIPTextModelWithProjection, T5EncoderModel

from library.device_utils import init_ipex, get_preferred_device
from networks import lora_sd3

init_ipex()

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import sd3_models, sd3_utils, strategy_sd3
from library.utils import load_safetensors


def get_noise(seed, latent, device="cpu"):
    # generator = torch.manual_seed(seed)
    generator = torch.Generator(device)
    generator.manual_seed(seed)
    return torch.randn(latent.size(), dtype=latent.dtype, layout=latent.layout, generator=generator, device=device)


def get_sigmas(sampling: sd3_utils.ModelSamplingDiscreteFlow, steps):
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
    initial_latent: Optional[torch.Tensor],
    seed: int,
    cond: Tuple[torch.Tensor, torch.Tensor],
    neg_cond: Tuple[torch.Tensor, torch.Tensor],
    mmdit: sd3_models.MMDiT,
    steps: int,
    cfg_scale: float,
    dtype: torch.dtype,
    device: str,
):
    if initial_latent is None:
        # latent = torch.ones(1, 16, height // 8, width // 8, device=device) * 0.0609 # this seems to be a bug in the original code. thanks to furusu for pointing it out
        latent = torch.zeros(1, 16, height // 8, width // 8, device=device)
    else:
        latent = initial_latent

    latent = latent.to(dtype).to(device)

    noise = get_noise(seed, latent, device)

    model_sampling = sd3_utils.ModelSamplingDiscreteFlow(shift=3.0)  # 3.0 is for SD3

    sigmas = get_sigmas(model_sampling, steps).to(device)
    # sigmas = sigmas[int(steps * (1 - denoise)) :] # do not support i2i

    # conditioning = fix_cond(conditioning)
    # neg_cond = fix_cond(neg_cond)
    # extra_args = {"cond": cond, "uncond": neg_cond, "cond_scale": guidance_scale}

    noise_scaled = model_sampling.noise_scaling(sigmas[0], noise, latent, max_denoise(model_sampling, sigmas))

    c_crossattn = torch.cat([cond[0], neg_cond[0]]).to(device).to(dtype)
    y = torch.cat([cond[1], neg_cond[1]]).to(device).to(dtype)

    x = noise_scaled.to(device).to(dtype)
    # print(x.shape)

    with torch.no_grad():
        for i in tqdm(range(len(sigmas) - 1)):
            sigma_hat = sigmas[i]

            timestep = model_sampling.timestep(sigma_hat).float()
            timestep = torch.FloatTensor([timestep, timestep]).to(device)

            x_c_nc = torch.cat([x, x], dim=0)
            # print(x_c_nc.shape, timestep.shape, c_crossattn.shape, y.shape)

            with torch.autocast(device_type=device.type, dtype=dtype):
                model_output = mmdit(x_c_nc, timestep, context=c_crossattn, y=y)
            model_output = model_output.float()
            batched = model_sampling.calculate_denoised(sigma_hat, model_output, x)

            pos_out, neg_out = batched.chunk(2)
            denoised = neg_out + (pos_out - neg_out) * cfg_scale
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

    latent = x
    latent = vae.process_out(latent)
    return latent


def generate_image(
    mmdit: sd3_models.MMDiT,
    vae: sd3_models.SDVAE,
    clip_l: CLIPTextModelWithProjection,
    clip_g: CLIPTextModelWithProjection,
    t5xxl: T5EncoderModel,
    steps: int,
    prompt: str,
    seed: int,
    target_width: int,
    target_height: int,
    device: str,
    negative_prompt: str,
    cfg_scale: float,
):
    # prepare embeddings
    logger.info("Encoding prompts...")

    # TODO support one-by-one offloading
    clip_l.to(device)
    clip_g.to(device)
    t5xxl.to(device)

    with torch.autocast(device_type=device.type, dtype=mmdit.dtype), torch.no_grad():
        tokens_and_masks = tokenize_strategy.tokenize(prompt)
        lg_out, t5_out, pooled, l_attn_mask, g_attn_mask, t5_attn_mask = encoding_strategy.encode_tokens(
            tokenize_strategy, [clip_l, clip_g, t5xxl], tokens_and_masks, args.apply_lg_attn_mask, args.apply_t5_attn_mask
        )
        cond = encoding_strategy.concat_encodings(lg_out, t5_out, pooled)

        tokens_and_masks = tokenize_strategy.tokenize(negative_prompt)
        lg_out, t5_out, pooled, neg_l_attn_mask, neg_g_attn_mask, neg_t5_attn_mask = encoding_strategy.encode_tokens(
            tokenize_strategy, [clip_l, clip_g, t5xxl], tokens_and_masks, args.apply_lg_attn_mask, args.apply_t5_attn_mask
        )
        neg_cond = encoding_strategy.concat_encodings(lg_out, t5_out, pooled)

    # attn masks are not used currently

    if args.offload:
        clip_l.to("cpu")
        clip_g.to("cpu")
        t5xxl.to("cpu")

    # generate image
    logger.info("Generating image...")
    mmdit.to(device)
    latent_sampled = do_sample(target_height, target_width, None, seed, cond, neg_cond, mmdit, steps, cfg_scale, sd3_dtype, device)
    if args.offload:
        mmdit.to("cpu")

    # latent to image
    vae.to(device)
    with torch.no_grad():
        image = vae.decode(latent_sampled)

    if args.offload:
        vae.to("cpu")

    image = image.float()
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
    decoded_np = decoded_np.astype(np.uint8)
    out_image = Image.fromarray(decoded_np)

    # save image
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    out_image.save(output_path)

    logger.info(f"Saved image to {output_path}")


if __name__ == "__main__":
    target_height = 1024
    target_width = 1024

    # steps = 50  # 28  # 50
    # cfg_scale = 5
    # seed = 1  # None  # 1

    device = get_preferred_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--clip_g", type=str, required=False)
    parser.add_argument("--clip_l", type=str, required=False)
    parser.add_argument("--t5xxl", type=str, required=False)
    parser.add_argument("--t5xxl_token_length", type=int, default=256, help="t5xxl token length, default: 256")
    parser.add_argument("--apply_lg_attn_mask", action="store_true")
    parser.add_argument("--apply_t5_attn_mask", action="store_true")
    parser.add_argument("--prompt", type=str, default="A photo of a cat")
    # parser.add_argument("--prompt2", type=str, default=None)  # do not support different prompts for text encoders
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--offload", action="store_true", help="Offload to CPU")
    parser.add_argument("--output_dir", type=str, default=".")
    # parser.add_argument("--do_not_use_t5xxl", action="store_true")
    # parser.add_argument("--attn_mode", type=str, default="torch", help="torch (SDPA) or xformers. default: torch")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--lora_weights",
        type=str,
        nargs="*",
        default=[],
        help="LoRA weights, only supports networks.lora_sd3, each argument is a `path;multiplier` (semi-colon separated)",
    )
    parser.add_argument("--merge_lora_weights", action="store_true", help="Merge LoRA weights to model")
    parser.add_argument("--width", type=int, default=target_width)
    parser.add_argument("--height", type=int, default=target_height)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    seed = args.seed
    steps = args.steps

    sd3_dtype = torch.float32
    if args.fp16:
        sd3_dtype = torch.float16
    elif args.bf16:
        sd3_dtype = torch.bfloat16

    loading_device = "cpu" if args.offload else device

    # load state dict
    logger.info(f"Loading SD3 models from {args.ckpt_path}...")
    # state_dict = load_file(args.ckpt_path)
    state_dict = load_safetensors(args.ckpt_path, loading_device, disable_mmap=True, dtype=sd3_dtype)

    # load text encoders
    clip_l = sd3_utils.load_clip_l(args.clip_l, sd3_dtype, loading_device, state_dict=state_dict)
    clip_g = sd3_utils.load_clip_g(args.clip_g, sd3_dtype, loading_device, state_dict=state_dict)
    t5xxl = sd3_utils.load_t5xxl(args.t5xxl, sd3_dtype, loading_device, state_dict=state_dict)

    # MMDiT and VAE
    vae = sd3_utils.load_vae(None, sd3_dtype, loading_device, state_dict=state_dict)
    mmdit = sd3_utils.load_mmdit(state_dict, sd3_dtype, loading_device)

    clip_l.to(sd3_dtype)
    clip_g.to(sd3_dtype)
    t5xxl.to(sd3_dtype)
    vae.to(sd3_dtype)
    mmdit.to(sd3_dtype)
    if not args.offload:
        # make sure to move to the device: some tensors are created in the constructor on the CPU
        clip_l.to(device)
        clip_g.to(device)
        t5xxl.to(device)
        vae.to(device)
        mmdit.to(device)

    clip_l.eval()
    clip_g.eval()
    t5xxl.eval()
    mmdit.eval()
    vae.eval()

    # load tokenizers
    logger.info("Loading tokenizers...")
    tokenize_strategy = strategy_sd3.Sd3TokenizeStrategy(args.t5xxl_token_length)
    encoding_strategy = strategy_sd3.Sd3TextEncodingStrategy()

    # LoRA
    lora_models: list[lora_sd3.LoRANetwork] = []
    for weights_file in args.lora_weights:
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 1.0

        weights_sd = load_file(weights_file)
        module = lora_sd3
        lora_model, _ = module.create_network_from_weights(multiplier, None, vae, [clip_l, clip_g, t5xxl], mmdit, weights_sd, True)

        if args.merge_lora_weights:
            lora_model.merge_to([clip_l, clip_g, t5xxl], mmdit, weights_sd)
        else:
            lora_model.apply_to([clip_l, clip_g, t5xxl], mmdit)
            info = lora_model.load_state_dict(weights_sd, strict=True)
            logger.info(f"Loaded LoRA weights from {weights_file}: {info}")
            lora_model.eval()
            lora_model.to(device)

        lora_models.append(lora_model)

    if not args.interactive:
        generate_image(
            mmdit,
            vae,
            clip_l,
            clip_g,
            t5xxl,
            args.steps,
            args.prompt,
            args.seed,
            args.width,
            args.height,
            device,
            args.negative_prompt,
            args.cfg_scale,
        )
    else:
        # loop for interactive
        width = args.width
        height = args.height
        steps = None
        cfg_scale = args.cfg_scale

        while True:
            print(
                "Enter prompt (empty to exit). Options: --w <width> --h <height> --s <steps> --d <seed>"
                " --n <negative prompt>, `--n -` for empty negative prompt"
                "Options are kept for the next prompt. Current options:"
                f" width={width}, height={height}, steps={steps}, seed={seed}, cfg_scale={cfg_scale}"
            )
            prompt = input()
            if prompt == "":
                break

            # parse options
            options = prompt.split("--")
            prompt = options[0].strip()
            seed = None
            negative_prompt = None
            for opt in options[1:]:
                try:
                    opt = opt.strip()
                    if opt.startswith("w"):
                        width = int(opt[1:].strip())
                    elif opt.startswith("h"):
                        height = int(opt[1:].strip())
                    elif opt.startswith("s"):
                        steps = int(opt[1:].strip())
                    elif opt.startswith("d"):
                        seed = int(opt[1:].strip())
                    elif opt.startswith("m"):
                        mutipliers = opt[1:].strip().split(",")
                        if len(mutipliers) != len(lora_models):
                            logger.error(f"Invalid number of multipliers, expected {len(lora_models)}")
                            continue
                        for i, lora_model in enumerate(lora_models):
                            lora_model.set_multiplier(float(mutipliers[i]))
                    elif opt.startswith("n"):
                        negative_prompt = opt[1:].strip()
                        if negative_prompt == "-":
                            negative_prompt = ""
                    elif opt.startswith("c"):
                        cfg_scale = float(opt[1:].strip())
                except ValueError as e:
                    logger.error(f"Invalid option: {opt}, {e}")

            generate_image(
                mmdit,
                vae,
                clip_l,
                clip_g,
                t5xxl,
                steps if steps is not None else args.steps,
                prompt,
                seed if seed is not None else args.seed,
                width,
                height,
                device,
                negative_prompt if negative_prompt is not None else args.negative_prompt,
                cfg_scale,
            )

    logger.info("Done!")
