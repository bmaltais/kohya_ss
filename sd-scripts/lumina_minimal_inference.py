# Minimum Inference Code for Lumina
# Based on flux_minimal_inference.py

import logging
import argparse
import math
import os
import random
import time
from typing import Optional

import einops
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import Gemma2Model
from library.flux_models import AutoEncoder

from library import (
    device_utils,
    lumina_models,
    lumina_train_util,
    lumina_util,
    sd3_train_utils,
    strategy_lumina,
)
import networks.lora_lumina as lora_lumina
from library.device_utils import get_preferred_device, init_ipex
from library.utils import setup_logging, str_to_dtype

init_ipex()
setup_logging()
logger = logging.getLogger(__name__)


def generate_image(
    model: lumina_models.NextDiT,
    gemma2: Gemma2Model,
    ae: AutoEncoder,
    prompt: str,
    system_prompt: str,
    seed: Optional[int],
    image_width: int,
    image_height: int,
    steps: int,
    guidance_scale: float,
    negative_prompt: Optional[str],
    args: argparse.Namespace,
    cfg_trunc_ratio: float = 0.25,
    renorm_cfg: float = 1.0,
):
    #
    # 0. Prepare arguments
    #
    device = get_preferred_device()
    if args.device:
        device = torch.device(args.device)

    dtype = str_to_dtype(args.dtype)
    ae_dtype = str_to_dtype(args.ae_dtype)
    gemma2_dtype = str_to_dtype(args.gemma2_dtype)

    #
    # 1. Prepare models
    #
    # model.to(device, dtype=dtype)
    model.to(dtype)
    model.eval()

    gemma2.to(device, dtype=gemma2_dtype)
    gemma2.eval()

    ae.to(ae_dtype)
    ae.eval()

    #
    # 2. Encode prompts
    #
    logger.info("Encoding prompts...")

    tokenize_strategy = strategy_lumina.LuminaTokenizeStrategy(system_prompt, args.gemma2_max_token_length)
    encoding_strategy = strategy_lumina.LuminaTextEncodingStrategy()

    tokens_and_masks = tokenize_strategy.tokenize(prompt)
    with torch.no_grad():
        gemma2_conds = encoding_strategy.encode_tokens(tokenize_strategy, [gemma2], tokens_and_masks)

    tokens_and_masks = tokenize_strategy.tokenize(
        negative_prompt, is_negative=True and not args.add_system_prompt_to_negative_prompt
    )
    with torch.no_grad():
        neg_gemma2_conds = encoding_strategy.encode_tokens(tokenize_strategy, [gemma2], tokens_and_masks)

    # Unpack Gemma2 outputs
    prompt_hidden_states, _, prompt_attention_mask = gemma2_conds
    uncond_hidden_states, _, uncond_attention_mask = neg_gemma2_conds

    if args.offload:
        print("Offloading models to CPU to save VRAM...")
        gemma2.to("cpu")
        device_utils.clean_memory()

    model.to(device)

    #
    # 3. Prepare latents
    #
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    logger.info(f"Seed: {seed}")
    torch.manual_seed(seed)

    latent_height = image_height // 8
    latent_width = image_width // 8
    latent_channels = 16

    latents = torch.randn(
        (1, latent_channels, latent_height, latent_width),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    #
    # 4. Denoise
    #
    logger.info("Denoising...")
    scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps

    # # compare with lumina_train_util.retrieve_timesteps
    # lumina_timestep = lumina_train_util.retrieve_timesteps(scheduler, num_inference_steps=steps)
    # print(f"Using timesteps: {timesteps}")
    # print(f"vs Lumina timesteps: {lumina_timestep}")  # should be the same

    with torch.autocast(device_type=device.type, dtype=dtype), torch.no_grad():
        latents = lumina_train_util.denoise(
            scheduler,
            model,
            latents.to(device),
            prompt_hidden_states.to(device),
            prompt_attention_mask.to(device),
            uncond_hidden_states.to(device),
            uncond_attention_mask.to(device),
            timesteps,
            guidance_scale,
            cfg_trunc_ratio,
            renorm_cfg,
        )

    if args.offload:
        model.to("cpu")
        device_utils.clean_memory()
        ae.to(device)

    #
    # 5. Decode latents
    #
    logger.info("Decoding image...")
    # latents = latents / ae.scale_factor + ae.shift_factor
    with torch.no_grad():
        image = ae.decode(latents.to(ae_dtype))
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")

    #
    # 6. Save image
    #
    pil_image = Image.fromarray(image[0])
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    seed_suffix = f"_{seed}"
    output_path = os.path.join(output_dir, f"image_{ts_str}{seed_suffix}.png")
    pil_image.save(output_path)
    logger.info(f"Image saved to {output_path}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Lumina DiT model path / Lumina DiTモデルのパス",
    )
    parser.add_argument(
        "--gemma2_path",
        type=str,
        default=None,
        required=True,
        help="Gemma2 model path / Gemma2モデルのパス",
    )
    parser.add_argument(
        "--ae_path",
        type=str,
        default=None,
        required=True,
        help="Autoencoder model path / Autoencoderモデルのパス",
    )
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the mountains", help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for image generation, default is empty")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for generated images")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=36, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--image_width", type=int, default=1024, help="Image width")
    parser.add_argument("--image_height", type=int, default=1024, help="Image height")
    parser.add_argument("--dtype", type=str, default="bf16", help="Data type for model (bf16, fp16, float)")
    parser.add_argument("--gemma2_dtype", type=str, default="bf16", help="Data type for Gemma2 (bf16, fp16, float)")
    parser.add_argument("--ae_dtype", type=str, default="bf16", help="Data type for Autoencoder (bf16, fp16, float)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda:0')")
    parser.add_argument("--offload", action="store_true", help="Offload models to CPU to save VRAM")
    parser.add_argument("--system_prompt", type=str, default="", help="System prompt for Gemma2 model")
    parser.add_argument("--add_system_prompt_to_negative_prompt", action="store_true", help="Add system prompt to negative prompt")
    parser.add_argument(
        "--gemma2_max_token_length",
        type=int,
        default=256,
        help="Max token length for Gemma2 tokenizer",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=6.0,
        help="Shift value for FlowMatchEulerDiscreteScheduler",
    )
    parser.add_argument(
        "--cfg_trunc_ratio",
        type=float,
        default=0.25,
        help="The ratio of the timestep interval to apply normalization-based guidance scale. For example, 0.25 means the first 25%% of timesteps will be guided.",
    )
    parser.add_argument(
        "--renorm_cfg",
        type=float,
        default=1.0,
        help="The factor to limit the maximum norm after guidance. Default: 1.0, 0.0 means no renormalization.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use flash attention for Lumina model",
    )
    parser.add_argument(
        "--use_sage_attn",
        action="store_true",
        help="Use sage attention for Lumina model",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        nargs="*",
        default=[],
        help="LoRA weights, each argument is a `path;multiplier` (semi-colon separated)",
    )
    parser.add_argument("--merge_lora_weights", action="store_true", help="Merge LoRA weights to model")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode for generating multiple images / 対話モードで複数の画像を生成する",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    logger.info("Loading models...")
    device = get_preferred_device()
    if args.device:
        device = torch.device(args.device)

    # Load Lumina DiT model
    model = lumina_util.load_lumina_model(
        args.pretrained_model_name_or_path,
        dtype=None,  # Load in fp32 and then convert
        device="cpu",
        use_flash_attn=args.use_flash_attn,
        use_sage_attn=args.use_sage_attn,
    )

    # Load Gemma2
    gemma2 = lumina_util.load_gemma2(args.gemma2_path, dtype=None, device="cpu")

    # Load Autoencoder
    ae = lumina_util.load_ae(args.ae_path, dtype=None, device="cpu")

    # LoRA
    lora_models = []
    for weights_file in args.lora_weights:
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 1.0

        weights_sd = load_file(weights_file)
        lora_model, _ = lora_lumina.create_network_from_weights(multiplier, None, ae, [gemma2], model, weights_sd, True)

        if args.merge_lora_weights:
            lora_model.merge_to([gemma2], model, weights_sd)
        else:
            lora_model.apply_to([gemma2], model)
            info = lora_model.load_state_dict(weights_sd, strict=True)
            logger.info(f"Loaded LoRA weights from {weights_file}: {info}")
            lora_model.to(device)
            lora_model.set_multiplier(multiplier)
            lora_model.eval()

        lora_models.append(lora_model)

    if not args.interactive:
        generate_image(
            model,
            gemma2,
            ae,
            args.prompt,
            args.system_prompt,
            args.seed,
            args.image_width,
            args.image_height,
            args.steps,
            args.guidance_scale,
            args.negative_prompt,
            args,
            args.cfg_trunc_ratio,
            args.renorm_cfg,
        )
    else:
        # Interactive mode loop
        image_width = args.image_width
        image_height = args.image_height
        steps = args.steps
        guidance_scale = args.guidance_scale
        cfg_trunc_ratio = args.cfg_trunc_ratio
        renorm_cfg = args.renorm_cfg

        print("Entering interactive mode.")
        while True:
            print(
                "\nEnter prompt (or 'exit'). Options: --w <int> --h <int> --s <int> --d <int> --g <float> --n <str> --ctr <float> --rcfg <float> --m <m1,m2...>"
            )
            user_input = input()
            if user_input.lower() == "exit":
                break
            if not user_input:
                continue

            # Parse options
            options = user_input.split("--")
            prompt = options[0].strip()

            # Set defaults for each generation
            seed = None  # New random seed each time unless specified
            negative_prompt = args.negative_prompt  # Reset to default

            for opt in options[1:]:
                try:
                    opt = opt.strip()
                    if not opt:
                        continue

                    key, value = (opt.split(None, 1) + [""])[:2]

                    if key == "w":
                        image_width = int(value)
                    elif key == "h":
                        image_height = int(value)
                    elif key == "s":
                        steps = int(value)
                    elif key == "d":
                        seed = int(value)
                    elif key == "g":
                        guidance_scale = float(value)
                    elif key == "n":
                        negative_prompt = value if value != "-" else ""
                    elif key == "ctr":
                        cfg_trunc_ratio = float(value)
                    elif key == "rcfg":
                        renorm_cfg = float(value)
                    elif key == "m":
                        multipliers = value.split(",")
                        if len(multipliers) != len(lora_models):
                            logger.error(f"Invalid number of multipliers, expected {len(lora_models)}")
                            continue
                        for i, lora_model in enumerate(lora_models):
                            lora_model.set_multiplier(float(multipliers[i].strip()))
                    else:
                        logger.warning(f"Unknown option: --{key}")

                except (ValueError, IndexError) as e:
                    logger.error(f"Invalid value for option --{key}: '{value}'. Error: {e}")

            generate_image(
                model,
                gemma2,
                ae,
                prompt,
                args.system_prompt,
                seed,
                image_width,
                image_height,
                steps,
                guidance_scale,
                negative_prompt,
                args,
                cfg_trunc_ratio,
                renorm_cfg,
            )

    logger.info("Done.")
