import argparse
import os
import gc
from typing import Dict, Optional, Union
import torch
from safetensors.torch import safe_open

from library.utils import setup_logging
from library.utils import str_to_dtype
from library.safetensors_utils import load_safetensors, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)


def merge_safetensors(
    dit_path: str,
    vae_path: Optional[str] = None,
    clip_l_path: Optional[str] = None,
    clip_g_path: Optional[str] = None,
    t5xxl_path: Optional[str] = None,
    output_path: str = "merged_model.safetensors",
    device: str = "cpu",
    save_precision: Optional[str] = None,
):
    """
    Merge multiple safetensors files into a single file

    Args:
        dit_path: Path to the DiT/MMDiT model
        vae_path: Path to the VAE model
        clip_l_path: Path to the CLIP-L model
        clip_g_path: Path to the CLIP-G model
        t5xxl_path: Path to the T5-XXL model
        output_path: Path to save the merged model
        device: Device to load tensors to
        save_precision: Target dtype for model weights (e.g. 'fp16', 'bf16')
    """
    logger.info("Starting to merge safetensors files...")

    # Convert save_precision string to torch dtype if specified
    if save_precision:
        target_dtype = str_to_dtype(save_precision)
    else:
        target_dtype = None

    # 1. Get DiT metadata if available
    metadata = None
    try:
        with safe_open(dit_path, framework="pt") as f:
            metadata = f.metadata()  # may be None
            if metadata:
                logger.info(f"Found metadata in DiT model: {metadata}")
    except Exception as e:
        logger.warning(f"Failed to read metadata from DiT model: {e}")

    # 2. Create empty merged state dict
    merged_state_dict = {}

    # 3. Load and merge each model with memory management

    # DiT/MMDiT - prefix: model.diffusion_model.
    # This state dict may have VAE keys.
    logger.info(f"Loading DiT model from {dit_path}")
    dit_state_dict = load_safetensors(dit_path, device=device, disable_mmap=True, dtype=target_dtype)
    logger.info(f"Adding DiT model with {len(dit_state_dict)} keys")
    for key, value in dit_state_dict.items():
        if key.startswith("model.diffusion_model.") or key.startswith("first_stage_model."):
            merged_state_dict[key] = value
        else:
            merged_state_dict[f"model.diffusion_model.{key}"] = value
    # Free memory
    del dit_state_dict
    gc.collect()

    # VAE - prefix: first_stage_model.
    # May be omitted if VAE is already included in DiT model.
    if vae_path:
        logger.info(f"Loading VAE model from {vae_path}")
        vae_state_dict = load_safetensors(vae_path, device=device, disable_mmap=True, dtype=target_dtype)
        logger.info(f"Adding VAE model with {len(vae_state_dict)} keys")
        for key, value in vae_state_dict.items():
            if key.startswith("first_stage_model."):
                merged_state_dict[key] = value
            else:
                merged_state_dict[f"first_stage_model.{key}"] = value
        # Free memory
        del vae_state_dict
        gc.collect()

    # CLIP-L - prefix: text_encoders.clip_l.
    if clip_l_path:
        logger.info(f"Loading CLIP-L model from {clip_l_path}")
        clip_l_state_dict = load_safetensors(clip_l_path, device=device, disable_mmap=True, dtype=target_dtype)
        logger.info(f"Adding CLIP-L model with {len(clip_l_state_dict)} keys")
        for key, value in clip_l_state_dict.items():
            if key.startswith("text_encoders.clip_l.transformer."):
                merged_state_dict[key] = value
            else:
                merged_state_dict[f"text_encoders.clip_l.transformer.{key}"] = value
        # Free memory
        del clip_l_state_dict
        gc.collect()

    # CLIP-G - prefix: text_encoders.clip_g.
    if clip_g_path:
        logger.info(f"Loading CLIP-G model from {clip_g_path}")
        clip_g_state_dict = load_safetensors(clip_g_path, device=device, disable_mmap=True, dtype=target_dtype)
        logger.info(f"Adding CLIP-G model with {len(clip_g_state_dict)} keys")
        for key, value in clip_g_state_dict.items():
            if key.startswith("text_encoders.clip_g.transformer."):
                merged_state_dict[key] = value
            else:
                merged_state_dict[f"text_encoders.clip_g.transformer.{key}"] = value
        # Free memory
        del clip_g_state_dict
        gc.collect()

    # T5-XXL - prefix: text_encoders.t5xxl.
    if t5xxl_path:
        logger.info(f"Loading T5-XXL model from {t5xxl_path}")
        t5xxl_state_dict = load_safetensors(t5xxl_path, device=device, disable_mmap=True, dtype=target_dtype)
        logger.info(f"Adding T5-XXL model with {len(t5xxl_state_dict)} keys")
        for key, value in t5xxl_state_dict.items():
            if key.startswith("text_encoders.t5xxl.transformer."):
                merged_state_dict[key] = value
            else:
                merged_state_dict[f"text_encoders.t5xxl.transformer.{key}"] = value
        # Free memory
        del t5xxl_state_dict
        gc.collect()

    # 4. Save merged state dict
    logger.info(f"Saving merged model to {output_path} with {len(merged_state_dict)} keys total")
    mem_eff_save_file(merged_state_dict, output_path, metadata)
    logger.info("Successfully merged safetensors files")


def main():
    parser = argparse.ArgumentParser(description="Merge Stable Diffusion 3.5 model components into a single safetensors file")
    parser.add_argument("--dit", required=True, help="Path to the DiT/MMDiT model")
    parser.add_argument("--vae", help="Path to the VAE model. May be omitted if VAE is included in DiT model")
    parser.add_argument("--clip_l", help="Path to the CLIP-L model")
    parser.add_argument("--clip_g", help="Path to the CLIP-G model")
    parser.add_argument("--t5xxl", help="Path to the T5-XXL model")
    parser.add_argument("--output", default="merged_model.safetensors", help="Path to save the merged model")
    parser.add_argument("--device", default="cpu", help="Device to load tensors to")
    parser.add_argument("--save_precision", type=str, help="Precision to save the model in (e.g., 'fp16', 'bf16', 'float16', etc.)")

    args = parser.parse_args()

    merge_safetensors(
        dit_path=args.dit,
        vae_path=args.vae,
        clip_l_path=args.clip_l,
        clip_g_path=args.clip_g,
        t5xxl_path=args.t5xxl,
        output_path=args.output,
        device=args.device,
        save_precision=args.save_precision,
    )


if __name__ == "__main__":
    main()
