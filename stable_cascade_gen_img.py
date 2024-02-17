import argparse
import math
import os
import time

from safetensors.torch import load_file, save_file
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPTextConfig
from PIL import Image
from accelerate import init_empty_weights

import library.stable_cascade as sc
import library.stable_cascade_utils as sc_utils
import library.device_utils as device_utils
from library import train_util
from library.sdxl_model_util import _load_state_dict_on_device


def calculate_latent_sizes(height=1024, width=1024, batch_size=4, compression_factor_b=42.67, compression_factor_a=4.0):
    resolution_multiple = 42.67
    latent_height = math.ceil(height / compression_factor_b)
    latent_width = math.ceil(width / compression_factor_b)
    stage_c_latent_shape = (batch_size, 16, latent_height, latent_width)

    latent_height = math.ceil(height / compression_factor_a)
    latent_width = math.ceil(width / compression_factor_a)
    stage_b_latent_shape = (batch_size, 4, latent_height, latent_width)

    return stage_c_latent_shape, stage_b_latent_shape


def main(args):
    device = device_utils.get_preferred_device()

    loading_device = device if not args.lowvram else "cpu"
    text_model_device = "cpu"

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    text_model_dtype = torch.float32

    # EfficientNet encoder
    effnet = sc_utils.load_effnet(args.effnet_checkpoint_path, loading_device)
    effnet.eval().requires_grad_(False).to(loading_device)

    generator_c = sc_utils.load_stage_c_model(args.stage_c_checkpoint_path, dtype=dtype, device=loading_device)
    generator_c.eval().requires_grad_(False).to(loading_device)

    generator_b = sc_utils.load_stage_b_model(args.stage_b_checkpoint_path, dtype=dtype, device=loading_device)
    generator_b.eval().requires_grad_(False).to(loading_device)

    # CLIP encoders
    print(f"Loading CLIP text model")

    tokenizer = sc_utils.load_tokenizer(args)

    text_model = sc_utils.load_clip_text_model(
        args.text_model_checkpoint_path, text_model_dtype, text_model_device, args.save_text_model
    )
    text_model = text_model.requires_grad_(False).to(text_model_dtype).to(text_model_device)

    # image_model = (
    #     CLIPVisionModelWithProjection.from_pretrained(clip_image_model_name).requires_grad_(False).to(dtype).to(device)
    # )

    # vqGAN
    stage_a = sc_utils.load_stage_a_model(args.stage_a_checkpoint_path, dtype=dtype, device=loading_device)
    stage_a.eval().requires_grad_(False)

    caption = "Cinematic photo of an anthropomorphic penguin sitting in a cafe reading a book and having a coffee"
    height, width = 1024, 1024
    stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=1)

    # 謎のクラス gdf
    gdf_c = sc.GDF(
        schedule=sc.CosineSchedule(clamp_range=[0.0001, 0.9999]),
        input_scaler=sc.VPScaler(),
        target=sc.EpsilonTarget(),
        noise_cond=sc.CosineTNoiseCond(),
        loss_weight=None,
    )
    gdf_b = sc.GDF(
        schedule=sc.CosineSchedule(clamp_range=[0.0001, 0.9999]),
        input_scaler=sc.VPScaler(),
        target=sc.EpsilonTarget(),
        noise_cond=sc.CosineTNoiseCond(),
        loss_weight=None,
    )

    # Stage C Parameters

    # extras.sampling_configs["cfg"] = 4
    # extras.sampling_configs["shift"] = 2
    # extras.sampling_configs["timesteps"] = 20
    # extras.sampling_configs["t_start"] = 1.0

    # # Stage B Parameters
    # extras_b.sampling_configs["cfg"] = 1.1
    # extras_b.sampling_configs["shift"] = 1
    # extras_b.sampling_configs["timesteps"] = 10
    # extras_b.sampling_configs["t_start"] = 1.0

    # PREPARE CONDITIONS
    cond_text, cond_pooled = sc.get_clip_conditions([caption], None, tokenizer, text_model)
    cond_text = cond_text.to(device, dtype=dtype)
    cond_pooled = cond_pooled.to(device, dtype=dtype)

    uncond_text, uncond_pooled = sc.get_clip_conditions([""], None, tokenizer, text_model)
    uncond_text = uncond_text.to(device, dtype=dtype)
    uncond_pooled = uncond_pooled.to(device, dtype=dtype)

    zero_img_emb = torch.zeros(1, 768, device=device)

    # 辞書にしたくないけど GDF から先の変更が面倒だからとりあえず辞書にしておく
    conditions = {"clip_text_pooled": cond_pooled, "clip": cond_pooled, "clip_text": cond_text, "clip_img": zero_img_emb}
    unconditions = {"clip_text_pooled": uncond_pooled, "clip": uncond_pooled, "clip_text": uncond_text, "clip_img": zero_img_emb}
    conditions_b = {}
    conditions_b.update(conditions)
    unconditions_b = {}
    unconditions_b.update(unconditions)

    # torch.manual_seed(42)

    if args.lowvram:
        generator_c = generator_c.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        sampling_c = gdf_c.sample(
            generator_c, conditions, stage_c_latent_shape, unconditions, device=device, cfg=4, shift=2, timesteps=20, t_start=1.0
        )
        for sampled_c, _, _ in tqdm(sampling_c, total=20):
            sampled_c = sampled_c

        conditions_b["effnet"] = sampled_c
        unconditions_b["effnet"] = torch.zeros_like(sampled_c)

    if args.lowvram:
        generator_c = generator_c.to(loading_device)
        device_utils.clean_memory_on_device(device)
        generator_b = generator_b.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        sampling_b = gdf_b.sample(
            generator_b,
            conditions_b,
            stage_b_latent_shape,
            unconditions_b,
            device=device,
            cfg=1.1,
            shift=1,
            timesteps=10,
            t_start=1.0,
        )
        for sampled_b, _, _ in tqdm(sampling_b, total=10):
            sampled_b = sampled_b

    if args.lowvram:
        generator_b = generator_b.to(loading_device)
        device_utils.clean_memory_on_device(device)
        stage_a = stage_a.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        sampled = stage_a.decode(sampled_b).float()
        print(sampled.shape, sampled.min(), sampled.max())

    if args.lowvram:
        stage_a = stage_a.to(loading_device)
        device_utils.clean_memory_on_device(device)

    # float 0-1 to PIL Image
    sampled = sampled.clamp(0, 1)
    sampled = sampled.mul(255).to(dtype=torch.uint8)
    sampled = sampled.permute(0, 2, 3, 1)
    sampled = sampled.cpu().numpy()
    sampled = Image.fromarray(sampled[0])

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.outdir, exist_ok=True)
    sampled.save(os.path.join(args.outdir, f"sampled_{timestamp_str}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    sc_utils.add_effnet_arguments(parser)
    train_util.add_tokenizer_arguments(parser)
    sc_utils.add_stage_a_arguments(parser)
    sc_utils.add_stage_b_arguments(parser)
    sc_utils.add_stage_c_arguments(parser)
    sc_utils.add_text_model_arguments(parser)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--outdir", type=str, default="../outputs", help="dir to write results to / 生成画像の出力先")
    parser.add_argument("--lowvram", action="store_true", help="if specified, use low VRAM mode")
    args = parser.parse_args()

    main(args)
