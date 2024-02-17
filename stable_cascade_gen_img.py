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
import library.device_utils as device_utils
from library.sdxl_model_util import _load_state_dict_on_device

clip_text_model_name: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"


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
    print(f"Loading EfficientNet encoder from {args.effnet_checkpoint_path}")
    effnet = sc.EfficientNetEncoder()
    effnet_checkpoint = load_file(args.effnet_checkpoint_path)
    info = effnet.load_state_dict(effnet_checkpoint if "state_dict" not in effnet_checkpoint else effnet_checkpoint["state_dict"])
    print(info)
    effnet.eval().requires_grad_(False).to(loading_device)
    del effnet_checkpoint

    # Generator
    print(f"Instantiating Stage C generator")
    with init_empty_weights():
        generator_c = sc.StageC()
    print(f"Loading Stage C generator from {args.stage_c_checkpoint_path}")
    stage_c_checkpoint = load_file(args.stage_c_checkpoint_path)
    print(f"Loading state dict")
    info = _load_state_dict_on_device(generator_c, stage_c_checkpoint, loading_device, dtype=dtype)
    print(info)
    generator_c.eval().requires_grad_(False).to(loading_device)

    print(f"Instantiating Stage B generator")
    with init_empty_weights():
        generator_b = sc.StageB()
    print(f"Loading Stage B generator from {args.stage_b_checkpoint_path}")
    stage_b_checkpoint = load_file(args.stage_b_checkpoint_path)
    print(f"Loading state dict")
    info = _load_state_dict_on_device(generator_b, stage_b_checkpoint, loading_device, dtype=dtype)
    print(info)
    generator_b.eval().requires_grad_(False).to(loading_device)

    # CLIP encoders
    print(f"Loading CLIP text model")

    # TODO 完全にオフラインで動かすには tokenizer もローカルに保存できるようにする必要がある
    tokenizer = AutoTokenizer.from_pretrained(clip_text_model_name)

    if args.save_text_model or args.text_model_checkpoint_path is None:
        print(f"Loading CLIP text model from {clip_text_model_name}")
        text_model = CLIPTextModelWithProjection.from_pretrained(clip_text_model_name)

        if args.save_text_model:
            sd = text_model.state_dict()
            print(f"Saving CLIP text model to {args.text_model_checkpoint_path}")
            save_file(sd, args.text_model_checkpoint_path)
    else:
        print(f"Loading CLIP text model from {args.text_model_checkpoint_path}")

        # copy from sdxl_model_util.py
        text_model2_cfg = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=1280,
            intermediate_size=5120,
            num_hidden_layers=32,
            num_attention_heads=20,
            max_position_embeddings=77,
            hidden_act="gelu",
            layer_norm_eps=1e-05,
            dropout=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            model_type="clip_text_model",
            projection_dim=1280,
            # torch_dtype="float32",
            # transformers_version="4.25.0.dev0",
        )
        with init_empty_weights():
            text_model = CLIPTextModelWithProjection(text_model2_cfg)

        text_model_checkpoint = load_file(args.text_model_checkpoint_path)
        info = _load_state_dict_on_device(text_model, text_model_checkpoint, text_model_device, dtype=text_model_dtype)
        print(info)

    text_model = text_model.requires_grad_(False).to(text_model_dtype).to(text_model_device)
    # image_model = (
    #     CLIPVisionModelWithProjection.from_pretrained(clip_image_model_name).requires_grad_(False).to(dtype).to(device)
    # )

    # vqGAN
    print(f"Loading Stage A vqGAN from {args.stage_a_checkpoint_path}")
    stage_a = sc.StageA().to(loading_device)
    stage_a_checkpoint = load_file(args.stage_a_checkpoint_path)
    info = stage_a.load_state_dict(
        stage_a_checkpoint if "state_dict" not in stage_a_checkpoint else stage_a_checkpoint["state_dict"]
    )
    print(info)
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
    cond_text, cond_pooled = sc.get_clip_conditions([caption], tokenizer, text_model)
    cond_text = cond_text.to(device, dtype=dtype)
    cond_pooled = cond_pooled.to(device, dtype=dtype)

    uncond_text, uncond_pooled = sc.get_clip_conditions([""], tokenizer, text_model)
    uncond_text = uncond_text.to(device, dtype=dtype)
    uncond_pooled = uncond_pooled.to(device, dtype=dtype)

    img_emb = torch.zeros(1, 768, device=device)

    # 辞書にしたくないけど GDF から先の変更が面倒だからとりあえず辞書にしておく
    conditions = {"clip_text_pooled": cond_pooled, "clip": cond_pooled, "clip_text": cond_text, "clip_img": img_emb}
    unconditions = {"clip_text_pooled": uncond_pooled, "clip": uncond_pooled, "clip_text": uncond_text, "clip_img": img_emb}
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
    parser.add_argument("--effnet_checkpoint_path", type=str, required=True)
    parser.add_argument("--stage_a_checkpoint_path", type=str, required=True)
    parser.add_argument("--stage_b_checkpoint_path", type=str, required=True)
    parser.add_argument("--stage_c_checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--text_model_checkpoint_path", type=str, required=False, default=None, help="if omitted, download from HuggingFace"
    )
    parser.add_argument("--save_text_model", action="store_true", help="if specified, save text model to corresponding path")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--outdir", type=str, default="../outputs", help="dir to write results to / 生成画像の出力先")
    parser.add_argument("--lowvram", action="store_true", help="if specified, use low VRAM mode")
    args = parser.parse_args()

    main(args)
