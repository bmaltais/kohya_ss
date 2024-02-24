import argparse
import importlib
import math
import os
import random
import time
import numpy as np

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
    # if args.xformers or args.sdpa:
    print(f"Stage C: use_xformers_or_sdpa: {args.xformers} {args.sdpa}")
    generator_c.set_use_xformers_or_sdpa(args.xformers, args.sdpa)

    generator_b = sc_utils.load_stage_b_model(args.stage_b_checkpoint_path, dtype=dtype, device=loading_device)
    generator_b.eval().requires_grad_(False).to(loading_device)
    # if args.xformers or args.sdpa:
    print(f"Stage B: use_xformers_or_sdpa: {args.xformers} {args.sdpa}")
    generator_b.set_use_xformers_or_sdpa(args.xformers, args.sdpa)

    # CLIP encoders
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

    # previewer
    if args.previewer_checkpoint_path is not None:
        previewer = sc_utils.load_previewer_model(args.previewer_checkpoint_path, dtype=dtype, device=loading_device)
        previewer.eval().requires_grad_(False)
    else:
        previewer = None

    # LoRA
    if args.network_module:
        for i, network_module in enumerate(args.network_module):
            print("import network module:", network_module)
            imported_module = importlib.import_module(network_module)

            network_mul = 1.0 if args.network_mul is None or len(args.network_mul) <= i else args.network_mul[i]

            net_kwargs = {}
            if args.network_args and i < len(args.network_args):
                network_args = args.network_args[i]
                # TODO escape special chars
                network_args = network_args.split(";")
                for net_arg in network_args:
                    key, value = net_arg.split("=")
                    net_kwargs[key] = value

            if args.network_weights is None or len(args.network_weights) <= i:
                raise ValueError("No weight. Weight is required.")

            network_weight = args.network_weights[i]
            print("load network weights from:", network_weight)

            network, weights_sd = imported_module.create_network_from_weights(
                network_mul, network_weight, effnet, text_model, generator_c, for_inference=True, **net_kwargs
            )
            if network is None:
                return

            mergeable = network.is_mergeable()
            assert mergeable, "not-mergeable network is not supported yet."

            network.merge_to(text_model, generator_c, weights_sd, dtype, device)

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
    b_cfg = 1.1
    b_shift = 1
    b_timesteps = 10
    b_t_start = 1.0

    # caption = "Cinematic photo of an anthropomorphic penguin sitting in a cafe reading a book and having a coffee"
    # height, width = 1024, 1024

    while True:
        print("type caption:")
        # if Ctrl+Z is pressed, it will raise EOFError
        try:
            caption = input()
        except EOFError:
            break

        caption = caption.strip()
        if caption == "":
            continue

        # parse options: '--w' and  '--h' for size, '--l' for cfg, '--s' for timesteps, '--f' for shift. if not specified, use default values
        # e.g. "caption --w 4 --h 4 --l 20 --s 20 --f 1.0"

        tokens = caption.split()
        width = height = 1024
        cfg = 4
        timesteps = 20
        shift = 2
        t_start = 1.0
        negative_prompt = ""
        seed = None

        caption_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if i == len(tokens) - 1:
                caption_tokens.append(token)
                i += 1
                continue

            if token == "--w":
                width = int(tokens[i + 1])
            elif token == "--h":
                height = int(tokens[i + 1])
            elif token == "--l":
                cfg = float(tokens[i + 1])
            elif token == "--s":
                timesteps = int(tokens[i + 1])
            elif token == "--f":
                shift = float(tokens[i + 1])
            elif token == "--t":
                t_start = float(tokens[i + 1])
            elif token == "--n":
                negative_prompt = tokens[i + 1]
            elif token == "--d":
                seed = int(tokens[i + 1])
            else:
                caption_tokens.append(token)
                i += 1
                continue

            i += 2

        caption = " ".join(caption_tokens)

        stage_c_latent_shape, stage_b_latent_shape = sc_utils.calculate_latent_sizes(height, width, batch_size=1)

        # PREPARE CONDITIONS
        # cond_text, cond_pooled = sc.get_clip_conditions([caption], None, tokenizer, text_model)
        input_ids = tokenizer(
            [caption], truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )["input_ids"].to(text_model.device)
        cond_text, cond_pooled = train_util.get_hidden_states_stable_cascade(
            tokenizer.model_max_length, input_ids, tokenizer, text_model
        )
        cond_text = cond_text.to(device, dtype=dtype)
        cond_pooled = cond_pooled.unsqueeze(1).to(device, dtype=dtype)

        # uncond_text, uncond_pooled = sc.get_clip_conditions([""], None, tokenizer, text_model)
        input_ids = tokenizer(
            [negative_prompt], truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )["input_ids"].to(text_model.device)
        uncond_text, uncond_pooled = train_util.get_hidden_states_stable_cascade(
            tokenizer.model_max_length, input_ids, tokenizer, text_model
        )
        uncond_text = uncond_text.to(device, dtype=dtype)
        uncond_pooled = uncond_pooled.unsqueeze(1).to(device, dtype=dtype)

        zero_img_emb = torch.zeros(1, 768, device=device)

        # 辞書にしたくないけど GDF から先の変更が面倒だからとりあえず辞書にしておく
        conditions = {"clip_text_pooled": cond_pooled, "clip": cond_pooled, "clip_text": cond_text, "clip_img": zero_img_emb}
        unconditions = {
            "clip_text_pooled": uncond_pooled,
            "clip": uncond_pooled,
            "clip_text": uncond_text,
            "clip_img": zero_img_emb,
        }
        conditions_b = {}
        conditions_b.update(conditions)
        unconditions_b = {}
        unconditions_b.update(unconditions)

        # seed everything
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            np.random.seed(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

        if args.lowvram:
            generator_c = generator_c.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            sampling_c = gdf_c.sample(
                generator_c,
                conditions,
                stage_c_latent_shape,
                unconditions,
                device=device,
                cfg=cfg,
                shift=shift,
                timesteps=timesteps,
                t_start=t_start,
            )
            for sampled_c, _, _ in tqdm(sampling_c, total=timesteps):
                sampled_c = sampled_c

            conditions_b["effnet"] = sampled_c
            unconditions_b["effnet"] = torch.zeros_like(sampled_c)

        if previewer is not None:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                preview = previewer(sampled_c)
                preview = preview.clamp(0, 1)
            preview = preview.permute(0, 2, 3, 1).squeeze(0)
            preview = preview.detach().float().cpu().numpy()
            preview = Image.fromarray((preview * 255).astype(np.uint8))

            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs(args.outdir, exist_ok=True)
            preview.save(os.path.join(args.outdir, f"preview_{timestamp_str}.png"))

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
                cfg=b_cfg,
                shift=b_shift,
                timesteps=b_timesteps,
                t_start=b_t_start,
            )
            for sampled_b, _, _ in tqdm(sampling_b, total=b_t_start):
                sampled_b = sampled_b

        if args.lowvram:
            generator_b = generator_b.to(loading_device)
            device_utils.clean_memory_on_device(device)
            stage_a = stage_a.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            sampled = stage_a.decode(sampled_b).float()
        # print(sampled.shape, sampled.min(), sampled.max())

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
    sc_utils.add_previewer_arguments(parser)
    sc_utils.add_text_model_arguments(parser)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--sdpa", action="store_true")
    parser.add_argument("--outdir", type=str, default="../outputs", help="dir to write results to / 生成画像の出力先")
    parser.add_argument("--lowvram", action="store_true", help="if specified, use low VRAM mode")
    parser.add_argument(
        "--network_module",
        type=str,
        default=None,
        nargs="*",
        help="additional network module to use / 追加ネットワークを使う時そのモジュール名",
    )
    parser.add_argument(
        "--network_weights", type=str, default=None, nargs="*", help="additional network weights to load / 追加ネットワークの重み"
    )
    parser.add_argument(
        "--network_mul", type=float, default=None, nargs="*", help="additional network multiplier / 追加ネットワークの効果の倍率"
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    args = parser.parse_args()

    main(args)
