import argparse
import json
import math
import os
import time
from typing import List
import numpy as np
import toml

import torch
import torchvision
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPTextConfig
from accelerate import init_empty_weights, Accelerator, PartialState
from PIL import Image

from library import stable_cascade as sc

from library.sdxl_model_util import _load_state_dict_on_device
from library.device_utils import clean_memory_on_device
from library.train_util import (
    save_sd_model_on_epoch_end_or_stepwise_common,
    save_sd_model_on_train_end_common,
    line_to_prompt_dict,
    get_hidden_states_stable_cascade,
)
from library import sai_model_spec


from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


CLIP_TEXT_MODEL_NAME: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_sc_te_outputs.npz"


def calculate_latent_sizes(height=1024, width=1024, batch_size=4, compression_factor_b=42.67, compression_factor_a=4.0):
    resolution_multiple = 42.67
    latent_height = math.ceil(height / compression_factor_b)
    latent_width = math.ceil(width / compression_factor_b)
    stage_c_latent_shape = (batch_size, 16, latent_height, latent_width)

    latent_height = math.ceil(height / compression_factor_a)
    latent_width = math.ceil(width / compression_factor_a)
    stage_b_latent_shape = (batch_size, 4, latent_height, latent_width)

    return stage_c_latent_shape, stage_b_latent_shape


# region load and save


def load_effnet(effnet_checkpoint_path, loading_device="cpu") -> sc.EfficientNetEncoder:
    logger.info(f"Loading EfficientNet encoder from {effnet_checkpoint_path}")
    effnet = sc.EfficientNetEncoder()
    effnet_checkpoint = load_file(effnet_checkpoint_path)
    info = effnet.load_state_dict(effnet_checkpoint if "state_dict" not in effnet_checkpoint else effnet_checkpoint["state_dict"])
    logger.info(info)
    del effnet_checkpoint
    return effnet


def load_tokenizer(args: argparse.Namespace):
    # TODO commonize with sdxl_train_util.load_tokenizers
    logger.info("prepare tokenizers")

    original_paths = [CLIP_TEXT_MODEL_NAME]
    tokenizers = []
    for i, original_path in enumerate(original_paths):
        tokenizer: CLIPTokenizer = None
        if args.tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(args.tokenizer_cache_dir, original_path.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)

        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(original_path)

        if args.tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        tokenizers.append(tokenizer)

    if hasattr(args, "max_token_length") and args.max_token_length is not None:
        logger.info(f"update token length: {args.max_token_length}")

    return tokenizers[0]


def load_stage_c_model(stage_c_checkpoint_path, dtype=None, device="cpu") -> sc.StageC:
    # Generator
    logger.info(f"Instantiating Stage C generator")
    with init_empty_weights():
        generator_c = sc.StageC()
    logger.info(f"Loading Stage C generator from {stage_c_checkpoint_path}")
    stage_c_checkpoint = load_file(stage_c_checkpoint_path)

    stage_c_checkpoint = convert_state_dict_mha_to_normal_attn(stage_c_checkpoint)

    logger.info(f"Loading state dict")
    info = _load_state_dict_on_device(generator_c, stage_c_checkpoint, device, dtype=dtype)
    logger.info(info)
    return generator_c


def load_stage_b_model(stage_b_checkpoint_path, dtype=None, device="cpu") -> sc.StageB:
    logger.info(f"Instantiating Stage B generator")
    with init_empty_weights():
        generator_b = sc.StageB()
    logger.info(f"Loading Stage B generator from {stage_b_checkpoint_path}")
    stage_b_checkpoint = load_file(stage_b_checkpoint_path)

    stage_b_checkpoint = convert_state_dict_mha_to_normal_attn(stage_b_checkpoint)

    logger.info(f"Loading state dict")
    info = _load_state_dict_on_device(generator_b, stage_b_checkpoint, device, dtype=dtype)
    logger.info(info)
    return generator_b


def load_clip_text_model(text_model_checkpoint_path, dtype=None, device="cpu", save_text_model=False):
    # CLIP encoders
    logger.info(f"Loading CLIP text model")
    if save_text_model or text_model_checkpoint_path is None:
        logger.info(f"Loading CLIP text model from {CLIP_TEXT_MODEL_NAME}")
        text_model = CLIPTextModelWithProjection.from_pretrained(CLIP_TEXT_MODEL_NAME)

        if save_text_model:
            sd = text_model.state_dict()
            logger.info(f"Saving CLIP text model to {text_model_checkpoint_path}")
            save_file(sd, text_model_checkpoint_path)
    else:
        logger.info(f"Loading CLIP text model from {text_model_checkpoint_path}")

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

        text_model_checkpoint = load_file(text_model_checkpoint_path)
        info = _load_state_dict_on_device(text_model, text_model_checkpoint, device, dtype=dtype)
        logger.info(info)

    return text_model


def load_stage_a_model(stage_a_checkpoint_path, dtype=None, device="cpu") -> sc.StageA:
    logger.info(f"Loading Stage A vqGAN from {stage_a_checkpoint_path}")
    stage_a = sc.StageA().to(device)
    stage_a_checkpoint = load_file(stage_a_checkpoint_path)
    info = stage_a.load_state_dict(
        stage_a_checkpoint if "state_dict" not in stage_a_checkpoint else stage_a_checkpoint["state_dict"]
    )
    logger.info(info)
    return stage_a


def load_previewer_model(previewer_checkpoint_path, dtype=None, device="cpu") -> sc.Previewer:
    logger.info(f"Loading Previewer from {previewer_checkpoint_path}")
    previewer = sc.Previewer().to(device)
    previewer_checkpoint = load_file(previewer_checkpoint_path)
    info = previewer.load_state_dict(
        previewer_checkpoint if "state_dict" not in previewer_checkpoint else previewer_checkpoint["state_dict"]
    )
    logger.info(info)
    return previewer


def convert_state_dict_mha_to_normal_attn(state_dict):
    # convert nn.MultiheadAttention to to_q/k/v and to_out
    print("convert_state_dict_mha_to_normal_attn")
    for key in list(state_dict.keys()):
        if "attention.attn." in key:
            if "in_proj_bias" in key:
                value = state_dict.pop(key)
                qkv = torch.chunk(value, 3, dim=0)
                state_dict[key.replace("in_proj_bias", "to_q.bias")] = qkv[0]
                state_dict[key.replace("in_proj_bias", "to_k.bias")] = qkv[1]
                state_dict[key.replace("in_proj_bias", "to_v.bias")] = qkv[2]
            elif "in_proj_weight" in key:
                value = state_dict.pop(key)
                qkv = torch.chunk(value, 3, dim=0)
                state_dict[key.replace("in_proj_weight", "to_q.weight")] = qkv[0]
                state_dict[key.replace("in_proj_weight", "to_k.weight")] = qkv[1]
                state_dict[key.replace("in_proj_weight", "to_v.weight")] = qkv[2]
            elif "out_proj.bias" in key:
                value = state_dict.pop(key)
                state_dict[key.replace("out_proj.bias", "to_out.bias")] = value
            elif "out_proj.weight" in key:
                value = state_dict.pop(key)
                state_dict[key.replace("out_proj.weight", "to_out.weight")] = value
    return state_dict


def convert_state_dict_normal_attn_to_mha(state_dict):
    # convert to_q/k/v and to_out to nn.MultiheadAttention
    for key in list(state_dict.keys()):
        if "attention.attn." in key:
            if "to_q.bias" in key:
                q = state_dict.pop(key)
                k = state_dict.pop(key.replace("to_q.bias", "to_k.bias"))
                v = state_dict.pop(key.replace("to_q.bias", "to_v.bias"))
                state_dict[key.replace("to_q.bias", "in_proj_bias")] = torch.cat([q, k, v])
            elif "to_q.weight" in key:
                q = state_dict.pop(key)
                k = state_dict.pop(key.replace("to_q.weight", "to_k.weight"))
                v = state_dict.pop(key.replace("to_q.weight", "to_v.weight"))
                state_dict[key.replace("to_q.weight", "in_proj_weight")] = torch.cat([q, k, v])
            elif "to_out.bias" in key:
                v = state_dict.pop(key)
                state_dict[key.replace("to_out.bias", "out_proj.bias")] = v
            elif "to_out.weight" in key:
                v = state_dict.pop(key)
                state_dict[key.replace("to_out.weight", "out_proj.weight")] = v
    return state_dict


def get_sai_model_spec(args, lora=False):
    timestamp = time.time()

    reso = args.resolution

    title = args.metadata_title if args.metadata_title is not None else args.output_name

    if args.min_timestep is not None or args.max_timestep is not None:
        min_time_step = args.min_timestep if args.min_timestep is not None else 0
        max_time_step = args.max_timestep if args.max_timestep is not None else 1000
        timesteps = (min_time_step, max_time_step)
    else:
        timesteps = None

    metadata = sai_model_spec.build_metadata(
        None,
        False,
        False,
        False,
        lora,
        False,
        timestamp,
        title=title,
        reso=reso,
        is_stable_diffusion_ckpt=False,
        author=args.metadata_author,
        description=args.metadata_description,
        license=args.metadata_license,
        tags=args.metadata_tags,
        timesteps=timesteps,
        clip_skip=args.clip_skip,  # None or int
        stable_cascade=True,
    )
    return metadata


def stage_c_saver_common(ckpt_file, stage_c, text_model, save_dtype, sai_metadata):
    state_dict = stage_c.state_dict()
    if save_dtype is not None:
        state_dict = {k: v.to(save_dtype) for k, v in state_dict.items()}

    state_dict = convert_state_dict_normal_attn_to_mha(state_dict)

    save_file(state_dict, ckpt_file, metadata=sai_metadata)

    # save text model
    if text_model is not None:
        text_model_sd = text_model.state_dict()

        if save_dtype is not None:
            text_model_sd = {k: v.to(save_dtype) for k, v in text_model_sd.items()}

        text_model_ckpt_file = os.path.splitext(ckpt_file)[0] + "_text_model.safetensors"
        save_file(text_model_sd, text_model_ckpt_file)


def save_stage_c_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    stage_c,
    text_model,
):
    def stage_c_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = get_sai_model_spec(args)
        stage_c_saver_common(ckpt_file, stage_c, text_model, save_dtype, sai_metadata)

    save_sd_model_on_epoch_end_or_stepwise_common(
        args, on_epoch_end, accelerator, True, True, epoch, num_train_epochs, global_step, stage_c_saver, None
    )


def save_stage_c_model_on_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    stage_c,
    text_model,
):
    def stage_c_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = get_sai_model_spec(args)
        stage_c_saver_common(ckpt_file, stage_c, text_model, save_dtype, sai_metadata)

    save_sd_model_on_train_end_common(args, True, True, epoch, global_step, stage_c_saver, None)


# endregion

# region sample generation


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    previewer,
    tokenizer,
    text_encoder,
    stage_c,
    gdf,
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
    if not os.path.isfile(args.sample_prompts):
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    # unwrap unet and text_encoder(s)
    stage_c = accelerator.unwrap_model(stage_c)
    text_encoder = accelerator.unwrap_model(text_encoder)

    # read prompts
    if args.sample_prompts.endswith(".txt"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif args.sample_prompts.endswith(".toml"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif args.sample_prompts.endswith(".json"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    # save random state to restore later
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    if distributed_state.num_processes <= 1:
        # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
        with torch.no_grad():
            for prompt_dict in prompts:
                sample_image_inference(
                    accelerator,
                    args,
                    tokenizer,
                    text_encoder,
                    stage_c,
                    previewer,
                    gdf,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
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
                        tokenizer,
                        text_encoder,
                        stage_c,
                        previewer,
                        gdf,
                        save_dir,
                        prompt_dict,
                        epoch,
                        steps,
                        prompt_replacement,
                    )

    # I'm not sure which of these is the correct way to clear the memory, but accelerator's device is used in the pipeline, so I'm using it here.
    # with torch.cuda.device(torch.cuda.current_device()):
    #     torch.cuda.empty_cache()
    clean_memory_on_device(accelerator.device)

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    tokenizer,
    text_model,
    stage_c,
    previewer,
    gdf,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    prompt_replacement,
):
    assert isinstance(prompt_dict, dict)
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 20)
    width = prompt_dict.get("width", 1024)
    height = prompt_dict.get("height", 1024)
    scale = prompt_dict.get("scale", 4)
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

    negative_prompt = "" if negative_prompt is None else negative_prompt
    cfg = scale
    timesteps = sample_steps
    shift = 2
    t_start = 1.0

    stage_c_latent_shape, _ = calculate_latent_sizes(height, width, batch_size=1)

    # PREPARE CONDITIONS
    input_ids = tokenizer(
        [prompt], truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )["input_ids"].to(text_model.device)
    cond_text, cond_pooled = get_hidden_states_stable_cascade(tokenizer.model_max_length, input_ids, tokenizer, text_model)

    input_ids = tokenizer(
        [negative_prompt], truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )["input_ids"].to(text_model.device)
    uncond_text, uncond_pooled = get_hidden_states_stable_cascade(tokenizer.model_max_length, input_ids, tokenizer, text_model)

    device = accelerator.device
    dtype = stage_c.dtype
    cond_text = cond_text.to(device, dtype=dtype)
    cond_pooled = cond_pooled.unsqueeze(1).to(device, dtype=dtype)

    uncond_text = uncond_text.to(device, dtype=dtype)
    uncond_pooled = uncond_pooled.unsqueeze(1).to(device, dtype=dtype)

    zero_img_emb = torch.zeros(1, 768, device=device)

    # 辞書にしたくないけど GDF から先の変更が面倒だからとりあえず辞書にしておく
    conditions = {"clip_text_pooled": cond_pooled, "clip": cond_pooled, "clip_text": cond_text, "clip_img": zero_img_emb}
    unconditions = {"clip_text_pooled": uncond_pooled, "clip": uncond_pooled, "clip_text": uncond_text, "clip_img": zero_img_emb}

    with torch.no_grad():  # , torch.cuda.amp.autocast(dtype=dtype):
        sampling_c = gdf.sample(
            stage_c,
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

    sampled_c = sampled_c.to(previewer.device, dtype=previewer.dtype)
    image = previewer(sampled_c)[0]
    image = torch.clamp(image, 0, 1)
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = image * 255
    image = image.astype(np.uint8)
    image = Image.fromarray(image)

    # adding accelerator.wait_for_everyone() here should sync up and ensure that sample images are saved in the same order as the original prompt list
    # but adding 'enum' to the filename should be enough

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict["enum"]
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    # wandb有効時のみログを送信
    try:
        wandb_tracker = accelerator.get_tracker("wandb")
        try:
            import wandb
        except ImportError:  # 事前に一度確認するのでここはエラー出ないはず
            raise ImportError("No wandb / wandb がインストールされていないようです")

        wandb_tracker.log({f"sample_{i}": wandb.Image(image)})
    except:  # wandb 無効時
        pass


# endregion


def add_effnet_arguments(parser):
    parser.add_argument(
        "--effnet_checkpoint_path",
        type=str,
        required=True,
        help="path to EfficientNet checkpoint / EfficientNetのチェックポイントのパス",
    )
    return parser


def add_text_model_arguments(parser):
    parser.add_argument(
        "--text_model_checkpoint_path",
        type=str,
        help="path to CLIP text model checkpoint / CLIPテキストモデルのチェックポイントのパス",
    )
    parser.add_argument("--save_text_model", action="store_true", help="if specified, save text model to corresponding path")
    return parser


def add_stage_a_arguments(parser):
    parser.add_argument(
        "--stage_a_checkpoint_path",
        type=str,
        required=True,
        help="path to Stage A checkpoint / Stage Aのチェックポイントのパス",
    )
    return parser


def add_stage_b_arguments(parser):
    parser.add_argument(
        "--stage_b_checkpoint_path",
        type=str,
        required=True,
        help="path to Stage B checkpoint / Stage Bのチェックポイントのパス",
    )
    return parser


def add_stage_c_arguments(parser):
    parser.add_argument(
        "--stage_c_checkpoint_path",
        type=str,
        required=True,
        help="path to Stage C checkpoint / Stage Cのチェックポイントのパス",
    )
    return parser


def add_previewer_arguments(parser):
    parser.add_argument(
        "--previewer_checkpoint_path",
        type=str,
        required=False,
        help="path to previewer checkpoint / previewerのチェックポイントのパス",
    )
    return parser


def add_training_arguments(parser):
    parser.add_argument(
        "--adaptive_loss_weight",
        action="store_true",
        help="if specified, use adaptive loss weight. if not, use P2 loss weight"
        + " / Adaptive Loss Weightを使用する。指定しない場合はP2 Loss Weightを使用する",
    )
