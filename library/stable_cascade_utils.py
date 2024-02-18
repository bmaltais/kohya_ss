import argparse
import os
import time
from typing import List
import numpy as np

import torch
import torchvision
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPTextConfig
from accelerate import init_empty_weights

from library import stable_cascade as sc
from library.train_util import (
    ImageInfo,
    load_image,
    trim_and_resize_if_required,
    save_latents_to_disk,
    HIGH_VRAM,
    save_text_encoder_outputs_to_disk,
)
from library.sdxl_model_util import _load_state_dict_on_device
from library.device_utils import clean_memory_on_device
from library.train_util import save_sd_model_on_epoch_end_or_stepwise_common, save_sd_model_on_train_end_common
from library import sai_model_spec


from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


CLIP_TEXT_MODEL_NAME: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

EFFNET_PREPROCESS = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
)

TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_sc_te_outputs.npz"
LATENTS_CACHE_SUFFIX = "_sc_latents.npz"


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


def is_disk_cached_latents_is_expected(reso, npz_path: str, flip_aug: bool):
    expected_latents_size = (reso[1] // 32, reso[0] // 32)  # bucket_resoはWxHなので注意

    if not os.path.exists(npz_path):
        return False

    npz = np.load(npz_path)
    if "latents" not in npz or "original_size" not in npz or "crop_ltrb" not in npz:  # old ver?
        return False
    if npz["latents"].shape[1:3] != expected_latents_size:
        return False

    if flip_aug:
        if "latents_flipped" not in npz:
            return False
        if npz["latents_flipped"].shape[1:3] != expected_latents_size:
            return False

    return True


def cache_batch_latents(
    effnet: sc.EfficientNetEncoder,
    cache_to_disk: bool,
    image_infos: List[ImageInfo],
    flip_aug: bool,
    random_crop: bool,
    device,
    dtype,
) -> None:
    r"""
    requires image_infos to have: absolute_path, bucket_reso, resized_size, latents_npz
    optionally requires image_infos to have: image
    if cache_to_disk is True, set info.latents_npz
        flipped latents is also saved if flip_aug is True
    if cache_to_disk is False, set info.latents
        latents_flipped is also set if flip_aug is True
    latents_original_size and latents_crop_ltrb are also set
    """
    images = []
    for info in image_infos:
        image = load_image(info.absolute_path) if info.image is None else np.array(info.image, np.uint8)
        # TODO 画像のメタデータが壊れていて、メタデータから割り当てたbucketと実際の画像サイズが一致しない場合があるのでチェック追加要
        image, original_size, crop_ltrb = trim_and_resize_if_required(random_crop, image, info.bucket_reso, info.resized_size)
        image = EFFNET_PREPROCESS(image)
        images.append(image)

        info.latents_original_size = original_size
        info.latents_crop_ltrb = crop_ltrb

    img_tensors = torch.stack(images, dim=0)
    img_tensors = img_tensors.to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = effnet(img_tensors).to("cpu")
        print(latents.shape)

    if flip_aug:
        img_tensors = torch.flip(img_tensors, dims=[3])
        with torch.no_grad():
            flipped_latents = effnet(img_tensors).to("cpu")
    else:
        flipped_latents = [None] * len(latents)

    for info, latent, flipped_latent in zip(image_infos, latents, flipped_latents):
        # check NaN
        if torch.isnan(latents).any() or (flipped_latent is not None and torch.isnan(flipped_latent).any()):
            raise RuntimeError(f"NaN detected in latents: {info.absolute_path}")

        if cache_to_disk:
            save_latents_to_disk(info.latents_npz, latent, info.latents_original_size, info.latents_crop_ltrb, flipped_latent)
        else:
            info.latents = latent
            if flip_aug:
                info.latents_flipped = flipped_latent

    if not HIGH_VRAM:
        clean_memory_on_device(device)


def cache_batch_text_encoder_outputs(image_infos, tokenizers, text_encoders, max_token_length, cache_to_disk, input_ids, dtype):
    # 75 トークン越えは未対応
    input_ids = input_ids.to(text_encoders[0].device)

    with torch.no_grad():
        b_hidden_state, b_pool = sc.get_clip_conditions(None, input_ids, tokenizers[0], text_encoders[0])

        b_hidden_state = b_hidden_state.detach().to("cpu")  # b,n*75+2,768
        b_pool = b_pool.detach().to("cpu")  # b,1280

    for info, hidden_state, pool in zip(image_infos, b_hidden_state, b_pool):
        if cache_to_disk:
            save_text_encoder_outputs_to_disk(info.text_encoder_outputs_npz, None, hidden_state, pool)
        else:
            info.text_encoder_outputs1 = hidden_state
            info.text_encoder_pool2 = pool


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
        required=True,
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


def get_sai_model_spec(args):
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
        False,
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
        state_dict = {k: v.to(save_dtype) for k, v in state_dict.items}

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


def cache_latents(self, effnet, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
    # マルチGPUには対応していないので、そちらはtools/cache_latents.pyを使うこと
    logger.info("caching latents.")

    image_infos = list(self.image_data.values())

    # sort by resolution
    image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

    # split by resolution
    batches = []
    batch = []
    logger.info("checking cache validity...")
    for info in tqdm(image_infos):
        subset = self.image_to_subset[info.image_key]

        if info.latents_npz is not None:  # fine tuning dataset
            continue

        # check disk cache exists and size of latents
        if cache_to_disk:
            info.latents_npz = os.path.splitext(info.absolute_path)[0] + LATENTS_CACHE_SUFFIX
            if not is_main_process:  # store to info only
                continue

            cache_available = is_disk_cached_latents_is_expected(info.bucket_reso, info.latents_npz, subset.flip_aug)

            if cache_available:  # do not add to batch
                continue

        # if last member of batch has different resolution, flush the batch
        if len(batch) > 0 and batch[-1].bucket_reso != info.bucket_reso:
            batches.append(batch)
            batch = []

        batch.append(info)

        # if number of data in batch is enough, flush the batch
        if len(batch) >= vae_batch_size:
            batches.append(batch)
            batch = []

    if len(batch) > 0:
        batches.append(batch)

    if cache_to_disk and not is_main_process:  # if cache to disk, don't cache latents in non-main process, set to info only
        return

    # iterate batches: batch doesn't have image, image will be loaded in cache_batch_latents and discarded
    logger.info("caching latents...")
    for batch in tqdm(batches, smoothing=1, total=len(batches)):
        cache_batch_latents(effnet, cache_to_disk, batch, subset.flip_aug, subset.random_crop)


# weight_dtypeを指定するとText Encoderそのもの、およひ出力がweight_dtypeになる
def cache_text_encoder_outputs(self, tokenizers, text_encoders, device, weight_dtype, cache_to_disk=False, is_main_process=True):
    # latentsのキャッシュと同様に、ディスクへのキャッシュに対応する
    # またマルチGPUには対応していないので、そちらはtools/cache_latents.pyを使うこと
    logger.info("caching text encoder outputs.")
    image_infos = list(self.image_data.values())

    logger.info("checking cache existence...")
    image_infos_to_cache = []
    for info in tqdm(image_infos):
        # subset = self.image_to_subset[info.image_key]
        if cache_to_disk:
            te_out_npz = os.path.splitext(info.absolute_path)[0] + TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX
            info.text_encoder_outputs_npz = te_out_npz

            if not is_main_process:  # store to info only
                continue

            if os.path.exists(te_out_npz):
                continue

        image_infos_to_cache.append(info)

    if cache_to_disk and not is_main_process:  # if cache to disk, don't cache latents in non-main process, set to info only
        return

    # prepare tokenizers and text encoders
    for text_encoder in text_encoders:
        text_encoder.to(device)
        if weight_dtype is not None:
            text_encoder.to(dtype=weight_dtype)

    # create batch
    batch = []
    batches = []
    for info in image_infos_to_cache:
        input_ids1 = self.get_input_ids(info.caption, tokenizers[0])
        batch.append((info, input_ids1, None))

        if len(batch) >= self.batch_size:
            batches.append(batch)
            batch = []

    if len(batch) > 0:
        batches.append(batch)

    # iterate batches: call text encoder and cache outputs for memory or disk
    logger.info("caching text encoder outputs...")
    for batch in tqdm(batches):
        infos, input_ids1, input_ids2 = zip(*batch)
        input_ids1 = torch.stack(input_ids1, dim=0)
        input_ids2 = torch.stack(input_ids2, dim=0) if input_ids2[0] is not None else None
        cache_batch_text_encoder_outputs(
            infos, tokenizers, text_encoders, self.max_token_length, cache_to_disk, input_ids1, weight_dtype
        )
