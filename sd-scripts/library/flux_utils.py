import json
import os
from dataclasses import replace
from typing import List, Optional, Tuple, Union

import einops
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPConfig, CLIPTextModel, T5Config, T5EncoderModel

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import flux_models
from library.safetensors_utils import load_safetensors

MODEL_VERSION_FLUX_V1 = "flux1"
MODEL_NAME_DEV = "dev"
MODEL_NAME_SCHNELL = "schnell"
MODEL_VERSION_CHROMA = "chroma"


def analyze_checkpoint_state(ckpt_path: str) -> Tuple[bool, bool, Tuple[int, int], List[str]]:
    """
    チェックポイントの状態を分析し、DiffusersかBFLか、devかschnellか、ブロック数を計算して返す。

    Args:
        ckpt_path (str): チェックポイントファイルまたはディレクトリのパス。

    Returns:
        Tuple[bool, bool, Tuple[int, int], List[str]]:
            - bool: Diffusersかどうかを示すフラグ。
            - bool: Schnellかどうかを示すフラグ。
            - Tuple[int, int]: ダブルブロックとシングルブロックの数。
            - List[str]: チェックポイントに含まれるキーのリスト。
    """
    # check the state dict: Diffusers or BFL, dev or schnell, number of blocks
    logger.info(f"Checking the state dict: Diffusers or BFL, dev or schnell")

    if os.path.isdir(ckpt_path):  # if ckpt_path is a directory, it is Diffusers
        ckpt_path = os.path.join(ckpt_path, "transformer", "diffusion_pytorch_model-00001-of-00003.safetensors")
    if "00001-of-00003" in ckpt_path:
        ckpt_paths = [ckpt_path.replace("00001-of-00003", f"0000{i}-of-00003") for i in range(1, 4)]
    else:
        ckpt_paths = [ckpt_path]

    keys = []
    for ckpt_path in ckpt_paths:
        with safe_open(ckpt_path, framework="pt") as f:
            keys.extend(f.keys())

    # if the key has annoying prefix, remove it
    if keys[0].startswith("model.diffusion_model."):
        keys = [key.replace("model.diffusion_model.", "") for key in keys]

    is_diffusers = "transformer_blocks.0.attn.add_k_proj.bias" in keys
    is_schnell = not ("guidance_in.in_layer.bias" in keys or "time_text_embed.guidance_embedder.linear_1.bias" in keys)

    # check number of double and single blocks
    if not is_diffusers:
        max_double_block_index = max(
            [int(key.split(".")[1]) for key in keys if key.startswith("double_blocks.") and key.endswith(".img_attn.proj.bias")]
        )
        max_single_block_index = max(
            [int(key.split(".")[1]) for key in keys if key.startswith("single_blocks.") and key.endswith(".modulation.lin.bias")]
        )
    else:
        max_double_block_index = max(
            [
                int(key.split(".")[1])
                for key in keys
                if key.startswith("transformer_blocks.") and key.endswith(".attn.add_k_proj.bias")
            ]
        )
        max_single_block_index = max(
            [
                int(key.split(".")[1])
                for key in keys
                if key.startswith("single_transformer_blocks.") and key.endswith(".attn.to_k.bias")
            ]
        )

    num_double_blocks = max_double_block_index + 1
    num_single_blocks = max_single_block_index + 1

    return is_diffusers, is_schnell, (num_double_blocks, num_single_blocks), ckpt_paths


def load_flow_model(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    model_type: str = "flux",
) -> Tuple[bool, flux_models.Flux]:
    if model_type == "flux":
        is_diffusers, is_schnell, (num_double_blocks, num_single_blocks), ckpt_paths = analyze_checkpoint_state(ckpt_path)
        name = MODEL_NAME_DEV if not is_schnell else MODEL_NAME_SCHNELL

        # build model
        logger.info(f"Building Flux model {name} from {'Diffusers' if is_diffusers else 'BFL'} checkpoint")
        with torch.device("meta"):
            params = flux_models.configs[name].params

            # set the number of blocks
            if params.depth != num_double_blocks:
                logger.info(f"Setting the number of double blocks from {params.depth} to {num_double_blocks}")
                params = replace(params, depth=num_double_blocks)
            if params.depth_single_blocks != num_single_blocks:
                logger.info(f"Setting the number of single blocks from {params.depth_single_blocks} to {num_single_blocks}")
                params = replace(params, depth_single_blocks=num_single_blocks)

            model = flux_models.Flux(params)
            if dtype is not None:
                model = model.to(dtype)

        # load_sft doesn't support torch.device
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = {}
        for ckpt_path in ckpt_paths:
            sd.update(load_safetensors(ckpt_path, device=device, disable_mmap=disable_mmap, dtype=dtype))

        # convert Diffusers to BFL
        if is_diffusers:
            logger.info("Converting Diffusers to BFL")
            sd = convert_diffusers_sd_to_bfl(sd, num_double_blocks, num_single_blocks)
            logger.info("Converted Diffusers to BFL")

        # if the key has annoying prefix, remove it
        for key in list(sd.keys()):
            new_key = key.replace("model.diffusion_model.", "")
            if new_key == key:
                break  # the model doesn't have annoying prefix
            sd[new_key] = sd.pop(key)

        info = model.load_state_dict(sd, strict=False, assign=True)
        logger.info(f"Loaded Flux: {info}")
        return is_schnell, model

    elif model_type == "chroma":
        from . import chroma_models

        # build model
        logger.info("Building Chroma model")
        with torch.device("meta"):
            model = chroma_models.Chroma(chroma_models.chroma_params)
            if dtype is not None:
                model = model.to(dtype)

        # load_sft doesn't support torch.device
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

        # if the key has annoying prefix, remove it
        for key in list(sd.keys()):
            new_key = key.replace("model.diffusion_model.", "")
            if new_key == key:
                break  # the model doesn't have annoying prefix
            sd[new_key] = sd.pop(key)

        info = model.load_state_dict(sd, strict=False, assign=True)
        logger.info(f"Loaded Chroma: {info}")
        is_schnell = False  # Chroma is not schnell
        return is_schnell, model

    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Supported types are 'flux' and 'chroma'.")


def load_ae(
    ckpt_path: str, dtype: torch.dtype, device: Union[str, torch.device], disable_mmap: bool = False
) -> flux_models.AutoEncoder:
    logger.info("Building AutoEncoder")
    with torch.device("meta"):
        # dev and schnell have the same AE params
        ae = flux_models.AutoEncoder(flux_models.configs[MODEL_NAME_DEV].ae_params).to(dtype)

    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = ae.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded AE: {info}")
    return ae


def load_controlnet(
    ckpt_path: Optional[str], is_schnell: bool, dtype: torch.dtype, device: Union[str, torch.device], disable_mmap: bool = False
):
    logger.info("Building ControlNet")
    name = MODEL_NAME_DEV if not is_schnell else MODEL_NAME_SCHNELL
    with torch.device(device):
        controlnet = flux_models.ControlNetFlux(flux_models.configs[name].params).to(dtype)

    if ckpt_path is not None:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
        info = controlnet.load_state_dict(sd, strict=False, assign=True)
        logger.info(f"Loaded ControlNet: {info}")
    return controlnet


def dummy_clip_l() -> torch.nn.Module:
    """
    Returns a dummy CLIP-L model with the output shape of (N, 77, 768).
    """
    return DummyCLIPL()


class DummyTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Parameter(torch.zeros(1))


class DummyCLIPL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_shape = (77, 1)  # Note: The original code had (77, 768), but we use (77, 1) for the dummy output

        # dtype and device from these parameters. train_network.py accesses them
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self.dummy_param_2 = torch.nn.Parameter(torch.zeros(1))
        self.dummy_param_3 = torch.nn.Parameter(torch.zeros(1))
        self.text_model = DummyTextModel()

    @property
    def device(self):
        return self.dummy_param.device

    @property
    def dtype(self):
        return self.dummy_param.dtype

    def forward(self, *args, **kwargs):
        """
        Returns a dummy output with the shape of (N, 77, 768).
        """
        batch_size = args[0].shape[0] if args else 1
        return {"pooler_output": torch.zeros(batch_size, *self.output_shape, device=self.device, dtype=self.dtype)}


def load_clip_l(
    ckpt_path: Optional[str],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> CLIPTextModel:
    logger.info("Building CLIP-L")
    CLIPL_CONFIG = {
        "_name_or_path": "clip-vit-large-patch14/",
        "architectures": ["CLIPModel"],
        "initializer_factor": 1.0,
        "logit_scale_init_value": 2.6592,
        "model_type": "clip",
        "projection_dim": 768,
        # "text_config": {
        "_name_or_path": "",
        "add_cross_attention": False,
        "architectures": None,
        "attention_dropout": 0.0,
        "bad_words_ids": None,
        "bos_token_id": 0,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "dropout": 0.0,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": 2,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
        "layer_norm_eps": 1e-05,
        "length_penalty": 1.0,
        "max_length": 20,
        "max_position_embeddings": 77,
        "min_length": 0,
        "model_type": "clip_text_model",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 12,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_hidden_layers": 12,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": 1,
        "prefix": None,
        "problem_type": None,
        "projection_dim": 768,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "sep_token_id": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": None,
        "torchscript": False,
        "transformers_version": "4.16.0.dev0",
        "use_bfloat16": False,
        "vocab_size": 49408,
        "hidden_act": "gelu",
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_attention_heads": 20,
        "num_hidden_layers": 32,
        # },
        # "text_config_dict": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "projection_dim": 768,
        # },
        # "torch_dtype": "float32",
        # "transformers_version": None,
    }
    config = CLIPConfig(**CLIPL_CONFIG)
    with init_empty_weights():
        clip = CLIPTextModel._from_config(config)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = clip.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded CLIP-L: {info}")
    return clip


def load_t5xxl(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> T5EncoderModel:
    T5_CONFIG_JSON = """
{
  "architectures": [
    "T5EncoderModel"
  ],
  "classifier_dropout": 0.0,
  "d_ff": 10240,
  "d_kv": 64,
  "d_model": 4096,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 24,
  "num_heads": 64,
  "num_layers": 24,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.41.2",
  "use_cache": true,
  "vocab_size": 32128
}
"""
    config = json.loads(T5_CONFIG_JSON)
    config = T5Config(**config)
    with init_empty_weights():
        t5xxl = T5EncoderModel._from_config(config)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = t5xxl.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded T5xxl: {info}")
    return t5xxl


def get_t5xxl_actual_dtype(t5xxl: T5EncoderModel) -> torch.dtype:
    # nn.Embedding is the first layer, but it could be casted to bfloat16 or float32
    return t5xxl.encoder.block[0].layer[0].SelfAttention.q.weight.dtype


def prepare_img_ids(batch_size: int, packed_latent_height: int, packed_latent_width: int):
    img_ids = torch.zeros(packed_latent_height, packed_latent_width, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_latent_height)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_latent_width)[None, :]
    img_ids = einops.repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    return img_ids


def unpack_latents(x: torch.Tensor, packed_latent_height: int, packed_latent_width: int) -> torch.Tensor:
    """
    x: [b (h w) (c ph pw)] -> [b c (h ph) (w pw)], ph=2, pw=2
    """
    x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)
    return x


def pack_latents(x: torch.Tensor) -> torch.Tensor:
    """
    x: [b c (h ph) (w pw)] -> [b (h w) (c ph pw)], ph=2, pw=2
    """
    x = einops.rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return x


# region Diffusers

NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38

BFL_TO_DIFFUSERS_MAP = {
    "time_in.in_layer.weight": ["time_text_embed.timestep_embedder.linear_1.weight"],
    "time_in.in_layer.bias": ["time_text_embed.timestep_embedder.linear_1.bias"],
    "time_in.out_layer.weight": ["time_text_embed.timestep_embedder.linear_2.weight"],
    "time_in.out_layer.bias": ["time_text_embed.timestep_embedder.linear_2.bias"],
    "vector_in.in_layer.weight": ["time_text_embed.text_embedder.linear_1.weight"],
    "vector_in.in_layer.bias": ["time_text_embed.text_embedder.linear_1.bias"],
    "vector_in.out_layer.weight": ["time_text_embed.text_embedder.linear_2.weight"],
    "vector_in.out_layer.bias": ["time_text_embed.text_embedder.linear_2.bias"],
    "guidance_in.in_layer.weight": ["time_text_embed.guidance_embedder.linear_1.weight"],
    "guidance_in.in_layer.bias": ["time_text_embed.guidance_embedder.linear_1.bias"],
    "guidance_in.out_layer.weight": ["time_text_embed.guidance_embedder.linear_2.weight"],
    "guidance_in.out_layer.bias": ["time_text_embed.guidance_embedder.linear_2.bias"],
    "txt_in.weight": ["context_embedder.weight"],
    "txt_in.bias": ["context_embedder.bias"],
    "img_in.weight": ["x_embedder.weight"],
    "img_in.bias": ["x_embedder.bias"],
    "double_blocks.().img_mod.lin.weight": ["norm1.linear.weight"],
    "double_blocks.().img_mod.lin.bias": ["norm1.linear.bias"],
    "double_blocks.().txt_mod.lin.weight": ["norm1_context.linear.weight"],
    "double_blocks.().txt_mod.lin.bias": ["norm1_context.linear.bias"],
    "double_blocks.().img_attn.qkv.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
    "double_blocks.().img_attn.qkv.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"],
    "double_blocks.().txt_attn.qkv.weight": ["attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"],
    "double_blocks.().txt_attn.qkv.bias": ["attn.add_q_proj.bias", "attn.add_k_proj.bias", "attn.add_v_proj.bias"],
    "double_blocks.().img_attn.norm.query_norm.scale": ["attn.norm_q.weight"],
    "double_blocks.().img_attn.norm.key_norm.scale": ["attn.norm_k.weight"],
    "double_blocks.().txt_attn.norm.query_norm.scale": ["attn.norm_added_q.weight"],
    "double_blocks.().txt_attn.norm.key_norm.scale": ["attn.norm_added_k.weight"],
    "double_blocks.().img_mlp.0.weight": ["ff.net.0.proj.weight"],
    "double_blocks.().img_mlp.0.bias": ["ff.net.0.proj.bias"],
    "double_blocks.().img_mlp.2.weight": ["ff.net.2.weight"],
    "double_blocks.().img_mlp.2.bias": ["ff.net.2.bias"],
    "double_blocks.().txt_mlp.0.weight": ["ff_context.net.0.proj.weight"],
    "double_blocks.().txt_mlp.0.bias": ["ff_context.net.0.proj.bias"],
    "double_blocks.().txt_mlp.2.weight": ["ff_context.net.2.weight"],
    "double_blocks.().txt_mlp.2.bias": ["ff_context.net.2.bias"],
    "double_blocks.().img_attn.proj.weight": ["attn.to_out.0.weight"],
    "double_blocks.().img_attn.proj.bias": ["attn.to_out.0.bias"],
    "double_blocks.().txt_attn.proj.weight": ["attn.to_add_out.weight"],
    "double_blocks.().txt_attn.proj.bias": ["attn.to_add_out.bias"],
    "single_blocks.().modulation.lin.weight": ["norm.linear.weight"],
    "single_blocks.().modulation.lin.bias": ["norm.linear.bias"],
    "single_blocks.().linear1.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "proj_mlp.weight"],
    "single_blocks.().linear1.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias", "proj_mlp.bias"],
    "single_blocks.().linear2.weight": ["proj_out.weight"],
    "single_blocks.().norm.query_norm.scale": ["attn.norm_q.weight"],
    "single_blocks.().norm.key_norm.scale": ["attn.norm_k.weight"],
    "single_blocks.().linear2.weight": ["proj_out.weight"],
    "single_blocks.().linear2.bias": ["proj_out.bias"],
    "final_layer.linear.weight": ["proj_out.weight"],
    "final_layer.linear.bias": ["proj_out.bias"],
    "final_layer.adaLN_modulation.1.weight": ["norm_out.linear.weight"],
    "final_layer.adaLN_modulation.1.bias": ["norm_out.linear.bias"],
}


def make_diffusers_to_bfl_map(num_double_blocks: int, num_single_blocks: int) -> dict[str, tuple[int, str]]:
    # make reverse map from diffusers map
    diffusers_to_bfl_map = {}  # key: diffusers_key, value: (index, bfl_key)
    for b in range(num_double_blocks):
        for key, weights in BFL_TO_DIFFUSERS_MAP.items():
            if key.startswith("double_blocks."):
                block_prefix = f"transformer_blocks.{b}."
                for i, weight in enumerate(weights):
                    diffusers_to_bfl_map[f"{block_prefix}{weight}"] = (i, key.replace("()", f"{b}"))
    for b in range(num_single_blocks):
        for key, weights in BFL_TO_DIFFUSERS_MAP.items():
            if key.startswith("single_blocks."):
                block_prefix = f"single_transformer_blocks.{b}."
                for i, weight in enumerate(weights):
                    diffusers_to_bfl_map[f"{block_prefix}{weight}"] = (i, key.replace("()", f"{b}"))
    for key, weights in BFL_TO_DIFFUSERS_MAP.items():
        if not (key.startswith("double_blocks.") or key.startswith("single_blocks.")):
            for i, weight in enumerate(weights):
                diffusers_to_bfl_map[weight] = (i, key)
    return diffusers_to_bfl_map


def convert_diffusers_sd_to_bfl(
    diffusers_sd: dict[str, torch.Tensor], num_double_blocks: int = NUM_DOUBLE_BLOCKS, num_single_blocks: int = NUM_SINGLE_BLOCKS
) -> dict[str, torch.Tensor]:
    diffusers_to_bfl_map = make_diffusers_to_bfl_map(num_double_blocks, num_single_blocks)

    # iterate over three safetensors files to reduce memory usage
    flux_sd = {}
    for diffusers_key, tensor in diffusers_sd.items():
        if diffusers_key in diffusers_to_bfl_map:
            index, bfl_key = diffusers_to_bfl_map[diffusers_key]
            if bfl_key not in flux_sd:
                flux_sd[bfl_key] = []
            flux_sd[bfl_key].append((index, tensor))
        else:
            logger.error(f"Error: Key not found in diffusers_to_bfl_map: {diffusers_key}")
            raise KeyError(f"Key not found in diffusers_to_bfl_map: {diffusers_key}")

    # concat tensors if multiple tensors are mapped to a single key, sort by index
    for key, values in flux_sd.items():
        if len(values) == 1:
            flux_sd[key] = values[0][1]
        else:
            flux_sd[key] = torch.cat([value[1] for value in sorted(values, key=lambda x: x[0])])

    # special case for final_layer.adaLN_modulation.1.weight and final_layer.adaLN_modulation.1.bias
    def swap_scale_shift(weight):
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    if "final_layer.adaLN_modulation.1.weight" in flux_sd:
        flux_sd["final_layer.adaLN_modulation.1.weight"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.weight"])
    if "final_layer.adaLN_modulation.1.bias" in flux_sd:
        flux_sd["final_layer.adaLN_modulation.1.bias"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.bias"])

    return flux_sd


# endregion
