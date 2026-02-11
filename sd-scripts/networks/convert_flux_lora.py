# convert key mapping and data format from some LoRA format to another
"""
Original LoRA format: Based on Black Forest Labs, QKV and MLP are unified into one module
alpha is scalar for each LoRA module

0 to 18
lora_unet_double_blocks_0_img_attn_proj.alpha torch.Size([])
lora_unet_double_blocks_0_img_attn_proj.lora_down.weight torch.Size([4, 3072])
lora_unet_double_blocks_0_img_attn_proj.lora_up.weight torch.Size([3072, 4])
lora_unet_double_blocks_0_img_attn_qkv.alpha torch.Size([])
lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight torch.Size([4, 3072])
lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight torch.Size([9216, 4])
lora_unet_double_blocks_0_img_mlp_0.alpha torch.Size([])
lora_unet_double_blocks_0_img_mlp_0.lora_down.weight torch.Size([4, 3072])
lora_unet_double_blocks_0_img_mlp_0.lora_up.weight torch.Size([12288, 4])
lora_unet_double_blocks_0_img_mlp_2.alpha torch.Size([])
lora_unet_double_blocks_0_img_mlp_2.lora_down.weight torch.Size([4, 12288])
lora_unet_double_blocks_0_img_mlp_2.lora_up.weight torch.Size([3072, 4])
lora_unet_double_blocks_0_img_mod_lin.alpha torch.Size([])
lora_unet_double_blocks_0_img_mod_lin.lora_down.weight torch.Size([4, 3072])
lora_unet_double_blocks_0_img_mod_lin.lora_up.weight torch.Size([18432, 4])
lora_unet_double_blocks_0_txt_attn_proj.alpha torch.Size([])
lora_unet_double_blocks_0_txt_attn_proj.lora_down.weight torch.Size([4, 3072])
lora_unet_double_blocks_0_txt_attn_proj.lora_up.weight torch.Size([3072, 4])
lora_unet_double_blocks_0_txt_attn_qkv.alpha torch.Size([])
lora_unet_double_blocks_0_txt_attn_qkv.lora_down.weight torch.Size([4, 3072])
lora_unet_double_blocks_0_txt_attn_qkv.lora_up.weight torch.Size([9216, 4])
lora_unet_double_blocks_0_txt_mlp_0.alpha torch.Size([])
lora_unet_double_blocks_0_txt_mlp_0.lora_down.weight torch.Size([4, 3072])
lora_unet_double_blocks_0_txt_mlp_0.lora_up.weight torch.Size([12288, 4])
lora_unet_double_blocks_0_txt_mlp_2.alpha torch.Size([])
lora_unet_double_blocks_0_txt_mlp_2.lora_down.weight torch.Size([4, 12288])
lora_unet_double_blocks_0_txt_mlp_2.lora_up.weight torch.Size([3072, 4])
lora_unet_double_blocks_0_txt_mod_lin.alpha torch.Size([])
lora_unet_double_blocks_0_txt_mod_lin.lora_down.weight torch.Size([4, 3072])
lora_unet_double_blocks_0_txt_mod_lin.lora_up.weight torch.Size([18432, 4])

0 to 37
lora_unet_single_blocks_0_linear1.alpha torch.Size([])
lora_unet_single_blocks_0_linear1.lora_down.weight torch.Size([4, 3072])
lora_unet_single_blocks_0_linear1.lora_up.weight torch.Size([21504, 4])
lora_unet_single_blocks_0_linear2.alpha torch.Size([])
lora_unet_single_blocks_0_linear2.lora_down.weight torch.Size([4, 15360])
lora_unet_single_blocks_0_linear2.lora_up.weight torch.Size([3072, 4])
lora_unet_single_blocks_0_modulation_lin.alpha torch.Size([])
lora_unet_single_blocks_0_modulation_lin.lora_down.weight torch.Size([4, 3072])
lora_unet_single_blocks_0_modulation_lin.lora_up.weight torch.Size([9216, 4])
"""
"""
ai-toolkit: Based on Diffusers, QKV and MLP are separated into 3 modules.
A is down, B is up. No alpha for each LoRA module.

0 to 18
transformer.transformer_blocks.0.attn.add_k_proj.lora_A.weight torch.Size([16, 3072])
transformer.transformer_blocks.0.attn.add_k_proj.lora_B.weight torch.Size([3072, 16])
transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight torch.Size([16, 3072])
transformer.transformer_blocks.0.attn.add_q_proj.lora_B.weight torch.Size([3072, 16])
transformer.transformer_blocks.0.attn.add_v_proj.lora_A.weight torch.Size([16, 3072])
transformer.transformer_blocks.0.attn.add_v_proj.lora_B.weight torch.Size([3072, 16])
transformer.transformer_blocks.0.attn.to_add_out.lora_A.weight torch.Size([16, 3072])
transformer.transformer_blocks.0.attn.to_add_out.lora_B.weight torch.Size([3072, 16])
transformer.transformer_blocks.0.attn.to_k.lora_A.weight torch.Size([16, 3072])
transformer.transformer_blocks.0.attn.to_k.lora_B.weight torch.Size([3072, 16])
transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight torch.Size([16, 3072])
transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight torch.Size([3072, 16])
transformer.transformer_blocks.0.attn.to_q.lora_A.weight torch.Size([16, 3072])
transformer.transformer_blocks.0.attn.to_q.lora_B.weight torch.Size([3072, 16])
transformer.transformer_blocks.0.attn.to_v.lora_A.weight torch.Size([16, 3072])
transformer.transformer_blocks.0.attn.to_v.lora_B.weight torch.Size([3072, 16])
transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight torch.Size([16, 3072])
transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight torch.Size([12288, 16])
transformer.transformer_blocks.0.ff.net.2.lora_A.weight torch.Size([16, 12288])
transformer.transformer_blocks.0.ff.net.2.lora_B.weight torch.Size([3072, 16])
transformer.transformer_blocks.0.ff_context.net.0.proj.lora_A.weight torch.Size([16, 3072])
transformer.transformer_blocks.0.ff_context.net.0.proj.lora_B.weight torch.Size([12288, 16])
transformer.transformer_blocks.0.ff_context.net.2.lora_A.weight torch.Size([16, 12288])
transformer.transformer_blocks.0.ff_context.net.2.lora_B.weight torch.Size([3072, 16])
transformer.transformer_blocks.0.norm1.linear.lora_A.weight torch.Size([16, 3072])
transformer.transformer_blocks.0.norm1.linear.lora_B.weight torch.Size([18432, 16])
transformer.transformer_blocks.0.norm1_context.linear.lora_A.weight torch.Size([16, 3072])
transformer.transformer_blocks.0.norm1_context.linear.lora_B.weight torch.Size([18432, 16])

0 to 37
transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight torch.Size([16, 3072])
transformer.single_transformer_blocks.0.attn.to_k.lora_B.weight torch.Size([3072, 16])
transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight torch.Size([16, 3072])
transformer.single_transformer_blocks.0.attn.to_q.lora_B.weight torch.Size([3072, 16])
transformer.single_transformer_blocks.0.attn.to_v.lora_A.weight torch.Size([16, 3072])
transformer.single_transformer_blocks.0.attn.to_v.lora_B.weight torch.Size([3072, 16])
transformer.single_transformer_blocks.0.norm.linear.lora_A.weight torch.Size([16, 3072])
transformer.single_transformer_blocks.0.norm.linear.lora_B.weight torch.Size([9216, 16])
transformer.single_transformer_blocks.0.proj_mlp.lora_A.weight torch.Size([16, 3072])
transformer.single_transformer_blocks.0.proj_mlp.lora_B.weight torch.Size([12288, 16])
transformer.single_transformer_blocks.0.proj_out.lora_A.weight torch.Size([16, 15360])
transformer.single_transformer_blocks.0.proj_out.lora_B.weight torch.Size([3072, 16])
"""
"""
xlabs: Unknown format.
0 to 18
double_blocks.0.processor.proj_lora1.down.weight torch.Size([16, 3072])
double_blocks.0.processor.proj_lora1.up.weight torch.Size([3072, 16])
double_blocks.0.processor.proj_lora2.down.weight torch.Size([16, 3072])
double_blocks.0.processor.proj_lora2.up.weight torch.Size([3072, 16])
double_blocks.0.processor.qkv_lora1.down.weight torch.Size([16, 3072])
double_blocks.0.processor.qkv_lora1.up.weight torch.Size([9216, 16])
double_blocks.0.processor.qkv_lora2.down.weight torch.Size([16, 3072])
double_blocks.0.processor.qkv_lora2.up.weight torch.Size([9216, 16])
"""


import argparse
from safetensors.torch import save_file
from safetensors import safe_open
import torch


from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def convert_to_sd_scripts(sds_sd, ait_sd, sds_key, ait_key):
    ait_down_key = ait_key + ".lora_A.weight"
    if ait_down_key not in ait_sd:
        return
    ait_up_key = ait_key + ".lora_B.weight"

    down_weight = ait_sd.pop(ait_down_key)
    sds_sd[sds_key + ".lora_down.weight"] = down_weight
    sds_sd[sds_key + ".lora_up.weight"] = ait_sd.pop(ait_up_key)
    rank = down_weight.shape[0]
    sds_sd[sds_key + ".alpha"] = torch.scalar_tensor(rank, dtype=down_weight.dtype, device=down_weight.device)


def convert_to_sd_scripts_cat(sds_sd, ait_sd, sds_key, ait_keys):
    ait_down_keys = [k + ".lora_A.weight" for k in ait_keys]
    if ait_down_keys[0] not in ait_sd:
        return
    ait_up_keys = [k + ".lora_B.weight" for k in ait_keys]

    down_weights = [ait_sd.pop(k) for k in ait_down_keys]
    up_weights = [ait_sd.pop(k) for k in ait_up_keys]

    # lora_down is concatenated along dim=0, so rank is multiplied by the number of splits
    rank = down_weights[0].shape[0]
    num_splits = len(ait_keys)
    sds_sd[sds_key + ".lora_down.weight"] = torch.cat(down_weights, dim=0)

    merged_up_weights = torch.zeros(
        (sum(w.shape[0] for w in up_weights), rank * num_splits),
        dtype=up_weights[0].dtype,
        device=up_weights[0].device,
    )

    i = 0
    for j, up_weight in enumerate(up_weights):
        merged_up_weights[i : i + up_weight.shape[0], j * rank : (j + 1) * rank] = up_weight
        i += up_weight.shape[0]

    sds_sd[sds_key + ".lora_up.weight"] = merged_up_weights

    # set alpha to new_rank
    new_rank = rank * num_splits
    sds_sd[sds_key + ".alpha"] = torch.scalar_tensor(new_rank, dtype=down_weights[0].dtype, device=down_weights[0].device)


def convert_ai_toolkit_to_sd_scripts(ait_sd):
    sds_sd = {}
    for i in range(19):
        convert_to_sd_scripts(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_img_attn_proj", f"transformer.transformer_blocks.{i}.attn.to_out.0"
        )
        convert_to_sd_scripts_cat(
            sds_sd,
            ait_sd,
            f"lora_unet_double_blocks_{i}_img_attn_qkv",
            [
                f"transformer.transformer_blocks.{i}.attn.to_q",
                f"transformer.transformer_blocks.{i}.attn.to_k",
                f"transformer.transformer_blocks.{i}.attn.to_v",
            ],
        )
        convert_to_sd_scripts(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_img_mlp_0", f"transformer.transformer_blocks.{i}.ff.net.0.proj"
        )
        convert_to_sd_scripts(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_img_mlp_2", f"transformer.transformer_blocks.{i}.ff.net.2"
        )
        convert_to_sd_scripts(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_img_mod_lin", f"transformer.transformer_blocks.{i}.norm1.linear"
        )
        convert_to_sd_scripts(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_txt_attn_proj", f"transformer.transformer_blocks.{i}.attn.to_add_out"
        )
        convert_to_sd_scripts_cat(
            sds_sd,
            ait_sd,
            f"lora_unet_double_blocks_{i}_txt_attn_qkv",
            [
                f"transformer.transformer_blocks.{i}.attn.add_q_proj",
                f"transformer.transformer_blocks.{i}.attn.add_k_proj",
                f"transformer.transformer_blocks.{i}.attn.add_v_proj",
            ],
        )
        convert_to_sd_scripts(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_txt_mlp_0", f"transformer.transformer_blocks.{i}.ff_context.net.0.proj"
        )
        convert_to_sd_scripts(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_txt_mlp_2", f"transformer.transformer_blocks.{i}.ff_context.net.2"
        )
        convert_to_sd_scripts(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_txt_mod_lin", f"transformer.transformer_blocks.{i}.norm1_context.linear"
        )

    for i in range(38):
        convert_to_sd_scripts_cat(
            sds_sd,
            ait_sd,
            f"lora_unet_single_blocks_{i}_linear1",
            [
                f"transformer.single_transformer_blocks.{i}.attn.to_q",
                f"transformer.single_transformer_blocks.{i}.attn.to_k",
                f"transformer.single_transformer_blocks.{i}.attn.to_v",
                f"transformer.single_transformer_blocks.{i}.proj_mlp",
            ],
        )
        convert_to_sd_scripts(
            sds_sd, ait_sd, f"lora_unet_single_blocks_{i}_linear2", f"transformer.single_transformer_blocks.{i}.proj_out"
        )
        convert_to_sd_scripts(
            sds_sd, ait_sd, f"lora_unet_single_blocks_{i}_modulation_lin", f"transformer.single_transformer_blocks.{i}.norm.linear"
        )

    if len(ait_sd) > 0:
        logger.warning(f"Unsuppored keys for sd-scripts: {ait_sd.keys()}")
    return sds_sd


def convert_to_ai_toolkit(sds_sd, ait_sd, sds_key, ait_key):
    if sds_key + ".lora_down.weight" not in sds_sd:
        return
    down_weight = sds_sd.pop(sds_key + ".lora_down.weight")

    # scale weight by alpha and dim
    rank = down_weight.shape[0]
    alpha = sds_sd.pop(sds_key + ".alpha").item()  # alpha is scalar
    scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here
    # print(f"rank: {rank}, alpha: {alpha}, scale: {scale}")

    # calculate scale_down and scale_up to keep the same value. if scale is 4, scale_down is 2 and scale_up is 2
    scale_down = scale
    scale_up = 1.0
    while scale_down * 2 < scale_up:
        scale_down *= 2
        scale_up /= 2
    # print(f"scale: {scale}, scale_down: {scale_down}, scale_up: {scale_up}")

    ait_sd[ait_key + ".lora_A.weight"] = down_weight * scale_down
    ait_sd[ait_key + ".lora_B.weight"] = sds_sd.pop(sds_key + ".lora_up.weight") * scale_up


def convert_to_ai_toolkit_cat(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
    if sds_key + ".lora_down.weight" not in sds_sd:
        return
    down_weight = sds_sd.pop(sds_key + ".lora_down.weight")
    up_weight = sds_sd.pop(sds_key + ".lora_up.weight")
    sd_lora_rank = down_weight.shape[0]

    # scale weight by alpha and dim
    alpha = sds_sd.pop(sds_key + ".alpha")
    scale = alpha / sd_lora_rank

    # calculate scale_down and scale_up
    scale_down = scale
    scale_up = 1.0
    while scale_down * 2 < scale_up:
        scale_down *= 2
        scale_up /= 2

    down_weight = down_weight * scale_down
    up_weight = up_weight * scale_up

    # calculate dims if not provided
    num_splits = len(ait_keys)
    if dims is None:
        dims = [up_weight.shape[0] // num_splits] * num_splits
    else:
        assert sum(dims) == up_weight.shape[0]

    # check upweight is sparse or not
    is_sparse = False
    if sd_lora_rank % num_splits == 0:
        ait_rank = sd_lora_rank // num_splits
        is_sparse = True
        i = 0
        for j in range(len(dims)):
            for k in range(len(dims)):
                if j == k:
                    continue
                is_sparse = is_sparse and torch.all(up_weight[i : i + dims[j], k * ait_rank : (k + 1) * ait_rank] == 0)
            i += dims[j]
        if is_sparse:
            logger.info(f"weight is sparse: {sds_key}")

    # make ai-toolkit weight
    ait_down_keys = [k + ".lora_A.weight" for k in ait_keys]
    ait_up_keys = [k + ".lora_B.weight" for k in ait_keys]
    if not is_sparse:
        # down_weight is copied to each split
        ait_sd.update({k: down_weight for k in ait_down_keys})

        # up_weight is split to each split
        ait_sd.update({k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))})
    else:
        # down_weight is chunked to each split
        ait_sd.update({k: v for k, v in zip(ait_down_keys, torch.chunk(down_weight, num_splits, dim=0))})

        # up_weight is sparse: only non-zero values are copied to each split
        i = 0
        for j in range(len(dims)):
            ait_sd[ait_up_keys[j]] = up_weight[i : i + dims[j], j * ait_rank : (j + 1) * ait_rank].contiguous()
            i += dims[j]


def convert_sd_scripts_to_ai_toolkit(sds_sd):
    ait_sd = {}
    for i in range(19):
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_img_attn_proj", f"transformer.transformer_blocks.{i}.attn.to_out.0"
        )
        convert_to_ai_toolkit_cat(
            sds_sd,
            ait_sd,
            f"lora_unet_double_blocks_{i}_img_attn_qkv",
            [
                f"transformer.transformer_blocks.{i}.attn.to_q",
                f"transformer.transformer_blocks.{i}.attn.to_k",
                f"transformer.transformer_blocks.{i}.attn.to_v",
            ],
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_img_mlp_0", f"transformer.transformer_blocks.{i}.ff.net.0.proj"
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_img_mlp_2", f"transformer.transformer_blocks.{i}.ff.net.2"
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_img_mod_lin", f"transformer.transformer_blocks.{i}.norm1.linear"
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_txt_attn_proj", f"transformer.transformer_blocks.{i}.attn.to_add_out"
        )
        convert_to_ai_toolkit_cat(
            sds_sd,
            ait_sd,
            f"lora_unet_double_blocks_{i}_txt_attn_qkv",
            [
                f"transformer.transformer_blocks.{i}.attn.add_q_proj",
                f"transformer.transformer_blocks.{i}.attn.add_k_proj",
                f"transformer.transformer_blocks.{i}.attn.add_v_proj",
            ],
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_txt_mlp_0", f"transformer.transformer_blocks.{i}.ff_context.net.0.proj"
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_txt_mlp_2", f"transformer.transformer_blocks.{i}.ff_context.net.2"
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_double_blocks_{i}_txt_mod_lin", f"transformer.transformer_blocks.{i}.norm1_context.linear"
        )

    for i in range(38):
        convert_to_ai_toolkit_cat(
            sds_sd,
            ait_sd,
            f"lora_unet_single_blocks_{i}_linear1",
            [
                f"transformer.single_transformer_blocks.{i}.attn.to_q",
                f"transformer.single_transformer_blocks.{i}.attn.to_k",
                f"transformer.single_transformer_blocks.{i}.attn.to_v",
                f"transformer.single_transformer_blocks.{i}.proj_mlp",
            ],
            dims=[3072, 3072, 3072, 12288],
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_single_blocks_{i}_linear2", f"transformer.single_transformer_blocks.{i}.proj_out"
        )
        convert_to_ai_toolkit(
            sds_sd, ait_sd, f"lora_unet_single_blocks_{i}_modulation_lin", f"transformer.single_transformer_blocks.{i}.norm.linear"
        )

    if len(sds_sd) > 0:
        logger.warning(f"Unsuppored keys for ai-toolkit: {sds_sd.keys()}")
    return ait_sd


def main(args):
    # load source safetensors
    logger.info(f"Loading source file {args.src_path}")
    state_dict = {}
    with safe_open(args.src_path, framework="pt") as f:
        metadata = f.metadata()
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    logger.info(f"Converting {args.src} to {args.dst} format")
    if args.src == "ai-toolkit" and args.dst == "sd-scripts":
        state_dict = convert_ai_toolkit_to_sd_scripts(state_dict)
    elif args.src == "sd-scripts" and args.dst == "ai-toolkit":
        state_dict = convert_sd_scripts_to_ai_toolkit(state_dict)

        # eliminate 'shared tensors' 
        for k in list(state_dict.keys()):
            state_dict[k] = state_dict[k].detach().clone()
    else:
        raise NotImplementedError(f"Conversion from {args.src} to {args.dst} is not supported")

    # save destination safetensors
    logger.info(f"Saving destination file {args.dst_path}")
    save_file(state_dict, args.dst_path, metadata=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LoRA format")
    parser.add_argument("--src", type=str, default="ai-toolkit", help="source format, ai-toolkit or sd-scripts")
    parser.add_argument("--dst", type=str, default="sd-scripts", help="destination format, ai-toolkit or sd-scripts")
    parser.add_argument("--src_path", type=str, default=None, help="source path")
    parser.add_argument("--dst_path", type=str, default=None, help="destination path")
    args = parser.parse_args()
    main(args)
