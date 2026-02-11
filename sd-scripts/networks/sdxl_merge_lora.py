import itertools
import math
import argparse
import os
import time
import concurrent.futures
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from library import sai_model_spec, sdxl_model_util, train_util
import library.model_util as model_util
import lora
import oft
from svd_merge_lora import format_lbws, get_lbw_block_index, LAYER26
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, model, metadata):
    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(model, file_name, metadata=metadata)
    else:
        torch.save(model, file_name)


def detect_method_from_training_model(models, dtype):
    for model in models:
        # TODO It is better to use key names to detect the method
        lora_sd, _ = load_state_dict(model, dtype)
        for key in tqdm(lora_sd.keys()):
            if "lora_up" in key or "lora_down" in key:
                return "LoRA"
            elif "oft_blocks" in key:
                return "OFT"


def merge_to_sd_model(text_encoder1, text_encoder2, unet, models, ratios, lbws, merge_dtype):
    text_encoder1.to(merge_dtype)
    text_encoder2.to(merge_dtype)
    unet.to(merge_dtype)

    # detect the method: OFT or LoRA_module
    method = detect_method_from_training_model(models, merge_dtype)
    logger.info(f"method:{method}")

    if lbws:
        lbws, _, LBW_TARGET_IDX = format_lbws(lbws)
    else:
        LBW_TARGET_IDX = []

    # create module map
    name_to_module = {}
    for i, root_module in enumerate([text_encoder1, text_encoder2, unet]):
        if method == "LoRA":
            if i <= 1:
                if i == 0:
                    prefix = lora.LoRANetwork.LORA_PREFIX_TEXT_ENCODER1
                else:
                    prefix = lora.LoRANetwork.LORA_PREFIX_TEXT_ENCODER2
                target_replace_modules = lora.LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
            else:
                prefix = lora.LoRANetwork.LORA_PREFIX_UNET
                target_replace_modules = (
                    lora.LoRANetwork.UNET_TARGET_REPLACE_MODULE + lora.LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
                )
        elif method == "OFT":
            prefix = oft.OFTNetwork.OFT_PREFIX_UNET
            # ALL_LINEAR includes ATTN_ONLY, so we don't need to specify ATTN_ONLY
            target_replace_modules = (
                oft.OFTNetwork.UNET_TARGET_REPLACE_MODULE_ALL_LINEAR + oft.OFTNetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
            )

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        name_to_module[lora_name] = child_module

    for model, ratio, lbw in itertools.zip_longest(models, ratios, lbws):
        logger.info(f"loading: {model}")
        lora_sd, _ = load_state_dict(model, merge_dtype)

        logger.info(f"merging...")

        if lbw:
            lbw_weights = [1] * 26
            for index, value in zip(LBW_TARGET_IDX, lbw):
                lbw_weights[index] = value
            logger.info(f"lbw: {dict(zip(LAYER26.keys(), lbw_weights))}")

        if method == "LoRA":
            for key in tqdm(lora_sd.keys()):
                if "lora_down" in key:
                    up_key = key.replace("lora_down", "lora_up")
                    alpha_key = key[: key.index("lora_down")] + "alpha"

                    # find original module for this lora
                    module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
                    if module_name not in name_to_module:
                        logger.info(f"no module found for LoRA weight: {key}")
                        continue
                    module = name_to_module[module_name]
                    # logger.info(f"apply {key} to {module}")

                    down_weight = lora_sd[key]
                    up_weight = lora_sd[up_key]

                    dim = down_weight.size()[0]
                    alpha = lora_sd.get(alpha_key, dim)
                    scale = alpha / dim

                    if lbw:
                        index = get_lbw_block_index(key, True)
                        is_lbw_target = index in LBW_TARGET_IDX
                        if is_lbw_target:
                            scale *= lbw_weights[index]  # keyがlbwの対象であれば、lbwの重みを掛ける

                    # W <- W + U * D
                    weight = module.weight
                    # logger.info(module_name, down_weight.size(), up_weight.size())
                    if len(weight.size()) == 2:
                        # linear
                        weight = weight + ratio * (up_weight @ down_weight) * scale
                    elif down_weight.size()[2:4] == (1, 1):
                        # conv2d 1x1
                        weight = (
                            weight
                            + ratio
                            * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                            * scale
                        )
                    else:
                        # conv2d 3x3
                        conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                        # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                        weight = weight + ratio * conved * scale

                    module.weight = torch.nn.Parameter(weight)

        elif method == "OFT":

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for key in tqdm(lora_sd.keys()):
                if "oft_blocks" in key:
                    oft_blocks = lora_sd[key]
                    dim = oft_blocks.shape[0]
                    break
            for key in tqdm(lora_sd.keys()):
                if "alpha" in key:
                    oft_blocks = lora_sd[key]
                    alpha = oft_blocks.item()
                    break

            def merge_to(key):
                if "alpha" in key:
                    return

                # find original module for this OFT
                module_name = ".".join(key.split(".")[:-1])
                if module_name not in name_to_module:
                    logger.info(f"no module found for OFT weight: {key}")
                    return
                module = name_to_module[module_name]

                # logger.info(f"apply {key} to {module}")

                oft_blocks = lora_sd[key]

                if isinstance(module, torch.nn.Linear):
                    out_dim = module.out_features
                elif isinstance(module, torch.nn.Conv2d):
                    out_dim = module.out_channels

                num_blocks = dim
                block_size = out_dim // dim
                constraint = (0 if alpha is None else alpha) * out_dim

                multiplier = 1
                if lbw:
                    index = get_lbw_block_index(key, False)
                    is_lbw_target = index in LBW_TARGET_IDX
                    if is_lbw_target:
                        multiplier *= lbw_weights[index]

                block_Q = oft_blocks - oft_blocks.transpose(1, 2)
                norm_Q = torch.norm(block_Q.flatten())
                new_norm_Q = torch.clamp(norm_Q, max=constraint)
                block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))
                I = torch.eye(block_size, device=oft_blocks.device).unsqueeze(0).repeat(num_blocks, 1, 1)
                block_R = torch.matmul(I + block_Q, (I - block_Q).inverse())
                block_R_weighted = multiplier * block_R + (1 - multiplier) * I
                R = torch.block_diag(*block_R_weighted)

                # get org weight
                org_sd = module.state_dict()
                org_weight = org_sd["weight"].to(device)

                R = R.to(org_weight.device, dtype=org_weight.dtype)

                if org_weight.dim() == 4:
                    weight = torch.einsum("oihw, op -> pihw", org_weight, R)
                else:
                    weight = torch.einsum("oi, op -> pi", org_weight, R)

                weight = weight.contiguous()  # Make Tensor contiguous; required due to ThreadPoolExecutor

                module.weight = torch.nn.Parameter(weight)

            # TODO multi-threading may cause OOM on CPU if cpu_count is too high and RAM is not enough
            max_workers = 1 if device.type != "cpu" else None  # avoid OOM on GPU
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(executor.map(merge_to, lora_sd.keys()), total=len(lora_sd.keys())))


def merge_lora_models(models, ratios, lbws, merge_dtype, concat=False, shuffle=False):
    base_alphas = {}  # alpha for merged model
    base_dims = {}

    # detect the method: OFT or LoRA_module
    method = detect_method_from_training_model(models, merge_dtype)
    if method == "OFT":
        raise ValueError(
            "OFT model is not supported for merging OFT models. / OFTモデルはOFTモデル同士のマージには対応していません"
        )

    if lbws:
        lbws, _, LBW_TARGET_IDX = format_lbws(lbws)
    else:
        LBW_TARGET_IDX = []

    merged_sd = {}
    v2 = None
    base_model = None
    for model, ratio, lbw in itertools.zip_longest(models, ratios, lbws):
        logger.info(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)

        if lbw:
            lbw_weights = [1] * 26
            for index, value in zip(LBW_TARGET_IDX, lbw):
                lbw_weights[index] = value
            logger.info(f"lbw: {dict(zip(LAYER26.keys(), lbw_weights))}")

        if lora_metadata is not None:
            if v2 is None:
                v2 = lora_metadata.get(train_util.SS_METADATA_KEY_V2, None)  # returns string, SDXLはv2がないのでFalseのはず
            if base_model is None:
                base_model = lora_metadata.get(train_util.SS_METADATA_KEY_BASE_MODEL_VERSION, None)

        # get alpha and dim
        alphas = {}  # alpha for current model
        dims = {}  # dims for current model
        for key in lora_sd.keys():
            if "alpha" in key:
                lora_module_name = key[: key.rfind(".alpha")]
                alpha = float(lora_sd[key].detach().numpy())
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha
            elif "lora_down" in key:
                lora_module_name = key[: key.rfind(".lora_down")]
                dim = lora_sd[key].size()[0]
                dims[lora_module_name] = dim
                if lora_module_name not in base_dims:
                    base_dims[lora_module_name] = dim

        for lora_module_name in dims.keys():
            if lora_module_name not in alphas:
                alpha = dims[lora_module_name]
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha

        logger.info(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        # merge
        logger.info(f"merging...")
        for key in tqdm(lora_sd.keys()):
            if "alpha" in key:
                continue

            if "lora_up" in key and concat:
                concat_dim = 1
            elif "lora_down" in key and concat:
                concat_dim = 0
            else:
                concat_dim = None

            lora_module_name = key[: key.rfind(".lora_")]

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            scale = math.sqrt(alpha / base_alpha) * ratio
            scale = abs(scale) if "lora_up" in key else scale  # マイナスの重みに対応する。

            if lbw:
                index = get_lbw_block_index(key, True)
                is_lbw_target = index in LBW_TARGET_IDX
                if is_lbw_target:
                    scale *= lbw_weights[index]  # keyがlbwの対象であれば、lbwの重みを掛ける

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == lora_sd[key].size() or concat_dim is not None
                ), f"weights shape mismatch merging v1 and v2, different dims? / 重みのサイズが合いません。v1とv2、または次元数の異なるモデルはマージできません"
                if concat_dim is not None:
                    merged_sd[key] = torch.cat([merged_sd[key], lora_sd[key] * scale], dim=concat_dim)
                else:
                    merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)
        if shuffle:
            key_down = lora_module_name + ".lora_down.weight"
            key_up = lora_module_name + ".lora_up.weight"
            dim = merged_sd[key_down].shape[0]
            perm = torch.randperm(dim)
            merged_sd[key_down] = merged_sd[key_down][perm]
            merged_sd[key_up] = merged_sd[key_up][:, perm]

    logger.info("merged model")
    logger.info(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    # check all dims are same
    dims_list = list(set(base_dims.values()))
    alphas_list = list(set(base_alphas.values()))
    all_same_dims = True
    all_same_alphas = True
    for dims in dims_list:
        if dims != dims_list[0]:
            all_same_dims = False
            break
    for alphas in alphas_list:
        if alphas != alphas_list[0]:
            all_same_alphas = False
            break

    # build minimum metadata
    dims = f"{dims_list[0]}" if all_same_dims else "Dynamic"
    alphas = f"{alphas_list[0]}" if all_same_alphas else "Dynamic"
    metadata = train_util.build_minimum_network_metadata(v2, base_model, "networks.lora", dims, alphas, None)

    return merged_sd, metadata


def merge(args):
    assert len(args.models) == len(
        args.ratios
    ), f"number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"
    if args.lbws:
        assert len(args.models) == len(
            args.lbws
        ), f"number of models must be equal to number of ratios / モデルの数と層別適用率の数は合わせてください"
    else:
        args.lbws = []  # zip_longestで扱えるようにlbws未使用時には空のリストにしておく

    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    if args.sd_model is not None:
        logger.info(f"loading SD model: {args.sd_model}")

        (
            text_model1,
            text_model2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_model_util.load_models_from_sdxl_checkpoint(sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, args.sd_model, "cpu")

        merge_to_sd_model(text_model1, text_model2, unet, args.models, args.ratios, args.lbws, merge_dtype)

        if args.no_metadata:
            sai_metadata = None
        else:
            merged_from = sai_model_spec.build_merged_from([args.sd_model] + args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                None, False, False, True, False, False, time.time(), title=title, merged_from=merged_from
            )

        logger.info(f"saving SD model to: {args.save_to}")
        sdxl_model_util.save_stable_diffusion_checkpoint(
            args.save_to, text_model1, text_model2, unet, 0, 0, ckpt_info, vae, logit_scale, sai_metadata, save_dtype
        )
    else:
        state_dict, metadata = merge_lora_models(args.models, args.ratios, args.lbws, merge_dtype, args.concat, args.shuffle)

        # cast to save_dtype before calculating hashes
        for key in list(state_dict.keys()):
            value = state_dict[key]
            if type(value) == torch.Tensor and value.dtype.is_floating_point and value.dtype != save_dtype:
                state_dict[key] = value.to(save_dtype)

        logger.info(f"calculating hashes and creating metadata...")

        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        if not args.no_metadata:
            merged_from = sai_model_spec.build_merged_from(args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                state_dict, False, False, True, True, False, time.time(), title=title, merged_from=merged_from
            )
            metadata.update(sai_metadata)

        logger.info(f"saving model to: {args.save_to}")
        save_to_file(args.save_to, state_dict, metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はマージ時の精度と同じ",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="precision in merging (float is recommended) / マージの計算時の精度（floatを推奨）",
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default=None,
        help="Stable Diffusion model to load: ckpt or safetensors file, merge LoRA models if omitted / 読み込むモデル、ckptまたはsafetensors。省略時はLoRAモデル同士をマージする",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="LoRA models to merge: ckpt or safetensors file / マージするLoRAモデル、ckptまたはsafetensors",
    )
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each model / それぞれのLoRAモデルの比率")
    parser.add_argument("--lbws", type=str, nargs="*", help="lbw for each model / それぞれのLoRAモデルの層別適用率")
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのメタデータを保存しない（LoRAの最低限のss_metadataは保存される）",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="concat lora instead of merge (The dim(rank) of the output LoRA is the sum of the input dims) / "
        + "マージの代わりに結合する（LoRAのdim(rank)は入力dimの合計になる）",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle lora weight./ " + "LoRAの重みをシャッフルする",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    merge(args)
