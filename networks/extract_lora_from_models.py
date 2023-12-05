# extract approximating LoRA by svd from two SD models
# The code is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo!

import argparse
import json
import os
import time
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from library import sai_model_spec, model_util, sdxl_model_util
import lora


# CLAMP_QUANTILE = 0.99
# MIN_DIFF = 1e-1


def save_to_file(file_name, model, state_dict, dtype):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(model, file_name)
    else:
        torch.save(model, file_name)


def svd(
    model_org=None,
    model_tuned=None,
    save_to=None,
    dim=4,
    v2=None,
    sdxl=None,
    conv_dim=None,
    v_parameterization=None,
    device=None,
    save_precision=None,
    clamp_quantile=0.99,
    min_diff=0.01,
    no_metadata=False,
):
    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    assert v2 != sdxl or (not v2 and not sdxl), "v2 and sdxl cannot be specified at the same time / v2とsdxlは同時に指定できません"
    if v_parameterization is None:
        v_parameterization = v2

    save_dtype = str_to_dtype(save_precision)

    # load models
    if not sdxl:
        print(f"loading original SD model : {model_org}")
        text_encoder_o, _, unet_o = model_util.load_models_from_stable_diffusion_checkpoint(v2, model_org)
        text_encoders_o = [text_encoder_o]
        print(f"loading tuned SD model : {model_tuned}")
        text_encoder_t, _, unet_t = model_util.load_models_from_stable_diffusion_checkpoint(v2, model_tuned)
        text_encoders_t = [text_encoder_t]
        model_version = model_util.get_model_version_str_for_sd1_sd2(v2, v_parameterization)
    else:
        print(f"loading original SDXL model : {model_org}")
        text_encoder_o1, text_encoder_o2, _, unet_o, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
            sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, model_org, "cpu"
        )
        text_encoders_o = [text_encoder_o1, text_encoder_o2]
        print(f"loading original SDXL model : {model_tuned}")
        text_encoder_t1, text_encoder_t2, _, unet_t, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
            sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, model_tuned, "cpu"
        )
        text_encoders_t = [text_encoder_t1, text_encoder_t2]
        model_version = sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0

    # create LoRA network to extract weights: Use dim (rank) as alpha
    if conv_dim is None:
        kwargs = {}
    else:
        kwargs = {"conv_dim": conv_dim, "conv_alpha": conv_dim}

    lora_network_o = lora.create_network(1.0, dim, dim, None, text_encoders_o, unet_o, **kwargs)
    lora_network_t = lora.create_network(1.0, dim, dim, None, text_encoders_t, unet_t, **kwargs)
    assert len(lora_network_o.text_encoder_loras) == len(
        lora_network_t.text_encoder_loras
    ), f"model version is different (SD1.x vs SD2.x) / それぞれのモデルのバージョンが違います（SD1.xベースとSD2.xベース） "

    # get diffs
    diffs = {}
    text_encoder_different = False
    for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.text_encoder_loras, lora_network_t.text_encoder_loras)):
        lora_name = lora_o.lora_name
        module_o = lora_o.org_module
        module_t = lora_t.org_module
        diff = module_t.weight - module_o.weight

        # Text Encoder might be same
        if not text_encoder_different and torch.max(torch.abs(diff)) > min_diff:
            text_encoder_different = True
            print(f"Text encoder is different. {torch.max(torch.abs(diff))} > {min_diff}")

        diff = diff.float()
        diffs[lora_name] = diff

    if not text_encoder_different:
        print("Text encoder is same. Extract U-Net only.")
        lora_network_o.text_encoder_loras = []
        diffs = {}

    for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.unet_loras, lora_network_t.unet_loras)):
        lora_name = lora_o.lora_name
        module_o = lora_o.org_module
        module_t = lora_t.org_module
        diff = module_t.weight - module_o.weight
        diff = diff.float()

        if args.device:
            diff = diff.to(args.device)

        diffs[lora_name] = diff

    # make LoRA with svd
    print("calculating by svd")
    lora_weights = {}
    with torch.no_grad():
        for lora_name, mat in tqdm(list(diffs.items())):
            # if conv_dim is None, diffs do not include LoRAs for conv2d-3x3
            conv2d = len(mat.size()) == 4
            kernel_size = None if not conv2d else mat.size()[2:4]
            conv2d_3x3 = conv2d and kernel_size != (1, 1)

            rank = dim if not conv2d_3x3 or conv_dim is None else conv_dim
            out_dim, in_dim = mat.size()[0:2]

            if device:
                mat = mat.to(device)

            # print(lora_name, mat.size(), mat.device, rank, in_dim, out_dim)
            rank = min(rank, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

            if conv2d:
                if conv2d_3x3:
                    mat = mat.flatten(start_dim=1)
                else:
                    mat = mat.squeeze()

            U, S, Vh = torch.linalg.svd(mat)

            U = U[:, :rank]
            S = S[:rank]
            U = U @ torch.diag(S)

            Vh = Vh[:rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp_quantile)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            if conv2d:
                U = U.reshape(out_dim, rank, 1, 1)
                Vh = Vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])

            U = U.to("cpu").contiguous()
            Vh = Vh.to("cpu").contiguous()

            lora_weights[lora_name] = (U, Vh)

    # make state dict for LoRA
    lora_sd = {}
    for lora_name, (up_weight, down_weight) in lora_weights.items():
        lora_sd[lora_name + ".lora_up.weight"] = up_weight
        lora_sd[lora_name + ".lora_down.weight"] = down_weight
        lora_sd[lora_name + ".alpha"] = torch.tensor(down_weight.size()[0])

    # load state dict to LoRA and save it
    lora_network_save, lora_sd = lora.create_network_from_weights(1.0, None, None, text_encoders_o, unet_o, weights_sd=lora_sd)
    lora_network_save.apply_to(text_encoders_o, unet_o)  # create internal module references for state_dict

    info = lora_network_save.load_state_dict(lora_sd)
    print(f"Loading extracted LoRA weights: {info}")

    dir_name = os.path.dirname(save_to)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # minimum metadata
    net_kwargs = {}
    if conv_dim is not None:
        net_kwargs["conv_dim"] = str(conv_dim)
        net_kwargs["conv_alpha"] = str(float(conv_dim))

    metadata = {
        "ss_v2": str(v2),
        "ss_base_model_version": model_version,
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(dim),
        "ss_network_alpha": str(float(dim)),
        "ss_network_args": json.dumps(net_kwargs),
    }

    if not no_metadata:
        title = os.path.splitext(os.path.basename(save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(None, v2, v_parameterization, sdxl, True, False, time.time(), title=title)
        metadata.update(sai_metadata)

    lora_network_save.save_weights(save_to, save_dtype, metadata)
    print(f"LoRA weights are saved to: {save_to}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="load Stable Diffusion v2.x model / Stable Diffusion 2.xのモデルを読み込む")
    parser.add_argument(
        "--v_parameterization",
        action="store_true",
        default=None,
        help="make LoRA metadata for v-parameterization (default is same to v2) / 作成するLoRAのメタデータにv-parameterization用と設定する（省略時はv2と同じ）",
    )
    parser.add_argument(
        "--sdxl", action="store_true", help="load Stable Diffusion SDXL base model / Stable Diffusion SDXL baseのモデルを読み込む"
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はfloat",
    )
    parser.add_argument(
        "--model_org",
        type=str,
        default=None,
        required=True,
        help="Stable Diffusion original model: ckpt or safetensors file / 元モデル、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--model_tuned",
        type=str,
        default=None,
        required=True,
        help="Stable Diffusion tuned model, LoRA is difference of `original to tuned`: ckpt or safetensors file / 派生モデル（生成されるLoRAは元→派生の差分になります）、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        required=True,
        help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors",
    )
    parser.add_argument("--dim", type=int, default=4, help="dimension (rank) of LoRA (default 4) / LoRAの次元数（rank）（デフォルト4）")
    parser.add_argument(
        "--conv_dim",
        type=int,
        default=None,
        help="dimension (rank) of LoRA for Conv2d-3x3 (default None, disabled) / LoRAのConv2d-3x3の次元数（rank）（デフォルトNone、適用なし）",
    )
    parser.add_argument("--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う")
    parser.add_argument(
        "--clamp_quantile",
        type=float,
        default=0.99,
        help="Quantile clamping value, float, (0-1). Default = 0.99 / 値をクランプするための分位点、float、(0-1)。デフォルトは0.99",
    )
    parser.add_argument(
        "--min_diff",
        type=float,
        default=0.01,
        help="Minimum difference between finetuned model and base to consider them different enough to extract, float, (0-1). Default = 0.01 /"
        + "LoRAを抽出するために元モデルと派生モデルの差分の最小値、float、(0-1)。デフォルトは0.01",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのメタデータを保存しない（LoRAの最低限のss_metadataは保存される）",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    svd(**vars(args))
