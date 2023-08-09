import argparse
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def is_unet_key(key):
    # VAE or TextEncoder, the last one is for SDXL
    return not ("first_stage_model" in key or "cond_stage_model" in key or "conditioner." in key)


TEXT_ENCODER_KEY_REPLACEMENTS = [
    ("cond_stage_model.transformer.embeddings.", "cond_stage_model.transformer.text_model.embeddings."),
    ("cond_stage_model.transformer.encoder.", "cond_stage_model.transformer.text_model.encoder."),
    ("cond_stage_model.transformer.final_layer_norm.", "cond_stage_model.transformer.text_model.final_layer_norm."),
]


# support for models with different text encoder keys
def replace_text_encoder_key(key):
    for rep_from, rep_to in TEXT_ENCODER_KEY_REPLACEMENTS:
        if key.startswith(rep_from):
            return True, rep_to + key[len(rep_from) :]
    return False, key


def merge(args):
    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float

    if args.saving_precision == "fp16":
        save_dtype = torch.float16
    elif args.saving_precision == "bf16":
        save_dtype = torch.bfloat16
    else:
        save_dtype = torch.float

    # check if all models are safetensors
    for model in args.models:
        if not model.endswith("safetensors"):
            print(f"Model {model} is not a safetensors model")
            exit()
        if not os.path.isfile(model):
            print(f"Model {model} does not exist")
            exit()

    assert len(args.models) == len(args.ratios) or args.ratios is None, "ratios must be the same length as models"

    # load and merge
    ratio = 1.0 / len(args.models)  # default
    supplementary_key_ratios = {}  # [key] = ratio, for keys not in all models, add later

    merged_sd = None
    first_model_keys = set()  # check missing keys in other models
    for i, model in enumerate(args.models):
        if args.ratios is not None:
            ratio = args.ratios[i]

        if merged_sd is None:
            # load first model
            print(f"Loading model {model}, ratio = {ratio}...")
            merged_sd = {}
            with safe_open(model, framework="pt", device=args.device) as f:
                for key in tqdm(f.keys()):
                    value = f.get_tensor(key)
                    _, key = replace_text_encoder_key(key)

                    first_model_keys.add(key)

                    if not is_unet_key(key) and args.unet_only:
                        supplementary_key_ratios[key] = 1.0  # use first model's value for VAE or TextEncoder
                        continue

                    value = ratio * value.to(dtype)  # first model's value * ratio
                    merged_sd[key] = value

            print(f"Model has {len(merged_sd)} keys " + ("(UNet only)" if args.unet_only else ""))
            continue

        # load other models
        print(f"Loading model {model}, ratio = {ratio}...")

        with safe_open(model, framework="pt", device=args.device) as f:
            model_keys = f.keys()
            for key in tqdm(model_keys):
                _, new_key = replace_text_encoder_key(key)
                if new_key not in merged_sd:
                    if args.show_skipped and new_key not in first_model_keys:
                        print(f"Skip: {new_key}")
                    continue

                value = f.get_tensor(key)
                merged_sd[new_key] = merged_sd[new_key] + ratio * value.to(dtype)

            # enumerate keys not in this model
            model_keys = set(model_keys)
            for key in merged_sd.keys():
                if key in model_keys:
                    continue
                print(f"Key {key} not in model {model}, use first model's value")
                if key in supplementary_key_ratios:
                    supplementary_key_ratios[key] += ratio
                else:
                    supplementary_key_ratios[key] = ratio

    # add supplementary keys' value (including VAE and TextEncoder)
    if len(supplementary_key_ratios) > 0:
        print("add first model's value")
        with safe_open(model, framework="pt", device=args.device) as f:
            for key in tqdm(f.keys()):
                _, new_key = replace_text_encoder_key(key)
                if new_key not in supplementary_key_ratios:
                    continue

                if is_unet_key(new_key): # not VAE or TextEncoder
                    print(f"Key {new_key} not in all models, ratio = {supplementary_key_ratios[new_key]}")

                value = f.get_tensor(key)  # original key

                if new_key not in merged_sd:
                    merged_sd[new_key] = supplementary_key_ratios[new_key] * value.to(dtype)
                else:
                    merged_sd[new_key] = merged_sd[new_key] + supplementary_key_ratios[new_key] * value.to(dtype)

    # save
    output_file = args.output
    if not output_file.endswith(".safetensors"):
        output_file = output_file + ".safetensors"

    print(f"Saving to {output_file}...")

    # convert to save_dtype
    for k in merged_sd.keys():
        merged_sd[k] = merged_sd[k].to(save_dtype)

    save_file(merged_sd, output_file)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge models")
    parser.add_argument("--models", nargs="+", type=str, help="Models to merge")
    parser.add_argument("--output", type=str, help="Output model")
    parser.add_argument("--ratios", nargs="+", type=float, help="Ratios of models, default is equal, total = 1.0")
    parser.add_argument("--unet_only", action="store_true", help="Only merge unet")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use, default is cpu")
    parser.add_argument(
        "--precision", type=str, default="float", choices=["float", "fp16", "bf16"], help="Calculation precision, default is float"
    )
    parser.add_argument(
        "--saving_precision",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="Saving precision, default is float",
    )
    parser.add_argument("--show_skipped", action="store_true", help="Show skipped keys (keys not in first model)")

    args = parser.parse_args()
    merge(args)
