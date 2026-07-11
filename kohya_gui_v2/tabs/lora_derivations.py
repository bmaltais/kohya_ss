"""Pure derivation functions for LoRA fields that are not a direct 1:1
FieldSpec -> TOML key mapping (per wargame plan Move 5's counter-move for
derived/composite fields; see wargame/reference/arch-matrix-lora.md section
3 for the source documentation of each derivation).

`derive(values, arch_key)` takes the raw widget-value dict (including
gui_only source widgets like `lora_type`, `xformers_radio`) and returns a
dict of TOML-key -> value overrides to merge on top of the plain FieldSpec
values before calling `build_run_config`.

Coverage note (2026-07-11, B1 checkpoint): the universal derivations below
are implemented and exercised by the equivalence harness. `network_module`
and `network_args`'s full LoRA_type-dependent branching (LyCORIS presets,
Flux1 block dims, LoRA+ ratios, GGPO, native LoHa/LoKr) is NOT yet fully
ported -- this is a known, tracked gap (see B1 status notes), not silently
missing. Presets that depend on those branches will show as equivalence
diffs until this is completed.
"""

import math
import os

NO_CLIP_SKIP_ARCHS = {"hunyuan_image", "anima", "lumina"}
NO_MAX_TOKEN_LENGTH_ARCHS = {"flux1", "hunyuan_image", "anima", "lumina"}
TRAIN_INPAINTING_SUPPORTED_ARCHS = {"sd15", "sd2", "sdxl"}
TEXT_ENCODER_OUTPUTS_ARCHS = {
    "sdxl",
    "flux1",
    "sd3",
    "hunyuan_image",
    "anima",
    "lumina",
}


def _count_dataset_steps(train_data_dir: str) -> int:
    """Sum `repeats * image_count` across `<repeats>_<name>` subfolders,
    matching the old GUI's folder-naming convention (lora_gui.py ~1464-1500).
    """
    if not train_data_dir or not os.path.isdir(train_data_dir):
        return 0
    total = 0
    for folder in os.listdir(train_data_dir):
        folder_path = os.path.join(train_data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        try:
            repeats = int(folder.split("_")[0])
        except ValueError:
            continue
        num_images = len(
            [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
        )
        total += repeats * num_images
    return total


def derive(values: dict, arch_key: str) -> dict:
    out = {}

    # lr_scheduler_num_cycles falls back to epoch when the widget is blank
    # (empty string, not merely 0 -- matches old GUI's `!= ""` check).
    raw_cycles = values.get("lr_scheduler_num_cycles")
    if raw_cycles in (None, ""):
        epoch = values.get("epoch")
        if epoch not in (None, ""):
            out["lr_scheduler_num_cycles"] = int(epoch)

    # max_train_steps: when unset (0/empty), old GUI computes it from the
    # dataset folder's repeats*image_count, batch size, grad accumulation,
    # epoch count, and a 2x regularization-image factor.
    raw_max_train_steps = values.get("max_train_steps")
    if raw_max_train_steps in (None, "", 0):
        train_data_dir = values.get("train_data_dir") or ""
        total_steps = _count_dataset_steps(train_data_dir)
        train_batch_size = int(values.get("train_batch_size") or 1)
        grad_accum = int(values.get("gradient_accumulation_steps") or 1)
        epoch = int(values.get("epoch") or 0)
        reg_factor = 2 if values.get("reg_data_dir") else 1
        if total_steps and epoch:
            out["max_train_steps"] = int(
                math.ceil(
                    float(total_steps)
                    / train_batch_size
                    / grad_accum
                    * epoch
                    * reg_factor
                )
            )

    # xformers radio -> xformers/sdpa mutually exclusive booleans. Matches
    # old GUI exactly (lora_gui.py ~2121/2143): True only on an exact string
    # match, None otherwise -- including for stale presets that still store
    # a plain boolean under the "xformers" key from an older GUI schema.
    xformers_choice = values.get("xformers")
    out["xformers"] = True if xformers_choice == "xformers" else None
    out["sdpa"] = True if xformers_choice == "sdpa" else None

    # wandb_run_name falls back to output_name when empty
    if not values.get("wandb_run_name"):
        out["wandb_run_name"] = values.get("output_name") or None

    # network_train_unet_only / network_train_text_encoder_only derived from
    # the text_encoder_lr vs unet_lr comparison (0 on one side => only the
    # other is trained), with a hunyuan_image override forcing unet-only.
    text_encoder_lr = values.get("text_encoder_lr") or 0
    unet_lr = values.get("unet_lr") or 0
    if arch_key == "hunyuan_image":
        out["network_train_unet_only"] = True
        out["network_train_text_encoder_only"] = None
    elif text_encoder_lr and not unet_lr:
        out["network_train_text_encoder_only"] = True
        out["network_train_unet_only"] = None
    elif unet_lr and not text_encoder_lr:
        out["network_train_unet_only"] = True
        out["network_train_text_encoder_only"] = None

    # lr_warmup_steps: prefer the direct step count if given; otherwise
    # derive from the lr_warmup PERCENTAGE * max_train_steps / 100. Old GUI
    # has a confirmed bug (lora_gui.py ~1552, non-dataset_config branch)
    # that omits the "* max_train_steps" multiplication -- v2 implements
    # the correct formula; the resulting divergence is a registered known
    # defect (see tests/test_v2_equivalence_lora.py KNOWN_DEFECT_MISMATCH).
    lr_warmup_steps = values.get("lr_warmup_steps") or 0
    lr_warmup_pct = values.get("lr_warmup") or 0
    if not lr_warmup_steps and lr_warmup_pct:
        max_train_steps = out.get("max_train_steps", values.get("max_train_steps") or 0)
        out["lr_warmup_steps"] = round(
            float(lr_warmup_pct) * float(max_train_steps) / 100
        )

    # noise_offset / noise_offset_random_strength / adaptive_noise_scale
    # only apply when noise_offset_type == "Original"; multires_noise_*
    # only apply when noise_offset_type == "Multires" (lora_gui.py
    # ~1934-2058).
    noise_offset_type = values.get("noise_offset_type")
    if noise_offset_type != "Original":
        out["noise_offset"] = None
        out["noise_offset_random_strength"] = None
        out["adaptive_noise_scale"] = None
    if noise_offset_type != "Multires":
        out["multires_noise_discount"] = None
        out["multires_noise_iterations"] = None

    # clip_skip forced None for archs that don't use a CLIP text encoder
    # the same way SD1/2/SDXL do
    if arch_key in NO_CLIP_SKIP_ARCHS:
        out["clip_skip"] = None
    elif not values.get("clip_skip"):
        out["clip_skip"] = None

    # max_token_length forced None for archs with a different tokenizer setup
    if arch_key in NO_MAX_TOKEN_LENGTH_ARCHS:
        out["max_token_length"] = None

    # train_inpainting only supported for sd15/sd2/sdxl
    if arch_key not in TRAIN_INPAINTING_SUPPORTED_ARCHS:
        out["train_inpainting"] = None

    # cache_text_encoder_outputs(_to_disk): True or None, never False, and
    # sourced from a per-arch prefixed widget (e.g. flux1_cache_text_encoder_
    # outputs, sdxl_cache_text_encoder_outputs, ...), only meaningful for the
    # archs with a separate text-encoder cache step.
    for key in ("cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk"):
        if arch_key in TEXT_ENCODER_OUTPUTS_ARCHS:
            source = values.get(f"{arch_key}_{key}", values.get(key))
            out[key] = True if source else None
        else:
            out[key] = None

    # no_half_vae: sdxl-only composite (sdxl and sdxl_no_half_vae)
    if arch_key == "sdxl":
        out["no_half_vae"] = True if values.get("sdxl_no_half_vae") else None
    else:
        out["no_half_vae"] = None

    # lr_scheduler_args: space-separated, double-quoted "key=value" textbox
    # -> list of strings; always a list, even when empty (lora_gui.py ~2017:
    # `str(lr_scheduler_args).replace('"', "").split()`, no None fallback).
    out["lr_scheduler_args"] = (
        str(values.get("lr_scheduler_args") or "").replace('"', "").split()
    )

    # optimizer_args: same transform, but old GUI special-cases a literal
    # empty-list widget value to None (lora_gui.py ~2079-2083: `if
    # optimizer_args != [] else None`) -- a string widget value (including
    # "") always goes through str().split(), which produces [] for "".
    raw_optimizer_args = values.get("optimizer_args")
    if raw_optimizer_args == []:
        out["optimizer_args"] = None
    else:
        out["optimizer_args"] = str(raw_optimizer_args or "").replace('"', "").split()

    # text_encoder_lr: GUI single value -> 2-element list [te_lr, te_lr]
    # (t5xxl_lr override is a further per-arch composite not yet ported --
    # tracked as a known gap, see DERIVATION_IN_PROGRESS in the harness).
    te_lr = values.get("text_encoder_lr")
    if te_lr not in (None, "", 0):
        out["text_encoder_lr"] = [te_lr, te_lr]
    else:
        out["text_encoder_lr"] = None

    return out
