"""Pure derivation functions for DreamBooth fields that are not a direct
1:1 FieldSpec -> TOML key mapping (per wargame plan Move 5's counter-move for
derived/composite fields; see wargame/reference/arch-matrix-dreambooth.md
section 3 for the source documentation of each derivation, and section 1
for the 4-architecture script-selection table this derive() mirrors).

`derive(values, arch_key)` takes the raw widget-value dict (including
gui_only source widgets like `sdxl_no_half_vae`, `flux1_clip_l`) and returns
a dict of TOML-key -> value overrides to merge on top of the plain
FieldSpec values before calling `build_run_config`.

Architecture keys used here: sd15/sd2 (default, train_db.py), sdxl, sd3,
flux1 -- see dreambooth_fields.py's ARCHITECTURE_CHOICES.
"""

import math
import os

BASE_ARCHS = {"sd15", "sd2"}
NO_HALF_VAE_ARCHS = {"sdxl"}
TRAIN_INPAINTING_SUPPORTED_ARCHS = {"sd15", "sd2", "sdxl"}
TEXT_ENCODER_OUTPUTS_ARCHS = {"sdxl", "sd3", "flux1"}
TEXT_ENCODER_OUTPUTS_TO_DISK_ARCHS = {"sd3", "flux1"}


def _count_dataset_steps(train_data_dir: str) -> int:
    """Sum `repeats * image_count` across `<repeats>_<name>` subfolders,
    matching the old GUI's folder-naming convention
    (dreambooth_gui.py:746-832).
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

    # max_train_steps: only recomputed from the dataset folder when no
    # external dataset_config is set and the raw widget value is 0
    # (dreambooth_gui.py:746-832). reg_data_dir doubles the step count.
    raw_max_train_steps = values.get("max_train_steps")
    if not values.get("dataset_config") and raw_max_train_steps in (None, "", 0):
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

    # lr_warmup_steps: prefer the direct step count if given; otherwise
    # treat lr_warmup as a raw percentage divided by 100 -- this tab's
    # documented formula (dreambooth_gui.py:834-844) does NOT multiply by
    # max_train_steps, unlike LoRA's formula (see lora_derivations.py); this
    # is the old GUI's actual behavior for this tab, ported faithfully, not
    # a defect to fix here.
    raw_lr_warmup_steps = values.get("lr_warmup_steps") or 0
    lr_warmup_pct = values.get("lr_warmup") or 0
    if raw_lr_warmup_steps and float(raw_lr_warmup_steps) > 0:
        out["lr_warmup_steps"] = int(raw_lr_warmup_steps)
    elif lr_warmup_pct:
        out["lr_warmup_steps"] = float(lr_warmup_pct) / 100
    else:
        out["lr_warmup_steps"] = 0

    # train_inpainting: silently forced off for SD3/FLUX.1 regardless of the
    # checkbox; when effectively True, also forces cache_latents and
    # cache_latents_to_disk to False (masks are generated per-step, cannot
    # use cached latents).
    train_inpainting = bool(values.get("train_inpainting"))
    if arch_key in ("sd3", "flux1"):
        train_inpainting = False
    out["train_inpainting"] = train_inpainting or None
    if train_inpainting:
        out["cache_latents"] = False
        out["cache_latents_to_disk"] = False

    # cache_text_encoder_outputs(_to_disk): OR of the per-arch prefixed
    # widget, only meaningful for the archs with a separate text-encoder
    # cache step. SDXL has no "to disk" variant.
    if arch_key in TEXT_ENCODER_OUTPUTS_ARCHS:
        source = values.get(f"{arch_key}_cache_text_encoder_outputs")
        out["cache_text_encoder_outputs"] = True if source else None
    else:
        out["cache_text_encoder_outputs"] = None

    if arch_key in TEXT_ENCODER_OUTPUTS_TO_DISK_ARCHS:
        source = values.get(f"{arch_key}_cache_text_encoder_outputs_to_disk")
        out["cache_text_encoder_outputs_to_disk"] = True if source else None
    else:
        out["cache_text_encoder_outputs_to_disk"] = None

    # no_half_vae: sdxl-only composite (sdxl and sdxl_no_half_vae).
    if arch_key in NO_HALF_VAE_ARCHS:
        out["no_half_vae"] = True if values.get("sdxl_no_half_vae") else None
    else:
        out["no_half_vae"] = None

    # fused_backward_pass: routed from one of three per-arch widgets --
    # sd3_fused_backward_pass (SD3), flux_fused_backward_pass (FLUX.1),
    # else the plain fused_backward_pass widget (base/SDXL).
    if arch_key == "sd3":
        out["fused_backward_pass"] = bool(values.get("sd3_fused_backward_pass"))
    elif arch_key == "flux1":
        out["fused_backward_pass"] = bool(values.get("flux_fused_backward_pass"))
    else:
        out["fused_backward_pass"] = bool(values.get("fused_backward_pass"))

    # train_text_encoder: SDXL-only, derived from whether either
    # text-encoder learning rate is set (dreambooth_gui.py:921).
    if arch_key == "sdxl":
        te1 = values.get("learning_rate_te1") or 0
        te2 = values.get("learning_rate_te2") or 0
        out["train_text_encoder"] = (
            bool(
                (te1 is not None and float(te1) > 0)
                or (te2 is not None and float(te2) > 0)
            )
            or None
        )
    else:
        out["train_text_encoder"] = None

    # xformers dropdown -> xformers/sdpa mutually exclusive booleans.
    xformers_choice = values.get("xformers")
    out["xformers"] = True if xformers_choice == "xformers" else None
    out["sdpa"] = True if xformers_choice == "sdpa" else None

    # learning_rate_te / learning_rate_te1 / learning_rate_te2: only one
    # family is meaningful per architecture (matrix rows 66-68).
    if arch_key == "sdxl":
        out["learning_rate_te"] = None
    else:
        out["learning_rate_te1"] = None
        out["learning_rate_te2"] = None

    # clip_l / t5xxl: two GUI-side source widgets alias to the same TOML
    # key depending on architecture (flux1_clip_l/flux1_t5xxl vs the plain
    # sd3 widgets clip_l/t5xxl).
    if arch_key == "flux1":
        out["clip_l"] = values.get("flux1_clip_l") or None
        out["t5xxl"] = values.get("flux1_t5xxl") or None
    elif arch_key == "sd3":
        out["clip_l"] = values.get("clip_l") or None
        out["t5xxl"] = values.get("t5xxl") or None
    else:
        out["clip_l"] = None
        out["t5xxl"] = None

    # wandb_run_name falls back to output_name when empty.
    if not values.get("wandb_run_name"):
        out["wandb_run_name"] = values.get("output_name") or None

    # lr_scheduler_num_cycles falls back to epoch when the widget is blank.
    raw_cycles = values.get("lr_scheduler_num_cycles")
    if raw_cycles in (None, ""):
        epoch = values.get("epoch")
        if epoch not in (None, ""):
            out["lr_scheduler_num_cycles"] = int(epoch)

    # lr_scheduler_args / optimizer_args: space-separated, quote-stripped
    # textbox -> list of strings (dreambooth_gui.py, same coercion as LoRA).
    out["lr_scheduler_args"] = (
        str(values.get("lr_scheduler_args") or "").replace('"', "").split()
    )
    raw_optimizer_args = values.get("optimizer_args")
    if raw_optimizer_args == []:
        out["optimizer_args"] = None
    else:
        out["optimizer_args"] = str(raw_optimizer_args or "").replace('"', "").split()

    # sample_prompts: raw textbox content is written to a prompt file under
    # output_dir; the TOML value is the file path, not the raw text
    # (dreambooth_gui.py, via kohya_gui.class_sample_images.create_prompt_file).
    sample_prompts = values.get("sample_prompts")
    output_dir = values.get("output_dir")
    if sample_prompts and output_dir:
        from kohya_gui.class_sample_images import create_prompt_file

        out["sample_prompts"] = create_prompt_file(sample_prompts, output_dir)

    # split_mode / train_blocks: present as GUI widgets (round-tripped
    # through save/open) but never reach the TOML for this tab -- flux_train.py
    # (the only FLUX.1 script this tab targets) doesn't accept them
    # (arch-matrix-dreambooth.md lines 166-167, 200).
    out["split_mode"] = None
    out["train_blocks"] = None

    return out
