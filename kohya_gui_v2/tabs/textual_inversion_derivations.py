"""Pure derivation functions for Textual Inversion fields that are not a
direct 1:1 FieldSpec -> TOML key mapping (per wargame plan Move 5's
counter-move for derived/composite fields; see
wargame/reference/arch-matrix-textual_inversion.md section 3 for the source
documentation of each derivation).

`derive(values, arch_key)` takes the raw widget-value dict and returns a
dict of TOML-key -> value overrides to merge on top of the plain FieldSpec
values before calling `build_run_config`.

Architecture keys: sd_v1v2 (train_textual_inversion.py), sdxl
(sdxl_train_textual_inversion.py) -- see textual_inversion_fields.py's
ARCHITECTURE_CHOICES. Both `no_half_vae`/`disable_mmap_load_safetensors`
are "soft" sdxl-only per the matrix (emitted for both archs but only
meaningful under sdxl; the falsy-drop already hides them for sd_v1v2 since
the source widgets default to off there) -- no extra gating code needed.
"""

import math
import os


def _count_dataset_steps(train_data_dir: str) -> int:
    """Sum `repeats * image_count` across `<repeats>_<name>` subfolders,
    matching the old GUI's folder-naming convention
    (textual_inversion_gui.py:607-696).
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

    # max_train_steps: skipped entirely when dataset_config is set (raw
    # widget value passes through); otherwise recomputed from the dataset
    # folder when the widget is 0 (textual_inversion_gui.py:607-696).
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

    # lr_warmup_steps: absolute step count wins if > 0; else lr_warmup
    # (percent widget) is treated as a bare fraction, not multiplied by
    # max_train_steps (textual_inversion_gui.py:699-708).
    raw_lr_warmup_steps = values.get("lr_warmup_steps") or 0
    lr_warmup_pct = values.get("lr_warmup") or 0
    if raw_lr_warmup_steps and float(raw_lr_warmup_steps) > 0:
        out["lr_warmup_steps"] = int(raw_lr_warmup_steps)
    elif lr_warmup_pct:
        out["lr_warmup_steps"] = float(lr_warmup_pct) / 100
    else:
        out["lr_warmup_steps"] = 0

    # lr_scheduler_num_cycles falls back to epoch when the widget is blank.
    raw_cycles = values.get("lr_scheduler_num_cycles")
    if raw_cycles in (None, ""):
        epoch = values.get("epoch")
        if epoch not in (None, ""):
            out["lr_scheduler_num_cycles"] = int(epoch)

    # xformers dropdown -> xformers/sdpa mutually exclusive booleans.
    xformers_choice = values.get("xformers")
    out["xformers"] = True if xformers_choice == "xformers" else None
    out["sdpa"] = True if xformers_choice == "sdpa" else None

    # template dropdown ("caption" / "object template" / "style template")
    # -> use_object_template/use_style_template mutually exclusive booleans.
    template_choice = values.get("template")
    out["use_object_template"] = True if template_choice == "object template" else None
    out["use_style_template"] = True if template_choice == "style template" else None

    # wandb_run_name falls back to output_name when empty.
    if not values.get("wandb_run_name"):
        out["wandb_run_name"] = values.get("output_name") or None

    # lr_scheduler_args / optimizer_args: space-separated, quote-stripped
    # textbox -> list of strings.
    out["lr_scheduler_args"] = (
        str(values.get("lr_scheduler_args") or "").replace('"', "").split()
    )
    raw_optimizer_args = values.get("optimizer_args")
    if raw_optimizer_args == []:
        out["optimizer_args"] = None
    else:
        out["optimizer_args"] = str(raw_optimizer_args or "").replace('"', "").split()

    # sample_prompts: raw textbox content written to a prompt file under
    # output_dir; the TOML value is the file path, not the raw text.
    sample_prompts = values.get("sample_prompts")
    output_dir = values.get("output_dir")
    if sample_prompts and output_dir:
        from kohya_gui.class_sample_images import create_prompt_file

        out["sample_prompts"] = create_prompt_file(sample_prompts, output_dir)

    # stop_text_encoder_training_pct never reaches the TOML for this tab --
    # neither trainer script accepts it (arch-matrix-textual_inversion.md
    # section 3).
    out["stop_text_encoder_training_pct"] = None

    return out
