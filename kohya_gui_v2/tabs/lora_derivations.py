"""Pure derivation functions for LoRA fields that are not a direct 1:1
FieldSpec -> TOML key mapping (per wargame plan Move 5's counter-move for
derived/composite fields; see wargame/reference/arch-matrix-lora.md section
3 for the source documentation of each derivation).

`derive(values, arch_key)` takes the raw widget-value dict (including
gui_only source widgets like `lora_type`, `xformers_radio`) and returns a
dict of TOML-key -> value overrides to merge on top of the plain FieldSpec
values before calling `build_run_config`.

Coverage note (2026-07-12): `network_module`/`network_args`'s LoRA_type-
dependent branching is now ported (see `_derive_network_module_and_args`
below), mirroring `kohya_gui/lora_gui.py:1626-1860`.
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

# Ported from kohya_gui/lora_gui.py:86-93.
LYCORIS_PRESETS_CHOICES = [
    "attn-mlp",
    "attn-only",
    "full",
    "full-lin",
    "unet-transformer-only",
    "unet-convblock-only",
]

# Native sd-scripts LoHa/LoKr (networks.loha / networks.lokr) -- distinct from
# LyCORIS/LoHa and LyCORIS/LoKr which use the third-party lycoris.kohya
# module. Ported from kohya_gui/lora_gui.py:889-949.


def _append_loraplus_network_args(
    network_args: str,
    loraplus_lr_ratio,
    loraplus_unet_lr_ratio,
    loraplus_text_encoder_lr_ratio,
) -> str:
    # sd-scripts only reads loraplus_* via --network_args (networks/lora.py's
    # create_network kwargs), not as top-level config keys, so they must be
    # appended here rather than written as their own config_toml_data entries.
    if loraplus_lr_ratio:
        network_args += f" loraplus_lr_ratio={loraplus_lr_ratio}"
    if loraplus_unet_lr_ratio:
        network_args += f" loraplus_unet_lr_ratio={loraplus_unet_lr_ratio}"
    if loraplus_text_encoder_lr_ratio:
        network_args += (
            f" loraplus_text_encoder_lr_ratio={loraplus_text_encoder_lr_ratio}"
        )
    return network_args


def _build_native_loha_lokr_network_args(
    *,
    is_lokr: bool,
    conv_dim=0,
    conv_alpha=1,
    use_tucker: bool = False,
    factor=-1,
    rank_dropout=0,
    module_dropout=0,
) -> str:
    """Build optional --network_args for networks.loha / networks.lokr.

    Only non-default optional values are emitted so the backend can apply its
    architecture auto-detection defaults (targets, exclude_patterns).
    conv_dim=0 means "do not train Conv2d 3x3+ layers".
    """
    parts: list[str] = []
    if conv_dim is not None and float(conv_dim) > 0:
        parts.append(f"conv_dim={int(conv_dim)}")
        if conv_alpha is not None:
            alpha_val = (
                int(conv_alpha) if float(conv_alpha) == int(conv_alpha) else conv_alpha
            )
            parts.append(f"conv_alpha={alpha_val}")
    if use_tucker:
        parts.append("use_tucker=True")
    if is_lokr and factor is not None and int(factor) != -1:
        parts.append(f"factor={int(factor)}")
    if rank_dropout is not None and float(rank_dropout) > 0:
        parts.append(f"rank_dropout={rank_dropout}")
    if module_dropout is not None and float(module_dropout) > 0:
        parts.append(f"module_dropout={module_dropout}")
    return (" " + " ".join(parts)) if parts else ""


def _derive_network_module_and_args(values: dict, arch_key: str) -> tuple:
    """Port of the LoRA_type -> network_module/network_args branching in
    kohya_gui/lora_gui.py:1626-1860 (train_model, right before
    config_toml_data is built).
    """
    lora_type = values.get("LoRA_type", "Standard")
    lycoris_preset = values.get("LyCORIS_preset", "full")
    conv_dim = values.get("conv_dim", 1)
    conv_alpha = values.get("conv_alpha", 1)
    use_tucker = values.get("use_tucker")
    rank_dropout = values.get("rank_dropout") or 0
    module_dropout = values.get("module_dropout") or 0
    bypass_mode = values.get("bypass_mode")
    dora_wd = values.get("dora_wd")
    use_scalar = values.get("use_scalar")
    rank_dropout_scale = values.get("rank_dropout_scale")
    train_norm = values.get("train_norm")
    constrain = values.get("constrain")
    rescaled = values.get("rescaled")
    unit = values.get("unit", 1)
    factor = values.get("factor", -1)
    use_cp = values.get("use_cp")
    decompose_both = values.get("decompose_both")
    train_on_input = values.get("train_on_input")

    network_module = None
    network_args = ""

    if lora_type == "LyCORIS/BOFT":
        network_module = "lycoris.kohya"
        network_args = f" preset={lycoris_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} module_dropout={module_dropout} use_tucker={use_tucker} rank_dropout={rank_dropout} rank_dropout_scale={rank_dropout_scale} algo=boft train_norm={train_norm}"

    if lora_type == "LyCORIS/Diag-OFT":
        network_module = "lycoris.kohya"
        network_args = f" preset={lycoris_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} module_dropout={module_dropout} use_tucker={use_tucker} rank_dropout={rank_dropout} rank_dropout_scale={rank_dropout_scale} constraint={constrain} rescaled={rescaled} algo=diag-oft train_norm={train_norm}"

    if lora_type == "LyCORIS/DyLoRA":
        network_module = "lycoris.kohya"
        network_args = f' preset={lycoris_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} use_tucker={use_tucker} block_size={unit} rank_dropout={rank_dropout} module_dropout={module_dropout} algo="dylora" train_norm={train_norm}'

    if lora_type == "LyCORIS/GLoRA":
        network_module = "lycoris.kohya"
        network_args = f' preset={lycoris_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} use_tucker={use_tucker} rank_dropout={rank_dropout} module_dropout={module_dropout} rank_dropout_scale={rank_dropout_scale} algo="glora" train_norm={train_norm}'

    if lora_type == "LyCORIS/iA3":
        network_module = "lycoris.kohya"
        network_args = f" preset={lycoris_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} train_on_input={train_on_input} algo=ia3"

    if lora_type in ("LoCon", "LyCORIS/LoCon"):
        network_module = "lycoris.kohya"
        network_args = f" preset={lycoris_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} use_tucker={use_tucker} rank_dropout={rank_dropout} bypass_mode={bypass_mode} dora_wd={dora_wd} module_dropout={module_dropout} use_tucker={use_tucker} use_scalar={use_scalar} rank_dropout_scale={rank_dropout_scale} algo=locon train_norm={train_norm}"

    if lora_type == "LyCORIS/LoHa":
        network_module = "lycoris.kohya"
        network_args = f" preset={lycoris_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} use_tucker={use_tucker} rank_dropout={rank_dropout} bypass_mode={bypass_mode} dora_wd={dora_wd} module_dropout={module_dropout} use_tucker={use_tucker} use_scalar={use_scalar} rank_dropout_scale={rank_dropout_scale} algo=loha train_norm={train_norm}"

    if lora_type == "LyCORIS/LoKr":
        network_module = "lycoris.kohya"
        network_args = f" preset={lycoris_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} use_tucker={use_tucker} rank_dropout={rank_dropout} bypass_mode={bypass_mode} dora_wd={dora_wd} module_dropout={module_dropout} factor={factor} use_cp={use_cp} use_scalar={use_scalar} decompose_both={decompose_both} rank_dropout_scale={rank_dropout_scale} algo=lokr train_norm={train_norm}"

    if lora_type == "LyCORIS/Native Fine-Tuning":
        network_module = "lycoris.kohya"
        network_args = f" preset={lycoris_preset} rank_dropout={rank_dropout} module_dropout={module_dropout} rank_dropout_scale={rank_dropout_scale} algo=full train_norm={train_norm}"

    if lora_type == "Flux1":
        network_module = "networks.lora_flux"
        kohya_lora_var_list = [
            "img_attn_dim",
            "img_mlp_dim",
            "img_mod_dim",
            "single_dim",
            "txt_attn_dim",
            "txt_mlp_dim",
            "txt_mod_dim",
            "single_mod_dim",
            "in_dims",
            "train_double_block_indices",
            "train_single_block_indices",
        ]
        train_lora_ggpo = values.get("train_lora_ggpo")
        if train_lora_ggpo:
            kohya_lora_var_list += ["ggpo_beta", "ggpo_sigma"]
        kohya_lora_vars = {
            key: values.get(key)
            for key in kohya_lora_var_list
            if values.get(key)
        }
        if values.get("split_mode"):
            kohya_lora_vars["train_blocks"] = "single"

        if values.get("split_qkv"):
            kohya_lora_vars["split_qkv"] = True
        if values.get("train_t5xxl"):
            kohya_lora_vars["train_t5xxl"] = True

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f" {key}={value}"

    if lora_type == "Flux1 OFT":
        network_module = "networks.oft_flux"
        if values.get("enable_all_linear"):
            network_args += " enable_all_linear=True"

    if lora_type == "HunyuanImage-2.1":
        network_module = "networks.lora_hunyuan_image"

    if lora_type == "Anima":
        network_module = "networks.lora_anima"

    if lora_type == "Lumina":
        network_module = "networks.lora_lumina"

    if lora_type == "Kohya LoHa":
        network_module = "networks.loha"
        network_args = _build_native_loha_lokr_network_args(
            is_lokr=False,
            conv_dim=conv_dim,
            conv_alpha=conv_alpha,
            use_tucker=use_tucker,
            factor=factor,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
        )

    if lora_type == "Kohya LoKr":
        network_module = "networks.lokr"
        network_args = _build_native_loha_lokr_network_args(
            is_lokr=True,
            conv_dim=conv_dim,
            conv_alpha=conv_alpha,
            use_tucker=use_tucker,
            factor=factor,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
        )

    if lora_type in ("Kohya LoCon", "Standard"):
        network_module = "networks.lora_sd3" if arch_key == "sd3" else "networks.lora"
        kohya_lora_var_list = [
            "down_lr_weight",
            "mid_lr_weight",
            "up_lr_weight",
            "block_lr_zero_threshold",
            "block_dims",
            "block_alphas",
            "conv_block_dims",
            "conv_block_alphas",
            "rank_dropout",
            "module_dropout",
        ]
        kohya_lora_vars = {
            key: values.get(key)
            for key in kohya_lora_var_list
            if values.get(key)
        }

        if lora_type == "Kohya LoCon":
            network_args += f' conv_dim="{conv_dim}" conv_alpha="{conv_alpha}"'

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f" {key}={value}"

    if lora_type == "LoRA-FA":
        network_module = "networks.lora_fa"
        kohya_lora_var_list = [
            "down_lr_weight",
            "mid_lr_weight",
            "up_lr_weight",
            "block_lr_zero_threshold",
            "block_dims",
            "block_alphas",
            "conv_block_dims",
            "conv_block_alphas",
            "rank_dropout",
            "module_dropout",
        ]
        kohya_lora_vars = {
            key: values.get(key)
            for key in kohya_lora_var_list
            if values.get(key)
        }
        network_args = ""
        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f" {key}={value}"

    if lora_type == "Kohya DyLoRA":
        network_module = "networks.dylora"
        kohya_lora_var_list = [
            "conv_dim",
            "conv_alpha",
            "down_lr_weight",
            "mid_lr_weight",
            "up_lr_weight",
            "block_lr_zero_threshold",
            "block_dims",
            "block_alphas",
            "conv_block_dims",
            "conv_block_alphas",
            "rank_dropout",
            "module_dropout",
            "unit",
        ]
        kohya_lora_vars = {
            key: values.get(key)
            for key in kohya_lora_var_list
            if values.get(key)
        }
        network_args = ""
        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f" {key}={value}"

    network_args = _append_loraplus_network_args(
        network_args,
        values.get("loraplus_lr_ratio"),
        values.get("loraplus_unet_lr_ratio"),
        values.get("loraplus_text_encoder_lr_ratio"),
    )

    return network_module, network_args.strip()


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
    # match, None otherwise. Only applies when values["xformers"] still
    # holds the raw v1 dropdown string -- a second derive() pass (every
    # do_train/do_save via _build_values) sees the live Gradio checkbox's
    # bool and must not reinterpret it as the dropdown string and null
    # xformers back out.
    xformers_choice = values.get("xformers")
    if isinstance(xformers_choice, str) and xformers_choice in (
        "xformers",
        "sdpa",
        "none",
    ):
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

    # lr_scheduler_args / optimizer_args: left as the raw widget string here.
    # FieldSpec.to_toml/from_toml (_to_arg_list/_from_arg_list) own the
    # textbox<->list round-trip for both the run-config build and the
    # import_json/normalize_widget_value display path; duplicating the split
    # here double-applied it (a widget already showing "[]" from a prior
    # derive() pass would get re-split into ["[]"], crashing
    # optimizer.py's `key, value = arg.split("=")` -- see dreambooth
    # optimizer_args regression, 2026-07-12).

    # text_encoder_lr: GUI single value -> 2-element list [te_lr, te_lr]
    # (t5xxl_lr override is a further per-arch composite not yet ported --
    # tracked as a known gap, see DERIVATION_IN_PROGRESS in the harness).
    te_lr = values.get("text_encoder_lr")
    if te_lr not in (None, "", 0):
        out["text_encoder_lr"] = [te_lr, te_lr]
    else:
        out["text_encoder_lr"] = None

    network_module, network_args = _derive_network_module_and_args(values, arch_key)
    out["network_module"] = network_module
    out["network_args"] = network_args

    return out
