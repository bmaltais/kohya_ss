"""Pure derivation functions for Finetune fields that are not a direct 1:1
FieldSpec -> TOML key mapping (per wargame plan Move 5's counter-move for
derived/composite fields; see wargame/reference/arch-matrix-finetune.md
section 4 for the source documentation of each derivation).

`derive(values, arch_key)` takes the raw widget-value dict (including
gui_only source widgets like `sdxl_no_half_vae`, `flux1_clip_l`) and returns
a dict of TOML-key -> value overrides to merge on top of the plain
FieldSpec values before calling `build_run_config`.

Simple "arch X only" or "AND <field> truthy" gates that don't require
composing multiple widgets are NOT reimplemented here -- FieldSpec.archs
(via for_selection) already hides/excludes fields per architecture. Note
finetune_tab.py calls build_run_config with zero_survives_false=True
(finetune_gui.py's own falsy-drop filter uses `value is False`, keeping
numeric 0 -- see config_io.py's build_run_config docstring), so fields the
old GUI explicitly guards with `!= 0 else None` need the explicit
ZERO_TO_NONE_FIELDS handling below; they are NOT dropped by default the way
they would be under the equality-based filter.

Architecture keys: base (fine_tune.py), sdxl, sd3, flux1, anima, lumina --
see finetune_fields.py's ARCHITECTURE_CHOICES.
"""

import math
import os

NO_HALF_VAE_ARCHS = {"sdxl"}
TRAIN_INPAINTING_SUPPRESSED_ARCHS = {"sd3", "flux1", "anima", "lumina"}
TEXT_ENCODER_OUTPUTS_ARCHS = {"sdxl", "sd3", "flux1", "anima", "lumina"}
TEXT_ENCODER_OUTPUTS_TO_DISK_ARCHS = {"sd3", "flux1", "anima", "lumina"}
NO_CLIP_SKIP_ARCHS = {"anima", "lumina"}
NO_MAX_TOKEN_LENGTH_ARCHS = {"anima", "lumina"}
SHOW_TIMESTEPS_ARCHS = {"flux1", "sd3", "anima", "lumina"}

# Fields the old GUI explicitly guards with `!= 0 else None` (or `> 0 else
# None` for fused_optimizer_groups) INDEPENDENT of the blanket falsy-drop
# filter -- arch-matrix-finetune.md's per-key "None if == 0" notes.
ZERO_TO_NONE_FIELDS = (
    "adaptive_noise_scale",
    "clip_skip",
    "fused_optimizer_groups",
    "ip_noise_gamma",
    "max_timestep",
    "max_train_epochs",
    "min_snr_gamma",
    "min_timestep",
    "multires_noise_iterations",
    "noise_offset",
    "sample_every_n_epochs",
    "sample_every_n_steps",
    "save_every_n_epochs",
    "save_every_n_steps",
    "save_last_n_epochs",
    "save_last_n_epochs_state",
    "save_last_n_steps",
    "save_last_n_steps_state",
    "seed",
    "v_pred_like_loss",
    "vae_batch_size",
)


def _zero_to_none(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value == 0:
        return None
    return value


def _count_image_files(image_folder: str) -> int:
    if not image_folder or not os.path.isdir(image_folder):
        return 0
    return len(
        [
            f
            for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
    )


def derive(values: dict, arch_key: str) -> dict:
    out = {}

    # enable_bucket: hardcoded True in finetune_gui.py's config_toml_data --
    # not backed by any widget at all (arch-matrix-finetune.md #83).
    out["enable_bucket"] = True

    # max_train_steps: recomputed from the image-folder file count when no
    # external dataset_config is set and the raw widget is 0
    # (finetune_gui.py:990-1026). flip_aug halves the resulting step count
    # (source images are doubled by the augmentation).
    raw_max_train_steps = values.get("max_train_steps")
    if not values.get("dataset_config") and raw_max_train_steps in (None, "", 0):
        image_folder = values.get("train_data_dir") or ""
        image_num = _count_image_files(image_folder)
        dataset_repeats = int(values.get("dataset_repeats") or 1)
        repeats = image_num * dataset_repeats
        train_batch_size = int(values.get("train_batch_size") or 1)
        grad_accum = int(values.get("gradient_accumulation_steps") or 1)
        epoch = int(values.get("epoch") or 0)
        if repeats and epoch:
            computed = int(
                math.ceil(float(repeats) / train_batch_size / grad_accum * epoch)
            )
            if values.get("flip_aug") and computed:
                computed = int(math.ceil(computed / 2))
            out["max_train_steps"] = computed

    # lr_warmup_steps: absolute step count wins if > 0; else lr_warmup
    # (percent widget) is treated as a bare fraction, NOT multiplied by
    # max_train_steps (finetune_gui.py:1030-1042 -- flagged in the matrix as
    # a possible pre-existing quirk, ported faithfully, not "fixed" here).
    raw_lr_warmup_steps = values.get("lr_warmup_steps") or 0
    lr_warmup_pct = values.get("lr_warmup") or 0
    if raw_lr_warmup_steps and float(raw_lr_warmup_steps) > 0:
        out["lr_warmup_steps"] = int(raw_lr_warmup_steps)
    elif lr_warmup_pct:
        out["lr_warmup_steps"] = float(lr_warmup_pct) / 100
    else:
        out["lr_warmup_steps"] = 0

    # train_inpainting: forced off for sd3/flux1/anima/lumina regardless of
    # the checkbox; when effectively True, also forces cache_latents and
    # cache_latents_to_disk to False.
    train_inpainting = bool(values.get("train_inpainting"))
    if arch_key in TRAIN_INPAINTING_SUPPRESSED_ARCHS:
        train_inpainting = False
    out["train_inpainting"] = train_inpainting or None
    if train_inpainting:
        out["cache_latents"] = False
        out["cache_latents_to_disk"] = False

    # cache_text_encoder_outputs(_to_disk): OR of the per-arch prefixed
    # widget. SDXL has no "to disk" variant.
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

    # no_half_vae: sdxl-only composite.
    if arch_key in NO_HALF_VAE_ARCHS:
        out["no_half_vae"] = True if values.get("sdxl_no_half_vae") else None
    else:
        out["no_half_vae"] = None

    # in_json: composite of the GUI's own `train_dir` widget (a metadata
    # storage folder, DISTINCT from `train_data_dir`/image_folder -- the
    # actual training images folder) + one of two metadata-filename
    # widgets, switched by use_latent_files (finetune_gui.py section 4).
    train_dir = values.get("train_dir") or ""
    use_latent_files = values.get("use_latent_files")
    if use_latent_files == "Yes":
        filename = values.get("latent_metadata_filename") or ""
    else:
        filename = values.get("caption_metadata_filename") or ""
    out["in_json"] = f"{train_dir}/{filename}" if train_dir and filename else None

    # xformers dropdown -> xformers/sdpa mutually exclusive booleans.
    xformers_choice = values.get("xformers")
    out["xformers"] = True if xformers_choice == "xformers" else None
    out["sdpa"] = True if xformers_choice == "sdpa" else None

    # clip_l / t5xxl: per-arch duplicate widgets collapse to one TOML key.
    if arch_key == "flux1":
        out["clip_l"] = values.get("flux1_clip_l") or None
        out["t5xxl"] = values.get("flux1_t5xxl") or None
    elif arch_key == "sd3":
        out["clip_l"] = values.get("clip_l") or None
        out["t5xxl"] = values.get("t5xxl") or None
    else:
        out["clip_l"] = None
        out["t5xxl"] = None

    # ae: lumina_ae if lumina, else flux1's ae widget if flux1, else None.
    if arch_key == "lumina":
        out["ae"] = values.get("lumina_ae") or None
    elif arch_key == "flux1":
        out["ae"] = values.get("ae") or None
    else:
        out["ae"] = None

    # fused_backward_pass: 3-way arch dispatch.
    if arch_key == "sd3":
        out["fused_backward_pass"] = bool(values.get("sd3_fused_backward_pass"))
    elif arch_key == "flux1":
        out["fused_backward_pass"] = bool(values.get("flux_fused_backward_pass"))
    else:
        out["fused_backward_pass"] = bool(values.get("fused_backward_pass"))

    # discrete_flow_shift: lumina/anima/flux1 3-way dispatch, float() cast
    # for the lumina/anima branches.
    if arch_key == "lumina":
        v = values.get("lumina_discrete_flow_shift")
        out["discrete_flow_shift"] = float(v) if v not in (None, "") else None
    elif arch_key == "anima":
        v = values.get("anima_discrete_flow_shift")
        out["discrete_flow_shift"] = float(v) if v not in (None, "") else None
    elif arch_key == "flux1":
        out["discrete_flow_shift"] = values.get("discrete_flow_shift")
    else:
        out["discrete_flow_shift"] = None

    # model_prediction_type: lumina/flux1 dispatch.
    if arch_key == "lumina":
        out["model_prediction_type"] = (
            values.get("lumina_model_prediction_type") or None
        )
    elif arch_key == "flux1":
        out["model_prediction_type"] = values.get("model_prediction_type") or None
    else:
        out["model_prediction_type"] = None

    # timestep_sampling: lumina/anima/flux1 3-way dispatch.
    if arch_key == "lumina":
        out["timestep_sampling"] = values.get("lumina_timestep_sampling") or None
    elif arch_key == "anima":
        out["timestep_sampling"] = values.get("anima_timestep_sampling") or None
    elif arch_key == "flux1":
        out["timestep_sampling"] = values.get("timestep_sampling") or None
    else:
        out["timestep_sampling"] = None

    # sigmoid_scale: anima/lumina dispatch, float() cast.
    if arch_key == "lumina":
        v = values.get("lumina_sigmoid_scale")
        out["sigmoid_scale"] = float(v) if v not in (None, "") else None
    elif arch_key == "anima":
        v = values.get("anima_sigmoid_scale")
        out["sigmoid_scale"] = float(v) if v not in (None, "") else None
    else:
        out["sigmoid_scale"] = None

    # clip_skip / max_token_length: forced off for anima/lumina, which don't
    # use a CLIP text encoder the same way the other architectures do.
    if arch_key in NO_CLIP_SKIP_ARCHS:
        out["clip_skip"] = None
    if arch_key in NO_MAX_TOKEN_LENGTH_ARCHS:
        out["max_token_length"] = None

    # vae: anima-only widget (GUI name anima_vae).
    out["vae"] = values.get("anima_vae") if arch_key == "anima" else None

    # show_timesteps_resolution depends on the sibling show_timesteps field,
    # not just its own gate/value (finetune_gui.py section 4).
    if arch_key not in SHOW_TIMESTEPS_ARCHS or not values.get("show_timesteps"):
        out["show_timesteps"] = None
        out["show_timesteps_resolution"] = None

    # attn_mode: anima-only, only emitted when set to something other than
    # the trainer's own default ("torch").
    if arch_key == "anima" and values.get("attn_mode") == "torch":
        out["attn_mode"] = None

    # compile_backend/compile_mode/compile_dynamic/compile_fullgraph/
    # compile_cache_size_limit: anima-only, all gated on `compile` itself
    # being truthy; compile_dynamic is further suppressed at "auto".
    if arch_key == "anima":
        if not values.get("compile"):
            out["compile_backend"] = None
            out["compile_mode"] = None
            out["compile_dynamic"] = None
            out["compile_fullgraph"] = None
            out["compile_cache_size_limit"] = None
        elif values.get("compile_dynamic") == "auto":
            out["compile_dynamic"] = None

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

    # split_mode / train_blocks: no widgets at all for this tab -- always
    # None, never reach the TOML (arch-matrix-finetune.md lines 198-199).
    out["split_mode"] = None
    out["train_blocks"] = None

    # Explicit `!= 0 else None` guards independent of the blanket falsy
    # drop (see ZERO_TO_NONE_FIELDS docstring above).
    for name in ZERO_TO_NONE_FIELDS:
        current = out[name] if name in out else values.get(name)
        out[name] = _zero_to_none(current)

    return out
