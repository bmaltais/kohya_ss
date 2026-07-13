"""Move 7 (checkpoint B9, Finetune portion): equivalence harness -- old GUI
vs kohya_gui_v2, same shape as test_v2_equivalence_lora.py (B8) /
test_v2_equivalence_dreambooth.py.

For each preset JSON, run it through the old GUI's train_model(print_only=True)
to capture the real run TOML, then run the same values through v2's field
registry + derivation + build_run_config, and diff the two dicts.
"""

import glob
import json
import os

import pytest

from conftest import (
    build_train_model_kwargs,
    run_train_model_and_load_toml,
)
from kohya_gui import finetune_gui
from kohya_gui_v2.config_io import build_run_config
from kohya_gui_v2.tabs.finetune_fields import FINETUNE_REGISTRY, derive

PRESET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "presets", "finetune"
)
PRESETS = sorted(p for p in glob.glob(os.path.join(PRESET_DIR, "*.json")))

# Tier 1: cosmetic, structural differences that are correct-by-design.
COSMETIC_ALLOWLIST = {
    "max_data_loader_n_workers",
    "output_dir",
    "output_name",
    "sample_prompts",
    "wandb_run_name",
    # persistent_data_loader_workers is a real store_true boolean
    # (sd-scripts/library/args.py:352) but old GUI's widget is a Number
    # with an explicit int() cast (arch-matrix-finetune.md), so old TOML
    # holds a literal 0/1 while v2's correctly-typed CHECKBOX FieldSpec
    # omits a False default -- both are behaviorally the trainer's
    # store_true default; the int-vs-omitted encoding is cosmetic only.
    "persistent_data_loader_workers",
    # These fields are absent from all four finetune preset fixtures.
    # build_train_model_kwargs' generic "default missing params to 0"
    # fallback doesn't match their real widget types/defaults (booleans
    # get a synthetic int 0 that survives old GUI's identity-based filter;
    # string fields get a synthetic int 0 instead of "" or their real
    # non-empty default). A test-harness fixture artifact (would not occur
    # with a real Gradio session where every widget has its true default),
    # not a genuine GUI-vs-v2 equivalence concern -- same category as
    # DreamBooth's fp8_base/train_inpainting/huber_scale/max_grad_norm.
    "debiased_estimation_loss",
    "fp8_base",
    "full_bf16",
    "ip_noise_gamma_random_strength",
    "log_tracker_name",
    "masked_loss",
    "noise_offset_random_strength",
    "save_model_as",
    "scale_v_pred_loss_like_noise_pred",
    "train_inpainting",
    "wandb_api_key",
    "weighted_captions",
    "huber_c",
    "huber_scale",
    "multires_noise_discount",
    "sample_sampler",
    # in_json's source `train_dir` widget (metadata storage folder, distinct
    # from train_data_dir) is absent from all four presets; conftest
    # defaults it to a synthetic int 0 (not a string), which old GUI
    # formats unconditionally into "0/<caption_metadata_filename>" while
    # v2's falsy check on the same synthetic 0 correctly treats it as unset.
    # Fixture artifact, not a real equivalence concern -- none of the
    # available finetune presets actually exercise the caption/latent
    # metadata pre-training pipeline.
    "in_json",
    # old GUI always writes an empty list for an unset optimizer/scheduler
    # args textbox; v2's FieldSpec.to_toml (_to_arg_list) returns None for
    # an empty string, which the falsy-drop filter omits entirely. Both
    # `optimizer_args = []` and an absent key are no-ops to sd-scripts
    # (`args.optimizer_args is not None and len(args.optimizer_args) > 0`).
    "lr_scheduler_args",
    "optimizer_args",
}

# GUI kwarg name -> v2 FieldSpec (TOML key) name.
RENAME_MAP = {
    "optimizer": "optimizer_type",
    "max_resolution": "resolution",
    "sd3_text_encoder_batch_size": "text_encoder_batch_size",
}

# Move 7 known-defect register (tier 2): old GUI provably emits a key no
# Finetune-family sd-scripts parser accepts (Move 4 gap analysis,
# wargame/reference/gap-analysis-finetune.md).
KNOWN_DEFECT_OLD_GUI_ONLY = {
    "noise_offset_type",  # dead across all 6 archs (gap-analysis summary)
}

NUMERIC_FIXUPS = (
    "max_train_steps",
    "max_train_epochs",
    "seed",
    "vae_batch_size",
    "save_every_n_steps",
    "save_every_n_epochs",
    "clip_skip",
    "max_token_length",
    "caption_dropout_every_n_epochs",
    "min_snr_gamma",
    "min_timestep",
    "max_timestep",
    "keep_tokens",
    "lr_warmup",
    "epoch",
    "gradient_accumulation_steps",
    "train_batch_size",
    "noise_offset",
    "noise_offset_random_strength",
    "adaptive_noise_scale",
    "multires_noise_iterations",
    "multires_noise_discount",
    "max_bucket_reso",
    "min_bucket_reso",
    "bucket_reso_steps",
    "dataset_repeats",
)

STRING_OVERRIDES = (
    "pretrained_model_name_or_path",
    "vae",
    "dataset_config",
    "logging_dir",
    "resume",
    "log_tracker_config",
    "lr_scheduler_args",
    "optimizer_args",
    "additional_parameters",
    "training_comment",
    "wandb_run_name",
    "output_name",
    "sample_prompts",
    "output_dir",
    "train_data_dir",
    "image_folder",
    "ae",
    "clip_l",
    "flux1_clip_l",
    "t5xxl",
    "flux1_t5xxl",
    "t5xxl_device",
    "t5xxl_dtype",
    "huggingface_repo_id",
    "huggingface_token",
    "huggingface_repo_type",
    "huggingface_repo_visibility",
    "huggingface_path_in_repo",
    "metadata_author",
    "metadata_description",
    "metadata_license",
    "metadata_tags",
    "metadata_title",
    "log_with",
    "log_config",
    "loss_type",
    "huber_schedule",
    "lr_scheduler_type",
    "dynamo_backend",
    "dynamo_mode",
    "extra_accelerate_launch_args",
    "model_prediction_type",
    "timestep_sampling",
    "train_blocks",
    "weighting_scheme",
    "clip_g",
    "in_json",
    "latent_metadata_filename",
    "caption_metadata_filename",
    "use_latent_files",
    "block_lr",
)


def _architecture_for(kwargs: dict) -> str:
    if kwargs.get("sdxl_checkbox"):
        return "sdxl"
    if kwargs.get("sd3_checkbox"):
        return "sd3"
    if kwargs.get("flux1_checkbox"):
        return "flux1"
    if kwargs.get("anima_checkbox"):
        return "anima"
    if kwargs.get("lumina_checkbox"):
        return "lumina"
    return "base"


@pytest.mark.parametrize(
    "preset_path", PRESETS, ids=[os.path.basename(p) for p in PRESETS]
)
def test_finetune_equivalence(preset_path):
    kwargs = build_train_model_kwargs(
        finetune_gui.train_model,
        preset_path,
        numeric_fixups=NUMERIC_FIXUPS,
        string_overrides=STRING_OVERRIDES,
    )
    for key, value in list(kwargs.items()):
        if isinstance(value, str):
            try:
                kwargs[key] = int(value)
            except ValueError:
                try:
                    kwargs[key] = float(value)
                except ValueError:
                    pass

    for key, default in (
        ("bucket_reso_steps", 64),
        ("min_bucket_reso", 256),
        ("max_bucket_reso", 2048),
        ("dataset_repeats", 1),
    ):
        if not kwargs.get(key):
            kwargs[key] = default

    if kwargs.get("pretrained_model_name_or_path") and not os.path.exists(
        kwargs["pretrained_model_name_or_path"]
    ):
        kwargs["pretrained_model_name_or_path"] = ""

    # train_dir (metadata storage folder) may carry the preset author's
    # local path; finetune_gui.py os.mkdir()'s it unconditionally when
    # non-empty, which fails for a path outside this machine's filesystem
    # root. Treat a non-existent train_dir as unset, like the model path.
    if kwargs.get("train_dir") and not os.path.isdir(
        os.path.dirname(str(kwargs["train_dir"])) or "."
    ):
        kwargs["train_dir"] = ""

    kwargs.setdefault("image_folder", "")
    if not os.path.isdir(kwargs["image_folder"]):
        kwargs["image_folder"] = ""
    if not kwargs["image_folder"]:
        kwargs["image_folder"] = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "test", "img"
        )

    old_toml = run_train_model_and_load_toml(finetune_gui, kwargs)

    arch_key = _architecture_for(kwargs)

    raw_values = dict(kwargs)
    raw_values["train_data_dir"] = raw_values.get("image_folder")
    for gui_name, v2_name in RENAME_MAP.items():
        if gui_name in raw_values:
            raw_values[v2_name] = raw_values[gui_name]

    # Keys actually present in the source preset JSON (plus their renamed
    # v2 counterparts). A FieldSpec absent from the real preset must default
    # to the FieldSpec's own default -- not to build_train_model_kwargs'
    # synthetic 0/False fill for missing train_model() positional params
    # (Move 7's gap-added-field invariant).
    with open(preset_path, encoding="utf-8") as f:
        cfg_keys = set(json.load(f).keys())
    for gui_name, v2_name in RENAME_MAP.items():
        if gui_name in cfg_keys:
            cfg_keys.add(v2_name)
    # train_data_dir (image_folder) is always synthesized by this harness
    # (falls back to test/img when absent from the preset, same as the
    # LoRA/DreamBooth harnesses) -- both sides always receive a real path,
    # so it's never a genuine "gap-added field" concern.
    cfg_keys.add("train_data_dir")
    if "xformers" in cfg_keys:
        cfg_keys.add("sdpa")

    # xformers/sdpa: the string->boolean split now happens once in
    # legacy_import.import_json (against the raw JSON value), not inside
    # derive() -- mirror that here since this harness calls derive()
    # directly rather than going through import_json.
    if "xformers" in raw_values:
        xformers_choice = raw_values["xformers"]
        raw_values["xformers"] = True if xformers_choice == "xformers" else None
        raw_values["sdpa"] = True if xformers_choice == "sdpa" else None

    v2_values = {}
    for spec in FINETUNE_REGISTRY:
        if spec.name in cfg_keys:
            v2_values[spec.name] = raw_values.get(spec.name, spec.default)
        else:
            v2_values[spec.name] = spec.default

    v2_values.update(derive(raw_values, arch_key))

    v2_toml = build_run_config(
        FINETUNE_REGISTRY,
        v2_values,
        arch_key=arch_key,
        training_type="finetune",
        zero_survives_false=True,
    )

    diff_keys = set(old_toml.keys()) ^ set(v2_toml.keys())
    shared_keys = set(old_toml.keys()) & set(v2_toml.keys())

    def _numeric_string_quirk(old_val, new_val):
        try:
            return float(old_val) == float(new_val)
        except (TypeError, ValueError):
            return False

    mismatched = {
        k
        for k in shared_keys
        if old_toml[k] != v2_toml[k]
        and k not in COSMETIC_ALLOWLIST
        and not _numeric_string_quirk(old_toml[k], v2_toml[k])
    }

    v2_only = set(v2_toml.keys()) - set(old_toml.keys())
    registry_names = set(FINETUNE_REGISTRY.names())
    v2_only_but_at_default = {
        k
        for k in v2_only
        if k in registry_names and v2_toml[k] == FINETUNE_REGISTRY[k].default
    }

    old_only = set(old_toml.keys()) - set(v2_toml.keys())
    old_only_known_defect = old_only & KNOWN_DEFECT_OLD_GUI_ONLY

    unexplained_missing_or_extra = (
        diff_keys - COSMETIC_ALLOWLIST - v2_only_but_at_default - old_only_known_defect
    )
    unexplained_mismatched = mismatched

    if unexplained_missing_or_extra or unexplained_mismatched:
        details = []
        for k in sorted(unexplained_missing_or_extra):
            details.append(
                f"  {k}: old={old_toml.get(k, '<absent>')!r} v2={v2_toml.get(k, '<absent>')!r}"
            )
        for k in sorted(unexplained_mismatched):
            details.append(f"  {k}: old={old_toml[k]!r} v2={v2_toml[k]!r}")
        pytest.fail(
            f"{len(details)} unexplained diff(s) for {os.path.basename(preset_path)} "
            f"(arch={arch_key}):\n" + "\n".join(details)
        )
