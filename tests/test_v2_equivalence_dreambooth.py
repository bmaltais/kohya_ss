"""Move 7 (checkpoint B9, DreamBooth portion): equivalence harness -- old
GUI vs kohya_gui_v2, same shape as test_v2_equivalence_lora.py (B8).

For each preset JSON, run it through the old GUI's train_model(print_only=True)
to capture the real run TOML, then run the same values through v2's field
registry + derivation + build_run_config, and diff the two dicts.
"""

import glob
import os

import pytest

from conftest import (
    build_train_model_kwargs,
    run_train_model_and_load_toml,
)
from kohya_gui import dreambooth_gui
from kohya_gui_v2.config_io import build_run_config
from kohya_gui_v2.tabs.dreambooth_fields import DREAMBOOTH_REGISTRY, derive

PRESET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "presets", "dreambooth"
)
PRESETS = sorted(glob.glob(os.path.join(PRESET_DIR, "*.json")))

# Tier 1: cosmetic, structural differences that are correct-by-design.
COSMETIC_ALLOWLIST = {
    "max_data_loader_n_workers",
    "output_dir",
    "output_name",
    "sample_prompts",
    "wandb_run_name",
    # persistent_data_loader_workers is a real store_true boolean in every
    # DreamBooth-family parser (sd-scripts/library/args.py:352), but the old
    # GUI's widget is a Number with an explicit int() cast
    # (arch-matrix-dreambooth.md), so old TOML holds a literal 0/1 while v2's
    # FieldSpec (correctly typed as Widget.CHECKBOX from real parser
    # introspection) treats a False widget value as a real boolean and omits
    # it -- both are behaviorally False/absent to the trainer's store_true
    # default; the int-vs-omitted encoding is cosmetic only.
    "persistent_data_loader_workers",
    # fp8_base/train_inpainting/huber_scale/max_grad_norm are absent from
    # both dreambooth preset fixtures entirely (not GUI-vs-v2 behavior --
    # build_train_model_kwargs' generic "default missing numeric-ish params
    # to 0" fallback doesn't match these fields' real widget defaults, e.g.
    # huber_scale/max_grad_norm default to 1.0 in the real GUI). Old GUI's
    # identity-based filter faithfully preserves that synthetic 0; v2 uses
    # the FieldSpec's real default instead. A test-harness fixture artifact,
    # not a genuine equivalence concern -- would not occur with a real
    # Gradio session where every widget always has its true default value.
    "fp8_base",
    "train_inpainting",
    "huber_scale",
    "max_grad_norm",
}

# GUI kwarg name -> v2 FieldSpec (TOML key) name.
RENAME_MAP = {
    "optimizer": "optimizer_type",
    "max_resolution": "resolution",
    "sd3_text_encoder_batch_size": "text_encoder_batch_size",
}

# Move 7 known-defect register (tier 2): old GUI provably emits a key no
# DreamBooth-family sd-scripts parser accepts (Move 4 gap analysis,
# wargame/reference/gap-analysis-dreambooth.md).
KNOWN_DEFECT_OLD_GUI_ONLY = {
    "epoch",
    "noise_offset_type",
    "disable_mmap_load_safetensors",  # SD1.x/2.x-only defect
    "fused_optimizer_groups",  # SD1.x/2.x-only defect
    # SDXL/SD3/FLUX.1-only defects: these three architectures reuse
    # Finetune's entry script, which lacks these DreamBooth-specific
    # concepts (gap-analysis-dreambooth.md summary). Old GUI emits them
    # universally; only train_db.py's parser actually declares them.
    "prior_loss_weight",
    "stop_text_encoder_training",
    "no_token_padding",
    # learning_rate_te: matrix-documented "SD1.x/2.x only (not sdxl)" gating
    # means old GUI still writes it for sd3/flux1 (only sdxl nulls it), but
    # neither sd3_train.py nor flux_train.py declare this arg -- a genuine
    # defect for those two archs, confirmed via gap-analysis-dreambooth.md.
    "learning_rate_te",
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
    "stop_text_encoder_training",
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
    "reg_data_dir",
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
)


def _architecture_for(kwargs: dict) -> str:
    if kwargs.get("sdxl"):
        return "sdxl"
    if kwargs.get("sd3_checkbox"):
        return "sd3"
    if kwargs.get("flux1_checkbox"):
        return "flux1"
    return "sd15"


@pytest.mark.parametrize(
    "preset_path", PRESETS, ids=[os.path.basename(p) for p in PRESETS]
)
def test_dreambooth_equivalence(preset_path):
    kwargs = build_train_model_kwargs(
        dreambooth_gui.train_model,
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
    ):
        if not kwargs.get(key):
            kwargs[key] = default

    if kwargs.get("pretrained_model_name_or_path") and not os.path.exists(
        kwargs["pretrained_model_name_or_path"]
    ):
        kwargs["pretrained_model_name_or_path"] = ""

    kwargs.setdefault("train_data_dir", "")
    if not os.path.isdir(kwargs["train_data_dir"]):
        kwargs["train_data_dir"] = ""
    if not kwargs["train_data_dir"]:
        kwargs["train_data_dir"] = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "test", "img"
        )

    old_toml = run_train_model_and_load_toml(dreambooth_gui, kwargs)

    arch_key = _architecture_for(kwargs)

    raw_values = dict(kwargs)
    for gui_name, v2_name in RENAME_MAP.items():
        if gui_name in raw_values:
            raw_values[v2_name] = raw_values[gui_name]

    # Keys actually present in the source preset JSON (plus their renamed
    # v2 counterparts). A FieldSpec absent from the real preset must default
    # to the FieldSpec's own default -- not to build_train_model_kwargs'
    # synthetic 0/False fill for missing train_model() positional params,
    # which would otherwise leak a fixture artifact into a gap-added (v2-only)
    # field and falsely perturb equivalence (Move 7's invariant test).
    with open(preset_path, encoding="utf-8") as f:
        import json as _json

        cfg_keys = set(_json.load(f).keys())
    for gui_name, v2_name in RENAME_MAP.items():
        if gui_name in cfg_keys:
            cfg_keys.add(v2_name)

    v2_values = {}
    for spec in DREAMBOOTH_REGISTRY:
        if spec.name in cfg_keys:
            v2_values[spec.name] = raw_values.get(spec.name, spec.default)
        else:
            v2_values[spec.name] = spec.default

    v2_values.update(derive(raw_values, arch_key))

    v2_toml = build_run_config(
        DREAMBOOTH_REGISTRY,
        v2_values,
        arch_key=arch_key,
        training_type="dreambooth",
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
    registry_names = set(DREAMBOOTH_REGISTRY.names())
    v2_only_but_at_default = {
        k
        for k in v2_only
        if k in registry_names and v2_toml[k] == DREAMBOOTH_REGISTRY[k].default
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
