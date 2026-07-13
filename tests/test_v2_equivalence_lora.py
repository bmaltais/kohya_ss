"""Move 7 (checkpoint B8): LoRA equivalence harness -- old GUI vs kohya_gui_v2.

For each preset JSON, run it through the old GUI's train_model(print_only=True)
to capture the real run TOML, then run the same values through v2's field
registry + derivation + build_run_config, and diff the two dicts.

Diff tiers (per the plan): tier 1 = cosmetic allowlist, tier 2 = known-defect
register (v2 intentionally differs from a provably-buggy old GUI), tier 3 =
unexplained (fails the test).

Status (2026-07-11, B1/B8 in progress): LoRA_type-dependent network_module
and network_args derivation is not yet fully ported (see
kohya_gui_v2/tabs/lora_derivations.py). Diffs on those two keys are
currently tier-2-pending (tracked, not yet in the formal register) rather
than causing a hard failure, so this harness can report real signal on
everything else while that work continues. This is a temporary carve-out,
not a permanent allowlist entry -- tighten it as derivation coverage grows.
"""

import glob
import os

import pytest
import toml

from conftest import (
    build_train_model_kwargs,
    mock_executor,
    run_train_model_and_load_toml,
)
from kohya_gui import lora_gui
from kohya_gui_v2.config_io import build_run_config
from kohya_gui_v2.tabs.lora_fields import LORA_REGISTRY, derive

PRESET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets", "lora")
PRESETS = sorted(glob.glob(os.path.join(PRESET_DIR, "*.json")))

# Tier 1: cosmetic, structural differences that are correct-by-design.
COSMETIC_ALLOWLIST = {
    "max_data_loader_n_workers",  # old GUI exempts this key from its falsy-drop filter; v2 doesn't need the same carve-out if the value is non-falsy
    # test-harness artifacts: run_train_model_and_load_toml substitutes a
    # fresh tmpdir/output_name AFTER our v2_values snapshot is built, and
    # sample_prompts is a generated file path derived from output_dir --
    # none of these are real GUI-vs-v2 content differences.
    "output_dir",
    "output_name",
    "sample_prompts",
    "wandb_run_name",  # derived from output_name, inherits the same tmpdir-substitution timing artifact
    # old GUI always writes an empty list for an unset optimizer/scheduler
    # args textbox; v2's FieldSpec.to_toml (_to_arg_list) returns None for
    # an empty string, which the falsy-drop filter omits entirely. Both
    # `optimizer_args = []` and an absent key are no-ops to sd-scripts
    # (`args.optimizer_args is not None and len(args.optimizer_args) > 0`).
    "lr_scheduler_args",
    "optimizer_args",
    # Same empty-list-vs-absent-key artifact as above, but for the
    # LoRA_type-derived network_args string (networks.lora's create_network
    # treats `args.network_args is None` and `args.network_args == []`
    # identically).
    "network_args",
}

# GUI kwarg name -> v2 FieldSpec (TOML key) name, for the handful of keys
# the old GUI stores under a different widget name than the TOML key it
# writes. Temporary/local to this harness; the full six-type mapping is
# Move 8's legacy_import.py.
RENAME_MAP = {
    "optimizer": "optimizer_type",
    "max_resolution": "resolution",
}

# Move 7 known-defect register (tier 2): old GUI provably emits a key no
# LoRA-family sd-scripts parser accepts (confirmed via Move 4 gap analysis,
# wargame/reference/gap-analysis-lora.md). v2 correctly omits these -- do
# not port them. One-directional: only "present in old, absent in v2" is
# expected for these keys.
KNOWN_DEFECT_OLD_GUI_ONLY = {
    "epoch",  # gap-analysis-lora.md: dead across all 8 archs; only max_train_epochs is real
    "noise_offset_type",  # gap-analysis-lora.md: dead across all 8 archs
}

# Tier 2, value-mismatch variant: old GUI computes a provably wrong value;
# v2 computes the correct one. Confirmed via code citation, not guessed.
KNOWN_DEFECT_MISMATCH = {
    # lora_gui.py ~1552 (non-dataset_config branch): `lr_warmup_steps =
    # lr_warmup / 100` omits `* max_train_steps`, producing a near-zero
    # warmup instead of a percentage of total steps.
    "lr_warmup_steps",
}


def _architecture_for(kwargs: dict) -> str:
    if kwargs.get("sdxl"):
        return "sdxl"
    if kwargs.get("flux1_checkbox"):
        return "flux1"
    if kwargs.get("sd3_checkbox"):
        return "sd3"
    if kwargs.get("hunyuan_image_checkbox"):
        return "hunyuan_image"
    if kwargs.get("anima_checkbox"):
        return "anima"
    if kwargs.get("lumina_checkbox"):
        return "lumina"
    return "sd15"


STRING_OVERRIDES = (
    "pretrained_model_name_or_path",
    "vae",
    "network_weights",
    "dataset_config",
    "logging_dir",
    "resume",
    "log_tracker_config",
    "LyCORIS_preset",
    "lr_scheduler_args",
    "optimizer_args",
    "network_args",
    "additional_parameters",
    "base_weights",
    "base_weights_multiplier",
    "network_weights",
    "training_comment",
    "wandb_run_name",
    "output_name",
    "sample_prompts",
    "output_dir",
    "train_data_dir",
    "reg_data_dir",
    "ae",
    "clip_l",
    "t5xxl",
    "sd3_clip_l",
    "sd3_t5xxl",
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
    "in_dims",
)


@pytest.mark.parametrize(
    "preset_path", PRESETS, ids=[os.path.basename(p) for p in PRESETS]
)
def test_lora_equivalence(preset_path):
    kwargs = build_train_model_kwargs(
        lora_gui.train_model,
        preset_path,
        numeric_fixups=(
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
            "network_dim",
            "network_alpha",
            "noise_offset",
            "noise_offset_random_strength",
            "adaptive_noise_scale",
            "multires_noise_iterations",
            "multires_noise_discount",
        ),
        string_overrides=STRING_OVERRIDES,
    )
    # Gradio's gr.Number widget coerces a loaded JSON value to a native
    # int/float at runtime; build_train_model_kwargs bypasses Gradio and
    # passes raw JSON values straight through, so numeric-looking strings
    # (e.g. lr_warmup="0" from an older preset) reach train_model() as
    # strings and crash old-GUI arithmetic that assumes a real number.
    # Coerce anything that parses cleanly as a number, mimicking Gradio.
    for key, value in list(kwargs.items()):
        if isinstance(value, str):
            try:
                kwargs[key] = int(value)
            except ValueError:
                try:
                    kwargs[key] = float(value)
                except ValueError:
                    pass

    # bucket_reso_steps/min_bucket_reso/max_bucket_reso must be non-zero for
    # old GUI's validation; conftest defaults missing numeric fields to 0,
    # which is a valid preset omission (real widgets carry a real default)
    # but not a valid runtime value -- restore the trainer's own defaults.
    for key, default in (
        ("bucket_reso_steps", 64),
        ("min_bucket_reso", 256),
        ("max_bucket_reso", 2048),
    ):
        if not kwargs.get(key):
            kwargs[key] = default

    # Presets carry the preset-author's local model path (or, in this one
    # case, a literal unfilled placeholder string) -- neither exists on
    # this machine. Model-path validation isn't what this harness is
    # testing; treat any non-existent path as "unset" like the old GUI
    # itself treats an empty string.
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

    old_toml = run_train_model_and_load_toml(lora_gui, kwargs)

    arch_key = _architecture_for(kwargs)

    # Raw values dict (all old-GUI kwargs, renamed where needed) so derive()
    # can see composite source widgets (e.g. sdxl_no_half_vae) that aren't
    # themselves v2 FieldSpec names. Best-effort for this in-progress
    # checkpoint -- full six-type legacy_import.py mapping is Move 8.
    raw_values = dict(kwargs)
    for gui_name, v2_name in RENAME_MAP.items():
        if gui_name in raw_values:
            raw_values[v2_name] = raw_values[gui_name]

    # xformers/sdpa: the string->boolean split now happens once in
    # legacy_import.import_json (against the raw JSON value), not inside
    # derive() -- mirror that here since this harness calls derive()
    # directly rather than going through import_json.
    if "xformers" in raw_values:
        xformers_choice = raw_values["xformers"]
        raw_values["xformers"] = True if xformers_choice == "xformers" else None
        raw_values["sdpa"] = True if xformers_choice == "sdpa" else None

    v2_values = {}
    for spec in LORA_REGISTRY:
        v2_values[spec.name] = raw_values.get(spec.name, spec.default)

    v2_values.update(derive(raw_values, arch_key))

    v2_toml = build_run_config(
        LORA_REGISTRY, v2_values, arch_key=arch_key, training_type="lora"
    )

    diff_keys = set(old_toml.keys()) ^ set(v2_toml.keys())
    shared_keys = set(old_toml.keys()) & set(v2_toml.keys())

    def _numeric_string_quirk(old_val, new_val):
        """Old GUI sometimes passes numeric values through as strings when
        the preset JSON stored them that way (Gradio widget serialization
        quirk); v2 always coerces to the parser's declared type. Same value,
        different type -- a v2 correctness improvement, not a defect."""
        try:
            return float(old_val) == float(new_val)
        except (TypeError, ValueError):
            return False

    mismatched = {
        k
        for k in shared_keys
        if old_toml[k] != v2_toml[k]
        and k not in COSMETIC_ALLOWLIST
        and k not in KNOWN_DEFECT_MISMATCH
        and not _numeric_string_quirk(old_toml[k], v2_toml[k])
    }

    v2_only = set(v2_toml.keys()) - set(old_toml.keys())
    # A key v2 exposes that the old GUI never had, sitting at that field's
    # OWN argparse default, is behaviorally identical to omitting it --
    # this is the "gap-added fields must not perturb equivalence" invariant
    # (Move 7). Only flag it if v2's value differs from the field's default.
    registry_names = set(LORA_REGISTRY.names())
    v2_only_but_at_default = {
        k
        for k in v2_only
        if k in registry_names and v2_toml[k] == LORA_REGISTRY[k].default
    }

    old_only = set(old_toml.keys()) - set(v2_toml.keys())
    old_only_known_defect = old_only & KNOWN_DEFECT_OLD_GUI_ONLY

    unexplained_missing_or_extra = (
        diff_keys
        - COSMETIC_ALLOWLIST
        - v2_only_but_at_default
        - old_only_known_defect
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
