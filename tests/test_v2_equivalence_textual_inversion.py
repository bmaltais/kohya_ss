"""Move 7 (checkpoint B9, Textual Inversion portion): equivalence harness --
old GUI vs kohya_gui_v2, same shape as test_v2_equivalence_lora.py (B8).

No preset corpus exists for Textual Inversion; this harness uses two
committed hand-built fixtures (tests/fixtures/v2/textual_inversion/, copied
from the pre-existing test/config/TI-AdamW8bit*.json fixtures used by
test_textual_inversion_gui.py) covering sd_v1v2 and sdxl.
"""

import glob
import json
import os

import pytest

from conftest import (
    build_train_model_kwargs,
    run_train_model_and_load_toml,
)
from kohya_gui import textual_inversion_gui
from kohya_gui_v2.config_io import build_run_config
from kohya_gui_v2.tabs.textual_inversion_fields import (
    TEXTUAL_INVERSION_REGISTRY,
    derive,
)

FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__), "fixtures", "v2", "textual_inversion"
)
FIXTURES = sorted(glob.glob(os.path.join(FIXTURE_DIR, "*.json")))

COSMETIC_ALLOWLIST = {
    "max_data_loader_n_workers",
    "output_dir",
    "output_name",
    "sample_prompts",
    "wandb_run_name",
}

RENAME_MAP = {
    "optimizer": "optimizer_type",
    "max_resolution": "resolution",
}

# Move 7 known-defect register (tier 2): old GUI provably emits a key no
# TI-family sd-scripts parser accepts (Move 4 gap analysis,
# wargame/reference/gap-analysis-textual_inversion.md). Also includes the
# seeded finding: stop_text_encoder_training_pct is preserved in the JSON
# round-trip but neither trainer script accepts it (gui_only=True in the
# registry; see textual_inversion_fields.py) -- it never appears in either
# side's run TOML, so it doesn't actually surface as a diff here.
KNOWN_DEFECT_OLD_GUI_ONLY = {
    "epoch",  # dead across both archs; only max_train_epochs is real
    "noise_offset_type",  # dead across both archs
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
    "num_vectors_per_token",
    "stop_text_encoder_training_pct",
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
    "wandb_run_name",
    "output_name",
    "sample_prompts",
    "output_dir",
    "train_data_dir",
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
    "lr_scheduler_type",
    "dynamo_backend",
    "dynamo_mode",
    "extra_accelerate_launch_args",
    "weights",
    "token_string",
    "init_word",
    "template",
)


def _architecture_for(kwargs: dict) -> str:
    if kwargs.get("sdxl"):
        return "sdxl"
    return "sd_v1v2"


@pytest.mark.parametrize(
    "fixture_path", FIXTURES, ids=[os.path.basename(p) for p in FIXTURES]
)
def test_textual_inversion_equivalence(fixture_path):
    kwargs = build_train_model_kwargs(
        textual_inversion_gui.train_model,
        fixture_path,
        numeric_fixups=NUMERIC_FIXUPS,
        string_overrides=STRING_OVERRIDES,
    )
    # JSON `null` for a string-typed field is preserved as-is by
    # build_train_model_kwargs (it only defaults truly-MISSING keys via
    # string_overrides); some fixtures store an explicit null for an unset
    # optional textbox, which crashes old-GUI string methods (e.g.
    # `.replace()` on extra_accelerate_launch_args). Old GUI's own Gradio
    # widgets never actually emit None for a Textbox, so this is a fixture
    # encoding quirk to normalize here, not a real equivalence concern.
    for key in STRING_OVERRIDES:
        if kwargs.get(key) is None:
            kwargs[key] = ""

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

    if not kwargs.get("token_string"):
        kwargs["token_string"] = "mytoken"
    if not kwargs.get("init_word"):
        kwargs["init_word"] = "*"

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

    old_toml = run_train_model_and_load_toml(textual_inversion_gui, kwargs)

    arch_key = _architecture_for(kwargs)

    raw_values = dict(kwargs)
    for gui_name, v2_name in RENAME_MAP.items():
        if gui_name in raw_values:
            raw_values[v2_name] = raw_values[gui_name]

    with open(fixture_path, encoding="utf-8") as f:
        cfg_keys = set(json.load(f).keys())
    for gui_name, v2_name in RENAME_MAP.items():
        if gui_name in cfg_keys:
            cfg_keys.add(v2_name)
    cfg_keys.add("train_data_dir")
    cfg_keys.add("token_string")
    cfg_keys.add("init_word")

    v2_values = {}
    for spec in TEXTUAL_INVERSION_REGISTRY:
        if spec.name in cfg_keys:
            v2_values[spec.name] = raw_values.get(spec.name, spec.default)
        else:
            v2_values[spec.name] = spec.default

    v2_values.update(derive(raw_values, arch_key))

    v2_toml = build_run_config(
        TEXTUAL_INVERSION_REGISTRY,
        v2_values,
        arch_key=arch_key,
        training_type="textual_inversion",
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
    registry_names = set(TEXTUAL_INVERSION_REGISTRY.names())
    v2_only_but_at_default = {
        k
        for k in v2_only
        if k in registry_names and v2_toml[k] == TEXTUAL_INVERSION_REGISTRY[k].default
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
            f"{len(details)} unexplained diff(s) for {os.path.basename(fixture_path)} "
            f"(arch={arch_key}):\n" + "\n".join(details)
        )
