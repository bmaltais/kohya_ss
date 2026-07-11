"""Move 7 (checkpoint B9, LeCo portion): equivalence harness -- old GUI vs
kohya_gui_v2, same shape as test_v2_equivalence_lora.py (B8).

No preset corpus exists for LeCo; this harness uses two committed
hand-built fixtures (tests/fixtures/v2/leco/) generated via
leco_gui.save_configuration, covering sd15 and sdxl.
"""

import glob
import json
import os

import pytest

from conftest import (
    build_train_model_kwargs,
    run_train_model_and_load_toml,
)
from kohya_gui import leco_gui
from kohya_gui_v2.config_io import build_run_config
from kohya_gui_v2.tabs.leco_fields import LECO_REGISTRY, derive

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "v2", "leco")
FIXTURES = sorted(glob.glob(os.path.join(FIXTURE_DIR, "*.json")))

PROMPTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "test", "config", "leco_prompts.toml"
)

COSMETIC_ALLOWLIST = {
    "output_dir",
    "output_name",
    "wandb_run_name",
}

RENAME_MAP = {
    "optimizer": "optimizer_type",
}

KNOWN_DEFECT_OLD_GUI_ONLY: set = set()

STRING_OVERRIDES = (
    "pretrained_model_name_or_path",
    "network_weights",
    "training_comment",
    "wandb_run_name",
    "output_name",
    "output_dir",
    "resume",
    "logging_dir",
    "log_with",
    "log_tracker_name",
    "log_tracker_config",
    "wandb_api_key",
    "network_args",
    "optimizer_args",
    "lr_scheduler_args",
    "dynamo_backend",
    "dynamo_mode",
    "extra_accelerate_launch_args",
    "gpu_ids",
    "network_module",
)

NUMERIC_FIXUPS = (
    "network_dim",
    "network_alpha",
    "network_dropout",
    "unet_lr",
    "lr_warmup_steps",
    "lr_scheduler_num_cycles",
    "lr_scheduler_power",
    "max_train_steps",
    "max_grad_norm",
    "max_denoising_steps",
    "leco_denoise_guidance_scale",
    "seed",
    "gradient_accumulation_steps",
    "clip_skip",
    "noise_offset",
    "min_snr_gamma",
    "save_every_n_steps",
    "save_last_n_steps",
    "save_last_n_steps_state",
    "num_cpu_threads_per_process",
    "num_processes",
    "num_machines",
    "main_process_port",
)


def _architecture_for(kwargs: dict) -> str:
    return "sdxl" if kwargs.get("sdxl") else "sd15"


@pytest.mark.parametrize(
    "fixture_path", FIXTURES, ids=[os.path.basename(p) for p in FIXTURES]
)
def test_leco_equivalence(fixture_path):
    kwargs = build_train_model_kwargs(
        leco_gui.train_model,
        fixture_path,
        numeric_fixups=NUMERIC_FIXUPS,
        string_overrides=STRING_OVERRIDES,
    )

    for key in STRING_OVERRIDES:
        if kwargs.get(key) is None:
            kwargs[key] = ""

    for key, value in list(kwargs.items()):
        if isinstance(value, str) and key not in ("network_module",):
            try:
                kwargs[key] = int(value)
            except ValueError:
                try:
                    kwargs[key] = float(value)
                except ValueError:
                    pass

    # prompts_file must point at a real, valid TOML file -- leco_gui.py's
    # own validate_toml_file() guard rejects anything else before the TOML
    # is even written.
    kwargs["prompts_file"] = PROMPTS_FILE

    old_toml = run_train_model_and_load_toml(leco_gui, kwargs)

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
    cfg_keys.add("prompts_file")

    v2_values = {}
    for spec in LECO_REGISTRY:
        if spec.name in cfg_keys:
            v2_values[spec.name] = raw_values.get(spec.name, spec.default)
        else:
            v2_values[spec.name] = spec.default

    v2_values.update(derive(raw_values, arch_key))

    v2_toml = build_run_config(
        LECO_REGISTRY, v2_values, arch_key=arch_key, training_type="leco"
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
    registry_names = set(LECO_REGISTRY.names())
    v2_only_but_at_default = {
        k
        for k in v2_only
        if k in registry_names and v2_toml[k] == LECO_REGISTRY[k].default
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
