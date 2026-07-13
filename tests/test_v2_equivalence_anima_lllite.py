"""Move 7 (checkpoint B9, Anima LLLite portion): equivalence harness -- old
GUI vs kohya_gui_v2, same shape as test_v2_equivalence_lora.py (B8).

No preset corpus exists for Anima LLLite; this harness uses two committed
hand-built fixtures (tests/fixtures/v2/anima_lllite/) generated via
anima_lllite_gui.save_configuration. Single architecture (no arch dropdown
concept), per arch-matrix-anima_lllite.md #1.
"""

import glob
import json
import os

import pytest

from conftest import (
    build_train_model_kwargs,
    run_train_model_and_load_toml,
)
from kohya_gui import anima_lllite_gui
from kohya_gui_v2.config_io import build_run_config
from kohya_gui_v2.tabs.anima_lllite_fields import ANIMA_LLLITE_REGISTRY, derive

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "v2", "anima_lllite")
FIXTURES = sorted(glob.glob(os.path.join(FIXTURE_DIR, "*.json")))

ARCH_KEY = "anima_lllite"

COSMETIC_ALLOWLIST = {
    "output_dir",
    "output_name",
    "wandb_run_name",
    # old GUI always writes an empty list for an unset optimizer/scheduler
    # args textbox; v2's FieldSpec.to_toml (_to_arg_list) returns None for
    # an empty string, which the falsy-drop filter omits entirely. Both
    # `optimizer_args = []` and an absent key are no-ops to sd-scripts
    # (`args.optimizer_args is not None and len(args.optimizer_args) > 0`).
    "lr_scheduler_args",
    "optimizer_args",
}

RENAME_MAP = {
    "optimizer": "optimizer_type",
    "anima_qwen3": "qwen3",
    "anima_vae": "vae",
    "anima_llm_adapter_path": "llm_adapter_path",
    "anima_t5_tokenizer_path": "t5_tokenizer_path",
    "anima_discrete_flow_shift": "discrete_flow_shift",
    "anima_timestep_sampling": "timestep_sampling",
    "anima_sigmoid_scale": "sigmoid_scale",
    "anima_qwen3_max_token_length": "qwen3_max_token_length",
    "anima_t5_max_token_length": "t5_max_token_length",
    "anima_attn_mode": "attn_mode",
    "anima_split_attn": "split_attn",
    "anima_vae_chunk_size": "vae_chunk_size",
    "anima_vae_disable_cache": "vae_disable_cache",
    "anima_qwen_image_vae_2d": "qwen_image_vae_2d",
    "anima_compile": "compile",
    "anima_torch_compile": "torch_compile",
    "anima_compile_backend": "compile_backend",
    "anima_compile_mode": "compile_mode",
    "anima_compile_dynamic": "compile_dynamic",
    "anima_compile_fullgraph": "compile_fullgraph",
    "anima_compile_cache_size_limit": "compile_cache_size_limit",
}

KNOWN_DEFECT_OLD_GUI_ONLY: set = set()

STRING_OVERRIDES = (
    "pretrained_model_name_or_path",
    "train_data_dir",
    "conditioning_data_dir",
    "dataset_config",
    "anima_qwen3",
    "anima_vae",
    "anima_llm_adapter_path",
    "anima_t5_tokenizer_path",
    "anima_timestep_sampling",
    "anima_attn_mode",
    "anima_compile_backend",
    "anima_compile_mode",
    "anima_compile_dynamic",
    "lllite_target_layers",
    "network_weights",
    "optimizer_args",
    "lr_scheduler_args",
    "resume",
    "logging_dir",
    "log_with",
    "log_tracker_name",
    "log_tracker_config",
    "wandb_api_key",
    "wandb_run_name",
    "output_name",
    "output_dir",
    "show_timesteps",
    "show_timesteps_resolution",
    "dynamo_backend",
    "dynamo_mode",
    "extra_accelerate_launch_args",
    "gpu_ids",
)

NUMERIC_FIXUPS = (
    "min_bucket_reso",
    "max_bucket_reso",
    "train_batch_size",
    "max_train_epochs",
    "max_train_steps",
    "seed",
    "gradient_accumulation_steps",
    "anima_discrete_flow_shift",
    "anima_sigmoid_scale",
    "anima_qwen3_max_token_length",
    "anima_t5_max_token_length",
    "anima_vae_chunk_size",
    "anima_compile_cache_size_limit",
    "cond_emb_dim",
    "lllite_mlp_dim",
    "lllite_cond_dim",
    "lllite_cond_resblocks",
    "lllite_dropout",
    "lllite_multiplier",
    "lllite_cond_in_channels",
    "learning_rate",
    "lr_warmup_steps",
    "lr_scheduler_num_cycles",
    "lr_scheduler_power",
    "max_grad_norm",
    "num_cpu_threads_per_process",
    "num_processes",
    "num_machines",
    "main_process_port",
    "save_every_n_epochs",
    "save_every_n_steps",
    "save_last_n_steps",
    "save_last_n_steps_state",
)


@pytest.mark.parametrize(
    "fixture_path", FIXTURES, ids=[os.path.basename(p) for p in FIXTURES]
)
def test_anima_lllite_equivalence(fixture_path):
    kwargs = build_train_model_kwargs(
        anima_lllite_gui.train_model,
        fixture_path,
        numeric_fixups=NUMERIC_FIXUPS,
        string_overrides=STRING_OVERRIDES,
    )

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

    if kwargs.get("pretrained_model_name_or_path") and not os.path.exists(
        kwargs["pretrained_model_name_or_path"]
    ):
        kwargs["pretrained_model_name_or_path"] = ""
    if kwargs.get("anima_llm_adapter_path") and not os.path.exists(
        str(kwargs["anima_llm_adapter_path"])
    ):
        kwargs["anima_llm_adapter_path"] = ""
    # anima_qwen3/anima_vae are hard-required (train_model rejects an empty
    # value outright, unlike the optional pretrained_model_name_or_path);
    # substitute a fake HF-style repo id (matches validate_model_path's
    # `[\w-]+/[\w-]+` pattern, which skips the local-existence check) when
    # the preset's real local path doesn't exist on this machine.
    for model_key in ("anima_qwen3", "anima_vae"):
        if not kwargs.get(model_key) or not os.path.exists(str(kwargs[model_key])):
            kwargs[model_key] = "test-org/test-model"
    if kwargs.get("anima_t5_tokenizer_path") and not os.path.isdir(
        str(kwargs["anima_t5_tokenizer_path"])
    ):
        kwargs["anima_t5_tokenizer_path"] = ""

    kwargs.setdefault("train_data_dir", "")
    if not os.path.isdir(str(kwargs["train_data_dir"])):
        kwargs["train_data_dir"] = ""
    if not kwargs["train_data_dir"] and not kwargs.get("dataset_config"):
        kwargs["train_data_dir"] = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "test", "img"
        )
    if not os.path.isdir(str(kwargs.get("conditioning_data_dir") or "")):
        kwargs["conditioning_data_dir"] = kwargs["train_data_dir"]

    old_toml = run_train_model_and_load_toml(anima_lllite_gui, kwargs)

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
    cfg_keys.add("conditioning_data_dir")

    v2_values = {}
    for spec in ANIMA_LLLITE_REGISTRY:
        if spec.name in cfg_keys:
            v2_values[spec.name] = raw_values.get(spec.name, spec.default)
        else:
            v2_values[spec.name] = spec.default

    v2_values.update(derive(raw_values, ARCH_KEY))

    v2_toml = build_run_config(
        ANIMA_LLLITE_REGISTRY,
        v2_values,
        arch_key=ARCH_KEY,
        training_type="anima_lllite",
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
    registry_names = set(ANIMA_LLLITE_REGISTRY.names())
    v2_only_but_at_default = {
        k
        for k in v2_only
        if k in registry_names and v2_toml[k] == ANIMA_LLLITE_REGISTRY[k].default
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
            f"{len(details)} unexplained diff(s) for {os.path.basename(fixture_path)}:\n"
            + "\n".join(details)
        )
