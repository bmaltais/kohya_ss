"""Unit tests for kohya_gui_v2.config_io (Move 3 core framework)."""

import os
import tempfile

from kohya_gui_v2.config_io import build_run_config, load_config, save_config
from kohya_gui_v2.fields import FieldRegistry, FieldSpec, Widget


def _synthetic_registry():
    return FieldRegistry(
        [
            FieldSpec(
                name="architecture",
                widget=Widget.DROPDOWN,
                default="sdxl",
                gui_only=True,
            ),
            FieldSpec(name="output_dir", widget=Widget.TEXTBOX, default=""),
            FieldSpec(name="learning_rate", widget=Widget.NUMBER, default=1e-4),
            FieldSpec(
                name="guidance_scale",
                widget=Widget.NUMBER,
                default=1.0,
                keep_if_falsy=True,
                archs=frozenset({"flux1"}),
            ),
            FieldSpec(
                name="network_dim",
                widget=Widget.NUMBER,
                default=8,
                training_types=frozenset({"lora"}),
            ),
            FieldSpec(name="cache_latents", widget=Widget.CHECKBOX, default=False),
        ]
    )


def test_save_load_idempotence():
    registry = _synthetic_registry()
    values = {
        "architecture": "flux1",
        "output_dir": "/tmp/out",
        "learning_rate": 5e-5,
        "guidance_scale": 0.0,
        "network_dim": 16,
        "cache_latents": True,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.toml")
        save_config(registry, values, path)
        loaded = load_config(registry, path)
        assert loaded == values

        # second save from the loaded values must be byte-identical
        path2 = os.path.join(tmpdir, "config2.toml")
        save_config(registry, loaded, path2)
        with open(path, encoding="utf-8") as f1, open(path2, encoding="utf-8") as f2:
            assert f1.read() == f2.read()


def test_load_missing_keys_fall_back_to_defaults():
    registry = _synthetic_registry()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.toml")
        save_config(registry, {"output_dir": "/x"}, path)
        loaded = load_config(registry, path)
        assert loaded["learning_rate"] == 1e-4  # default, not in file
        assert loaded["output_dir"] == "/x"


def test_sectioned_toml_equals_flat_toml():
    registry = _synthetic_registry()
    with tempfile.TemporaryDirectory() as tmpdir:
        flat_path = os.path.join(tmpdir, "flat.toml")
        with open(flat_path, "w", encoding="utf-8") as f:
            f.write('output_dir = "/tmp/out"\nlearning_rate = 5e-5\n')

        sectioned_path = os.path.join(tmpdir, "sectioned.toml")
        with open(sectioned_path, "w", encoding="utf-8") as f:
            f.write('[training]\noutput_dir = "/tmp/out"\nlearning_rate = 5e-5\n')

        assert load_config(registry, flat_path) == load_config(registry, sectioned_path)


def test_keep_if_falsy_flag_honored_in_run_config():
    registry = _synthetic_registry()
    values = {
        "architecture": "flux1",
        "output_dir": "",  # falsy, not keep_if_falsy -> dropped
        "learning_rate": 1e-4,
        "guidance_scale": 0.0,  # falsy, keep_if_falsy=True -> kept
        "network_dim": 8,
        "cache_latents": False,  # falsy, not keep_if_falsy -> dropped
    }
    run_config = build_run_config(
        registry, values, arch_key="flux1", training_type="lora"
    )
    assert "output_dir" not in run_config
    assert "cache_latents" not in run_config
    assert run_config["guidance_scale"] == 0.0
    assert "architecture" not in run_config  # gui_only, never in run config


def test_build_run_config_filters_by_arch_and_training_type():
    registry = _synthetic_registry()
    values = {
        "architecture": "sdxl",
        "output_dir": "/tmp/out",
        "learning_rate": 1e-4,
        "guidance_scale": 3.5,  # flux1-only, not applicable to sdxl
        "network_dim": 8,
        "cache_latents": True,
    }
    run_config = build_run_config(
        registry, values, arch_key="sdxl", training_type="dreambooth"
    )
    assert "guidance_scale" not in run_config  # arch mismatch
    assert "network_dim" not in run_config  # training_type mismatch (lora-only)
    assert run_config["output_dir"] == "/tmp/out"
