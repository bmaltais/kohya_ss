"""Legacy JSON presets must open/save without Gradio dropdown rejections.

Regression for dynamo_backend=\"no\" and max_token_length=75 (and stringified
\"75\") which the v2 FieldSpecs originally omitted from choices even though
every legacy preset and the v1 GUI use them.
"""

import json
import os
import tempfile

import pytest

from kohya_gui_v2.config_io import load_config, save_config
from kohya_gui_v2.fields import FieldSpec, Widget
from kohya_gui_v2.legacy_import import REGISTRIES, import_json
from kohya_gui_v2.tabs.lora_fields import LORA_REGISTRY

PRESETS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets")


def test_normalize_widget_value_soft_matches_stringified_int():
    spec = FieldSpec(
        name="max_token_length",
        widget=Widget.DROPDOWN,
        default=75,
        choices=[75, 150, 225],
        from_toml=lambda v: int(v) if v not in (None, "") else v,
    )
    assert spec.normalize_widget_value("75") == 75
    assert spec.normalize_widget_value(150) == 150
    assert spec.normalize_widget_value("999") == 75  # clamp to default


def test_normalize_widget_value_accepts_legacy_dynamo_no():
    spec = FieldSpec(
        name="dynamo_backend",
        widget=Widget.DROPDOWN,
        default="no",
        choices=["no", "eager", "inductor"],
    )
    assert spec.normalize_widget_value("no") == "no"
    assert spec.normalize_widget_value("inductor") == "inductor"
    assert spec.normalize_widget_value("not_a_backend") == "no"


def test_lora_registry_includes_legacy_dropdown_values():
    dynamo = LORA_REGISTRY["dynamo_backend"]
    assert dynamo.default == "no"
    assert "no" in dynamo.choices
    assert "tensort" not in dynamo.choices
    assert "tensorrt" in dynamo.choices

    token = LORA_REGISTRY["max_token_length"]
    assert token.default == 75
    assert 75 in token.choices
    assert token.choices == [75, 150, 225]


@pytest.mark.parametrize(
    "preset_rel",
    [
        os.path.join("lora", "flux1D - adamw8bit fp8.json"),
        os.path.join("lora", "loha-sd15.json"),
    ],
)
def test_import_legacy_json_preserves_no_and_75(preset_rel):
    path = os.path.join(PRESETS_ROOT, preset_rel)
    result = import_json(path, training_type="lora")
    assert result.values["dynamo_backend"] == "no"
    # flux preset uses int 75; some older presets use string \"75\"
    assert result.values["max_token_length"] == 75
    # values must be legal Gradio Dropdown choices
    assert result.values["dynamo_backend"] in LORA_REGISTRY["dynamo_backend"].choices
    assert (
        result.values["max_token_length"] in LORA_REGISTRY["max_token_length"].choices
    )


def test_import_save_load_preserves_dropdown_values_for_gradio():
    """Open legacy JSON → save TOML → reload must keep values Gradio accepts."""
    path = os.path.join(PRESETS_ROOT, "lora", "flux1D - adamw8bit fp8.json")
    result = import_json(path, training_type="lora")
    registry = REGISTRIES["lora"]
    values = dict(result.values)
    values["architecture"] = result.architecture

    with tempfile.TemporaryDirectory() as tmp:
        toml_path = os.path.join(tmp, "out.toml")
        save_config(registry, values, toml_path)
        reloaded = load_config(registry, toml_path)

    dynamo = registry["dynamo_backend"]
    token = registry["max_token_length"]
    assert reloaded["dynamo_backend"] in dynamo.choices
    assert reloaded["max_token_length"] in token.choices
    # defaults are dropped on save; after reload they reappear as defaults
    assert reloaded["dynamo_backend"] == "no"
    assert reloaded["max_token_length"] == 75


def test_toml_with_explicit_legacy_values_loads():
    registry = LORA_REGISTRY
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "legacy.toml")
        with open(path, "w", encoding="utf-8") as f:
            f.write('dynamo_backend = "no"\nmax_token_length = 75\n')
        loaded = load_config(registry, path)
    assert loaded["dynamo_backend"] == "no"
    assert loaded["max_token_length"] == 75
