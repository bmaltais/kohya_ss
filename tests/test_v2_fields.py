"""Unit tests for kohya_gui_v2.fields (Move 3 core framework)."""

import pytest

from kohya_gui_v2.fields import FieldRegistry, FieldSpec, Widget


def test_duplicate_names_rejected():
    specs = [
        FieldSpec(name="learning_rate", widget=Widget.NUMBER, default=1e-4),
        FieldSpec(name="learning_rate", widget=Widget.NUMBER, default=2e-4),
    ]
    with pytest.raises(ValueError, match="Duplicate FieldSpec name"):
        FieldRegistry(specs)


def test_registry_preserves_insertion_order():
    specs = [
        FieldSpec(name="c", widget=Widget.TEXTBOX),
        FieldSpec(name="a", widget=Widget.TEXTBOX),
        FieldSpec(name="b", widget=Widget.TEXTBOX),
    ]
    registry = FieldRegistry(specs)
    assert registry.names() == ["c", "a", "b"]
    assert list(registry) == specs


def test_getitem_by_name():
    spec = FieldSpec(name="seed", widget=Widget.NUMBER, default=42)
    registry = FieldRegistry([spec])
    assert registry["seed"] is spec


def test_for_selection_filters_by_arch_and_training_type():
    universal = FieldSpec(name="output_dir", widget=Widget.TEXTBOX)
    flux_only = FieldSpec(
        name="guidance_scale", widget=Widget.NUMBER, archs=frozenset({"flux1"})
    )
    lora_only = FieldSpec(
        name="network_dim",
        widget=Widget.NUMBER,
        training_types=frozenset({"lora"}),
    )
    registry = FieldRegistry([universal, flux_only, lora_only])

    names = {
        s.name for s in registry.for_selection(arch_key="flux1", training_type="lora")
    }
    assert names == {"output_dir", "guidance_scale", "network_dim"}

    names = {
        s.name for s in registry.for_selection(arch_key="sdxl", training_type="lora")
    }
    assert names == {"output_dir", "network_dim"}

    names = {
        s.name
        for s in registry.for_selection(arch_key="flux1", training_type="dreambooth")
    }
    assert names == {"output_dir", "guidance_scale"}


def test_keep_if_falsy_flag_defaults_false():
    spec = FieldSpec(name="guidance_scale", widget=Widget.NUMBER)
    assert spec.keep_if_falsy is False

    spec_keep = FieldSpec(
        name="guidance_scale", widget=Widget.NUMBER, keep_if_falsy=True
    )
    assert spec_keep.keep_if_falsy is True


def test_to_toml_from_toml_coercion_roundtrip():
    spec = FieldSpec(
        name="network_dim",
        widget=Widget.NUMBER,
        to_toml=lambda v: int(v),
        from_toml=lambda v: float(v),
    )
    assert spec.coerce_to_toml(4.0) == 4
    assert isinstance(spec.coerce_to_toml(4.0), int)
    assert spec.coerce_from_toml(4) == 4.0
    assert isinstance(spec.coerce_from_toml(4), float)


def test_gui_only_field_never_confused_with_trainer_arg():
    spec = FieldSpec(name="architecture", widget=Widget.DROPDOWN, gui_only=True)
    assert spec.gui_only is True
