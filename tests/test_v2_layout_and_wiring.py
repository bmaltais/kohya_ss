"""Regression tests for UX-overhaul review fixes:

- wiring order is registry.names() (not layout creation order)
- multi-input rows form after sorting non-adjacent shared row keys
- bilingual argparse help is stripped to English-only info text
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import pytest

from kohya_gui_v2.fields import FieldSpec, Widget
from kohya_gui_v2.layout_map import LAYOUT, FieldLayout
from kohya_gui_v2.tab_builder import _pack_rows, _sort_specs_for_rows
from kohya_gui_v2.tabs.lora_fields import LORA_REGISTRY
from kohya_gui_v2.tabs.lora_tab import lora_tab

# generators live under scripts/ — put it on path for unit tests
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from gen_fields_common import english_only_help  # noqa: E402


def test_wiring_order_matches_registry_names_not_layout_order():
    """ordered_names / ordered_components must follow registry.names().

    Layout creates widgets section-by-section (and may sort within sections
    for row packing); save/open/train must still zip in registry order.
    """
    with gr.Blocks():
        _arch, ordered_names, ordered_components = lora_tab(headless=True)

    expected = [n for n in LORA_REGISTRY.names() if n != "architecture"]
    assert ordered_names == expected
    assert len(ordered_components) == len(expected)
    # identity: component list position i corresponds to registry name i
    for i, name in enumerate(ordered_names):
        assert name == expected[i]


def test_pack_rows_groups_non_adjacent_shared_row_keys():
    """Alphabetical registry order must not prevent multi-field rows.

    Simulate three specs that share a row key but would not be adjacent in
    alphabetical order; after sort+pack they must form one multi-field row.
    """
    # Use real layout_map row keys if the noise family is mapped; else inject.
    names = [
        "adaptive_noise_scale",
        "alpha_mask",  # different / no shared row — spacer
        "noise_offset",
        "noise_offset_random_strength",
    ]
    # Build specs; monkeypatch via temporary LAYOUT entries if needed
    shared = "test.shared_noise_row"
    originals = {}
    for n in ("adaptive_noise_scale", "noise_offset", "noise_offset_random_strength"):
        if n in LAYOUT:
            originals[n] = LAYOUT[n]
            LAYOUT[n] = FieldLayout(
                section=LAYOUT[n].section,
                row=shared,
                label=LAYOUT[n].label,
                info=LAYOUT[n].info,
            )
        else:
            LAYOUT[n] = FieldLayout(section="advanced", row=shared, label=n, info=None)
            originals[n] = None

    try:
        specs = [
            FieldSpec(name=n, widget=Widget.NUMBER, default=0, group="advanced")
            for n in names
            if n != "alpha_mask"
        ] + [
            FieldSpec(
                name="alpha_mask",
                widget=Widget.CHECKBOX,
                default=False,
                group="advanced",
            )
        ]
        # Alphabetical-ish input order (as registry would give)
        specs_alpha = sorted(specs, key=lambda s: s.name)
        packed = _pack_rows(_sort_specs_for_rows(specs_alpha))
        multi = [row for row in packed if len(row) > 1]
        assert multi, "expected at least one multi-field row after sort+pack"
        multi_names = {s.name for s in multi[0]}
        assert multi_names == {
            "adaptive_noise_scale",
            "noise_offset",
            "noise_offset_random_strength",
        }
    finally:
        for n, prev in originals.items():
            if prev is None:
                LAYOUT.pop(n, None)
            else:
                LAYOUT[n] = prev


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "path to ae (*.sft or *.safetensors) / aeのパス（*.sftまたは*.safetensors）",
            "path to ae (*.sft or *.safetensors)",
        ),
        (
            "use alpha channel as mask for training / 画像のアルファチャンネルをlossのマスクに使用する",
            "use alpha channel as mask for training",
        ),
        ("English only help text", "English only help text"),
        ("  spaced   words  / 日本語 ", "spaced words"),
    ],
)
def test_english_only_help_strips_japanese(raw, expected):
    assert english_only_help(raw) == expected


def test_generated_lora_infos_have_no_cjk():
    import re

    cjk = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    offenders = [s.name for s in LORA_REGISTRY if s.info and cjk.search(s.info)]
    assert offenders == [], f"CJK still present in info for: {offenders[:10]}"
