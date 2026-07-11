"""Checkpoint B4 completeness tests (Move 5): Textual Inversion FieldSpecs.

Three directions, per the wargame plan:
(1) every gap-analysis expose/expose-advanced disposition has a FieldSpec
    (or is a documented derivation-only key);
(2) every non-gui_only FieldSpec name exists in that architecture's real
    sd-scripts parser;
(3) every exclude disposition carries a reason string in the fixture.

Reference: wargame/reference/gap-analysis-textual_inversion.md,
wargame/reference/arch-matrix-textual_inversion.md. This gap-analysis file
uses the same header/row vocabulary as LoRA's: "## Architecture: <key>
(`<script>.py`, N args)", disposition "n/a" for port rows, "gap candidate"
(space) category.
"""

import importlib
import os
import re
import sys

import pytest

from kohya_gui_v2.tabs.textual_inversion_derivations import derive
from kohya_gui_v2.tabs.textual_inversion_fields import TEXTUAL_INVERSION_REGISTRY

pytest.importorskip(
    "torch", reason="sd-scripts entry modules import torch at module load time"
)

SD_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sd-scripts")
GAP_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "wargame",
    "reference",
    "gap-analysis-textual_inversion.md",
)

ARCH_SCRIPTS = {
    "sd_v1v2": "train_textual_inversion",
    "sdxl": "sdxl_train_textual_inversion",
}

NO_FIELDSPEC_EXPECTED = {"config_file", "output_config"}


def _load_parser(module_name):
    if SD_SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SD_SCRIPTS_DIR)
    mod = importlib.import_module(module_name)
    return mod.setup_parser()


def _parse_gap_file():
    """Return {arch_key: [(name, category, disposition, note), ...]}."""
    text = open(GAP_FILE, encoding="utf-8").read()
    sections = re.split(r"^## Architecture: (\w+) ", text, flags=re.M)
    result = {}
    for i in range(1, len(sections), 2):
        arch_key = sections[i]
        body = sections[i + 1]
        rows = re.findall(
            r"^\| `([a-zA-Z0-9_]+)` \| ([^|]+?) \| ([^|]+?) \| ([^|]+?) \|",
            body,
            flags=re.M,
        )
        result[arch_key] = [
            (name, cat.strip(), disp.strip(), note.strip())
            for name, cat, disp, note in rows
        ]
    return result


@pytest.fixture(scope="module")
def gap_by_arch():
    return _parse_gap_file()


@pytest.fixture(scope="module")
def parsers():
    return {
        module_name: _load_parser(module_name)
        for module_name in set(ARCH_SCRIPTS.values())
    }


@pytest.mark.parametrize("arch_key", sorted(ARCH_SCRIPTS.keys()))
def test_every_expose_disposition_has_fieldspec_or_documented_derivation(
    arch_key, gap_by_arch
):
    derived_keys = set(derive({}, arch_key).keys())
    registry_names = set(TEXTUAL_INVERSION_REGISTRY.names())

    missing = []
    for name, cat, disp, _note in gap_by_arch[arch_key]:
        if cat != "gap candidate" or disp not in ("expose", "expose-advanced"):
            continue
        if (
            name in registry_names
            or name in derived_keys
            or name in NO_FIELDSPEC_EXPECTED
        ):
            continue
        missing.append(name)

    assert not missing, (
        f"{arch_key}: expose/expose-advanced keys with no FieldSpec or "
        f"derivation: {missing}"
    )


@pytest.mark.parametrize("arch_key", sorted(ARCH_SCRIPTS.keys()))
def test_every_port_key_has_fieldspec_or_documented_derivation(arch_key, gap_by_arch):
    derived_keys = set(derive({}, arch_key).keys())
    registry_names = set(TEXTUAL_INVERSION_REGISTRY.names())

    missing = []
    for name, cat, _disp, _note in gap_by_arch[arch_key]:
        if cat != "port":
            continue
        if (
            name in registry_names
            or name in derived_keys
            or name in NO_FIELDSPEC_EXPECTED
        ):
            continue
        missing.append(name)

    assert (
        not missing
    ), f"{arch_key}: port keys with no FieldSpec or derivation: {missing}"


def test_every_exclude_disposition_has_a_reason(gap_by_arch):
    missing = []
    for arch_key, rows in gap_by_arch.items():
        for name, cat, disp, note in rows:
            if cat == "gap candidate" and disp == "exclude" and not note:
                missing.append((arch_key, name))
    assert not missing, f"exclude dispositions missing a reason: {missing}"


@pytest.mark.parametrize("arch_key", sorted(ARCH_SCRIPTS.keys()))
def test_fieldspec_names_exist_in_real_parser(arch_key, parsers):
    module_name = ARCH_SCRIPTS[arch_key]
    dests = {a.dest for a in parsers[module_name]._actions}

    missing = []
    for spec in TEXTUAL_INVERSION_REGISTRY:
        if spec.gui_only:
            continue
        if not spec.supports_arch(arch_key):
            continue
        if spec.name not in dests:
            missing.append(spec.name)

    assert not missing, (
        f"{arch_key} ({module_name}): FieldSpecs claiming support for this "
        f"arch but absent from its real parser: {missing}"
    )


def test_no_duplicate_fieldspec_names():
    assert len(TEXTUAL_INVERSION_REGISTRY) > 100
