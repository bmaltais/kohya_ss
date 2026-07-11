"""Checkpoint B6 completeness tests (Move 5): Anima LLLite FieldSpecs.

Three directions, per the wargame plan:
(1) every gap-analysis expose/expose-advanced disposition has a FieldSpec
    (or is a documented derivation-only key);
(2) every non-gui_only FieldSpec name exists in the real sd-scripts parser;
(3) every exclude disposition carries a reason string in the fixture.

This is the last of the six training types (Move 5, B1-B6 complete after
this checkpoint) and the only single-architecture tab in the project --
see wargame/reference/arch-matrix-anima_lllite.md #1.

Reference: wargame/reference/gap-analysis-anima_lllite.md,
wargame/reference/arch-matrix-anima_lllite.md. Same header/row vocabulary
as LoRA's ("## Architecture: <key> (`<script>.py`, N args, single
architecture)").
"""

import importlib
import os
import re
import sys

import pytest

from kohya_gui_v2.tabs.anima_lllite_derivations import derive
from kohya_gui_v2.tabs.anima_lllite_fields import ANIMA_LLLITE_REGISTRY

pytest.importorskip(
    "torch", reason="sd-scripts entry modules import torch at module load time"
)

SD_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sd-scripts")
GAP_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "wargame",
    "reference",
    "gap-analysis-anima_lllite.md",
)

MODULE_NAME = "anima_train_control_net_lllite"
ARCH_KEY = "anima_lllite"

NO_FIELDSPEC_EXPECTED = {"config_file", "output_config"}


def _load_parser():
    if SD_SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SD_SCRIPTS_DIR)
    mod = importlib.import_module(MODULE_NAME)
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
def gap_rows():
    return _parse_gap_file()[ARCH_KEY]


@pytest.fixture(scope="module")
def parser():
    return _load_parser()


def test_every_expose_disposition_has_fieldspec_or_documented_derivation(gap_rows):
    derived_keys = set(derive({}, ARCH_KEY).keys())
    registry_names = set(ANIMA_LLLITE_REGISTRY.names())

    missing = []
    for name, cat, disp, _note in gap_rows:
        if cat != "gap candidate" or disp not in ("expose", "expose-advanced"):
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
    ), f"expose/expose-advanced keys with no FieldSpec or derivation: {missing}"


def test_every_port_key_has_fieldspec_or_documented_derivation(gap_rows):
    derived_keys = set(derive({}, ARCH_KEY).keys())
    registry_names = set(ANIMA_LLLITE_REGISTRY.names())

    missing = []
    for name, cat, _disp, _note in gap_rows:
        if cat != "port":
            continue
        if (
            name in registry_names
            or name in derived_keys
            or name in NO_FIELDSPEC_EXPECTED
        ):
            continue
        missing.append(name)

    assert not missing, f"port keys with no FieldSpec or derivation: {missing}"


def test_every_exclude_disposition_has_a_reason(gap_rows):
    missing = []
    for name, cat, disp, note in gap_rows:
        if cat == "gap candidate" and disp == "exclude" and not note:
            missing.append(name)
    assert not missing, f"exclude dispositions missing a reason: {missing}"


def test_fieldspec_names_exist_in_real_parser(parser):
    dests = {a.dest for a in parser._actions}

    missing = []
    for spec in ANIMA_LLLITE_REGISTRY:
        if spec.gui_only:
            continue
        if spec.name not in dests:
            missing.append(spec.name)

    assert not missing, (
        f"FieldSpecs claiming support for {MODULE_NAME} but absent from its "
        f"real parser: {missing}"
    )


def test_no_duplicate_fieldspec_names():
    assert len(ANIMA_LLLITE_REGISTRY) > 100
