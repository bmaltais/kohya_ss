"""Checkpoint B2 completeness tests (Move 5): DreamBooth FieldSpecs.

Three directions, per the wargame plan:
(1) every gap-analysis expose/expose-advanced disposition has a FieldSpec
    (or is a documented derivation-only key);
(2) every non-gui_only FieldSpec name exists in that architecture's real
    sd-scripts parser;
(3) every exclude disposition carries a reason string in the fixture.

Reference: wargame/reference/gap-analysis-dreambooth.md,
wargame/reference/arch-matrix-dreambooth.md.
"""

import importlib
import os
import re
import sys

import pytest

from kohya_gui_v2.tabs.dreambooth_derivations import derive
from kohya_gui_v2.tabs.dreambooth_fields import DREAMBOOTH_REGISTRY

pytest.importorskip(
    "torch", reason="sd-scripts entry modules import torch at module load time"
)

SD_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sd-scripts")
GAP_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "wargame",
    "reference",
    "gap-analysis-dreambooth.md",
)

ARCH_SCRIPTS = {
    "sd15": "train_db",
    "sd2": "train_db",
    "sdxl": "sdxl_train",
    "sd3": "sd3_train",
    "flux1": "flux_train",
}

ARCH_SECTION = {
    "sd15": "SD1.x/2.x (default)",
    "sd2": "SD1.x/2.x (default)",
    "sdxl": "SDXL",
    "sd3": "SD3/3.5",
    "flux1": "FLUX.1",
}

# Keys that are intentionally FieldSpec-free: architecture-dead widgets
# (never emitted for this tab, matrix-confirmed) or accelerate-launch/meta
# args excluded mechanically by the generator.
NO_FIELDSPEC_EXPECTED = {"split_mode", "train_blocks", "config_file", "output_config"}


def _load_parser(module_name):
    if SD_SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SD_SCRIPTS_DIR)
    mod = importlib.import_module(module_name)
    return mod.setup_parser()


def _parse_gap_file():
    """Return {section_title: [(name, category, disposition, note), ...]}."""
    text = open(GAP_FILE, encoding="utf-8").read()
    sections = re.split(r"^## (.+?)\n", text, flags=re.M)
    result = {}
    for i in range(1, len(sections), 2):
        title = sections[i].strip()
        body = sections[i + 1]
        rows = re.findall(
            r"^\| `([a-zA-Z0-9_]+)` \| ([^|]+?) \| ([^|]+?) \| ([^|]+?) \|",
            body,
            flags=re.M,
        )
        result[title] = [
            (name, cat.strip(), disp.strip(), note.strip())
            for name, cat, disp, note in rows
        ]
    return result


@pytest.fixture(scope="module")
def gap_by_section():
    return _parse_gap_file()


@pytest.fixture(scope="module")
def parsers():
    return {
        module_name: _load_parser(module_name)
        for module_name in set(ARCH_SCRIPTS.values())
    }


def _section_for(prefix, gap_by_section):
    for title in gap_by_section:
        if title.startswith(prefix):
            return title
    raise KeyError(prefix)


@pytest.mark.parametrize("arch_key", sorted(ARCH_SCRIPTS.keys()))
def test_every_expose_disposition_has_fieldspec_or_documented_derivation(
    arch_key, gap_by_section
):
    section = _section_for(ARCH_SECTION[arch_key], gap_by_section)
    derived_keys = set(derive({}, arch_key).keys())
    registry_names = set(DREAMBOOTH_REGISTRY.names())

    missing = []
    for name, cat, disp, _note in gap_by_section[section]:
        if cat != "gap-candidate" or disp not in ("expose", "expose-advanced"):
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
def test_every_port_key_has_fieldspec_or_documented_derivation(
    arch_key, gap_by_section
):
    section = _section_for(ARCH_SECTION[arch_key], gap_by_section)
    derived_keys = set(derive({}, arch_key).keys())
    registry_names = set(DREAMBOOTH_REGISTRY.names())

    missing = []
    for name, cat, _disp, _note in gap_by_section[section]:
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


def test_every_exclude_disposition_has_a_reason(gap_by_section):
    missing = []
    for section, rows in gap_by_section.items():
        for name, cat, disp, note in rows:
            if cat == "gap-candidate" and disp == "exclude" and not note:
                missing.append((section, name))
    assert not missing, f"exclude dispositions missing a reason: {missing}"


@pytest.mark.parametrize("arch_key", sorted(ARCH_SCRIPTS.keys()))
def test_fieldspec_names_exist_in_real_parser(arch_key, parsers):
    module_name = ARCH_SCRIPTS[arch_key]
    dests = {a.dest for a in parsers[module_name]._actions}

    missing = []
    for spec in DREAMBOOTH_REGISTRY:
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
    # FieldRegistry construction already raises on duplicates; this asserts
    # the registry loaded at all and is non-trivially sized.
    assert len(DREAMBOOTH_REGISTRY) > 100
