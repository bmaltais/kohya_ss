"""Checkpoint B5 completeness tests (Move 5): LeCo FieldSpecs.

Three directions, per the wargame plan:
(1) every gap-analysis expose/expose-advanced disposition has a FieldSpec
    (or is a documented derivation-only key);
(2) every non-gui_only FieldSpec name exists in that architecture's real
    sd-scripts parser;
(3) every exclude disposition carries a reason string in the fixture.

Plus a dedicated check for the confirmed LeCo bug this checkpoint exists to
avoid reproducing: `prompts_file` must carry `required_cli=True` treatment
(leco_fields.py's REQUIRED_CLI_FIELDS) so v2's command builder always
passes it on argv, not only inside the --config_file TOML (see
wargame/reference/arch-matrix-leco.md section 3).

Reference: wargame/reference/gap-analysis-leco.md,
wargame/reference/arch-matrix-leco.md. Same header/row vocabulary as
LoRA's, except the base architecture section is shared by sd15 and sd2
("## Architecture: sd15/sd2 (...)").
"""

import importlib
import os
import re
import sys

import pytest

from kohya_gui_v2.tabs.leco_derivations import derive
from kohya_gui_v2.tabs.leco_fields import LECO_REGISTRY, REQUIRED_CLI_FIELDS

pytest.importorskip(
    "torch", reason="sd-scripts entry modules import torch at module load time"
)

SD_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sd-scripts")
GAP_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "wargame",
    "reference",
    "gap-analysis-leco.md",
)

ARCH_SCRIPTS = {
    "sd15": "train_leco",
    "sd2": "train_leco",
    "sdxl": "sdxl_train_leco",
}

ARCH_SECTION_TOKEN = {
    "sd15": "sd15/sd2",
    "sd2": "sd15/sd2",
    "sdxl": "sdxl",
}

NO_FIELDSPEC_EXPECTED = {"config_file", "output_config"}


def _load_parser(module_name):
    if SD_SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SD_SCRIPTS_DIR)
    mod = importlib.import_module(module_name)
    return mod.setup_parser()


def _parse_gap_file():
    """Return {section_token: [(name, category, disposition, note), ...]}."""
    text = open(GAP_FILE, encoding="utf-8").read()
    sections = re.split(r"^## Architecture: (\S+) ", text, flags=re.M)
    result = {}
    for i in range(1, len(sections), 2):
        token = sections[i]
        body = sections[i + 1]
        rows = re.findall(
            r"^\| `([a-zA-Z0-9_]+)` \| ([^|]+?) \| ([^|]+?) \| ([^|]+?) \|",
            body,
            flags=re.M,
        )
        result[token] = [
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


@pytest.mark.parametrize("arch_key", sorted(ARCH_SCRIPTS.keys()))
def test_every_expose_disposition_has_fieldspec_or_documented_derivation(
    arch_key, gap_by_section
):
    section = ARCH_SECTION_TOKEN[arch_key]
    derived_keys = set(derive({}, arch_key).keys())
    registry_names = set(LECO_REGISTRY.names())

    missing = []
    for name, cat, disp, _note in gap_by_section[section]:
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
def test_every_port_key_has_fieldspec_or_documented_derivation(
    arch_key, gap_by_section
):
    section = ARCH_SECTION_TOKEN[arch_key]
    derived_keys = set(derive({}, arch_key).keys())
    registry_names = set(LECO_REGISTRY.names())

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
            if cat == "gap candidate" and disp == "exclude" and not note:
                missing.append((section, name))
    assert not missing, f"exclude dispositions missing a reason: {missing}"


@pytest.mark.parametrize("arch_key", sorted(ARCH_SCRIPTS.keys()))
def test_fieldspec_names_exist_in_real_parser(arch_key, parsers):
    module_name = ARCH_SCRIPTS[arch_key]
    dests = {a.dest for a in parsers[module_name]._actions}

    missing = []
    for spec in LECO_REGISTRY:
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


def test_prompts_file_is_required_on_cli():
    """The confirmed bug this checkpoint must not reproduce: both LeCo
    scripts declare --prompts_file as required=True and argparse enforces
    it against sys.argv before --config_file's TOML is read. v2's registry
    must flag prompts_file so the command builder always emits it on argv.
    """
    assert "prompts_file" in REQUIRED_CLI_FIELDS
    assert "prompts_file" in LECO_REGISTRY.names()


@pytest.mark.parametrize("module_name", sorted(set(ARCH_SCRIPTS.values())))
def test_prompts_file_really_is_required_in_the_real_parser(module_name, parsers):
    required_dests = {
        a.dest for a in parsers[module_name]._actions if getattr(a, "required", False)
    }
    assert "prompts_file" in required_dests


def test_no_duplicate_fieldspec_names():
    assert len(LECO_REGISTRY) > 100
