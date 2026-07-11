"""Move 8: legacy JSON preset importer tests.

Covers the plan's three acceptance criteria: (1) every preset JSON imports
without error and is routed to the training type its own folder implies,
(2) a converted TOML loads back to the same effective values as the
original JSON import (the "import(json) == load(converted toml)"
invariant), (3) unrecognized keys are surfaced, not silently dropped --
exercised with one deliberately polluted fixture.
"""

import glob
import json
import os
import tempfile

import pytest

from kohya_gui_v2.config_io import load_config, save_config
from kohya_gui_v2.legacy_import import REGISTRIES, detect_training_type, import_json

PRESETS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets")
FIXTURES_ROOT = os.path.join(os.path.dirname(__file__), "fixtures", "v2")

# (glob pattern, expected training_type) -- folder name is ground truth for
# the presets/ corpus; the hand-built fixtures are named by training type.
CORPUS = [
    (os.path.join(PRESETS_ROOT, "lora", "*.json"), "lora"),
    (os.path.join(PRESETS_ROOT, "dreambooth", "*.json"), "dreambooth"),
    (os.path.join(PRESETS_ROOT, "finetune", "*.json"), "finetune"),
    (os.path.join(FIXTURES_ROOT, "textual_inversion", "*.json"), "textual_inversion"),
    (os.path.join(FIXTURES_ROOT, "leco", "*.json"), "leco"),
    (os.path.join(FIXTURES_ROOT, "anima_lllite", "*.json"), "anima_lllite"),
]


def _all_fixtures():
    out = []
    for pattern, expected_type in CORPUS:
        for path in sorted(glob.glob(pattern)):
            out.append((path, expected_type))
    return out


ALL_FIXTURES = _all_fixtures()


@pytest.mark.parametrize(
    "path,expected_type",
    ALL_FIXTURES,
    ids=[os.path.basename(p) for p, _ in ALL_FIXTURES],
)
def test_import_json_succeeds_and_detects_correct_type(path, expected_type):
    result = import_json(path)  # autodetect, not given the expected type
    assert result.training_type == expected_type
    assert result.architecture
    assert isinstance(result.values, dict)
    assert len(result.values) > 0


@pytest.mark.parametrize(
    "path,expected_type",
    ALL_FIXTURES,
    ids=[os.path.basename(p) for p, _ in ALL_FIXTURES],
)
def test_import_then_save_then_load_round_trips(path, expected_type):
    """import(json) -> save -> load reproduces the same effective values.

    A FieldSpec that derive() explicitly nulls out (not applicable to the
    detected architecture) round-trips to that spec's own default after a
    save/load cycle, since save_config drops default-valued keys and
    load_config falls back to the default for anything absent -- so a
    normalized comparison (None counted as "at default") is the correct
    equivalence check here, not raw dict equality.
    """
    result = import_json(path, training_type=expected_type)
    registry = REGISTRIES[expected_type]

    values = dict(result.values)
    values["architecture"] = result.architecture
    values["training_type"] = expected_type

    with tempfile.TemporaryDirectory() as tmp:
        toml_path = os.path.join(tmp, "converted.toml")
        save_config(registry, values, toml_path)
        reloaded = load_config(registry, toml_path)

    mismatches = []
    for spec in registry:
        original = values.get(spec.name)
        effective_original = spec.default if original is None else original
        # Normalize through the same to_toml/from_toml coercion save_config
        # /load_config apply internally (e.g. a preset storing "0" as a
        # string for a Number field) -- import_json returns raw JSON values,
        # coercion is save_config's job, not the importer's.
        effective_original = spec.coerce_from_toml(
            spec.coerce_to_toml(effective_original)
        )
        if effective_original != reloaded.get(spec.name):
            mismatches.append((spec.name, effective_original, reloaded.get(spec.name)))

    assert not mismatches, f"{os.path.basename(path)}: {mismatches[:10]}"


def test_polluted_fixture_surfaces_unrecognized_keys():
    """A JSON preset with a genuinely unknown key must not fail import, and
    the unknown key must be surfaced (not silently dropped) -- the counter-move
    documented in the wargame plan for this Move.
    """
    source_path = os.path.join(PRESETS_ROOT, "lora", "loha-sd15.json")
    with open(source_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["totally_unknown_key_from_a_future_gui_version"] = "surprise"

    with tempfile.TemporaryDirectory() as tmp:
        polluted_path = os.path.join(tmp, "polluted.json")
        with open(polluted_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        result = import_json(polluted_path, training_type="lora")

    assert "totally_unknown_key_from_a_future_gui_version" in result.unrecognized_keys


def test_detect_training_type_matches_folder_for_every_preset():
    for pattern, expected_type in CORPUS:
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            assert detect_training_type(cfg) == expected_type, path
