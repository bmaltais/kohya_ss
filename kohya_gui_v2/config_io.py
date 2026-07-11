"""TOML config save/load/run-config construction.

The file the GUI saves via `save_config` is, byte-for-byte, a file that
`sd-scripts/library/args.py::read_config_from_file` can consume via
`--config_file` (see wargame plan, Move 1 recon: confirmed via
tests/test_v2_config_file_tolerance.py that all six trainer parsers tolerate
unknown keys, so GUI-only bookkeeping keys like "architecture" and
"training_type" can live directly in the same flat TOML without a filtered
sibling copy -- Fork R1 in the plan is not active).
"""

import os

import toml

from .fields import FieldRegistry


def _flatten_toml_dict(data: dict) -> dict:
    """Flatten one level of [section] tables, matching sd-scripts'
    read_config_from_file behavior (library/args.py:1170-1180): non-dict
    values are kept as-is, dict values have their keys merged up to the top
    level. Ensures a sectioned TOML and a flat TOML with the same logical
    content load identically.
    """
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def save_config(registry: FieldRegistry, values: dict, path: str) -> None:
    """Save `values` (name -> widget value) as a flat, sorted TOML file.

    Keys whose value equals the FieldSpec default are dropped to keep saved
    files small and readable -- this is a save-time convenience, not a
    correctness requirement (open_config falls back to defaults for any
    missing key).
    """
    out = {}
    for spec in registry:
        if spec.name not in values:
            continue
        value = values[spec.name]
        if value == spec.default:
            continue
        out[spec.name] = spec.coerce_to_toml(value)

    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    with open(path, "w", encoding="utf-8") as f:
        toml.dump(dict(sorted(out.items())), f)


def load_config(registry: FieldRegistry, path: str) -> dict:
    """Load a TOML file (flat or sectioned) and return name -> value for
    every FieldSpec, falling back to the spec's default when the key is
    absent from the file. Unknown keys present in the file but not backed
    by any FieldSpec are ignored here (callers that need to warn about them,
    e.g. legacy JSON import, do that separately).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = toml.load(f)
    flat = _flatten_toml_dict(raw)

    values = {}
    for spec in registry:
        if spec.name in flat:
            values[spec.name] = spec.coerce_from_toml(flat[spec.name])
        else:
            values[spec.name] = spec.default
    return values


def build_run_config(
    registry: FieldRegistry,
    values: dict,
    arch_key: str,
    training_type: str,
    keep_falsy_default: bool = False,
) -> dict:
    """Build the dict that gets written to the run-time TOML passed via
    --config_file. Drops GUI-only keys, drops keys not applicable to the
    selected architecture/training type, and drops falsy values (""/False/
    None) unless the FieldSpec opts out via keep_if_falsy=True or the
    backend's keep_falsy_default is True.
    """
    out = {}
    for spec in registry.for_selection(arch_key=arch_key, training_type=training_type):
        if spec.gui_only:
            continue
        if spec.name not in values:
            continue
        value = values[spec.name]
        if not (spec.keep_if_falsy or keep_falsy_default):
            if value in ("", False, None):
                continue
        out[spec.name] = spec.coerce_to_toml(value)
    return dict(sorted(out.items()))
