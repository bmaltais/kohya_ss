"""Batch-convert legacy JSON presets to v2 TOML siblings (Move 8).

Walks `presets/**/*.json`, imports each one through
`kohya_gui_v2.legacy_import.import_json` (autodetecting training type), and
writes a sibling `.toml` next to it via `kohya_gui_v2.config_io.save_config`
-- the exact same import + save code path a v2 tab's Open dialog uses, per
the wargame plan's "single code path" requirement (the converter must not
carry its own key mapping). The original JSON presets are never deleted.

Usage: uv run python tools/convert_presets_to_toml.py [--dry-run]
"""

import argparse
import glob
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kohya_gui_v2.config_io import save_config
from kohya_gui_v2.legacy_import import REGISTRIES, import_json


def convert_all(preset_root: str, dry_run: bool = False) -> list:
    """Convert every `*.json` under `preset_root` to a sibling `.toml`.
    Returns a list of (json_path, toml_path, training_type, unrecognized_keys).
    """
    results = []
    for json_path in sorted(
        glob.glob(os.path.join(preset_root, "**", "*.json"), recursive=True)
    ):
        toml_path = os.path.splitext(json_path)[0] + ".toml"
        result = import_json(json_path)
        if not dry_run:
            save_config(REGISTRIES[result.training_type], result.values, toml_path)
        results.append(
            (json_path, toml_path, result.training_type, result.unrecognized_keys)
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--presets-dir",
        default=os.path.join(PROJECT_ROOT, "presets"),
        help="Root directory to walk for *.json presets (default: ./presets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be converted without writing any .toml files",
    )
    args = parser.parse_args()

    results = convert_all(args.presets_dir, dry_run=args.dry_run)

    by_type: dict = {}
    for json_path, toml_path, training_type, unrecognized in results:
        by_type.setdefault(training_type, []).append(json_path)
        if unrecognized:
            print(
                f"[{training_type}] {os.path.relpath(json_path, PROJECT_ROOT)}: "
                f"ignored legacy keys: {', '.join(unrecognized)}"
            )

    print(f"\nConverted {len(results)} preset(s):")
    for training_type, paths in sorted(by_type.items()):
        print(f"  {training_type}: {len(paths)}")

    if args.dry_run:
        print("\n(dry run -- no .toml files written)")


if __name__ == "__main__":
    main()
