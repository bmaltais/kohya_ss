"""One-off generator: build kohya_gui_v2/tabs/anima_lllite_fields_generated.py
from wargame/reference/gap-analysis-anima_lllite.md (port/defect/gap-candidate)
plus real argparse introspection (widget type, default, choices) for the
Anima LLLite tab's single architecture.

Not part of the shipped package -- a build-time tool, run once (re-run if
the gap analysis changes). Output is a plain, hand-editable Python file.

Same header/row vocabulary as LoRA's gap-analysis file. Unlike every other
training type in this project, this tab is architecture-fixed (single
model family + ControlNet-LLLite adapter, no model-checkbox row) --
FieldSpec.archs is always None (universal) since there is only one
architecture to gate against.
"""

import importlib
import os
import re
import sys

SD_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sd-scripts"
)
GAP_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "wargame",
    "reference",
    "gap-analysis-anima_lllite.md",
)

MODULE_NAME = "anima_train_control_net_lllite"

ALWAYS_EXCLUDE = {
    "config_file",
    "output_config",
    "help",
    # Accelerate-launch-only concepts: anima_lllite_gui.py's train_model
    # consumes these directly into the accelerate-launch CLI prefix and
    # never writes them into config_toml_data, even though the real trainer
    # parser also declares dynamo_backend (hence gap-analysis classifies it
    # as a gap-candidate, not a defect). Exposing it as a v2 TOML FieldSpec
    # would collide with the existing accelerate-launch-only GUI parameter
    # of the same name and perturb equivalence (same fix as LeCo, Move 7).
    "dynamo_backend",
    "dynamo_mode",
    "dynamo_use_fullgraph",
    "dynamo_use_dynamic",
    "num_processes",
    "num_machines",
    "num_cpu_threads_per_process",
    "multi_gpu",
    "gpu_ids",
    "main_process_port",
    "extra_accelerate_launch_args",
}

WIDGET_ENUM_IMPORT = "from ..fields import FieldSpec, Widget"


def load_parser(module_name):
    if SD_SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SD_SCRIPTS_DIR)
    mod = importlib.import_module(module_name)
    return mod.setup_parser()


def action_to_widget_and_default(action):
    """Map an argparse.Action to (Widget enum name, default repr, coercer,
    choices repr or None)."""
    if action.nargs == 0 and isinstance(action.default, bool):
        return "CHECKBOX", repr(bool(action.default)), "None", None
    if action.choices:
        coercer = "None"
        if action.type in (int,):
            coercer = "_to_int"
        elif action.type in (float,):
            coercer = "_to_float"
        # Always include the action's own default among the choices --
        # some parsers declare a default that isn't itself in `choices`,
        # and Gradio's gr.Dropdown emits a UserWarning + falls back to
        # allow_custom_value behavior whenever the initial value isn't a
        # member of `choices`.
        choices_list = list(action.choices)
        if action.default not in choices_list:
            choices_list = [action.default] + choices_list
        return "DROPDOWN", repr(action.default), coercer, repr(choices_list)
    if action.nargs not in (None, 1) or isinstance(action.default, (list, tuple)):
        return (
            "TEXTBOX",
            repr(action.default if action.default is not None else ""),
            "None",
            None,
        )
    if action.type in (int,):
        return (
            "NUMBER",
            repr(action.default if action.default is not None else 0),
            "_to_int",
            None,
        )
    if action.type in (float,):
        return (
            "NUMBER",
            repr(action.default if action.default is not None else 0.0),
            "_to_float",
            None,
        )
    return (
        "TEXTBOX",
        repr(action.default if action.default is not None else ""),
        "None",
        None,
    )


def parse_gap_file(path):
    """Return {arch_key: {name: (category, disposition)}}."""
    text = open(path, encoding="utf-8").read()
    sections = re.split(r"^## Architecture: (\w+) ", text, flags=re.M)
    result = {}
    for i in range(1, len(sections), 2):
        arch_key = sections[i]
        body = sections[i + 1]
        rows = re.findall(
            r"^\| `([a-zA-Z0-9_]+)` \| ([^|]+?) \| ([^|]+?) \|",
            body,
            flags=re.M,
        )
        result[arch_key] = {
            name: (cat.strip(), disp.strip()) for name, cat, disp in rows
        }
    return result


def main():
    gap = parse_gap_file(GAP_FILE)
    rows = gap["anima_lllite"]

    parser = load_parser(MODULE_NAME)
    actions = {a.dest: a for a in parser._actions if a.dest != "help"}

    key_meta: dict = {}

    for name, (cat, disp) in rows.items():
        if name in ALWAYS_EXCLUDE:
            continue
        if cat.startswith("defect"):
            continue
        if cat == "gap candidate" and disp not in ("expose", "expose-advanced"):
            continue
        if name in actions:
            widget, default_repr, coercer, choices_repr = action_to_widget_and_default(
                actions[name]
            )
            key_meta[name] = (widget, default_repr, coercer, choices_repr)

    lines = []
    lines.append(
        '"""Anima LLLite FieldSpecs -- generated by scripts/gen_anima_lllite_fields.py'
    )
    lines.append(
        "from wargame/reference/gap-analysis-anima_lllite.md + arch-matrix-anima_lllite.md."
    )
    lines.append("")
    lines.append("Regenerate with: uv run python scripts/gen_anima_lllite_fields.py")
    lines.append(
        "Hand edits are fine but will be overwritten on regeneration -- move any"
    )
    lines.append(
        "permanent hand corrections into the generator or a follow-up patch step."
    )
    lines.append('"""')
    lines.append("")
    lines.append(WIDGET_ENUM_IMPORT)
    lines.append("")
    lines.append("def _to_int(v):")
    lines.append('    return int(v) if v not in (None, "") else v')
    lines.append("")
    lines.append("def _to_float(v):")
    lines.append('    return float(v) if v not in (None, "") else v')
    lines.append("")
    lines.append("ANIMA_LLLITE_FIELDS = [")

    for name in sorted(key_meta.keys()):
        widget, default_repr, coercer, choices_repr = key_meta[name]
        coerce_kwargs = ""
        if coercer != "None":
            coerce_kwargs = f", to_toml={coercer}, from_toml={coercer}"
        choices_kwargs = f", choices={choices_repr}" if choices_repr else ""
        lines.append(
            f"    FieldSpec(name={name!r}, widget=Widget.{widget}, default={default_repr}, "
            f'archs=None, training_types=frozenset({{"anima_lllite"}}), group="anima_lllite_generated"'
            f"{coerce_kwargs}{choices_kwargs}),"
        )

    lines.append("]")
    lines.append("")

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "kohya_gui_v2",
        "tabs",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "anima_lllite_fields_generated.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(key_meta)} FieldSpecs to {out_path}")


if __name__ == "__main__":
    main()
