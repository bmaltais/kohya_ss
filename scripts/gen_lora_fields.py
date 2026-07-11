"""One-off generator: build kohya_gui_v2/tabs/lora_fields.py from
wargame/reference/gap-analysis-lora.md (port/defect/gap-candidate per
architecture) plus real argparse introspection (widget type, default,
choices) for the 8 LoRA architectures' sd-scripts parsers.

Not part of the shipped package -- a build-time tool, run once (re-run if
the gap analysis changes). Output is a plain, hand-editable Python file.
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
    "gap-analysis-lora.md",
)

ARCH_SCRIPTS = {
    "sd15": "train_network",
    "sd2": "train_network",
    "sdxl": "sdxl_train_network",
    "flux1": "flux_train_network",
    "sd3": "sd3_train_network",
    "hunyuan_image": "hunyuan_image_train_network",
    "anima": "anima_train_network",
    "lumina": "lumina_train_network",
}

# Keys excluded from FieldSpec generation entirely: accelerate-launch /
# distributed / meta args already covered elsewhere in the framework, or
# genuinely not user-facing for a single-GPU-first GUI. Mirrors the "exclude"
# disposition philosophy but applied mechanically to known meta-arg names
# that appear across all parsers.
ALWAYS_EXCLUDE = {
    "config_file",
    "output_config",
    "help",
}

WIDGET_ENUM_IMPORT = "from ..fields import FieldSpec, Widget"


def load_parser(module_name):
    if SD_SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SD_SCRIPTS_DIR)
    mod = importlib.import_module(module_name)
    return mod.setup_parser()


def action_to_widget_and_default(action):
    """Map an argparse.Action to (Widget enum name, default repr, to_toml repr)."""
    if action.nargs == 0 and isinstance(action.default, bool):
        return "CHECKBOX", repr(bool(action.default)), "None"
    if action.choices:
        coercer = "None"
        if action.type in (int,):
            coercer = "_to_int"
        elif action.type in (float,):
            coercer = "_to_float"
        return "DROPDOWN", repr(action.default), coercer
    if action.nargs not in (None, 1) or isinstance(action.default, (list, tuple)):
        # nargs-based multi-value args (e.g. text_encoder_lr) aren't a plain
        # scalar; don't force int/float coercion on them.
        return (
            "TEXTBOX",
            repr(action.default if action.default is not None else ""),
            "None",
        )
    if action.type in (int,):
        return (
            "NUMBER",
            repr(action.default if action.default is not None else 0),
            "_to_int",
        )
    if action.type in (float,):
        return (
            "NUMBER",
            repr(action.default if action.default is not None else 0.0),
            "_to_float",
        )
    return "TEXTBOX", repr(action.default if action.default is not None else ""), "None"


def parse_gap_file(path):
    """Return {arch_key: {name: (category, disposition)}}."""
    text = open(path, encoding="utf-8").read()
    sections = re.split(r"^## Architecture: (\w+) ", text, flags=re.M)
    result = {}
    # sections[0] is preamble; then alternating (arch_key, body)
    for i in range(1, len(sections), 2):
        arch_key = sections[i]
        body = sections[i + 1]
        rows = re.findall(
            r"^\| `([a-zA-Z0-9_]+)` \| ([\w][\w /-]*) \| ([\w][\w-]*|n/a) \|",
            body,
            flags=re.M,
        )
        result[arch_key] = {
            name: (cat.strip(), disp.strip()) for name, cat, disp in rows
        }
    return result


def main():
    gap = parse_gap_file(GAP_FILE)

    parsers = {}
    for arch, module_name in ARCH_SCRIPTS.items():
        if module_name not in parsers:
            parsers[module_name] = load_parser(module_name)

    action_by_module_and_name = {}
    for module_name, parser in parsers.items():
        action_by_module_and_name[module_name] = {
            a.dest: a for a in parser._actions if a.dest not in ("help",)
        }

    # Union of all included keys (port + gap-candidates with expose/expose-advanced),
    # tracking which architectures support each.
    key_archs: dict = {}
    key_meta: dict = {}  # name -> (widget, default_repr) from first arch that has it

    for arch, rows in gap.items():
        module_name = ARCH_SCRIPTS[arch]
        actions = action_by_module_and_name[module_name]
        for name, (cat, disp) in rows.items():
            if name in ALWAYS_EXCLUDE:
                continue
            if cat.startswith("defect"):
                continue
            if cat == "gap candidate" and disp not in ("expose", "expose-advanced"):
                continue
            # cat == "port" or included gap-candidate
            key_archs.setdefault(name, set()).add(arch)
            if name not in key_meta and name in actions:
                widget, default_repr, coercer = action_to_widget_and_default(
                    actions[name]
                )
                key_meta[name] = (widget, default_repr, coercer)

    all_archs = set(ARCH_SCRIPTS.keys())
    lines = []
    lines.append('"""LoRA FieldSpecs -- generated by scripts/gen_lora_fields.py from')
    lines.append(
        "wargame/reference/gap-analysis-lora.md + wargame/reference/arch-matrix-lora.md."
    )
    lines.append("")
    lines.append("Regenerate with: uv run python scripts/gen_lora_fields.py")
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
    lines.append("LORA_FIELDS = [")

    for name in sorted(key_meta.keys()):
        widget, default_repr, coercer = key_meta[name]
        archs = key_archs[name]
        archs_repr = "None" if archs == all_archs else f"frozenset({sorted(archs)!r})"
        coerce_kwargs = ""
        if coercer != "None":
            coerce_kwargs = f", to_toml={coercer}, from_toml={coercer}"
        lines.append(
            f"    FieldSpec(name={name!r}, widget=Widget.{widget}, default={default_repr}, "
            f'archs={archs_repr}, training_types=frozenset({{"lora"}}), group="lora_generated"'
            f"{coerce_kwargs}),"
        )

    lines.append("]")
    lines.append("")

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "kohya_gui_v2",
        "tabs",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lora_fields_generated.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(key_meta)} FieldSpecs to {out_path}")
    print(f"Architectures: {sorted(all_archs)}")


if __name__ == "__main__":
    main()
