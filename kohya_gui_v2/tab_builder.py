"""Shared training-tab builder for all v2 training types.

Consolidates the six near-identical ``*_tab.py`` modules: accordion sections,
multi-field rows, per-component arch visibility, save/open/train wiring, and
the live client-side parameter filter.

CRITICAL CONTRACT: components may be *created* in section order, but every
event's input/output list is rebuilt from ``registry.names()`` order so
save/open/train never mis-zip values.
"""

from __future__ import annotations

import html
import json
import os
from collections import OrderedDict
from typing import Callable, Optional

import gradio as gr

from kohya_gui.class_command_executor import CommandExecutor
from kohya_gui.common_gui import get_file_path, get_saveasfile_path, setup_environment

from .builder import build_components
from .config_io import build_run_config, load_config, save_config
from .fields import FieldRegistry, FieldSpec
from .layout_map import (
    OPEN_BY_DEFAULT,
    SECTION_ORDER,
    SECTION_TITLES,
    layout_for,
)
from .legacy_import import import_json
from .registry import SD_SCRIPTS_BACKEND

# Cap inputs per gr.Row (legacy GUI typically packs 2–4)
_MAX_PER_ROW = 4


def _filter_js(training_type: str) -> str:
    """Client-side filter scoped to one training tab.

    Matches against a pre-built haystack per elem_id; only toggles the
    ``v2-filter-hidden`` class so Gradio arch visibility (inline styles)
    is never overwritten.
    """
    # training_type is a safe identifier (snake_case); interpolate into JS
    return f"""
(query) => {{
  const q = (query || "").toLowerCase().trim();
  const filterNode = document.getElementById("v2_{training_type}_filter");
  let root = filterNode ? filterNode.closest(".v2-training-tab") : null;
  root = root || document;
  const mapEl = root.querySelector("[data-v2-filter-map]");
  let haystack = {{}};
  if (mapEl) {{
    try {{ haystack = JSON.parse(mapEl.getAttribute("data-v2-filter-map") || "{{}}"); }}
    catch (e) {{ haystack = {{}}; }}
  }}
  const fields = root.querySelectorAll(".v2-field");
  const openStateKey = "data-v2-filter-prev-open";
  const accordions = root.querySelectorAll(".v2-section");

  if (q) {{
    accordions.forEach((acc) => {{
      if (!acc.hasAttribute(openStateKey)) {{
        const lw = acc.querySelector(".label-wrap");
        const isOpen = lw ? lw.classList.contains("open") : true;
        acc.setAttribute(openStateKey, isOpen ? "1" : "0");
      }}
    }});
  }}

  const matchCount = new Map();
  accordions.forEach((acc) => matchCount.set(acc, 0));

  fields.forEach((el) => {{
    let id = el.id || "";
    if (!id) {{
      const withId = el.closest("[id]");
      if (withId) id = withId.id;
    }}
    if (!haystack[id]) {{
      for (const key of Object.keys(haystack)) {{
        try {{
          if (el.id === key || (el.closest && el.closest("#" + CSS.escape(key)))) {{
            id = key;
            break;
          }}
        }} catch (e) {{ /* ignore invalid selectors */ }}
      }}
    }}
    const hay = (haystack[id] || id || el.innerText || "").toLowerCase();
    const match = !q || hay.includes(q);
    if (match) el.classList.remove("v2-filter-hidden");
    else el.classList.add("v2-filter-hidden");
    const section = el.closest(".v2-section");
    if (section && match) matchCount.set(section, (matchCount.get(section) || 0) + 1);
  }});

  accordions.forEach((acc) => {{
    if (!q) {{
      acc.classList.remove("v2-filter-hidden");
      const prev = acc.getAttribute(openStateKey);
      const lw = acc.querySelector(".label-wrap");
      if (prev !== null && lw) {{
        const shouldOpen = prev === "1";
        const isOpen = lw.classList.contains("open");
        if (shouldOpen !== isOpen) lw.click();
        acc.removeAttribute(openStateKey);
      }}
      return;
    }}
    const count = matchCount.get(acc) || 0;
    if (count === 0) acc.classList.add("v2-filter-hidden");
    else {{
      acc.classList.remove("v2-filter-hidden");
      const lw = acc.querySelector(".label-wrap");
      if (lw && !lw.classList.contains("open")) lw.click();
    }}
  }});
  return query;
}}
"""


def _specs_for_layout(registry: FieldRegistry) -> list[FieldSpec]:
    """All registry fields except architecture (which is a top-level control)."""
    out = []
    for spec in registry:
        if spec.gui_only and spec.name == "architecture":
            continue
        out.append(spec)
    return out


def _group_specs_by_section(
    specs: list[FieldSpec],
) -> "OrderedDict[str, list[FieldSpec]]":
    buckets: dict[str, list[FieldSpec]] = {s: [] for s in SECTION_ORDER}
    extra: dict[str, list[FieldSpec]] = {}
    for spec in specs:
        section = spec.group or layout_for(spec.name).section
        if section not in buckets:
            extra.setdefault(section, []).append(spec)
        else:
            buckets[section].append(spec)
    ordered: "OrderedDict[str, list[FieldSpec]]" = OrderedDict()
    for s in SECTION_ORDER:
        if buckets[s]:
            ordered[s] = buckets[s]
    for s, items in extra.items():
        ordered[s] = items
    return ordered


def _pack_rows(specs: list[FieldSpec]) -> list[list[FieldSpec]]:
    """Pack specs sharing a layout row key into chunks of ≤_MAX_PER_ROW."""
    rows: list[list[FieldSpec]] = []
    current_key: Optional[str] = object()  # sentinel
    current: list[FieldSpec] = []

    def flush():
        nonlocal current
        if current:
            rows.append(current)
            current = []

    for spec in specs:
        lay = layout_for(spec.name)
        key = lay.row
        if key is None:
            flush()
            rows.append([spec])
            current_key = object()
            continue
        if key != current_key:
            flush()
            current_key = key
            current = [spec]
        else:
            current.append(spec)
            if len(current) >= _MAX_PER_ROW:
                flush()
                current_key = key  # allow continuation under same key
    flush()
    return rows


def build_training_tab(
    registry: FieldRegistry,
    derive: Callable[[dict, str], dict],
    training_type: str,
    arch_choices: list,
    arch_scripts: dict,
    headless: bool = False,
    config=None,
    config_placeholder: str = "",
    default_arch: str = "sd15",
    required_cli_fields: Optional[frozenset] = None,
):
    """Build one full training tab. Returns (architecture, ordered_names, ordered_components).

    ``required_cli_fields``: optional set of arg names that must appear on
    argv (not only in --config_file TOML) — used by LeCo for prompts_file.
    """

    gr.Checkbox(value=headless, visible=False)  # keep parity with prior tabs

    with gr.Column(elem_classes=["v2-training-tab"]):
        with gr.Accordion("Configuration File", open=False, elem_classes=["v2-config"]):
            with gr.Row():
                config_file_name = gr.Textbox(
                    label="Config file",
                    value="",
                    placeholder=config_placeholder
                    or f"./config_{training_type}_v2.toml",
                )
                button_open = gr.Button("Open")
                button_save = gr.Button("Save")
                button_save_as = gr.Button("Save as")

        with gr.Row(elem_classes=["v2-toolbar"]):
            architecture = gr.Dropdown(
                choices=arch_choices,
                value=default_arch if default_arch in arch_choices else arch_choices[0],
                label="Architecture",
                elem_id=f"v2_{training_type}_architecture",
                scale=2,
            )
            filter_box = gr.Textbox(
                label="Filter parameters",
                placeholder="Type to filter…",
                value="",
                elem_id=f"v2_{training_type}_filter",
                elem_classes=["v2-filter"],
                scale=3,
            )

        layout_specs = _specs_for_layout(registry)
        by_section = _group_specs_by_section(layout_specs)

        components: dict = {}
        section_accordions: dict = {}
        filter_haystack: dict[str, str] = {}

        for section, section_specs in by_section.items():
            title = SECTION_TITLES.get(section, section.replace("_", " ").title())
            is_open = section in OPEN_BY_DEFAULT
            with gr.Accordion(
                title,
                open=is_open,
                elem_classes=["v2-section"],
                elem_id=f"v2_{training_type}_section_{section}",
            ) as acc:
                # data attribute for filter JS (Gradio may not pass arbitrary attrs;
                # we also mirror via a hidden HTML map below)
                section_accordions[section] = acc
                for row_specs in _pack_rows(section_specs):
                    ctx = (
                        gr.Row(elem_classes=["v2-field-row"])
                        if len(row_specs) > 1
                        else gr.Column()
                    )
                    with ctx:
                        for spec in row_specs:
                            # Single-spec registry so build_components stays pure
                            single = type(registry)([spec])
                            built = build_components(
                                single,
                                config=config,
                                elem_id_prefix=f"v2_{training_type}",
                                elem_classes=["v2-field"],
                            )
                            for name, comp in built.items():
                                components[name] = comp
                                lay = layout_for(name)
                                label = lay.label or spec.label or name
                                info = lay.info or spec.info or ""
                                filter_haystack[f"v2_{training_type}_{name}"] = (
                                    f"{name} {label} {info}"
                                )

        # Haystack for filter JS (escaped into a data attribute)
        section_ids = {s: f"v2_{training_type}_section_{s}" for s in section_accordions}
        hay_attr = html.escape(json.dumps(filter_haystack), quote=True)
        sec_attr = html.escape(json.dumps(section_ids), quote=True)
        gr.HTML(
            value=(
                f'<div style="display:none" class="v2-filter-map" '
                f'data-v2-filter-map="{hay_attr}" '
                f'data-v2-sections="{sec_attr}"></div>'
            ),
            visible=True,
            elem_classes=["v2-filter-map-host"],
        )

        # Re-tag section accordions: Gradio Accordion doesn't take data-*,
        # inject via elem_id which filter JS already uses; also set classes.
        # (section keys already in elem_id)

        button_print = gr.Button("Print training command")
        executor = CommandExecutor(headless=headless)

    ordered_names = [n for n in registry.names() if n != "architecture"]
    missing = [n for n in ordered_names if n not in components]
    if missing:
        raise RuntimeError(
            f"tab_builder failed to create components for: {missing[:10]}..."
        )
    ordered_components = [components[n] for n in ordered_names]
    ordered_specs = [registry[n] for n in ordered_names]

    # Accordion list in section order for arch visibility
    section_keys = list(section_accordions.keys())
    section_components = [section_accordions[k] for k in section_keys]

    def apply_architecture(arch_key):
        field_updates = [
            gr.update(visible=spec.supports_arch(arch_key)) for spec in ordered_specs
        ]
        # Hide empty accordions
        section_visible = {s: False for s in section_keys}
        for spec in ordered_specs:
            if spec.supports_arch(arch_key):
                sec = spec.group or layout_for(spec.name).section
                if sec in section_visible:
                    section_visible[sec] = True
        acc_updates = [gr.update(visible=section_visible[s]) for s in section_keys]
        return field_updates + acc_updates

    arch_outputs = ordered_components + section_components
    architecture.change(
        fn=apply_architecture, inputs=[architecture], outputs=arch_outputs
    )

    # Client-side filter (Route A)
    filter_box.change(
        fn=None,
        js=_filter_js(training_type),
        inputs=[filter_box],
        outputs=[filter_box],
    )

    def _build_values(arch_key, *component_values):
        raw = dict(zip(ordered_names, component_values))
        raw["architecture"] = arch_key
        raw.update(derive(raw, arch_key))
        return raw

    def do_save(arch_key, path, save_as, *component_values):
        values = _build_values(arch_key, *component_values)
        if save_as or not path:
            path = get_saveasfile_path(
                path, defaultextension=".toml", extension_name="TOML files (*.toml)"
            )
        if not path:
            return gr.Textbox()
        save_config(registry, values, path)
        return gr.Textbox(value=path)

    def do_open(path):
        if not path or not os.path.isfile(path):
            path = get_file_path(
                path,
                default_extension=".toml",
                extension_name="TOML/JSON files (*.toml *.json)",
            )
        if not path or not os.path.isfile(path):
            return (
                [gr.Textbox()]
                + [gr.update() for _ in ordered_components]
                + [gr.Dropdown()]
            )
        if path.lower().endswith(".json"):
            result = import_json(path, training_type=training_type)
            values = result.values
            arch_key = result.architecture
            if result.unrecognized_keys:
                gr.Warning(
                    "Ignored legacy keys not recognized by v2: "
                    + ", ".join(result.unrecognized_keys)
                )
        else:
            values = load_config(registry, path)
            arch_key = values.get("architecture", default_arch)
        return (
            [gr.Textbox(value=path)]
            + [gr.update(value=values.get(n)) for n in ordered_names]
            + [gr.Dropdown(value=arch_key)]
        )

    def do_train(arch_key, print_only, *component_values):
        values = _build_values(arch_key, *component_values)
        run_config = build_run_config(
            registry, values, arch_key=arch_key, training_type=training_type
        )

        import tempfile
        from datetime import datetime

        output_dir = values.get("output_dir") or tempfile.gettempdir()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        toml_path = os.path.join(output_dir, f"config_{training_type}_v2-{ts}.toml")
        os.makedirs(output_dir, exist_ok=True)
        import toml as toml_lib

        with open(toml_path, "w", encoding="utf-8") as f:
            toml_lib.dump(run_config, f)

        script = arch_scripts.get(arch_key, next(iter(arch_scripts.values())))
        run_cmd = list(SD_SCRIPTS_BACKEND.launcher) + [
            os.path.join(SD_SCRIPTS_BACKEND.script_root, script),
        ]
        # Some trainers declare required=True CLI args that argparse enforces
        # before --config_file is read (e.g. LeCo --prompts_file).
        if required_cli_fields:
            for field_name in required_cli_fields:
                value = run_config.get(field_name) or values.get(field_name)
                if value:
                    run_cmd += [f"--{field_name}", str(value)]
        run_cmd += [
            SD_SCRIPTS_BACKEND.config_file_flag,
            toml_path,
        ]

        if print_only:
            gr.Info(" ".join(run_cmd))
            return
        executor.execute_command(run_cmd=run_cmd, env=setup_environment())

    button_save.click(
        do_save,
        inputs=[architecture, config_file_name, gr.Checkbox(value=False, visible=False)]
        + ordered_components,
        outputs=[config_file_name],
    )
    button_save_as.click(
        do_save,
        inputs=[architecture, config_file_name, gr.Checkbox(value=True, visible=False)]
        + ordered_components,
        outputs=[config_file_name],
    )
    button_open.click(
        do_open,
        inputs=[config_file_name],
        outputs=[config_file_name] + ordered_components + [architecture],
    ).then(fn=apply_architecture, inputs=[architecture], outputs=arch_outputs)

    button_print.click(
        do_train,
        inputs=[architecture, gr.Checkbox(value=True, visible=False)]
        + ordered_components,
    )
    executor.button_run.click(
        do_train,
        inputs=[architecture, gr.Checkbox(value=False, visible=False)]
        + ordered_components,
    )

    return architecture, ordered_names, ordered_components
