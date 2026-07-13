"""Shared training-tab builder for all v2 training types.

Consolidates the six near-identical ``*_tab.py`` modules: accordion sections,
multi-field rows, per-component arch visibility, save/open/train wiring, and
the live client-side parameter filter.

CRITICAL CONTRACT: components may be *created* in section order, but every
event's input/output list is rebuilt from ``registry.names()`` order so
save/open/train never mis-zip values.
"""

from __future__ import annotations

import contextlib
import html
import json
import os
import shlex
import time
from collections import OrderedDict
from typing import Callable, Optional

import gradio as gr

from kohya_gui.class_command_executor import CommandExecutor
from kohya_gui.common_gui import (
    get_any_file_path,
    get_file_path,
    get_folder_path,
    get_saveasfile_path,
    setup_environment,
)

from .builder import build_components
from .config_io import build_run_config, load_config, save_config
from .fields import FieldRegistry, FieldSpec
from .layout_map import (
    OPEN_BY_DEFAULT,
    SECTION_ORDER,
    SECTION_TITLES,
    layout_for,
    path_kind_for,
)
from .legacy_import import import_json
from .registry import SD_SCRIPTS_BACKEND

# Cap inputs per gr.Row (legacy GUI typically packs 2–4)
_MAX_PER_ROW = 4


def _accelerate_launch_flags(values: dict) -> list:
    """Build the `accelerate launch [flags] <script>` flags that must sit
    ahead of the script path -- these are launcher-level options consumed by
    `accelerate` itself, not by any sd-scripts trainer, so they never belong
    in the --config_file TOML. Ported field-for-field from
    kohya_gui/class_accelerate_launch.py::AccelerateLaunch.run_cmd.
    """
    flags: list = []
    dynamo_backend = values.get("dynamo_backend")
    if dynamo_backend:
        flags += ["--dynamo_backend", dynamo_backend]

    dynamo_mode = values.get("dynamo_mode")
    if dynamo_mode:
        flags += ["--dynamo_mode", dynamo_mode]

    if values.get("dynamo_use_fullgraph"):
        flags.append("--dynamo_use_fullgraph")

    if values.get("dynamo_use_dynamic"):
        flags.append("--dynamo_use_dynamic")

    extra_args = values.get("extra_accelerate_launch_args") or ""
    if extra_args:
        for arg in extra_args.replace('"', "").split():
            flags.append(shlex.quote(arg))

    gpu_ids = values.get("gpu_ids") or ""
    if gpu_ids:
        flags += ["--gpu_ids", shlex.quote(gpu_ids)]

    main_process_port = int(values.get("main_process_port") or 0)
    if main_process_port > 0:
        flags += ["--main_process_port", str(main_process_port)]

    mixed_precision = values.get("mixed_precision")
    if mixed_precision:
        flags += ["--mixed_precision", shlex.quote(mixed_precision)]

    if values.get("multi_gpu"):
        flags.append("--multi_gpu")

    num_processes = int(values.get("num_processes") or 0)
    if num_processes > 0:
        flags += ["--num_processes", str(num_processes)]

    num_machines = int(values.get("num_machines") or 0)
    if num_machines > 0:
        flags += ["--num_machines", str(num_machines)]

    num_cpu_threads_per_process = int(values.get("num_cpu_threads_per_process") or 0)
    if num_cpu_threads_per_process > 0:
        flags += ["--num_cpu_threads_per_process", str(num_cpu_threads_per_process)]

    return flags


def _filter_js(training_type: str) -> str:
    """Client-side filter scoped to one training tab.

    Matches against a pre-built haystack per elem_id; only toggles the
    ``v2-filter-hidden`` class so Gradio arch visibility (inline styles)
    is never overwritten.

    Hiding happens at three nested levels so a non-matching field never
    leaves a visible trace of itself behind:
      1. the field's "hideable unit" -- its own ``.v2-field`` for a bare
         field, or the wrapping ``.v2-path-row`` for a path field (so its
         Browse button, which carries no searchable text of its own, hides
         together with the field instead of being orphaned);
      2. the shared ``.v2-row-group`` (the Row/Column `_pack_rows` packed it
         into) collapses entirely once every hideable unit inside it is
         hidden, instead of leaving an empty slim box;
      3. the enclosing accordion section, as before.
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
  const rowGroups = root.querySelectorAll(".v2-row-group");
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

  if (!q) {{
    fields.forEach((el) => el.classList.remove("v2-filter-hidden"));
    root.querySelectorAll(".v2-path-row").forEach((el) => el.classList.remove("v2-filter-hidden"));
    rowGroups.forEach((el) => el.classList.remove("v2-filter-hidden"));
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
    // Hide at the path-row level (field + Browse button together) when the
    // field is wrapped in one; otherwise hide the field itself.
    const hideTarget = el.closest(".v2-path-row") || el;
    if (match) hideTarget.classList.remove("v2-filter-hidden");
    else if (q) hideTarget.classList.add("v2-filter-hidden");
    const section = el.closest(".v2-section");
    if (section && match) matchCount.set(section, (matchCount.get(section) || 0) + 1);
  }});

  if (q) {{
    rowGroups.forEach((group) => {{
      const units = group.querySelectorAll(".v2-field, .v2-path-row");
      let anyVisible = false;
      units.forEach((u) => {{
        if (!u.classList.contains("v2-filter-hidden")) anyVisible = true;
      }});
      if (anyVisible) group.classList.remove("v2-filter-hidden");
      else group.classList.add("v2-filter-hidden");
    }});
  }}

  accordions.forEach((acc) => {{
    if (!q) {{
      acc.classList.remove("v2-filter-hidden");
      const prev = acc.getAttribute(openStateKey);
      const lw = acc.querySelector(".label-wrap");
      // Restore prior open/closed by clicking Gradio's accordion header.
      // Relies on Gradio-internal .label-wrap.open DOM — may need a revisit
      // on major Gradio upgrades; class-only hide/show is the hard requirement.
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


def _sort_specs_for_rows(specs: list[FieldSpec]) -> list[FieldSpec]:
    """Stable-sort so specs that share a layout row key become adjacent.

    FieldSpecs are stored alphabetically in the registry, but legacy row
    membership groups fields that may sit far apart alphabetically. Event
    wiring is rebuilt from ``registry.names()`` regardless of creation order,
    so reordering for layout is safe.
    """
    first_row_index: dict[str, int] = {}
    for i, spec in enumerate(specs):
        key = layout_for(spec.name).row
        if key is not None and key not in first_row_index:
            first_row_index[key] = i

    def sort_key(pair: tuple[int, FieldSpec]) -> tuple[int, int]:
        i, spec = pair
        key = layout_for(spec.name).row
        group = first_row_index[key] if key is not None else i
        return (group, i)

    return [spec for _, spec in sorted(enumerate(specs), key=sort_key)]


def _pack_rows(specs: list[FieldSpec]) -> list[list[FieldSpec]]:
    """Pack specs sharing a layout row key into chunks of ≤_MAX_PER_ROW.

    Callers should pass specs already ordered by ``_sort_specs_for_rows``.
    """
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
        path_buttons: dict = {}
        path_rows: dict = {}
        section_accordions: dict = {}
        filter_haystack: dict[str, str] = {}

        for section, section_specs in by_section.items():
            title = SECTION_TITLES.get(section, section.replace("_", " ").title())
            is_open = section in OPEN_BY_DEFAULT
            with gr.Accordion(
                title,
                open=is_open,
                elem_classes=["v2-section", f"v2-section-{section}"],
                elem_id=f"v2_{training_type}_section_{section}",
            ) as acc:
                # data attribute for filter JS (Gradio may not pass arbitrary attrs;
                # we also mirror via a hidden HTML map below)
                section_accordions[section] = acc
                # Cluster shared row keys before packing (registry order is alpha)
                for row_specs in _pack_rows(_sort_specs_for_rows(section_specs)):
                    # "v2-row-group" is always present (Row or Column) so the
                    # filter can collapse the whole packed row once every
                    # field inside it is hidden, instead of leaving an empty
                    # slim box (the previously-observed "residual lines").
                    ctx = (
                        gr.Row(elem_classes=["v2-field-row", "v2-row-group"])
                        if len(row_specs) > 1
                        else gr.Column(elem_classes=["v2-row-group"])
                    )
                    with ctx:
                        for spec in row_specs:
                            # Single-spec registry so build_components stays pure
                            single = type(registry)([spec])
                            kind = path_kind_for(spec.name)
                            # Fields with a legacy Browse-dialog equivalent get an
                            # inner Row pairing the field with a "📁 Browse"
                            # button (kohya_gui/common_gui.get_folder_path /
                            # get_any_file_path — same native dialogs the
                            # legacy GUI and musubi_tuner_gui's path_field()
                            # use), instead of the bare textbox v2 rendered
                            # before. The button is tracked separately from
                            # `components` so it never enters the
                            # registry-order wiring contract.
                            path_row = (
                                gr.Row(elem_classes=["v2-path-row"]) if kind else None
                            )
                            if kind:
                                path_rows[spec.name] = path_row
                            field_ctx = path_row if kind else contextlib.nullcontext()
                            with field_ctx:
                                built = build_components(
                                    single,
                                    config=config,
                                    elem_id_prefix=f"v2_{training_type}",
                                    elem_classes=["v2-field"],
                                )
                                if kind:
                                    textbox = next(iter(built.values()))
                                    textbox.scale = 1
                                    icon = "📂" if kind == "folder" else "📄"
                                    # scale=0 means "don't flex-grow, size to
                                    # content/min_width" in Gradio; scale=1
                                    # (like the textbox) would claim an equal
                                    # share of the row instead of staying a
                                    # compact icon button.
                                    button = gr.Button(
                                        icon,
                                        scale=0,
                                        min_width=0,
                                        elem_classes=[
                                            "v2-path-browse",
                                            f"v2-path-browse-{kind}",
                                        ],
                                    )
                                    fn = (
                                        get_folder_path
                                        if kind == "folder"
                                        else get_any_file_path
                                    )
                                    button.click(
                                        fn=fn,
                                        inputs=[textbox],
                                        outputs=[textbox],
                                        show_progress=False,
                                    )
                                    path_buttons[spec.name] = button
                            for name, comp in built.items():
                                components[name] = comp
                                lay = layout_for(name)
                                label = lay.label or spec.label or name
                                info = lay.info or spec.info or ""
                                filter_haystack[f"v2_{training_type}_{name}"] = (
                                    f"{name} {label} {info}"
                                )

        # Haystack for filter JS (escaped into a data attribute)
        hay_attr = html.escape(json.dumps(filter_haystack), quote=True)
        gr.HTML(
            value=(
                f'<div style="display:none" class="v2-filter-map" '
                f'data-v2-filter-map="{hay_attr}"></div>'
            ),
            visible=True,
            elem_classes=["v2-filter-map-host"],
        )

        button_print = gr.Button("Print training command")
        executor = CommandExecutor(headless=headless)
        run_state = gr.Textbox(value="0", visible=False)

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
    # Browse buttons hide alongside their field so an arch-hidden field never
    # leaves an orphaned, useless button visible next to it. The wrapping Row
    # itself must also be toggled -- toggling only the child Textbox/Button
    # can leave the Row's own flex box stuck at a stale size (observed as an
    # orphaned Browse button with no visible field when a path field flips
    # from hidden to visible after the initial render).
    path_button_names = [n for n in ordered_names if n in path_buttons]
    path_button_components = [path_buttons[n] for n in path_button_names]
    path_row_components = [path_rows[n] for n in path_button_names]

    def _spec_visible(spec: FieldSpec, arch_key, raw_values: dict) -> bool:
        return spec.supports_arch(arch_key) and spec.supports_visibility(raw_values)

    def apply_visibility(arch_key, *component_values):
        raw_values = dict(zip(ordered_names, component_values))
        field_visible = {
            spec.name: _spec_visible(spec, arch_key, raw_values) for spec in ordered_specs
        }
        field_updates = [gr.update(visible=field_visible[spec.name]) for spec in ordered_specs]
        # Hide empty accordions
        section_visible = {s: False for s in section_keys}
        for spec in ordered_specs:
            if field_visible[spec.name]:
                sec = spec.group or layout_for(spec.name).section
                if sec in section_visible:
                    section_visible[sec] = True
        acc_updates = [gr.update(visible=section_visible[s]) for s in section_keys]
        button_updates = [gr.update(visible=field_visible[n]) for n in path_button_names]
        row_updates = [gr.update(visible=field_visible[n]) for n in path_button_names]
        return field_updates + acc_updates + button_updates + row_updates

    arch_outputs = (
        ordered_components
        + section_components
        + path_button_components
        + path_row_components
    )
    # Components whose current value some spec's visibility depends on must
    # also trigger a recompute (not just the architecture dropdown) -- e.g. a
    # LoRA_type dropdown controlling conv_dim's visibility. Empty for tabs
    # where no FieldSpec sets visible_when (a pure no-op, same as before).
    visibility_dep_names: set = set()
    for spec in ordered_specs:
        if spec.visible_when is not None:
            visibility_dep_names.update(spec.visible_when.deps)
    visibility_trigger_components = [
        components[n] for n in ordered_names if n in visibility_dep_names
    ]

    architecture.change(
        fn=apply_visibility,
        inputs=[architecture] + ordered_components,
        outputs=arch_outputs,
    )
    for trigger in visibility_trigger_components:
        trigger.change(
            fn=apply_visibility,
            inputs=[architecture] + ordered_components,
            outputs=arch_outputs,
        )

    # Client-side filter (Route A): .input fires on user typing only (not
    # programmatic value updates during Open), and has no outputs so the JS
    # cannot echo the value back into a re-trigger loop.
    filter_box.input(
        fn=None,
        js=_filter_js(training_type),
        inputs=[filter_box],
        outputs=None,
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
                default_extension=".toml .json",
                extension_name="TOML/JSON files",
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

    def _missing_required_fields(values: dict) -> list[str]:
        """Fields that are always required for a real run but silently
        default to "" when a user clicks Train/Print without ever loading a
        preset (fresh page, or Save-as/New-tab with a blank form). Their
        absence doesn't fail fast in v2 -- it surfaces as an opaque
        traceback from deep inside sd-scripts (e.g. dreambooth_dataset.py's
        `assert resolution is not None`) or, worse, a silently-wrong run
        (blank output_dir falling back to the OS temp dir). Only check
        fields the registry actually declares, since not every training
        type uses all of them (e.g. LeCo has no resolution/train_data_dir).
        """
        checks = (
            ("pretrained_model_name_or_path", "Pretrained model name or path"),
            ("train_data_dir", "Image folder"),
            ("resolution", "Resolution"),
            ("output_dir", "Output directory"),
        )
        field_names = {spec.name for spec in registry}
        missing = []
        for name, label in checks:
            if name in field_names and not str(values.get(name) or "").strip():
                missing.append(label)
        return missing

    def do_train(arch_key, print_only, *component_values):
        values = _build_values(arch_key, *component_values)
        missing = _missing_required_fields(values)
        if missing:
            gr.Warning(
                "Cannot start training -- missing required field(s): "
                + ", ".join(missing)
                + ". Did you forget to open a config file (Open button) before training?"
            )
            return gr.Button(), gr.Button(), gr.Textbox()
        run_config = build_run_config(
            registry,
            values,
            arch_key=arch_key,
            training_type=training_type,
            zero_survives_false=training_type in ("dreambooth", "finetune"),
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
        run_cmd = (
            list(SD_SCRIPTS_BACKEND.launcher)
            + _accelerate_launch_flags(values)
            + [os.path.join(SD_SCRIPTS_BACKEND.script_root, script)]
        )
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
            return gr.Button(), gr.Button(), gr.Textbox()

        executor.execute_command(run_cmd=run_cmd, env=setup_environment())
        return (
            gr.Button(visible=False or headless),
            gr.Button(visible=True),
            gr.Textbox(value=str(time.time())),
        )

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
    ).then(
        fn=apply_visibility,
        inputs=[architecture] + ordered_components,
        outputs=arch_outputs,
    )

    button_print.click(
        do_train,
        inputs=[architecture, gr.Checkbox(value=True, visible=False)]
        + ordered_components,
        outputs=[executor.button_run, executor.button_stop_training, run_state],
    )
    executor.button_run.click(
        do_train,
        inputs=[architecture, gr.Checkbox(value=False, visible=False)]
        + ordered_components,
        outputs=[executor.button_run, executor.button_stop_training, run_state],
        show_progress=False,
    )

    # run_state's value changes on every real (non-print) launch; that change
    # event drives wait_for_training_to_end, which blocks until the
    # subprocess exits and then flips the buttons back -- same run_state
    # relay the legacy *_gui.py tabs use (e.g. kohya_gui/lora_gui.py's
    # run_state.change wiring) so Start training reliably flips to Stop
    # training and back.
    run_state.change(
        fn=executor.wait_for_training_to_end,
        outputs=[executor.button_run, executor.button_stop_training],
    )
    executor.button_stop_training.click(
        executor.kill_command,
        outputs=[executor.button_run, executor.button_stop_training],
    )

    return architecture, ordered_names, ordered_components
