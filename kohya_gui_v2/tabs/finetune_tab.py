"""Finetune v2 tab: generic build_tab() assembly + Save/Open/Save-as/Print/
Start/Stop wiring (Move 6, checkpoint B7).
"""

import os

import gradio as gr

from kohya_gui.class_command_executor import CommandExecutor
from kohya_gui.common_gui import get_file_path, get_saveasfile_path, setup_environment

from ..builder import build_components, visible_groups_for
from ..config_io import build_run_config, load_config, save_config
from ..registry import SD_SCRIPTS_BACKEND
from .finetune_derivations import derive
from .finetune_fields import ARCHITECTURE_CHOICES, FINETUNE_REGISTRY

# Registered per architecture in Move 4/5's matrix; single source of truth
# for which sd-scripts script each architecture launches.
ARCHITECTURE_SCRIPTS = {
    "base": "fine_tune.py",
    "sdxl": "sdxl_train.py",
    "sd3": "sd3_train.py",
    "flux1": "flux_train.py",
    "anima": "anima_train.py",
    "lumina": "lumina_train.py",
}


def finetune_tab(headless: bool = False, config=None):
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    with gr.Accordion("Configuration File", open=False):
        with gr.Row():
            config_file_name = gr.Textbox(
                label="Config file", value="", placeholder="./config_finetune_v2.toml"
            )
            button_open = gr.Button("Open")
            button_save = gr.Button("Save")
            button_save_as = gr.Button("Save as")

    architecture = gr.Dropdown(
        choices=ARCHITECTURE_CHOICES, value="base", label="Architecture"
    )

    groups: dict = {}
    components: dict = {}
    for spec in FINETUNE_REGISTRY:
        if spec.gui_only and spec.name == "architecture":
            continue
        group_key = spec.group
        if group_key not in groups:
            groups[group_key] = gr.Column(visible=True)
        with groups[group_key]:
            comp_dict = build_components(type(FINETUNE_REGISTRY)([spec]), config=config)
            components.update(comp_dict)

    def apply_architecture(arch_key):
        visible = visible_groups_for(
            FINETUNE_REGISTRY, arch_key=arch_key, training_type="finetune"
        )
        return [gr.Column(visible=(g in visible)) for g in groups]

    architecture.change(
        fn=apply_architecture, inputs=[architecture], outputs=list(groups.values())
    )

    button_print = gr.Button("Print training command")
    executor = CommandExecutor(headless=headless)

    ordered_names = [n for n in FINETUNE_REGISTRY.names() if n != "architecture"]
    ordered_components = [components[n] for n in ordered_names]

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
        save_config(FINETUNE_REGISTRY, values, path)
        return gr.Textbox(value=path)

    def do_open(path):
        if not path or not os.path.isfile(path):
            path = get_file_path(
                path, default_extension=".toml", extension_name="TOML files (*.toml)"
            )
        if not path or not os.path.isfile(path):
            return (
                [gr.Textbox()]
                + [gr.update() for _ in ordered_components]
                + [gr.Dropdown()]
            )
        values = load_config(FINETUNE_REGISTRY, path)
        arch_key = values.get("architecture", "base")
        return (
            [gr.Textbox(value=path)]
            + [gr.update(value=values.get(n)) for n in ordered_names]
            + [gr.Dropdown(value=arch_key)]
        )

    def do_train(arch_key, print_only, *component_values):
        values = _build_values(arch_key, *component_values)
        run_config = build_run_config(
            FINETUNE_REGISTRY, values, arch_key=arch_key, training_type="finetune"
        )

        import tempfile
        from datetime import datetime

        output_dir = values.get("output_dir") or tempfile.gettempdir()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        toml_path = os.path.join(output_dir, f"config_finetune_v2-{ts}.toml")
        os.makedirs(output_dir, exist_ok=True)
        import toml as toml_lib

        with open(toml_path, "w", encoding="utf-8") as f:
            toml_lib.dump(run_config, f)

        script = ARCHITECTURE_SCRIPTS.get(arch_key, "fine_tune.py")
        run_cmd = list(SD_SCRIPTS_BACKEND.launcher) + [
            os.path.join(SD_SCRIPTS_BACKEND.script_root, script),
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
    ).then(fn=apply_architecture, inputs=[architecture], outputs=list(groups.values()))

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
