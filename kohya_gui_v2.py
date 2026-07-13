import os
import sys
import argparse
import contextlib
import gradio as gr

from kohya_gui.class_gui_config import KohyaSSGUIConfig
from kohya_gui.custom_logging import setup_logging

from kohya_gui_v2.placeholder_tab import placeholder_tab
from kohya_gui_v2.tabs.anima_lllite_tab import anima_lllite_tab
from kohya_gui_v2.tabs.dreambooth_tab import dreambooth_tab
from kohya_gui_v2.tabs.finetune_tab import finetune_tab
from kohya_gui_v2.tabs.leco_tab import leco_tab
from kohya_gui_v2.tabs.lora_tab import lora_tab
from kohya_gui_v2.tabs.textual_inversion_tab import textual_inversion_tab

project_dir = os.path.dirname(os.path.abspath(__file__))


def read_file_content(file_path):
    with contextlib.suppress(FileNotFoundError):
        with open(file_path, "r", encoding="utf8") as file:
            return file.read()
    return ""


def initialize_ui_interface(config, release_info, headless=False):
    # v2 reuses the legacy stylesheet verbatim (tooltip reveal, folder-button
    # chrome, .styler/.ver-class fixes) so both GUIs render identically off
    # one source, then layers only v2-specific additions (section tints,
    # filter, row spacing) on top. Never fork the shared rules into a second,
    # independently-drifting copy.
    css = (
        read_file_content("./assets/style.css")
        + "\n"
        + read_file_content("./assets/style_v2.css")
    )

    # Same hover-info tooltip script the legacy GUI uses.
    enable_info_tooltip = True
    if config is not None:
        try:
            enable_info_tooltip = config.get("settings.enable_info_tooltip", True)
        except Exception:
            enable_info_tooltip = True
    info_tooltip_js = read_file_content("./assets/js/info_tooltip.js")
    # Native browser tooltip for the path-field Browse buttons (Gradio's
    # Button has no info= slot). Delegated on mouseover -- like
    # info_tooltip.js -- rather than a DOMContentLoaded pass, since these
    # buttons exist inside Gradio's client-rendered tree, not the static
    # head-injected HTML.
    path_browse_tooltip_js = """
document.addEventListener("mouseover", function (e) {
    const btn = e.target.closest && e.target.closest(".v2-path-browse");
    if (btn && !btn.title) {
        btn.title = btn.classList.contains("v2-path-browse-folder")
            ? "Browse for a folder"
            : "Browse for a file";
    }
}, true);
"""
    head = (
        f'<script type="text/javascript">'
        f"window.KOHYA_INFO_TOOLTIP_ENABLED = {str(bool(enable_info_tooltip)).lower()};"
        f"</script>"
        f'<script type="text/javascript">{info_tooltip_js}</script>'
        f'<script type="text/javascript">{path_browse_tooltip_js}</script>'
    )

    ui_interface = gr.Blocks(
        css=css,
        head=head,
        title=f"Kohya_ss GUI v2 (preview) {release_info}",
        theme=gr.themes.Default(),
        elem_classes=["v2-app"],
    )
    with ui_interface:
        with gr.Tab("LoRA"):
            lora_tab(headless=headless, config=config)

        with gr.Tab("DreamBooth"):
            dreambooth_tab(headless=headless, config=config)

        with gr.Tab("Finetune"):
            finetune_tab(headless=headless, config=config)

        with gr.Tab("Textual Inversion"):
            textual_inversion_tab(headless=headless, config=config)

        with gr.Tab("LeCo"):
            leco_tab(headless=headless, config=config)

        with gr.Tab("Anima LLLite"):
            anima_lllite_tab(headless=headless, config=config)

        with gr.Tab("Preview"):
            placeholder_tab()

        gr.Markdown(
            f"<div class='ver-class'>kohya_gui_v2 preview - {release_info}</div>"
        )

    return ui_interface


def UI(**kwargs):
    log.info(f"headless: {kwargs.get('headless', False)}")

    release_info = read_file_content("./.release")

    config_file_path = kwargs.get("config") or "./config.toml"
    config = KohyaSSGUIConfig(config_file_path=config_file_path)
    if config.is_config_loaded():
        log.info(f"Loaded default GUI values from '{config_file_path}'...")

    ui_interface = initialize_ui_interface(
        config, release_info, headless=kwargs.get("headless", False)
    )

    launch_params = {
        "server_name": kwargs.get("listen"),
        "server_port": (
            kwargs.get("server_port", 0) if kwargs.get("server_port", 0) > 0 else None
        ),
        "inbrowser": kwargs.get("inbrowser", False),
        "share": kwargs.get("share", False),
        "debug": kwargs.get("debug", False),
        "allowed_paths": config.allowed_paths,
    }
    launch_params = {k: v for k, v in launch_params.items() if v is not None}

    ui_interface.launch(**launch_params)


def initialize_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config.toml",
        help="Path to the toml config file for interface defaults",
    )
    parser.add_argument("--debug", action="store_true", help="Debug on")
    parser.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1",
        help="IP to listen on for connections to Gradio",
    )
    parser.add_argument(
        "--server_port", type=int, default=0, help="Port to run the server listener on"
    )
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--share", action="store_true", help="Share the gradio UI")
    parser.add_argument(
        "--headless", action="store_true", help="Is the server headless"
    )
    return parser


if __name__ == "__main__":
    parser = initialize_arg_parser()
    args = parser.parse_args()

    log = setup_logging(debug=args.debug)

    UI(**vars(args))
