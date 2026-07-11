import os
import sys
import argparse
import contextlib
import gradio as gr

from kohya_gui.class_gui_config import KohyaSSGUIConfig
from kohya_gui.custom_logging import setup_logging

from kohya_gui_v2.placeholder_tab import placeholder_tab

project_dir = os.path.dirname(os.path.abspath(__file__))


def read_file_content(file_path):
    with contextlib.suppress(FileNotFoundError):
        with open(file_path, "r", encoding="utf8") as file:
            return file.read()
    return ""


def initialize_ui_interface(config, release_info):
    css = read_file_content("./assets/style.css")

    ui_interface = gr.Blocks(
        css=css,
        title=f"Kohya_ss GUI v2 (preview) {release_info}",
        theme=gr.themes.Default(),
    )
    with ui_interface:
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

    ui_interface = initialize_ui_interface(config, release_info)

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
