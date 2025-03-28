import os
import sys
import argparse
import subprocess
import contextlib
import gradio as gr

from kohya_gui.class_gui_config import KohyaSSGUIConfig
from kohya_gui.dreambooth_gui import dreambooth_tab
from kohya_gui.finetune_gui import finetune_tab
from kohya_gui.textual_inversion_gui import ti_tab
from kohya_gui.utilities import utilities_tab
from kohya_gui.lora_gui import lora_tab
from kohya_gui.class_lora_tab import LoRATools
from kohya_gui.custom_logging import setup_logging
from kohya_gui.localization_ext import add_javascript

PYTHON = sys.executable
project_dir = os.path.dirname(os.path.abspath(__file__))

# Function to read file content, suppressing any FileNotFoundError
def read_file_content(file_path):
    with contextlib.suppress(FileNotFoundError):
        with open(file_path, "r", encoding="utf8") as file:
            return file.read()
    return ""

# Function to initialize the Gradio UI interface
def initialize_ui_interface(config, headless, use_shell, release_info, readme_content):
    # Load custom CSS if available
    css = read_file_content("./assets/style.css")

    # Create the main Gradio Blocks interface
    ui_interface = gr.Blocks(css=css, title=f"Kohya_ss GUI {release_info}", theme=gr.themes.Default())
    with ui_interface:
        # Create tabs for different functionalities
        with gr.Tab("Dreambooth"):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = dreambooth_tab(headless=headless, config=config, use_shell_flag=use_shell)
        with gr.Tab("LoRA"):
            lora_tab(headless=headless, config=config, use_shell_flag=use_shell)
        with gr.Tab("Textual Inversion"):
            ti_tab(headless=headless, config=config, use_shell_flag=use_shell)
        with gr.Tab("Finetuning"):
            finetune_tab(headless=headless, config=config, use_shell_flag=use_shell)
        with gr.Tab("Utilities"):
            # Utilities tab requires inputs from the Dreambooth tab
            utilities_tab(
                train_data_dir_input=train_data_dir_input,
                reg_data_dir_input=reg_data_dir_input,
                output_dir_input=output_dir_input,
                logging_dir_input=logging_dir_input,
                headless=headless,
                config=config,
            )
            with gr.Tab("LoRA"):
                _ = LoRATools(headless=headless)
        with gr.Tab("About"):
            # About tab to display release information and README content
            gr.Markdown(f"kohya_ss GUI release {release_info}")
            with gr.Tab("README"):
                gr.Markdown(readme_content)

        # Display release information in a div element
        gr.Markdown(f"<div class='ver-class'>{release_info}</div>")

    return ui_interface

# Function to configure and launch the UI
def UI(**kwargs):
    # Add custom JavaScript if specified
    add_javascript(kwargs.get("language"))
    log.info(f"headless: {kwargs.get('headless', False)}")

    # Load release and README information
    release_info = read_file_content("./.release")
    readme_content = read_file_content("./README.md")

    # Load configuration from the specified file
    config = KohyaSSGUIConfig(config_file_path=kwargs.get("config"))
    if config.is_config_loaded():
        log.info(f"Loaded default GUI values from '{kwargs.get('config')}'...")

    # Determine if shell should be used for running external commands
    use_shell = not kwargs.get("do_not_use_shell", False) and config.get("settings.use_shell", True)
    if use_shell:
        log.info("Using shell=True when running external commands...")

    # Initialize the Gradio UI interface
    ui_interface = initialize_ui_interface(config, kwargs.get("headless", False), use_shell, release_info, readme_content)

    # Construct launch parameters using dictionary comprehension
    launch_params = {
        "server_name": kwargs.get("listen"),
        "auth": (kwargs["username"], kwargs["password"]) if kwargs.get("username") and kwargs.get("password") else None,
        "server_port": kwargs.get("server_port", 0) if kwargs.get("server_port", 0) > 0 else None,
        "inbrowser": kwargs.get("inbrowser", False),
        "share": False if kwargs.get("do_not_share", False) else kwargs.get("share", False),
        "root_path": kwargs.get("root_path", None),
        "debug": kwargs.get("debug", False),
    }
  
    # This line filters out any key-value pairs from `launch_params` where the value is `None`, ensuring only valid parameters are passed to the `launch` function.
    launch_params = {k: v for k, v in launch_params.items() if v is not None}

    # Launch the Gradio interface with the specified parameters
    ui_interface.launch(**launch_params)

# Function to initialize argument parser for command-line arguments
def initialize_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.toml", help="Path to the toml config file for interface defaults")
    parser.add_argument("--debug", action="store_true", help="Debug on")
    parser.add_argument("--listen", type=str, default="127.0.0.1", help="IP to listen on for connections to Gradio")
    parser.add_argument("--username", type=str, default="", help="Username for authentication")
    parser.add_argument("--password", type=str, default="", help="Password for authentication")
    parser.add_argument("--server_port", type=int, default=0, help="Port to run the server listener on")
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--share", action="store_true", help="Share the gradio UI")
    parser.add_argument("--headless", action="store_true", help="Is the server headless")
    parser.add_argument("--language", type=str, default=None, help="Set custom language")
    parser.add_argument("--use-ipex", action="store_true", help="Use IPEX environment")
    parser.add_argument("--use-rocm", action="store_true", help="Use ROCm environment")
    parser.add_argument("--do_not_use_shell", action="store_true", help="Enforce not to use shell=True when running external commands")
    parser.add_argument("--do_not_share", action="store_true", help="Do not share the gradio UI")
    parser.add_argument("--requirements", type=str, default=None, help="requirements file to use for validation")
    parser.add_argument("--root_path", type=str, default=None, help="`root_path` for Gradio to enable reverse proxy support. e.g. /kohya_ss")
    parser.add_argument("--noverify", action="store_true", help="Disable requirements verification")
    return parser

if __name__ == "__main__":
    # Initialize argument parser and parse arguments
    parser = initialize_arg_parser()
    args = parser.parse_args()

    # Set up logging based on the debug flag
    log = setup_logging(debug=args.debug)

    # Verify requirements unless `noverify` flag is set
    if args.noverify:
        log.warning("Skipping requirements verification.")
    else:
        # Run the validation command to verify requirements
        validation_command = [PYTHON, os.path.join(project_dir, "setup", "validate_requirements.py")]
        
        if args.requirements is not None:
            validation_command.append(f"--requirements={args.requirements}")
            
        subprocess.run(validation_command, check=True)

    # Launch the UI with the provided arguments
    UI(**vars(args))