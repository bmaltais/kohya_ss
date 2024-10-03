import gradio as gr
import os
import argparse
from kohya_gui.class_gui_config import KohyaSSGUIConfig
from kohya_gui.dreambooth_gui import dreambooth_tab
from kohya_gui.finetune_gui import finetune_tab
from kohya_gui.textual_inversion_gui import ti_tab
from kohya_gui.utilities import utilities_tab
from kohya_gui.lora_gui import lora_tab
from kohya_gui.class_lora_tab import LoRATools

from kohya_gui.custom_logging import setup_logging
from kohya_gui.localization_ext import add_javascript


def UI(**kwargs):
    add_javascript(kwargs.get("language"))
    css = ""

    headless = kwargs.get("headless", False)
    log.info(f"headless: {headless}")

    if os.path.exists("./assets/style.css"):
        with open(os.path.join("./assets/style.css"), "r", encoding="utf8") as file:
            log.debug("Load CSS...")
            css += file.read() + "\n"

    if os.path.exists("./.release"):
        with open(os.path.join("./.release"), "r", encoding="utf8") as file:
            release = file.read()

    if os.path.exists("./README.md"):
        with open(os.path.join("./README.md"), "r", encoding="utf8") as file:
            README = file.read()

    interface = gr.Blocks(
        css=css, title=f"Kohya_ss GUI {release}", theme=gr.themes.Default()
    )

    config = KohyaSSGUIConfig(config_file_path=kwargs.get("config"))

    if config.is_config_loaded():
        log.info(f"Loaded default GUI values from '{kwargs.get('config')}'...")

    use_shell_flag = True
    # if os.name == "posix":
    #     use_shell_flag = True
        
    use_shell_flag = config.get("settings.use_shell", use_shell_flag)
        
    if kwargs.get("do_not_use_shell", False):
        use_shell_flag = False
        
    if use_shell_flag:
        log.info("Using shell=True when running external commands...")

    with interface:
        with gr.Tab("Dreambooth"):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = dreambooth_tab(
                headless=headless, config=config, use_shell_flag=use_shell_flag
            )
        with gr.Tab("LoRA"):
            lora_tab(headless=headless, config=config, use_shell_flag=use_shell_flag)
        with gr.Tab("Textual Inversion"):
            ti_tab(headless=headless, config=config, use_shell_flag=use_shell_flag)
        with gr.Tab("Finetuning"):
            finetune_tab(
                headless=headless, config=config, use_shell_flag=use_shell_flag
            )
        with gr.Tab("Utilities"):
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
            gr.Markdown(f"kohya_ss GUI release {release}")
            with gr.Tab("README"):
                gr.Markdown(README)

        htmlStr = f"""
        <html>
            <body>
                <div class="ver-class">{release}</div>
            </body>
        </html>
        """
        gr.HTML(htmlStr)
    # Show the interface
    launch_kwargs = {}
    username = kwargs.get("username")
    password = kwargs.get("password")
    server_port = kwargs.get("server_port", 0)
    inbrowser = kwargs.get("inbrowser", False)
    share = kwargs.get("share", False)
    do_not_share = kwargs.get("do_not_share", False)
    server_name = kwargs.get("listen")
    root_path = kwargs.get("root_path", None)
    debug = kwargs.get("debug", False)

    launch_kwargs["server_name"] = server_name
    if username and password:
        launch_kwargs["auth"] = (username, password)
    if server_port > 0:
        launch_kwargs["server_port"] = server_port
    if inbrowser:
        launch_kwargs["inbrowser"] = inbrowser
    if do_not_share:
        launch_kwargs["share"] = False
    else:
        if share:
            launch_kwargs["share"] = share
    if root_path:
        launch_kwargs["root_path"] = root_path
    if debug:
        launch_kwargs["debug"] = True
    interface.launch(**launch_kwargs)


if __name__ == "__main__":
    # torch.cuda.set_per_process_memory_fraction(0.48)
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
        "--username", type=str, default="", help="Username for authentication"
    )
    parser.add_argument(
        "--password", type=str, default="", help="Password for authentication"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=0,
        help="Port to run the server listener on",
    )
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--share", action="store_true", help="Share the gradio UI")
    parser.add_argument(
        "--headless", action="store_true", help="Is the server headless"
    )
    parser.add_argument(
        "--language", type=str, default=None, help="Set custom language"
    )

    parser.add_argument("--use-ipex", action="store_true", help="Use IPEX environment")
    parser.add_argument("--use-rocm", action="store_true", help="Use ROCm environment")

    parser.add_argument(
        "--do_not_use_shell", action="store_true", help="Enforce not to use shell=True when running external commands"
    )

    parser.add_argument(
        "--do_not_share", action="store_true", help="Do not share the gradio UI"
    )

    parser.add_argument(
        "--root_path", type=str, default=None, help="`root_path` for Gradio to enable reverse proxy support. e.g. /kohya_ss"
    )

    args = parser.parse_args()

    # Set up logging
    log = setup_logging(debug=args.debug)

    UI(**vars(args))
