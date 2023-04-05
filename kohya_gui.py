import gradio as gr
import os
import argparse
from dreambooth_gui import dreambooth_tab
from finetune_gui import finetune_tab
from textual_inversion_gui import ti_tab
from library.utilities import utilities_tab
from library.extract_lora_gui import gradio_extract_lora_tab
from library.extract_lycoris_locon_gui import gradio_extract_lycoris_locon_tab
from library.merge_lora_gui import gradio_merge_lora_tab
from library.resize_lora_gui import gradio_resize_lora_tab
from lora_gui import lora_tab

import yaml


def UI(**kwargs):
    css = ''

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            print('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Blocks(css=css, title='Kohya_ss GUI')

    with interface:
        with gr.Tab('Dreambooth'):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = dreambooth_tab()
        with gr.Tab('Dreambooth LoRA'):
            lora_tab()
        with gr.Tab('Dreambooth TI'):
            ti_tab()
        with gr.Tab('Finetune'):
            finetune_tab()
        with gr.Tab('Utilities'):
            utilities_tab(
                train_data_dir_input=train_data_dir_input,
                reg_data_dir_input=reg_data_dir_input,
                output_dir_input=output_dir_input,
                logging_dir_input=logging_dir_input,
                enable_copy_info_button=True,
            )
            gradio_extract_lora_tab()
            gradio_extract_lycoris_locon_tab()
            gradio_merge_lora_tab()
            gradio_resize_lora_tab()

    # Show the interface
    launch_kwargs = {}
    username = kwargs.get('username')
    password = kwargs.get('password')
    server_port = kwargs.get('server_port', 0)
    inbrowser = kwargs.get('inbrowser', False)
    share = kwargs.get('share', False)
    server_name = kwargs.get('listen')

    launch_kwargs['server_name'] = server_name
    if username and password:
        launch_kwargs['auth'] = (username, password)
    if server_port > 0:
        launch_kwargs['server_port'] = server_port
    if inbrowser:
        launch_kwargs['inbrowser'] = inbrowser
    if share:
        launch_kwargs['share'] = share
    interface.launch(**launch_kwargs)


def load_config(_config_file):
    if os.path.isfile(_config_file):
        with open(_config_file, "r") as f:
            _config_data = yaml.safe_load(f)
    else:
        _config_data = None
    return _config_data


if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)

    """
    Argument priority order: CLI arguments > install_config.yml config file > default script values.

    1. Define default_args dictionary with default values for each argument.
    2. If a configuration file is found and contains the "kohya_gui_arguments" section,
       update the default_args dictionary with the values from the configuration file.
    3. Initialize the argparse.ArgumentParser() object and add arguments with updated
       default values from the default_args dictionary.
    4. When calling parser.parse_args(), any command-line arguments provided by the user
       will override the corresponding default values in the parser. If a command-line
       argument is not provided for a specific parameter, the parser will use the default
       value (which be from the config file, if present, or the script's default values).
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", "--file", dest="config_file", default="install_config.yaml",
                        help="Path to the configuration file.")
    _file_args, _ = parser.parse_known_args()
    config_file = _file_args.config_file
    config_data = load_config(config_file)

    # Define the default arguments first
    default_args = {
        '--listen': '127.0.0.1',
        '--username': '',
        '--password': '',
        '--server_port': 0,
        '--inbrowser': False,
        '--share': False
    }

    # Update the default arguments with values from the config file
    if config_data and "kohya_gui_arguments" in config_data:
        for arg in config_data["kohya_gui_arguments"]:
            long = arg["long"]
            default = arg["default"]
            default_args[long] = default

    # Add arguments to the parser with updated default values
    parser.add_argument(
        '--listen',
        type=str,
        default=default_args['--listen'],
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default=default_args['--username'], help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default=default_args['--password'], help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=default_args['--server_port'],
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', default=default_args['--inbrowser'], help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', default=default_args['--share'], help='Share the gradio UI'
    )

    args = parser.parse_args()

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
    )
