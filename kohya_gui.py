#!/usr/bin/env python3

import logging
import platform
import re
from datetime import datetime

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
from library.extract_lora_from_dylora_gui import gradio_extract_dylora_tab
from library.merge_lycoris_gui import gradio_merge_lycoris_tab
from lora_gui import lora_tab


def UI(**kwargs):
    css = ''

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            print('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Blocks(css=css, title='Kohya_ss GUI', theme=gr.themes.Default())

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
            gradio_extract_dylora_tab()
            gradio_extract_lora_tab()
            gradio_extract_lycoris_locon_tab()
            gradio_merge_lora_tab()
            gradio_merge_lycoris_tab()
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


# This custom action was added so that the v option could be used Windows-style with integers (-v 3) setting the
# verbosity and Unix style (-vvv).
class CountOccurrencesAction(argparse.Action):
    def __call__(self, _parser, namespace, values, option_string=None):
        # If value is a string, check if it's a single integer
        try:
            count = int(values)
            setattr(namespace, self.dest, count)
        except ValueError:
            # If value is not a single integer, check if it's a valid verbosity string
            if not bool(re.search('[^v]', values)):
                # We add a single v because .count starts at zero and returns v - 1.
                count = (values + 'v').count('v')
                setattr(namespace, self.dest, count)
            else:
                logging.error('Invalid verbosity level')
                exit(1)

        # Check if verbosity level is a non-negative integer
        if getattr(namespace, self.dest) < 0:
            logging.error('Verbosity level must be a positive integer')
            exit(1)


def get_logs_dir(_args):
    if getattr(_args, "log-dir"):
        os.path.expanduser(getattr(_args, "log-dir"))
        _logs_dir = os.path.abspath(getattr(_args, "log-dir"))
    else:
        logs_base = os.path.join(os.path.expanduser("~"), ".kohya_ss")
        _logs_dir = os.path.join(logs_base, "logs")

    os.makedirs(_logs_dir, exist_ok=True)
    return _logs_dir


class CustomFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(fmt='%(levelname)s: %(message)s')

    def format(self, record):
        if record.levelno == logging.CRITICAL:
            return f"{record.getMessage()}"
        else:
            return f"{record.levelname}: {record.getMessage()}"

    @staticmethod
    def generate_log_filename(_logs_dir):
        now = datetime.now()
        current_time_str = now.strftime("%Y-%m-%dT%H%M%S")  # ISO 8601 format without timezone information

        counter = 0
        while True:
            counter_suffix = f"{counter}" if counter > 0 else ""
            log_filename = f"install_log_{current_time_str}{counter_suffix}.log"
            log_filepath = os.path.join(_logs_dir, log_filename)

            if not os.path.exists(log_filepath):
                break
            counter += 1

        return log_filepath


if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '-l', '--log-dir',
        default=None,
        type=str,
        help='Override the default log directory.',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )
    parser.add_argument(
        '-v', '--verbosity',
        default=0,
        type=int,
        help='Increase verbosity levels. Use multiple times (e.g., -vvv) or specify number (e.g., -v 4).',
        action=CountOccurrencesAction
    )

    args = parser.parse_args()

    # Initialize log_level with a default value
    log_level = logging.ERROR

    # Set logging level based on the verbosity count
    # print(f"Verbosity: {args.verbosity}")
    if args.verbosity == 0:
        log_level = logging.ERROR
    elif args.verbosity == 1:
        log_level = logging.WARNING
    elif args.verbosity == 2:
        log_level = logging.INFO
    elif args.verbosity >= 3:
        log_level = logging.DEBUG

    # Configure logging
    # noinspection SpellCheckingInspection
    setattr(args, "log-dir", os.path.abspath(get_logs_dir(args)))
    log_file = CustomFormatter.generate_log_filename(getattr(args, "log-dir"))
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())

    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_file, mode='w')])
    logging.getLogger().setLevel(log_level)

    # Replace 'root' with an empty string in the logger name
    for handler in logging.getLogger().handlers:
        handler.setFormatter(CustomFormatter())

    logging.debug("")
    if args.verbosity >= 3:
        # Get system information
        system = platform.system()
        release = platform.release()
        version = platform.version()
        machine = platform.machine()
        processor = platform.processor()

        # Get Python information
        python_version = platform.python_version()
        python_implementation = platform.python_implementation()
        python_compiler = platform.python_compiler()

        logging.debug(f"System Information:\nSystem: {system}\nRelease: {release}\nVersion: {version}\n"
                      f"Machine: {machine}\nProcessor: {processor}")
        logging.debug(f"Python Information:\nVersion: {python_version}\nImplementation: "
                      f"{python_implementation}\nCompiler: {python_compiler}")

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
    )
