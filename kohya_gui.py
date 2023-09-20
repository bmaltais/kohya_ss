import gradio as gr
import os
import argparse
from dreambooth_gui import dreambooth_tab
from finetune_gui import finetune_tab
from textual_inversion_gui import ti_tab
from library.utilities import utilities_tab
from lora_gui import lora_tab
from library.class_lora_tab import LoRATools

import os
from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()


def UI(**kwargs):
    css = ''

    headless = kwargs.get('headless', False)
    log.info(f'headless: {headless}')

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            log.info('Load CSS...')
            css += file.read() + '\n'

    if os.path.exists('./.release'):
        with open(os.path.join('./.release'), 'r', encoding='utf8') as file:
            release = file.read()

    if os.path.exists('./README.md'):
        with open(os.path.join('./README.md'), 'r', encoding='utf8') as file:
            README = file.read()

    interface = gr.Blocks(
        css=css, title=f'Kohya_ss GUI {release}', theme=gr.themes.Default()
    )

    with interface:
        with gr.Tab('Dreambooth'):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = dreambooth_tab(headless=headless)
        with gr.Tab('LoRA'):
            lora_tab(headless=headless)
        with gr.Tab('Textual Inversion'):
            ti_tab(headless=headless)
        with gr.Tab('Finetuning'):
            finetune_tab(headless=headless)
        with gr.Tab('Utilities'):
            utilities_tab(
                train_data_dir_input=train_data_dir_input,
                reg_data_dir_input=reg_data_dir_input,
                output_dir_input=output_dir_input,
                logging_dir_input=logging_dir_input,
                enable_copy_info_button=True,
                headless=headless,
            )
            with gr.Tab('LoRA'):
                _ = LoRATools(headless=headless)
        with gr.Tab('About'):
            gr.Markdown(f'kohya_ss GUI release {release}')
            with gr.Tab('README'):
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
        '--headless', action='store_true', help='Is the server headless'
    )

    args = parser.parse_args()

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
        headless=args.headless,
    )
