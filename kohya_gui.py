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

    args = parser.parse_args()

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
    )
