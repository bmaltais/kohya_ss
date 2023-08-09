# v1: initial release
# v2: add open and save folder icons
# v3: Add new Utilities tab for Dreambooth folder preparation
# v3.1: Adding captionning of images to utilities

import gradio as gr
import os
import argparse
from library.basic_caption_gui import gradio_basic_caption_gui_tab
from library.convert_model_gui import gradio_convert_model_tab
from library.blip_caption_gui import gradio_blip_caption_gui_tab
from library.git_caption_gui import gradio_git_caption_gui_tab
from library.wd14_caption_gui import gradio_wd14_caption_gui_tab
from library.manual_caption_gui import gradio_manual_caption_gui_tab
from library.group_images_gui import gradio_group_images_gui_tab


def utilities_tab(
    train_data_dir_input=gr.Textbox(),
    reg_data_dir_input=gr.Textbox(),
    output_dir_input=gr.Textbox(),
    logging_dir_input=gr.Textbox(),
    enable_copy_info_button=bool(False),
    enable_dreambooth_tab=True,
    headless=False
):
    with gr.Tab('Captioning'):
        gradio_basic_caption_gui_tab(headless=headless)
        gradio_blip_caption_gui_tab(headless=headless)
        gradio_git_caption_gui_tab(headless=headless)
        gradio_wd14_caption_gui_tab(headless=headless)
        gradio_manual_caption_gui_tab(headless=headless)
    gradio_convert_model_tab(headless=headless)
    gradio_group_images_gui_tab(headless=headless)

    return (
        train_data_dir_input,
        reg_data_dir_input,
        output_dir_input,
        logging_dir_input,
    )


def UI(**kwargs):
    css = ''

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            print('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Blocks(css=css)

    with interface:
        utilities_tab()

    # Show the interface
    launch_kwargs = {}
    if not kwargs.get('username', None) == '':
        launch_kwargs['auth'] = (
            kwargs.get('username', None),
            kwargs.get('password', None),
        )
    if kwargs.get('server_port', 0) > 0:
        launch_kwargs['server_port'] = kwargs.get('server_port', 0)
    if kwargs.get('inbrowser', False):
        launch_kwargs['inbrowser'] = kwargs.get('inbrowser', False)
    print(launch_kwargs)
    interface.launch(**launch_kwargs)


if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)
    parser = argparse.ArgumentParser()
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

    args = parser.parse_args()

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
    )
