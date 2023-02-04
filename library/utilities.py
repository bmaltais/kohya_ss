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


def utilities_tab(
    train_data_dir_input=gr.Textbox(),
    reg_data_dir_input=gr.Textbox(),
    output_dir_input=gr.Textbox(),
    logging_dir_input=gr.Textbox(),
    enable_copy_info_button=bool(False),
    enable_dreambooth_tab=True,
):
    with gr.Tab('Captioning'):
        gradio_basic_caption_gui_tab()
        gradio_blip_caption_gui_tab()
        gradio_git_caption_gui_tab()
        gradio_wd14_caption_gui_tab()
    gradio_convert_model_tab()

    return (
        train_data_dir_input,
        reg_data_dir_input,
        output_dir_input,
        logging_dir_input,
    )


def UI(username, password):
    css = ''

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            print('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Blocks(css=css)

    with interface:
        utilities_tab()

    # Show the interface
    if not username == '':
        interface.launch(auth=(username, password))
    else:
        interface.launch()


if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )

    args = parser.parse_args()

    UI(username=args.username, password=args.password)
