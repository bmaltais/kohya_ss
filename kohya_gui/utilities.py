# v1: initial release
# v2: add open and save folder icons
# v3: Add new Utilities tab for Dreambooth folder preparation
# v3.1: Adding captionning of images to utilities

import gradio as gr
import os

from .basic_caption_gui import gradio_basic_caption_gui_tab
from .convert_model_gui import gradio_convert_model_tab
from .blip_caption_gui import gradio_blip_caption_gui_tab
from .blip2_caption_gui import gradio_blip2_caption_gui_tab
from .git_caption_gui import gradio_git_caption_gui_tab
from .wd14_caption_gui import gradio_wd14_caption_gui_tab
from .manual_caption_gui import gradio_manual_caption_gui_tab
from .group_images_gui import gradio_group_images_gui_tab


def utilities_tab(
    train_data_dir_input=gr.Dropdown(),
    reg_data_dir_input=gr.Dropdown(),
    output_dir_input=gr.Dropdown(),
    logging_dir_input=gr.Dropdown(),
    enable_copy_info_button=bool(False),
    enable_dreambooth_tab=True,
    headless=False
):
    with gr.Tab('Captioning'):
        gradio_basic_caption_gui_tab(headless=headless)
        gradio_blip_caption_gui_tab(headless=headless)
        gradio_blip2_caption_gui_tab(headless=headless)
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
