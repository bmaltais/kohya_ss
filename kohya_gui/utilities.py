import gradio as gr

from .basic_caption_gui import gradio_basic_caption_gui_tab
from .convert_model_gui import gradio_convert_model_tab
from .blip_caption_gui import gradio_blip_caption_gui_tab
from .blip2_caption_gui import gradio_blip2_caption_gui_tab
from .git_caption_gui import gradio_git_caption_gui_tab
from .wd14_caption_gui import gradio_wd14_caption_gui_tab
from .manual_caption_gui import gradio_manual_caption_gui_tab
from .group_images_gui import gradio_group_images_gui_tab
from .class_gui_config import KohyaSSGUIConfig


def utilities_tab(
    train_data_dir_input=gr.Dropdown(),
    reg_data_dir_input=gr.Dropdown(),
    output_dir_input=gr.Dropdown(),
    logging_dir_input=gr.Dropdown(),
    enable_copy_info_button=bool(False),
    enable_dreambooth_tab=True,
    headless=False,
    config: KohyaSSGUIConfig = {},
    use_shell_flag: bool = False,
):
    with gr.Tab("Captioning"):
        gradio_basic_caption_gui_tab(headless=headless)
        gradio_blip_caption_gui_tab(headless=headless, use_shell=use_shell_flag)
        gradio_blip2_caption_gui_tab(headless=headless)
        gradio_git_caption_gui_tab(headless=headless, use_shell=use_shell_flag)
        gradio_wd14_caption_gui_tab(headless=headless, config=config, use_shell=use_shell_flag)
        gradio_manual_caption_gui_tab(headless=headless)
    gradio_convert_model_tab(headless=headless, use_shell=use_shell_flag)
    gradio_group_images_gui_tab(headless=headless, use_shell=use_shell_flag)

    return (
        train_data_dir_input,
        reg_data_dir_input,
        output_dir_input,
        logging_dir_input,
    )
