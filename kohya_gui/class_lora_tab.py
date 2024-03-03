import gradio as gr
from .merge_lora_gui import GradioMergeLoRaTab
from .svd_merge_lora_gui import gradio_svd_merge_lora_tab
from .verify_lora_gui import gradio_verify_lora_tab
from .resize_lora_gui import gradio_resize_lora_tab
from .extract_lora_gui import gradio_extract_lora_tab
from .convert_lcm_gui import gradio_convert_lcm_tab
from .extract_lycoris_locon_gui import gradio_extract_lycoris_locon_tab
from .extract_lora_from_dylora_gui import gradio_extract_dylora_tab
from .merge_lycoris_gui import gradio_merge_lycoris_tab

# Deprecated code
from .dataset_balancing_gui import gradio_dataset_balancing_tab
from .dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)


class LoRATools:
    def __init__(self, train_data_dir=None, reg_data_dir=None, output_dir=None, logging_dir=None, headless: bool = False):
        self.headless = headless

        gr.Markdown(
            'This section provide LoRA tools to help setup your dataset...'
        )
        gradio_extract_dylora_tab(headless=headless)
        gradio_convert_lcm_tab(headless=headless)
        gradio_extract_lora_tab(headless=headless)
        gradio_extract_lycoris_locon_tab(headless=headless)
        gradio_merge_lora_tab = GradioMergeLoRaTab()
        gradio_merge_lycoris_tab(headless=headless)
        gradio_svd_merge_lora_tab(headless=headless)
        gradio_resize_lora_tab(headless=headless)
        gradio_verify_lora_tab(headless=headless)
        if train_data_dir is not None:
            with gr.Tab('Dataset Preparation'):
                gradio_dreambooth_folder_creation_tab(
                    train_data_dir_input=train_data_dir,
                    reg_data_dir_input=reg_data_dir,
                    output_dir_input=output_dir,
                    logging_dir_input=logging_dir,
                    headless=headless,
                )
                gradio_dataset_balancing_tab(headless=headless)
