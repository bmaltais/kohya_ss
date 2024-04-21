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


class LoRATools:
    def __init__(
        self,
        headless: bool = False,
        use_shell_flag: bool = False,
    ):
        gr.Markdown("This section provide various LoRA tools...")
        gradio_extract_dylora_tab(headless=headless, use_shell=use_shell_flag)
        gradio_convert_lcm_tab(headless=headless, use_shell=use_shell_flag)
        gradio_extract_lora_tab(headless=headless, use_shell=use_shell_flag)
        gradio_extract_lycoris_locon_tab(headless=headless, use_shell=use_shell_flag)
        gradio_merge_lora_tab = GradioMergeLoRaTab(use_shell=use_shell_flag)
        gradio_merge_lycoris_tab(headless=headless, use_shell=use_shell_flag)
        gradio_svd_merge_lora_tab(headless=headless, use_shell=use_shell_flag)
        gradio_resize_lora_tab(headless=headless, use_shell=use_shell_flag)
        gradio_verify_lora_tab(headless=headless)
