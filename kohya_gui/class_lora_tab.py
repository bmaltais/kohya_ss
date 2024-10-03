import gradio as gr
from .merge_lora_gui import GradioMergeLoRaTab
from .svd_merge_lora_gui import gradio_svd_merge_lora_tab
from .verify_lora_gui import gradio_verify_lora_tab
from .resize_lora_gui import gradio_resize_lora_tab
from .extract_lora_gui import gradio_extract_lora_tab
from .flux_extract_lora_gui import gradio_flux_extract_lora_tab
from .convert_lcm_gui import gradio_convert_lcm_tab
from .extract_lycoris_locon_gui import gradio_extract_lycoris_locon_tab
from .extract_lora_from_dylora_gui import gradio_extract_dylora_tab
from .merge_lycoris_gui import gradio_merge_lycoris_tab
from .flux_merge_lora_gui import GradioFluxMergeLoRaTab


class LoRATools:
    def __init__(
        self,
        headless: bool = False,
    ):
        gr.Markdown("This section provide various LoRA tools...")
        gradio_extract_dylora_tab(headless=headless)
        gradio_convert_lcm_tab(headless=headless)
        gradio_extract_lora_tab(headless=headless)
        gradio_flux_extract_lora_tab(headless=headless)
        gradio_extract_lycoris_locon_tab(headless=headless)
        gradio_merge_lora_tab = GradioMergeLoRaTab()
        gradio_merge_lycoris_tab(headless=headless)
        gradio_svd_merge_lora_tab(headless=headless)
        gradio_resize_lora_tab(headless=headless)
        gradio_verify_lora_tab(headless=headless)
        GradioFluxMergeLoRaTab(headless=headless)
