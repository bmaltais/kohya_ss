import gradio as gr
from .class_gui_config import KohyaSSGUIConfig

class SDXLParameters:
    def __init__(
        self,
        sdxl_checkbox: gr.Checkbox,
        show_sdxl_cache_text_encoder_outputs: bool = True,
        config: KohyaSSGUIConfig = {},
    ):
        self.sdxl_checkbox = sdxl_checkbox
        self.show_sdxl_cache_text_encoder_outputs = show_sdxl_cache_text_encoder_outputs
        self.config = config
        
        self.initialize_accordion()

    def initialize_accordion(self):
        with gr.Accordion(
            visible=False, open=True, label="SDXL Specific Parameters"
        ) as self.sdxl_row:
            with gr.Row():
                self.sdxl_cache_text_encoder_outputs = gr.Checkbox(
                    label="Cache text encoder outputs",
                    info="Cache the outputs of the text encoders. This option is useful to reduce the GPU memory usage. This option cannot be used with options for shuffling or dropping the captions.",
                    value=self.config.get("sdxl.sdxl_cache_text_encoder_outputs", False),
                    visible=self.show_sdxl_cache_text_encoder_outputs,
                )
                self.sdxl_no_half_vae = gr.Checkbox(
                    label="No half VAE",
                    info="Disable the half-precision (mixed-precision) VAE. VAE for SDXL seems to produce NaNs in some cases. This option is useful to avoid the NaNs.",
                    value=self.config.get("sdxl.sdxl_no_half_vae", False),
                )

        self.sdxl_checkbox.change(
            lambda sdxl_checkbox: gr.Accordion(visible=sdxl_checkbox),
            inputs=[self.sdxl_checkbox],
            outputs=[self.sdxl_row],
        )
