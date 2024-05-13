import gradio as gr
from .class_gui_config import KohyaSSGUIConfig

class SDXLParameters:
    def __init__(
        self,
        sdxl_checkbox: gr.Checkbox,
        show_sdxl_cache_text_encoder_outputs: bool = True,
        config: KohyaSSGUIConfig = {},
        trainer: str = "",
    ):
        self.sdxl_checkbox = sdxl_checkbox
        self.show_sdxl_cache_text_encoder_outputs = show_sdxl_cache_text_encoder_outputs
        self.config = config
        self.trainer = trainer
        
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
                self.fused_backward_pass = gr.Checkbox(
                    label="Fused backward pass",
                    info="Enable fused backward pass. This option is useful to reduce the GPU memory usage. Can't be used if Fused optimizer groups is > 0. Only AdaFactor is supported",
                    value=self.config.get("sdxl.fused_backward_pass", False),
                    visible=self.trainer == "finetune" or self.trainer == "dreambooth",
                )
                self.fused_optimizer_groups = gr.Number(
                    label="Fused optimizer groups",
                    info="Number of optimizer groups to fuse. This option is useful to reduce the GPU memory usage. Can't be used if Fused backward pass is enabled. Since the effect is limited to a certain number, it is recommended to specify 4-10.",
                    value=self.config.get("sdxl.fused_optimizer_groups", 0),
                    minimum=0,
                    step=1,
                    visible=self.trainer == "finetune" or self.trainer == "dreambooth",
                )
                self.disable_mmap_load_safetensors = gr.Checkbox(
                    label="Disable mmap load safe tensors",
                    info="Disable memory mapping when loading the model's .safetensors in SDXL.",
                    value=self.config.get("sdxl.disable_mmap_load_safetensors", False),
                )

                self.fused_backward_pass.change(
                    lambda fused_backward_pass: gr.Number(
                        interactive=not fused_backward_pass
                    ),
                    inputs=[self.fused_backward_pass],
                    outputs=[self.fused_optimizer_groups],
                )
                self.fused_optimizer_groups.change(
                    lambda fused_optimizer_groups: gr.Checkbox(
                        interactive=fused_optimizer_groups == 0
                    ),
                    inputs=[self.fused_optimizer_groups],
                    outputs=[self.fused_backward_pass],
                )


        self.sdxl_checkbox.change(
            lambda sdxl_checkbox: gr.Accordion(visible=sdxl_checkbox),
            inputs=[self.sdxl_checkbox],
            outputs=[self.sdxl_row],
        )
