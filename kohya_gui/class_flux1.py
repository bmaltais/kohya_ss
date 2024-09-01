import gradio as gr
from typing import Tuple
from .common_gui import (
    get_any_file_path,
    document_symbol,
)


class flux1Training:
    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        training_type: str = "",
        config: dict = {},
        flux1_checkbox: gr.Checkbox = False,
    ) -> None:
        self.headless = headless
        self.finetuning = finetuning
        self.training_type = training_type
        self.config = config
        self.flux1_checkbox = flux1_checkbox

        # Define the behavior for changing noise offset type.
        def noise_offset_type_change(
            noise_offset_type: str,
        ) -> Tuple[gr.Group, gr.Group]:
            if noise_offset_type == "Original":
                return (gr.Group(visible=True), gr.Group(visible=False))
            else:
                return (gr.Group(visible=False), gr.Group(visible=True))

        with gr.Accordion(
            "Flux.1", open=True, elem_id="flux1_tab", visible=False
        ) as flux1_accordion:
            with gr.Group():
                with gr.Row():
                    self.ae = gr.Textbox(
                        label="VAE Path",
                        placeholder="Path to VAE model",
                        value=self.config.get("flux1.ae", ""),
                        interactive=True,
                    )
                    self.ae_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.ae_button.click(
                        get_any_file_path,
                        outputs=self.ae,
                        show_progress=False,
                    )
                    
                    self.clip_l = gr.Textbox(
                        label="CLIP-L Path",
                        placeholder="Path to CLIP-L model",
                        value=self.config.get("flux1.clip_l", ""),
                        interactive=True,
                    )
                    self.clip_l_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.clip_l_button.click(
                        get_any_file_path,
                        outputs=self.clip_l,
                        show_progress=False,
                    )

                    self.t5xxl = gr.Textbox(
                        label="T5-XXL Path",
                        placeholder="Path to T5-XXL model",
                        value=self.config.get("flux1.t5xxl", ""),
                        interactive=True,
                    )
                    self.t5xxl_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.t5xxl_button.click(
                        get_any_file_path,
                        outputs=self.t5xxl,
                        show_progress=False,
                    )

                with gr.Row():
                    
                    self.discrete_flow_shift = gr.Number(
                        label="Discrete Flow Shift",
                        value=self.config.get("flux1.discrete_flow_shift", 3.0),
                        info="Discrete flow shift for the Euler Discrete Scheduler, default is 3.0",
                        minimum=-1024,
                        maximum=1024,
                        step=.01,
                        interactive=True,
                    )
                    self.model_prediction_type = gr.Dropdown(
                        label="Model Prediction Type",
                        choices=["raw", "additive", "sigma_scaled"],
                        value=self.config.get("flux1.timestep_sampling", "sigma_scaled"),
                        interactive=True,
                    )
                    self.timestep_sampling = gr.Dropdown(
                        label="Timestep Sampling",
                        choices=["flux_shift ", "sigma", "shift", "sigmoid", "uniform"],
                        value=self.config.get("flux1.timestep_sampling", "sigma"),
                        interactive=True,
                    )
                    self.apply_t5_attn_mask = gr.Checkbox(
                        label="Apply T5 Attention Mask",
                        value=self.config.get("flux1.apply_t5_attn_mask", False),
                        info="Apply attention mask to T5-XXL encode and FLUX double blocks ",
                        interactive=True,
                    )
                with gr.Row(visible=True if not finetuning else False):
                    self.split_mode = gr.Checkbox(
                        label="Split Mode",
                        value=self.config.get("flux1.split_mode", False),
                        info="Split mode for Flux1",
                        interactive=True,
                    )
                    self.train_blocks = gr.Dropdown(
                        label="Train Blocks",
                        choices=["all", "double", "single"],
                        value=self.config.get("flux1.train_blocks", "all"),
                        interactive=True,
                    )
                    self.split_qkv = gr.Checkbox(
                        label="Split QKV",
                        value=self.config.get("flux1.split_qkv", False),
                        info="Split the projection layers of q/k/v/txt in the attention",
                        interactive=True,
                    )
                with gr.Row():
                    self.guidance_scale = gr.Number(
                        label="Guidance Scale",
                        value=self.config.get("flux1.guidance_scale", 3.5),
                        info="Guidance scale for Flux1",
                        minimum=0,
                        maximum=1024,
                        step=.1,
                        interactive=True,
                    )
                    self.t5xxl_max_token_length =  gr.Number(
                        label="T5-XXL Max Token Length",
                        value=self.config.get("flux1.t5xxl_max_token_length", 512),
                        info="Max token length for T5-XXL",
                        minimum=0,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )
                    
                with gr.Row():
                    self.flux1_cache_text_encoder_outputs = gr.Checkbox(
                        label="Cache Text Encoder Outputs",
                        value=self.config.get("flux1.cache_text_encoder_outputs", False),
                        info="Cache text encoder outputs to speed up inference",
                        interactive=True,
                    )
                    self.flux1_cache_text_encoder_outputs_to_disk = gr.Checkbox(
                        label="Cache Text Encoder Outputs to Disk",
                        value=self.config.get(
                            "flux1.cache_text_encoder_outputs_to_disk", False
                        ),
                        info="Cache text encoder outputs to disk to speed up inference",
                        interactive=True,
                    )
                    self.mem_eff_save = gr.Checkbox(
                        label="Memory Efficient Save",
                        value=self.config.get("flux1.mem_eff_save", False),
                        info="[Experimentsl] Enable memory efficient save. We do not recommend using it unless you are familiar with the code.",
                        interactive=True,
                    )
                    
                with gr.Row(visible=True if finetuning else False):
                    self.blockwise_fused_optimizers = gr.Checkbox(
                        label="Blockwise Fused Optimizer",
                        value=self.config.get("flux1.blockwise_fused_optimizers", False),
                        info="Enable blockwise optimizers for fused backward pass and optimizer step. Any optimizer can be used.",
                        interactive=True,
                    )
                    self.cpu_offload_checkpointing = gr.Checkbox(
                        label="CPU Offload Checkpointing",
                        value=self.config.get("flux1.cpu_offload_checkpointing", False),
                        info="[Experimental] Enable offloading of tensors to CPU during checkpointing",
                        interactive=True,
                    )
                    self.single_blocks_to_swap = gr.Slider(
                        label="Single Blocks to swap",
                        value=self.config.get("flux1.single_blocks_to_swap", 0),
                        info="[Experimental] Sets the number of 'single_blocks' (~320MB) to swap during the forward and backward passes.",
                        minimum=0,
                        maximum=19,
                        step=1,
                        interactive=True,
                    )
                    self.double_blocks_to_swap = gr.Slider(
                        label="Double Blocks to swap",
                        value=self.config.get("flux1.double_blocks_to_swap", 0),
                        info="[Experimental] Sets the number of 'double_blocks' (~640MB) to swap during the forward and backward passes.",
                        minimum=0,
                        maximum=38,
                        step=1,
                        interactive=True,
                    )
                    self.flux_fused_backward_pass = gr.Checkbox(
                        label="Fused Backward Pass",
                        value=self.config.get("flux1.fused_backward_pass", False),
                        info="Enables the fusing of the optimizer step into the backward pass for each parameter.  Only Adafactor optimizer is supported.",
                        interactive=True,
                    )

                self.flux1_checkbox.change(
                    lambda flux1_checkbox: gr.Accordion(visible=flux1_checkbox),
                    inputs=[self.flux1_checkbox],
                    outputs=[flux1_accordion],
                )
