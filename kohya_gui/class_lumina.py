import gradio as gr
from .common_gui import (
    get_any_file_path,
    document_symbol,
)


class luminaTraining:
    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        training_type: str = "",
        config: dict = {},
        lumina_checkbox: gr.Checkbox = False,
    ) -> None:
        self.headless = headless
        self.finetuning = finetuning
        self.training_type = training_type
        self.config = config
        self.lumina_checkbox = lumina_checkbox

        with gr.Accordion(
            "Lumina Image 2.0",
            open=True,
            visible=False,
            elem_classes=["lumina_background"],
        ) as lumina_accordion:
            with gr.Group():
                with gr.Row():
                    self.gemma2 = gr.Textbox(
                        label="Gemma2 Path",
                        placeholder="Path to Gemma2 text encoder model",
                        value=self.config.get("lumina.gemma2", ""),
                        interactive=True,
                    )
                    self.gemma2_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.gemma2_button.click(
                        get_any_file_path,
                        outputs=self.gemma2,
                        show_progress=False,
                    )

                    self.ae = gr.Textbox(
                        label="AE Path",
                        placeholder="Path to Lumina AutoEncoder (same family as FLUX AE)",
                        value=self.config.get("lumina.ae", ""),
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

                with gr.Row():
                    self.discrete_flow_shift = gr.Number(
                        label="Discrete Flow Shift",
                        value=self.config.get("lumina.discrete_flow_shift", 6.0),
                        info="Discrete flow shift for the Euler Discrete Scheduler, default is 6.0",
                        minimum=-1024,
                        maximum=1024,
                        step=0.01,
                        interactive=True,
                    )
                    self.model_prediction_type = gr.Dropdown(
                        label="Model Prediction Type",
                        choices=["raw", "additive", "sigma_scaled"],
                        value=self.config.get("lumina.model_prediction_type", "raw"),
                        interactive=True,
                    )
                    self.timestep_sampling = gr.Dropdown(
                        label="Timestep Sampling",
                        choices=[
                            "sigma",
                            "uniform",
                            "sigmoid",
                            "shift",
                            "nextdit_shift",
                            "flux_shift",
                        ],
                        value=self.config.get("lumina.timestep_sampling", "shift"),
                        interactive=True,
                    )
                    self.sigmoid_scale = gr.Number(
                        label="Sigmoid Scale",
                        value=self.config.get("lumina.sigmoid_scale", 1.0),
                        info='Scale factor for sigmoid timestep sampling (only used when timestep sampling is "sigmoid")',
                        minimum=0.0,
                        maximum=1024,
                        step=0.01,
                        interactive=True,
                    )

                with gr.Row():
                    self.gemma2_max_token_length = gr.Number(
                        label="Gemma2 Max Token Length",
                        value=self.config.get("lumina.gemma2_max_token_length", 256),
                        info="Maximum token length for Gemma2. Backend default is 256 when omitted.",
                        minimum=1,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )
                    self.system_prompt = gr.Textbox(
                        label="System Prompt",
                        placeholder=(
                            "You are an assistant designed to generate "
                            "high-quality images based on user prompts."
                        ),
                        value=self.config.get("lumina.system_prompt", ""),
                        interactive=True,
                    )

                with gr.Row():
                    self.use_flash_attn = gr.Checkbox(
                        label="Use Flash Attention",
                        value=self.config.get("lumina.use_flash_attn", False),
                        info="Requires flash-attn package",
                        interactive=True,
                    )
                    self.use_sage_attn = gr.Checkbox(
                        label="Use Sage Attention",
                        value=self.config.get("lumina.use_sage_attn", False),
                        interactive=True,
                    )
                    self.cache_text_encoder_outputs = gr.Checkbox(
                        label="Cache Text Encoder Outputs",
                        value=self.config.get(
                            "lumina.cache_text_encoder_outputs", False
                        ),
                        info="Cache Gemma2 text encoder outputs to reduce memory usage",
                        interactive=True,
                    )
                    self.cache_text_encoder_outputs_to_disk = gr.Checkbox(
                        label="Cache Text Encoder Outputs to Disk",
                        value=self.config.get(
                            "lumina.cache_text_encoder_outputs_to_disk", False
                        ),
                        info="Cache text encoder outputs to disk",
                        interactive=True,
                    )

                self.lumina_checkbox.change(
                    lambda lumina_checkbox: gr.Accordion(visible=lumina_checkbox),
                    inputs=[self.lumina_checkbox],
                    outputs=[lumina_accordion],
                )
