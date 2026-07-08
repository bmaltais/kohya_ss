import gradio as gr
from .common_gui import (
    get_any_file_path,
    document_symbol,
)


class hunyuanImageTraining:
    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        training_type: str = "",
        config: dict = {},
        hunyuan_image_checkbox: gr.Checkbox = False,
    ) -> None:
        self.headless = headless
        self.finetuning = finetuning
        self.training_type = training_type
        self.config = config
        self.hunyuan_image_checkbox = hunyuan_image_checkbox

        with gr.Accordion(
            "HunyuanImage-2.1",
            open=True,
            visible=False,
            elem_classes=["hunyuan_image_background"],
        ) as hunyuan_image_accordion:
            with gr.Group():
                with gr.Row():
                    self.text_encoder = gr.Textbox(
                        label="Qwen2.5-VL Path",
                        placeholder="Path to Qwen2.5-VL text encoder model",
                        value=self.config.get("hunyuan_image.text_encoder", ""),
                        interactive=True,
                    )
                    self.text_encoder_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.text_encoder_button.click(
                        get_any_file_path,
                        outputs=self.text_encoder,
                        show_progress=False,
                    )

                    self.byt5 = gr.Textbox(
                        label="byT5 Path",
                        placeholder="Path to byT5 text encoder model",
                        value=self.config.get("hunyuan_image.byt5", ""),
                        interactive=True,
                    )
                    self.byt5_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.byt5_button.click(
                        get_any_file_path,
                        outputs=self.byt5,
                        show_progress=False,
                    )

                    self.vae = gr.Textbox(
                        label="VAE Path",
                        placeholder="Path to HunyuanImage-2.1 VAE model",
                        value=self.config.get("hunyuan_image.vae", ""),
                        interactive=True,
                    )
                    self.vae_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.vae_button.click(
                        get_any_file_path,
                        outputs=self.vae,
                        show_progress=False,
                    )

                with gr.Row():
                    self.discrete_flow_shift = gr.Number(
                        label="Discrete Flow Shift",
                        value=self.config.get("hunyuan_image.discrete_flow_shift", 5.0),
                        info="Discrete flow shift for the Euler Discrete Scheduler, default is 5.0",
                        minimum=-1024,
                        maximum=1024,
                        step=0.01,
                        interactive=True,
                    )
                    self.model_prediction_type = gr.Dropdown(
                        label="Model Prediction Type",
                        choices=["raw", "additive", "sigma_scaled"],
                        value=self.config.get(
                            "hunyuan_image.model_prediction_type", "raw"
                        ),
                        interactive=True,
                    )
                    self.timestep_sampling = gr.Dropdown(
                        label="Timestep Sampling",
                        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift"],
                        value=self.config.get(
                            "hunyuan_image.timestep_sampling", "sigma"
                        ),
                        interactive=True,
                    )
                    self.sigmoid_scale = gr.Number(
                        label="Sigmoid Scale",
                        value=self.config.get("hunyuan_image.sigmoid_scale", 1.0),
                        info='Scale factor for sigmoid timestep sampling (only used when timestep sampling is "sigmoid")',
                        minimum=0.0,
                        maximum=1024,
                        step=0.01,
                        interactive=True,
                    )

                with gr.Row():
                    self.attn_mode = gr.Dropdown(
                        label="Attention Mode",
                        choices=["torch", "xformers", "flash", "sageattn"],
                        value=self.config.get("hunyuan_image.attn_mode", "torch"),
                        interactive=True,
                    )
                    self.split_attn = gr.Checkbox(
                        label="Split Attention",
                        value=self.config.get("hunyuan_image.split_attn", False),
                        info="Split attention computation to reduce memory usage. Required when using xformers with batch size > 1.",
                        interactive=True,
                    )
                    self.fp8_scaled = gr.Checkbox(
                        label="FP8 Scaled",
                        value=self.config.get("hunyuan_image.fp8_scaled", False),
                        info="Use scaled fp8 for the DiT model to reduce VRAM usage",
                        interactive=True,
                    )
                    self.fp8_vl = gr.Checkbox(
                        label="FP8 VLM",
                        value=self.config.get("hunyuan_image.fp8_vl", False),
                        info="Use fp8 for the Qwen2.5-VL text encoder",
                        interactive=True,
                    )

                with gr.Row():
                    self.text_encoder_cpu = gr.Checkbox(
                        label="Text Encoder on CPU",
                        value=self.config.get("hunyuan_image.text_encoder_cpu", False),
                        info="Run the Qwen2.5-VL and byT5 text encoders on CPU to reduce VRAM usage",
                        interactive=True,
                    )
                    self.vae_chunk_size = gr.Number(
                        label="VAE Chunk Size",
                        value=self.config.get("hunyuan_image.vae_chunk_size", 0),
                        info="Chunk size for VAE decoding to reduce memory usage. 0 means no chunking, 16 is recommended if enabled.",
                        minimum=0,
                        maximum=1024,
                        step=1,
                        interactive=True,
                    )
                    self.hunyuan_image_cache_text_encoder_outputs = gr.Checkbox(
                        label="Cache Text Encoder Outputs",
                        value=self.config.get(
                            "hunyuan_image.cache_text_encoder_outputs", False
                        ),
                        info="Cache Qwen2.5-VL and byT5 text encoder outputs to reduce memory usage",
                        interactive=True,
                    )
                    self.hunyuan_image_cache_text_encoder_outputs_to_disk = gr.Checkbox(
                        label="Cache Text Encoder Outputs to Disk",
                        value=self.config.get(
                            "hunyuan_image.cache_text_encoder_outputs_to_disk", False
                        ),
                        info="Cache text encoder outputs to disk",
                        interactive=True,
                    )

                self.hunyuan_image_checkbox.change(
                    lambda hunyuan_image_checkbox: gr.Accordion(
                        visible=hunyuan_image_checkbox
                    ),
                    inputs=[self.hunyuan_image_checkbox],
                    outputs=[hunyuan_image_accordion],
                )
