import gradio as gr
from .common_gui import (
    get_any_file_path,
    get_folder_path,
    document_symbol,
)


class animaTraining:
    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        training_type: str = "",
        config: dict = {},
        anima_checkbox: gr.Checkbox = False,
    ) -> None:
        self.headless = headless
        self.finetuning = finetuning
        self.training_type = training_type
        self.config = config
        self.anima_checkbox = anima_checkbox

        with gr.Accordion(
            "Anima",
            open=True,
            visible=False,
            elem_classes=["anima_background"],
        ) as anima_accordion:
            with gr.Group():
                with gr.Row():
                    self.qwen3 = gr.Textbox(
                        label="Qwen3 Path",
                        placeholder="Path to Qwen3-0.6B text encoder model (file or directory)",
                        value=self.config.get("anima.qwen3", ""),
                        interactive=True,
                    )
                    self.qwen3_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.qwen3_button.click(
                        get_any_file_path,
                        outputs=self.qwen3,
                        show_progress=False,
                    )

                    self.vae = gr.Textbox(
                        label="VAE Path",
                        placeholder="Path to Qwen-Image VAE model",
                        value=self.config.get("anima.vae", ""),
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
                    self.llm_adapter_path = gr.Textbox(
                        label="LLM Adapter Path",
                        placeholder="(Optional) Path to separate LLM adapter weights",
                        info="If omitted, the adapter is loaded from the DiT file when the key llm_adapter.out_proj.weight exists",
                        value=self.config.get("anima.llm_adapter_path", ""),
                        interactive=True,
                    )
                    self.llm_adapter_path_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.llm_adapter_path_button.click(
                        get_any_file_path,
                        outputs=self.llm_adapter_path,
                        show_progress=False,
                    )

                    self.t5_tokenizer_path = gr.Textbox(
                        label="T5 Tokenizer Path",
                        placeholder="(Optional) Path to T5 tokenizer directory",
                        info="If omitted, uses the bundled tokenizer at configs/t5_old/",
                        value=self.config.get("anima.t5_tokenizer_path", ""),
                        interactive=True,
                    )
                    self.t5_tokenizer_path_button = gr.Button(
                        "\U0001f4c2",
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.t5_tokenizer_path_button.click(
                        get_folder_path,
                        outputs=self.t5_tokenizer_path,
                        show_progress=False,
                    )

                with gr.Row():
                    self.discrete_flow_shift = gr.Number(
                        label="Discrete Flow Shift",
                        value=self.config.get("anima.discrete_flow_shift", 1.0),
                        info="Timestep distribution shift for rectified flow training, used when Timestep Sampling is 'shift'. Default is 1.0",
                        minimum=-1024,
                        maximum=1024,
                        step=0.01,
                        interactive=True,
                    )
                    self.timestep_sampling = gr.Dropdown(
                        label="Timestep Sampling",
                        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift"],
                        value=self.config.get("anima.timestep_sampling", "sigmoid"),
                        interactive=True,
                    )
                    self.sigmoid_scale = gr.Number(
                        label="Sigmoid Scale",
                        value=self.config.get("anima.sigmoid_scale", 1.0),
                        info='Scale factor for sigmoid timestep sampling (used when Timestep Sampling is "sigmoid", "shift" or "flux_shift")',
                        minimum=0.0,
                        maximum=1024,
                        step=0.01,
                        interactive=True,
                    )

                with gr.Row():
                    self.qwen3_max_token_length = gr.Number(
                        label="Qwen3 Max Token Length",
                        value=self.config.get("anima.qwen3_max_token_length", 512),
                        info="Maximum token length for the Qwen3 tokenizer",
                        minimum=0,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )
                    self.t5_max_token_length = gr.Number(
                        label="T5 Max Token Length",
                        value=self.config.get("anima.t5_max_token_length", 512),
                        info="Maximum token length for the T5 tokenizer",
                        minimum=0,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )
                    self.attn_mode = gr.Dropdown(
                        label="Attention Mode",
                        choices=["torch", "xformers", "flash", "sageattn"],
                        value=self.config.get("anima.attn_mode", "torch"),
                        info="xformers requires Split Attention. sageattn does not support training (inference only)",
                        interactive=True,
                    )

                with gr.Row():
                    self.split_attn = gr.Checkbox(
                        label="Split Attention",
                        value=self.config.get("anima.split_attn", False),
                        info="Split attention computation to reduce memory usage. Required when using Attention Mode xformers.",
                        interactive=True,
                    )
                    self.vae_chunk_size = gr.Number(
                        label="VAE Chunk Size",
                        value=self.config.get("anima.vae_chunk_size", 0),
                        info="Spatial chunk size for VAE encoding/decoding to reduce memory usage. Must be an even number. 0 means no chunking (default).",
                        minimum=0,
                        maximum=1024,
                        step=2,
                        interactive=True,
                    )
                    self.vae_disable_cache = gr.Checkbox(
                        label="VAE Disable Cache",
                        value=self.config.get("anima.vae_disable_cache", False),
                        info="Disable internal VAE caching mechanism to reduce memory usage (faster, but differs from official behavior)",
                        interactive=True,
                    )

                with gr.Row():
                    self.cache_text_encoder_outputs = gr.Checkbox(
                        label="Cache Text Encoder Outputs",
                        value=self.config.get(
                            "anima.cache_text_encoder_outputs", False
                        ),
                        info="Cache Qwen3 text encoder outputs to reduce VRAM usage. Recommended when not training Text Encoder LoRA.",
                        interactive=True,
                    )
                    self.cache_text_encoder_outputs_to_disk = gr.Checkbox(
                        label="Cache Text Encoder Outputs to Disk",
                        value=self.config.get(
                            "anima.cache_text_encoder_outputs_to_disk", False
                        ),
                        info="Cache text encoder outputs to disk. Automatically enables Cache Text Encoder Outputs.",
                        interactive=True,
                    )
                    self.unsloth_offload_checkpointing = gr.Checkbox(
                        label="Unsloth Offload Checkpointing",
                        value=self.config.get(
                            "anima.unsloth_offload_checkpointing", False
                        ),
                        info="Offload activations to CPU RAM using async non-blocking transfers (faster than CPU Offload Checkpointing). Cannot be combined with CPU Offload Checkpointing or Blocks to swap.",
                        interactive=True,
                    )

                self.anima_checkbox.change(
                    lambda anima_checkbox: gr.Accordion(visible=anima_checkbox),
                    inputs=[self.anima_checkbox],
                    outputs=[anima_accordion],
                )
