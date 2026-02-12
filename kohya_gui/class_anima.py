import gradio as gr
from .common_gui import (
    get_any_file_path,
    document_symbol,
)


class animaTraining:
    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        config: dict = {},
        anima_checkbox: gr.Checkbox = False,
    ) -> None:
        self.headless = headless
        self.finetuning = finetuning
        self.config = config
        self.anima_checkbox = anima_checkbox

        with gr.Accordion(
            "Anima", open=True, visible=False, elem_classes=["anima_background"]
        ) as anima_accordion:
            with gr.Group():
                gr.Markdown("### Anima Model Paths")
                with gr.Row():
                    self.qwen3 = gr.Textbox(
                        label="Qwen3-0.6B Text Encoder Path",
                        placeholder="Path to Qwen3-0.6B model directory or .safetensors",
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

                    self.anima_vae = gr.Textbox(
                        label="Qwen-Image VAE Path",
                        placeholder="Path to Qwen-Image VAE .safetensors or .pth",
                        value=self.config.get("anima.anima_vae", ""),
                        interactive=True,
                    )
                    self.anima_vae_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.anima_vae_button.click(
                        get_any_file_path,
                        outputs=self.anima_vae,
                        show_progress=False,
                    )

                with gr.Row():
                    self.llm_adapter_path = gr.Textbox(
                        label="LLM Adapter Path (Optional)",
                        placeholder="Path to LLM adapter .safetensors. If empty, loaded from DiT if present.",
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
                        label="T5 Tokenizer Path (Optional)",
                        placeholder="Path to T5 tokenizer directory. If empty, uses bundled configs/t5_old/.",
                        value=self.config.get("anima.t5_tokenizer_path", ""),
                        interactive=True,
                    )
                    self.t5_tokenizer_path_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.t5_tokenizer_path_button.click(
                        get_any_file_path,
                        outputs=self.t5_tokenizer_path,
                        show_progress=False,
                    )

                gr.Markdown("### Anima Training Parameters")
                with gr.Row():
                    self.anima_timestep_sampling = gr.Dropdown(
                        label="Timestep Sampling",
                        choices=["sigmoid", "sigma", "uniform", "shift", "flux_shift"],
                        value=self.config.get("anima.anima_timestep_sampling", "sigmoid"),
                        info="Timestep sampling method. Same options as FLUX training. Default: sigmoid.",
                        interactive=True,
                    )
                    self.anima_discrete_flow_shift = gr.Number(
                        label="Discrete Flow Shift",
                        value=self.config.get("anima.anima_discrete_flow_shift", 1.0),
                        info="Shift for timestep distribution in Rectified Flow. Default 1.0. Used when timestep_sampling=shift.",
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        interactive=True,
                    )
                    self.anima_sigmoid_scale = gr.Number(
                        label="Sigmoid Scale",
                        value=self.config.get("anima.anima_sigmoid_scale", 1.0),
                        info="Scale factor for sigmoid/shift/flux_shift timestep sampling. Default 1.0.",
                        minimum=0.001,
                        maximum=100.0,
                        step=0.1,
                        interactive=True,
                    )

                with gr.Row():
                    self.qwen3_max_token_length = gr.Number(
                        label="Qwen3 Max Token Length",
                        value=self.config.get("anima.qwen3_max_token_length", 512),
                        info="Maximum token length for Qwen3 tokenizer. Default 512.",
                        minimum=1,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )
                    self.t5_max_token_length = gr.Number(
                        label="T5 Max Token Length",
                        value=self.config.get("anima.t5_max_token_length", 512),
                        info="Maximum token length for T5 tokenizer. Default 512.",
                        minimum=1,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )
                    self.anima_split_attn = gr.Checkbox(
                        label="Split Attention",
                        value=self.config.get("anima.anima_split_attn", False),
                        info="Split attention computation to reduce memory. Required when using xformers attn_mode.",
                        interactive=True,
                    )

                gr.Markdown("### Memory & Speed")
                with gr.Row():
                    self.anima_cache_text_encoder_outputs = gr.Checkbox(
                        label="Cache Text Encoder Outputs",
                        value=self.config.get("anima.anima_cache_text_encoder_outputs", True),
                        info="Cache Qwen3 outputs to reduce VRAM. Enabled by default: TE LoRA is not supported at inference for Anima.",
                        interactive=True,
                    )
                    self.anima_cache_text_encoder_outputs_to_disk = gr.Checkbox(
                        label="Cache Text Encoder Outputs to Disk",
                        value=self.config.get("anima.anima_cache_text_encoder_outputs_to_disk", False),
                        info="Cache text encoder outputs to disk.",
                        interactive=True,
                    )
                    self.anima_blocks_to_swap = gr.Slider(
                        label="Blocks to Swap",
                        value=self.config.get("anima.anima_blocks_to_swap", 0),
                        info="Number of Transformer blocks to swap CPU<->GPU. 28-block model: max 26. Reduces VRAM at cost of speed.",
                        minimum=0,
                        maximum=34,
                        step=1,
                        interactive=True,
                    )
                    self.anima_unsloth_offload_checkpointing = gr.Checkbox(
                        label="Unsloth Offload Checkpointing",
                        value=self.config.get("anima.anima_unsloth_offload_checkpointing", False),
                        info="Offload activations to CPU RAM using async non-blocking transfers. Faster than cpu_offload_checkpointing. Cannot combine with blocks_to_swap.",
                        interactive=True,
                    )
                    self.anima_torch_compile = gr.Checkbox(
                        label="torch.compile",
                        value=self.config.get("anima.anima_torch_compile", False),
                        info="JIT-compile DiT with torch.compile (inductor backend). Can speed up training ~10-30%. Incompatible with Unsloth Offload Checkpointing.",
                        interactive=True,
                    )
                    self.anima_disable_mmap_load_safetensors = gr.Checkbox(
                        label="Disable mmap Load",
                        value=self.config.get("anima.anima_disable_mmap_load_safetensors", False),
                        info="Disable mmap for safetensors loading. Speeds up model loading on WSL2 or network drives.",
                        interactive=True,
                    )

                with gr.Row():
                    self.vae_chunk_size = gr.Number(
                        label="VAE Chunk Size",
                        value=self.config.get("anima.vae_chunk_size", 0),
                        info="Chunk size for Qwen-Image VAE processing to reduce VRAM. 0 = no chunking.",
                        minimum=0,
                        maximum=1024,
                        step=8,
                        interactive=True,
                    )
                    self.vae_disable_cache = gr.Checkbox(
                        label="VAE Disable Cache",
                        value=self.config.get("anima.vae_disable_cache", False),
                        info="Disable internal caching in Qwen-Image VAE to reduce VRAM.",
                        interactive=True,
                    )
                    self.anima_train_llm_adapter = gr.Checkbox(
                        label="Train LLM Adapter LoRA",
                        value=self.config.get("anima.anima_train_llm_adapter", False),
                        info="Apply LoRA to LLM Adapter blocks (6-layer transformer bridge from Qwen3 to T5-compatible space). Only supported with LoRA type 'Anima', ignored for LyCORIS variants.",
                        interactive=True,
                    )

                self.anima_checkbox.change(
                    lambda anima_checkbox: gr.Accordion(visible=anima_checkbox),
                    inputs=[self.anima_checkbox],
                    outputs=[anima_accordion],
                )