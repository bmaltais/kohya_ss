import gradio as gr
from .common_gui import (
    get_any_file_path,
    get_folder_path,
    document_symbol,
)


class mossImageTraining:
    """MOSS-Image (OpenMOSS) LoRA training options.

    Wraps `sd-scripts/moss_image_train_network.py` (sd-scripts >= 0.12.1):
    a 5B single-stream DiT with the Qwen3.5-9B text encoder and the FLUX.2
    VAE. See sd-scripts/docs/moss_image_train_network.md.
    """

    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        training_type: str = "",
        config: dict = {},
        moss_image_checkbox: gr.Checkbox = None,
    ) -> None:
        self.headless = headless
        self.finetuning = finetuning
        self.training_type = training_type
        self.config = config
        if moss_image_checkbox is None:
            raise ValueError("moss_image_checkbox must be provided")
        self.moss_image_checkbox = moss_image_checkbox

        with gr.Accordion(
            "MOSS-Image",
            open=True,
            visible=False,
            elem_classes=["moss_image_background"],
        ) as moss_image_accordion:
            with gr.Group():
                with gr.Row():
                    self.text_encoder = gr.Textbox(
                        label="Text Encoder Path",
                        placeholder="Path to Qwen3.5-9B text encoder checkpoint",
                        info=(
                            "First shard of the diffusers format "
                            "(model-00001-of-00004.safetensors) or the ComfyUI "
                            "single file. W4A8 GPTQ files are not supported."
                        ),
                        value=self.config.get("moss_image.text_encoder", ""),
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

                    self.vae = gr.Textbox(
                        label="VAE Path",
                        placeholder="Path to FLUX.2 VAE (flux2-vae.safetensors)",
                        info="ComfyUI / BFL format flux2-vae.safetensors recommended; the diffusers-format VAE also works.",
                        value=self.config.get("moss_image.vae", ""),
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
                    self.tokenizer_cache_dir = gr.Textbox(
                        label="Tokenizer Cache Dir",
                        placeholder="(Optional) HF cache dir or local tokenizer directory",
                        info=(
                            "The tokenizer is downloaded from Hugging Face "
                            "(OpenMOSS-Team/MOSS-Image-Base, subfolder "
                            "tokenizer) unless a cache directory or a local "
                            "directory containing tokenizer_config.json is "
                            "given here (offline training)."
                        ),
                        value=self.config.get("moss_image.tokenizer_cache_dir", ""),
                        interactive=True,
                    )
                    self.tokenizer_cache_dir_button = gr.Button(
                        "\U0001f4c2",
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.tokenizer_cache_dir_button.click(
                        get_folder_path,
                        outputs=self.tokenizer_cache_dir,
                        show_progress=False,
                    )

                    self.max_token_length = gr.Number(
                        label="Max Token Length",
                        value=self.config.get("moss_image.max_token_length", 4096),
                        info="Maximum number of text tokens including the chat template (default and maximum: 4096).",
                        minimum=1,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )

                with gr.Row():
                    self.discrete_flow_shift = gr.Number(
                        label="Discrete Flow Shift",
                        value=self.config.get("moss_image.discrete_flow_shift", 1.0),
                        info="Timestep distribution shift for rectified flow training (default: 1.0, the training grid of MOSS-Image).",
                        minimum=-1024,
                        maximum=1024,
                        step=0.01,
                        interactive=True,
                    )
                    self.timestep_sampling = gr.Dropdown(
                        label="Timestep Sampling",
                        choices=[
                            "sigma",
                            "uniform",
                            "sigmoid",
                            "shift",
                            "flux_shift",
                            "truncated_logit_normal",
                        ],
                        value=self.config.get(
                            "moss_image.timestep_sampling", "sigmoid"
                        ),
                        info=(
                            "The pre-training distribution is reproduced with "
                            "truncated_logit_normal + logit_mean -1.09 + "
                            "logit_std 0.95 (set in Additional parameters)."
                        ),
                        interactive=True,
                    )
                    self.sigmoid_scale = gr.Number(
                        label="Sigmoid Scale",
                        value=self.config.get("moss_image.sigmoid_scale", 1.0),
                        info="Scale factor for sigmoid (logit_normal) timestep sampling (default: 1.0).",
                        minimum=0.0,
                        maximum=1024,
                        step=0.01,
                        interactive=True,
                    )

                with gr.Row():
                    self.attn_mode = gr.Dropdown(
                        label="Attention Mode",
                        choices=["torch", "xformers", "flash", "sageattn"],
                        value=self.config.get("moss_image.attn_mode", "torch"),
                        info="xformers requires Split Attention. sageattn does not support training (inference only).",
                        interactive=True,
                    )
                    self.split_attn = gr.Checkbox(
                        label="Split Attention",
                        value=self.config.get("moss_image.split_attn", False),
                        info="Split attention computation to reduce memory usage. Required when using Attention Mode xformers.",
                        interactive=True,
                    )
                    self.sample_uncond_skip_layers = gr.Textbox(
                        label="Sample Uncond Skip Layers",
                        value=self.config.get(
                            "moss_image.sample_uncond_skip_layers", "22"
                        ),
                        info=(
                            "Comma separated DiT block indices skipped in the "
                            "negative (unconditional) branch of CFG during "
                            "sample image generation. Default is 22 "
                            "(official). Empty disables skipping."
                        ),
                        interactive=True,
                    )

                with gr.Row():
                    self.convrot_int8 = gr.Checkbox(
                        label="ConvRot INT8 (DiT)",
                        value=self.config.get("moss_image.convrot_int8", False),
                        info=(
                            "Quantize the frozen DiT base weights to ConvRot "
                            "INT8 at load time (alternative to fp8, which "
                            "MOSS-Image does not support). Fused int8 kernels "
                            "require triton (triton-windows on Windows); "
                            "without it a slower dequantizing fallback with "
                            "the same memory saving is used. ComfyUI "
                            "pre-quantized *_int8_convrot.safetensors files "
                            "are detected automatically."
                        ),
                        interactive=True,
                    )
                    self.convrot_int8_text_encoder = gr.Checkbox(
                        label="ConvRot INT8 (Text Encoder)",
                        value=self.config.get(
                            "moss_image.convrot_int8_text_encoder", False
                        ),
                        info="Quantize the Qwen3.5-9B text encoder to ConvRot INT8 at load time (about 16GB -> 9GB). Pre-quantized files are detected automatically.",
                        interactive=True,
                    )
                    self.convrot_int8_bwd = gr.Dropdown(
                        label="ConvRot INT8 Backward",
                        choices=["bf16", "int8"],
                        value=self.config.get("moss_image.convrot_int8_bwd", "bf16"),
                        info=(
                            "Backward mode of the ConvRot INT8 layers. bf16 "
                            "(default): transient dequantized matmul, most "
                            "accurate. int8: fused int8 GEMM for grad_x "
                            "(requires triton and CUDA)."
                        ),
                        interactive=True,
                    )

                with gr.Row():
                    self.unsloth_offload_checkpointing = gr.Checkbox(
                        label="Unsloth Offload Checkpointing",
                        value=self.config.get(
                            "moss_image.unsloth_offload_checkpointing", False
                        ),
                        info="Offload activations to CPU RAM using async non-blocking transfers (faster than CPU Offload Checkpointing). Cannot be combined with CPU Offload Checkpointing or Blocks to swap.",
                        interactive=True,
                    )
                    self.cache_text_encoder_outputs = gr.Checkbox(
                        label="Cache Text Encoder Outputs",
                        value=self.config.get(
                            "moss_image.cache_text_encoder_outputs", False
                        ),
                        info="Cache Qwen3.5-9B outputs to reduce VRAM usage. Recommended (and required to keep the text encoder off the GPU). Incompatible with Text Encoder LoRA training.",
                        interactive=True,
                    )
                    self.cache_text_encoder_outputs_to_disk = gr.Checkbox(
                        label="Cache Text Encoder Outputs to Disk",
                        value=self.config.get(
                            "moss_image.cache_text_encoder_outputs_to_disk", False
                        ),
                        info="Cache text encoder outputs to disk. Automatically enables Cache Text Encoder Outputs.",
                        interactive=True,
                    )

                with gr.Row():
                    self.caption_candidates = gr.Checkbox(
                        label="Multi-Caption Candidates (Tag Dropout)",
                        value=self.config.get("moss_image.caption_candidates", False),
                        info=(
                            "Generate tag-dropout caption variants and let the "
                            "trainer pick one at random every step. Works WITH "
                            "Cache Text Encoder Outputs (which forbids "
                            "--caption_tag_dropout_rate). Writes <image>.multi "
                            "next to each image and a derived "
                            "<name>.moss-multi.toml; the original captions and "
                            "dataset TOML are never modified. Requires a dataset "
                            "config TOML with image_dir subsets."
                        ),
                        interactive=True,
                    )
                    self.caption_candidate_count = gr.Number(
                        label="Caption Candidates",
                        value=self.config.get("moss_image.caption_candidate_count", 3),
                        info="Total candidates per image: the original caption plus dropout variants. Cache size and caching time scale with this number.",
                        minimum=2,
                        maximum=8,
                        step=1,
                        interactive=True,
                    )
                    self.caption_tag_dropout_rate = gr.Number(
                        label="Caption Tag Dropout Rate",
                        value=self.config.get(
                            "moss_image.caption_tag_dropout_rate", 0.15
                        ),
                        info="Probability of dropping each tag in the variants. The first Keep n tokens tags are always kept.",
                        minimum=0.05,
                        maximum=0.9,
                        step=0.05,
                        interactive=True,
                    )
                    self.caption_candidates_regenerate = gr.Checkbox(
                        label="Regenerate Candidates",
                        value=self.config.get(
                            "moss_image.caption_candidates_regenerate", False
                        ),
                        info="Rewrite existing .multi files. Needed after changing the candidate count, dropout rate or Keep n tokens; otherwise existing files are reused.",
                        interactive=True,
                    )

                self.moss_image_checkbox.change(
                    lambda moss_image_checkbox: gr.Accordion(
                        visible=moss_image_checkbox
                    ),
                    inputs=[self.moss_image_checkbox],
                    outputs=[moss_image_accordion],
                )
