import gradio as gr
import os
from typing import Tuple
from .common_gui import (
    get_folder_path,
    get_any_file_path,
    list_files,
    list_dirs,
    create_refresh_button,
    document_symbol
)


class AdvancedTraining:
    """
    This class configures and initializes the advanced training settings for a machine learning model,
    including options for headless operation, fine-tuning, training type selection, and default directory paths.

    Attributes:
        headless (bool): If True, run without the Gradio interface.
        finetuning (bool): If True, enables fine-tuning of the model.
        training_type (str): Specifies the type of training to perform.
        no_token_padding (gr.Checkbox): Checkbox to disable token padding.
        gradient_accumulation_steps (gr.Slider): Slider to set the number of gradient accumulation steps.
        weighted_captions (gr.Checkbox): Checkbox to enable weighted captions.
    """

    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        training_type: str = "",
        config: dict = {},
    ) -> None:
        """
        Initializes the AdvancedTraining class with given settings.

        Parameters:
            headless (bool): Run in headless mode without GUI.
            finetuning (bool): Enable model fine-tuning.
            training_type (str): The type of training to be performed.
            config (dict): Configuration options for the training process.
        """
        self.headless = headless
        self.finetuning = finetuning
        self.training_type = training_type
        self.config = config

        # Determine the current directories for VAE and output, falling back to defaults if not specified.
        self.current_vae_dir = self.config.get("vae_dir", "./models/vae")
        self.current_state_dir = self.config.get("state_dir", "./outputs")
        self.current_log_tracker_config_dir = self.config.get(
            "log_tracker_config_dir", "./logs"
        )

        # Define the behavior for changing noise offset type.
        def noise_offset_type_change(
            noise_offset_type: str,
        ) -> Tuple[gr.Group, gr.Group]:
            """
            Returns a tuple of Gradio Groups with visibility set based on the noise offset type.

            Parameters:
                noise_offset_type (str): The selected noise offset type.

            Returns:
                Tuple[gr.Group, gr.Group]: A tuple containing two Gradio Group elements with their visibility set.
            """
            if noise_offset_type == "Original":
                return (gr.Group(visible=True), gr.Group(visible=False))
            else:
                return (gr.Group(visible=False), gr.Group(visible=True))

        # GUI elements are only visible when not fine-tuning.
        with gr.Row(visible=not finetuning):
            # Exclude token padding option for LoRA training type.
            if training_type != "lora":
                self.no_token_padding = gr.Checkbox(
                    label="No token padding", value=False
                )
            self.gradient_accumulation_steps = gr.Slider(
                label="Gradient accumulate steps",
                info="Number of updates steps to accumulate before performing a backward/update pass",
                value="1",
                minimum=1,
                maximum=120,
                step=1,
            )
            self.weighted_captions = gr.Checkbox(label="Weighted captions", value=False)
        with gr.Group(), gr.Row(visible=not finetuning):
            self.prior_loss_weight = gr.Number(label="Prior loss weight", value=1.0)

            def list_vae_files(path):
                self.current_vae_dir = path if not path == "" else "."
                return list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

            self.vae = gr.Dropdown(
                label="VAE (Optional: Path to checkpoint of vae for training)",
                interactive=True,
                choices=[""] + list_vae_files(self.current_vae_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                self.vae,
                lambda: None,
                lambda: {"choices": [""] + list_vae_files(self.current_vae_dir)},
                "open_folder_small",
            )
            self.vae_button = gr.Button(
                "📂", elem_id="open_folder_small", visible=(not headless)
            )
            self.vae_button.click(
                get_any_file_path,
                outputs=self.vae,
                show_progress=False,
            )

            self.vae.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_vae_files(path)),
                inputs=self.vae,
                outputs=self.vae,
                show_progress=False,
            )

        with gr.Row():
            self.additional_parameters = gr.Textbox(
                label="Additional parameters",
                placeholder='(Optional) Use to provide additional parameters not handled by the GUI. Eg: --some_parameters "value"',
            )
        with gr.Row():
            self.save_every_n_steps = gr.Number(
                label="Save every N steps",
                value=0,
                precision=0,
                info="(Optional) The model is saved every specified steps",
            )
            self.save_last_n_steps = gr.Number(
                label="Save last N steps",
                value=0,
                precision=0,
                info="(Optional) Save only the specified number of models (old models will be deleted)",
            )
            self.save_last_n_steps_state = gr.Number(
                label="Save last N steps state",
                value=0,
                precision=0,
                info="(Optional) Save only the specified number of states (old models will be deleted)",
            )
        with gr.Row():

            def full_options_update(full_fp16, full_bf16):
                full_fp16_active = True
                full_bf16_active = True

                if full_fp16:
                    full_bf16_active = False
                if full_bf16:
                    full_fp16_active = False
                return gr.Checkbox(
                    interactive=full_fp16_active,
                ), gr.Checkbox(interactive=full_bf16_active)

            self.keep_tokens = gr.Slider(
                label="Keep n tokens", value="0", minimum=0, maximum=32, step=1
            )
            self.clip_skip = gr.Slider(
                label="Clip skip", value="1", minimum=1, maximum=12, step=1
            )
            self.max_token_length = gr.Dropdown(
                label="Max Token Length",
                choices=[
                    "75",
                    "150",
                    "225",
                ],
                value="75",
            )

        with gr.Row():
            if training_type == "lora":
                self.fp8_base = gr.Checkbox(
                    label="fp8 base training (experimental)",
                    info="U-Net and Text Encoder can be trained with fp8 (experimental)",
                    value=False,
                )
            self.full_fp16 = gr.Checkbox(
                label="Full fp16 training (experimental)",
                value=False,
            )
            self.full_bf16 = gr.Checkbox(
                label="Full bf16 training (experimental)",
                value=False,
                info="Required bitsandbytes >= 0.36.0",
            )

            self.full_fp16.change(
                full_options_update,
                inputs=[self.full_fp16, self.full_bf16],
                outputs=[self.full_fp16, self.full_bf16],
            )
            self.full_bf16.change(
                full_options_update,
                inputs=[self.full_fp16, self.full_bf16],
                outputs=[self.full_fp16, self.full_bf16],
            )

        with gr.Row():
            self.gradient_checkpointing = gr.Checkbox(
                label="Gradient checkpointing", value=False
            )
            self.shuffle_caption = gr.Checkbox(label="Shuffle caption", value=False)
            self.persistent_data_loader_workers = gr.Checkbox(
                label="Persistent data loader", value=False
            )
            self.mem_eff_attn = gr.Checkbox(
                label="Memory efficient attention", value=False
            )
        with gr.Row():
            self.xformers = gr.Dropdown(
                label="CrossAttention",
                choices=["none", "sdpa", "xformers"],
                value="xformers",
            )
            self.color_aug = gr.Checkbox(label="Color augmentation", value=False)
            self.flip_aug = gr.Checkbox(label="Flip augmentation", value=False)
            self.min_snr_gamma = gr.Slider(
                label="Min SNR gamma",
                value=0,
                minimum=0,
                maximum=20,
                step=1,
                info="Recommended value of 5 when used",
            )
        with gr.Row():
            # self.sdpa = gr.Checkbox(label='Use sdpa', value=False, info='Use sdpa for CrossAttention')
            self.bucket_no_upscale = gr.Checkbox(
                label="Don't upscale bucket resolution", value=True
            )
            self.bucket_reso_steps = gr.Slider(
                label="Bucket resolution steps",
                value=64,
                minimum=1,
                maximum=128,
            )
            self.random_crop = gr.Checkbox(
                label="Random crop instead of center crop", value=False
            )
            self.v_pred_like_loss = gr.Slider(
                label="V Pred like loss",
                value=0,
                minimum=0,
                maximum=1,
                step=0.01,
                info="Recommended value of 0.5 when used",
            )

        with gr.Row():
            self.min_timestep = gr.Slider(
                label="Min Timestep",
                value=0,
                step=1,
                minimum=0,
                maximum=1000,
                info="Values greater than 0 will make the model more img2img focussed. 0 = image only",
            )
            self.max_timestep = gr.Slider(
                label="Max Timestep",
                value=1000,
                step=1,
                minimum=0,
                maximum=1000,
                info="Values lower than 1000 will make the model more img2img focussed. 1000 = noise only",
            )

        with gr.Row():
            self.noise_offset_type = gr.Dropdown(
                label="Noise offset type",
                choices=[
                    "Original",
                    "Multires",
                ],
                value="Original",
            )
            with gr.Row(visible=True) as self.noise_offset_original:
                self.noise_offset = gr.Slider(
                    label="Noise offset",
                    value=0,
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    info='Recommended values are 0.05 - 0.15',
                )
                self.adaptive_noise_scale = gr.Slider(
                    label="Adaptive noise scale",
                    value=0,
                    minimum=-1,
                    maximum=1,
                    step=0.001,
                    info="(Experimental, Optional) Since the latent is close to a normal distribution, it may be a good idea to specify a value around 1/10 the noise offset.",
                )
            with gr.Row(visible=False) as self.noise_offset_multires:
                self.multires_noise_iterations = gr.Slider(
                    label="Multires noise iterations",
                    value=0,
                    minimum=0,
                    maximum=64,
                    step=1,
                  info='Enable multires noise (recommended values are 6-10)',
                )
                self.multires_noise_discount = gr.Slider(
                    label="Multires noise discount",
                    value=0,
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    info='Recommended values are 0.8. For LoRAs with small datasets, 0.1-0.3',
                )
            self.noise_offset_type.change(
                noise_offset_type_change,
                inputs=[self.noise_offset_type],
                outputs=[
                    self.noise_offset_original,
                    self.noise_offset_multires,
                ],
            )
        with gr.Row():
            self.caption_dropout_every_n_epochs = gr.Number(
                label="Dropout caption every n epochs", value=0
            )
            self.caption_dropout_rate = gr.Slider(
                label="Rate of caption dropout", value=0, minimum=0, maximum=1
            )
            self.vae_batch_size = gr.Slider(
                label="VAE batch size", minimum=0, maximum=32, value=0, step=1
            )
        with gr.Group(), gr.Row():
            self.save_state = gr.Checkbox(label="Save training state", value=False)

            def list_state_dirs(path):
                self.current_state_dir = path if not path == "" else "."
                return list(list_dirs(path))

            self.resume = gr.Dropdown(
                label='Resume from saved training state (path to "last-state" state folder)',
                choices=[""] + list_state_dirs(self.current_state_dir),
                value="",
                interactive=True,
                allow_custom_value=True,
            )
            create_refresh_button(
                self.resume,
                lambda: None,
                lambda: {"choices": [""] + list_state_dirs(self.current_state_dir)},
                "open_folder_small",
            )
            self.resume_button = gr.Button(
                "📂", elem_id="open_folder_small", visible=(not headless)
            )
            self.resume_button.click(
                get_folder_path,
                outputs=self.resume,
                show_progress=False,
            )
            self.resume.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_state_dirs(path)),
                inputs=self.resume,
                outputs=self.resume,
                show_progress=False,
            )
            self.max_data_loader_n_workers = gr.Textbox(
                label="Max num workers for DataLoader",
                placeholder="(Optional) Override number of epoch. Default: 8",
                value="0",
            )
        with gr.Row():
            self.num_processes = gr.Number(
                label="Number of processes", value=1, precision=0, minimum=1
            )
            self.num_machines = gr.Number(
                label="Number of machines", value=1, precision=0, minimum=1
            )
            self.multi_gpu = gr.Checkbox(label="Multi GPU", value=False)
            self.gpu_ids = gr.Textbox(
                label="GPU IDs", value="", placeholder="example: 0,1"
            )
        with gr.Row():
            self.use_wandb = gr.Checkbox(
                label="WANDB Logging",
                value=False,
                info="If unchecked, tensorboard will be used as the default for logging.",
            )
            self.wandb_api_key = gr.Textbox(
                label="WANDB API Key",
                value="",
                placeholder="(Optional)",
                info="Users can obtain and/or generate an api key in the their user settings on the website: https://wandb.ai/login",
            )
            self.wandb_run_name = gr.Textbox(
                label="WANDB run name",
                value="",
                placeholder="(Optional)",
                info="The name of the specific wandb session",
            )
        with gr.Group(), gr.Row():

            def list_log_tracker_config_files(path):
                self.current_log_tracker_config_dir = path if not path == "" else "."
                return list(list_files(path, exts=[".json"], all=True))

            self.log_tracker_name = gr.Textbox(
                label="Log tracker name",
                value="",
                placeholder="(Optional)",
                info="Name of tracker to use for logging, default is script-specific default name",
            )
            self.log_tracker_config = gr.Dropdown(
                label="Log tracker config",
                choices=[""]
                + list_log_tracker_config_files(self.current_log_tracker_config_dir),
                value="",
                info="Path to tracker config file to use for logging",
                interactive=True,
                allow_custom_value=True,
            )
            create_refresh_button(
                self.log_tracker_config,
                lambda: None,
                lambda: {
                    "choices": [""]
                    + list_log_tracker_config_files(self.current_log_tracker_config_dir)
                },
                "open_folder_small",
            )
            self.log_tracker_config_button = gr.Button(
                document_symbol, elem_id="open_folder_small", visible=(not headless)
            )
            self.log_tracker_config_button.click(
                get_any_file_path,
                outputs=self.log_tracker_config,
                show_progress=False,
            )
            self.log_tracker_config.change(
                fn=lambda path: gr.Dropdown(
                    choices=[""] + list_log_tracker_config_files(path)
                ),
                inputs=self.log_tracker_config,
                outputs=self.log_tracker_config,
                show_progress=False,
            )
        with gr.Row():
            self.scale_v_pred_loss_like_noise_pred = gr.Checkbox(
                label="Scale v prediction loss",
                value=False,
                info="Only for SD v2 models. By scaling the loss according to the time step, the weights of global noise prediction and local noise prediction become the same, and the improvement of details may be expected.",
            )
            self.debiased_estimation_loss = gr.Checkbox(
                label="Debiased Estimation loss",
                value=False,
                info="Automates the processing of noise, allowing for faster model fitting, as well as balancing out color issues",
            )
