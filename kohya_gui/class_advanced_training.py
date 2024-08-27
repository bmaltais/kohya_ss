import gradio as gr
from typing import Tuple
from .common_gui import (
    get_folder_path,
    get_any_file_path,
    list_files,
    list_dirs,
    create_refresh_button,
    document_symbol,
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
        self.current_vae_dir = self.config.get("advanced.vae_dir", "./models/vae")
        self.current_state_dir = self.config.get("advanced.state_dir", "./outputs")
        self.current_log_tracker_config_dir = self.config.get(
            "advanced.log_tracker_config_dir", "./logs"
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
                    label="No token padding",
                    value=self.config.get("advanced.no_token_padding", False),
                )
            self.gradient_accumulation_steps = gr.Slider(
                label="Gradient accumulate steps",
                info="Number of updates steps to accumulate before performing a backward/update pass",
                value=self.config.get("advanced.gradient_accumulation_steps", 1),
                minimum=1,
                maximum=120,
                step=1,
            )
            self.weighted_captions = gr.Checkbox(
                label="Weighted captions",
                value=self.config.get("advanced.weighted_captions", False),
            )
        with gr.Group(), gr.Row(visible=not finetuning):
            self.prior_loss_weight = gr.Number(
                label="Prior loss weight",
                value=self.config.get("advanced.prior_loss_weight", 1.0),
            )

            def list_vae_files(path):
                self.current_vae_dir = path if not path == "" else "."
                return list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

            self.vae = gr.Dropdown(
                label="VAE (Optional: Path to checkpoint of vae for training)",
                interactive=True,
                choices=[self.config.get("advanced.vae_dir", "")]
                + list_vae_files(self.current_vae_dir),
                value=self.config.get("advanced.vae_dir", ""),
                allow_custom_value=True,
            )
            create_refresh_button(
                self.vae,
                lambda: None,
                lambda: {
                    "choices": [self.config.get("advanced.vae_dir", "")]
                    + list_vae_files(self.current_vae_dir)
                },
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
                fn=lambda path: gr.Dropdown(
                    choices=[self.config.get("advanced.vae_dir", "")]
                    + list_vae_files(path)
                ),
                inputs=self.vae,
                outputs=self.vae,
                show_progress=False,
            )

        with gr.Row():
            self.additional_parameters = gr.Textbox(
                label="Additional parameters",
                placeholder='(Optional) Use to provide additional parameters not handled by the GUI. Eg: --some_parameters "value"',
                value=self.config.get("advanced.additional_parameters", ""),
            )
        with gr.Accordion("Scheduled Huber Loss", open=False):
            with gr.Row():
                self.loss_type = gr.Dropdown(
                    label="Loss type",
                    choices=["huber", "smooth_l1", "l1", "l2"],
                    value=self.config.get("advanced.loss_type", "l2"),
                    info="The type of loss to use and whether it's scheduled based on the timestep",
                )
                self.huber_schedule = gr.Dropdown(
                    label="Huber schedule",
                    choices=[
                        "constant",
                        "exponential",
                        "snr",
                    ],
                    value=self.config.get("advanced.huber_schedule", "snr"),
                    info="The type of loss to use and whether it's scheduled based on the timestep",
                )
                self.huber_c = gr.Number(
                    label="Huber C",
                    value=self.config.get("advanced.huber_c", 0.1),
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    info="The huber loss parameter. Only used if one of the huber loss modes (huber or smooth l1) is selected with loss_type",
                )

        with gr.Row():
            self.save_every_n_steps = gr.Number(
                label="Save every N steps",
                value=self.config.get("advanced.save_every_n_steps", 0),
                precision=0,
                info="(Optional) The model is saved every specified steps",
            )
            self.save_last_n_steps = gr.Number(
                label="Save last N steps",
                value=self.config.get("advanced.save_last_n_steps", 0),
                precision=0,
                info="(Optional) Save only the specified number of models (old models will be deleted)",
            )
            self.save_last_n_steps_state = gr.Number(
                label="Save last N steps state",
                value=self.config.get("advanced.save_last_n_steps_state", 0),
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
                label="Keep n tokens",
                value=self.config.get("advanced.keep_tokens", 0),
                minimum=0,
                maximum=32,
                step=1,
            )
            self.clip_skip = gr.Slider(
                label="Clip skip",
                value=self.config.get("advanced.clip_skip", 1),
                minimum=0,
                maximum=12,
                step=1,
            )
            self.max_token_length = gr.Dropdown(
                label="Max Token Length",
                choices=[
                    75,
                    150,
                    225,
                ],
                info="max token length of text encoder",
                value=self.config.get("advanced.max_token_length", 75),
            )

        with gr.Row():
            self.fp8_base = gr.Checkbox(
                label="fp8 base",
                info="Use fp8 for base model",
                value=self.config.get("advanced.fp8_base", False),
            )
            self.fp8_base_unet  = gr.Checkbox(
                label="fp8 base unet",
                info="Flux can be trained with fp8, and CLIP-L can be trained with bf16/fp16.",
                value=self.config.get("advanced.fp8_base_unet", False),
            )
            self.full_fp16 = gr.Checkbox(
                label="Full fp16 training (experimental)",
                value=self.config.get("advanced.full_fp16", False),
            )
            self.full_bf16 = gr.Checkbox(
                label="Full bf16 training (experimental)",
                value=self.config.get("advanced.full_bf16", False),
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
            self.highvram = gr.Checkbox(
                label="highvram",
                value=self.config.get("advanced.highvram", False),
                info="Disable low VRAM optimization. e.g. do not clear CUDA cache after each latent caching (for machines which have bigger VRAM)",
                interactive=True,
            )
            self.lowvram = gr.Checkbox(
                label="lowvram",
                value=self.config.get("advanced.lowvram", False),
                info="Enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle)",
                interactive=True,
            )

        with gr.Row():
            self.gradient_checkpointing = gr.Checkbox(
                label="Gradient checkpointing",
                value=self.config.get("advanced.gradient_checkpointing", False),
            )
            self.shuffle_caption = gr.Checkbox(
                label="Shuffle caption",
                value=self.config.get("advanced.shuffle_caption", False),
            )
            self.persistent_data_loader_workers = gr.Checkbox(
                label="Persistent data loader",
                value=self.config.get("advanced.persistent_data_loader_workers", False),
            )
            self.mem_eff_attn = gr.Checkbox(
                label="Memory efficient attention",
                value=self.config.get("advanced.mem_eff_attn", False),
            )
        with gr.Row():
            self.xformers = gr.Dropdown(
                label="CrossAttention",
                choices=["none", "sdpa", "xformers"],
                value=self.config.get("advanced.xformers", "xformers"),
            )
            self.color_aug = gr.Checkbox(
                label="Color augmentation",
                value=self.config.get("advanced.color_aug", False),
                info="Enable weak color augmentation",
            )
            self.flip_aug = gr.Checkbox(
                label="Flip augmentation",
                value=getattr(self.config, "advanced.flip_aug", False),
                info="Enable horizontal flip augmentation",
            )
            self.masked_loss = gr.Checkbox(
                label="Masked loss",
                value=self.config.get("advanced.masked_loss", False),
                info="Apply mask for calculating loss. conditioning_data_dir is required for dataset",
            )
        with gr.Row():
            self.scale_v_pred_loss_like_noise_pred = gr.Checkbox(
                label="Scale v prediction loss",
                value=self.config.get(
                    "advanced.scale_v_pred_loss_like_noise_pred", False
                ),
                info="Only for SD v2 models. By scaling the loss according to the time step, the weights of global noise prediction and local noise prediction become the same, and the improvement of details may be expected.",
            )
            self.min_snr_gamma = gr.Slider(
                label="Min SNR gamma",
                value=self.config.get("advanced.min_snr_gamma", 0),
                minimum=0,
                maximum=20,
                step=1,
                info="Recommended value of 5 when used",
            )
            self.debiased_estimation_loss = gr.Checkbox(
                label="Debiased Estimation loss",
                value=self.config.get("advanced.debiased_estimation_loss", False),
                info="Automates the processing of noise, allowing for faster model fitting, as well as balancing out color issues. Do not use if Min SNR gamma is specified.",
            )
        with gr.Row():
            # self.sdpa = gr.Checkbox(label='Use sdpa', value=False, info='Use sdpa for CrossAttention')
            self.bucket_no_upscale = gr.Checkbox(
                label="Don't upscale bucket resolution",
                value=self.config.get("advanced.bucket_no_upscale", True),
            )
            self.bucket_reso_steps = gr.Slider(
                label="Bucket resolution steps",
                value=self.config.get("advanced.bucket_reso_steps", 64),
                minimum=1,
                maximum=128,
            )
            self.random_crop = gr.Checkbox(
                label="Random crop instead of center crop",
                value=self.config.get("advanced.random_crop", False),
            )
            self.v_pred_like_loss = gr.Slider(
                label="V Pred like loss",
                value=self.config.get("advanced.v_pred_like_loss", 0),
                minimum=0,
                maximum=1,
                step=0.01,
                info="Recommended value of 0.5 when used",
            )

        with gr.Row():
            self.min_timestep = gr.Slider(
                label="Min Timestep",
                value=self.config.get("advanced.min_timestep", 0),
                step=1,
                minimum=0,
                maximum=1000,
                info="Values greater than 0 will make the model more img2img focussed. 0 = image only",
            )
            self.max_timestep = gr.Slider(
                label="Max Timestep",
                value=self.config.get("advanced.max_timestep", 1000),
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
                value=self.config.get("advanced.noise_offset_type", "Original"),
                scale=1,
            )
            with gr.Row(visible=True) as self.noise_offset_original:
                self.noise_offset = gr.Slider(
                    label="Noise offset",
                    value=self.config.get("advanced.noise_offset", 0),
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    info="Recommended values are 0.05 - 0.15",
                )
                self.noise_offset_random_strength = gr.Checkbox(
                    label="Noise offset random strength",
                    value=self.config.get(
                        "advanced.noise_offset_random_strength", False
                    ),
                    info="Use random strength between 0~noise_offset for noise offset",
                )
                self.adaptive_noise_scale = gr.Slider(
                    label="Adaptive noise scale",
                    value=self.config.get("advanced.adaptive_noise_scale", 0),
                    minimum=-1,
                    maximum=1,
                    step=0.001,
                    info="Add `latent mean absolute value * this value` to noise_offset",
                )
            with gr.Row(visible=False) as self.noise_offset_multires:
                self.multires_noise_iterations = gr.Slider(
                    label="Multires noise iterations",
                    value=self.config.get("advanced.multires_noise_iterations", 0),
                    minimum=0,
                    maximum=64,
                    step=1,
                    info="Enable multires noise (recommended values are 6-10)",
                )
                self.multires_noise_discount = gr.Slider(
                    label="Multires noise discount",
                    value=self.config.get("advanced.multires_noise_discount", 0.3),
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    info="Recommended values are 0.8. For LoRAs with small datasets, 0.1-0.3",
                )
            with gr.Row(visible=True):
                self.ip_noise_gamma = gr.Slider(
                    label="IP noise gamma",
                    value=self.config.get("advanced.ip_noise_gamma", 0),
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    info="enable input perturbation noise. used for regularization. recommended value: around 0.1",
                )
                self.ip_noise_gamma_random_strength = gr.Checkbox(
                    label="IP noise gamma random strength",
                    value=self.config.get(
                        "advanced.ip_noise_gamma_random_strength", False
                    ),
                    info="Use random strength between 0~ip_noise_gamma for input perturbation noise",
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
                label="Dropout caption every n epochs",
                value=self.config.get("advanced.caption_dropout_every_n_epochs", 0),
            )
            self.caption_dropout_rate = gr.Slider(
                label="Rate of caption dropout",
                value=self.config.get("advanced.caption_dropout_rate", 0),
                minimum=0,
                maximum=1,
            )
            self.vae_batch_size = gr.Slider(
                label="VAE batch size",
                minimum=0,
                maximum=32,
                value=self.config.get("advanced.vae_batch_size", 0),
                step=1,
            )
        with gr.Group(), gr.Row():
            self.save_state = gr.Checkbox(
                label="Save training state",
                value=self.config.get("advanced.save_state", False),
                info="Save training state (including optimizer states etc.) when saving models"
            )

            self.save_state_on_train_end = gr.Checkbox(
                label="Save training state at end of training",
                value=self.config.get("advanced.save_state_on_train_end", False),
                info="Save training state (including optimizer states etc.) on train end"
            )

            def list_state_dirs(path):
                self.current_state_dir = path if not path == "" else "."
                return list(list_dirs(path))

            self.resume = gr.Dropdown(
                label='Resume from saved training state (path to "last-state" state folder)',
                choices=[self.config.get("advanced.state_dir", "")]
                + list_state_dirs(self.current_state_dir),
                value=self.config.get("advanced.state_dir", ""),
                interactive=True,
                allow_custom_value=True,
                info="Saved state to resume training from"
            )
            create_refresh_button(
                self.resume,
                lambda: None,
                lambda: {
                    "choices": [self.config.get("advanced.state_dir", "")]
                    + list_state_dirs(self.current_state_dir)
                },
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
                fn=lambda path: gr.Dropdown(
                    choices=[self.config.get("advanced.state_dir", "")]
                    + list_state_dirs(path)
                ),
                inputs=self.resume,
                outputs=self.resume,
                show_progress=False,
            )
            self.max_data_loader_n_workers = gr.Number(
                label="Max num workers for DataLoader",
                info="Override number of epoch. Default: 0",
                step=1,
                minimum=0,
                value=self.config.get("advanced.max_data_loader_n_workers", 0),
            )
        with gr.Row():
            self.log_with = gr.Dropdown(
                label="Logging",
                choices=["","wandb", "tensorboard","all"],
                value="",
                info="Loggers to use, tensorboard will be used as the default.",
            )
            self.wandb_api_key = gr.Textbox(
                label="WANDB API Key",
                value=self.config.get("advanced.wandb_api_key", ""),
                placeholder="(Optional)",
                info="Users can obtain and/or generate an api key in the their user settings on the website: https://wandb.ai/login",
            )
            self.wandb_run_name = gr.Textbox(
                label="WANDB run name",
                value=self.config.get("advanced.wandb_run_name", ""),
                placeholder="(Optional)",
                info="The name of the specific wandb session",
            )
        with gr.Group(), gr.Row():

            def list_log_tracker_config_files(path):
                self.current_log_tracker_config_dir = path if not path == "" else "."
                return list(list_files(path, exts=[".json"], all=True))

            self.log_config = gr.Checkbox(
                label="Log config",
                value=self.config.get("advanced.log_config", False),
                info="Log training parameter to WANDB",
            )
            self.log_tracker_name = gr.Textbox(
                label="Log tracker name",
                value=self.config.get("advanced.log_tracker_name", ""),
                placeholder="(Optional)",
                info="Name of tracker to use for logging, default is script-specific default name",
            )
            self.log_tracker_config = gr.Dropdown(
                label="Log tracker config",
                choices=[self.config.get("log_tracker_config_dir", "")]
                + list_log_tracker_config_files(self.current_log_tracker_config_dir),
                value=self.config.get("log_tracker_config_dir", ""),
                info="Path to tracker config file to use for logging",
                interactive=True,
                allow_custom_value=True,
            )
            create_refresh_button(
                self.log_tracker_config,
                lambda: None,
                lambda: {
                    "choices": [self.config.get("log_tracker_config_dir", "")]
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
                    choices=[self.config.get("log_tracker_config_dir", "")]
                    + list_log_tracker_config_files(path)
                ),
                inputs=self.log_tracker_config,
                outputs=self.log_tracker_config,
                show_progress=False,
            )
