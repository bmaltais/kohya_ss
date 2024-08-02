import gradio as gr
from typing import Tuple
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

class BasicTraining:
    """
    This class configures and initializes the basic training settings for a machine learning model,
    including options for SDXL, learning rate, learning rate scheduler, and training epochs.

    Attributes:
        sdxl_checkbox (gr.Checkbox): Checkbox to enable SDXL training.
        learning_rate_value (str): Initial learning rate value.
        lr_scheduler_value (str): Initial learning rate scheduler value.
        lr_warmup_value (str): Initial learning rate warmup value.
        finetuning (bool): If True, enables fine-tuning of the model.
        dreambooth (bool): If True, enables Dreambooth training.
    """

    def __init__(
        self,
        sdxl_checkbox: gr.Checkbox,
        learning_rate_value: float = "1e-6",
        lr_scheduler_value: str = "constant",
        lr_warmup_value: float = "0",
        finetuning: bool = False,
        dreambooth: bool = False,
        config: dict = {},
    ) -> None:
        """
        Initializes the BasicTraining object with the given parameters.

        Args:
            sdxl_checkbox (gr.Checkbox): Checkbox to enable SDXL training.
            learning_rate_value (str): Initial learning rate value.
            lr_scheduler_value (str): Initial learning rate scheduler value.
            lr_warmup_value (str): Initial learning rate warmup value.
            finetuning (bool): If True, enables fine-tuning of the model.
            dreambooth (bool): If True, enables Dreambooth training.
        """
        self.sdxl_checkbox = sdxl_checkbox
        self.learning_rate_value = learning_rate_value
        self.lr_scheduler_value = lr_scheduler_value
        self.lr_warmup_value = lr_warmup_value
        self.finetuning = finetuning
        self.dreambooth = dreambooth
        self.config = config
        self.old_lr_warmup = 0

        # Initialize the UI components
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        """
        Initializes the UI components for the training settings.
        """
        # Initialize the training controls
        self.init_training_controls()
        # Initialize the precision and resources controls
        self.init_precision_and_resources_controls()
        # Initialize the learning rate and optimizer controls
        self.init_lr_and_optimizer_controls()
        # Initialize the gradient and learning rate controls
        self.init_grad_and_lr_controls()
        # Initialize the learning rate controls
        self.init_learning_rate_controls()
        # Initialize the scheduler controls
        self.init_scheduler_controls()
        # Initialize the resolution and bucket controls
        self.init_resolution_and_bucket_controls()
        # Setup the behavior of the SDXL checkbox
        self.setup_sdxl_checkbox_behavior()

    def init_training_controls(self) -> None:
        """
        Initializes the training controls for the model.
        """
        # Create a row for the training controls
        with gr.Row():
            # Initialize the train batch size slider
            self.train_batch_size = gr.Slider(
                minimum=1,
                maximum=64,
                label="Train batch size",
                value=1,
                step=self.config.get("basic.train_batch_size", 1),
            )
            # Initialize the epoch number input
            self.epoch = gr.Number(
                label="Epoch", value=self.config.get("basic.epoch", 1), precision=0
            )
            # Initialize the maximum train epochs input
            self.max_train_epochs = gr.Number(
                label="Max train epoch",
                info="training epochs (overrides max_train_steps). 0 = no override",
                step=1,
                # precision=0,
                minimum=0,
                value=self.config.get("basic.max_train_epochs", 0),
            )
            # Initialize the maximum train steps input
            self.max_train_steps = gr.Number(
                label="Max train steps",
                info="Overrides # training steps. 0 = no override",
                step=1,
                # precision=0,
                value=self.config.get("basic.max_train_steps", 1600),
            )
            # Initialize the save every N epochs input
            self.save_every_n_epochs = gr.Number(
                label="Save every N epochs",
                value=self.config.get("basic.save_every_n_epochs", 1),
                precision=0,
            )
            # Initialize the caption extension input
            self.caption_extension = gr.Dropdown(
                label="Caption file extension",
                choices=["", ".cap", ".caption", ".txt"],
                value=".txt",
                interactive=True,
            )

    def init_precision_and_resources_controls(self) -> None:
        """
        Initializes the precision and resources controls for the model.
        """
        with gr.Row():
            # Initialize the seed textbox
            self.seed = gr.Number(
                label="Seed",
                # precision=0,
                step=1,
                minimum=0,
                value=self.config.get("basic.seed", 0),
                info="Set to 0 to make random",
            )
            # Initialize the cache latents checkbox
            self.cache_latents = gr.Checkbox(
                label="Cache latents",
                value=self.config.get("basic.cache_latents", True),
            )
            # Initialize the cache latents to disk checkbox
            self.cache_latents_to_disk = gr.Checkbox(
                label="Cache latents to disk",
                value=self.config.get("basic.cache_latents_to_disk", False),
            )

    def init_lr_and_optimizer_controls(self) -> None:
        """
        Initializes the learning rate and optimizer controls for the model.
        """
        with gr.Row():
            # Initialize the learning rate scheduler dropdown
            self.lr_scheduler = gr.Dropdown(
                label="LR Scheduler",
                choices=[
                    "adafactor",
                    "constant",
                    "constant_with_warmup",
                    "cosine",
                    "cosine_with_restarts",
                    "linear",
                    "piecewise_constant",
                    "polynomial",
                ],
                value=self.config.get("basic.lr_scheduler", self.lr_scheduler_value),
            )
            
            # Initialize the learning rate scheduler type dropdown
            self.lr_scheduler_type = gr.Dropdown(
                label="LR Scheduler type",
                info="(Optional) custom scheduler module name",
                choices=[
                    "",
                    "CosineAnnealingLR",
                ],
                value=self.config.get("basic.lr_scheduler_type", ""),
                allow_custom_value=True,
            )
            
            # Initialize the optimizer dropdown
            self.optimizer = gr.Dropdown(
                label="Optimizer",
                choices=[
                    "AdamW",
                    "AdamW8bit",
                    "Adafactor",
                    "DAdaptation",
                    "DAdaptAdaGrad",
                    "DAdaptAdam",
                    "DAdaptAdan",
                    "DAdaptAdanIP",
                    "DAdaptAdamPreprint",
                    "DAdaptLion",
                    "DAdaptSGD",
                    "Lion",
                    "Lion8bit",
                    "PagedAdamW8bit",
                    "PagedAdamW32bit",
                    "PagedLion8bit",
                    "Prodigy",
                    "SGDNesterov",
                    "SGDNesterov8bit",
                ],
                value=self.config.get("basic.optimizer", "AdamW8bit"),
                interactive=True,
            )

    def init_grad_and_lr_controls(self) -> None:
        """
        Initializes the gradient and learning rate controls for the model.
        """
        with gr.Row():
            # Initialize the maximum gradient norm slider
            self.max_grad_norm = gr.Slider(
                label="Max grad norm",
                value=self.config.get("basic.max_grad_norm", 1.0),
                minimum=0.0,
                maximum=1.0,
                interactive=True,
            )
            # Initialize the learning rate scheduler extra arguments textbox
            self.lr_scheduler_args = gr.Textbox(
                label="LR scheduler extra arguments",
                lines=2,
                placeholder="(Optional) eg: milestones=[1,10,30,50] gamma=0.1",
                value=self.config.get("basic.lr_scheduler_args", ""),
            )
            # Initialize the optimizer extra arguments textbox
            self.optimizer_args = gr.Textbox(
                label="Optimizer extra arguments",
                lines=2,
                placeholder="(Optional) eg: relative_step=True scale_parameter=True warmup_init=True",
                value=self.config.get("basic.optimizer_args", ""),
            )

    def init_learning_rate_controls(self) -> None:
        """
        Initializes the learning rate controls for the model.
        """
        with gr.Row():
            # Adjust visibility based on training modes
            lr_label = (
                "Learning rate Unet"
                if self.finetuning or self.dreambooth
                else "Learning rate"
            )
            # Initialize the learning rate number input
            self.learning_rate = gr.Number(
                label=lr_label,
                value=self.config.get("basic.learning_rate", self.learning_rate_value),
                minimum=-1,
                maximum=1,
                info="Set to 0 to not train the Unet",
            )
            # Initialize the learning rate TE number input
            self.learning_rate_te = gr.Number(
                label="Learning rate TE",
                value=self.config.get(
                    "basic.learning_rate_te", self.learning_rate_value
                ),
                visible=self.finetuning or self.dreambooth,
                minimum=-1,
                maximum=1,
                info="Set to 0 to not train the Text Encoder",
            )
            # Initialize the learning rate TE1 number input
            self.learning_rate_te1 = gr.Number(
                label="Learning rate TE1",
                value=self.config.get(
                    "basic.learning_rate_te1", self.learning_rate_value
                ),
                visible=False,
                minimum=-1,
                maximum=1,
                info="Set to 0 to not train the Text Encoder 1",
            )
            # Initialize the learning rate TE2 number input
            self.learning_rate_te2 = gr.Number(
                label="Learning rate TE2",
                value=self.config.get(
                    "basic.learning_rate_te2", self.learning_rate_value
                ),
                visible=False,
                minimum=-1,
                maximum=1,
                info="Set to 0 to not train the Text Encoder 2",
            )
            # Initialize the learning rate warmup slider
            self.lr_warmup = gr.Slider(
                label="LR warmup (% of total steps)",
                value=self.config.get("basic.lr_warmup", self.lr_warmup_value),
                minimum=0,
                maximum=100,
                step=1,
            )
            
            def lr_scheduler_changed(scheduler, value):
                if scheduler == "constant":
                    self.old_lr_warmup = value
                    value = 0
                    interactive=False
                    info="Can't use LR warmup with LR Scheduler constant... setting to 0 and disabling field..."
                else:
                    if self.old_lr_warmup != 0:
                        value = self.old_lr_warmup
                        self.old_lr_warmup = 0
                    interactive=True
                    info=""
                return gr.Slider(value=value, interactive=interactive, info=info)
            
            self.lr_scheduler.change(
                lr_scheduler_changed,
                inputs=[self.lr_scheduler, self.lr_warmup],
                outputs=self.lr_warmup,
            )

    def init_scheduler_controls(self) -> None:
        """
        Initializes the scheduler controls for the model.
        """
        with gr.Row(visible=not self.finetuning):
            # Initialize the learning rate scheduler number of cycles textbox
            self.lr_scheduler_num_cycles = gr.Number(
                label="LR # cycles",
                minimum=1,
                # precision=0, # round to nearest integer
                step=1, # Increment value by 1
                info="Number of restarts for cosine scheduler with restarts",
                value=self.config.get("basic.lr_scheduler_num_cycles", 1),
            )
            # Initialize the learning rate scheduler power textbox
            self.lr_scheduler_power = gr.Number(
                label="LR power",
                minimum=0.0,
                step=0.01,
                info="Polynomial power for polynomial scheduler",
                value=self.config.get("basic.lr_scheduler_power", 1.0),
            )

    def init_resolution_and_bucket_controls(self) -> None:
        """
        Initializes the resolution and bucket controls for the model.
        """
        with gr.Row(visible=not self.finetuning):
            # Initialize the maximum resolution textbox
            self.max_resolution = gr.Textbox(
                label="Max resolution",
                value=self.config.get("basic.max_resolution", "512,512"),
                placeholder="512,512",
            )
            # Initialize the stop text encoder training slider
            self.stop_text_encoder_training = gr.Slider(
                minimum=-1,
                maximum=100,
                value=self.config.get("basic.stop_text_encoder_training", 0),
                step=1,
                label="Stop TE (% of total steps)",
            )
            # Initialize the enable buckets checkbox
            self.enable_bucket = gr.Checkbox(
                label="Enable buckets",
                value=self.config.get("basic.enable_bucket", True),
            )
            # Initialize the minimum bucket resolution slider
            self.min_bucket_reso = gr.Slider(
                label="Minimum bucket resolution",
                value=self.config.get("basic.min_bucket_reso", 256),
                minimum=64,
                maximum=4096,
                step=64,
                info="Minimum size in pixel a bucket can be (>= 64)",
            )
            # Initialize the maximum bucket resolution slider
            self.max_bucket_reso = gr.Slider(
                label="Maximum bucket resolution",
                value=self.config.get("basic.max_bucket_reso", 2048),
                minimum=64,
                maximum=4096,
                step=64,
                info="Maximum size in pixel a bucket can be (>= 64)",
            )

    def setup_sdxl_checkbox_behavior(self) -> None:
        """
        Sets up the behavior of the SDXL checkbox based on the finetuning and dreambooth flags.
        """
        self.sdxl_checkbox.change(
            self.update_learning_rate_te,
            inputs=[
                self.sdxl_checkbox,
                gr.Checkbox(value=self.finetuning, visible=False),
                gr.Checkbox(value=self.dreambooth, visible=False),
            ],
            outputs=[
                self.learning_rate_te,
                self.learning_rate_te1,
                self.learning_rate_te2,
            ],
        )

    def update_learning_rate_te(
        self,
        sdxl_checkbox: gr.Checkbox,
        finetuning: bool,
        dreambooth: bool,
    ) -> Tuple[gr.Number, gr.Number, gr.Number]:
        """
        Updates the visibility of the learning rate TE, TE1, and TE2 based on the SDXL checkbox and finetuning/dreambooth flags.

        Args:
            sdxl_checkbox (gr.Checkbox): The SDXL checkbox.
            finetuning (bool): Whether finetuning is enabled.
            dreambooth (bool): Whether dreambooth is enabled.

        Returns:
            Tuple[gr.Number, gr.Number, gr.Number]: A tuple containing the updated visibility for learning rate TE, TE1, and TE2.
        """
        # Determine the visibility condition based on finetuning and dreambooth flags
        visibility_condition = finetuning or dreambooth
        # Return a tuple of gr.Number instances with updated visibility
        return (
            gr.Number(visible=(not sdxl_checkbox and visibility_condition)),
            gr.Number(visible=(sdxl_checkbox and visibility_condition)),
            gr.Number(visible=(sdxl_checkbox and visibility_condition)),
        )
