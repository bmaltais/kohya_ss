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


class sd3Training:
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
        sd3_checkbox: gr.Checkbox = False,
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
        self.sd3_checkbox = sd3_checkbox

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

        with gr.Accordion(
            "SD3", open=False, elem_id="sd3_tab", visible=False
        ) as sd3_accordion:
            with gr.Group():
                gr.Markdown("### SD3 Specific Parameters")
                with gr.Row():
                    self.weighting_scheme = gr.Dropdown(
                        label="Weighting Scheme",
                        choices=["logit_normal", "sigma_sqrt", "mode", "cosmap"],
                        value=self.config.get("sd3.weighting_scheme", "logit_normal"),
                        interactive=True,
                    )
                    self.logit_mean = gr.Number(
                        label="Logit Mean",
                        value=self.config.get("sd3.logit_mean", 0.0),
                        interactive=True,
                    )
                    self.logit_std = gr.Number(
                        label="Logit Std",
                        value=self.config.get("sd3.logit_std", 1.0),
                        interactive=True,
                    )
                    self.mode_scale = gr.Number(
                        label="Mode Scale",
                        value=self.config.get("sd3.mode_scale", 1.29),
                        interactive=True,
                    )

                with gr.Row():
                    self.clip_l = gr.Textbox(
                        label="CLIP-L Path",
                        placeholder="Path to CLIP-L model",
                        value=self.config.get("sd3.clip_l", ""),
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

                    self.clip_g = gr.Textbox(
                        label="CLIP-G Path",
                        placeholder="Path to CLIP-G model",
                        value=self.config.get("sd3.clip_g", ""),
                        interactive=True,
                    )
                    self.clip_g_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.clip_g_button.click(
                        get_any_file_path,
                        outputs=self.clip_g,
                        show_progress=False,
                    )

                    self.t5xxl = gr.Textbox(
                        label="T5-XXL Path",
                        placeholder="Path to T5-XXL model",
                        value=self.config.get("sd3.t5xxl", ""),
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
                    self.save_clip = gr.Checkbox(
                        label="Save CLIP models",
                        value=self.config.get("sd3.save_clip", False),
                        interactive=True,
                    )
                    self.save_t5xxl = gr.Checkbox(
                        label="Save T5-XXL model",
                        value=self.config.get("sd3.save_t5xxl", False),
                        interactive=True,
                    )

                with gr.Row():
                    self.t5xxl_device = gr.Textbox(
                        label="T5-XXL Device",
                        placeholder="Device for T5-XXL (e.g., cuda:0)",
                        value=self.config.get("sd3.t5xxl_device", ""),
                        interactive=True,
                    )
                    self.t5xxl_dtype = gr.Dropdown(
                        label="T5-XXL Dtype",
                        choices=["float32", "fp16", "bf16"],
                        value=self.config.get("sd3.t5xxl_dtype", "bf16"),
                        interactive=True,
                    )
                    self.sd3_text_encoder_batch_size = gr.Number(
                        label="Text Encoder Batch Size",
                        value=self.config.get("sd3.text_encoder_batch_size", 1),
                        minimum=1,
                        maximum=1024,
                        step=1,
                        interactive=True,
                    )
                    self.sd3_cache_text_encoder_outputs = gr.Checkbox(
                        label="Cache Text Encoder Outputs",
                        value=self.config.get("sd3.cache_text_encoder_outputs", False),
                        info="Cache text encoder outputs to speed up inference",
                        interactive=True,
                    )
                    self.sd3_cache_text_encoder_outputs_to_disk = gr.Checkbox(
                        label="Cache Text Encoder Outputs to Disk",
                        value=self.config.get(
                            "sd3.cache_text_encoder_outputs_to_disk", False
                        ),
                        info="Cache text encoder outputs to disk to speed up inference",
                        interactive=True,
                    )

                self.sd3_checkbox.change(
                    lambda sd3_checkbox: gr.Accordion(visible=sd3_checkbox),
                    inputs=[self.sd3_checkbox],
                    outputs=[sd3_accordion],
                )
