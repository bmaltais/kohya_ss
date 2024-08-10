import gradio as gr
import os

from .common_gui import (
    get_file_path,
    get_folder_path,
    set_pretrained_model_name_or_path_input,
    scriptdir,
    list_dirs,
    list_files,
    create_refresh_button,
)
from .class_gui_config import KohyaSSGUIConfig

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

default_models = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned",
    "stabilityai/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-2-base",
    "stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2",
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
]


class SourceModel:
    def __init__(
        self,
        save_model_as_choices=[
            "same as source model",
            "ckpt",
            "diffusers",
            "diffusers_safetensors",
            "safetensors",
        ],
        save_precision_choices=[
            "float",
            "fp16",
            "bf16",
        ],
        headless=False,
        finetuning=False,
        config: KohyaSSGUIConfig = {},
    ):
        self.headless = headless
        self.save_model_as_choices = save_model_as_choices
        self.finetuning = finetuning
        self.config = config

        # Set default directories if not provided
        self.current_models_dir = self.config.get(
            "model.models_dir", os.path.join(scriptdir, "models")
        )
        self.current_train_data_dir = self.config.get(
            "model.train_data_dir", os.path.join(scriptdir, "data")
        )
        self.current_dataset_config_dir = self.config.get(
            "model.dataset_config", os.path.join(scriptdir, "dataset_config")
        )

        model_checkpoints = list(
            list_files(
                self.current_models_dir, exts=[".ckpt", ".safetensors"], all=True
            )
        )

        def list_models(path):
            self.current_models_dir = (
                path if os.path.isdir(path) else os.path.dirname(path)
            )
            return default_models + list(
                list_files(path, exts=[".ckpt", ".safetensors"], all=True)
            )

        def list_train_data_dirs(path):
            self.current_train_data_dir = path if not path == "" else "."
            return list(list_dirs(self.current_train_data_dir))

        def list_dataset_config_dirs(path: str) -> list:
            """
            List directories and toml files in the dataset_config directory.

            Parameters:
            - path (str): The path to list directories and files from.

            Returns:
            - list: A list of directories and files.
            """
            current_dataset_config_dir = path if not path == "" else "."
            # Lists all .json files in the current configuration directory, used for populating dropdown choices.
            return list(
                list_files(current_dataset_config_dir, exts=[".toml"], all=True)
            )

        with gr.Accordion("Model", open=True):
            with gr.Column(), gr.Group():
                model_ext = gr.Textbox(value="*.safetensors *.ckpt", visible=False)
                model_ext_name = gr.Textbox(value="Model types", visible=False)

                # Define the input elements
                with gr.Row():
                    with gr.Column(), gr.Row():
                        self.model_list = gr.Textbox(visible=False, value="")
                        self.pretrained_model_name_or_path = gr.Dropdown(
                            label="Pretrained model name or path",
                            choices=default_models + model_checkpoints,
                            value=self.config.get("model.models_dir", "runwayml/stable-diffusion-v1-5"),
                            allow_custom_value=True,
                            visible=True,
                            min_width=100,
                        )
                        create_refresh_button(
                            self.pretrained_model_name_or_path,
                            lambda: None,
                            lambda: {"choices": list_models(self.current_models_dir)},
                            "open_folder_small",
                        )

                        self.pretrained_model_name_or_path_file = gr.Button(
                            document_symbol,
                            elem_id="open_folder_small",
                            elem_classes=["tool"],
                            visible=(not headless),
                        )
                        self.pretrained_model_name_or_path_file.click(
                            get_file_path,
                            inputs=[self.pretrained_model_name_or_path, model_ext, model_ext_name],
                            outputs=self.pretrained_model_name_or_path,
                            show_progress=False,
                        )
                        self.pretrained_model_name_or_path_folder = gr.Button(
                            folder_symbol,
                            elem_id="open_folder_small",
                            elem_classes=["tool"],
                            visible=(not headless),
                        )
                        self.pretrained_model_name_or_path_folder.click(
                            get_folder_path,
                            inputs=self.pretrained_model_name_or_path,
                            outputs=self.pretrained_model_name_or_path,
                            show_progress=False,
                        )

                    with gr.Column(), gr.Row():
                        self.output_name = gr.Textbox(
                            label="Trained Model output name",
                            placeholder="(Name of the model to output)",
                            value=self.config.get("model.output_name", "last"),
                            interactive=True,
                        )
                with gr.Row():
                    with gr.Column(), gr.Row():
                        self.train_data_dir = gr.Dropdown(
                            label=(
                                "Image folder (containing training images subfolders)"
                                if not finetuning
                                else "Image folder (containing training images)"
                            ),
                            choices=[""]
                            + list_train_data_dirs(self.current_train_data_dir),
                            value=self.config.get("model.train_data_dir", ""),
                            interactive=True,
                            allow_custom_value=True,
                        )
                        create_refresh_button(
                            self.train_data_dir,
                            lambda: None,
                            lambda: {
                                "choices": [""]
                                + list_train_data_dirs(self.current_train_data_dir)
                            },
                            "open_folder_small",
                        )
                        self.train_data_dir_folder = gr.Button(
                            "📂",
                            elem_id="open_folder_small",
                            elem_classes=["tool"],
                            visible=(not self.headless),
                        )
                        self.train_data_dir_folder.click(
                            get_folder_path,
                            outputs=self.train_data_dir,
                            show_progress=False,
                        )
                    with gr.Column(), gr.Row():
                        # Toml directory dropdown
                        self.dataset_config = gr.Dropdown(
                            label="Dataset config file (Optional. Select the toml configuration file to use for the dataset)",
                            choices=[self.config.get("model.dataset_config", "")]
                            + list_dataset_config_dirs(self.current_dataset_config_dir),
                            value=self.config.get("model.dataset_config", ""),
                            interactive=True,
                            allow_custom_value=True,
                        )
                        # Refresh button for dataset_config directory
                        create_refresh_button(
                            self.dataset_config,
                            lambda: None,
                            lambda: {
                                "choices": [""]
                                + list_dataset_config_dirs(
                                    self.current_dataset_config_dir
                                )
                            },
                            "open_folder_small",
                        )
                        # Toml directory button
                        self.dataset_config_folder = gr.Button(
                            document_symbol,
                            elem_id="open_folder_small",
                            elem_classes=["tool"],
                            visible=(not self.headless),
                        )

                        # Toml directory button click event
                        self.dataset_config_folder.click(
                            get_file_path,
                            inputs=[
                                self.dataset_config,
                                gr.Textbox(value="*.toml", visible=False),
                                gr.Textbox(value="Dataset config types", visible=False),
                            ],
                            outputs=self.dataset_config,
                            show_progress=False,
                        )
                        # Change event for dataset_config directory dropdown
                        self.dataset_config.change(
                            fn=lambda path: gr.Dropdown(
                                choices=[""] + list_dataset_config_dirs(path)
                            ),
                            inputs=self.dataset_config,
                            outputs=self.dataset_config,
                            show_progress=False,
                        )

                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            self.v2 = gr.Checkbox(
                                label="v2", value=False, visible=False, min_width=60,
                                interactive=True,
                            )
                            self.v_parameterization = gr.Checkbox(
                                label="v_parameterization",
                                value=False,
                                visible=False,
                                min_width=130,
                                interactive=True,
                            )
                            self.sdxl_checkbox = gr.Checkbox(
                                label="SDXL",
                                value=False,
                                visible=False,
                                min_width=60,
                                interactive=True,
                            )
                            self.sd3_checkbox = gr.Checkbox(
                                label="SD3",
                                value=False,
                                visible=False,
                                min_width=60,
                                interactive=True,
                            )
                            self.flux1_checkbox = gr.Checkbox(
                                label="Flux.1",
                                value=False,
                                visible=False,
                                min_width=60,
                                interactive=True,
                            )

                            def toggle_checkboxes(v2, v_parameterization, sdxl_checkbox, sd3_checkbox, flux1_checkbox):
                                # Check if all checkboxes are unchecked
                                if not v2 and not v_parameterization and not sdxl_checkbox and not sd3_checkbox and not flux1_checkbox:
                                    # If all unchecked, return new interactive checkboxes
                                    return (
                                        gr.Checkbox(interactive=True),  # v2 checkbox
                                        gr.Checkbox(interactive=True),  # v_parameterization checkbox
                                        gr.Checkbox(interactive=True),  # sdxl_checkbox
                                        gr.Checkbox(interactive=True),  # sd3_checkbox
                                        gr.Checkbox(interactive=True),  # sd3_checkbox
                                    )
                                else:
                                    # If any checkbox is checked, return checkboxes with current interactive state
                                    return (
                                        gr.Checkbox(interactive=v2),  # v2 checkbox
                                        gr.Checkbox(interactive=v_parameterization),  # v_parameterization checkbox
                                        gr.Checkbox(interactive=sdxl_checkbox),  # sdxl_checkbox
                                        gr.Checkbox(interactive=sd3_checkbox),  # sd3_checkbox
                                        gr.Checkbox(interactive=flux1_checkbox),  # flux1_checkbox
                                    )

                            self.v2.change(
                                fn=toggle_checkboxes,
                                inputs=[self.v2, self.v_parameterization, self.sdxl_checkbox, self.sd3_checkbox, self.flux1_checkbox],
                                outputs=[self.v2, self.v_parameterization, self.sdxl_checkbox, self.sd3_checkbox, self.flux1_checkbox],
                                show_progress=False,
                            )
                            self.v_parameterization.change(
                                fn=toggle_checkboxes,
                                inputs=[self.v2, self.v_parameterization, self.sdxl_checkbox, self.sd3_checkbox, self.flux1_checkbox],
                                outputs=[self.v2, self.v_parameterization, self.sdxl_checkbox, self.sd3_checkbox, self.flux1_checkbox],
                                show_progress=False,
                            )
                            self.sdxl_checkbox.change(
                                fn=toggle_checkboxes,
                                inputs=[self.v2, self.v_parameterization, self.sdxl_checkbox, self.sd3_checkbox, self.flux1_checkbox],
                                outputs=[self.v2, self.v_parameterization, self.sdxl_checkbox, self.sd3_checkbox, self.flux1_checkbox],
                                show_progress=False,
                            )
                            self.sd3_checkbox.change(
                                fn=toggle_checkboxes,
                                inputs=[self.v2, self.v_parameterization, self.sdxl_checkbox, self.sd3_checkbox, self.flux1_checkbox],
                                outputs=[self.v2, self.v_parameterization, self.sdxl_checkbox, self.sd3_checkbox, self.flux1_checkbox],
                                show_progress=False,
                            )
                            self.flux1_checkbox.change(
                                fn=toggle_checkboxes,
                                inputs=[self.v2, self.v_parameterization, self.sdxl_checkbox, self.sd3_checkbox, self.flux1_checkbox],
                                outputs=[self.v2, self.v_parameterization, self.sdxl_checkbox, self.sd3_checkbox, self.flux1_checkbox],
                                show_progress=False,
                            )
                    with gr.Column():
                        gr.Group(visible=False)

                with gr.Row():
                    self.training_comment = gr.Textbox(
                        label="Training comment",
                        placeholder="(Optional) Add training comment to be included in metadata",
                        interactive=True,
                        value=self.config.get("model.training_comment", ""),
                    )

                with gr.Row():
                    self.save_model_as = gr.Radio(
                        save_model_as_choices,
                        label="Save trained model as",
                        value=self.config.get("model.save_model_as", "safetensors"),
                    )
                    self.save_precision = gr.Radio(
                        save_precision_choices,
                        label="Save precision",
                        value=self.config.get("model.save_precision", "fp16"),
                    )

                self.pretrained_model_name_or_path.change(
                    fn=lambda path: set_pretrained_model_name_or_path_input(
                        path, refresh_method=list_models
                    ),
                    inputs=[
                        self.pretrained_model_name_or_path,
                    ],
                    outputs=[
                        self.pretrained_model_name_or_path,
                        self.v2,
                        self.v_parameterization,
                        self.sdxl_checkbox,
                        self.sd3_checkbox,
                        self.flux1_checkbox,
                    ],
                    show_progress=False,
                )

                self.train_data_dir.change(
                    fn=lambda path: gr.Dropdown(
                        choices=[""] + list_train_data_dirs(path)
                    ),
                    inputs=self.train_data_dir,
                    outputs=self.train_data_dir,
                    show_progress=False,
                )
