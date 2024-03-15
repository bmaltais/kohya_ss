import gradio as gr
import os

from .common_gui import (
    get_any_file_path,
    get_folder_path,
    set_pretrained_model_name_or_path_input,
    scriptdir,
    list_dirs,
    list_files,
)

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄

default_models = [
    'stabilityai/stable-diffusion-xl-base-1.0',
    'stabilityai/stable-diffusion-xl-refiner-1.0',
    'stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned',
    'stabilityai/stable-diffusion-2-1-base',
    'stabilityai/stable-diffusion-2-base',
    'stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned',
    'stabilityai/stable-diffusion-2-1',
    'stabilityai/stable-diffusion-2',
    'runwayml/stable-diffusion-v1-5',
    'CompVis/stable-diffusion-v1-4',
]

class SourceModel:
    def __init__(
        self,
        save_model_as_choices=[
            'same as source model',
            'ckpt',
            'diffusers',
            'diffusers_safetensors',
            'safetensors',
        ],
        save_precision_choices=[
            "float",
            "fp16",
            "bf16",
        ],
        headless=False,
        default_data_dir=None,
        finetuning=False,
    ):
        self.headless = headless
        self.save_model_as_choices = save_model_as_choices
        self.finetuning = finetuning

        from .common_gui import create_refresh_button

        default_data_dir = default_data_dir if default_data_dir is not None else os.path.join(scriptdir, "outputs")
        default_train_dir = default_data_dir if default_data_dir is not None else os.path.join(scriptdir, "data")
        model_checkpoints = list(list_files(default_data_dir, exts=[".ckpt", ".safetensors"], all=True))
        self.current_data_dir = default_data_dir
        self.current_train_dir = default_train_dir

        def list_models(path):
            self.current_data_dir = path if os.path.isdir(path) else os.path.dirname(path)
            return default_models + list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

        def list_train_dirs(path):
            self.current_train_dir = path if os.path.isdir(path) else os.path.dirname(path)
            return list(list_dirs(path))

        if default_data_dir is not None and default_data_dir.strip() != "" and not os.path.exists(default_data_dir):
            os.makedirs(default_data_dir, exist_ok=True)

        with gr.Column(), gr.Group():
            # Define the input elements
            with gr.Row():
              with gr.Column(), gr.Row():
                self.model_list = gr.Textbox(visible=False, value="")
                self.pretrained_model_name_or_path = gr.Dropdown(
                    label='Pretrained model name or path',
                    choices=default_models + model_checkpoints,
                    value='runwayml/stable-diffusion-v1-5',
                    allow_custom_value=True,
                    visible=True,
                    min_width=100,
                )
                create_refresh_button(self.pretrained_model_name_or_path, lambda: None, lambda: {"choices": list_models(self.current_data_dir)},"open_folder_small")

                self.pretrained_model_name_or_path_file = gr.Button(
                    document_symbol,
                    elem_id='open_folder_small',
                    elem_classes=['tool'],
                    visible=(not headless),
                )
                self.pretrained_model_name_or_path_file.click(
                    get_any_file_path,
                    inputs=self.pretrained_model_name_or_path,
                    outputs=self.pretrained_model_name_or_path,
                    show_progress=False,
                )
                self.pretrained_model_name_or_path_folder = gr.Button(
                    folder_symbol,
                    elem_id='open_folder_small',
                    elem_classes=['tool'],
                    visible=(not headless),
                )
                self.pretrained_model_name_or_path_folder.click(
                    get_folder_path,
                    inputs=self.pretrained_model_name_or_path,
                    outputs=self.pretrained_model_name_or_path,
                    show_progress=False,
                )

              with gr.Column(), gr.Row():
                self.train_data_dir = gr.Dropdown(
                    label='Image folder (containing training images subfolders)' if not finetuning else 'Image folder (containing training images)',
                    choices=[""] + list_train_dirs(default_train_dir),
                    value="",
                    interactive=True,
                    allow_custom_value=True,
                )
                create_refresh_button(self.train_data_dir, lambda: None, lambda: {"choices": list_train_dirs(self.current_train_dir)}, "open_folder_small")
                self.train_data_dir_folder = gr.Button(
                    '📂', elem_id='open_folder_small', elem_classes=["tool"], visible=(not self.headless)
                )
                self.train_data_dir_folder.click(
                    get_folder_path,
                    outputs=self.train_data_dir,
                    show_progress=False,
                )

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        self.v2 = gr.Checkbox(label='v2', value=False, visible=False, min_width=60)
                        self.v_parameterization = gr.Checkbox(
                            label='v_parameterization', value=False, visible=False, min_width=130,
                        )
                        self.sdxl_checkbox = gr.Checkbox(
                            label='SDXL', value=False, visible=False, min_width=60,
                        )
                with gr.Column():
                    gr.Box(visible=False)

            with gr.Row():
                self.output_name = gr.Textbox(
                    label='Trained Model output name',
                    placeholder='(Name of the model to output)',
                    value='last',
                    interactive=True,
                )
                self.training_comment = gr.Textbox(
                    label='Training comment',
                    placeholder='(Optional) Add training comment to be included in metadata',
                    interactive=True,
                )

            with gr.Row():
                self.save_model_as = gr.Radio(
                    save_model_as_choices,
                    label="Save trained model as",
                    value="safetensors",
                )
                self.save_precision = gr.Radio(
                    save_precision_choices,
                    label="Save precision",
                    value="fp16",
                )

            self.pretrained_model_name_or_path.change(
                fn=lambda path: set_pretrained_model_name_or_path_input(path, refresh_method=list_models),
                inputs=[
                    self.pretrained_model_name_or_path,
                ],
                outputs=[
                    self.pretrained_model_name_or_path,
                    self.v2,
                    self.v_parameterization,
                    self.sdxl_checkbox,
                ],
                show_progress=False,
            )

            self.train_data_dir.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_train_dirs(path)),
                inputs=self.train_data_dir,
                outputs=self.train_data_dir,
                show_progress=False,
            )
