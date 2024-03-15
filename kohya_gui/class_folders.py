import gradio as gr
import os
from .common_gui import get_folder_path, scriptdir, list_dirs


class Folders:
    def __init__(self, finetune=False, train_data_dir: gr.Dropdown = None, data_dir=None, output_dir=None, logging_dir=None, headless=False):
        from .common_gui import create_refresh_button

        self.headless = headless

        default_data_dir = data_dir if data_dir is not None else os.path.join(scriptdir, "data")
        default_output_dir = output_dir if output_dir is not None else os.path.join(scriptdir, "outputs")
        default_logging_dir = logging_dir if logging_dir is not None else os.path.join(scriptdir, "logs")
        default_reg_data_dir = default_data_dir

        self.current_data_dir = default_data_dir
        self.current_output_dir = default_output_dir
        self.current_logging_dir = default_logging_dir


        if default_data_dir is not None and default_data_dir.strip() != "" and not os.path.exists(default_data_dir):
            os.makedirs(default_data_dir, exist_ok=True)
        if default_output_dir is not None and default_output_dir.strip() != "" and not os.path.exists(default_output_dir):
            os.makedirs(default_output_dir, exist_ok=True)
        if default_logging_dir is not None and default_logging_dir.strip() != "" and not os.path.exists(default_logging_dir):
            os.makedirs(default_logging_dir, exist_ok=True)

        def list_data_dirs(path):
            self.current_data_dir = path
            return list(list_dirs(path))

        def list_output_dirs(path):
            self.current_output_dir = path
            return list(list_dirs(path))

        def list_logging_dirs(path):
            self.current_logging_dir = path
            return list(list_dirs(path))

        with gr.Row():
            self.output_dir = gr.Dropdown(
                label=f'Output folder to output trained model',
                choices=[""] + list_output_dirs(default_output_dir),
                value="",
                interactive=True,
                allow_custom_value=True,
            )
            create_refresh_button(self.output_dir, lambda: None, lambda: {"choices": list_output_dirs(self.current_output_dir)}, "open_folder_small")
            self.output_dir_folder = gr.Button(
                '📂', elem_id='open_folder_small', elem_classes=["tool"], visible=(not self.headless)
            )
            self.output_dir_folder.click(
                get_folder_path,
                outputs=self.output_dir,
                show_progress=False,
            )

            self.reg_data_dir = gr.Dropdown(
                label='Regularisation folder (Optional. containing reqularization images)' if not finetune else 'Train config folder (Optional. where config files will be saved)',
                choices=[""] + list_data_dirs(default_reg_data_dir),
                value="",
                interactive=True,
                allow_custom_value=True,
            )
            create_refresh_button(self.reg_data_dir, lambda: None, lambda: {"choices": list_data_dirs(self.current_data_dir)}, "open_folder_small")
            self.reg_data_dir_folder = gr.Button(
                '📂', elem_id='open_folder_small', elem_classes=["tool"], visible=(not self.headless)
            )
            self.reg_data_dir_folder.click(
                get_folder_path,
                outputs=self.reg_data_dir,
                show_progress=False,
            )
        with gr.Row():
            self.logging_dir = gr.Dropdown(
                label='Logging folder (Optional. to enable logging and output Tensorboard log)',
                choices=[""] + list_logging_dirs(default_logging_dir),
                value="",
                interactive=True,
                allow_custom_value=True,
            )
            create_refresh_button(self.logging_dir, lambda: None, lambda: {"choices": list_logging_dirs(self.current_logging_dir)}, "open_folder_small")
            self.logging_dir_folder = gr.Button(
                '📂', elem_id='open_folder_small', elem_classes=["tool"], visible=(not self.headless)
            )
            self.logging_dir_folder.click(
                get_folder_path,
                outputs=self.logging_dir,
                show_progress=False,
            )

            self.output_dir.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_output_dirs(path)),
                inputs=self.output_dir,
                outputs=self.output_dir,
                show_progress=False,
            )
            self.reg_data_dir.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_data_dirs(path)),
                inputs=self.reg_data_dir,
                outputs=self.reg_data_dir,
                show_progress=False,
            )
            self.logging_dir.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_logging_dirs(path)),
                inputs=self.logging_dir,
                outputs=self.logging_dir,
                show_progress=False,
            )
