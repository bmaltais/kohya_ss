import gradio as gr
import os

from .common_gui import list_files


class ConfigurationFile:
    def __init__(self, headless=False, output_dir: gr.Dropdown = None):
        from .common_gui import create_refresh_button

        self.headless = headless
        self.output_dir = None

        def update_configs(output_dir):
            self.output_dir = output_dir
            return gr.Dropdown(choices=[""] + list(list_files(output_dir, exts=[".json"], all=True)))

        def list_configs(path):
            self.output_dir = path
            return list(list_files(path, exts=[".json"], all=True))

        with gr.Group():
            with gr.Row():
                self.config_file_name = gr.Dropdown(
                    label='Load/Save Config file',
                    choices=[""] + list_configs(self.output_dir),
                    value="",
                    interactive=True,
                    allow_custom_value=True,
                )
                create_refresh_button(self.config_file_name, lambda: None, lambda: {"choices": list_configs(self.output_dir)}, "open_folder_small")
                self.button_open_config = gr.Button(
                    '📂',
                    elem_id='open_folder_small',
                    elem_classes=['tool'],
                    visible=(not self.headless),
                )
                self.button_save_config = gr.Button(
                    '💾',
                    elem_id='open_folder_small',
                    elem_classes=['tool'],
                )
                self.button_load_config = gr.Button(
                    '↩️ ',
                    elem_id='open_folder_small',
                    elem_classes=['tool'],
                )

            self.config_file_name.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_configs(path)),
                inputs=self.config_file_name,
                outputs=self.config_file_name,
                show_progress=False,
            )

            output_dir.change(
                fn=update_configs,
                inputs=output_dir,
                outputs=self.config_file_name,
            )
