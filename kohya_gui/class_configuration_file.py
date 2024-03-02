import gradio as gr


class ConfigurationFile:
    def __init__(self, headless=False):
        self.headless = headless
        with gr.Accordion('Configuration file', open=False):
            with gr.Row():
                self.button_open_config = gr.Button(
                    'Open 📂',
                    elem_id='open_folder',
                    visible=(not self.headless),
                )
                self.button_save_config = gr.Button(
                    'Save 💾',
                    elem_id='open_folder',
                )
                self.button_save_as_config = gr.Button(
                    'Save as... 💾',
                    elem_id='open_folder',
                    visible=(not self.headless),
                )
                self.config_file_name = gr.Textbox(
                    label='',
                    placeholder="type the configuration file path or use the 'Open' button above to select it...",
                    interactive=True,
                )
                self.button_load_config = gr.Button(
                    'Load 💾', elem_id='open_folder'
                )
