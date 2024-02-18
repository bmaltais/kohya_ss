import gradio as gr

### Stable Cascade Parameters class
class StableCascadeParameters:
    def __init__(
        self, show_stable_cascade_cache_text_encoder_outputs: bool = True
    ):
        self.show_stable_cascade_cache_text_encoder_outputs = (
            show_stable_cascade_cache_text_encoder_outputs
        )

        with gr.Accordion(
            visible=True, open=True, label='Stable Cascade Specific Parameters'
        ):
            with gr.Row():
                self.effnet_checkpoint_path = gr.Textbox(
                    label="effnet checkpoint path"
                )
