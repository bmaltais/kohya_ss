import gradio as gr

### Stable Cascade Parameters class
class StableCascadeParameters:
    def __init__(
        self, stable_cascade_checkbox, show_stable_cascade_cache_text_encoder_outputs: bool = True
    ):
        self.stable_cascade_checkbox = stable_cascade_checkbox
        self.show_stable_cascade_cache_text_encoder_outputs = (
            show_stable_cascade_cache_text_encoder_outputs
        )

        with gr.Accordion(
            visible=False, open=True, label='Stable Cascade Specific Parameters'
        ) as self.stable_cascade_row:
            with gr.Row():
                self.effnet_checkpoint_path = gr.Textbox(
                    label="effnet checkpoint path"
                )
                self.stable_cascade_cache_text_encoder_outputs = gr.Checkbox(
                    label='Cache text encoder outputs',
                    info='Cache the outputs of the text encoders. This option is useful to reduce the GPU memory usage. This option cannot be used with options for shuffling or dropping the captions.',
                    value=False,
                    visible=show_stable_cascade_cache_text_encoder_outputs,
                )
                self.stable_cascade_no_half_vae = gr.Checkbox(
                    label='No half VAE',
                    info='Disable the half-precision (mixed-precision) VAE. VAE for Stable Cascade seems to produce NaNs in some cases. This option is useful to avoid the NaNs.',
                    value=True,
                )

        self.stable_cascade_checkbox.change(
            lambda stable_cascade_checkbox: gr.Accordion(visible=stable_cascade_checkbox),
            inputs=[self.stable_cascade_checkbox],
            outputs=[self.stable_cascade_row],
        )
