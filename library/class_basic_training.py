import gradio as gr
import os

class BasicTraining:
    def __init__(
        self,
        learning_rate_value='1e-6',
        lr_scheduler_value='constant',
        lr_warmup_value='0',
        finetuning: bool = False,
    ):
        self.learning_rate_value = learning_rate_value
        self.lr_scheduler_value = lr_scheduler_value
        self.lr_warmup_value = lr_warmup_value
        self.finetuning = finetuning

        with gr.Row():
            self.train_batch_size = gr.Slider(
                minimum=1,
                maximum=64,
                label='Train batch size',
                value=1,
                step=1,
            )
            self.epoch = gr.Number(label='Epoch', value=1, precision=0)
            self.max_train_epochs = gr.Textbox(
                label='Max train epoch',
                placeholder='(Optional) Enforce number of epoch',
            )
            self.save_every_n_epochs = gr.Number(
                label='Save every N epochs', value=1, precision=0
            )
            self.caption_extension = gr.Textbox(
                label='Caption Extension',
                placeholder='(Optional) Extension for caption files. default: .caption',
            )
        with gr.Row():
            self.mixed_precision = gr.Dropdown(
                label='Mixed precision',
                choices=[
                    'no',
                    'fp16',
                    'bf16',
                ],
                value='fp16',
            )
            self.save_precision = gr.Dropdown(
                label='Save precision',
                choices=[
                    'float',
                    'fp16',
                    'bf16',
                ],
                value='fp16',
            )
            self.num_cpu_threads_per_process = gr.Slider(
                minimum=1,
                maximum=os.cpu_count(),
                step=1,
                label='Number of CPU threads per core',
                value=2,
            )
            self.seed = gr.Textbox(
                label='Seed', placeholder='(Optional) eg:1234'
            )
            self.cache_latents = gr.Checkbox(label='Cache latents', value=True)
            self.cache_latents_to_disk = gr.Checkbox(
                label='Cache latents to disk', value=False
            )
        with gr.Row():
            self.learning_rate = gr.Number(
                label='Learning rate', value=learning_rate_value
            )
            self.lr_scheduler = gr.Dropdown(
                label='LR Scheduler',
                choices=[
                    'adafactor',
                    'constant',
                    'constant_with_warmup',
                    'cosine',
                    'cosine_with_restarts',
                    'linear',
                    'polynomial',
                ],
                value=lr_scheduler_value,
            )
            self.lr_warmup = gr.Slider(
                label='LR warmup (% of steps)',
                value=lr_warmup_value,
                minimum=0,
                maximum=100,
                step=1,
            )
            self.optimizer = gr.Dropdown(
                label='Optimizer',
                choices=[
                    'AdamW',
                    'AdamW8bit',
                    'Adafactor',
                    'DAdaptation',
                    'DAdaptAdaGrad',
                    'DAdaptAdam',
                    'DAdaptAdan',
                    'DAdaptAdanIP',
                    'DAdaptAdamPreprint',
                    'DAdaptLion',
                    'DAdaptSGD',
                    'Lion',
                    'Lion8bit',
                    "PagedAdamW8bit",
                    "PagedLion8bit",
                    'Prodigy',
                    'SGDNesterov',
                    'SGDNesterov8bit',
                ],
                value='AdamW8bit',
                interactive=True,
            )
        with gr.Row(visible=not finetuning):
            self.lr_scheduler_num_cycles = gr.Textbox(
                label='LR number of cycles',
                placeholder='(Optional) For Cosine with restart and polynomial only',
            )

            self.lr_scheduler_power = gr.Textbox(
                label='LR power',
                placeholder='(Optional) For Cosine with restart and polynomial only',
            )
        with gr.Row():
            self.optimizer_args = gr.Textbox(
                label='Optimizer extra arguments',
                placeholder='(Optional) eg: relative_step=True scale_parameter=True warmup_init=True',
            )
        with gr.Row(visible=not finetuning):
            self.max_resolution = gr.Textbox(
                label='Max resolution',
                value='512,512',
                placeholder='512,512',
            )
            self.stop_text_encoder_training = gr.Slider(
                minimum=-1,
                maximum=100,
                value=0,
                step=1,
                label='Stop text encoder training',
            )
        with gr.Row(visible=not finetuning):
            self.enable_bucket = gr.Checkbox(label='Enable buckets', value=True)
            self.min_bucket_reso = gr.Slider(label='Minimum bucket resolution', value=256, minimum=64, maximum=4096, step=64, info='Minimum size in pixel a bucket can be (>= 64)')
            self.max_bucket_reso = gr.Slider(label='Maximum bucket resolution', value=2048, minimum=64, maximum=4096, step=64, info='Maximum size in pixel a bucket can be (>= 64)')
            