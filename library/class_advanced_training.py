import gradio as gr
from .common_gui import get_folder_path, get_any_file_path

class AdvancedTraining:
    def __init__(
        self,
        headless=False,
        finetuning: bool = False
    ):
        self.headless = headless
        self.finetuning = finetuning
        def noise_offset_type_change(noise_offset_type):
            if noise_offset_type == 'Original':
                return (gr.Group.update(visible=True), gr.Group.update(visible=False))
            else:
                return (gr.Group.update(visible=False), gr.Group.update(visible=True))

        with gr.Row(visible=not finetuning):
            self.no_token_padding = gr.Checkbox(
                label='No token padding', value=False
            )
            self.gradient_accumulation_steps = gr.Slider(
                label='Gradient accumulate steps', 
                info='Number of updates steps to accumulate before performing a backward/update pass',
                value='1',
                minimum=1, maximum=120,
                step=1
            )
            self.weighted_captions = gr.Checkbox(
                label='Weighted captions', value=False
            )
        with gr.Row(visible=not finetuning):
            self.prior_loss_weight = gr.Number(
                label='Prior loss weight', value=1.0
            )
            self.vae = gr.Textbox(
                label='VAE',
                placeholder='(Optional) path to checkpoint of vae to replace for training',
            )
            self.vae_button = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            self.vae_button.click(
                get_any_file_path,
                outputs=self.vae,
                show_progress=False,
            )

        with gr.Row():
            self.additional_parameters = gr.Textbox(
                label='Additional parameters',
                placeholder='(Optional) Use to provide additional parameters not handled by the GUI. Eg: --some_parameters "value"',
            )
        with gr.Row():
            self.save_every_n_steps = gr.Number(
                label='Save every N steps',
                value=0,
                precision=0,
                info='(Optional) The model is saved every specified steps',
            )
            self.save_last_n_steps = gr.Number(
                label='Save last N steps',
                value=0,
                precision=0,
                info='(Optional) Save only the specified number of models (old models will be deleted)',
            )
            self.save_last_n_steps_state = gr.Number(
                label='Save last N steps state',
                value=0,
                precision=0,
                info='(Optional) Save only the specified number of states (old models will be deleted)',
            )
        with gr.Row():
            def full_options_update(full_fp16, full_bf16):
                full_fp16_active = True
                full_bf16_active = True
                
                if full_fp16:
                    full_bf16_active = False
                if full_bf16:
                    full_fp16_active = False
                return gr.Checkbox.update(interactive=full_fp16_active, ), gr.Checkbox.update(interactive=full_bf16_active)
            
            self.keep_tokens = gr.Slider(
                label='Keep n tokens', value='0', minimum=0, maximum=32, step=1
            )
            self.clip_skip = gr.Slider(
                label='Clip skip', value='1', minimum=1, maximum=12, step=1
            )
            self.max_token_length = gr.Dropdown(
                label='Max Token Length',
                choices=[
                    '75',
                    '150',
                    '225',
                ],
                value='75',
            )
            self.full_fp16 = gr.Checkbox(
                label='Full fp16 training (experimental)', value=False,
            )
            self.full_bf16 = gr.Checkbox(
                label='Full bf16 training (experimental)', value=False, info='Required bitsandbytes >= 0.36.0'
            )
            self.full_fp16.change(full_options_update, inputs=[self.full_fp16, self.full_bf16], outputs=[self.full_fp16, self.full_bf16])
            self.full_bf16.change(full_options_update, inputs=[self.full_fp16, self.full_bf16], outputs=[self.full_fp16, self.full_bf16])
            
        with gr.Row():
            self.gradient_checkpointing = gr.Checkbox(
                label='Gradient checkpointing', value=False
            )
            self.shuffle_caption = gr.Checkbox(label='Shuffle caption', value=False)
            self.persistent_data_loader_workers = gr.Checkbox(
                label='Persistent data loader', value=False
            )
            self.mem_eff_attn = gr.Checkbox(
                label='Memory efficient attention', value=False
            )
        with gr.Row():
            # This use_8bit_adam element should be removed in a future release as it is no longer used
            # use_8bit_adam = gr.Checkbox(
            #     label='Use 8bit adam', value=False, visible=False
            # )
            # self.xformers = gr.Checkbox(label='Use xformers', value=True, info='Use xformers for CrossAttention')
            self.xformers = gr.Dropdown(label='CrossAttention', choices=["none", "sdpa", "xformers"], value='xformers')
            self.color_aug = gr.Checkbox(label='Color augmentation', value=False)
            self.flip_aug = gr.Checkbox(label='Flip augmentation', value=False)
            self.min_snr_gamma = gr.Slider(
                label='Min SNR gamma', value=0, minimum=0, maximum=20, step=1
            )
        with gr.Row():
            # self.sdpa = gr.Checkbox(label='Use sdpa', value=False, info='Use sdpa for CrossAttention')
            self.bucket_no_upscale = gr.Checkbox(
                label="Don't upscale bucket resolution", value=True
            )
            self.bucket_reso_steps = gr.Slider(
                label='Bucket resolution steps', value=64, minimum=1, maximum=128
            )
            self.random_crop = gr.Checkbox(
                label='Random crop instead of center crop', value=False
            )
        
        with gr.Row():
            self.min_timestep = gr.Slider(
                label='Min Timestep',
                value=0,
                step=1,
                minimum=0,
                maximum=1000,
                info='Values greater than 0 will make the model more img2img focussed. 0 = image only'
            )
            self.max_timestep = gr.Slider(
                label='Max Timestep',
                value=1000,
                step=1,
                minimum=0,
                maximum=1000,
                info='Values lower than 1000 will make the model more img2img focussed. 1000 = noise only',
            )
        
        with gr.Row():
            self.noise_offset_type = gr.Dropdown(
                label='Noise offset type',
                choices=[
                    'Original',
                    'Multires',
                ],
                value='Original',
            )
            with gr.Row(visible=True) as self.noise_offset_original:
                self.noise_offset = gr.Slider(
                    label='Noise offset',
                    value=0,
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    info='recommended values are 0.05 - 0.15',
                )
                self.adaptive_noise_scale = gr.Slider(
                    label='Adaptive noise scale',
                    value=0,
                    minimum=-1,
                    maximum=1,
                    step=0.001,
                    info='(Experimental, Optional) Since the latent is close to a normal distribution, it may be a good idea to specify a value around 1/10 the noise offset.',
                )
            with gr.Row(visible=False) as self.noise_offset_multires:
                self.multires_noise_iterations = gr.Slider(
                    label='Multires noise iterations',
                    value=0,
                    minimum=0,
                    maximum=64,
                    step=1,
                    info='enable multires noise (recommended values are 6-10)',
                )
                self.multires_noise_discount = gr.Slider(
                    label='Multires noise discount',
                    value=0,
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    info='recommended values are 0.8. For LoRAs with small datasets, 0.1-0.3',
                )
            self.noise_offset_type.change(
                noise_offset_type_change,
                inputs=[self.noise_offset_type],
                outputs=[self.noise_offset_original, self.noise_offset_multires]
            )
        with gr.Row():
            self.caption_dropout_every_n_epochs = gr.Number(
                label='Dropout caption every n epochs', value=0
            )
            self.caption_dropout_rate = gr.Slider(
                label='Rate of caption dropout', value=0, minimum=0, maximum=1
            )
            self.vae_batch_size = gr.Slider(
                label='VAE batch size', minimum=0, maximum=32, value=0, step=1
            )
        with gr.Row():
            self.save_state = gr.Checkbox(label='Save training state', value=False)
            self.resume = gr.Textbox(
                label='Resume from saved training state',
                placeholder='path to "last-state" state folder to resume from',
            )
            self.resume_button = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            self.resume_button.click(
                get_folder_path,
                outputs=self.resume,
                show_progress=False,
            )
            # self.max_train_epochs = gr.Textbox(
            #     label='Max train epoch',
            #     placeholder='(Optional) Override number of epoch',
            # )
            self.max_data_loader_n_workers = gr.Textbox(
                label='Max num workers for DataLoader',
                placeholder='(Optional) Override number of epoch. Default: 8',
                value='0',
            )
        with gr.Row():
            self.wandb_api_key = gr.Textbox(
                label='WANDB API Key',
                value='',
                placeholder='(Optional)',
                info='Users can obtain and/or generate an api key in the their user settings on the website: https://wandb.ai/login',
            )
            self.use_wandb = gr.Checkbox(
                label='WANDB Logging',
                value=False,
                info='If unchecked, tensorboard will be used as the default for logging.',
            )
            self.scale_v_pred_loss_like_noise_pred = gr.Checkbox(
                label='Scale v prediction loss',
                value=False,
                info='Only for SD v2 models. By scaling the loss according to the time step, the weights of global noise prediction and local noise prediction become the same, and the improvement of details may be expected.',
            )
            