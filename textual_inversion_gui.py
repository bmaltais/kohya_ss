# v1: initial release
# v2: add open and save folder icons
# v3: Add new Utilities tab for Dreambooth folder preparation
# v3.1: Adding captionning of images to utilities

import gradio as gr
import json
import math
import os
import subprocess
import pathlib
import argparse
from datetime import datetime
from library.common_gui import (
    get_file_path,
    get_saveasfile_path,
    color_aug_changed,
    save_inference_file,
    run_cmd_advanced_training,
    run_cmd_training,
    update_my_data,
    check_if_model_exist,
    output_message,
    verify_image_folder_pattern,
    SaveConfigFile,
    save_to_file
)
from library.class_configuration_file import ConfigurationFile
from library.class_source_model import SourceModel
from library.class_basic_training import BasicTraining
from library.class_advanced_training import AdvancedTraining
from library.class_folders import Folders
from library.class_sdxl_parameters import SDXLParameters
from library.class_command_executor import CommandExecutor
from library.tensorboard_gui import (
    gradio_tensorboard,
    start_tensorboard,
    stop_tensorboard,
)
from library.dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from library.utilities import utilities_tab
from library.class_sample_images import SampleImages, run_cmd_sample

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = CommandExecutor()

def save_configuration(
    save_as,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training,
    min_bucket_reso,
    max_bucket_reso,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    color_aug,
    flip_aug,
    clip_skip,
    vae,
    output_name,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    mem_eff_attn,
    gradient_accumulation_steps,
    model_list,
    token_string,
    init_word,
    num_vectors_per_token,
    max_train_steps,
    weights,
    template,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    min_timestep,
    max_timestep,
    sdxl_no_half_vae
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    original_file_path = file_path

    save_as_bool = True if save_as.get('label') == 'True' else False

    if save_as_bool:
        log.info('Save as...')
        file_path = get_saveasfile_path(file_path)
    else:
        log.info('Save...')
        if file_path == None or file_path == '':
            file_path = get_saveasfile_path(file_path)

    # log.info(file_path)

    if file_path == None or file_path == '':
        return original_file_path  # In case a file_path was provided and the user decide to cancel the open action

    # Extract the destination directory from the file path
    destination_directory = os.path.dirname(file_path)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    SaveConfigFile(parameters=parameters, file_path=file_path, exclusion=['file_path', 'save_as'])

    return file_path


def open_configuration(
    ask_for_file,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training,
    min_bucket_reso,
    max_bucket_reso,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    color_aug,
    flip_aug,
    clip_skip,
    vae,
    output_name,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    mem_eff_attn,
    gradient_accumulation_steps,
    model_list,
    token_string,
    init_word,
    num_vectors_per_token,
    max_train_steps,
    weights,
    template,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    min_timestep,
    max_timestep,
    sdxl_no_half_vae
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    ask_for_file = True if ask_for_file.get('label') == 'True' else False

    original_file_path = file_path

    if ask_for_file:
        file_path = get_file_path(file_path)

    if not file_path == '' and not file_path == None:
        # load variables from JSON file
        with open(file_path, 'r') as f:
            my_data = json.load(f)
            log.info('Loading config...')
            # Update values to fix deprecated use_8bit_adam checkbox and set appropriate optimizer if it is set to True
            my_data = update_my_data(my_data)
    else:
        file_path = original_file_path  # In case a file_path was provided and the user decide to cancel the open action
        my_data = {}

    values = [file_path]
    for key, value in parameters:
        # Set the value in the dictionary to the corresponding value in `my_data`, or the default value if not found
        if not key in ['ask_for_file', 'file_path']:
            values.append(my_data.get(key, value))
    return tuple(values)


def train_model(
    headless,
    print_only,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training_pct,
    min_bucket_reso,
    max_bucket_reso,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    color_aug,
    flip_aug,
    clip_skip,
    vae,
    output_name,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    mem_eff_attn,
    gradient_accumulation_steps,
    model_list,  # Keep this. Yes, it is unused here but required given the common list used
    token_string,
    init_word,
    num_vectors_per_token,
    max_train_steps,
    weights,
    template,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    min_timestep,
    max_timestep,
    sdxl_no_half_vae
):
    # Get list of function parameters and values
    parameters = list(locals().items())
    
    print_only_bool = True if print_only.get('label') == 'True' else False
    log.info(f'Start training TI...')

    headless_bool = True if headless.get('label') == 'True' else False

    if pretrained_model_name_or_path == '':
        output_message(
            msg='Source model information is missing', headless=headless_bool
        )
        return

    if train_data_dir == '':
        output_message(
            msg='Image folder path is missing', headless=headless_bool
        )
        return

    if not os.path.exists(train_data_dir):
        output_message(
            msg='Image folder does not exist', headless=headless_bool
        )
        return

    if not verify_image_folder_pattern(train_data_dir):
        return

    if reg_data_dir != '':
        if not os.path.exists(reg_data_dir):
            output_message(
                msg='Regularisation folder does not exist',
                headless=headless_bool,
            )
            return

        if not verify_image_folder_pattern(reg_data_dir):
            return

    if output_dir == '':
        output_message(
            msg='Output folder path is missing', headless=headless_bool
        )
        return

    if token_string == '':
        output_message(msg='Token string is missing', headless=headless_bool)
        return

    if init_word == '':
        output_message(msg='Init word is missing', headless=headless_bool)
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if check_if_model_exist(
        output_name, output_dir, save_model_as, headless_bool
    ):
        return

    # if float(noise_offset) > 0 and (
    #     multires_noise_iterations > 0 or multires_noise_discount > 0
    # ):
    #     output_message(
    #         msg="noise offset and multires_noise can't be set at the same time. Only use one or the other.",
    #         title='Error',
    #         headless=headless_bool,
    #     )
    #     return

    # if optimizer == 'Adafactor' and lr_warmup != '0':
    #     output_message(
    #         msg="Warning: lr_scheduler is set to 'Adafactor', so 'LR warmup (% of steps)' will be considered 0.",
    #         title='Warning',
    #         headless=headless_bool,
    #     )
    #     lr_warmup = '0'

    # Get a list of all subfolders in train_data_dir
    subfolders = [
        f
        for f in os.listdir(train_data_dir)
        if os.path.isdir(os.path.join(train_data_dir, f))
    ]

    total_steps = 0

    # Loop through each subfolder and extract the number of repeats
    for folder in subfolders:
        # Extract the number of repeats from the folder name
        repeats = int(folder.split('_')[0])

        # Count the number of images in the folder
        num_images = len(
            [
                f
                for f, lower_f in (
                    (file, file.lower())
                    for file in os.listdir(
                        os.path.join(train_data_dir, folder)
                    )
                )
                if lower_f.endswith(('.jpg', '.jpeg', '.png', '.webp'))
            ]
        )

        # Calculate the total number of steps for this folder
        steps = repeats * num_images
        total_steps += steps

        # Print the result
        log.info(f'Folder {folder}: {steps} steps')

    # Print the result
    # log.info(f"{total_steps} total steps")

    if reg_data_dir == '':
        reg_factor = 1
    else:
        log.info(
            'Regularisation images are used... Will double the number of steps required...'
        )
        reg_factor = 2

    # calculate max_train_steps
    if max_train_steps == '':
        max_train_steps = int(
            math.ceil(
                float(total_steps)
                / int(train_batch_size)
                / int(gradient_accumulation_steps)
                * int(epoch)
                * int(reg_factor)
            )
        )
    else:
        max_train_steps = int(max_train_steps)

    log.info(f'max_train_steps = {max_train_steps}')

    # calculate stop encoder training
    if stop_text_encoder_training_pct == None:
        stop_text_encoder_training = 0
    else:
        stop_text_encoder_training = math.ceil(
            float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
        )
    log.info(f'stop_text_encoder_training = {stop_text_encoder_training}')

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    log.info(f'lr_warmup_steps = {lr_warmup_steps}')

    run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process}'
    if sdxl:
        run_cmd += f' "./sdxl_train_textual_inversion.py"'
    else:
        run_cmd += f' "./train_textual_inversion.py"'
        
    if v2:
        run_cmd += ' --v2'
    if v_parameterization:
        run_cmd += ' --v_parameterization'
    if enable_bucket:
        run_cmd += f' --enable_bucket --min_bucket_reso={min_bucket_reso} --max_bucket_reso={max_bucket_reso}'
    if no_token_padding:
        run_cmd += ' --no_token_padding'
    run_cmd += (
        f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'
    )
    run_cmd += f' --train_data_dir="{train_data_dir}"'
    if len(reg_data_dir):
        run_cmd += f' --reg_data_dir="{reg_data_dir}"'
    run_cmd += f' --resolution="{max_resolution}"'
    run_cmd += f' --output_dir="{output_dir}"'
    if not logging_dir == '':
        run_cmd += f' --logging_dir="{logging_dir}"'
    if not stop_text_encoder_training == 0:
        run_cmd += (
            f' --stop_text_encoder_training={stop_text_encoder_training}'
        )
    if not save_model_as == 'same as source model':
        run_cmd += f' --save_model_as={save_model_as}'
    # if not resume == '':
    #     run_cmd += f' --resume={resume}'
    if not float(prior_loss_weight) == 1.0:
        run_cmd += f' --prior_loss_weight={prior_loss_weight}'
    if not vae == '':
        run_cmd += f' --vae="{vae}"'
    if not output_name == '':
        run_cmd += f' --output_name="{output_name}"'
    if not lr_scheduler_num_cycles == '':
        run_cmd += f' --lr_scheduler_num_cycles="{lr_scheduler_num_cycles}"'
    else:
        run_cmd += f' --lr_scheduler_num_cycles="{epoch}"'
    if not lr_scheduler_power == '':
        run_cmd += f' --lr_scheduler_power="{lr_scheduler_power}"'
    if int(max_token_length) > 75:
        run_cmd += f' --max_token_length={max_token_length}'
    if not max_train_epochs == '':
        run_cmd += f' --max_train_epochs="{max_train_epochs}"'
    if not max_data_loader_n_workers == '':
        run_cmd += (
            f' --max_data_loader_n_workers="{max_data_loader_n_workers}"'
        )
    if int(gradient_accumulation_steps) > 1:
        run_cmd += f' --gradient_accumulation_steps={int(gradient_accumulation_steps)}'
    
    if sdxl_no_half_vae:
        run_cmd += f' --no_half_vae'

    run_cmd += run_cmd_training(
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        train_batch_size=train_batch_size,
        max_train_steps=max_train_steps,
        save_every_n_epochs=save_every_n_epochs,
        mixed_precision=mixed_precision,
        save_precision=save_precision,
        seed=seed,
        caption_extension=caption_extension,
        cache_latents=cache_latents,
        cache_latents_to_disk=cache_latents_to_disk,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
    )

    run_cmd += run_cmd_advanced_training(
        max_train_epochs=max_train_epochs,
        max_data_loader_n_workers=max_data_loader_n_workers,
        max_token_length=max_token_length,
        resume=resume,
        save_state=save_state,
        mem_eff_attn=mem_eff_attn,
        clip_skip=clip_skip,
        flip_aug=flip_aug,
        color_aug=color_aug,
        shuffle_caption=shuffle_caption,
        gradient_checkpointing=gradient_checkpointing,
        full_fp16=full_fp16,
        xformers=xformers,
        # use_8bit_adam=use_8bit_adam,
        keep_tokens=keep_tokens,
        persistent_data_loader_workers=persistent_data_loader_workers,
        bucket_no_upscale=bucket_no_upscale,
        random_crop=random_crop,
        bucket_reso_steps=bucket_reso_steps,
        caption_dropout_every_n_epochs=caption_dropout_every_n_epochs,
        caption_dropout_rate=caption_dropout_rate,
        noise_offset_type=noise_offset_type,
        noise_offset=noise_offset,
        adaptive_noise_scale=adaptive_noise_scale,
        multires_noise_iterations=multires_noise_iterations,
        multires_noise_discount=multires_noise_discount,
        additional_parameters=additional_parameters,
        vae_batch_size=vae_batch_size,
        min_snr_gamma=min_snr_gamma,
        save_every_n_steps=save_every_n_steps,
        save_last_n_steps=save_last_n_steps,
        save_last_n_steps_state=save_last_n_steps_state,
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        scale_v_pred_loss_like_noise_pred=scale_v_pred_loss_like_noise_pred,
        min_timestep=min_timestep,
        max_timestep=max_timestep,
    )
    run_cmd += f' --token_string="{token_string}"'
    run_cmd += f' --init_word="{init_word}"'
    run_cmd += f' --num_vectors_per_token={num_vectors_per_token}'
    if not weights == '':
        run_cmd += f' --weights="{weights}"'
    if template == 'object template':
        run_cmd += f' --use_object_template'
    elif template == 'style template':
        run_cmd += f' --use_style_template'

    run_cmd += run_cmd_sample(
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_sampler,
        sample_prompts,
        output_dir,
    )

    if print_only_bool:
        log.warning(
            'Here is the trainer command as a reference. It will not be executed:\n'
        )
        print(run_cmd)
        
        save_to_file(run_cmd)
    else:
        # Saving config file for model
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(output_dir, f'{output_name}_{formatted_datetime}.json')
        
        log.info(f'Saving training config to {file_path}...')

        SaveConfigFile(parameters=parameters, file_path=file_path, exclusion=['file_path', 'save_as', 'headless', 'print_only'])
        
        log.info(run_cmd)

        # Run the command
        
        executor.execute_command(run_cmd=run_cmd)

        # check if output_dir/last is a folder... therefore it is a diffuser model
        last_dir = pathlib.Path(f'{output_dir}/{output_name}')

        if not last_dir.is_dir():
            # Copy inference model for v2 if required
            save_inference_file(
                output_dir, v2, v_parameterization, output_name
            )


def ti_tab(
    headless=False,
):
    dummy_db_true = gr.Label(value=True, visible=False)
    dummy_db_false = gr.Label(value=False, visible=False)
    dummy_headless = gr.Label(value=headless, visible=False)
    
    with gr.Tab('Training'):
        gr.Markdown('Train a TI using kohya textual inversion python code...')
        
        # Setup Configuration Files Gradio
        config = ConfigurationFile(headless)

        source_model = SourceModel(
            save_model_as_choices=[
                'ckpt',
                'safetensors',
            ],
            headless=headless,
        )

        with gr.Tab('Folders'):
            folders = Folders(headless=headless)
        with gr.Tab('Parameters'):
            with gr.Tab('Basic', elem_id='basic_tab'):
                with gr.Row():
                    weights = gr.Textbox(
                        label='Resume TI training',
                        placeholder='(Optional) Path to existing TI embeding file to keep training',
                    )
                    weights_file_input = gr.Button(
                        '📂', elem_id='open_folder_small', visible=(not headless)
                    )
                    weights_file_input.click(
                        get_file_path,
                        outputs=weights,
                        show_progress=False,
                    )
                with gr.Row():
                    token_string = gr.Textbox(
                        label='Token string',
                        placeholder='eg: cat',
                    )
                    init_word = gr.Textbox(
                        label='Init word',
                        value='*',
                    )
                    num_vectors_per_token = gr.Slider(
                        minimum=1,
                        maximum=75,
                        value=1,
                        step=1,
                        label='Vectors',
                    )
                    max_train_steps = gr.Textbox(
                        label='Max train steps',
                        placeholder='(Optional) Maximum number of steps',
                    )
                    template = gr.Dropdown(
                        label='Template',
                        choices=[
                            'caption',
                            'object template',
                            'style template',
                        ],
                        value='caption',
                    )
                basic_training = BasicTraining(
                    learning_rate_value='1e-5',
                    lr_scheduler_value='cosine',
                    lr_warmup_value='10',
                )
                    
                # Add SDXL Parameters
                sdxl_params = SDXLParameters(source_model.sdxl_checkbox, show_sdxl_cache_text_encoder_outputs=False)
                
            with gr.Tab('Advanced', elem_id='advanced_tab'):
                advanced_training = AdvancedTraining(headless=headless)
                advanced_training.color_aug.change(
                    color_aug_changed,
                    inputs=[advanced_training.color_aug],
                    outputs=[basic_training.cache_latents],
                )
            
            with gr.Tab('Samples', elem_id='samples_tab'):
                sample = SampleImages()

        with gr.Tab('Tools'):
            gr.Markdown(
                'This section provide Dreambooth tools to help setup your dataset...'
            )
            gradio_dreambooth_folder_creation_tab(
                train_data_dir_input=folders.train_data_dir,
                reg_data_dir_input=folders.reg_data_dir,
                output_dir_input=folders.output_dir,
                logging_dir_input=folders.logging_dir,
                headless=headless,
            )

        with gr.Row():
            button_run = gr.Button('Start training', variant='primary')
            
            button_stop_training = gr.Button('Stop training')

        button_print = gr.Button('Print training command')

        # Setup gradio tensorboard buttons
        button_start_tensorboard, button_stop_tensorboard = gradio_tensorboard()

        button_start_tensorboard.click(
            start_tensorboard,
            inputs=folders.logging_dir,
            show_progress=False,
        )

        button_stop_tensorboard.click(
            stop_tensorboard,
            show_progress=False,
        )

        settings_list = [
            source_model.pretrained_model_name_or_path,
            source_model.v2,
            source_model.v_parameterization,
            source_model.sdxl_checkbox,
            folders.logging_dir,
            folders.train_data_dir,
            folders.reg_data_dir,
            folders.output_dir,
            basic_training.max_resolution,
            basic_training.learning_rate,
            basic_training.lr_scheduler,
            basic_training.lr_warmup,
            basic_training.train_batch_size,
            basic_training.epoch,
            basic_training.save_every_n_epochs,
            basic_training.mixed_precision,
            basic_training.save_precision,
            basic_training.seed,
            basic_training.num_cpu_threads_per_process,
            basic_training.cache_latents,
            basic_training.cache_latents_to_disk,
            basic_training.caption_extension,
            basic_training.enable_bucket,
            advanced_training.gradient_checkpointing,
            advanced_training.full_fp16,
            advanced_training.no_token_padding,
            basic_training.stop_text_encoder_training,
            basic_training.min_bucket_reso,
            basic_training.max_bucket_reso,
            advanced_training.xformers,
            source_model.save_model_as,
            advanced_training.shuffle_caption,
            advanced_training.save_state,
            advanced_training.resume,
            advanced_training.prior_loss_weight,
            advanced_training.color_aug,
            advanced_training.flip_aug,
            advanced_training.clip_skip,
            advanced_training.vae,
            folders.output_name,
            advanced_training.max_token_length,
            basic_training.max_train_epochs,
            advanced_training.max_data_loader_n_workers,
            advanced_training.mem_eff_attn,
            advanced_training.gradient_accumulation_steps,
            source_model.model_list,
            token_string,
            init_word,
            num_vectors_per_token,
            max_train_steps,
            weights,
            template,
            advanced_training.keep_tokens,
            basic_training.lr_scheduler_num_cycles,
            basic_training.lr_scheduler_power,
            advanced_training.persistent_data_loader_workers,
            advanced_training.bucket_no_upscale,
            advanced_training.random_crop,
            advanced_training.bucket_reso_steps,
            advanced_training.caption_dropout_every_n_epochs,
            advanced_training.caption_dropout_rate,
            basic_training.optimizer,
            basic_training.optimizer_args,
            advanced_training.noise_offset_type,
            advanced_training.noise_offset,
            advanced_training.adaptive_noise_scale,
            advanced_training.multires_noise_iterations,
            advanced_training.multires_noise_discount,
            sample.sample_every_n_steps,
            sample.sample_every_n_epochs,
            sample.sample_sampler,
            sample.sample_prompts,
            advanced_training.additional_parameters,
            advanced_training.vae_batch_size,
            advanced_training.min_snr_gamma,
            advanced_training.save_every_n_steps,
            advanced_training.save_last_n_steps,
            advanced_training.save_last_n_steps_state,
            advanced_training.use_wandb,
            advanced_training.wandb_api_key,
            advanced_training.scale_v_pred_loss_like_noise_pred,
            advanced_training.min_timestep,
            advanced_training.max_timestep,
            sdxl_params.sdxl_no_half_vae,
        ]

        config.button_open_config.click(
            open_configuration,
            inputs=[dummy_db_true, config.config_file_name] + settings_list,
            outputs=[config.config_file_name] + settings_list,
            show_progress=False,
        )

        config.button_load_config.click(
            open_configuration,
            inputs=[dummy_db_false, config.config_file_name] + settings_list,
            outputs=[config.config_file_name] + settings_list,
            show_progress=False,
        )

        config.button_save_config.click(
            save_configuration,
            inputs=[dummy_db_false, config.config_file_name] + settings_list,
            outputs=[config.config_file_name],
            show_progress=False,
        )

        config.button_save_as_config.click(
            save_configuration,
            inputs=[dummy_db_true, config.config_file_name] + settings_list,
            outputs=[config.config_file_name],
            show_progress=False,
        )

        button_run.click(
            train_model,
            inputs=[dummy_headless] + [dummy_db_false] + settings_list,
            show_progress=False,
        )
        
        button_stop_training.click(
            executor.kill_command
        )

        button_print.click(
            train_model,
            inputs=[dummy_headless] + [dummy_db_true] + settings_list,
            show_progress=False,
        )

        return (
            folders.train_data_dir,
            folders.reg_data_dir,
            folders.output_dir,
            folders.logging_dir,
        )


def UI(**kwargs):
    css = ''

    headless = kwargs.get('headless', False)
    log.info(f'headless: {headless}')

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            log.info('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Blocks(
        css=css, title='Kohya_ss GUI', theme=gr.themes.Default()
    )

    with interface:
        with gr.Tab('Dreambooth TI'):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = ti_tab(headless=headless)
        with gr.Tab('Utilities'):
            utilities_tab(
                train_data_dir_input=train_data_dir_input,
                reg_data_dir_input=reg_data_dir_input,
                output_dir_input=output_dir_input,
                logging_dir_input=logging_dir_input,
                enable_copy_info_button=True,
                headless=headless,
            )

    # Show the interface
    launch_kwargs = {}
    username = kwargs.get('username')
    password = kwargs.get('password')
    server_port = kwargs.get('server_port', 0)
    inbrowser = kwargs.get('inbrowser', False)
    share = kwargs.get('share', False)
    server_name = kwargs.get('listen')

    launch_kwargs['server_name'] = server_name
    if username and password:
        launch_kwargs['auth'] = (username, password)
    if server_port > 0:
        launch_kwargs['server_port'] = server_port
    if inbrowser:
        launch_kwargs['inbrowser'] = inbrowser
    if share:
        launch_kwargs['share'] = share
    interface.launch(**launch_kwargs)


if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )
    parser.add_argument(
        '--headless', action='store_true', help='Is the server headless'
    )

    args = parser.parse_args()

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
        headless=args.headless,
    )
