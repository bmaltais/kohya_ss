import gradio as gr
import json
import math
import os
import subprocess
import pathlib
import argparse
from datetime import datetime
from library.common_gui import (
    get_folder_path,
    get_file_path,
    get_saveasfile_path,
    save_inference_file,
    run_cmd_advanced_training,
    color_aug_changed,
    run_cmd_training,
    update_my_data,
    check_if_model_exist,
    SaveConfigFile,
    save_to_file
)
from library.class_configuration_file import ConfigurationFile
from library.class_source_model import SourceModel
from library.class_basic_training import BasicTraining
from library.class_advanced_training import AdvancedTraining
from library.class_sdxl_parameters import SDXLParameters
from library.class_command_executor import CommandExecutor
from library.tensorboard_gui import (
    gradio_tensorboard,
    start_tensorboard,
    stop_tensorboard,
)
from library.utilities import utilities_tab
from library.class_sample_images import SampleImages, run_cmd_sample

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = CommandExecutor()

# from easygui import msgbox

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄

PYTHON = 'python3' if os.name == 'posix' else './venv/Scripts/python.exe'


def save_configuration(
    save_as,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl_checkbox,
    train_dir,
    image_folder,
    output_dir,
    logging_dir,
    max_resolution,
    min_bucket_reso,
    max_bucket_reso,
    batch_size,
    flip_aug,
    caption_metadata_filename,
    latent_metadata_filename,
    full_path,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    dataset_repeats,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    train_text_encoder,
    full_bf16,
    create_caption,
    create_buckets,
    save_model_as,
    caption_extension,
    # use_8bit_adam,
    xformers,
    clip_skip,
    save_state,
    resume,
    gradient_checkpointing,
    gradient_accumulation_steps,
    mem_eff_attn,
    shuffle_caption,
    output_name,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    full_fp16,
    color_aug,
    model_list,
    cache_latents,
    cache_latents_to_disk,
    use_latent_files,
    keep_tokens,
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
    weighted_captions,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    min_timestep,
    max_timestep,
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
    sdxl_checkbox,
    train_dir,
    image_folder,
    output_dir,
    logging_dir,
    max_resolution,
    min_bucket_reso,
    max_bucket_reso,
    batch_size,
    flip_aug,
    caption_metadata_filename,
    latent_metadata_filename,
    full_path,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    dataset_repeats,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    train_text_encoder,
    full_bf16,
    create_caption,
    create_buckets,
    save_model_as,
    caption_extension,
    # use_8bit_adam,
    xformers,
    clip_skip,
    save_state,
    resume,
    gradient_checkpointing,
    gradient_accumulation_steps,
    mem_eff_attn,
    shuffle_caption,
    output_name,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    full_fp16,
    color_aug,
    model_list,
    cache_latents,
    cache_latents_to_disk,
    use_latent_files,
    keep_tokens,
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
    weighted_captions,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    min_timestep,
    max_timestep,
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
    sdxl_checkbox,
    train_dir,
    image_folder,
    output_dir,
    logging_dir,
    max_resolution,
    min_bucket_reso,
    max_bucket_reso,
    batch_size,
    flip_aug,
    caption_metadata_filename,
    latent_metadata_filename,
    full_path,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    dataset_repeats,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    train_text_encoder,
    full_bf16,
    generate_caption_database,
    generate_image_buckets,
    save_model_as,
    caption_extension,
    # use_8bit_adam,
    xformers,
    clip_skip,
    save_state,
    resume,
    gradient_checkpointing,
    gradient_accumulation_steps,
    mem_eff_attn,
    shuffle_caption,
    output_name,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    full_fp16,
    color_aug,
    model_list,  # Keep this. Yes, it is unused here but required given the common list used
    cache_latents,
    cache_latents_to_disk,
    use_latent_files,
    keep_tokens,
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
    weighted_captions,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    min_timestep,
    max_timestep,
):
    # Get list of function parameters and values
    parameters = list(locals().items())
    
    print_only_bool = True if print_only.get('label') == 'True' else False
    log.info(f'Start Finetuning...')

    headless_bool = True if headless.get('label') == 'True' else False

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

    # create caption json file
    if generate_caption_database:
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        run_cmd = f'{PYTHON} finetune/merge_captions_to_metadata.py'
        if caption_extension == '':
            run_cmd += f' --caption_extension=".caption"'
        else:
            run_cmd += f' --caption_extension={caption_extension}'
        run_cmd += f' "{image_folder}"'
        run_cmd += f' "{train_dir}/{caption_metadata_filename}"'
        if full_path:
            run_cmd += f' --full_path'

        log.info(run_cmd)

        if not print_only_bool:
            # Run the command
            if os.name == 'posix':
                os.system(run_cmd)
            else:
                subprocess.run(run_cmd)

    # create images buckets
    if generate_image_buckets:
        run_cmd = f'{PYTHON} finetune/prepare_buckets_latents.py'
        run_cmd += f' "{image_folder}"'
        run_cmd += f' "{train_dir}/{caption_metadata_filename}"'
        run_cmd += f' "{train_dir}/{latent_metadata_filename}"'
        run_cmd += f' "{pretrained_model_name_or_path}"'
        run_cmd += f' --batch_size={batch_size}'
        run_cmd += f' --max_resolution={max_resolution}'
        run_cmd += f' --min_bucket_reso={min_bucket_reso}'
        run_cmd += f' --max_bucket_reso={max_bucket_reso}'
        run_cmd += f' --mixed_precision={mixed_precision}'
        # if flip_aug:
        #     run_cmd += f' --flip_aug'
        if full_path:
            run_cmd += f' --full_path'
        if sdxl_no_half_vae:
            log.info('Using mixed_precision = no because no half vae is selected...')
            run_cmd += f' --mixed_precision="no"'

        log.info(run_cmd)

        if not print_only_bool:
            # Run the command
            if os.name == 'posix':
                os.system(run_cmd)
            else:
                subprocess.run(run_cmd)

    image_num = len(
        [
            f
            for f, lower_f in (
                (file, file.lower()) for file in os.listdir(image_folder)
            )
            if lower_f.endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
    )
    log.info(f'image_num = {image_num}')

    repeats = int(image_num) * int(dataset_repeats)
    log.info(f'repeats = {str(repeats)}')

    # calculate max_train_steps
    max_train_steps = int(
        math.ceil(
            float(repeats)
            / int(train_batch_size)
            / int(gradient_accumulation_steps)
            * int(epoch)
        )
    )

    # Divide by two because flip augmentation create two copied of the source images
    if flip_aug:
        max_train_steps = int(math.ceil(float(max_train_steps) / 2))

    log.info(f'max_train_steps = {max_train_steps}')

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    log.info(f'lr_warmup_steps = {lr_warmup_steps}')

    run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process}'
    if sdxl_checkbox:
        run_cmd += f' "./sdxl_train.py"'
    else:
        run_cmd += f' "./fine_tune.py"'

    if v2:
        run_cmd += ' --v2'
    if v_parameterization:
        run_cmd += ' --v_parameterization'
    if train_text_encoder:
        run_cmd += ' --train_text_encoder'
    if full_bf16:
        run_cmd += ' --full_bf16'
    if weighted_captions:
        run_cmd += ' --weighted_captions'
    run_cmd += (
        f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'
    )
    if use_latent_files == 'Yes':
        run_cmd += f' --in_json="{train_dir}/{latent_metadata_filename}"'
    else:
        run_cmd += f' --in_json="{train_dir}/{caption_metadata_filename}"'
    run_cmd += f' --train_data_dir="{image_folder}"'
    run_cmd += f' --output_dir="{output_dir}"'
    if not logging_dir == '':
        run_cmd += f' --logging_dir="{logging_dir}"'
    run_cmd += f' --dataset_repeats={dataset_repeats}'
    run_cmd += f' --learning_rate={learning_rate}'

    run_cmd += ' --enable_bucket'
    run_cmd += f' --resolution="{max_resolution}"'
    run_cmd += f' --min_bucket_reso={min_bucket_reso}'
    run_cmd += f' --max_bucket_reso={max_bucket_reso}'

    if not save_model_as == 'same as source model':
        run_cmd += f' --save_model_as={save_model_as}'
    if int(gradient_accumulation_steps) > 1:
        run_cmd += f' --gradient_accumulation_steps={int(gradient_accumulation_steps)}'
    # if save_state:
    #     run_cmd += ' --save_state'
    # if not resume == '':
    #     run_cmd += f' --resume={resume}'
    if not output_name == '':
        run_cmd += f' --output_name="{output_name}"'
    if int(max_token_length) > 75:
        run_cmd += f' --max_token_length={max_token_length}'
        
    if sdxl_cache_text_encoder_outputs:
        run_cmd += f' --cache_text_encoder_outputs'
        
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


def remove_doublequote(file_path):
    if file_path != None:
        file_path = file_path.replace('"', '')

    return file_path


def finetune_tab(headless=False):
    dummy_db_true = gr.Label(value=True, visible=False)
    dummy_db_false = gr.Label(value=False, visible=False)
    dummy_headless = gr.Label(value=headless, visible=False)
    with gr.Tab('Training'):
        gr.Markdown('Train a custom model using kohya finetune python code...')

        # Setup Configuration Files Gradio
        config = ConfigurationFile(headless)

        source_model = SourceModel(headless=headless)

        with gr.Tab('Folders'):
            with gr.Row():
                train_dir = gr.Textbox(
                    label='Training config folder',
                    placeholder='folder where the training configuration files will be saved',
                )
                train_dir_folder = gr.Button(
                    folder_symbol,
                    elem_id='open_folder_small',
                    visible=(not headless),
                )
                train_dir_folder.click(
                    get_folder_path,
                    outputs=train_dir,
                    show_progress=False,
                )

                image_folder = gr.Textbox(
                    label='Training Image folder',
                    placeholder='folder where the training images are located',
                )
                image_folder_input_folder = gr.Button(
                    folder_symbol,
                    elem_id='open_folder_small',
                    visible=(not headless),
                )
                image_folder_input_folder.click(
                    get_folder_path,
                    outputs=image_folder,
                    show_progress=False,
                )
            with gr.Row():
                output_dir = gr.Textbox(
                    label='Model output folder',
                    placeholder='folder where the model will be saved',
                )
                output_dir_input_folder = gr.Button(
                    folder_symbol,
                    elem_id='open_folder_small',
                    visible=(not headless),
                )
                output_dir_input_folder.click(
                    get_folder_path,
                    outputs=output_dir,
                    show_progress=False,
                )

                logging_dir = gr.Textbox(
                    label='Logging folder',
                    placeholder='Optional: enable logging and output TensorBoard log to this folder',
                )
                logging_dir_input_folder = gr.Button(
                    folder_symbol,
                    elem_id='open_folder_small',
                    visible=(not headless),
                )
                logging_dir_input_folder.click(
                    get_folder_path,
                    outputs=logging_dir,
                    show_progress=False,
                )
            with gr.Row():
                output_name = gr.Textbox(
                    label='Model output name',
                    placeholder='Name of the model to output',
                    value='last',
                    interactive=True,
                )
            train_dir.change(
                remove_doublequote,
                inputs=[train_dir],
                outputs=[train_dir],
            )
            image_folder.change(
                remove_doublequote,
                inputs=[image_folder],
                outputs=[image_folder],
            )
            output_dir.change(
                remove_doublequote,
                inputs=[output_dir],
                outputs=[output_dir],
            )
        with gr.Tab('Dataset preparation'):
            with gr.Row():
                max_resolution = gr.Textbox(
                    label='Resolution (width,height)', value='512,512'
                )
                min_bucket_reso = gr.Textbox(
                    label='Min bucket resolution', value='256'
                )
                max_bucket_reso = gr.Textbox(
                    label='Max bucket resolution', value='1024'
                )
                batch_size = gr.Textbox(label='Batch size', value='1')
            with gr.Row():
                create_caption = gr.Checkbox(
                    label='Generate caption metadata', value=True
                )
                create_buckets = gr.Checkbox(
                    label='Generate image buckets metadata', value=True
                )
                use_latent_files = gr.Dropdown(
                    label='Use latent files',
                    choices=[
                        'No',
                        'Yes',
                    ],
                    value='Yes',
                )
            with gr.Accordion('Advanced parameters', open=False):
                with gr.Row():
                    caption_metadata_filename = gr.Textbox(
                        label='Caption metadata filename', value='meta_cap.json'
                    )
                    latent_metadata_filename = gr.Textbox(
                        label='Latent metadata filename', value='meta_lat.json'
                    )
                with gr.Row():
                    full_path = gr.Checkbox(label='Use full path', value=True)
                    weighted_captions = gr.Checkbox(
                        label='Weighted captions', value=False
                    )
        with gr.Tab('Parameters'):
            with gr.Tab('Basic', elem_id='basic_tab'):
                basic_training = BasicTraining(learning_rate_value='1e-5', finetuning=True)
            
                # Add SDXL Parameters
                sdxl_params = SDXLParameters(source_model.sdxl_checkbox)
            
                with gr.Row():
                    dataset_repeats = gr.Textbox(label='Dataset repeats', value=40)
                    train_text_encoder = gr.Checkbox(
                        label='Train text encoder', value=True
                    )
                    
            with gr.Tab('Advanced', elem_id='advanced_tab'):
                with gr.Row():
                    gradient_accumulation_steps = gr.Number(
                        label='Gradient accumulate steps', value='1'
                    )
                advanced_training = AdvancedTraining(headless=headless, finetuning=True)
                advanced_training.color_aug.change(
                    color_aug_changed,
                    inputs=[advanced_training.color_aug],
                    outputs=[basic_training.cache_latents],  # Not applicable to fine_tune.py
                )
            
            with gr.Tab('Samples', elem_id='samples_tab'):
                sample = SampleImages()

        with gr.Row():
            button_run = gr.Button('Start training', variant='primary')
            
            button_stop_training = gr.Button('Stop training')

        button_print = gr.Button('Print training command')

        # Setup gradio tensorboard buttons
        button_start_tensorboard, button_stop_tensorboard = gradio_tensorboard()

        button_start_tensorboard.click(
            start_tensorboard,
            inputs=logging_dir,
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
            train_dir,
            image_folder,
            output_dir,
            logging_dir,
            max_resolution,
            min_bucket_reso,
            max_bucket_reso,
            batch_size,
            advanced_training.flip_aug,
            caption_metadata_filename,
            latent_metadata_filename,
            full_path,
            basic_training.learning_rate,
            basic_training.lr_scheduler,
            basic_training.lr_warmup,
            dataset_repeats,
            basic_training.train_batch_size,
            basic_training.epoch,
            basic_training.save_every_n_epochs,
            basic_training.mixed_precision,
            basic_training.save_precision,
            basic_training.seed,
            basic_training.num_cpu_threads_per_process,
            train_text_encoder,
            advanced_training.full_bf16,
            create_caption,
            create_buckets,
            source_model.save_model_as,
            basic_training.caption_extension,
            advanced_training.xformers,
            advanced_training.clip_skip,
            advanced_training.save_state,
            advanced_training.resume,
            advanced_training.gradient_checkpointing,
            gradient_accumulation_steps,
            advanced_training.mem_eff_attn,
            advanced_training.shuffle_caption,
            output_name,
            advanced_training.max_token_length,
            basic_training.max_train_epochs,
            advanced_training.max_data_loader_n_workers,
            advanced_training.full_fp16,
            advanced_training.color_aug,
            source_model.model_list,
            basic_training.cache_latents,
            basic_training.cache_latents_to_disk,
            use_latent_files,
            advanced_training.keep_tokens,
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
            weighted_captions,
            advanced_training.save_every_n_steps,
            advanced_training.save_last_n_steps,
            advanced_training.save_last_n_steps_state,
            advanced_training.use_wandb,
            advanced_training.wandb_api_key,
            advanced_training.scale_v_pred_loss_like_noise_pred,
            sdxl_params.sdxl_cache_text_encoder_outputs,
            sdxl_params.sdxl_no_half_vae,
            advanced_training.min_timestep,
            advanced_training.max_timestep,
        ]

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
        
    with gr.Tab('Guides'):
        gr.Markdown(
            'This section provide Various Finetuning guides and information...'
        )
        top_level_path = './docs/Finetuning/top_level.md'
        if os.path.exists(top_level_path):
            with open(os.path.join(top_level_path), 'r', encoding='utf8') as file:
                guides_top_level = file.read() + '\n'
        gr.Markdown(guides_top_level)


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
        with gr.Tab('Finetune'):
            finetune_tab(headless=headless)
        with gr.Tab('Utilities'):
            utilities_tab(enable_dreambooth_tab=False, headless=headless)

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
