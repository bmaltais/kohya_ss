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
from library.common_gui import (
    get_folder_path,
    remove_doublequote,
    get_file_path,
    get_any_file_path,
    get_saveasfile_path,
    color_aug_changed,
    save_inference_file,
    gradio_advanced_training,
    run_cmd_advanced_training,
    run_cmd_training,
    gradio_training,
    gradio_config,
    gradio_source_model,
    # set_legacy_8bitadam,
    update_my_data,
    check_if_model_exist,
)
from library.tensorboard_gui import (
    gradio_tensorboard,
    start_tensorboard,
    stop_tensorboard,
)
from library.dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from library.utilities import utilities_tab
from library.sampler_gui import sample_gradio_config, run_cmd_sample
from easygui import msgbox

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄


def save_configuration(
    save_as,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
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
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training,
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
    keep_tokens,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    noise_offset,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,weighted_captions,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    original_file_path = file_path

    save_as_bool = True if save_as.get('label') == 'True' else False

    if save_as_bool:
        print('Save as...')
        file_path = get_saveasfile_path(file_path)
    else:
        print('Save...')
        if file_path == None or file_path == '':
            file_path = get_saveasfile_path(file_path)

    # print(file_path)

    if file_path == None or file_path == '':
        return original_file_path  # In case a file_path was provided and the user decide to cancel the open action

    # Return the values of the variables as a dictionary
    variables = {
        name: value
        for name, value in parameters  # locals().items()
        if name
        not in [
            'file_path',
            'save_as',
        ]
    }

    # Extract the destination directory from the file path
    destination_directory = os.path.dirname(file_path)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Save the data to the selected file
    with open(file_path, 'w') as file:
        json.dump(variables, file, indent=2)

    return file_path


def open_configuration(
    ask_for_file,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
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
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training,
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
    keep_tokens,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    noise_offset,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,weighted_captions,
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
            print('Loading config...')
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
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
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
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training_pct,
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
    keep_tokens,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    noise_offset,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,weighted_captions,
):
    if pretrained_model_name_or_path == '':
        msgbox('Source model information is missing')
        return

    if train_data_dir == '':
        msgbox('Image folder path is missing')
        return

    if not os.path.exists(train_data_dir):
        msgbox('Image folder does not exist')
        return

    if reg_data_dir != '':
        if not os.path.exists(reg_data_dir):
            msgbox('Regularisation folder does not exist')
            return

    if output_dir == '':
        msgbox('Output folder path is missing')
        return

    if check_if_model_exist(output_name, output_dir, save_model_as):
        return
    
    if optimizer == 'Adafactor' and lr_warmup != '0':
        msgbox("Warning: lr_scheduler is set to 'Adafactor', so 'LR warmup (% of steps)' will be considered 0.", title="Warning")
        lr_warmup = '0'

    # Get a list of all subfolders in train_data_dir, excluding hidden folders
    subfolders = [
        f
        for f in os.listdir(train_data_dir)
        if os.path.isdir(os.path.join(train_data_dir, f))
        and not f.startswith('.')
    ]

    # Check if subfolders are present. If not let the user know and return
    if not subfolders:
        print(
            '\033[33mNo subfolders were found in',
            train_data_dir,
            " can't train\...033[0m",
        )
        return

    total_steps = 0

    # Loop through each subfolder and extract the number of repeats
    for folder in subfolders:
        # Extract the number of repeats from the folder name
        try:
            repeats = int(folder.split('_')[0])
        except ValueError:
            print(
                '\033[33mSubfolder',
                folder,
                "does not have a proper repeat value, please correct the name or remove it... can't train...\033[0m",
            )
            continue

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

        if num_images == 0:
            print(f'{folder} folder contain no images, skipping...')
        else:
            # Calculate the total number of steps for this folder
            steps = repeats * num_images
            total_steps += steps

            # Print the result
            print('\033[33mFolder', folder, ':', steps, 'steps\033[0m')

    if total_steps == 0:
        print(
            '\033[33mNo images were found in folder',
            train_data_dir,
            '... please rectify!\033[0m',
        )
        return

    # Print the result
    # print(f"{total_steps} total steps")

    if reg_data_dir == '':
        reg_factor = 1
    else:
        print(
            '\033[94mRegularisation images are used... Will double the number of steps required...\033[0m'
        )
        reg_factor = 2

    # calculate max_train_steps
    max_train_steps = int(
        math.ceil(
            float(total_steps)
            / int(train_batch_size)
            * int(epoch)
            * int(reg_factor)
        )
    )
    print(f'max_train_steps = {max_train_steps}')

    # calculate stop encoder training
    if int(stop_text_encoder_training_pct) == -1:
        stop_text_encoder_training = -1
    elif stop_text_encoder_training_pct == None:
        stop_text_encoder_training = 0
    else:
        stop_text_encoder_training = math.ceil(
            float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
        )
    print(f'stop_text_encoder_training = {stop_text_encoder_training}')

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    print(f'lr_warmup_steps = {lr_warmup_steps}')

    run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process} "train_db.py"'
    if v2:
        run_cmd += ' --v2'
    if v_parameterization:
        run_cmd += ' --v_parameterization'
    if enable_bucket:
        run_cmd += ' --enable_bucket'
    if no_token_padding:
        run_cmd += ' --no_token_padding'
    if weighted_captions:
        run_cmd += ' --weighted_captions'
    run_cmd += (
        f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'
    )
    run_cmd += f' --train_data_dir="{train_data_dir}"'
    if len(reg_data_dir):
        run_cmd += f' --reg_data_dir="{reg_data_dir}"'
    run_cmd += f' --resolution={max_resolution}'
    run_cmd += f' --output_dir="{output_dir}"'
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
        noise_offset=noise_offset,
        additional_parameters=additional_parameters,
        vae_batch_size=vae_batch_size,
        min_snr_gamma=min_snr_gamma,
    )

    run_cmd += run_cmd_sample(
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_sampler,
        sample_prompts,
        output_dir,
    )

    print(run_cmd)

    # Run the command
    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)

    # check if output_dir/last is a folder... therefore it is a diffuser model
    last_dir = pathlib.Path(f'{output_dir}/{output_name}')

    if not last_dir.is_dir():
        # Copy inference model for v2 if required
        save_inference_file(output_dir, v2, v_parameterization, output_name)


def dreambooth_tab(
    train_data_dir=gr.Textbox(),
    reg_data_dir=gr.Textbox(),
    output_dir=gr.Textbox(),
    logging_dir=gr.Textbox(),
):
    dummy_db_true = gr.Label(value=True, visible=False)
    dummy_db_false = gr.Label(value=False, visible=False)
    gr.Markdown('Train a custom model using kohya dreambooth python code...')
    (
        button_open_config,
        button_save_config,
        button_save_as_config,
        config_file_name,
        button_load_config,
    ) = gradio_config()

    (
        pretrained_model_name_or_path,
        v2,
        v_parameterization,
        save_model_as,
        model_list,
    ) = gradio_source_model()

    with gr.Tab('Folders'):
        with gr.Row():
            train_data_dir = gr.Textbox(
                label='Image folder',
                placeholder='Folder where the training folders containing the images are located',
            )
            train_data_dir_input_folder = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            train_data_dir_input_folder.click(
                get_folder_path,
                outputs=train_data_dir,
                show_progress=False,
            )
            reg_data_dir = gr.Textbox(
                label='Regularisation folder',
                placeholder='(Optional) Folder where where the regularization folders containing the images are located',
            )
            reg_data_dir_input_folder = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            reg_data_dir_input_folder.click(
                get_folder_path,
                outputs=reg_data_dir,
                show_progress=False,
            )
        with gr.Row():
            output_dir = gr.Textbox(
                label='Model output folder',
                placeholder='Folder to output trained model',
            )
            output_dir_input_folder = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            output_dir_input_folder.click(get_folder_path, outputs=output_dir)
            logging_dir = gr.Textbox(
                label='Logging folder',
                placeholder='Optional: enable logging and output TensorBoard log to this folder',
            )
            logging_dir_input_folder = gr.Button(
                '📂', elem_id='open_folder_small'
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
        train_data_dir.change(
            remove_doublequote,
            inputs=[train_data_dir],
            outputs=[train_data_dir],
        )
        reg_data_dir.change(
            remove_doublequote,
            inputs=[reg_data_dir],
            outputs=[reg_data_dir],
        )
        output_dir.change(
            remove_doublequote,
            inputs=[output_dir],
            outputs=[output_dir],
        )
        logging_dir.change(
            remove_doublequote,
            inputs=[logging_dir],
            outputs=[logging_dir],
        )
    with gr.Tab('Training parameters'):
        (
            learning_rate,
            lr_scheduler,
            lr_warmup,
            train_batch_size,
            epoch,
            save_every_n_epochs,
            mixed_precision,
            save_precision,
            num_cpu_threads_per_process,
            seed,
            caption_extension,
            cache_latents,
            optimizer,
            optimizer_args,
        ) = gradio_training(
            learning_rate_value='1e-5',
            lr_scheduler_value='cosine',
            lr_warmup_value='10',
        )
        with gr.Row():
            max_resolution = gr.Textbox(
                label='Max resolution',
                value='512,512',
                placeholder='512,512',
            )
            stop_text_encoder_training = gr.Slider(
                minimum=-1,
                maximum=100,
                value=0,
                step=1,
                label='Stop text encoder training',
            )
            enable_bucket = gr.Checkbox(label='Enable buckets', value=True)
        with gr.Accordion('Advanced Configuration', open=False):
            with gr.Row():
                no_token_padding = gr.Checkbox(
                    label='No token padding', value=False
                )
                gradient_accumulation_steps = gr.Number(
                    label='Gradient accumulate steps', value='1'
                )
                weighted_captions = gr.Checkbox(
                    label='Weighted captions', value=False
                )
            with gr.Row():
                prior_loss_weight = gr.Number(
                    label='Prior loss weight', value=1.0
                )
                vae = gr.Textbox(
                    label='VAE',
                    placeholder='(Optiona) path to checkpoint of vae to replace for training',
                )
                vae_button = gr.Button('📂', elem_id='open_folder_small')
                vae_button.click(
                    get_any_file_path,
                    outputs=vae,
                    show_progress=False,
                )
            (
                # use_8bit_adam,
                xformers,
                full_fp16,
                gradient_checkpointing,
                shuffle_caption,
                color_aug,
                flip_aug,
                clip_skip,
                mem_eff_attn,
                save_state,
                resume,
                max_token_length,
                max_train_epochs,
                max_data_loader_n_workers,
                keep_tokens,
                persistent_data_loader_workers,
                bucket_no_upscale,
                random_crop,
                bucket_reso_steps,
                caption_dropout_every_n_epochs,
                caption_dropout_rate,
                noise_offset,
                additional_parameters,
                vae_batch_size,
                min_snr_gamma,
            ) = gradio_advanced_training()
            color_aug.change(
                color_aug_changed,
                inputs=[color_aug],
                outputs=[cache_latents],
            )

        (
            sample_every_n_steps,
            sample_every_n_epochs,
            sample_sampler,
            sample_prompts,
        ) = sample_gradio_config()

    with gr.Tab('Tools'):
        gr.Markdown(
            'This section provide Dreambooth tools to help setup your dataset...'
        )
        gradio_dreambooth_folder_creation_tab(
            train_data_dir_input=train_data_dir,
            reg_data_dir_input=reg_data_dir,
            output_dir_input=output_dir,
            logging_dir_input=logging_dir,
        )

    button_run = gr.Button('Train model', variant='primary')

    # Setup gradio tensorboard buttons
    button_start_tensorboard, button_stop_tensorboard = gradio_tensorboard()

    button_start_tensorboard.click(
        start_tensorboard,
        inputs=logging_dir,
        show_progress=False,
    )

    button_stop_tensorboard.click(
        stop_tensorboard,
        show_progress=False,
    )

    settings_list = [
        pretrained_model_name_or_path,
        v2,
        v_parameterization,
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
        caption_extension,
        enable_bucket,
        gradient_checkpointing,
        full_fp16,
        no_token_padding,
        stop_text_encoder_training,
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
        keep_tokens,
        persistent_data_loader_workers,
        bucket_no_upscale,
        random_crop,
        bucket_reso_steps,
        caption_dropout_every_n_epochs,
        caption_dropout_rate,
        optimizer,
        optimizer_args,
        noise_offset,
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_sampler,
        sample_prompts,
        additional_parameters,
        vae_batch_size,
        min_snr_gamma,
        weighted_captions,
    ]

    button_open_config.click(
        open_configuration,
        inputs=[dummy_db_true, config_file_name] + settings_list,
        outputs=[config_file_name] + settings_list,
        show_progress=False,
    )

    button_load_config.click(
        open_configuration,
        inputs=[dummy_db_false, config_file_name] + settings_list,
        outputs=[config_file_name] + settings_list,
        show_progress=False,
    )

    button_save_config.click(
        save_configuration,
        inputs=[dummy_db_false, config_file_name] + settings_list,
        outputs=[config_file_name],
        show_progress=False,
    )

    button_save_as_config.click(
        save_configuration,
        inputs=[dummy_db_true, config_file_name] + settings_list,
        outputs=[config_file_name],
        show_progress=False,
    )

    button_run.click(
        train_model,
        inputs=settings_list,
        show_progress=False,
    )

    return (
        train_data_dir,
        reg_data_dir,
        output_dir,
        logging_dir,
    )


def UI(**kwargs):
    css = ''

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            print('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Blocks(css=css)

    with interface:
        with gr.Tab('Dreambooth'):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = dreambooth_tab()
        with gr.Tab('Utilities'):
            utilities_tab(
                train_data_dir_input=train_data_dir_input,
                reg_data_dir_input=reg_data_dir_input,
                output_dir_input=output_dir_input,
                logging_dir_input=logging_dir_input,
                enable_copy_info_button=True,
            )

    # Show the interface
    launch_kwargs = {}
    if not kwargs.get('username', None) == '':
        launch_kwargs['auth'] = (
            kwargs.get('username', None),
            kwargs.get('password', None),
        )
    if kwargs.get('server_port', 0) > 0:
        launch_kwargs['server_port'] = kwargs.get('server_port', 0)
    if kwargs.get('inbrowser', False):
        launch_kwargs['inbrowser'] = kwargs.get('inbrowser', False)
    print(launch_kwargs)
    interface.launch(**launch_kwargs)


if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)
    parser = argparse.ArgumentParser()
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

    args = parser.parse_args()

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
    )
