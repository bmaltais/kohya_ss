import gradio as gr
import json
import math
import os
import subprocess
import pathlib
import argparse
from library.common_gui import (
    get_folder_path,
    get_file_path,
    get_saveasfile_path,
    save_inference_file,
    gradio_advanced_training,
    run_cmd_advanced_training,
    gradio_training,
    run_cmd_advanced_training,
    gradio_config,
    gradio_source_model,
    color_aug_changed,
    run_cmd_training,
    # set_legacy_8bitadam,
    update_my_data,
    check_if_model_exist,
)
from library.tensorboard_gui import (
    gradio_tensorboard,
    start_tensorboard,
    stop_tensorboard,
)
from library.utilities import utilities_tab
from library.sampler_gui import sample_gradio_config, run_cmd_sample
from easygui import msgbox

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
    noise_offset,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,weighted_captions,
):
    if check_if_model_exist(output_name, output_dir, save_model_as):
        return
    
    if optimizer == 'Adafactor' and lr_warmup != '0':
        msgbox("Warning: lr_scheduler is set to 'Adafactor', so 'LR warmup (% of steps)' will be considered 0.", title="Warning")
        lr_warmup = '0'

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

        print(run_cmd)

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

        print(run_cmd)

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
    print(f'image_num = {image_num}')

    repeats = int(image_num) * int(dataset_repeats)
    print(f'repeats = {str(repeats)}')

    # calculate max_train_steps
    max_train_steps = int(
        math.ceil(float(repeats) / int(train_batch_size) * int(epoch))
    )

    # Divide by two because flip augmentation create two copied of the source images
    if flip_aug:
        max_train_steps = int(math.ceil(float(max_train_steps) / 2))

    print(f'max_train_steps = {max_train_steps}')

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    print(f'lr_warmup_steps = {lr_warmup_steps}')

    run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process} "./fine_tune.py"'
    if v2:
        run_cmd += ' --v2'
    if v_parameterization:
        run_cmd += ' --v_parameterization'
    if train_text_encoder:
        run_cmd += ' --train_text_encoder'
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
    run_cmd += f' --resolution={max_resolution}'
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


def remove_doublequote(file_path):
    if file_path != None:
        file_path = file_path.replace('"', '')

    return file_path


def finetune_tab():
    dummy_db_true = gr.Label(value=True, visible=False)
    dummy_db_false = gr.Label(value=False, visible=False)
    gr.Markdown('Train a custom model using kohya finetune python code...')

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
            train_dir = gr.Textbox(
                label='Training config folder',
                placeholder='folder where the training configuration files will be saved',
            )
            train_dir_folder = gr.Button(
                folder_symbol, elem_id='open_folder_small'
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
                folder_symbol, elem_id='open_folder_small'
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
                folder_symbol, elem_id='open_folder_small'
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
                folder_symbol, elem_id='open_folder_small'
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
        ) = gradio_training(learning_rate_value='1e-5')
        with gr.Row():
            dataset_repeats = gr.Textbox(label='Dataset repeats', value=40)
            train_text_encoder = gr.Checkbox(
                label='Train text encoder', value=True
            )
        with gr.Accordion('Advanced parameters', open=False):
            with gr.Row():
                gradient_accumulation_steps = gr.Number(
                    label='Gradient accumulate steps', value='1'
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
                outputs=[cache_latents],  # Not applicable to fine_tune.py
            )

        (
            sample_every_n_steps,
            sample_every_n_epochs,
            sample_sampler,
            sample_prompts,
        ) = sample_gradio_config()

    button_run = gr.Button('Train model', variant='primary')

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
        pretrained_model_name_or_path,
        v2,
        v_parameterization,
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

    button_run.click(train_model, inputs=settings_list)

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


def UI(**kwargs):

    css = ''

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            print('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Blocks(css=css)

    with interface:
        with gr.Tab('Finetune'):
            finetune_tab()
        with gr.Tab('Utilities'):
            utilities_tab(enable_dreambooth_tab=False)

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
