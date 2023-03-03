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
    gradio_training,
    gradio_config,
    gradio_source_model,
    run_cmd_training,
    set_legacy_8bitadam,
    update_optimizer,
)
from library.dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from library.tensorboard_gui import (
    gradio_tensorboard,
    start_tensorboard,
    stop_tensorboard,
)
from library.dataset_balancing_gui import gradio_dataset_balancing_tab
from library.utilities import utilities_tab
from library.merge_lora_gui import gradio_merge_lora_tab
from library.verify_lora_gui import gradio_verify_lora_tab
from library.resize_lora_gui import gradio_resize_lora_tab
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
    use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    lora_network_weights,
    color_aug,
    flip_aug,
    clip_skip,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    network_alpha,
    training_comment,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs, caption_dropout_rate,
    optimizer,
    optimizer_args,noise_offset,
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
    use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    lora_network_weights,
    color_aug,
    flip_aug,
    clip_skip,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    network_alpha,
    training_comment,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs, caption_dropout_rate,
    optimizer,
    optimizer_args,noise_offset,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    original_file_path = file_path
    file_path = get_file_path(file_path)

    if not file_path == '' and not file_path == None:
        # load variables from JSON file
        with open(file_path, 'r') as f:
            my_data = json.load(f)
            print('Loading config...')
            # Update values to fix deprecated use_8bit_adam checkbox and set appropriate optimizer if it is set to True
            my_data = update_optimizer(my_data)
    else:
        file_path = original_file_path  # In case a file_path was provided and the user decide to cancel the open action
        my_data = {}

    values = [file_path]
    for key, value in parameters:
        # Set the value in the dictionary to the corresponding value in `my_data`, or the default value if not found
        if not key in ['file_path']:
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
    use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    lora_network_weights,
    color_aug,
    flip_aug,
    clip_skip,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,  # Keep this. Yes, it is unused here but required given the common list used
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    network_alpha,
    training_comment,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs, caption_dropout_rate,
    optimizer,
    optimizer_args,noise_offset,
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

    if int(bucket_reso_steps) < 1:
        msgbox('Bucket resolution steps need to be greater than 0')
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if stop_text_encoder_training_pct > 0:
        msgbox(
            'Output "stop text encoder training" is not yet supported. Ignoring'
        )
        stop_text_encoder_training_pct = 0

    # If string is empty set string to 0.
    if text_encoder_lr == '':
        text_encoder_lr = 0
    if unet_lr == '':
        unet_lr = 0

    # if (float(text_encoder_lr) == 0) and (float(unet_lr) == 0):
    #     msgbox(
    #         'At least one Learning Rate value for "Text encoder" or "Unet" need to be provided'
    #     )
    #     return

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
                for f in os.listdir(os.path.join(train_data_dir, folder))
                if f.endswith('.jpg')
                or f.endswith('.jpeg')
                or f.endswith('.png')
                or f.endswith('.webp')
            ]
        )

        # Calculate the total number of steps for this folder
        steps = repeats * num_images
        total_steps += steps

        # Print the result
        print(f'Folder {folder}: {steps} steps')

    # calculate max_train_steps
    max_train_steps = int(
        math.ceil(
            float(total_steps)
            / int(train_batch_size)
            * int(epoch)
            # * int(reg_factor)
        )
    )
    print(f'max_train_steps = {max_train_steps}')

    # calculate stop encoder training
    if stop_text_encoder_training_pct == None:
        stop_text_encoder_training = 0
    else:
        stop_text_encoder_training = math.ceil(
            float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
        )
    print(f'stop_text_encoder_training = {stop_text_encoder_training}')

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    print(f'lr_warmup_steps = {lr_warmup_steps}')

    run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process} "train_network.py"'

    # run_cmd += f' --caption_dropout_rate="0.1" --caption_dropout_every_n_epochs=1'   # --random_crop'

    if v2:
        run_cmd += ' --v2'
    if v_parameterization:
        run_cmd += ' --v_parameterization'
    if enable_bucket:
        run_cmd += ' --enable_bucket'
    if no_token_padding:
        run_cmd += ' --no_token_padding'
    run_cmd += (
        f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'
    )
    run_cmd += f' --train_data_dir="{train_data_dir}"'
    if len(reg_data_dir):
        run_cmd += f' --reg_data_dir="{reg_data_dir}"'
    run_cmd += f' --resolution={max_resolution}'
    run_cmd += f' --output_dir="{output_dir}"'
    run_cmd += f' --logging_dir="{logging_dir}"'
    run_cmd += f' --network_alpha="{network_alpha}"'
    if not training_comment == '':
        run_cmd += f' --training_comment="{training_comment}"'
    if not stop_text_encoder_training == 0:
        run_cmd += (
            f' --stop_text_encoder_training={stop_text_encoder_training}'
        )
    if not save_model_as == 'same as source model':
        run_cmd += f' --save_model_as={save_model_as}'
    if not float(prior_loss_weight) == 1.0:
        run_cmd += f' --prior_loss_weight={prior_loss_weight}'
    run_cmd += f' --network_module=networks.lora'

    if not (float(text_encoder_lr) == 0) or not (float(unet_lr) == 0):
        if not (float(text_encoder_lr) == 0) and not (float(unet_lr) == 0):
            run_cmd += f' --text_encoder_lr={text_encoder_lr}'
            run_cmd += f' --unet_lr={unet_lr}'
        elif not (float(text_encoder_lr) == 0):
            run_cmd += f' --text_encoder_lr={text_encoder_lr}'
            run_cmd += f' --network_train_text_encoder_only'
        else:
            run_cmd += f' --unet_lr={unet_lr}'
            run_cmd += f' --network_train_unet_only'
    else:
        if float(text_encoder_lr) == 0:
            msgbox('Please input learning rate values.')
            return

    run_cmd += f' --network_dim={network_dim}'

    if not lora_network_weights == '':
        run_cmd += f' --network_weights="{lora_network_weights}"'
    if int(gradient_accumulation_steps) > 1:
        run_cmd += f' --gradient_accumulation_steps={int(gradient_accumulation_steps)}'
    if not output_name == '':
        run_cmd += f' --output_name="{output_name}"'
    if not lr_scheduler_num_cycles == '':
        run_cmd += f' --lr_scheduler_num_cycles="{lr_scheduler_num_cycles}"'
    else:
        run_cmd += f' --lr_scheduler_num_cycles="{epoch}"'
    if not lr_scheduler_power == '':
        run_cmd += f' --lr_scheduler_power="{lr_scheduler_power}"'

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
        use_8bit_adam=use_8bit_adam,
        keep_tokens=keep_tokens,
        persistent_data_loader_workers=persistent_data_loader_workers,
        bucket_no_upscale=bucket_no_upscale,
        random_crop=random_crop,
        bucket_reso_steps=bucket_reso_steps,
        caption_dropout_every_n_epochs=caption_dropout_every_n_epochs,
        caption_dropout_rate=caption_dropout_rate,
        noise_offset=noise_offset,
    )

    print(run_cmd)
    # Run the command
    os.system(run_cmd)

    # check if output_dir/last is a folder... therefore it is a diffuser model
    last_dir = pathlib.Path(f'{output_dir}/{output_name}')

    if not last_dir.is_dir():
        # Copy inference model for v2 if required
        save_inference_file(output_dir, v2, v_parameterization, output_name)


def lora_tab(
    train_data_dir_input=gr.Textbox(),
    reg_data_dir_input=gr.Textbox(),
    output_dir_input=gr.Textbox(),
    logging_dir_input=gr.Textbox(),
):
    dummy_db_true = gr.Label(value=True, visible=False)
    dummy_db_false = gr.Label(value=False, visible=False)
    gr.Markdown(
        'Train a custom model using kohya train network LoRA python code...'
    )
    (
        button_open_config,
        button_save_config,
        button_save_as_config,
        config_file_name,
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
            train_data_dir_folder = gr.Button('📂', elem_id='open_folder_small')
            train_data_dir_folder.click(
                get_folder_path, outputs=train_data_dir
            )
            reg_data_dir = gr.Textbox(
                label='Regularisation folder',
                placeholder='(Optional) Folder where where the regularization folders containing the images are located',
            )
            reg_data_dir_folder = gr.Button('📂', elem_id='open_folder_small')
            reg_data_dir_folder.click(get_folder_path, outputs=reg_data_dir)
        with gr.Row():
            output_dir = gr.Textbox(
                label='Output folder',
                placeholder='Folder to output trained model',
            )
            output_dir_folder = gr.Button('📂', elem_id='open_folder_small')
            output_dir_folder.click(get_folder_path, outputs=output_dir)
            logging_dir = gr.Textbox(
                label='Logging folder',
                placeholder='Optional: enable logging and output TensorBoard log to this folder',
            )
            logging_dir_folder = gr.Button('📂', elem_id='open_folder_small')
            logging_dir_folder.click(get_folder_path, outputs=logging_dir)
        with gr.Row():
            output_name = gr.Textbox(
                label='Model output name',
                placeholder='(Name of the model to output)',
                value='last',
                interactive=True,
            )
            training_comment = gr.Textbox(
                label='Training comment',
                placeholder='(Optional) Add training comment to be included in metadata',
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
        with gr.Row():
            lora_network_weights = gr.Textbox(
                label='LoRA network weights',
                placeholder='{Optional) Path to existing LoRA network weights to resume training',
            )
            lora_network_weights_file = gr.Button(
                document_symbol, elem_id='open_folder_small'
            )
            lora_network_weights_file.click(
                get_any_file_path,
                inputs=[lora_network_weights],
                outputs=lora_network_weights,
            )
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
            learning_rate_value='0.0001',
            lr_scheduler_value='cosine',
            lr_warmup_value='10',
        )
        with gr.Row():
            text_encoder_lr = gr.Textbox(
                label='Text Encoder learning rate',
                value='5e-5',
                placeholder='Optional',
            )
            unet_lr = gr.Textbox(
                label='Unet learning rate',
                value='0.0001',
                placeholder='Optional',
            )
            network_dim = gr.Slider(
                minimum=4,
                maximum=1024,
                label='Network Rank (Dimension)',
                value=8,
                step=4,
                interactive=True,
            )
            network_alpha = gr.Slider(
                minimum=4,
                maximum=1024,
                label='Network Alpha',
                value=1,
                step=4,
                interactive=True,
            )
        with gr.Row():
            max_resolution = gr.Textbox(
                label='Max resolution',
                value='512,512',
                placeholder='512,512',
            )
            stop_text_encoder_training = gr.Slider(
                minimum=0,
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
            with gr.Row():
                prior_loss_weight = gr.Number(
                    label='Prior loss weight', value=1.0
                )
                lr_scheduler_num_cycles = gr.Textbox(
                    label='LR number of cycles',
                    placeholder='(Optional) For Cosine with restart and polynomial only',
                )

                lr_scheduler_power = gr.Textbox(
                    label='LR power',
                    placeholder='(Optional) For Cosine with restart and polynomial only',
                )
            (
                use_8bit_adam,
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
                caption_dropout_every_n_epochs, caption_dropout_rate,noise_offset,
            ) = gradio_advanced_training()
            color_aug.change(
                color_aug_changed,
                inputs=[color_aug],
                outputs=[cache_latents],
            )
        
        optimizer.change(
            set_legacy_8bitadam,
            inputs=[optimizer, use_8bit_adam],
            outputs=[optimizer, use_8bit_adam],
        )

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
        gradio_dataset_balancing_tab()
        gradio_merge_lora_tab()
        gradio_resize_lora_tab()
        gradio_verify_lora_tab()

    button_run = gr.Button('Train model', variant='primary')
    
    # Setup gradio tensorboard buttons
    button_start_tensorboard, button_stop_tensorboard = gradio_tensorboard()
    
    button_start_tensorboard.click(
        start_tensorboard,
        inputs=logging_dir,
    )
    
    button_stop_tensorboard.click(
        stop_tensorboard,
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
        use_8bit_adam,
        xformers,
        save_model_as,
        shuffle_caption,
        save_state,
        resume,
        prior_loss_weight,
        text_encoder_lr,
        unet_lr,
        network_dim,
        lora_network_weights,
        color_aug,
        flip_aug,
        clip_skip,
        gradient_accumulation_steps,
        mem_eff_attn,
        output_name,
        model_list,
        max_token_length,
        max_train_epochs,
        max_data_loader_n_workers,
        network_alpha,
        training_comment,
        keep_tokens,
        lr_scheduler_num_cycles,
        lr_scheduler_power,
        persistent_data_loader_workers,
        bucket_no_upscale,
        random_crop,
        bucket_reso_steps,
        caption_dropout_every_n_epochs, caption_dropout_rate,
        optimizer,
        optimizer_args,noise_offset,
    ]

    button_open_config.click(
        open_configuration,
        inputs=[config_file_name] + settings_list,
        outputs=[config_file_name] + settings_list,
    )

    button_save_config.click(
        save_configuration,
        inputs=[dummy_db_false, config_file_name] + settings_list,
        outputs=[config_file_name],
    )

    button_save_as_config.click(
        save_configuration,
        inputs=[dummy_db_true, config_file_name] + settings_list,
        outputs=[config_file_name],
    )

    button_run.click(
        train_model,
        inputs=settings_list,
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
        with gr.Tab('LoRA'):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = lora_tab()
        with gr.Tab('Utilities'):
            utilities_tab(
                train_data_dir_input=train_data_dir_input,
                reg_data_dir_input=reg_data_dir_input,
                output_dir_input=output_dir_input,
                logging_dir_input=logging_dir_input,
                enable_copy_info_button=True,
            )

    # Show the interface
    launch_kwargs={}
    if not kwargs.get('username', None) == '':
        launch_kwargs["auth"] = (kwargs.get('username', None), kwargs.get('password', None))
    if kwargs.get('server_port', 0) > 0:
        launch_kwargs["server_port"] = kwargs.get('server_port', 0)
    if kwargs.get('inbrowser', False):        
        launch_kwargs["inbrowser"] = kwargs.get('inbrowser', False)
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
        '--server_port', type=int, default=0, help='Port to run the server listener on'
    )
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")

    args = parser.parse_args()

    UI(username=args.username, password=args.password, inbrowser=args.inbrowser, server_port=args.server_port)