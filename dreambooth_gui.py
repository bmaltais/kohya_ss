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
)
from library.dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from library.utilities import utilities_tab
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
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    original_file_path = file_path
    file_path = get_file_path(file_path)

    if not file_path == '' and not file_path == None:
        # load variables from JSON file
        with open(file_path, 'r') as f:
            my_data_db = json.load(f)
            print('Loading config...')
    else:
        file_path = original_file_path  # In case a file_path was provided and the user decide to cancel the open action
        my_data_db = {}

    values = [file_path]
    for key, value in parameters:
        # Set the value in the dictionary to the corresponding value in `my_data`, or the default value if not found
        if not key in ['file_path']:
            values.append(my_data_db.get(key, value))
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

    # Print the result
    # print(f"{total_steps} total steps")

    if reg_data_dir == '':
        reg_factor = 1
    else:
        print(
            'Regularisation images are used... Will double the number of steps required...'
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
    if stop_text_encoder_training_pct == None:
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
    )

    print(run_cmd)
    # Run the command
    subprocess.run(run_cmd)

    # check if output_dir/last is a folder... therefore it is a diffuser model
    last_dir = pathlib.Path(f'{output_dir}/{output_name}')

    if not last_dir.is_dir():
        # Copy inference model for v2 if required
        save_inference_file(output_dir, v2, v_parameterization, output_name)


def UI(username, password):
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
    if not username == '':
        interface.launch(auth=(username, password))
    else:
        interface.launch()


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
                get_folder_path, outputs=train_data_dir
            )
            reg_data_dir = gr.Textbox(
                label='Regularisation folder',
                placeholder='(Optional) Folder where where the regularization folders containing the images are located',
            )
            reg_data_dir_input_folder = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            reg_data_dir_input_folder.click(
                get_folder_path, outputs=reg_data_dir
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
                get_folder_path, outputs=logging_dir
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
                vae = gr.Textbox(
                    label='VAE',
                    placeholder='(Optiona) path to checkpoint of vae to replace for training',
                )
                vae_button = gr.Button('📂', elem_id='open_folder_small')
                vae_button.click(get_any_file_path, outputs=vae)
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
            ) = gradio_advanced_training()
            color_aug.change(
                color_aug_changed,
                inputs=[color_aug],
                outputs=[cache_latents],
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

    button_run = gr.Button('Train model')

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


if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )

    args = parser.parse_args()

    UI(username=args.username, password=args.password)
