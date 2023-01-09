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
import shutil
import argparse
from library.common_gui import (
    get_folder_path,
    remove_doublequote,
    get_file_path,
    get_any_file_path,
    get_saveasfile_path,
    color_aug_changed,
    save_inference_file,
    set_pretrained_model_name_or_path_input,
)
from library.dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from library.dataset_balancing_gui import gradio_dataset_balancing_tab
from library.utilities import utilities_tab
from library.merge_lora_gui import gradio_merge_lora_tab
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
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latent,
    caption_extention,
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
    pretrained_model_name_or_path_input,
    v2_input,
    v_parameterization_input,
    logging_dir_input,
    train_data_dir_input,
    reg_data_dir_input,
    output_dir_input,
    max_resolution_input,
    lr_scheduler_input,
    lr_warmup_input,
    train_batch_size_input,
    epoch_input,
    save_every_n_epochs_input,
    mixed_precision_input,
    save_precision_input,
    seed_input,
    num_cpu_threads_per_process_input,
    cache_latent_input,
    caption_extention_input,
    enable_bucket_input,
    gradient_checkpointing,
    full_fp16_input,
    no_token_padding_input,
    stop_text_encoder_training_input,
    use_8bit_adam_input,
    xformers_input,
    save_model_as_dropdown,
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
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    original_file_path = file_path
    file_path = get_file_path(file_path)

    if not file_path == '' and not file_path == None:
        # load variables from JSON file
        with open(file_path, 'r') as f:
            my_data_lora = json.load(f)
            print("Loading config...")
    else:
        file_path = original_file_path  # In case a file_path was provided and the user decide to cancel the open action
        my_data_lora = {}
    
    values = [file_path]
    for key, value in parameters:
        # Set the value in the dictionary to the corresponding value in `my_data`, or the default value if not found
        if not key in ['file_path']:
            values.append(my_data_lora.get(key, value))
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
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latent,
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

    # If string is empty set string to 0.
    if text_encoder_lr == '':
        text_encoder_lr = 0
    if unet_lr == '':
        unet_lr = 0

    if (float(text_encoder_lr) == 0) and (float(unet_lr) == 0):
        msgbox(
            'At least one Learning Rate value for "Text encoder" or "Unet" need to be provided'
        )
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
    if v2:
        run_cmd += ' --v2'
    if v_parameterization:
        run_cmd += ' --v_parameterization'
    if cache_latent:
        run_cmd += ' --cache_latents'
    if enable_bucket:
        run_cmd += ' --enable_bucket'
    if gradient_checkpointing:
        run_cmd += ' --gradient_checkpointing'
    if full_fp16:
        run_cmd += ' --full_fp16'
    if no_token_padding:
        run_cmd += ' --no_token_padding'
    if use_8bit_adam:
        run_cmd += ' --use_8bit_adam'
    if xformers:
        run_cmd += ' --xformers'
    if shuffle_caption:
        run_cmd += ' --shuffle_caption'
    if save_state:
        run_cmd += ' --save_state'
    if color_aug:
        run_cmd += ' --color_aug'
    if flip_aug:
        run_cmd += ' --flip_aug'
    if mem_eff_attn:
        run_cmd += ' --mem_eff_attn'
    run_cmd += (
        f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'
    )
    run_cmd += f' --train_data_dir="{train_data_dir}"'
    if len(reg_data_dir):
        run_cmd += f' --reg_data_dir="{reg_data_dir}"'
    run_cmd += f' --resolution={max_resolution}'
    run_cmd += f' --output_dir="{output_dir}"'
    run_cmd += f' --train_batch_size={train_batch_size}'
    # run_cmd += f' --learning_rate={learning_rate}'
    run_cmd += f' --lr_scheduler={lr_scheduler}'
    run_cmd += f' --lr_warmup_steps={lr_warmup_steps}'
    run_cmd += f' --max_train_steps={max_train_steps}'
    run_cmd += f' --use_8bit_adam'
    run_cmd += f' --xformers'
    run_cmd += f' --mixed_precision={mixed_precision}'
    run_cmd += f' --save_every_n_epochs={save_every_n_epochs}'
    run_cmd += f' --seed={seed}'
    run_cmd += f' --save_precision={save_precision}'
    run_cmd += f' --logging_dir="{logging_dir}"'
    if not caption_extension == '':
        run_cmd += f' --caption_extension={caption_extension}'
    if not stop_text_encoder_training == 0:
        run_cmd += (
            f' --stop_text_encoder_training={stop_text_encoder_training}'
        )
    if not save_model_as == 'same as source model':
        run_cmd += f' --save_model_as={save_model_as}'
    if not resume == '':
        run_cmd += f' --resume="{resume}"'
    if not float(prior_loss_weight) == 1.0:
        run_cmd += f' --prior_loss_weight={prior_loss_weight}'
    run_cmd += f' --network_module=networks.lora'
    if not float(text_encoder_lr) == 0:
        run_cmd += f' --text_encoder_lr={text_encoder_lr}'
    else:
        run_cmd += f' --network_train_unet_only'
    if not float(unet_lr) == 0:
        run_cmd += f' --unet_lr={unet_lr}'
    else:
        run_cmd += f' --network_train_text_encoder_only'
    # if network_train == 'Text encoder only':
    #     run_cmd += f' --network_train_text_encoder_only'
    # elif network_train == 'Unet only':
    #     run_cmd += f' --network_train_unet_only'
    run_cmd += f' --network_dim={network_dim}'
    if not lora_network_weights == '':
        run_cmd += f' --network_weights="{lora_network_weights}"'
    if int(clip_skip) > 1:
        run_cmd += f' --clip_skip={str(clip_skip)}'
    if int(gradient_accumulation_steps) > 1:
        run_cmd += f' --gradient_accumulation_steps={int(gradient_accumulation_steps)}'
    # if not vae == '':
    #     run_cmd += f' --vae="{vae}"'
    if not output_name == '':
        run_cmd += f' --output_name="{output_name}"'

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
    if not username == '':
        interface.launch(auth=(username, password))
    else:
        interface.launch()


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
    with gr.Accordion('Configuration file', open=False):
        with gr.Row():
            button_open_config = gr.Button('Open 📂', elem_id='open_folder')
            button_save_config = gr.Button('Save 💾', elem_id='open_folder')
            button_save_as_config = gr.Button(
                'Save as... 💾', elem_id='open_folder'
            )
            config_file_name = gr.Textbox(
                label='',
                placeholder="type the configuration file path or use the 'Open' button above to select it...",
                interactive=True,
            )
        # config_file_name.change(
        #     remove_doublequote,
        #     inputs=[config_file_name],
        #     outputs=[config_file_name],
        # )
    with gr.Tab('Source model'):
        # Define the input elements
        with gr.Row():
            pretrained_model_name_or_path_input = gr.Textbox(
                label='Pretrained model name or path',
                placeholder='enter the path to custom model or name of pretrained model',
            )
            pretrained_model_name_or_path_file = gr.Button(
                document_symbol, elem_id='open_folder_small'
            )
            pretrained_model_name_or_path_file.click(
                get_any_file_path,
                inputs=[pretrained_model_name_or_path_input],
                outputs=pretrained_model_name_or_path_input,
            )
            pretrained_model_name_or_path_folder = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            pretrained_model_name_or_path_folder.click(
                get_folder_path,
                outputs=pretrained_model_name_or_path_input,
            )
            model_list = gr.Dropdown(
                label='(Optional) Model Quick Pick',
                choices=[
                    'custom',
                    'stabilityai/stable-diffusion-2-1-base',
                    'stabilityai/stable-diffusion-2-base',
                    'stabilityai/stable-diffusion-2-1',
                    'stabilityai/stable-diffusion-2',
                    'runwayml/stable-diffusion-v1-5',
                    'CompVis/stable-diffusion-v1-4',
                ],
            )
            save_model_as_dropdown = gr.Dropdown(
                label='Save trained model as',
                choices=[
                    'same as source model',
                    'ckpt',
                    'diffusers',
                    'diffusers_safetensors',
                    'safetensors',
                ],
                value='same as source model',
            )

        with gr.Row():
            v2_input = gr.Checkbox(label='v2', value=True)
            v_parameterization_input = gr.Checkbox(
                label='v_parameterization', value=False
            )
        pretrained_model_name_or_path_input.change(
            remove_doublequote,
            inputs=[pretrained_model_name_or_path_input],
            outputs=[pretrained_model_name_or_path_input],
        )
        model_list.change(
            set_pretrained_model_name_or_path_input,
            inputs=[model_list, v2_input, v_parameterization_input],
            outputs=[
                pretrained_model_name_or_path_input,
                v2_input,
                v_parameterization_input,
            ],
        )

    with gr.Tab('Folders'):
        with gr.Row():
            train_data_dir_input = gr.Textbox(
                label='Image folder',
                placeholder='Folder where the training folders containing the images are located',
            )
            train_data_dir_input_folder = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            train_data_dir_input_folder.click(
                get_folder_path, outputs=train_data_dir_input
            )
            reg_data_dir_input = gr.Textbox(
                label='Regularisation folder',
                placeholder='(Optional) Folder where where the regularization folders containing the images are located',
            )
            reg_data_dir_input_folder = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            reg_data_dir_input_folder.click(
                get_folder_path, outputs=reg_data_dir_input
            )
        with gr.Row():
            output_dir_input = gr.Textbox(
                label='Output folder',
                placeholder='Folder to output trained model',
            )
            output_dir_input_folder = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            output_dir_input_folder.click(
                get_folder_path, outputs=output_dir_input
            )
            logging_dir_input = gr.Textbox(
                label='Logging folder',
                placeholder='Optional: enable logging and output TensorBoard log to this folder',
            )
            logging_dir_input_folder = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            logging_dir_input_folder.click(
                get_folder_path, outputs=logging_dir_input
            )
        with gr.Row():
            output_name = gr.Textbox(
                label='Model output name',
                placeholder='Name of the model to output',
                value='last',
                interactive=True,
            )
        train_data_dir_input.change(
            remove_doublequote,
            inputs=[train_data_dir_input],
            outputs=[train_data_dir_input],
        )
        reg_data_dir_input.change(
            remove_doublequote,
            inputs=[reg_data_dir_input],
            outputs=[reg_data_dir_input],
        )
        output_dir_input.change(
            remove_doublequote,
            inputs=[output_dir_input],
            outputs=[output_dir_input],
        )
        logging_dir_input.change(
            remove_doublequote,
            inputs=[logging_dir_input],
            outputs=[logging_dir_input],
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
        with gr.Row():
            lr_scheduler_input = gr.Dropdown(
                label='LR Scheduler',
                choices=[
                    'constant',
                    'constant_with_warmup',
                    'cosine',
                    'cosine_with_restarts',
                    'linear',
                    'polynomial',
                ],
                value='cosine',
            )
            lr_warmup_input = gr.Textbox(label='LR warmup (% of steps)', value=10)
        with gr.Row():
            text_encoder_lr = gr.Textbox(
                label='Text Encoder learning rate',
                value="5e-5",
                placeholder='Optional',
            )
            unet_lr = gr.Textbox(
                label='Unet learning rate', value="1e-3", placeholder='Optional'
            )
            network_dim = gr.Slider(
                minimum=1,
                maximum=128,
                label='Network Dimension',
                value=8,
                step=1,
                interactive=True,
            )
        with gr.Row():
            train_batch_size_input = gr.Slider(
                minimum=1,
                maximum=32,
                label='Train batch size',
                value=1,
                step=1,
            )
            epoch_input = gr.Textbox(label='Epoch', value=1)
            save_every_n_epochs_input = gr.Textbox(
                label='Save every N epochs', value=1
            )
        with gr.Row():
            mixed_precision_input = gr.Dropdown(
                label='Mixed precision',
                choices=[
                    'no',
                    'fp16',
                    'bf16',
                ],
                value='fp16',
            )
            save_precision_input = gr.Dropdown(
                label='Save precision',
                choices=[
                    'float',
                    'fp16',
                    'bf16',
                ],
                value='fp16',
            )
            num_cpu_threads_per_process_input = gr.Slider(
                minimum=1,
                maximum=os.cpu_count(),
                step=1,
                label='Number of CPU threads per process',
                value=os.cpu_count(),
            )
        with gr.Row():
            seed_input = gr.Textbox(label='Seed', value=1234)
            max_resolution_input = gr.Textbox(
                label='Max resolution',
                value='512,512',
                placeholder='512,512',
            )
        with gr.Row():
            caption_extention_input = gr.Textbox(
                label='Caption Extension',
                placeholder='(Optional) Extension for caption files. default: .caption',
            )
            stop_text_encoder_training_input = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                step=1,
                label='Stop text encoder training',
            )
        with gr.Row():
            enable_bucket_input = gr.Checkbox(
                label='Enable buckets', value=True
            )
            cache_latent_input = gr.Checkbox(label='Cache latent', value=True)
            use_8bit_adam_input = gr.Checkbox(
                label='Use 8bit adam', value=True
            )
            xformers_input = gr.Checkbox(label='Use xformers', value=True)
        with gr.Accordion('Advanced Configuration', open=False):
            with gr.Row():
                full_fp16_input = gr.Checkbox(
                    label='Full fp16 training (experimental)', value=False
                )
                no_token_padding_input = gr.Checkbox(
                    label='No token padding', value=False
                )

                gradient_checkpointing = gr.Checkbox(
                    label='Gradient checkpointing', value=False
                )
                gradient_accumulation_steps = gr.Number(
                    label='Gradient accumulate steps', value='1'
                )

                shuffle_caption = gr.Checkbox(
                    label='Shuffle caption', value=False
                )
            with gr.Row():
                prior_loss_weight = gr.Number(
                    label='Prior loss weight', value=1.0
                )
                color_aug = gr.Checkbox(
                    label='Color augmentation', value=False
                )
                flip_aug = gr.Checkbox(label='Flip augmentation', value=False)
                color_aug.change(
                    color_aug_changed,
                    inputs=[color_aug],
                    outputs=[cache_latent_input],
                )
                clip_skip = gr.Slider(
                    label='Clip skip', value='1', minimum=1, maximum=12, step=1
                )
                mem_eff_attn = gr.Checkbox(
                    label='Memory efficient attention', value=False
                )
            with gr.Row():
                save_state = gr.Checkbox(
                    label='Save training state', value=False
                )
                resume = gr.Textbox(
                    label='Resume from saved training state',
                    placeholder='path to "last-state" state folder to resume from',
                )
                resume_button = gr.Button('📂', elem_id='open_folder_small')
                resume_button.click(get_folder_path, outputs=resume)
                # vae = gr.Textbox(
                #     label='VAE',
                #     placeholder='(Optiona) path to checkpoint of vae to replace for training',
                # )
                # vae_button = gr.Button('📂', elem_id='open_folder_small')
                # vae_button.click(get_any_file_path, outputs=vae)
    with gr.Tab('Tools'):
        gr.Markdown(
            'This section provide Dreambooth tools to help setup your dataset...'
        )
        gradio_dreambooth_folder_creation_tab(
            train_data_dir_input=train_data_dir_input,
            reg_data_dir_input=reg_data_dir_input,
            output_dir_input=output_dir_input,
            logging_dir_input=logging_dir_input,
        )
        gradio_dataset_balancing_tab()
        gradio_merge_lora_tab()

    button_run = gr.Button('Train model')

    settings_list = [
        pretrained_model_name_or_path_input,
        v2_input,
        v_parameterization_input,
        logging_dir_input,
        train_data_dir_input,
        reg_data_dir_input,
        output_dir_input,
        max_resolution_input,
        lr_scheduler_input,
        lr_warmup_input,
        train_batch_size_input,
        epoch_input,
        save_every_n_epochs_input,
        mixed_precision_input,
        save_precision_input,
        seed_input,
        num_cpu_threads_per_process_input,
        cache_latent_input,
        caption_extention_input,
        enable_bucket_input,
        gradient_checkpointing,
        full_fp16_input,
        no_token_padding_input,
        stop_text_encoder_training_input,
        use_8bit_adam_input,
        xformers_input,
        save_model_as_dropdown,
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
        train_data_dir_input,
        reg_data_dir_input,
        output_dir_input,
        logging_dir_input,
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
