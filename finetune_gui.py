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
    get_file_path,
    get_any_file_path,
    get_saveasfile_path,
)
from library.utilities import utilities_tab

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
    use_8bit_adam,
    xformers,
    clip_skip,
):
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

    if file_path == None:
        return original_file_path

    # Return the values of the variables as a dictionary
    variables = {
        'pretrained_model_name_or_path': pretrained_model_name_or_path,
        'v2': v2,
        'v_parameterization': v_parameterization,
        'train_dir': train_dir,
        'image_folder': image_folder,
        'output_dir': output_dir,
        'logging_dir': logging_dir,
        'max_resolution': max_resolution,
        'min_bucket_reso': min_bucket_reso,
        'max_bucket_reso': max_bucket_reso,
        'batch_size': batch_size,
        'flip_aug': flip_aug,
        'caption_metadata_filename': caption_metadata_filename,
        'latent_metadata_filename': latent_metadata_filename,
        'full_path': full_path,
        'learning_rate': learning_rate,
        'lr_scheduler': lr_scheduler,
        'lr_warmup': lr_warmup,
        'dataset_repeats': dataset_repeats,
        'train_batch_size': train_batch_size,
        'epoch': epoch,
        'save_every_n_epochs': save_every_n_epochs,
        'mixed_precision': mixed_precision,
        'save_precision': save_precision,
        'seed': seed,
        'num_cpu_threads_per_process': num_cpu_threads_per_process,
        'train_text_encoder': train_text_encoder,
        'create_buckets': create_buckets,
        'create_caption': create_caption,
        'save_model_as': save_model_as,
        'caption_extension': caption_extension,
        'use_8bit_adam': use_8bit_adam,
        'xformers': xformers,
        'clip_skip': clip_skip,
    }

    # Save the data to the selected file
    with open(file_path, 'w') as file:
        json.dump(variables, file)

    return file_path


def open_config_file(
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
    use_8bit_adam,
    xformers,
    clip_skip,
):
    original_file_path = file_path
    file_path = get_file_path(file_path)

    if file_path != '' and file_path != None:
        print(file_path)
        # load variables from JSON file
        with open(file_path, 'r') as f:
            my_data = json.load(f)
    else:
        file_path = original_file_path   # In case a file_path was provided and the user decide to cancel the open action
        my_data = {}

    # Return the values of the variables as a dictionary
    return (
        file_path,
        my_data.get(
            'pretrained_model_name_or_path', pretrained_model_name_or_path
        ),
        my_data.get('v2', v2),
        my_data.get('v_parameterization', v_parameterization),
        my_data.get('train_dir', train_dir),
        my_data.get('image_folder', image_folder),
        my_data.get('output_dir', output_dir),
        my_data.get('logging_dir', logging_dir),
        my_data.get('max_resolution', max_resolution),
        my_data.get('min_bucket_reso', min_bucket_reso),
        my_data.get('max_bucket_reso', max_bucket_reso),
        my_data.get('batch_size', batch_size),
        my_data.get('flip_aug', flip_aug),
        my_data.get('caption_metadata_filename', caption_metadata_filename),
        my_data.get('latent_metadata_filename', latent_metadata_filename),
        my_data.get('full_path', full_path),
        my_data.get('learning_rate', learning_rate),
        my_data.get('lr_scheduler', lr_scheduler),
        my_data.get('lr_warmup', lr_warmup),
        my_data.get('dataset_repeats', dataset_repeats),
        my_data.get('train_batch_size', train_batch_size),
        my_data.get('epoch', epoch),
        my_data.get('save_every_n_epochs', save_every_n_epochs),
        my_data.get('mixed_precision', mixed_precision),
        my_data.get('save_precision', save_precision),
        my_data.get('seed', seed),
        my_data.get(
            'num_cpu_threads_per_process', num_cpu_threads_per_process
        ),
        my_data.get('train_text_encoder', train_text_encoder),
        my_data.get('create_buckets', create_buckets),
        my_data.get('create_caption', create_caption),
        my_data.get('save_model_as', save_model_as),
        my_data.get('caption_extension', caption_extension),
        my_data.get('use_8bit_adam', use_8bit_adam),
        my_data.get('xformers', xformers),
        my_data.get('clip_skip', clip_skip),
    )


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
    use_8bit_adam,
    xformers,
    clip_skip,
):
    def save_inference_file(output_dir, v2, v_parameterization):
        # Copy inference model for v2 if required
        if v2 and v_parameterization:
            print(f'Saving v2-inference-v.yaml as {output_dir}/last.yaml')
            shutil.copy(
                f'./v2_inference/v2-inference-v.yaml',
                f'{output_dir}/last.yaml',
            )
        elif v2:
            print(f'Saving v2-inference.yaml as {output_dir}/last.yaml')
            shutil.copy(
                f'./v2_inference/v2-inference.yaml',
                f'{output_dir}/last.yaml',
            )

    # create caption json file
    if generate_caption_database:
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        run_cmd = (
            f'./venv/Scripts/python.exe finetune/merge_captions_to_metadata.py'
        )
        if caption_extension == '':
            run_cmd += f' --caption_extension=".txt"'
        else:
            run_cmd += f' --caption_extension={caption_extension}'
        run_cmd += f' "{image_folder}"'
        run_cmd += f' "{train_dir}/{caption_metadata_filename}"'
        if full_path:
            run_cmd += f' --full_path'

        print(run_cmd)

        # Run the command
        subprocess.run(run_cmd)

    # create images buckets
    if generate_image_buckets:
        run_cmd = (
            f'./venv/Scripts/python.exe finetune/prepare_buckets_latents.py'
        )
        run_cmd += f' "{image_folder}"'
        run_cmd += f' "{train_dir}/{caption_metadata_filename}"'
        run_cmd += f' "{train_dir}/{latent_metadata_filename}"'
        run_cmd += f' "{pretrained_model_name_or_path}"'
        run_cmd += f' --batch_size={batch_size}'
        run_cmd += f' --max_resolution={max_resolution}'
        run_cmd += f' --min_bucket_reso={min_bucket_reso}'
        run_cmd += f' --max_bucket_reso={max_bucket_reso}'
        run_cmd += f' --mixed_precision={mixed_precision}'
        if flip_aug:
            run_cmd += f' --flip_aug'
        if full_path:
            run_cmd += f' --full_path'

        print(run_cmd)

        # Run the command
        subprocess.run(run_cmd)

    image_num = len(
        [f for f in os.listdir(image_folder) if f.endswith('.npz')]
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
    if use_8bit_adam:
        run_cmd += f' --use_8bit_adam'
    if xformers:
        run_cmd += f' --xformers'
    run_cmd += (
        f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'
    )
    run_cmd += f' --in_json="{train_dir}/{latent_metadata_filename}"'
    run_cmd += f' --train_data_dir="{image_folder}"'
    run_cmd += f' --output_dir="{output_dir}"'
    if not logging_dir == '':
        run_cmd += f' --logging_dir="{logging_dir}"'
    run_cmd += f' --train_batch_size={train_batch_size}'
    run_cmd += f' --dataset_repeats={dataset_repeats}'
    run_cmd += f' --learning_rate={learning_rate}'
    run_cmd += f' --lr_scheduler={lr_scheduler}'
    run_cmd += f' --lr_warmup_steps={lr_warmup_steps}'
    run_cmd += f' --max_train_steps={max_train_steps}'
    run_cmd += f' --mixed_precision={mixed_precision}'
    run_cmd += f' --save_every_n_epochs={save_every_n_epochs}'
    run_cmd += f' --seed={seed}'
    run_cmd += f' --save_precision={save_precision}'
    if not save_model_as == 'same as source model':
        run_cmd += f' --save_model_as={save_model_as}'
    if int(clip_skip) > 1:
        run_cmd += f' --clip_skip={str(clip_skip)}'

    print(run_cmd)
    # Run the command
    subprocess.run(run_cmd)

    # check if output_dir/last is a folder... therefore it is a diffuser model
    last_dir = pathlib.Path(f'{output_dir}/last')

    if not last_dir.is_dir():
        # Copy inference model for v2 if required
        save_inference_file(output_dir, v2, v_parameterization)


def set_pretrained_model_name_or_path_input(value, v2, v_parameterization):
    # define a list of substrings to search for
    substrings_v2 = [
        'stabilityai/stable-diffusion-2-1-base',
        'stabilityai/stable-diffusion-2-base',
    ]

    # check if $v2 and $v_parameterization are empty and if $pretrained_model_name_or_path contains any of the substrings in the v2 list
    if str(value) in substrings_v2:
        print('SD v2 model detected. Setting --v2 parameter')
        v2 = True
        v_parameterization = False

        return value, v2, v_parameterization

    # define a list of substrings to search for v-objective
    substrings_v_parameterization = [
        'stabilityai/stable-diffusion-2-1',
        'stabilityai/stable-diffusion-2',
    ]

    # check if $v2 and $v_parameterization are empty and if $pretrained_model_name_or_path contains any of the substrings in the v_parameterization list
    if str(value) in substrings_v_parameterization:
        print(
            'SD v2 v_parameterization detected. Setting --v2 parameter and --v_parameterization'
        )
        v2 = True
        v_parameterization = True

        return value, v2, v_parameterization

    # define a list of substrings to v1.x
    substrings_v1_model = [
        'CompVis/stable-diffusion-v1-4',
        'runwayml/stable-diffusion-v1-5',
    ]

    if str(value) in substrings_v1_model:
        v2 = False
        v_parameterization = False

        return value, v2, v_parameterization

    if value == 'custom':
        value = ''
        v2 = False
        v_parameterization = False

        return value, v2, v_parameterization


def remove_doublequote(file_path):
    if file_path != None:
        file_path = file_path.replace('"', '')

    return file_path


def UI(username, password):

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
    if not username == '':
        interface.launch(auth=(username, password))
    else:
        interface.launch()


def finetune_tab():
    dummy_ft_true = gr.Label(value=True, visible=False)
    dummy_ft_false = gr.Label(value=False, visible=False)
    gr.Markdown('Train a custom model using kohya finetune python code...')
    with gr.Accordion('Configuration file', open=False):
        with gr.Row():
            button_open_config = gr.Button(
                f'Open {folder_symbol}', elem_id='open_folder'
            )
            button_save_config = gr.Button(
                f'Save {save_style_symbol}', elem_id='open_folder'
            )
            button_save_as_config = gr.Button(
                f'Save as... {save_style_symbol}',
                elem_id='open_folder',
            )
        config_file_name = gr.Textbox(
            label='', placeholder='type file path or use buttons...'
        )
        config_file_name.change(
            remove_doublequote,
            inputs=[config_file_name],
            outputs=[config_file_name],
        )
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
                inputs=pretrained_model_name_or_path_input,
                outputs=pretrained_model_name_or_path_input,
            )
            pretrained_model_name_or_path_folder = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            pretrained_model_name_or_path_folder.click(
                get_folder_path,
                inputs=pretrained_model_name_or_path_input,
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
            train_dir_input = gr.Textbox(
                label='Training config folder',
                placeholder='folder where the training configuration files will be saved',
            )
            train_dir_folder = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            train_dir_folder.click(get_folder_path, outputs=train_dir_input)

            image_folder_input = gr.Textbox(
                label='Training Image folder',
                placeholder='folder where the training images are located',
            )
            image_folder_input_folder = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            image_folder_input_folder.click(
                get_folder_path, outputs=image_folder_input
            )
        with gr.Row():
            output_dir_input = gr.Textbox(
                label='Output folder',
                placeholder='folder where the model will be saved',
            )
            output_dir_input_folder = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            output_dir_input_folder.click(
                get_folder_path, outputs=output_dir_input
            )

            logging_dir_input = gr.Textbox(
                label='Logging folder',
                placeholder='Optional: enable logging and output TensorBoard log to this folder',
            )
            logging_dir_input_folder = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            logging_dir_input_folder.click(
                get_folder_path, outputs=logging_dir_input
            )
        train_dir_input.change(
            remove_doublequote,
            inputs=[train_dir_input],
            outputs=[train_dir_input],
        )
        image_folder_input.change(
            remove_doublequote,
            inputs=[image_folder_input],
            outputs=[image_folder_input],
        )
        output_dir_input.change(
            remove_doublequote,
            inputs=[output_dir_input],
            outputs=[output_dir_input],
        )
    with gr.Tab('Dataset preparation'):
        with gr.Row():
            max_resolution_input = gr.Textbox(
                label='Resolution (width,height)', value='512,512'
            )
            min_bucket_reso = gr.Textbox(
                label='Min bucket resolution', value='256'
            )
            max_bucket_reso = gr.Textbox(
                label='Max bucket resolution', value='1024'
            )
            batch_size = gr.Textbox(label='Batch size', value='1')
        with gr.Accordion('Advanced parameters', open=False):
            with gr.Row():
                caption_metadata_filename = gr.Textbox(
                    label='Caption metadata filename', value='meta_cap.json'
                )
                latent_metadata_filename = gr.Textbox(
                    label='Latent metadata filename', value='meta_lat.json'
                )
                full_path = gr.Checkbox(label='Use full path', value=True)
                flip_aug = gr.Checkbox(label='Flip augmentation', value=False)
    with gr.Tab('Training parameters'):
        with gr.Row():
            learning_rate_input = gr.Textbox(label='Learning rate', value=1e-6)
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
                value='constant',
            )
            lr_warmup_input = gr.Textbox(label='LR warmup', value=0)
        with gr.Row():
            dataset_repeats_input = gr.Textbox(
                label='Dataset repeats', value=40
            )
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
            seed_input = gr.Textbox(label='Seed', value=1234)
        with gr.Row():
            caption_extention_input = gr.Textbox(
                label='Caption Extension',
                placeholder='(Optional) Extension for caption files. default: .txt',
            )
            train_text_encoder_input = gr.Checkbox(
                label='Train text encoder', value=True
            )
        with gr.Accordion('Advanced parameters', open=False):
            with gr.Row():
                use_8bit_adam = gr.Checkbox(label='Use 8bit adam', value=True)
                xformers = gr.Checkbox(label='Use xformers', value=True)
                clip_skip = gr.Slider(
                    label='Clip skip', value='1', minimum=1, maximum=12, step=1
                )
    with gr.Box():
        with gr.Row():
            create_caption = gr.Checkbox(
                label='Generate caption metadata', value=True
            )
            create_buckets = gr.Checkbox(
                label='Generate image buckets metadata', value=True
            )

    button_run = gr.Button('Train model')

    settings_list = [
        pretrained_model_name_or_path_input,
        v2_input,
        v_parameterization_input,
        train_dir_input,
        image_folder_input,
        output_dir_input,
        logging_dir_input,
        max_resolution_input,
        min_bucket_reso,
        max_bucket_reso,
        batch_size,
        flip_aug,
        caption_metadata_filename,
        latent_metadata_filename,
        full_path,
        learning_rate_input,
        lr_scheduler_input,
        lr_warmup_input,
        dataset_repeats_input,
        train_batch_size_input,
        epoch_input,
        save_every_n_epochs_input,
        mixed_precision_input,
        save_precision_input,
        seed_input,
        num_cpu_threads_per_process_input,
        train_text_encoder_input,
        create_caption,
        create_buckets,
        save_model_as_dropdown,
        caption_extention_input,
        use_8bit_adam,
        xformers,
        clip_skip,
    ]

    button_run.click(train_model, inputs=settings_list)

    button_open_config.click(
        open_config_file,
        inputs=[config_file_name] + settings_list,
        outputs=[config_file_name] + settings_list,
    )

    button_save_config.click(
        save_configuration,
        inputs=[dummy_ft_false, config_file_name] + settings_list,
        outputs=[config_file_name],
    )

    button_save_as_config.click(
        save_configuration,
        inputs=[dummy_ft_true, config_file_name] + settings_list,
        outputs=[config_file_name],
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
