from tkinter import filedialog, Tk
import os
import gradio as gr
from easygui import msgbox
import shutil
import sys

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄


def get_dir_and_file(file_path):
    dir_path, file_name = os.path.split(file_path)
    return (dir_path, file_name)


def has_ext_files(directory, extension):
    # Iterate through all the files in the directory
    for file in os.listdir(directory):
        # If the file name ends with extension, return True
        if file.endswith(extension):
            return True
    # If no extension files were found, return False
    return False


def get_file_path(
    file_path='', defaultextension='.json', extension_name='Config files'
):
    current_file_path = file_path
    # print(f'current file path: {current_file_path}')

    initial_dir, initial_file = get_dir_and_file(file_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=(
            (f'{extension_name}', f'{defaultextension}'),
            ('All files', '*'),
        ),
        defaultextension=defaultextension,
        initialfile=initial_file,
        initialdir=initial_dir,
    )
    root.destroy()

    if file_path == '':
        file_path = current_file_path

    return file_path


def get_any_file_path(file_path=''):
    current_file_path = file_path
    # print(f'current file path: {current_file_path}')

    initial_dir, initial_file = get_dir_and_file(file_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        initialfile=initial_file,
    )
    root.destroy()

    if file_path == '':
        file_path = current_file_path

    return file_path


def remove_doublequote(file_path):
    if file_path != None:
        file_path = file_path.replace('"', '')

    return file_path


def get_folder_path(folder_path=''):
    current_folder_path = folder_path

    initial_dir, initial_file = get_dir_and_file(folder_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    folder_path = filedialog.askdirectory(initialdir=initial_dir)
    root.destroy()

    if folder_path == '':
        folder_path = current_folder_path

    return folder_path


def get_saveasfile_path(
    file_path='', defaultextension='.json', extension_name='Config files'
):
    current_file_path = file_path
    # print(f'current file path: {current_file_path}')

    initial_dir, initial_file = get_dir_and_file(file_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    save_file_path = filedialog.asksaveasfile(
        filetypes=(
            (f'{extension_name}', f'{defaultextension}'),
            ('All files', '*'),
        ),
        defaultextension=defaultextension,
        initialdir=initial_dir,
        initialfile=initial_file,
    )
    root.destroy()

    # print(save_file_path)

    if save_file_path == None:
        file_path = current_file_path
    else:
        print(save_file_path.name)
        file_path = save_file_path.name

    # print(file_path)

    return file_path


def get_saveasfilename_path(
    file_path='', extensions='*', extension_name='Config files'
):
    current_file_path = file_path
    # print(f'current file path: {current_file_path}')

    initial_dir, initial_file = get_dir_and_file(file_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    save_file_path = filedialog.asksaveasfilename(
        filetypes=((f'{extension_name}', f'{extensions}'), ('All files', '*')),
        defaultextension=extensions,
        initialdir=initial_dir,
        initialfile=initial_file,
    )
    root.destroy()

    if save_file_path == '':
        file_path = current_file_path
    else:
        # print(save_file_path)
        file_path = save_file_path

    return file_path


def add_pre_postfix(
    folder='', prefix='', postfix='', caption_file_ext='.caption'
):
    if not has_ext_files(folder, caption_file_ext):
        msgbox(
            f'No files with extension {caption_file_ext} were found in {folder}...'
        )
        return

    if prefix == '' and postfix == '':
        return

    files = [f for f in os.listdir(folder) if f.endswith(caption_file_ext)]
    if not prefix == '':
        prefix = f'{prefix} '
    if not postfix == '':
        postfix = f' {postfix}'

    for file in files:
        with open(os.path.join(folder, file), 'r+') as f:
            content = f.read()
            content = content.rstrip()
            f.seek(0, 0)
            f.write(f'{prefix}{content}{postfix}')
    f.close()


def find_replace(folder='', caption_file_ext='.caption', find='', replace=''):
    print('Running caption find/replace')
    if not has_ext_files(folder, caption_file_ext):
        msgbox(
            f'No files with extension {caption_file_ext} were found in {folder}...'
        )
        return

    if find == '':
        return

    files = [f for f in os.listdir(folder) if f.endswith(caption_file_ext)]
    for file in files:
        with open(os.path.join(folder, file), 'r', errors='ignore') as f:
            content = f.read()
            f.close
        content = content.replace(find, replace)
        with open(os.path.join(folder, file), 'w') as f:
            f.write(content)
            f.close()


def color_aug_changed(color_aug):
    if color_aug:
        msgbox(
            'Disabling "Cache latent" because "Color augmentation" has been selected...'
        )
        return gr.Checkbox.update(value=False, interactive=False)
    else:
        return gr.Checkbox.update(value=True, interactive=True)


def save_inference_file(output_dir, v2, v_parameterization, output_name):
    # List all files in the directory
    files = os.listdir(output_dir)

    # Iterate over the list of files
    for file in files:
        # Check if the file starts with the value of output_name
        if file.startswith(output_name):
            # Check if it is a file or a directory
            if os.path.isfile(os.path.join(output_dir, file)):
                # Split the file name and extension
                file_name, ext = os.path.splitext(file)

                # Copy the v2-inference-v.yaml file to the current file, with a .yaml extension
                if v2 and v_parameterization:
                    print(
                        f'Saving v2-inference-v.yaml as {output_dir}/{file_name}.yaml'
                    )
                    shutil.copy(
                        f'./v2_inference/v2-inference-v.yaml',
                        f'{output_dir}/{file_name}.yaml',
                    )
                elif v2:
                    print(
                        f'Saving v2-inference.yaml as {output_dir}/{file_name}.yaml'
                    )
                    shutil.copy(
                        f'./v2_inference/v2-inference.yaml',
                        f'{output_dir}/{file_name}.yaml',
                    )


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

    ###
    ### Gradio common GUI section
    ###


def gradio_config():
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
    return (
        button_open_config,
        button_save_config,
        button_save_as_config,
        config_file_name,
    )


def gradio_source_model():
    with gr.Tab('Source model'):
        # Define the input elements
        with gr.Row():
            pretrained_model_name_or_path = gr.Textbox(
                label='Pretrained model name or path',
                placeholder='enter the path to custom model or name of pretrained model',
            )
            pretrained_model_name_or_path_file = gr.Button(
                document_symbol, elem_id='open_folder_small'
            )
            pretrained_model_name_or_path_file.click(
                get_any_file_path,
                inputs=pretrained_model_name_or_path,
                outputs=pretrained_model_name_or_path,
            )
            pretrained_model_name_or_path_folder = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            pretrained_model_name_or_path_folder.click(
                get_folder_path,
                inputs=pretrained_model_name_or_path,
                outputs=pretrained_model_name_or_path,
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
            save_model_as = gr.Dropdown(
                label='Save trained model as',
                choices=[
                    'same as source model',
                    'ckpt',
                    'diffusers',
                    'diffusers_safetensors',
                    'safetensors',
                ],
                value='safetensors',
            )

        with gr.Row():
            v2 = gr.Checkbox(label='v2', value=True)
            v_parameterization = gr.Checkbox(
                label='v_parameterization', value=False
            )
        model_list.change(
            set_pretrained_model_name_or_path_input,
            inputs=[model_list, v2, v_parameterization],
            outputs=[
                pretrained_model_name_or_path,
                v2,
                v_parameterization,
            ],
        )
    return (
        pretrained_model_name_or_path,
        v2,
        v_parameterization,
        save_model_as,
        model_list,
    )


def gradio_training(
    learning_rate_value='1e-6',
    lr_scheduler_value='constant',
    lr_warmup_value='0',
):
    with gr.Row():
        train_batch_size = gr.Slider(
            minimum=1,
            maximum=32,
            label='Train batch size',
            value=1,
            step=1,
        )
        epoch = gr.Textbox(label='Epoch', value=1)
        save_every_n_epochs = gr.Textbox(label='Save every N epochs', value=1)
        caption_extension = gr.Textbox(
            label='Caption Extension',
            placeholder='(Optional) Extension for caption files. default: .caption',
        )
    with gr.Row():
        mixed_precision = gr.Dropdown(
            label='Mixed precision',
            choices=[
                'no',
                'fp16',
                'bf16',
            ],
            value='fp16',
        )
        save_precision = gr.Dropdown(
            label='Save precision',
            choices=[
                'float',
                'fp16',
                'bf16',
            ],
            value='fp16',
        )
        num_cpu_threads_per_process = gr.Slider(
            minimum=1,
            maximum=os.cpu_count(),
            step=1,
            label='Number of CPU threads per core',
            value=2,
        )
        seed = gr.Textbox(label='Seed', value=1234)
        cache_latents = gr.Checkbox(label='Cache latent', value=True)
    with gr.Row():
        learning_rate = gr.Textbox(
            label='Learning rate', value=learning_rate_value
        )
        lr_scheduler = gr.Dropdown(
            label='LR Scheduler',
            choices=[
                'constant',
                'constant_with_warmup',
                'cosine',
                'cosine_with_restarts',
                'linear',
                'polynomial',
            ],
            value=lr_scheduler_value,
        )
        lr_warmup = gr.Textbox(
            label='LR warmup (% of steps)', value=lr_warmup_value
        )
        optimizer = gr.Dropdown(
            label='Optimizer',
            choices=[
                'AdamW',
                'AdamW8bit',
                'Adafactor',
                'DAdaptation',
                'Lion',
                'SGDNesterov',
                'SGDNesterov8bit'
            ],
            value="AdamW",
            interactive=True,
        )
    with gr.Row():
        optimizer_args = gr.Textbox(
            label='Optimizer extra arguments', placeholder='(Optional) eg: relative_step=True scale_parameter=True warmup_init=True'
        )
    return (
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
    )


def run_cmd_training(**kwargs):
    options = [
        f' --learning_rate="{kwargs.get("learning_rate", "")}"'
        if kwargs.get('learning_rate')
        else '',
        f' --lr_scheduler="{kwargs.get("lr_scheduler", "")}"'
        if kwargs.get('lr_scheduler')
        else '',
        f' --lr_warmup_steps="{kwargs.get("lr_warmup_steps", "")}"'
        if kwargs.get('lr_warmup_steps')
        else '',
        f' --train_batch_size="{kwargs.get("train_batch_size", "")}"'
        if kwargs.get('train_batch_size')
        else '',
        f' --max_train_steps="{kwargs.get("max_train_steps", "")}"'
        if kwargs.get('max_train_steps')
        else '',
        f' --save_every_n_epochs="{kwargs.get("save_every_n_epochs", "")}"'
        if kwargs.get('save_every_n_epochs')
        else '',
        f' --mixed_precision="{kwargs.get("mixed_precision", "")}"'
        if kwargs.get('mixed_precision')
        else '',
        f' --save_precision="{kwargs.get("save_precision", "")}"'
        if kwargs.get('save_precision')
        else '',
        f' --seed="{kwargs.get("seed", "")}"' if kwargs.get('seed') else '',
        f' --caption_extension="{kwargs.get("caption_extension", "")}"'
        if kwargs.get('caption_extension')
        else '',
        ' --cache_latents' if kwargs.get('cache_latents') else '',
        # ' --use_lion_optimizer' if kwargs.get('optimizer') == 'Lion' else '',
        f' --optimizer_type="{kwargs.get("optimizer", "AdamW")}"',
        f' --optimizer_args {kwargs.get("optimizer_args", "")}' if not kwargs.get('optimizer_args') == '' else '',
    ]
    run_cmd = ''.join(options)
    return run_cmd

# # This function takes a dictionary of keyword arguments and returns a string that can be used to run a command-line training script
# def run_cmd_training(**kwargs):
#     arg_map = {
#         'learning_rate': ' --learning_rate="{}"',
#         'lr_scheduler': ' --lr_scheduler="{}"',
#         'lr_warmup_steps': ' --lr_warmup_steps="{}"',
#         'train_batch_size': ' --train_batch_size="{}"',
#         'max_train_steps': ' --max_train_steps="{}"',
#         'save_every_n_epochs': ' --save_every_n_epochs="{}"',
#         'mixed_precision': ' --mixed_precision="{}"',
#         'save_precision': ' --save_precision="{}"',
#         'seed': ' --seed="{}"',
#         'caption_extension': ' --caption_extension="{}"',
#         'cache_latents': ' --cache_latents',
#         'optimizer': ' --use_lion_optimizer' if kwargs.get('optimizer') == 'Lion' else '',
#     }

#     options = [arg_map[key].format(value) for key, value in kwargs.items() if key in arg_map and value]

#     cmd = ''.join(options)

#     return cmd


def gradio_advanced_training():
    with gr.Row():
        keep_tokens = gr.Slider(
            label='Keep n tokens', value='0', minimum=0, maximum=32, step=1
        )
        clip_skip = gr.Slider(
            label='Clip skip', value='1', minimum=1, maximum=12, step=1
        )
        max_token_length = gr.Dropdown(
            label='Max Token Length',
            choices=[
                '75',
                '150',
                '225',
            ],
            value='75',
        )
        full_fp16 = gr.Checkbox(
            label='Full fp16 training (experimental)', value=False
        )
    with gr.Row():
        gradient_checkpointing = gr.Checkbox(
            label='Gradient checkpointing', value=False
        )
        shuffle_caption = gr.Checkbox(label='Shuffle caption', value=False)
        persistent_data_loader_workers = gr.Checkbox(
            label='Persistent data loader', value=False
        )
        mem_eff_attn = gr.Checkbox(
            label='Memory efficient attention', value=False
        )
    with gr.Row():
        use_8bit_adam = gr.Checkbox(label='Use 8bit adam', value=sys.platform.startswith('linux'))
        xformers = gr.Checkbox(label='Use xformers', value=True)
        color_aug = gr.Checkbox(label='Color augmentation', value=False)
        flip_aug = gr.Checkbox(label='Flip augmentation', value=False)
    with gr.Row():
        bucket_no_upscale = gr.Checkbox(
            label="Don't upscale bucket resolution", value=True
        )
        bucket_reso_steps = gr.Number(
            label='Bucket resolution steps', value=64
        )
        random_crop = gr.Checkbox(
            label='Random crop instead of center crop', value=False
        )
        noise_offset = gr.Textbox(
            label='Noise offset (0 - 1)', placeholder='(Oprional) eg: 0.1'
        )
        
    with gr.Row():
        caption_dropout_every_n_epochs = gr.Number(
            label="Dropout caption every n epochs",
            value=0
        )
        caption_dropout_rate = gr.Slider(
            label="Rate of caption dropout",
            value=0,
            minimum=0,
            maximum=1
        )
    with gr.Row():
        save_state = gr.Checkbox(label='Save training state', value=False)
        resume = gr.Textbox(
            label='Resume from saved training state',
            placeholder='path to "last-state" state folder to resume from',
        )
        resume_button = gr.Button('📂', elem_id='open_folder_small')
        resume_button.click(get_folder_path, outputs=resume)
        max_train_epochs = gr.Textbox(
            label='Max train epoch',
            placeholder='(Optional) Override number of epoch',
        )
        max_data_loader_n_workers = gr.Textbox(
            label='Max num workers for DataLoader',
            placeholder='(Optional) Override number of epoch. Default: 8',
        )
    return (
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
    )


def run_cmd_advanced_training(**kwargs):
    options = [
        f' --max_train_epochs="{kwargs.get("max_train_epochs", "")}"'
        if kwargs.get('max_train_epochs')
        else '',
        f' --max_data_loader_n_workers="{kwargs.get("max_data_loader_n_workers", "")}"'
        if kwargs.get('max_data_loader_n_workers')
        else '',
        f' --max_token_length={kwargs.get("max_token_length", "")}'
        if int(kwargs.get('max_token_length', 75)) > 75
        else '',
        f' --clip_skip={kwargs.get("clip_skip", "")}'
        if int(kwargs.get('clip_skip', 1)) > 1
        else '',
        f' --resume="{kwargs.get("resume", "")}"'
        if kwargs.get('resume')
        else '',
        f' --keep_tokens="{kwargs.get("keep_tokens", "")}"'
        if int(kwargs.get('keep_tokens', 0)) > 0
        else '',
        f' --caption_dropout_every_n_epochs="{kwargs.get("caption_dropout_every_n_epochs", "")}"'
        if int(kwargs.get('caption_dropout_every_n_epochs', 0)) > 0
        else '',
        f' --caption_dropout_rate="{kwargs.get("caption_dropout_rate", "")}"'
        if float(kwargs.get('caption_dropout_rate', 0)) > 0
        else '',
        
        f' --bucket_reso_steps={int(kwargs.get("bucket_reso_steps", 1))}'
        if int(kwargs.get('bucket_reso_steps', 64)) >= 1
        else '',
        
        ' --save_state' if kwargs.get('save_state') else '',
        ' --mem_eff_attn' if kwargs.get('mem_eff_attn') else '',
        ' --color_aug' if kwargs.get('color_aug') else '',
        ' --flip_aug' if kwargs.get('flip_aug') else '',
        ' --shuffle_caption' if kwargs.get('shuffle_caption') else '',
        ' --gradient_checkpointing'
        if kwargs.get('gradient_checkpointing')
        else '',
        ' --full_fp16' if kwargs.get('full_fp16') else '',
        ' --xformers' if kwargs.get('xformers') else '',
        ' --use_8bit_adam' if kwargs.get('use_8bit_adam') else '',
        ' --persistent_data_loader_workers'
        if kwargs.get('persistent_data_loader_workers')
        else '',
        ' --bucket_no_upscale' if kwargs.get('bucket_no_upscale') else '',
        ' --random_crop' if kwargs.get('random_crop') else '',
        f' --noise_offset={float(kwargs.get("noise_offset", 0))}'
        if not kwargs.get('noise_offset', '') == ''
        else '',
    ]
    run_cmd = ''.join(options)
    return run_cmd

# def run_cmd_advanced_training(**kwargs):
#     arg_map = {
#         'max_train_epochs': ' --max_train_epochs="{}"',
#         'max_data_loader_n_workers': ' --max_data_loader_n_workers="{}"',
#         'max_token_length': ' --max_token_length={}' if int(kwargs.get('max_token_length', 75)) > 75 else '',
#         'clip_skip': ' --clip_skip={}' if int(kwargs.get('clip_skip', 1)) > 1 else '',
#         'resume': ' --resume="{}"',
#         'keep_tokens': ' --keep_tokens="{}"' if int(kwargs.get('keep_tokens', 0)) > 0 else '',
#         'caption_dropout_every_n_epochs': ' --caption_dropout_every_n_epochs="{}"' if int(kwargs.get('caption_dropout_every_n_epochs', 0)) > 0 else '',
#         'caption_dropout_rate': ' --caption_dropout_rate="{}"' if float(kwargs.get('caption_dropout_rate', 0)) > 0 else '',
#         'bucket_reso_steps': ' --bucket_reso_steps={:d}' if int(kwargs.get('bucket_reso_steps', 64)) >= 1 else '',
#         'save_state': ' --save_state',
#         'mem_eff_attn': ' --mem_eff_attn',
#         'color_aug': ' --color_aug',
#         'flip_aug': ' --flip_aug',
#         'shuffle_caption': ' --shuffle_caption',
#         'gradient_checkpointing': ' --gradient_checkpointing',
#         'full_fp16': ' --full_fp16',
#         'xformers': ' --xformers',
#         'use_8bit_adam': ' --use_8bit_adam',
#         'persistent_data_loader_workers': ' --persistent_data_loader_workers',
#         'bucket_no_upscale': ' --bucket_no_upscale',
#         'random_crop': ' --random_crop',
#     }

#     options = [arg_map[key].format(value) for key, value in kwargs.items() if key in arg_map and value]

#     cmd = ''.join(options)

#     return cmd