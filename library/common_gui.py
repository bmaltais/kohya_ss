from tkinter import filedialog, Tk
from easygui import msgbox
import os
import gradio as gr
import easygui
import shutil
import sys

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄

# define a list of substrings to search for v2 base models
V2_BASE_MODELS = [
    'stabilityai/stable-diffusion-2-1-base',
    'stabilityai/stable-diffusion-2-base',
]

# define a list of substrings to search for v_parameterization models
V_PARAMETERIZATION_MODELS = [
    'stabilityai/stable-diffusion-2-1',
    'stabilityai/stable-diffusion-2',
]

# define a list of substrings to v1.x models
V1_MODELS = [
    'CompVis/stable-diffusion-v1-4',
    'runwayml/stable-diffusion-v1-5',
]

# define a list of substrings to search for
ALL_PRESET_MODELS = V2_BASE_MODELS + V_PARAMETERIZATION_MODELS + V1_MODELS

ENV_EXCLUSION = ['COLAB_GPU', 'RUNPOD_POD_ID']


def check_if_model_exist(output_name, output_dir, save_model_as):
    if save_model_as in ['diffusers', 'diffusers_safetendors']:
        ckpt_folder = os.path.join(output_dir, output_name)
        if os.path.isdir(ckpt_folder):
            msg = f'A diffuser model with the same name {ckpt_folder} already exists. Do you want to overwrite it?'
            if not easygui.ynbox(msg, 'Overwrite Existing Model?'):
                print(
                    'Aborting training due to existing model with same name...'
                )
                return True
    elif save_model_as in ['ckpt', 'safetensors']:
        ckpt_file = os.path.join(output_dir, output_name + '.' + save_model_as)
        if os.path.isfile(ckpt_file):
            msg = f'A model with the same file name {ckpt_file} already exists. Do you want to overwrite it?'
            if not easygui.ynbox(msg, 'Overwrite Existing Model?'):
                print(
                    'Aborting training due to existing model with same name...'
                )
                return True
    else:
        print(
            'Can\'t verify if existing model exist when save model is set a "same as source model", continuing to train model...'
        )
        return False

    return False


def update_my_data(my_data):
    # Update the optimizer based on the use_8bit_adam flag
    use_8bit_adam = my_data.get('use_8bit_adam', False)
    my_data.setdefault('optimizer', 'AdamW8bit' if use_8bit_adam else 'AdamW')

    # Update model_list to custom if empty or pretrained_model_name_or_path is not a preset model
    model_list = my_data.get('model_list', [])
    pretrained_model_name_or_path = my_data.get('pretrained_model_name_or_path', '')
    if not model_list or pretrained_model_name_or_path not in ALL_PRESET_MODELS:
        my_data['model_list'] = 'custom'

    # Convert epoch and save_every_n_epochs values to int if they are strings
    for key in ['epoch', 'save_every_n_epochs']:
        value = my_data.get(key, -1)
        if isinstance(value, str) and value.isdigit():
            my_data[key] = int(value)
        elif not value:
            my_data[key] = -1

    # Update LoRA_type if it is set to LoCon
    if my_data.get('LoRA_type', 'Standard') == 'LoCon':
        my_data['LoRA_type'] = 'LyCORIS/LoCon'

    # Update model save choices due to changes for LoRA and TI training
    if (
        (my_data.get('LoRA_type') or my_data.get('num_vectors_per_token'))
        and my_data.get('save_model_as') not in ['safetensors', 'ckpt']
    ):
        message = (
            'Updating save_model_as to safetensors because the current value in the config file is no longer applicable to {}'
        )
        if my_data.get('LoRA_type'):
            print(message.format('LoRA'))
        if my_data.get('num_vectors_per_token'):
            print(message.format('TI'))
        my_data['save_model_as'] = 'safetensors'

    return my_data


def get_dir_and_file(file_path):
    dir_path, file_name = os.path.split(file_path)
    return (dir_path, file_name)


# def has_ext_files(directory, extension):
#     # Iterate through all the files in the directory
#     for file in os.listdir(directory):
#         # If the file name ends with extension, return True
#         if file.endswith(extension):
#             return True
#     # If no extension files were found, return False
#     return False


def get_file_path(
    file_path='', default_extension='.json', extension_name='Config files'
):
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != 'darwin':
        current_file_path = file_path
        # print(f'current file path: {current_file_path}')

        initial_dir, initial_file = get_dir_and_file(file_path)

        # Create a hidden Tkinter root window
        root = Tk()
        root.wm_attributes('-topmost', 1)
        root.withdraw()

        # Show the open file dialog and get the selected file path
        file_path = filedialog.askopenfilename(
            filetypes=(
                (extension_name, f'*{default_extension}'),
                ('All files', '*.*'),
            ),
            defaultextension=default_extension,
            initialfile=initial_file,
            initialdir=initial_dir,
        )

        # Destroy the hidden root window
        root.destroy()

        # If no file is selected, use the current file path
        if not file_path:
            file_path = current_file_path
        current_file_path = file_path
        # print(f'current file path: {current_file_path}')

    return file_path


def get_any_file_path(file_path=''):
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != 'darwin':
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


# def set_legacy_8bitadam(optimizer, use_8bit_adam):
#     if optimizer == 'AdamW8bit':
#         # use_8bit_adam = True
#         return gr.Dropdown.update(value=optimizer), gr.Checkbox.update(
#             value=True, interactive=False, visible=True
#         )
#     else:
#         # use_8bit_adam = False
#         return gr.Dropdown.update(value=optimizer), gr.Checkbox.update(
#             value=False, interactive=False, visible=True
#         )


def get_folder_path(folder_path=''):
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != 'darwin':
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
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != 'darwin':
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
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != 'darwin':
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
    folder: str = '',
    prefix: str = '',
    postfix: str = '',
    caption_file_ext: str = '.caption',
) -> None:
    """
    Add prefix and/or postfix to the content of caption files within a folder.
    If no caption files are found, create one with the requested prefix and/or postfix.

    Args:
        folder (str): Path to the folder containing caption files.
        prefix (str, optional): Prefix to add to the content of the caption files.
        postfix (str, optional): Postfix to add to the content of the caption files.
        caption_file_ext (str, optional): Extension of the caption files.
    """

    if prefix == '' and postfix == '':
        return

    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [
        f for f in os.listdir(folder) if f.lower().endswith(image_extensions)
    ]

    for image_file in image_files:
        caption_file_name = os.path.splitext(image_file)[0] + caption_file_ext
        caption_file_path = os.path.join(folder, caption_file_name)

        if not os.path.exists(caption_file_path):
            with open(caption_file_path, 'w') as f:
                separator = ' ' if prefix and postfix else ''
                f.write(f'{prefix}{separator}{postfix}')
        else:
            with open(caption_file_path, 'r+') as f:
                content = f.read()
                content = content.rstrip()
                f.seek(0, 0)

                prefix_separator = ' ' if prefix else ''
                postfix_separator = ' ' if postfix else ''
                f.write(
                    f'{prefix}{prefix_separator}{content}{postfix_separator}{postfix}'
                )


def has_ext_files(folder_path: str, file_extension: str) -> bool:
    """
    Check if there are any files with the specified extension in the given folder.

    Args:
        folder_path (str): Path to the folder containing files.
        file_extension (str): Extension of the files to look for.

    Returns:
        bool: True if files with the specified extension are found, False otherwise.
    """
    for file in os.listdir(folder_path):
        if file.endswith(file_extension):
            return True
    return False


def find_replace(
    folder_path: str = '',
    caption_file_ext: str = '.caption',
    search_text: str = '',
    replace_text: str = '',
) -> None:
    """
    Find and replace text in caption files within a folder.

    Args:
        folder_path (str, optional): Path to the folder containing caption files.
        caption_file_ext (str, optional): Extension of the caption files.
        search_text (str, optional): Text to search for in the caption files.
        replace_text (str, optional): Text to replace the search text with.
    """
    print('Running caption find/replace')

    if not has_ext_files(folder_path, caption_file_ext):
        msgbox(
            f'No files with extension {caption_file_ext} were found in {folder_path}...'
        )
        return

    if search_text == '':
        return

    caption_files = [
        f for f in os.listdir(folder_path) if f.endswith(caption_file_ext)
    ]

    for caption_file in caption_files:
        with open(
            os.path.join(folder_path, caption_file), 'r', errors='ignore'
        ) as f:
            content = f.read()

        content = content.replace(search_text, replace_text)

        with open(os.path.join(folder_path, caption_file), 'w') as f:
            f.write(content)


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


def set_pretrained_model_name_or_path_input(
    model_list, pretrained_model_name_or_path, v2, v_parameterization
):
    # check if $v2 and $v_parameterization are empty and if $pretrained_model_name_or_path contains any of the substrings in the v2 list
    if str(model_list) in V2_BASE_MODELS:
        print('SD v2 model detected. Setting --v2 parameter')
        v2 = True
        v_parameterization = False
        pretrained_model_name_or_path = str(model_list)

    # check if $v2 and $v_parameterization are empty and if $pretrained_model_name_or_path contains any of the substrings in the v_parameterization list
    if str(model_list) in V_PARAMETERIZATION_MODELS:
        print(
            'SD v2 v_parameterization detected. Setting --v2 parameter and --v_parameterization'
        )
        v2 = True
        v_parameterization = True
        pretrained_model_name_or_path = str(model_list)

    if str(model_list) in V1_MODELS:
        v2 = False
        v_parameterization = False
        pretrained_model_name_or_path = str(model_list)

    if model_list == 'custom':
        if (
            str(pretrained_model_name_or_path) in V1_MODELS
            or str(pretrained_model_name_or_path) in V2_BASE_MODELS
            or str(pretrained_model_name_or_path) in V_PARAMETERIZATION_MODELS
        ):
            pretrained_model_name_or_path = ''
            v2 = False
            v_parameterization = False
    return model_list, pretrained_model_name_or_path, v2, v_parameterization


def set_v2_checkbox(model_list, v2, v_parameterization):
    # check if $v2 and $v_parameterization are empty and if $pretrained_model_name_or_path contains any of the substrings in the v2 list
    if str(model_list) in V2_BASE_MODELS:
        v2 = True
        v_parameterization = False

    # check if $v2 and $v_parameterization are empty and if $pretrained_model_name_or_path contains any of the substrings in the v_parameterization list
    if str(model_list) in V_PARAMETERIZATION_MODELS:
        v2 = True
        v_parameterization = True

    if str(model_list) in V1_MODELS:
        v2 = False
        v_parameterization = False

    return v2, v_parameterization


def set_model_list(
    model_list,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
):

    if not pretrained_model_name_or_path in ALL_PRESET_MODELS:
        model_list = 'custom'
    else:
        model_list = pretrained_model_name_or_path

    return model_list, v2, v_parameterization


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
            button_load_config = gr.Button('Load 💾', elem_id='open_folder')
            config_file_name.change(
                remove_doublequote,
                inputs=[config_file_name],
                outputs=[config_file_name],
            )
    return (
        button_open_config,
        button_save_config,
        button_save_as_config,
        config_file_name,
        button_load_config,
    )


def get_pretrained_model_name_or_path_file(
    model_list, pretrained_model_name_or_path
):
    pretrained_model_name_or_path = get_any_file_path(
        pretrained_model_name_or_path
    )
    set_model_list(model_list, pretrained_model_name_or_path)


def gradio_source_model(save_model_as_choices = [
                    'same as source model',
                    'ckpt',
                    'diffusers',
                    'diffusers_safetensors',
                    'safetensors',
                ]):
    with gr.Tab('Source model'):
        # Define the input elements
        with gr.Row():
            pretrained_model_name_or_path = gr.Textbox(
                label='Pretrained model name or path',
                placeholder='enter the path to custom model or name of pretrained model',
                value='runwayml/stable-diffusion-v1-5',
            )
            pretrained_model_name_or_path_file = gr.Button(
                document_symbol, elem_id='open_folder_small'
            )
            pretrained_model_name_or_path_file.click(
                get_any_file_path,
                inputs=pretrained_model_name_or_path,
                outputs=pretrained_model_name_or_path,
                show_progress=False,
            )
            pretrained_model_name_or_path_folder = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            pretrained_model_name_or_path_folder.click(
                get_folder_path,
                inputs=pretrained_model_name_or_path,
                outputs=pretrained_model_name_or_path,
                show_progress=False,
            )
            model_list = gr.Dropdown(
                label='Model Quick Pick',
                choices=[
                    'custom',
                    'stabilityai/stable-diffusion-2-1-base',
                    'stabilityai/stable-diffusion-2-base',
                    'stabilityai/stable-diffusion-2-1',
                    'stabilityai/stable-diffusion-2',
                    'runwayml/stable-diffusion-v1-5',
                    'CompVis/stable-diffusion-v1-4',
                ],
                value='runwayml/stable-diffusion-v1-5',
            )
            save_model_as = gr.Dropdown(
                label='Save trained model as',
                choices=save_model_as_choices,
                value='safetensors',
            )

        with gr.Row():
            v2 = gr.Checkbox(label='v2', value=False)
            v_parameterization = gr.Checkbox(
                label='v_parameterization', value=False
            )
            v2.change(
                set_v2_checkbox,
                inputs=[model_list, v2, v_parameterization],
                outputs=[v2, v_parameterization],
                show_progress=False,
            )
            v_parameterization.change(
                set_v2_checkbox,
                inputs=[model_list, v2, v_parameterization],
                outputs=[v2, v_parameterization],
                show_progress=False,
            )
        model_list.change(
            set_pretrained_model_name_or_path_input,
            inputs=[
                model_list,
                pretrained_model_name_or_path,
                v2,
                v_parameterization,
            ],
            outputs=[
                model_list,
                pretrained_model_name_or_path,
                v2,
                v_parameterization,
            ],
            show_progress=False,
        )
        # Update the model list and parameters when user click outside the button or field
        pretrained_model_name_or_path.change(
            set_model_list,
            inputs=[
                model_list,
                pretrained_model_name_or_path,
                v2,
                v_parameterization,
            ],
            outputs=[
                model_list,
                v2,
                v_parameterization,
            ],
            show_progress=False,
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
            maximum=64,
            label='Train batch size',
            value=1,
            step=1,
        )
        epoch = gr.Number(label='Epoch', value=1, precision=0)
        save_every_n_epochs = gr.Number(
            label='Save every N epochs', value=1, precision=0
        )
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
        seed = gr.Textbox(label='Seed', placeholder='(Optional) eg:1234')
        cache_latents = gr.Checkbox(label='Cache latent', value=True)
    with gr.Row():
        learning_rate = gr.Textbox(
            label='Learning rate', value=learning_rate_value
        )
        lr_scheduler = gr.Dropdown(
            label='LR Scheduler',
            choices=[
                'adafactor',
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
                'SGDNesterov8bit',
            ],
            value='AdamW8bit',
            interactive=True,
        )
    with gr.Row():
        optimizer_args = gr.Textbox(
            label='Optimizer extra arguments',
            placeholder='(Optional) eg: relative_step=True scale_parameter=True warmup_init=True',
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
        f' --save_every_n_epochs="{int(kwargs.get("save_every_n_epochs", 1))}"'
        if int(kwargs.get('save_every_n_epochs'))
        else '',
        f' --mixed_precision="{kwargs.get("mixed_precision", "")}"'
        if kwargs.get('mixed_precision')
        else '',
        f' --save_precision="{kwargs.get("save_precision", "")}"'
        if kwargs.get('save_precision')
        else '',
        f' --seed="{kwargs.get("seed", "")}"'
        if kwargs.get('seed') != ''
        else '',
        f' --caption_extension="{kwargs.get("caption_extension", "")}"'
        if kwargs.get('caption_extension')
        else '',
        ' --cache_latents' if kwargs.get('cache_latents') else '',
        # ' --use_lion_optimizer' if kwargs.get('optimizer') == 'Lion' else '',
        f' --optimizer_type="{kwargs.get("optimizer", "AdamW")}"',
        f' --optimizer_args {kwargs.get("optimizer_args", "")}'
        if not kwargs.get('optimizer_args') == ''
        else '',
    ]
    run_cmd = ''.join(options)
    return run_cmd


def gradio_advanced_training():
    with gr.Row():
        additional_parameters = gr.Textbox(
            label='Additional parameters',
            placeholder='(Optional) Use to provide additional parameters not handled by the GUI. Eg: --some_parameters "value"',
        )
    with gr.Row():
        save_every_n_steps = gr.Number(
            label='Save every N steps', value=0, precision=0, info='(Optional) The model is saved every specified steps'
        )
        save_last_n_steps = gr.Number(
            label='Save last N steps', value=0, precision=0, info='(Optional) Save only the specified number of models (old models will be deleted)'
        )
        save_last_n_steps_state = gr.Number(
            label='Save last N steps', value=0, precision=0, info='(Optional) Save only the specified number of states (old models will be deleted)'
        )
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
        # This use_8bit_adam element should be removed in a future release as it is no longer used
        # use_8bit_adam = gr.Checkbox(
        #     label='Use 8bit adam', value=False, visible=False
        # )
        xformers = gr.Checkbox(label='Use xformers', value=True)
        color_aug = gr.Checkbox(label='Color augmentation', value=False)
        flip_aug = gr.Checkbox(label='Flip augmentation', value=False)
        min_snr_gamma = gr.Slider(label='Min SNR gamma', value = 0, minimum=0, maximum=20, step=1)
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
            label='Dropout caption every n epochs', value=0
        )
        caption_dropout_rate = gr.Slider(
            label='Rate of caption dropout', value=0, minimum=0, maximum=1
        )
        vae_batch_size = gr.Slider(
            label='VAE batch size',
            minimum=0,
            maximum=32,
            value=0,
            step=1
        )
    with gr.Row():
        save_state = gr.Checkbox(label='Save training state', value=False)
        resume = gr.Textbox(
            label='Resume from saved training state',
            placeholder='path to "last-state" state folder to resume from',
        )
        resume_button = gr.Button('📂', elem_id='open_folder_small')
        resume_button.click(
            get_folder_path,
            outputs=resume,
            show_progress=False,
        )
        max_train_epochs = gr.Textbox(
            label='Max train epoch',
            placeholder='(Optional) Override number of epoch',
        )
        max_data_loader_n_workers = gr.Textbox(
            label='Max num workers for DataLoader',
            placeholder='(Optional) Override number of epoch. Default: 8',
            value="0",
        )
    return (
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
        save_every_n_steps,
        save_last_n_steps,
        save_last_n_steps_state,
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
        f' --caption_dropout_every_n_epochs="{int(kwargs.get("caption_dropout_every_n_epochs", 0))}"'
        if int(kwargs.get('caption_dropout_every_n_epochs', 0)) > 0
        else '',
        f' --caption_dropout_rate="{float(kwargs.get("caption_dropout_rate", 0))}"'
        if float(kwargs.get('caption_dropout_rate', 0)) > 0
        else '',
        f' --vae_batch_size="{kwargs.get("vae_batch_size", 0)}"'
        if int(kwargs.get('vae_batch_size', 0)) > 0
        else '',
        f' --bucket_reso_steps={int(kwargs.get("bucket_reso_steps", 1))}'
        if int(kwargs.get('bucket_reso_steps', 64)) >= 1
        else '',
        f' --save_every_n_steps="{int(kwargs.get("save_every_n_steps", 0))}"'
        if int(kwargs.get('save_every_n_steps')) > 0
        else '',
        f' --save_last_n_steps="{int(kwargs.get("save_last_n_steps", 0))}"'
        if int(kwargs.get('save_last_n_steps')) > 0
        else '',
        f' --save_last_n_steps_state="{int(kwargs.get("save_last_n_steps_state", 0))}"'
        if int(kwargs.get('save_last_n_steps_state')) > 0
        else '',
        f' --min_snr_gamma={int(kwargs.get("min_snr_gamma", 0))}'
        if int(kwargs.get('min_snr_gamma', 0)) >= 1
        else '',
        ' --save_state' if kwargs.get('save_state') else '',
        ' --mem_eff_attn' if kwargs.get('mem_eff_attn') else '',
        ' --color_aug' if kwargs.get('color_aug') else '',
        ' --flip_aug' if kwargs.get('flip_aug') else '',
        ' --shuffle_caption' if kwargs.get('shuffle_caption') else '',
        ' --gradient_checkpointing' if kwargs.get('gradient_checkpointing')
        else '',
        ' --full_fp16' if kwargs.get('full_fp16') else '',
        ' --xformers' if kwargs.get('xformers') else '',
        # ' --use_8bit_adam' if kwargs.get('use_8bit_adam') else '',
        ' --persistent_data_loader_workers'
        if kwargs.get('persistent_data_loader_workers')
        else '',
        ' --bucket_no_upscale' if kwargs.get('bucket_no_upscale') else '',
        ' --random_crop' if kwargs.get('random_crop') else '',
        f' --noise_offset={float(kwargs.get("noise_offset", 0))}'
        if not kwargs.get('noise_offset', '') == ''
        else '',
        f' {kwargs.get("additional_parameters", "")}',
    ]
    run_cmd = ''.join(options)
    return run_cmd
