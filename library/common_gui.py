from tkinter import filedialog, Tk
from easygui import msgbox
import os
import re
import gradio as gr
import easygui
import shutil
import sys

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

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


def check_if_model_exist(
    output_name, output_dir, save_model_as, headless=False
):
    if headless:
        log.info(
            'Headless mode, skipping verification if model already exist... if model already exist it will be overwritten...'
        )
        return False

    if save_model_as in ['diffusers', 'diffusers_safetendors']:
        ckpt_folder = os.path.join(output_dir, output_name)
        if os.path.isdir(ckpt_folder):
            msg = f'A diffuser model with the same name {ckpt_folder} already exists. Do you want to overwrite it?'
            if not easygui.ynbox(msg, 'Overwrite Existing Model?'):
                log.info(
                    'Aborting training due to existing model with same name...'
                )
                return True
    elif save_model_as in ['ckpt', 'safetensors']:
        ckpt_file = os.path.join(output_dir, output_name + '.' + save_model_as)
        if os.path.isfile(ckpt_file):
            msg = f'A model with the same file name {ckpt_file} already exists. Do you want to overwrite it?'
            if not easygui.ynbox(msg, 'Overwrite Existing Model?'):
                log.info(
                    'Aborting training due to existing model with same name...'
                )
                return True
    else:
        log.info(
            'Can\'t verify if existing model exist when save model is set a "same as source model", continuing to train model...'
        )
        return False

    return False


def output_message(msg='', title='', headless=False):
    if headless:
        log.info(msg)
    else:
        msgbox(msg=msg, title=title)


def update_my_data(my_data):
    # Update the optimizer based on the use_8bit_adam flag
    use_8bit_adam = my_data.get('use_8bit_adam', False)
    my_data.setdefault('optimizer', 'AdamW8bit' if use_8bit_adam else 'AdamW')

    # Update model_list to custom if empty or pretrained_model_name_or_path is not a preset model
    model_list = my_data.get('model_list', [])
    pretrained_model_name_or_path = my_data.get(
        'pretrained_model_name_or_path', ''
    )
    if (
        not model_list
        or pretrained_model_name_or_path not in ALL_PRESET_MODELS
    ):
        my_data['model_list'] = 'custom'

    # Convert values to int if they are strings
    for key in ['epoch', 'save_every_n_epochs', 'lr_warmup']:
        value = my_data.get(key, 0)
        if isinstance(value, str) and value.strip().isdigit():
            my_data[key] = int(value)
        elif not value:
            my_data[key] = 0

    # Convert values to float if they are strings
    for key in ['noise_offset', 'learning_rate', 'text_encoder_lr', 'unet_lr']:
        value = my_data.get(key, 0)
        if isinstance(value, str) and value.strip().isdigit():
            my_data[key] = float(value)
        elif not value:
            my_data[key] = 0

    # Update LoRA_type if it is set to LoCon
    if my_data.get('LoRA_type', 'Standard') == 'LoCon':
        my_data['LoRA_type'] = 'LyCORIS/LoCon'

    # Update model save choices due to changes for LoRA and TI training
    if (
        my_data.get('LoRA_type') or my_data.get('num_vectors_per_token')
    ) and my_data.get('save_model_as') not in ['safetensors', 'ckpt']:
        message = 'Updating save_model_as to safetensors because the current value in the config file is no longer applicable to {}'
        if my_data.get('LoRA_type'):
            log.info(message.format('LoRA'))
        if my_data.get('num_vectors_per_token'):
            log.info(message.format('TI'))
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
    if (
        not any(var in os.environ for var in ENV_EXCLUSION)
        and sys.platform != 'darwin'
    ):
        current_file_path = file_path
        # log.info(f'current file path: {current_file_path}')

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
        # log.info(f'current file path: {current_file_path}')

    return file_path


def get_any_file_path(file_path=''):
    if (
        not any(var in os.environ for var in ENV_EXCLUSION)
        and sys.platform != 'darwin'
    ):
        current_file_path = file_path
        # log.info(f'current file path: {current_file_path}')

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
    if (
        not any(var in os.environ for var in ENV_EXCLUSION)
        and sys.platform != 'darwin'
    ):
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
    if (
        not any(var in os.environ for var in ENV_EXCLUSION)
        and sys.platform != 'darwin'
    ):
        current_file_path = file_path
        # log.info(f'current file path: {current_file_path}')

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

        # log.info(save_file_path)

        if save_file_path == None:
            file_path = current_file_path
        else:
            log.info(save_file_path.name)
            file_path = save_file_path.name

        # log.info(file_path)

    return file_path


def get_saveasfilename_path(
    file_path='', extensions='*', extension_name='Config files'
):
    if (
        not any(var in os.environ for var in ENV_EXCLUSION)
        and sys.platform != 'darwin'
    ):
        current_file_path = file_path
        # log.info(f'current file path: {current_file_path}')

        initial_dir, initial_file = get_dir_and_file(file_path)

        root = Tk()
        root.wm_attributes('-topmost', 1)
        root.withdraw()
        save_file_path = filedialog.asksaveasfilename(
            filetypes=(
                (f'{extension_name}', f'{extensions}'),
                ('All files', '*'),
            ),
            defaultextension=extensions,
            initialdir=initial_dir,
            initialfile=initial_file,
        )
        root.destroy()

        if save_file_path == '':
            file_path = current_file_path
        else:
            # log.info(save_file_path)
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
            with open(caption_file_path, 'w', encoding='utf8') as f:
                separator = ' ' if prefix and postfix else ''
                f.write(f'{prefix}{separator}{postfix}')
        else:
            with open(caption_file_path, 'r+', encoding='utf8') as f:
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
    log.info('Running caption find/replace')

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
                    log.info(
                        f'Saving v2-inference-v.yaml as {output_dir}/{file_name}.yaml'
                    )
                    shutil.copy(
                        f'./v2_inference/v2-inference-v.yaml',
                        f'{output_dir}/{file_name}.yaml',
                    )
                elif v2:
                    log.info(
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
        log.info('SD v2 model detected. Setting --v2 parameter')
        v2 = True
        v_parameterization = False
        pretrained_model_name_or_path = str(model_list)

    # check if $v2 and $v_parameterization are empty and if $pretrained_model_name_or_path contains any of the substrings in the v_parameterization list
    if str(model_list) in V_PARAMETERIZATION_MODELS:
        log.info(
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


def gradio_config(headless=False):
    with gr.Accordion('Configuration file', open=False):
        with gr.Row():
            button_open_config = gr.Button(
                'Open 📂', elem_id='open_folder', visible=(not headless)
            )
            button_save_config = gr.Button(
                'Save 💾', elem_id='open_folder', visible=(not headless)
            )
            button_save_as_config = gr.Button(
                'Save as... 💾', elem_id='open_folder', visible=(not headless)
            )
            config_file_name = gr.Textbox(
                label='',
                placeholder="输入配置文件的路径或使用上面的'打开'按钮选择它...",
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


def gradio_source_model(
    save_model_as_choices=[
        'same as source model',
        'ckpt',
        'diffusers',
        'diffusers_safetensors',
        'safetensors',
    ],
    headless=False,
):
    with gr.Tab('Source model'):
        # Define the input elements
        with gr.Row():
            pretrained_model_name_or_path = gr.Textbox(
                label='Pretrained model name or path',
                placeholder='enter the path to custom model or name of pretrained model',
                value='runwayml/stable-diffusion-v1-5',
                info='输入预训练模型的名称或路径。这可以是存储在远程服务器上的公开模型名称，或者是本地计算机上的模型文件路径'
            )
            pretrained_model_name_or_path_file = gr.Button(
                document_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            pretrained_model_name_or_path_file.click(
                get_any_file_path,
                inputs=pretrained_model_name_or_path,
                outputs=pretrained_model_name_or_path,
                show_progress=False,
            )
            pretrained_model_name_or_path_folder = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
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
                info='选择预训练模型的快速选择。这可以是存储在远程服务器上的公开模型名称，或者是本地计算机上的模型文件路径'
            )
            save_model_as = gr.Dropdown(
                label='Save trained model as',
                choices=save_model_as_choices,
                value='safetensors',
                info='选择训练模型的保存方式。如果选择与源模型相同，则将覆盖源模型'
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
            info='训练批次大小。如果您的显卡内存不足，可以尝试减小此值'
        )
        epoch = gr.Number(label='Epoch', value=1, precision=0,info='训练的轮数')
        save_every_n_epochs = gr.Number(
            label='Save every N epochs', value=1, precision=0,info='每N轮保存一次模型'
        )
        caption_extension = gr.Textbox(
            label='Caption Extension',
            placeholder='(可选) 标题文件的扩展名。默认为.caption'
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
            info='当我们在训练神经网络时，我们通常需要大量的计算，这可能需要很高的计算精度。但在许多情况下，我们并不总是需要那么高的精度，例如当我们进行一些简单的运算时，或者在早期的训练阶段。在这些情况下，我们可以使用较低的精度（如fp16或bf16），这可以减少我们的内存需求，加快我们的计算速度，并可能在某些情况下提高我们的模型性能。然而，这并不总是最好的选择，因为使用较低的精度可能会导致数值不稳定性问题，特别是在需要高精度的计算中。因此，如果你选择了"no"，那就意味着你没有启用混合精度训练，所有的计算都会以全精度（通常是float32）进行'
        )
        save_precision = gr.Dropdown(
            label='Save precision',
            choices=[
                'float',
                'fp16',
                'bf16',
            ],
            value='fp16',
            info='这个选项决定了你保存模型时使用的精度。通常情况下，我们会希望保存模型时使用全精度，以确保我们的模型性能。然而，使用全精度保存模型会占用大量的存储空间。因此，如果你的存储空间有限，你可以选择使用较低的精度（如fp16或bf16）来保存你的模型。这可以大大减少模型的存储大小，但可能会牺牲一些模型性能。'
        )
        num_cpu_threads_per_process = gr.Slider(
            minimum=1,
            maximum=os.cpu_count(),
            step=1,
            label='Number of CPU threads per core',
            value=2,
            info='每个核心的CPU线程数。如果您的CPU支持，可以尝试使用此选项来加快训练速度'
        )
        seed = gr.Textbox(label='Seed', placeholder='(Optional) eg:1234',
                          info='随机种子。如果您想要重现您的训练结果，可以尝试使用此选项')
        cache_latents = gr.Checkbox(label='Cache latents', value=True)
        cache_latents_to_disk = gr.Checkbox(
            label='Cache latents to disk', value=False,
            info='将潜在变量缓存到磁盘。如果您的显卡内存不足，可以尝试使用此选项'
        )
    with gr.Row():
        learning_rate = gr.Number(
            label='Learning rate', value=learning_rate_value,
            info='学习率是机器学习中的一个超参数，它决定了模型在训练过程中参数更新的速度。学习率过高，可能导致训练过程震荡不收敛；学习率过低，训练过程可能会过慢。对于大多数优化算法，理想的学习率通常需要通过实验来确定。在一些情况下，你可能会希望使用适应性学习率，这意味着学习率会在训练过程中自动调整。'
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
            info='学习率调度器是一种用于调整学习率的策略。随着训练的进行，我们可能希望逐渐减小学习率。这是因为在训练的开始，模型离最优解还很远，此时可以使用较大的学习率进行快速学习；而当模型接近最优解时，我们希望使用较小的学习率进行细致的调整。学习率调度器就是实现这个策略的工具，它会根据预定的策略（例如，每n步降低学习率）来动态地调整学习率。'
        )
        lr_warmup = gr.Slider(
            label='LR warmup (% of steps)',
            value=lr_warmup_value,
            minimum=0,
            maximum=100,
            step=1,
            info='学习率预热是一种训练策略，训练初期采用较小的学习率，逐渐提升到预设的学习率。其目的是避免训练初期，模型对数据的拟合过快而错过全局最优解。该选项允许用户设定预热阶段占总训练步数的比例。'
        )
        optimizer = gr.Dropdown(
            label='Optimizer',
            choices=[
                'AdamW',
                'AdamW8bit',
                'Adafactor',
                'DAdaptation',
                'DAdaptAdaGrad',
                'DAdaptAdam',
                'DAdaptAdan',
                'DAdaptAdanIP',
                'DAdaptAdamPreprint',
                'DAdaptLion',
                'DAdaptSGD',
                'Lion',
                'Lion8bit',
                'Prodigy',
                'SGDNesterov',
                'SGDNesterov8bit',
            ],
            value='AdamW8bit',
            interactive=True,
            info='''优化器用于在训练过程中更新和调整模型参数以减小模型误差。不同的优化器适用于不同类型的任务和模型。例如：
            AdamW：AdamW是优化器Adam的改进版本，它更准确地处理了权重衰减。通常，对于多种类型的深度学习任务（包括图像分类、自然语言处理等），AdamW都是一个相对稳定、表现良好的选择。
            AdamW8bit：这是AdamW的一个变体，它使用了较少的比特（8比特）进行计算，这可以减小内存使用并加速计算。它适用于那些对内存有严格限制或需要更快训练速度的情况
            Adafactor：Adafactor是一种类似Adam的优化器，但内存需求更小。它主要用于大型语言模型训练，例如Transformer。
            DAdapt系列的优化器：这些优化器可以自动调整学习率，适应不同的训练阶段。这些优化器在处理需要应对训练不稳定性和噪声的复杂问题时表现良好。
            Lion和Lion8bit：Lion是一种先进的优化器，可以自动调整学习率。如果你正在训练一个复杂的模型，并且希望让优化器尽可能地自动调整学习率，那么Lion可能是个好选择。
            Prodigy：这是一种先进的优化器，专为训练大型模型设计。如果你在训练一个大型模型，并希望优化器能够自动地调整学习率以适应训练的不同阶段，Prodigy可能是一个很好的选择。
            SGDNesterov和SGDNesterov8bit：这是随机梯度下降（SGD）优化器的一个变体，SGD优化器常用于深度学习模型的训练，包括CNN、RNN等。Nesterov动量可以加速训练过程。在内存充足、计算资源充足的情况下，SGDNesterov优化器常常是训练深度模型的首选。'''
        )
    with gr.Row():
        optimizer_args = gr.Textbox(
            label='Optimizer extra arguments',
            # placeholder='(Optional) eg: relative_step=True scale_parameter=True warmup_init=True',
            placeholder='优化器的额外参数。如果您不确定如何设置这些参数，可以尝试使用默认值'
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
        cache_latents_to_disk,
        optimizer,
        optimizer_args,
    )

def get_int_or_default(kwargs, key, default_value=0):
    value = kwargs.get(key, default_value)
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        return int(value)
    elif isinstance(value, float):
        return int(value)
    else:
        log.info(f'{key} is not an int, float or a string, setting value to {default_value}')
        return default_value
    
def get_float_or_default(kwargs, key, default_value=0.0):
    value = kwargs.get(key, default_value)
    if isinstance(value, float):
        return value
    elif isinstance(value, int):
        return float(value)
    elif isinstance(value, str):
        return float(value)
    else:
        log.info(f'{key} is not an int, float or a string, setting value to {default_value}')
        return default_value

def get_str_or_default(kwargs, key, default_value=""):
    value = kwargs.get(key, default_value)
    if isinstance(value, str):
        return value
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, str):
        return str(value)
    else:
        return default_value

def run_cmd_training(**kwargs):
    run_cmd = ''
    
    learning_rate = kwargs.get("learning_rate", "")
    if learning_rate:
        run_cmd += f' --learning_rate="{learning_rate}"'
    
    lr_scheduler = kwargs.get("lr_scheduler", "")
    if lr_scheduler:
        run_cmd += f' --lr_scheduler="{lr_scheduler}"'
    
    lr_warmup_steps = kwargs.get("lr_warmup_steps", "")
    if lr_warmup_steps:
        if lr_scheduler == 'constant':
            log.info('Can\'t use LR warmup with LR Scheduler constant... ignoring...')
        else:
            run_cmd += f' --lr_warmup_steps="{lr_warmup_steps}"'
    
    train_batch_size = kwargs.get("train_batch_size", "")
    if train_batch_size:
        run_cmd += f' --train_batch_size="{train_batch_size}"'
    
    max_train_steps = kwargs.get("max_train_steps", "")
    if max_train_steps:
        run_cmd += f' --max_train_steps="{max_train_steps}"'
    
    save_every_n_epochs = kwargs.get("save_every_n_epochs")
    if save_every_n_epochs:
        run_cmd += f' --save_every_n_epochs="{int(save_every_n_epochs)}"'
    
    mixed_precision = kwargs.get("mixed_precision", "")
    if mixed_precision:
        run_cmd += f' --mixed_precision="{mixed_precision}"'
    
    save_precision = kwargs.get("save_precision", "")
    if save_precision:
        run_cmd += f' --save_precision="{save_precision}"'
    
    seed = kwargs.get("seed", "")
    if seed != '':
        run_cmd += f' --seed="{seed}"'
    
    caption_extension = kwargs.get("caption_extension", "")
    if caption_extension:
        run_cmd += f' --caption_extension="{caption_extension}"'
    
    cache_latents = kwargs.get('cache_latents')
    if cache_latents:
        run_cmd += ' --cache_latents'
    
    cache_latents_to_disk = kwargs.get('cache_latents_to_disk')
    if cache_latents_to_disk:
        run_cmd += ' --cache_latents_to_disk'
    
    optimizer_type = kwargs.get("optimizer", "AdamW")
    run_cmd += f' --optimizer_type="{optimizer_type}"'
    
    optimizer_args = kwargs.get("optimizer_args", "")
    if optimizer_args != '':
        run_cmd += f' --optimizer_args {optimizer_args}'
    
    return run_cmd


def gradio_advanced_training(headless=False):
    def noise_offset_type_change(noise_offset_type):
        if noise_offset_type == 'Original':
            return (gr.Group.update(visible=True), gr.Group.update(visible=False))
        else:
            return (gr.Group.update(visible=False), gr.Group.update(visible=True))
    with gr.Row():
        additional_parameters = gr.Textbox(
            label='Additional parameters',
            # placeholder='(Optional) Use to provide additional parameters not handled by the GUI. Eg: --some_parameters "value"',
            placeholder='用于提供GUI无法处理的额外参数。例如：--some_parameters "value"',        
        )
    with gr.Row():
        save_every_n_steps = gr.Number(
            label='Save every N steps',
            value=0,
            precision=0,
            # info='(Optional) The model is saved every specified steps',
            info='（可选）模型将在每个指定步骤保存',
        )
        save_last_n_steps = gr.Number(
            label='Save last N steps',
            value=0,
            precision=0,
            # info='(Optional) Save only the specified number of models (old models will be deleted)',
            info='（可选）仅保存指定数量的模型（旧模型将被删除）',
        )
        save_last_n_steps_state = gr.Number(
            label='Save last N states',
            value=0,
            precision=0,
            # info='(Optional) Save only the specified number of states (old models will be deleted)',
            info='（可选）仅保存指定数量的状态（旧模型将被删除）',
        )
    with gr.Row():
        keep_tokens = gr.Slider(
            label='Keep n tokens', value='0', minimum=0, maximum=32, step=1,
            info='（可选）随机打乱caption时，保留前N个不变（逗号分隔）',
        )
        clip_skip = gr.Slider(
            label='Clip skip', value='1', minimum=1, maximum=12, step=1,
            info='（可选）跳过训练的步骤数',
        )
        max_token_length = gr.Dropdown(
            label='Max Token Length',
            choices=[
                '75',
                '150',
                '225',
            ],
            value='75',
            info='（可选）最大令牌长度。这可以提高训练速度，但可能会导致模型性能下降',
        )
        full_fp16 = gr.Checkbox(
            label='Full fp16 training (experimental)', value=False,
            info='（实验性）使用fp16进行完整的训练。这可以提高训练速度，但可能会导致模型性能下降',
        )
    with gr.Row():
        gradient_checkpointing = gr.Checkbox(
            label='Gradient checkpointing', value=False,
            info='(实验性) 使用梯度检查点来减少内存使用',
        )
        shuffle_caption = gr.Checkbox(label='Shuffle caption', value=False,
            info='随机对caption进行排序',)
        persistent_data_loader_workers = gr.Checkbox(
            label='Persistent data loader', value=False,
            info='在训练过程中，持续使用数据加载器，而不是在每个epoch中重新创建数据加载器。这可以提高训练速度，但是可能会导致内存泄漏',
        )
        mem_eff_attn = gr.Checkbox(
            label='Memory efficient attention', value=False,
            info='使用内存效率的注意力，以减少内存使用',
        )
    with gr.Row():
        # This use_8bit_adam element should be removed in a future release as it is no longer used
        # use_8bit_adam = gr.Checkbox(
        #     label='Use 8bit adam', value=False, visible=False
        # )
        xformers = gr.Checkbox(label='Use xformers', value=True,
                               info='使用xformers库，而不是原始的transformers库')
        color_aug = gr.Checkbox(label='Color augmentation', value=False,
                                info='通过改变图像的颜色来增加数据的多样性，以提高模型的泛化能力')
        flip_aug = gr.Checkbox(label='Flip augmentation', value=False,info='通过翻转图像来增加数据的多样性，以提高模型的泛化能力')
        min_snr_gamma = gr.Slider(
            label='Min SNR gamma', value=0, minimum=0, maximum=20, step=1,
            info='最小的信噪比伽马值，用于在训练过程中对图像进行噪声增强',
        )
    with gr.Row():
        bucket_no_upscale = gr.Checkbox(
            label="Don't upscale bucket resolution", value=True,
            info='不要在训练过程中提高bucket的分辨率，这可以减少内存使用',
        )
        bucket_reso_steps = gr.Slider(
            label='Bucket resolution steps', value=64, minimum=1, maximum=128,
            info='在训练过程中，每个epoch提高bucket的分辨率的步数',
        )
        random_crop = gr.Checkbox(
            label='Random crop instead of center crop', value=False,
            info='在训练过程中，使用随机裁剪而不是中心裁剪',
        )
    
    with gr.Row():
        noise_offset_type = gr.Dropdown(
            label='Noise offset type',
            info='使用的噪声偏移类型。默认为原始类型，但对于某些模型，多分辨率类型可能更好。这是实验性的，可选的。',
            choices=[
                'Original',
                'Multires',
            ],
            value='Original',
        )
        with gr.Row(visible=True) as noise_offset_original:
            noise_offset = gr.Slider(
                label='Noise offset',
                value=0,
                minimum=0,
                maximum=1,
                step=0.01,
                info='噪声强度。增大这个值，可以增加模型训练中的随机性，可能会使模型更加健壮，但也可能会引入过多的噪声，影响模型的精度。推荐的值在0.05到0.15之间',
            )
            adaptive_noise_scale = gr.Slider(
                label='Adaptive noise scale',
                value=0,
                minimum=-1,
                maximum=1,
                step=0.001,
                info='用于调整噪声的规模。由于潜在变量（模型内部表示数据的方式）通常接近正态分布，因此可能会选择一个接近噪声偏移1/10的值。这样可以保证噪声不会太大，也不会太小，以便模型能够有效地学习',
            )
        with gr.Row(visible=False) as noise_offset_multires:
            multires_noise_iterations = gr.Slider(
                label='Multires noise iterations',
                value=0,
                minimum=0,
                maximum=64,
                step=1,
                info='多分辨率噪声的次数。多分辨率噪声意味着在不同的细节级别（如高频细节和低频细节）上添加不同的噪声。推荐的值在6到10之间',
            )
            multires_noise_discount = gr.Slider(
                label='Multires noise discount',
                value=0,
                minimum=0,
                maximum=1,
                step=0.01,
                info='调整多分辨率噪声影响的参数。数值越大，噪声在细节级别上的影响越大。对于小型数据集，建议设置较小的值（如0.1到0.3），这可能是因为小型数据集可能更容易受到噪声的影响',
            
            )
        noise_offset_type.change(
            noise_offset_type_change,
            inputs=[noise_offset_type],
            outputs=[noise_offset_original, noise_offset_multires]
        )
    with gr.Row():
        caption_dropout_every_n_epochs = gr.Number(
            label='Dropout caption every n epochs', value=0,
            info='在训练过程中，每隔n个epoch，随机删除caption中的单词。这可以提高模型的泛化能力，但是可能会降低模型的精度',
        )
        caption_dropout_rate = gr.Slider(
            label='Rate of caption dropout', value=0, minimum=0, maximum=1,
            info='caption中单词被删除的概率',
        )
        vae_batch_size = gr.Slider(
            label='VAE batch size', minimum=0, maximum=32, value=0, step=1,
            info='VAE的batch size。如果设置为0，则不使用VAE',
        )
    with gr.Row():
        save_state = gr.Checkbox(label='Save training state', value=False,info='如果被勾选，将保存训练状态，以便在训练过程中恢复')
        resume = gr.Textbox(
            label='Resume from saved training state',
            placeholder='path to "last-state" state folder to resume from',
            info='如果设置了这个参数，模型将从上次保存的状态中恢复训练',
        )
        resume_button = gr.Button(
            '📂', elem_id='open_folder_small', visible=(not headless)
        )
        resume_button.click(
            get_folder_path,
            outputs=resume,
            show_progress=False,
        )
        max_train_epochs = gr.Textbox(
            label='Max train epoch',
            placeholder='(Optional) Override number of epoch',
            info='训练最大周期数(epoch)。每个epoch代表模型看完一次全部的训练数据。',
        )
        max_data_loader_n_workers = gr.Textbox(
            label='Max num workers for DataLoader',
            placeholder='(Optional) Override number of epoch. Default: 8',
            value='0',
            info='数据加载的最大工作线程数。在PyTorch中，DataLoader可以并行地加载数据，这个数值设定了同时工作的线程数，以便更快地加载数据。默认值是8',
        )
    with gr.Row():
        wandb_api_key = gr.Textbox(
            label='WANDB API Key',
            value='',
            placeholder='(Optional)',
            info='Weights & Biases（WANDB）API密钥。Weights & Biases是一个机器学习项目的实验跟踪和可视化工具。用户需要在WANDB网站生成自己的API密钥，然后输入到这个选项里: https://wandb.ai/login',
        )
        use_wandb = gr.Checkbox(
            label='WANDB Logging',
            value=False,
            info='如果被勾选，将使用Weights & Biases进行日志记录。如果没有勾选，将使用TensorBoard作为默认的日志记录工具',
        )
        scale_v_pred_loss_like_noise_pred = gr.Checkbox(
            label='Scale v prediction loss',
            value=False,
            info='只对SD v2模型有效。如果被勾选，将根据时间步长来调整预测损失，使得全局噪声预测和局部噪声预测的权重相同，从而可能期待提高细节的改进',
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
        noise_offset_type,
        noise_offset,
        adaptive_noise_scale,
        multires_noise_iterations,
        multires_noise_discount,
        additional_parameters,
        vae_batch_size,
        min_snr_gamma,
        save_every_n_steps,
        save_last_n_steps,
        save_last_n_steps_state,
        use_wandb,
        wandb_api_key,
        scale_v_pred_loss_like_noise_pred,
    )


def run_cmd_advanced_training(**kwargs):
    run_cmd = ''
    
    max_train_epochs = kwargs.get("max_train_epochs", "")
    if max_train_epochs:
        run_cmd += f' --max_train_epochs={max_train_epochs}'
        
    max_data_loader_n_workers = kwargs.get("max_data_loader_n_workers", "")
    if max_data_loader_n_workers:
        run_cmd += f' --max_data_loader_n_workers="{max_data_loader_n_workers}"'
    
    max_token_length = int(kwargs.get("max_token_length", 75))
    if max_token_length > 75:
        run_cmd += f' --max_token_length={max_token_length}'
        
    clip_skip = int(kwargs.get("clip_skip", 1))
    if clip_skip > 1:
        run_cmd += f' --clip_skip={clip_skip}'
        
    resume = kwargs.get("resume", "")
    if resume:
        run_cmd += f' --resume="{resume}"'
        
    keep_tokens = int(kwargs.get("keep_tokens", 0))
    if keep_tokens > 0:
        run_cmd += f' --keep_tokens="{keep_tokens}"'
        
    caption_dropout_every_n_epochs = int(kwargs.get("caption_dropout_every_n_epochs", 0))
    if caption_dropout_every_n_epochs > 0:
        run_cmd += f' --caption_dropout_every_n_epochs="{caption_dropout_every_n_epochs}"'
    
    caption_dropout_rate = float(kwargs.get("caption_dropout_rate", 0))
    if caption_dropout_rate > 0:
        run_cmd += f' --caption_dropout_rate="{caption_dropout_rate}"'
        
    vae_batch_size = int(kwargs.get("vae_batch_size", 0))
    if vae_batch_size > 0:
        run_cmd += f' --vae_batch_size="{vae_batch_size}"'
        
    bucket_reso_steps = int(kwargs.get("bucket_reso_steps", 64))
    run_cmd += f' --bucket_reso_steps={bucket_reso_steps}'
        
    save_every_n_steps = int(kwargs.get("save_every_n_steps", 0))
    if save_every_n_steps > 0:
        run_cmd += f' --save_every_n_steps="{save_every_n_steps}"'
        
    save_last_n_steps = int(kwargs.get("save_last_n_steps", 0))
    if save_last_n_steps > 0:
        run_cmd += f' --save_last_n_steps="{save_last_n_steps}"'
        
    save_last_n_steps_state = int(kwargs.get("save_last_n_steps_state", 0))
    if save_last_n_steps_state > 0:
        run_cmd += f' --save_last_n_steps_state="{save_last_n_steps_state}"'
        
    min_snr_gamma = int(kwargs.get("min_snr_gamma", 0))
    if min_snr_gamma >= 1:
        run_cmd += f' --min_snr_gamma={min_snr_gamma}'
    
    save_state = kwargs.get('save_state')
    if save_state:
        run_cmd += ' --save_state'
        
    mem_eff_attn = kwargs.get('mem_eff_attn')
    if mem_eff_attn:
        run_cmd += ' --mem_eff_attn'
    
    color_aug = kwargs.get('color_aug')
    if color_aug:
        run_cmd += ' --color_aug'
    
    flip_aug = kwargs.get('flip_aug')
    if flip_aug:
        run_cmd += ' --flip_aug'
    
    shuffle_caption = kwargs.get('shuffle_caption')
    if shuffle_caption:
        run_cmd += ' --shuffle_caption'
    
    gradient_checkpointing = kwargs.get('gradient_checkpointing')
    if gradient_checkpointing:
        run_cmd += ' --gradient_checkpointing'
    
    full_fp16 = kwargs.get('full_fp16')
    if full_fp16:
        run_cmd += ' --full_fp16'
    
    xformers = kwargs.get('xformers')
    if xformers:
        run_cmd += ' --xformers'
    
    persistent_data_loader_workers = kwargs.get('persistent_data_loader_workers')
    if persistent_data_loader_workers:
        run_cmd += ' --persistent_data_loader_workers'
    
    bucket_no_upscale = kwargs.get('bucket_no_upscale')
    if bucket_no_upscale:
        run_cmd += ' --bucket_no_upscale'
    
    random_crop = kwargs.get('random_crop')
    if random_crop:
        run_cmd += ' --random_crop'
        
    scale_v_pred_loss_like_noise_pred = kwargs.get('scale_v_pred_loss_like_noise_pred')
    if scale_v_pred_loss_like_noise_pred:
        run_cmd += ' --scale_v_pred_loss_like_noise_pred'
        
    noise_offset_type = kwargs.get('noise_offset_type', 'Original')
    if noise_offset_type == 'Original':
        noise_offset = float(kwargs.get("noise_offset", 0))
        if noise_offset > 0:
            run_cmd += f' --noise_offset={noise_offset}'
        
        adaptive_noise_scale = float(kwargs.get("adaptive_noise_scale", 0))
        if adaptive_noise_scale != 0 and noise_offset > 0:
            run_cmd += f' --adaptive_noise_scale={adaptive_noise_scale}'
    else:
        multires_noise_iterations = int(kwargs.get("multires_noise_iterations", 0))
        if multires_noise_iterations > 0:
            run_cmd += f' --multires_noise_iterations="{multires_noise_iterations}"'
        
        multires_noise_discount = float(kwargs.get("multires_noise_discount", 0))
        if multires_noise_discount > 0:
            run_cmd += f' --multires_noise_discount="{multires_noise_discount}"'
    
    additional_parameters = kwargs.get("additional_parameters", "")
    if additional_parameters:
        run_cmd += f' {additional_parameters}'
    
    use_wandb = kwargs.get('use_wandb')
    if use_wandb:
        run_cmd += ' --log_with wandb'
    
    wandb_api_key = kwargs.get("wandb_api_key", "")
    if wandb_api_key:
        run_cmd += f' --wandb_api_key="{wandb_api_key}"'
        
    return run_cmd

def verify_image_folder_pattern(folder_path):
    false_response = True # temporarily set to true to prevent stopping training in case of false positive
    true_response = True

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        log.error(f"The provided path '{folder_path}' is not a valid folder. Please follow the folder structure documentation found at docs\image_folder_structure.md ...")
        return false_response

    # Create a regular expression pattern to match the required sub-folder names
    # The pattern should start with one or more digits (\d+) followed by an underscore (_)
    # After the underscore, it should match one or more word characters (\w+), which can be letters, numbers, or underscores
    # Example of a valid pattern matching name: 123_example_folder
    pattern = r'^\d+_\w+'

    # Get the list of sub-folders in the directory
    subfolders = [
        os.path.join(folder_path, subfolder)
        for subfolder in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, subfolder))
    ]

    # Check the pattern of each sub-folder
    matching_subfolders = [subfolder for subfolder in subfolders if re.match(pattern, os.path.basename(subfolder))]

    # Print non-matching sub-folders
    non_matching_subfolders = set(subfolders) - set(matching_subfolders)
    if non_matching_subfolders:
        log.error(f"The following folders do not match the required pattern <number>_<text>: {', '.join(non_matching_subfolders)}")
        log.error(f"Please follow the folder structure documentation found at docs\image_folder_structure.md ...")
        return false_response

    # Check if no sub-folders exist
    if not matching_subfolders:
        log.error(f"No image folders found in {folder_path}. Please follow the folder structure documentation found at docs\image_folder_structure.md ...")
        return false_response

    log.info(f'Valid image folder names found in: {folder_path}')
    return true_response
