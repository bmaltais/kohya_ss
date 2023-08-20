import gradio as gr
from easygui import msgbox
import subprocess
import os
from .common_gui import get_folder_path, add_pre_postfix

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

PYTHON = 'python3' if os.name == 'posix' else './venv/Scripts/python.exe'


def caption_images(
    train_data_dir,
    caption_ext,
    batch_size,
    max_data_loader_n_workers,
    max_length,
    model_id,
    prefix,
    postfix,
):
    # Check for images_dir_input
    if train_data_dir == '':
        msgbox('Image folder is missing...')
        return

    if caption_ext == '':
        msgbox('Please provide an extension for the caption files.')
        return

    log.info(f'GIT captioning files in {train_data_dir}...')
    run_cmd = f'{PYTHON} finetune/make_captions_by_git.py'
    if not model_id == '':
        run_cmd += f' --model_id="{model_id}"'
    run_cmd += f' --batch_size="{int(batch_size)}"'
    run_cmd += (
        f' --max_data_loader_n_workers="{int(max_data_loader_n_workers)}"'
    )
    run_cmd += f' --max_length="{int(max_length)}"'
    if caption_ext != '':
        run_cmd += f' --caption_extension="{caption_ext}"'
    run_cmd += f' "{train_data_dir}"'

    log.info(run_cmd)

    # Run the command
    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)

    # Add prefix and postfix
    add_pre_postfix(
        folder=train_data_dir,
        caption_file_ext=caption_ext,
        prefix=prefix,
        postfix=postfix,
    )

    log.info('...captioning done')


###
# Gradio UI
###


def gradio_git_caption_gui_tab(headless=False):
    with gr.Tab('GIT Captioning'):
        gr.Markdown(
            'This utility will use GIT to caption files for each images in a folder.'
        )
        with gr.Row():
            train_data_dir = gr.Textbox(
                label='Image folder to caption',
                placeholder='Directory containing the images to caption',
                interactive=True,
            )
            button_train_data_dir_input = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            button_train_data_dir_input.click(
                get_folder_path,
                outputs=train_data_dir,
                show_progress=False,
            )
        with gr.Row():
            caption_ext = gr.Textbox(
                label='Caption file extension',
                placeholder='Extention for caption file. eg: .caption, .txt',
                value='.txt',
                interactive=True,
            )

            prefix = gr.Textbox(
                label='Prefix to add to BLIP caption',
                placeholder='(Optional)',
                interactive=True,
            )

            postfix = gr.Textbox(
                label='Postfix to add to BLIP caption',
                placeholder='(Optional)',
                interactive=True,
            )

            batch_size = gr.Number(
                value=1, label='Batch size', interactive=True
            )

        with gr.Row():
            max_data_loader_n_workers = gr.Number(
                value=2, label='Number of workers', interactive=True
            )
            max_length = gr.Number(
                value=75, label='Max length', interactive=True
            )
            model_id = gr.Textbox(
                label='Model',
                placeholder='(Optional) model id for GIT in Hugging Face',
                interactive=True,
            )

        caption_button = gr.Button('Caption images')

        caption_button.click(
            caption_images,
            inputs=[
                train_data_dir,
                caption_ext,
                batch_size,
                max_data_loader_n_workers,
                max_length,
                model_id,
                prefix,
                postfix,
            ],
            show_progress=False,
        )
