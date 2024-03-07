import gradio as gr
from easygui import msgbox
import subprocess
from .common_gui import get_folder_path, add_pre_postfix, find_replace, scriptdir, list_dirs
import os
import sys

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

PYTHON = sys.executable

def caption_images(
    caption_text,
    images_dir,
    overwrite,
    caption_ext,
    prefix,
    postfix,
    find_text,
    replace_text,
):
    # Check if images_dir is provided
    if not images_dir:
        msgbox(
            'Image folder is missing. Please provide the directory containing the images to caption.'
        )
        return

    # Check if caption_ext is provided
    if not caption_ext:
        msgbox('Please provide an extension for the caption files.')
        return

    if caption_text:
        log.info(f'Captioning files in {images_dir} with {caption_text}...')

        # Build the command to run caption.py
        run_cmd = fr'{PYTHON} "{scriptdir}/sd-scripts/tools/caption.py"'
        run_cmd += f' --caption_text="{caption_text}"'

        # Add optional flags to the command
        if overwrite:
            run_cmd += f' --overwrite'
        if caption_ext:
            run_cmd += f' --caption_file_ext="{caption_ext}"'

        run_cmd += f' "{images_dir}"'

        log.info(run_cmd)

        env = os.environ.copy()
        env['PYTHONPATH'] = fr"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"

        # Run the command based on the operating system
        subprocess.run(run_cmd, shell=True, env=env)

    # Check if overwrite option is enabled
    if overwrite:
        if prefix or postfix:
            # Add prefix and postfix to caption files
            add_pre_postfix(
                folder=images_dir,
                caption_file_ext=caption_ext,
                prefix=prefix,
                postfix=postfix,
            )
        if find_text:
            # Find and replace text in caption files
            find_replace(
                folder_path=images_dir,
                caption_file_ext=caption_ext,
                search_text=find_text,
                replace_text=replace_text,
            )
    else:
        if prefix or postfix:
            # Show a message if modification is not possible without overwrite option enabled
            msgbox(
                'Could not modify caption files with requested change because the "Overwrite existing captions in folder" option is not selected.'
            )

    log.info('Captioning done.')


# Gradio UI
def gradio_basic_caption_gui_tab(headless=False, default_images_dir=None):
    from .common_gui import create_refresh_button

    default_images_dir = default_images_dir if default_images_dir is not None else os.path.join(scriptdir, "data")
    current_images_dir = default_images_dir

    def list_images_dirs(path):
        current_images_dir = path
        return list(list_dirs(path))

    with gr.Tab('Basic Captioning'):
        gr.Markdown(
            'This utility allows you to create simple caption files for each image in a folder.'
        )
        with gr.Group(), gr.Row():
            images_dir = gr.Dropdown(
                label='Image folder to caption (containing the images to caption)',
                choices=list_images_dirs(default_images_dir),
                value="",
                interactive=True,
                allow_custom_value=True,
            )
            create_refresh_button(images_dir, lambda: None, lambda: {"choices": list_images_dir(current_images_dir)},"open_folder_small")
            folder_button = gr.Button(
                '📂', elem_id='open_folder_small', elem_classes=["tool"], visible=(not headless)
            )
            folder_button.click(
                get_folder_path,
                outputs=images_dir,
                show_progress=False,
            )
            caption_ext = gr.Textbox(
                label='Caption file extension',
                placeholder='Extension for caption file (e.g., .caption, .txt)',
                value='.txt',
                interactive=True,
            )
            overwrite = gr.Checkbox(
                label='Overwrite existing captions in folder',
                interactive=True,
                value=False,
            )
        with gr.Row():
            prefix = gr.Textbox(
                label='Prefix to add to caption',
                placeholder='(Optional)',
                interactive=True,
            )
            caption_text = gr.Textbox(
                label='Caption text',
                placeholder='e.g., "by some artist". Leave empty if you only want to add a prefix or postfix.',
                interactive=True,
            )
            postfix = gr.Textbox(
                label='Postfix to add to caption',
                placeholder='(Optional)',
                interactive=True,
            )
        with gr.Group(), gr.Row():
            find_text = gr.Textbox(
                label='Find text',
                placeholder='e.g., "by some artist". Leave empty if you only want to add a prefix or postfix.',
                interactive=True,
            )
            replace_text = gr.Textbox(
                label='Replacement text',
                placeholder='e.g., "by some artist". Leave empty if you want to replace with nothing.',
                interactive=True,
            )
            caption_button = gr.Button('Caption images')
            caption_button.click(
                caption_images,
                inputs=[
                    caption_text,
                    images_dir,
                    overwrite,
                    caption_ext,
                    prefix,
                    postfix,
                    find_text,
                    replace_text,
                ],
                show_progress=False,
            )

        images_dir.change(
            fn=lambda path: gr.Dropdown().update(choices=list_images_dirs(path)),
            inputs=images_dir,
            outputs=images_dir,
            show_progress=False,
        )
