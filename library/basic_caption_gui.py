import gradio as gr
from easygui import msgbox
import subprocess
from .common_gui import get_folder_path, add_pre_postfix, find_replace


def caption_images(
    caption_text_input,
    images_dir_input,
    overwrite_input,
    caption_file_ext,
    prefix,
    postfix,
    find,
    replace,
):
    # Check for images_dir_input
    if images_dir_input == '':
        msgbox('Image folder is missing...')
        return

    if caption_file_ext == '':
        msgbox('Please provide an extension for the caption files.')
        return

    if not caption_text_input == '':
        print(
            f'Captioning files in {images_dir_input} with {caption_text_input}...'
        )
        run_cmd = f'python "tools/caption.py"'
        run_cmd += f' --caption_text="{caption_text_input}"'
        if overwrite_input:
            run_cmd += f' --overwrite'
        if caption_file_ext != '':
            run_cmd += f' --caption_file_ext="{caption_file_ext}"'
        run_cmd += f' "{images_dir_input}"'

        print(run_cmd)

        # Run the command
        subprocess.run(run_cmd)

    if overwrite_input:
        if not prefix == '' or not postfix == '':
            # Add prefix and postfix
            add_pre_postfix(
                folder=images_dir_input,
                caption_file_ext=caption_file_ext,
                prefix=prefix,
                postfix=postfix,
            )
        if not find == '':
            find_replace(
                folder=images_dir_input,
                caption_file_ext=caption_file_ext,
                find=find,
                replace=replace,
            )
    else:
        if not prefix == '' or not postfix == '':
            msgbox(
                'Could not modify caption files with requested change because the "Overwrite existing captions in folder" option is not selected...'
            )

    print('...captioning done')


###
# Gradio UI
###


def gradio_basic_caption_gui_tab():
    with gr.Tab('Basic Captioning'):
        gr.Markdown(
            'This utility will allow the creation of simple caption files for each images in a folder.'
        )
        with gr.Row():
            images_dir_input = gr.Textbox(
                label='Image folder to caption',
                placeholder='Directory containing the images to caption',
                interactive=True,
            )
            button_images_dir_input = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            button_images_dir_input.click(
                get_folder_path, outputs=images_dir_input
            )
            caption_file_ext = gr.Textbox(
                label='Caption file extension',
                placeholder='Extention for caption file. eg: .caption, .txt',
                value='.txt',
                interactive=True,
            )
            overwrite_input = gr.Checkbox(
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
            caption_text_input = gr.Textbox(
                label='Caption text',
                placeholder='Eg: , by some artist. Leave empty if you just want to add pre or postfix',
                interactive=True,
            )
            postfix = gr.Textbox(
                label='Postfix to add to caption',
                placeholder='(Optional)',
                interactive=True,
            )
        with gr.Row():
            find = gr.Textbox(
                label='Find text',
                placeholder='Eg: , by some artist. Leave empty if you just want to add pre or postfix',
                interactive=True,
            )
            replace = gr.Textbox(
                label='Replacement text',
                placeholder='Eg: , by some artist. Leave empty if you just want to replace with nothing',
                interactive=True,
            )
        caption_button = gr.Button('Caption images')

        caption_button.click(
            caption_images,
            inputs=[
                caption_text_input,
                images_dir_input,
                overwrite_input,
                caption_file_ext,
                prefix,
                postfix,
                find,
                replace,
            ],
        )
