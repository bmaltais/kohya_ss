import gradio as gr
from easygui import msgbox
import subprocess
from .common_gui import get_folder_path


def caption_images(
    caption_text_input, images_dir_input, overwrite_input, caption_file_ext
):
    # Check for caption_text_input
    if caption_text_input == '':
        msgbox('Caption text is missing...')
        return

    # Check for images_dir_input
    if images_dir_input == '':
        msgbox('Image folder is missing...')
        return

    print(
        f'Captionning files in {images_dir_input} with {caption_text_input}...'
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

    print('...captionning done')


###
# Gradio UI
###


def gradio_caption_gui_tab():
    with gr.Tab('Captionning'):
        gr.Markdown(
            'This utility will allow the creation of caption files for each images in a folder.'
        )
        with gr.Row():
            caption_text_input = gr.Textbox(
                label='Caption text',
                placeholder='Eg: , by some artist',
                interactive=True,
            )
            overwrite_input = gr.Checkbox(
                label='Overwrite existing captions in folder',
                interactive=True,
                value=False,
            )
            caption_file_ext = gr.Textbox(
                label='Caption file extension',
                placeholder='(Optional) Default: .caption',
                interactive=True,
            )
        with gr.Row():
            images_dir_input = gr.Textbox(
                label='Image forder to caption',
                placeholder='Directory containing the images to caption',
                interactive=True,
            )
            button_images_dir_input = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            button_images_dir_input.click(
                get_folder_path, outputs=images_dir_input
            )
        caption_button = gr.Button('Caption images')

        caption_button.click(
            caption_images,
            inputs=[
                caption_text_input,
                images_dir_input,
                overwrite_input,
                caption_file_ext,
            ],
        )
