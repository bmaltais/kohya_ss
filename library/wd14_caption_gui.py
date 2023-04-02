import os
import subprocess

import gradio as gr

from .common_gui_functions import get_folder_path


def replace_underscore_with_space(folder_path, file_extension):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(file_extension):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                file_content = file.read()
            new_file_content = file_content.replace('_', ' ')
            with open(file_path, 'w') as file:
                file.write(new_file_content)

def caption_images(
    train_data_dir, caption_extension, batch_size, thresh, replace_underscores
):
    # Check for caption_text_input
    # if caption_text_input == "":
    #     show_message_box("Caption text is missing...")
    #     return

    # Check for images_dir_input
    if train_data_dir == '':
        show_message_box('Image folder is missing...')
        return

    if caption_extension == '':
        show_message_box('Please provide an extension for the caption files.')
        return

    print(f'Captioning files in {train_data_dir}...')
    run_cmd = f'accelerate launch "./finetune/tag_images_by_wd14_tagger.py"'
    run_cmd += f' --batch_size="{int(batch_size)}"'
    run_cmd += f' --thresh="{thresh}"'
    run_cmd += f' --caption_extension="{caption_extension}"'
    run_cmd += f' "{train_data_dir}"'

    print(run_cmd)

    # Run the command
    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)
        
    if replace_underscores:
        replace_underscore_with_space(train_data_dir, caption_extension)

    print('...captioning done')


###
# Gradio UI
###


def gradio_wd14_caption_gui_tab():
    with gr.Tab('WD14 Captioning'):
        gr.Markdown(
            'This utility will use WD14 to caption files for each images in a folder.'
        )
        with gr.Row():
            train_data_dir = gr.Textbox(
                label='Image folder to caption',
                placeholder='Directory containing the images to caption',
                interactive=True,
            )
            button_train_data_dir_input = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            button_train_data_dir_input.click(
                get_folder_path,
                outputs=train_data_dir,
                show_progress=False,
            )

            caption_extension = gr.Textbox(
                label='Caption file extension',
                placeholder='Extention for caption file. eg: .caption, .txt',
                value='.txt',
                interactive=True,
            )
            thresh = gr.Number(value=0.35, label='Threshold')

            batch_size = gr.Number(
                value=1, label='Batch size', interactive=True
            )

            replace_underscores = gr.Checkbox(
                label='Replace underscores in filenames with spaces',
                value=False,
                interactive=True,
            )

        caption_button = gr.Button('Caption images')

        caption_button.click(
            caption_images,
            inputs=[
                train_data_dir,
                caption_extension,
                batch_size,
                thresh,
                replace_underscores,
            ],
            show_progress=False,
        )
