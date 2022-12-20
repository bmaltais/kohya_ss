import gradio as gr
from easygui import msgbox
import subprocess
import os
from .common_gui import get_folder_path, add_pre_postfix


def caption_images(
    train_data_dir,
    caption_file_ext,
    batch_size,
    num_beams,
    top_p,
    max_length,
    min_length,
    beam_search,
    prefix,
    postfix,
):
    # Check for caption_text_input
    # if caption_text_input == "":
    #     msgbox("Caption text is missing...")
    #     return

    # Check for images_dir_input
    if train_data_dir == '':
        msgbox('Image folder is missing...')
        return

    print(f'Captioning files in {train_data_dir}...')
    run_cmd = f'.\\venv\\Scripts\\python.exe "finetune/make_captions.py"'
    run_cmd += f' --batch_size="{int(batch_size)}"'
    run_cmd += f' --num_beams="{int(num_beams)}"'
    run_cmd += f' --top_p="{top_p}"'
    run_cmd += f' --max_length="{int(max_length)}"'
    run_cmd += f' --min_length="{int(min_length)}"'
    if beam_search:
        run_cmd += f' --beam_search'
    if caption_file_ext != '':
        run_cmd += f' --caption_extension="{caption_file_ext}"'
    run_cmd += f' "{train_data_dir}"'
    run_cmd += f' --caption_weights="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"'

    print(run_cmd)

    # Run the command
    subprocess.run(run_cmd)

    # Add prefix and postfix
    add_pre_postfix(
        folder=train_data_dir,
        caption_file_ext=caption_file_ext,
        prefix=prefix,
        postfix=postfix,
    )

    print('...captioning done')


###
# Gradio UI
###


def gradio_blip_caption_gui_tab():
    with gr.Tab('BLIP Captioning'):
        gr.Markdown(
            'This utility will use BLIP to caption files for each images in a folder.'
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
                get_folder_path, outputs=train_data_dir
            )
        with gr.Row():
            caption_file_ext = gr.Textbox(
                label='Caption file extension',
                placeholder='(Optional) Default: .caption',
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
            beam_search = gr.Checkbox(
                label='Use beam search', interactive=True, value=True
            )
            num_beams = gr.Number(
                value=1, label='Number of beams', interactive=True
            )
            top_p = gr.Number(value=0.9, label='Top p', interactive=True)
            max_length = gr.Number(
                value=75, label='Max length', interactive=True
            )
            min_length = gr.Number(
                value=5, label='Min length', interactive=True
            )

        caption_button = gr.Button('Caption images')

        caption_button.click(
            caption_images,
            inputs=[
                train_data_dir,
                caption_file_ext,
                batch_size,
                num_beams,
                top_p,
                max_length,
                min_length,
                beam_search,
                prefix,
                postfix,
            ],
        )
