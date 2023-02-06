import gradio as gr
from easygui import msgbox
import subprocess
from .common_gui import get_folder_path


def caption_images(train_data_dir, caption_extension, batch_size, thresh):
    # Check for caption_text_input
    # if caption_text_input == "":
    #     msgbox("Caption text is missing...")
    #     return

    # Check for images_dir_input
    if train_data_dir == '':
        msgbox('Image folder is missing...')
        return

    if caption_extension == '':
        msgbox('Please provide an extension for the caption files.')
        return

    print(f'Captioning files in {train_data_dir}...')
    run_cmd = f'accelerate launch "./finetune/tag_images_by_wd14_tagger.py"'
    run_cmd += f' --batch_size="{int(batch_size)}"'
    run_cmd += f' --thresh="{thresh}"'
    if caption_extension != '':
        run_cmd += f' --caption_extension="{caption_extension}"'
    run_cmd += f' "{train_data_dir}"'

    print(run_cmd)

    # Run the command
    subprocess.run(run_cmd)

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
                get_folder_path, outputs=train_data_dir
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

        caption_button = gr.Button('Caption images')

        caption_button.click(
            caption_images,
            inputs=[train_data_dir, caption_extension, batch_size, thresh],
        )
