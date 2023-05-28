import gradio as gr
from easygui import msgbox
import subprocess
from .common_gui import get_folder_path
import os

PYTHON = 'python3' if os.name == 'posix' else './venv/Scripts/python.exe'

def group_images(
    input_folder,
    output_folder,
    group_size,
    include_subfolders,
    do_not_copy_other_files
):
    if input_folder == '':
        msgbox('Input folder is missing...')
        return

    if output_folder == '':
        msgbox('Please provide an output folder.')
        return

    print(f'Grouping images in {input_folder}...')

    run_cmd = f'{PYTHON} "{os.path.join("tools","group_images.py")}"'
    run_cmd += f' "{input_folder}"'
    run_cmd += f' "{output_folder}"'
    run_cmd += f' {(group_size)}'
    if include_subfolders:
        run_cmd += f' --include_subfolders'
    if do_not_copy_other_files:
        run_cmd += f' --do_not_copy_other_files'

    print(run_cmd)

    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)

    print('...grouping done')


def gradio_group_images_gui_tab(headless=False):
    with gr.Tab('Group Images'):
        gr.Markdown('This utility will group images in a folder based on their aspect ratio.')
        
        with gr.Row():
            input_folder = gr.Textbox(
                label='Input folder',
                placeholder='Directory containing the images to group',
                interactive=True,
            )
            button_input_folder = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            button_input_folder.click(
                get_folder_path,
                outputs=input_folder,
                show_progress=False,
            )

            output_folder = gr.Textbox(
                label='Output folder',
                placeholder='Directory where the grouped images will be stored',
                interactive=True,
            )
            button_output_folder = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            button_output_folder.click(
                get_folder_path,
                outputs=output_folder,
                show_progress=False,
            )
        with gr.Row():
            group_size = gr.Slider(
                label='Group size',
                info='Number of images to group together',
                value='4',
                minimum=1, maximum=64, step=1,
                interactive=True,
            )

            include_subfolders = gr.Checkbox(
                label='Include Subfolders',
                value=False,
                info='Include images in subfolders as well',
            )

            do_not_copy_other_files = gr.Checkbox(
                label='Do not copy other files',
                value=False,
                info='Do not copy other files in the input folder to the output folder',
            )

        group_images_button = gr.Button('Group images')

        group_images_button.click(
            group_images,
            inputs=[
                input_folder,
                output_folder,
                group_size,
                include_subfolders,
                do_not_copy_other_files
            ],
            show_progress=False,
        )
