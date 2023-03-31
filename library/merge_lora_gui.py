import os
import subprocess

import gradio as gr

from .common_gui_functions import (
    get_file_path, get_saveasfile_path,
)

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄
PYTHON = 'python3' if os.name == 'posix' else './venv/Scripts/python.exe'


def merge_lora(
    lora_a_model,
    lora_b_model,
    ratio,
    save_to,
    precision,
    save_precision,
):
    # Check for caption_text_input
    if lora_a_model == '':
        show_message_box('Invalid model A file')
        return

    if lora_b_model == '':
        show_message_box('Invalid model B file')
        return

    # Check if source model exist
    if not os.path.isfile(lora_a_model):
        show_message_box('The provided model A is not a file')
        return

    if not os.path.isfile(lora_b_model):
        show_message_box('The provided model B is not a file')
        return

    ratio_a = ratio
    ratio_b = 1 - ratio

    run_cmd = f'{PYTHON} "{os.path.join("networks","merge_lora.py")}"'
    run_cmd += f' --save_precision {save_precision}'
    run_cmd += f' --precision {precision}'
    run_cmd += f' --save_to "{save_to}"'
    run_cmd += f' --models "{lora_a_model}" "{lora_b_model}"'
    run_cmd += f' --ratios {ratio_a} {ratio_b}'

    print(run_cmd)

    # Run the command
    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)


###
# Gradio UI
###


def gradio_merge_lora_tab():
    with gr.Tab('Merge LoRA'):
        gr.Markdown('This utility can merge two LoRA networks together.')

        lora_ext = gr.Textbox(value='*.safetensors *.pt', visible=False)
        lora_ext_name = gr.Textbox(value='LoRA model types', visible=False)

        with gr.Row():
            lora_a_model = gr.Textbox(
                label='LoRA model "A"',
                placeholder='Path to the LoRA A model',
                interactive=True,
            )
            button_lora_a_model_file = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            button_lora_a_model_file.click(
                lambda *args, **kwargs: get_file_path(*args),
                inputs=[lora_a_model, lora_ext, lora_ext_name],
                outputs=lora_a_model,
                show_progress=False,
            )

            lora_b_model = gr.Textbox(
                label='LoRA model "B"',
                placeholder='Path to the LoRA B model',
                interactive=True,
            )
            button_lora_b_model_file = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            button_lora_b_model_file.click(
                lambda *args, **kwargs: get_file_path(*args),
                inputs=[lora_b_model, lora_ext, lora_ext_name],
                outputs=lora_b_model,
                show_progress=False,
            )
        with gr.Row():
            ratio = gr.Slider(
                label='Merge ratio (eg: 0.7 mean 70% of model A and 30% of model B',
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.5,
                interactive=True,
            )

        with gr.Row():
            save_to = gr.Textbox(
                label='Save to',
                placeholder='path for the file to save...',
                interactive=True,
            )
            button_save_to = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            button_save_to.click(
                get_saveasfile_path,
                inputs=[save_to, lora_ext, lora_ext_name],
                outputs=save_to,
                show_progress=False,
            )
            precision = gr.Dropdown(
                label='Merge precision',
                choices=['fp16', 'bf16', 'float'],
                value='float',
                interactive=True,
            )
            save_precision = gr.Dropdown(
                label='Save precision',
                choices=['fp16', 'bf16', 'float'],
                value='float',
                interactive=True,
            )

        convert_button = gr.Button('Merge model')

        convert_button.click(
            merge_lora,
            inputs=[
                lora_a_model,
                lora_b_model,
                ratio,
                save_to,
                precision,
                save_precision,
            ],
            show_progress=False,
        )
