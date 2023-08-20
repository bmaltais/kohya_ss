import gradio as gr
from easygui import msgbox
import subprocess
import os
from .common_gui import (
    get_saveasfilename_path,
    get_file_path,
)

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄
PYTHON = 'python3' if os.name == 'posix' else './venv/Scripts/python.exe'


def merge_lycoris(
    base_model,
    lycoris_model,
    weight,
    output_name,
    dtype,
    device,
    is_v2,
):
    log.info('Merge model...')

    run_cmd = f'{PYTHON} "{os.path.join("tools","merge_lycoris.py")}"'
    run_cmd += f' "{base_model}"'
    run_cmd += f' "{lycoris_model}"'
    run_cmd += f' "{output_name}"'
    run_cmd += f' --weight {weight}'
    run_cmd += f' --device {device}'
    run_cmd += f' --dtype {dtype}'
    if is_v2:
        run_cmd += f' --is_v2'

    log.info(run_cmd)

    # Run the command
    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)

    log.info('Done merging...')


###
# Gradio UI
###


def gradio_merge_lycoris_tab(headless=False):
    with gr.Tab('Merge LyCORIS'):
        gr.Markdown(
            'This utility can merge a LyCORIS model into a SD checkpoint.'
        )

        lora_ext = gr.Textbox(value='*.safetensors *.pt', visible=False)
        lora_ext_name = gr.Textbox(value='LoRA model types', visible=False)
        ckpt_ext = gr.Textbox(value='*.safetensors *.ckpt', visible=False)
        ckpt_ext_name = gr.Textbox(value='SD model types', visible=False)

        with gr.Row():
            base_model = gr.Textbox(
                label='SD Model',
                placeholder='(Optional) Stable Diffusion base model',
                interactive=True,
                info='Provide a SD file path that you want to merge with the LyCORIS file',
            )
            base_model_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            base_model_file.click(
                get_file_path,
                inputs=[base_model, ckpt_ext, ckpt_ext_name],
                outputs=base_model,
                show_progress=False,
            )

        with gr.Row():
            lycoris_model = gr.Textbox(
                label='LyCORIS model',
                placeholder='Path to the LyCORIS model',
                interactive=True,
            )
            button_lycoris_model_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_lycoris_model_file.click(
                get_file_path,
                inputs=[lycoris_model, lora_ext, lora_ext_name],
                outputs=lycoris_model,
                show_progress=False,
            )

        with gr.Row():
            weight = gr.Slider(
                label='Model A merge ratio (eg: 0.5 mean 50%)',
                minimum=0,
                maximum=1,
                step=0.01,
                value=1.0,
                interactive=True,
            )

        with gr.Row():
            output_name = gr.Textbox(
                label='Save to',
                placeholder='path for the checkpoint file to save...',
                interactive=True,
            )
            button_output_name = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_output_name.click(
                get_saveasfilename_path,
                inputs=[output_name, lora_ext, lora_ext_name],
                outputs=output_name,
                show_progress=False,
            )
            dtype = gr.Dropdown(
                label='Save dtype',
                choices=[
                    'float',
                    'float16',
                    'float32',
                    'float64',
                    'bfloat',
                    'bfloat16',
                ],
                value='float16',
                interactive=True,
            )

            device = gr.Dropdown(
                label='Device',
                choices=[
                    'cpu',
                    #  'cuda',
                ],
                value='cpu',
                interactive=True,
            )

            is_v2 = gr.Checkbox(label='is v2', value=False, interactive=True)

        merge_button = gr.Button('Merge model')

        merge_button.click(
            merge_lycoris,
            inputs=[
                base_model,
                lycoris_model,
                weight,
                output_name,
                dtype,
                device,
                is_v2,
            ],
            show_progress=False,
        )
