import gradio as gr
from easygui import msgbox
import subprocess
import os
import sys
from .common_gui import (
    get_saveasfilename_path,
    get_file_path,
    scriptdir,
)

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄

PYTHON = sys.executable


def merge_lycoris(
    base_model,
    lycoris_model,
    weight,
    output_name,
    dtype,
    device,
    is_sdxl,
    is_v2,
):
    log.info('Merge model...')

    run_cmd = fr'{PYTHON} "{scriptdir}/tools/merge_lycoris.py"'
    run_cmd += f' "{base_model}"'
    run_cmd += f' "{lycoris_model}"'
    run_cmd += f' "{output_name}"'
    run_cmd += f' --weight {weight}'
    run_cmd += f' --device {device}'
    run_cmd += f' --dtype {dtype}'
    if is_sdxl:
        run_cmd += f' --is_sdxl'
    if is_v2:
        run_cmd += f' --is_v2'

    log.info(run_cmd)

    env = os.environ.copy()
    env['PYTHONPATH'] = fr"{scriptdir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Run the command
    subprocess.run(run_cmd, shell=True, env=env)

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
                elem_classes=['tool'],
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
                elem_classes=['tool'],
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
                elem_classes=['tool'],
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
                    'cuda',
                ],
                value='cpu',
                interactive=True,
            )

            is_sdxl = gr.Checkbox(label='is sdxl', value=False, interactive=True)
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
                is_sdxl,
                is_v2,
            ],
            show_progress=False,
        )
