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


def extract_dylora(
    model,
    save_to,
    unit,
):
    # Check for caption_text_input
    if model == '':
        msgbox('Invalid DyLoRA model file')
        return

    # Check if source model exist
    if not os.path.isfile(model):
        msgbox('The provided DyLoRA model is not a file')
        return

    run_cmd = (
        fr'{PYTHON} "{scriptdir}/sd-scripts/networks/extract_lora_from_dylora.py"'
    )
    run_cmd += f' --save_to "{save_to}"'
    run_cmd += f' --model "{model}"'
    run_cmd += f' --unit {unit}'

    log.info(run_cmd)

    env = os.environ.copy()
    env['PYTHONPATH'] = fr"{scriptdir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Run the command
    subprocess.run(run_cmd, shell=True, env=env)

    log.info('Done extracting DyLoRA...')


###
# Gradio UI
###


def gradio_extract_dylora_tab(headless=False):
    with gr.Tab('Extract DyLoRA'):
        gr.Markdown(
            'This utility can extract a DyLoRA network from a finetuned model.'
        )
        lora_ext = gr.Textbox(value='*.safetensors *.pt', visible=False)
        lora_ext_name = gr.Textbox(value='LoRA model types', visible=False)

        with gr.Row():
            model = gr.Textbox(
                label='DyLoRA model',
                placeholder='Path to the DyLoRA model to extract from',
                interactive=True,
            )
            button_model_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                elem_classes=['tool'],
                visible=(not headless),
            )
            button_model_file.click(
                get_file_path,
                inputs=[model, lora_ext, lora_ext_name],
                outputs=model,
                show_progress=False,
            )

            save_to = gr.Textbox(
                label='Save to',
                placeholder='path where to save the extracted LoRA model...',
                interactive=True,
            )
            button_save_to = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                elem_classes=['tool'],
                visible=(not headless),
            )
            button_save_to.click(
                get_saveasfilename_path,
                inputs=[save_to, lora_ext, lora_ext_name],
                outputs=save_to,
                show_progress=False,
            )
            unit = gr.Slider(
                minimum=1,
                maximum=256,
                label='Network Dimension (Rank)',
                value=1,
                step=1,
                interactive=True,
            )

        extract_button = gr.Button('Extract LoRA model')

        extract_button.click(
            extract_dylora,
            inputs=[
                model,
                save_to,
                unit,
            ],
            show_progress=False,
        )
