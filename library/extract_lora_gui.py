import gradio as gr
from easygui import msgbox
import subprocess
import os
from .common_gui import (
    get_saveasfilename_path,
    get_any_file_path,
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


def extract_lora(
    model_tuned,
    model_org,
    save_to,
    save_precision,
    dim,
    v2,
    conv_dim,
    device,
):
    # Check for caption_text_input
    if model_tuned == '':
        msgbox('Invalid finetuned model file')
        return

    if model_org == '':
        msgbox('Invalid base model file')
        return

    # Check if source model exist
    if not os.path.isfile(model_tuned):
        msgbox('The provided finetuned model is not a file')
        return

    if not os.path.isfile(model_org):
        msgbox('The provided base model is not a file')
        return

    run_cmd = (
        f'{PYTHON} "{os.path.join("networks","extract_lora_from_models.py")}"'
    )
    run_cmd += f' --save_precision {save_precision}'
    run_cmd += f' --save_to "{save_to}"'
    run_cmd += f' --model_org "{model_org}"'
    run_cmd += f' --model_tuned "{model_tuned}"'
    run_cmd += f' --dim {dim}'
    run_cmd += f' --device {device}'
    if conv_dim > 0:
        run_cmd += f' --conv_dim {conv_dim}'
    if v2:
        run_cmd += f' --v2'

    log.info(run_cmd)

    # Run the command
    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)


###
# Gradio UI
###


def gradio_extract_lora_tab(headless=False):
    with gr.Tab('Extract LoRA'):
        gr.Markdown(
            'This utility can extract a LoRA network from a finetuned model.'
        )
        lora_ext = gr.Textbox(value='*.safetensors *.pt', visible=False)
        lora_ext_name = gr.Textbox(value='LoRA model types', visible=False)
        model_ext = gr.Textbox(value='*.ckpt *.safetensors', visible=False)
        model_ext_name = gr.Textbox(value='Model types', visible=False)

        with gr.Row():
            model_tuned = gr.Textbox(
                label='Finetuned model',
                placeholder='Path to the finetuned model to extract',
                interactive=True,
            )
            button_model_tuned_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_model_tuned_file.click(
                get_file_path,
                inputs=[model_tuned, model_ext, model_ext_name],
                outputs=model_tuned,
                show_progress=False,
            )

            model_org = gr.Textbox(
                label='Stable Diffusion base model',
                placeholder='Stable Diffusion original model: ckpt or safetensors file',
                interactive=True,
            )
            button_model_org_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_model_org_file.click(
                get_file_path,
                inputs=[model_org, model_ext, model_ext_name],
                outputs=model_org,
                show_progress=False,
            )
        with gr.Row():
            save_to = gr.Textbox(
                label='Save to',
                placeholder='path where to save the extracted LoRA model...',
                interactive=True,
            )
            button_save_to = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_save_to.click(
                get_saveasfilename_path,
                inputs=[save_to, lora_ext, lora_ext_name],
                outputs=save_to,
                show_progress=False,
            )
            save_precision = gr.Dropdown(
                label='Save precision',
                choices=['fp16', 'bf16', 'float'],
                value='float',
                interactive=True,
            )
        with gr.Row():
            dim = gr.Slider(
                minimum=4,
                maximum=1024,
                label='Network Dimension (Rank)',
                value=128,
                step=1,
                interactive=True,
            )
            conv_dim = gr.Slider(
                minimum=0,
                maximum=1024,
                label='Conv Dimension (Rank)',
                value=128,
                step=1,
                interactive=True,
            )
            v2 = gr.Checkbox(label='v2', value=False, interactive=True)
            device = gr.Dropdown(
                label='Device',
                choices=[
                    'cpu',
                    'cuda',
                ],
                value='cuda',
                interactive=True,
            )

        extract_button = gr.Button('Extract LoRA model')

        extract_button.click(
            extract_lora,
            inputs=[
                model_tuned,
                model_org,
                save_to,
                save_precision,
                dim,
                v2,
                conv_dim,
                device,
            ],
            show_progress=False,
        )
