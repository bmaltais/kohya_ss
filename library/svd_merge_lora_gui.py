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


def svd_merge_lora(
    lora_a_model,
    lora_b_model,
    lora_c_model,
    lora_d_model,
    ratio_a,
    ratio_b,
    ratio_c,
    ratio_d,
    save_to,
    precision,
    save_precision,
    new_rank,
    new_conv_rank,
    device,
):
    # Check if the output file already exists
    if os.path.isfile(save_to):
        print(f"Output file '{save_to}' already exists. Aborting.")
        return
    
    # Check if the ratio total is equal to one. If not mormalise to 1
    total_ratio = ratio_a + ratio_b + ratio_c + ratio_d
    if total_ratio != 1:
        ratio_a /= total_ratio
        ratio_b /= total_ratio
        ratio_c /= total_ratio
        ratio_d /= total_ratio

    run_cmd = f'{PYTHON} "{os.path.join("networks","svd_merge_lora.py")}"'
    run_cmd += f' --save_precision {save_precision}'
    run_cmd += f' --precision {precision}'
    run_cmd += f' --save_to "{save_to}"'

    run_cmd_models = ' --models'
    run_cmd_ratios = ' --ratios'
    # Add non-empty models and their ratios to the command
    if lora_a_model:
        if not os.path.isfile(lora_a_model):
            msgbox('The provided model A is not a file')
            return
        run_cmd_models += f' "{lora_a_model}"'
        run_cmd_ratios += f' {ratio_a}'
    if lora_b_model:
        if not os.path.isfile(lora_b_model):
            msgbox('The provided model B is not a file')
            return
        run_cmd_models += f' "{lora_b_model}"'
        run_cmd_ratios += f' {ratio_b}'
    if lora_c_model:
        if not os.path.isfile(lora_c_model):
            msgbox('The provided model C is not a file')
            return
        run_cmd_models += f' "{lora_c_model}"'
        run_cmd_ratios += f' {ratio_c}'
    if lora_d_model:
        if not os.path.isfile(lora_d_model):
            msgbox('The provided model D is not a file')
            return
        run_cmd_models += f' "{lora_d_model}"'
        run_cmd_ratios += f' {ratio_d}'

    run_cmd += run_cmd_models
    run_cmd += run_cmd_ratios
    run_cmd += f' --device {device}'
    run_cmd += f' --new_rank "{new_rank}"'
    run_cmd += f' --new_conv_rank "{new_conv_rank}"'

    log.info(run_cmd)

    # Run the command
    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)


###
# Gradio UI
###


def gradio_svd_merge_lora_tab(headless=False):
    with gr.Tab('Merge LoRA (SVD)'):
        gr.Markdown('This utility can merge two LoRA networks together into a new LoRA.')

        lora_ext = gr.Textbox(value='*.safetensors *.pt', visible=False)
        lora_ext_name = gr.Textbox(value='LoRA model types', visible=False)

        with gr.Row():
            lora_a_model = gr.Textbox(
                label='LoRA model "A"',
                placeholder='Path to the LoRA A model',
                interactive=True,
            )
            button_lora_a_model_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_lora_a_model_file.click(
                get_file_path,
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
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_lora_b_model_file.click(
                get_file_path,
                inputs=[lora_b_model, lora_ext, lora_ext_name],
                outputs=lora_b_model,
                show_progress=False,
            )
        with gr.Row():
            ratio_a = gr.Slider(
                label='Merge ratio model A',
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.25,
                interactive=True,
            )
            ratio_b = gr.Slider(
                label='Merge ratio model B',
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.25,
                interactive=True,
            )
        with gr.Row():
            lora_c_model = gr.Textbox(
                label='LoRA model "C"',
                placeholder='Path to the LoRA C model',
                interactive=True,
            )
            button_lora_c_model_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_lora_c_model_file.click(
                get_file_path,
                inputs=[lora_c_model, lora_ext, lora_ext_name],
                outputs=lora_c_model,
                show_progress=False,
            )

            lora_d_model = gr.Textbox(
                label='LoRA model "D"',
                placeholder='Path to the LoRA D model',
                interactive=True,
            )
            button_lora_d_model_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_lora_d_model_file.click(
                get_file_path,
                inputs=[lora_d_model, lora_ext, lora_ext_name],
                outputs=lora_d_model,
                show_progress=False,
            )
        with gr.Row():
            ratio_c = gr.Slider(
                label='Merge ratio model C',
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.25,
                interactive=True,
            )
            ratio_d = gr.Slider(
                label='Merge ratio model D',
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.25,
                interactive=True,
            )
        with gr.Row():
            new_rank = gr.Slider(
                label='New Rank',
                minimum=1,
                maximum=1024,
                step=1,
                value=128,
                interactive=True,
            )
            new_conv_rank = gr.Slider(
                label='New Conv Rank',
                minimum=1,
                maximum=1024,
                step=1,
                value=128,
                interactive=True,
            )

        with gr.Row():
            save_to = gr.Textbox(
                label='Save to',
                placeholder='path for the new LoRA file to save...',
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
        with gr.Row():
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
            device = gr.Dropdown(
                label='Device',
                choices=[
                    'cpu',
                    'cuda',
                ],
                value='cuda',
                interactive=True,
            )

        convert_button = gr.Button('Merge model')

        convert_button.click(
            svd_merge_lora,
            inputs=[
                lora_a_model,
                lora_b_model,
                lora_c_model,
                lora_d_model,
                ratio_a,
                ratio_b,
                ratio_c,
                ratio_d,
                save_to,
                precision,
                save_precision,
                new_rank,
                new_conv_rank,
                device,
            ],
            show_progress=False,
        )
