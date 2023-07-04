import gradio as gr
from easygui import msgbox
import subprocess
import os
from .common_gui import get_saveasfilename_path, get_file_path

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

PYTHON = 'python3' if os.name == 'posix' else './venv/Scripts/python.exe'
folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄


def resize_lora(
    model,
    new_rank,
    save_to,
    save_precision,
    device,
    dynamic_method,
    dynamic_param,
    verbose,
):
    # Check for caption_text_input
    if model == '':
        msgbox('Invalid model file')
        return

    # Check if source model exist
    if not os.path.isfile(model):
        msgbox('The provided model is not a file')
        return

    if dynamic_method == 'sv_ratio':
        if float(dynamic_param) < 2:
            msgbox(
                f'Dynamic parameter for {dynamic_method} need to be 2 or greater...'
            )
            return

    if dynamic_method == 'sv_fro' or dynamic_method == 'sv_cumulative':
        if float(dynamic_param) < 0 or float(dynamic_param) > 1:
            msgbox(
                f'Dynamic parameter for {dynamic_method} need to be between 0 and 1...'
            )
            return

    # Check if save_to end with one of the defines extension. If not add .safetensors.
    if not save_to.endswith(('.pt', '.safetensors')):
        save_to += '.safetensors'

    if device == '':
        device = 'cuda'

    run_cmd = f'{PYTHON} "{os.path.join("networks","resize_lora.py")}"'
    run_cmd += f' --save_precision {save_precision}'
    run_cmd += f' --save_to "{save_to}"'
    run_cmd += f' --model "{model}"'
    run_cmd += f' --new_rank {new_rank}'
    run_cmd += f' --device {device}'
    if not dynamic_method == 'None':
        run_cmd += f' --dynamic_method {dynamic_method}'
        run_cmd += f' --dynamic_param {dynamic_param}'
    if verbose:
        run_cmd += f' --verbose'

    log.info(run_cmd)

    # Run the command
    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)

    log.info('Done resizing...')


###
# Gradio UI
###


def gradio_resize_lora_tab(headless=False):
    with gr.Tab('Resize LoRA'):
        gr.Markdown('This utility can resize a LoRA.')

        lora_ext = gr.Textbox(value='*.safetensors *.pt', visible=False)
        lora_ext_name = gr.Textbox(value='LoRA model types', visible=False)

        with gr.Row():
            model = gr.Textbox(
                label='Source LoRA',
                placeholder='Path to the LoRA to resize',
                interactive=True,
            )
            button_lora_a_model_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_lora_a_model_file.click(
                get_file_path,
                inputs=[model, lora_ext, lora_ext_name],
                outputs=model,
                show_progress=False,
            )
        with gr.Row():
            save_to = gr.Textbox(
                label='Save to',
                placeholder='path for the LoRA file to save...',
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
            new_rank = gr.Slider(
                label='Desired LoRA rank',
                minimum=1,
                maximum=1024,
                step=1,
                value=4,
                interactive=True,
            )
            dynamic_method = gr.Dropdown(
                choices=['None', 'sv_ratio', 'sv_fro', 'sv_cumulative'],
                value='sv_fro',
                label='Dynamic method',
                interactive=True,
            )
            dynamic_param = gr.Textbox(
                label='Dynamic parameter',
                value='0.9',
                interactive=True,
                placeholder='Value for the dynamic method selected.',
            )
        with gr.Row():
            
            verbose = gr.Checkbox(label='Verbose', value=True)
            save_precision = gr.Dropdown(
                label='Save precision',
                choices=['fp16', 'bf16', 'float'],
                value='fp16',
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

        convert_button = gr.Button('Resize model')

        convert_button.click(
            resize_lora,
            inputs=[
                model,
                new_rank,
                save_to,
                save_precision,
                device,
                dynamic_method,
                dynamic_param,
                verbose,
            ],
            show_progress=False,
        )
