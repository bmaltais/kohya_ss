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


def extract_lycoris_locon(
    db_model,
    base_model,
    output_name,
    device,
    is_v2,
    mode,
    linear_dim,
    conv_dim,
    linear_threshold,
    conv_threshold,
    linear_ratio,
    conv_ratio,
    linear_quantile,
    conv_quantile,
    use_sparse_bias,
    sparsity,
    disable_cp,
):
    # Check for caption_text_input
    if db_model == '':
        msgbox('Invalid finetuned model file')
        return

    if base_model == '':
        msgbox('Invalid base model file')
        return

    # Check if source model exist
    if not os.path.isfile(db_model):
        msgbox('The provided finetuned model is not a file')
        return

    if not os.path.isfile(base_model):
        msgbox('The provided base model is not a file')
        return

    run_cmd = f'{PYTHON} "{os.path.join("tools","lycoris_locon_extract.py")}"'
    if is_v2:
        run_cmd += f' --is_v2'
    run_cmd += f' --device {device}'
    run_cmd += f' --mode {mode}'
    run_cmd += f' --safetensors'
    if mode == 'fixed':
        run_cmd += f' --linear_dim {linear_dim}'
        run_cmd += f' --conv_dim {conv_dim}'
    if mode == 'threshold':
        run_cmd += f' --linear_threshold {linear_threshold}'
        run_cmd += f' --conv_threshold {conv_threshold}'
    if mode == 'ratio':
        run_cmd += f' --linear_ratio {linear_ratio}'
        run_cmd += f' --conv_ratio {conv_ratio}'
    if mode == 'quantile':
        run_cmd += f' --linear_quantile {linear_quantile}'
        run_cmd += f' --conv_quantile {conv_quantile}'
    if use_sparse_bias:
        run_cmd += f' --use_sparse_bias'
    run_cmd += f' --sparsity {sparsity}'
    if disable_cp:
        run_cmd += f' --disable_cp'
    run_cmd += f' "{base_model}"'
    run_cmd += f' "{db_model}"'
    run_cmd += f' "{output_name}"'

    log.info(run_cmd)

    # Run the command
    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)


###
# Gradio UI
###
# def update_mode(mode):
#     # 'fixed', 'threshold','ratio','quantile'
#     if mode == 'fixed':
#         return gr.Row.update(visible=True), gr.Row.update(visible=False), gr.Row.update(visible=False), gr.Row.update(visible=False)
#     if mode == 'threshold':
#         return gr.Row.update(visible=False), gr.Row.update(visible=True), gr.Row.update(visible=False), gr.Row.update(visible=False)
#     if mode == 'ratio':
#         return gr.Row.update(visible=False), gr.Row.update(visible=False), gr.Row.update(visible=True), gr.Row.update(visible=False)
#     if mode == 'threshold':
#         return gr.Row.update(visible=False), gr.Row.update(visible=False), gr.Row.update(visible=False), gr.Row.update(visible=True)


def update_mode(mode):
    # Create a list of possible mode values
    modes = ['fixed', 'threshold', 'ratio', 'quantile']

    # Initialize an empty list to store visibility updates
    updates = []

    # Iterate through the possible modes
    for m in modes:
        # Add a visibility update for each mode, setting it to True if the input mode matches the current mode in the loop
        updates.append(gr.Row.update(visible=(mode == m)))

    # Return the visibility updates as a tuple
    return tuple(updates)


def gradio_extract_lycoris_locon_tab(headless=False):
    with gr.Tab('Extract LyCORIS LoCON'):
        gr.Markdown(
            'This utility can extract a LyCORIS LoCon network from a finetuned model.'
        )
        lora_ext = gr.Textbox(
            value='*.safetensors', visible=False
        )   # lora_ext = gr.Textbox(value='*.safetensors *.pt', visible=False)
        lora_ext_name = gr.Textbox(value='LoRA model types', visible=False)
        model_ext = gr.Textbox(value='*.safetensors *.ckpt', visible=False)
        model_ext_name = gr.Textbox(value='Model types', visible=False)

        with gr.Row():
            db_model = gr.Textbox(
                label='Finetuned model',
                placeholder='Path to the finetuned model to extract',
                interactive=True,
            )
            button_db_model_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_db_model_file.click(
                get_file_path,
                inputs=[db_model, model_ext, model_ext_name],
                outputs=db_model,
                show_progress=False,
            )

            base_model = gr.Textbox(
                label='Stable Diffusion base model',
                placeholder='Stable Diffusion original model: ckpt or safetensors file',
                interactive=True,
            )
            button_base_model_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_base_model_file.click(
                get_file_path,
                inputs=[base_model, model_ext, model_ext_name],
                outputs=base_model,
                show_progress=False,
            )
        with gr.Row():
            output_name = gr.Textbox(
                label='Save to',
                placeholder='path where to save the extracted LoRA model...',
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
            device = gr.Dropdown(
                label='Device',
                choices=[
                    'cpu',
                    'cuda',
                ],
                value='cuda',
                interactive=True,
            )
            is_v2 = gr.Checkbox(label='is v2', value=False, interactive=True)
        mode = gr.Dropdown(
            label='Mode',
            choices=['fixed', 'threshold', 'ratio', 'quantile'],
            value='fixed',
            interactive=True,
        )
        with gr.Row(visible=True) as fixed:
            linear_dim = gr.Slider(
                minimum=1,
                maximum=1024,
                label='Network Dimension',
                value=1,
                step=1,
                interactive=True,
            )
            conv_dim = gr.Slider(
                minimum=1,
                maximum=1024,
                label='Conv Dimension',
                value=1,
                step=1,
                interactive=True,
            )
        with gr.Row(visible=False) as threshold:
            linear_threshold = gr.Slider(
                minimum=0,
                maximum=1,
                label='Linear threshold',
                value=0.65,
                step=0.01,
                interactive=True,
                info='The higher the value, the smaller the file. Recommended starting value: 0.65',
            )
            conv_threshold = gr.Slider(
                minimum=0,
                maximum=1,
                label='Conv threshold',
                value=0.65,
                step=0.01,
                interactive=True,
                info='The higher the value, the smaller the file. Recommended starting value: 0.65',
            )
        with gr.Row(visible=False) as ratio:
            linear_ratio = gr.Slider(
                minimum=0,
                maximum=1,
                label='Linear ratio',
                value=0.75,
                step=0.01,
                interactive=True,
                info='The higher the value, the smaller the file. Recommended starting value: 0.75',
            )
            conv_ratio = gr.Slider(
                minimum=0,
                maximum=1,
                label='Conv ratio',
                value=0.75,
                step=0.01,
                interactive=True,
                info='The higher the value, the smaller the file. Recommended starting value: 0.75',
            )
        with gr.Row(visible=False) as quantile:
            linear_quantile = gr.Slider(
                minimum=0,
                maximum=1,
                label='Linear quantile',
                value=0.75,
                step=0.01,
                interactive=True,
                info='The higher the value, the larger the file. Recommended starting value: 0.75',
            )
            conv_quantile = gr.Slider(
                minimum=0,
                maximum=1,
                label='Conv quantile',
                value=0.75,
                step=0.01,
                interactive=True,
                info='The higher the value, the larger the file. Recommended starting value: 0.75',
            )
        with gr.Row():
            use_sparse_bias = gr.Checkbox(
                label='Use sparse biais', value=False, interactive=True
            )
            sparsity = gr.Slider(
                minimum=0,
                maximum=1,
                label='Sparsity',
                value=0.98,
                step=0.01,
                interactive=True,
            )
            disable_cp = gr.Checkbox(
                label='Disable CP decomposition', value=False, interactive=True
            )
        mode.change(
            update_mode,
            inputs=[mode],
            outputs=[
                fixed,
                threshold,
                ratio,
                quantile,
            ],
        )

        extract_button = gr.Button('Extract LyCORIS LoCon')

        extract_button.click(
            extract_lycoris_locon,
            inputs=[
                db_model,
                base_model,
                output_name,
                device,
                is_v2,
                mode,
                linear_dim,
                conv_dim,
                linear_threshold,
                conv_threshold,
                linear_ratio,
                conv_ratio,
                linear_quantile,
                conv_quantile,
                use_sparse_bias,
                sparsity,
                disable_cp,
            ],
            show_progress=False,
        )
