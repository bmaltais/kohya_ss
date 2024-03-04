import gradio as gr
from easygui import msgbox
import subprocess
import os
import sys
from .common_gui import (
    get_saveasfilename_path,
    get_file_path,
    is_file_writable,
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


def extract_lora(
    model_tuned,
    model_org,
    save_to,
    save_precision,
    dim,
    v2,
    sdxl,
    conv_dim,
    clamp_quantile,
    min_diff,
    device,
    load_original_model_to,
    load_tuned_model_to,
    load_precision,
):
    # Check for caption_text_input
    if model_tuned == '':
        log.info('Invalid finetuned model file')
        return

    if model_org == '':
        log.info('Invalid base model file')
        return

    # Check if source model exist
    if not os.path.isfile(model_tuned):
        log.info('The provided finetuned model is not a file')
        return

    if not os.path.isfile(model_org):
        log.info('The provided base model is not a file')
        return

    if not is_file_writable(save_to):
        return

    run_cmd = (
        fr'{PYTHON} "{scriptdir}/sd-scripts/networks/extract_lora_from_models.py"'
    )
    run_cmd += f' --load_precision {load_precision}'
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
    if sdxl:
        run_cmd += f' --sdxl'
    run_cmd += f' --clamp_quantile {clamp_quantile}'
    run_cmd += f' --min_diff {min_diff}'
    if sdxl:
        run_cmd += f' --load_original_model_to {load_original_model_to}'
        run_cmd += f' --load_tuned_model_to {load_tuned_model_to}'

    log.info(run_cmd)

    env = os.environ.copy()
    env['PYTHONPATH'] = fr"{scriptdir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Run the command
    subprocess.run(run_cmd, shell=True, env=env)


###
# Gradio UI
###


def gradio_extract_lora_tab(headless=False):
    def change_sdxl(sdxl):
        return gr.Dropdown(visible=sdxl), gr.Dropdown(visible=sdxl)
            
    
    
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
                elem_classes=['tool'],
                visible=(not headless),
            )
            button_model_tuned_file.click(
                get_file_path,
                inputs=[model_tuned, model_ext, model_ext_name],
                outputs=model_tuned,
                show_progress=False,
            )
            load_tuned_model_to = gr.Dropdown(
                label='Load finetuned model to',
                choices=['cpu', 'cuda', 'cuda:0'],
                value='cpu',
                interactive=True, scale=1,
                info="only for SDXL",
                visible=False,
            )
        with gr.Row():
            model_org = gr.Textbox(
                label='Stable Diffusion base model',
                placeholder='Stable Diffusion original model: ckpt or safetensors file',
                interactive=True,
            )
            button_model_org_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                elem_classes=['tool'],
                visible=(not headless),
            )
            button_model_org_file.click(
                get_file_path,
                inputs=[model_org, model_ext, model_ext_name],
                outputs=model_org,
                show_progress=False,
            )
            load_original_model_to = gr.Dropdown(
                label='Load Stable Diffusion base model to',
                choices=['cpu', 'cuda', 'cuda:0'],
                value='cpu',
                interactive=True, scale=1,
                info="only for SDXL",
                visible=False,
            )
        with gr.Row():
            save_to = gr.Textbox(
                label='Save to',
                placeholder='path where to save the extracted LoRA model...',
                interactive=True,
                scale=2,
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
            save_precision = gr.Dropdown(
                label='Save precision',
                choices=['fp16', 'bf16', 'float'],
                value='fp16',
                interactive=True, scale=1,
            )
            load_precision = gr.Dropdown(
                label='Load precision',
                choices=['fp16', 'bf16', 'float'],
                value='fp16',
                interactive=True, scale=1,
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
            clamp_quantile = gr.Number(
                label='Clamp Quantile',
                value=1,
                interactive=True,
            )
            min_diff = gr.Number(
                label='Minimum difference',
                value=0.01,
                interactive=True,
            )
        with gr.Row():
            v2 = gr.Checkbox(label='v2', value=False, interactive=True)
            sdxl = gr.Checkbox(label='SDXL', value=False, interactive=True)
            device = gr.Dropdown(
                label='Device',
                choices=[
                    'cpu',
                    'cuda',
                ],
                value='cuda',
                interactive=True,
            )
            
            sdxl.change(change_sdxl, inputs=sdxl, outputs=[load_tuned_model_to, load_original_model_to])

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
                sdxl,
                conv_dim,
                clamp_quantile,
                min_diff,
                device,
                load_original_model_to,
                load_tuned_model_to,
                load_precision,
            ],
            show_progress=False,
        )
