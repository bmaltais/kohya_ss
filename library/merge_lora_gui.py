# Standard library imports
import os
import subprocess

# Third-party imports
import gradio as gr
from easygui import msgbox

# Local module imports
from .common_gui import get_saveasfilename_path, get_file_path
from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄
PYTHON = 'python3' if os.name == 'posix' else './venv/Scripts/python.exe'


def check_model(model):
    if not model:
        return True
    if not os.path.isfile(model):
        msgbox(f'The provided {model} is not a file')
        return False
    return True


def verify_conditions(sd_model, lora_models):
    lora_models_count = sum(1 for model in lora_models if model)
    if sd_model and lora_models_count >= 1:
        return True
    elif not sd_model and lora_models_count >= 2:
        return True
    return False


def merge_lora(
    sd_model,
    sdxl_model,
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
):
    log.info('Merge model...')
    models = [sd_model, lora_a_model, lora_b_model, lora_c_model, lora_d_model]
    lora_models = models[1:]
    ratios = [ratio_a, ratio_b, ratio_c, ratio_d]

    if not verify_conditions(sd_model, lora_models):
        log.info(
            'Warning: Either provide at least one LoRa model along with the sd_model or at least two LoRa models if no sd_model is provided.'
        )
        return

    for model in models:
        if not check_model(model):
            return

    if not sdxl_model:
        run_cmd = f'{PYTHON} "{os.path.join("networks","merge_lora.py")}"'
    else:
        run_cmd = f'{PYTHON} "{os.path.join("networks","sdxl_merge_lora.py")}"'
    if sd_model:
        run_cmd += f' --sd_model "{sd_model}"'
    run_cmd += f' --save_precision {save_precision}'
    run_cmd += f' --precision {precision}'
    run_cmd += f' --save_to "{save_to}"'

    # Create a space-separated string of non-empty models (from the second element onwards), enclosed in double quotes
    models_cmd = ' '.join([f'"{model}"' for model in lora_models if model])

    # Create a space-separated string of non-zero ratios corresponding to non-empty LoRa models
    valid_ratios = [ratios[i] for i, model in enumerate(lora_models) if model]
    ratios_cmd = ' '.join([str(ratio) for ratio in valid_ratios])

    if models_cmd:
        run_cmd += f' --models {models_cmd}'
        run_cmd += f' --ratios {ratios_cmd}'

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


def gradio_merge_lora_tab(headless=False):
    with gr.Tab('Merge LoRA'):
        gr.Markdown(
            'This utility can merge up to 4 LoRA together or alternatively merge up to 4 LoRA into a SD checkpoint.'
        )

        lora_ext = gr.Textbox(value='*.safetensors *.pt', visible=False)
        lora_ext_name = gr.Textbox(value='LoRA model types', visible=False)
        ckpt_ext = gr.Textbox(value='*.safetensors *.ckpt', visible=False)
        ckpt_ext_name = gr.Textbox(value='SD model types', visible=False)

        with gr.Row():
            sd_model = gr.Textbox(
                label='SD Model',
                placeholder='(Optional) Stable Diffusion model',
                interactive=True,
                info='Provide a SD file path IF you want to merge it with LoRA files',
            )
            sd_model_file = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            sd_model_file.click(
                get_file_path,
                inputs=[sd_model, ckpt_ext, ckpt_ext_name],
                outputs=sd_model,
                show_progress=False,
            )
            sdxl_model = gr.Checkbox(label='SDXL model', value=False)

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
                label='Model A merge ratio (eg: 0.5 mean 50%)',
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.0,
                interactive=True,
            )

            ratio_b = gr.Slider(
                label='Model B merge ratio (eg: 0.5 mean 50%)',
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.0,
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
                label='Model C merge ratio (eg: 0.5 mean 50%)',
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.0,
                interactive=True,
            )

            ratio_d = gr.Slider(
                label='Model D merge ratio (eg: 0.5 mean 50%)',
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.0,
                interactive=True,
            )

        with gr.Row():
            save_to = gr.Textbox(
                label='Save to',
                placeholder='path for the file to save...',
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
            precision = gr.Dropdown(
                label='Merge precision',
                choices=['fp16', 'bf16', 'float'],
                value='float',
                interactive=True,
            )
            save_precision = gr.Dropdown(
                label='Save precision',
                choices=['fp16', 'bf16', 'float'],
                value='fp16',
                interactive=True,
            )

        merge_button = gr.Button('Merge model')

        merge_button.click(
            merge_lora,
            inputs=[
                sd_model,
                sdxl_model,
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
            ],
            show_progress=False,
        )
