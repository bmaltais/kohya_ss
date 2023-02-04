import gradio as gr
from easygui import msgbox
import subprocess
import os
from .common_gui import get_saveasfilename_path, get_file_path

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄


def resize_lora(
    model, new_rank, save_to, save_precision, device,
):
    # Check for caption_text_input
    if model == '':
        msgbox('Invalid model file')
        return

    # Check if source model exist
    if not os.path.isfile(model):
        msgbox('The provided model is not a file')
        return
    
    if device == '':
        device = 'cuda'

    run_cmd = f'.\\venv\Scripts\python.exe "networks\\resize_lora.py"'
    run_cmd += f' --save_precision {save_precision}'
    run_cmd += f' --save_to {save_to}'
    run_cmd += f' --model {model}'
    run_cmd += f' --new_rank {new_rank}'
    run_cmd += f' --device {device}'

    print(run_cmd)

    # Run the command
    subprocess.run(run_cmd)


###
# Gradio UI
###


def gradio_resize_lora_tab():
    with gr.Tab('Resize LoRA'):
        gr.Markdown(
            'This utility can resize a LoRA.'
        )
        
        lora_ext = gr.Textbox(value='*.pt *.safetensors', visible=False)
        lora_ext_name = gr.Textbox(value='LoRA model types', visible=False)
        
        with gr.Row():
            model = gr.Textbox(
                label='Source LoRA',
                placeholder='Path to the LoRA to resize',
                interactive=True,
            )
            button_lora_a_model_file = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            button_lora_a_model_file.click(
                get_file_path,
                inputs=[model, lora_ext, lora_ext_name],
                outputs=model,
            )
        with gr.Row():
            new_rank = gr.Slider(label="Desired LoRA rank", minimum=1, maximum=1024, step=1, value=4,
                interactive=True,)
        
        with gr.Row():
            save_to = gr.Textbox(
                label='Save to',
                placeholder='path for the LoRA file to save...',
                interactive=True,
            )
            button_save_to = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            button_save_to.click(
                get_saveasfilename_path, inputs=[save_to, lora_ext, lora_ext_name], outputs=save_to
            )
            save_precision = gr.Dropdown(
                label='Save precison',
                choices=['fp16', 'bf16', 'float'],
                value='fp16',
                interactive=True,
            )
            device = gr.Textbox(
                label='Device',
                placeholder='{Optional) device to use, cuda for GPU. Default: cuda',
                interactive=True,
            )

        convert_button = gr.Button('Resize model')

        convert_button.click(
            resize_lora,
            inputs=[model, new_rank, save_to, save_precision, device,
            ],
        )
