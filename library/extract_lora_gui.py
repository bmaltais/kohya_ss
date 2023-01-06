import gradio as gr
from easygui import msgbox
import subprocess
import os
from .common_gui import get_saveasfilename_path, get_any_file_path, get_file_path

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄


def extract_lora(
    model_tuned, model_org, save_to, save_precision, dim, v2,
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

    run_cmd = f'.\\venv\Scripts\python.exe "networks\extract_lora_from_models.py"'
    run_cmd += f' --save_precision {save_precision}'
    run_cmd += f' --save_to "{save_to}"'
    run_cmd += f' --model_org "{model_org}"'
    run_cmd += f' --model_tuned "{model_tuned}"'
    run_cmd += f' --dim {dim}'
    if v2:
        run_cmd += f' --v2'

    print(run_cmd)

    # Run the command
    subprocess.run(run_cmd)


###
# Gradio UI
###


def gradio_extract_lora_tab():
    with gr.Tab('Extract LoRA'):
        gr.Markdown(
            'This utility can extract a LoRA network from a finetuned model.'
        )
        lora_ext = gr.Textbox(value='*.pt *.safetensors', visible=False)
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
                folder_symbol, elem_id='open_folder_small'
            )
            button_model_tuned_file.click(
                get_file_path,
                inputs=[model_tuned, model_ext, model_ext_name],
                outputs=model_tuned,
            )
            
            model_org = gr.Textbox(
                label='Stable Diffusion base model',
                placeholder='Stable Diffusion original model: ckpt or safetensors file',
                interactive=True,
            )
            button_model_org_file = gr.Button(
                folder_symbol, elem_id='open_folder_small'
            )
            button_model_org_file.click(
                get_file_path,
                inputs=[model_org, model_ext, model_ext_name],
                outputs=model_org,
            )
        with gr.Row():
            save_to = gr.Textbox(
                label='Save to',
                placeholder='path where to save the extracted LoRA model...',
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
                value='float',
                interactive=True,
            )
        with gr.Row():
            dim = gr.Slider(
                minimum=1,
                maximum=128,
                label='Network Dimension',
                value=8,
                step=1,
                interactive=True,
            )
            v2 = gr.Checkbox(label='v2', value=False, interactive=True)

        extract_button = gr.Button('Extract LoRA model')

        extract_button.click(
            extract_lora,
            inputs=[model_tuned, model_org, save_to, save_precision, dim, v2
            ],
        )
