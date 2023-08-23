import gradio as gr
from easygui import msgbox
import subprocess
import os
import shutil
from .common_gui import get_folder_path, get_file_path

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄
PYTHON = 'python3' if os.name == 'posix' else './venv/Scripts/python.exe'


def convert_model(
    source_model_input,
    source_model_type,
    target_model_folder_input,
    target_model_name_input,
    target_model_type,
    target_save_precision_type,
    unet_use_linear_projection,
):
    # Check for caption_text_input
    if source_model_type == '':
        msgbox('Invalid source model type')
        return

    # Check if source model exist
    if os.path.isfile(source_model_input):
        log.info('The provided source model is a file')
    elif os.path.isdir(source_model_input):
        log.info('The provided model is a folder')
    else:
        msgbox('The provided source model is neither a file nor a folder')
        return

    # Check if source model exist
    if os.path.isdir(target_model_folder_input):
        log.info('The provided model folder exist')
    else:
        msgbox('The provided target folder does not exist')
        return

    run_cmd = f'{PYTHON} "tools/convert_diffusers20_original_sd.py"'

    v1_models = [
        'runwayml/stable-diffusion-v1-5',
        'CompVis/stable-diffusion-v1-4',
    ]

    # check if v1 models
    if str(source_model_type) in v1_models:
        log.info('SD v1 model specified. Setting --v1 parameter')
        run_cmd += ' --v1'
    else:
        log.info('SD v2 model specified. Setting --v2 parameter')
        run_cmd += ' --v2'

    if not target_save_precision_type == 'unspecified':
        run_cmd += f' --{target_save_precision_type}'

    if (
        target_model_type == 'diffuser'
        or target_model_type == 'diffuser_safetensors'
    ):
        run_cmd += f' --reference_model="{source_model_type}"'

    if target_model_type == 'diffuser_safetensors':
        run_cmd += ' --use_safetensors'

    # Fix for stabilityAI diffusers format. When saving v2 models in Diffusers format in training scripts and conversion scripts,
    # it was found that the U-Net configuration is different from those of Hugging Face's stabilityai models (this repository is
    # "use_linear_projection": false, stabilityai is true). Please note that the weight shapes are different, so please be careful
    # when using the weight files directly.

    if unet_use_linear_projection:
        run_cmd += ' --unet_use_linear_projection'

    run_cmd += f' "{source_model_input}"'

    if (
        target_model_type == 'diffuser'
        or target_model_type == 'diffuser_safetensors'
    ):
        target_model_path = os.path.join(
            target_model_folder_input, target_model_name_input
        )
        run_cmd += f' "{target_model_path}"'
    else:
        target_model_path = os.path.join(
            target_model_folder_input,
            f'{target_model_name_input}.{target_model_type}',
        )
        run_cmd += f' "{target_model_path}"'

    log.info(run_cmd)

    # Run the command
    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)

    if (
        not target_model_type == 'diffuser'
        or target_model_type == 'diffuser_safetensors'
    ):

        v2_models = [
            'stabilityai/stable-diffusion-2-1-base',
            'stabilityai/stable-diffusion-2-base',
        ]
        v_parameterization = [
            'stabilityai/stable-diffusion-2-1',
            'stabilityai/stable-diffusion-2',
        ]

        if str(source_model_type) in v2_models:
            inference_file = os.path.join(
                target_model_folder_input, f'{target_model_name_input}.yaml'
            )
            log.info(f'Saving v2-inference.yaml as {inference_file}')
            shutil.copy(
                f'./v2_inference/v2-inference.yaml',
                f'{inference_file}',
            )

        if str(source_model_type) in v_parameterization:
            inference_file = os.path.join(
                target_model_folder_input, f'{target_model_name_input}.yaml'
            )
            log.info(f'Saving v2-inference-v.yaml as {inference_file}')
            shutil.copy(
                f'./v2_inference/v2-inference-v.yaml',
                f'{inference_file}',
            )


#   parser = argparse.ArgumentParser()
#   parser.add_argument("--v1", action='store_true',
#                       help='load v1.x model (v1 or v2 is required to load checkpoint) / 1.xのモデルを読み込む')
#   parser.add_argument("--v2", action='store_true',
#                       help='load v2.0 model (v1 or v2 is required to load checkpoint) / 2.0のモデルを読み込む')
#   parser.add_argument("--fp16", action='store_true',
#                       help='load as fp16 (Diffusers only) and save as fp16 (checkpoint only) / fp16形式で読み込み（Diffusers形式のみ対応）、保存する（checkpointのみ対応）')
#   parser.add_argument("--bf16", action='store_true', help='save as bf16 (checkpoint only) / bf16形式で保存する（checkpointのみ対応）')
#   parser.add_argument("--float", action='store_true',
#                       help='save as float (checkpoint only) / float(float32)形式で保存する（checkpointのみ対応）')
#   parser.add_argument("--epoch", type=int, default=0, help='epoch to write to checkpoint / checkpointに記録するepoch数の値')
#   parser.add_argument("--global_step", type=int, default=0,
#                       help='global_step to write to checkpoint / checkpointに記録するglobal_stepの値')
#   parser.add_argument("--reference_model", type=str, default=None,
#                       help="reference model for schduler/tokenizer, required in saving Diffusers, copy schduler/tokenizer from this / scheduler/tokenizerのコピー元のDiffusersモデル、Diffusers形式で保存するときに必要")

#   parser.add_argument("model_to_load", type=str, default=None,
#                       help="model to load: checkpoint file or Diffusers model's directory / 読み込むモデル、checkpointかDiffusers形式モデルのディレクトリ")
#   parser.add_argument("model_to_save", type=str, default=None,
#                       help="model to save: checkpoint (with extension) or Diffusers model's directory (without extension) / 変換後のモデル、拡張子がある場合はcheckpoint、ない場合はDiffusesモデルとして保存")


###
# Gradio UI
###


def gradio_convert_model_tab(headless=False):
    with gr.Tab('Convert model'):
        gr.Markdown(
            'This utility can be used to convert from one stable diffusion model format to another.'
        )

        model_ext = gr.Textbox(value='*.safetensors *.ckpt', visible=False)
        model_ext_name = gr.Textbox(value='Model types', visible=False)

        with gr.Row():
            source_model_input = gr.Textbox(
                label='Source model',
                placeholder='path to source model folder of file to convert...',
                interactive=True,
            )
            button_source_model_dir = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_source_model_dir.click(
                get_folder_path,
                outputs=source_model_input,
                show_progress=False,
            )

            button_source_model_file = gr.Button(
                document_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_source_model_file.click(
                get_file_path,
                inputs=[source_model_input, model_ext, model_ext_name],
                outputs=source_model_input,
                show_progress=False,
            )

            source_model_type = gr.Dropdown(
                label='Source model type',
                choices=[
                    'stabilityai/stable-diffusion-2-1-base',
                    'stabilityai/stable-diffusion-2-base',
                    'stabilityai/stable-diffusion-2-1',
                    'stabilityai/stable-diffusion-2',
                    'runwayml/stable-diffusion-v1-5',
                    'CompVis/stable-diffusion-v1-4',
                ],
            )
        with gr.Row():
            target_model_folder_input = gr.Textbox(
                label='Target model folder',
                placeholder='path to target model folder of file name to create...',
                interactive=True,
            )
            button_target_model_folder = gr.Button(
                folder_symbol,
                elem_id='open_folder_small',
                visible=(not headless),
            )
            button_target_model_folder.click(
                get_folder_path,
                outputs=target_model_folder_input,
                show_progress=False,
            )

            target_model_name_input = gr.Textbox(
                label='Target model name',
                placeholder='target model name...',
                interactive=True,
            )
            target_model_type = gr.Dropdown(
                label='Target model type',
                choices=[
                    'diffuser',
                    'diffuser_safetensors',
                    'ckpt',
                    'safetensors',
                ],
            )
            target_save_precision_type = gr.Dropdown(
                label='Target model precision',
                choices=['unspecified', 'fp16', 'bf16', 'float'],
                value='unspecified',
            )
            unet_use_linear_projection = gr.Checkbox(
                label='UNet linear projection',
                value=False,
                info="Enable for Hugging Face's stabilityai models",
            )

        convert_button = gr.Button('Convert model')

        convert_button.click(
            convert_model,
            inputs=[
                source_model_input,
                source_model_type,
                target_model_folder_input,
                target_model_name_input,
                target_model_type,
                target_save_precision_type,
                unet_use_linear_projection,
            ],
            show_progress=False,
        )
