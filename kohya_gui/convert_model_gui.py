import gradio as gr
from easygui import msgbox
import subprocess
import os
import sys
from .common_gui import get_folder_path, get_file_path, scriptdir, list_files, list_dirs

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

PYTHON = sys.executable


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
    if source_model_type == "":
        msgbox("Invalid source model type")
        return

    # Check if source model exist
    if os.path.isfile(source_model_input):
        log.info("The provided source model is a file")
    elif os.path.isdir(source_model_input):
        log.info("The provided model is a folder")
    else:
        msgbox("The provided source model is neither a file nor a folder")
        return

    # Check if source model exist
    if os.path.isdir(target_model_folder_input):
        log.info("The provided model folder exist")
    else:
        msgbox("The provided target folder does not exist")
        return

    run_cmd = (
        rf'"{PYTHON}" "{scriptdir}/sd-scripts/tools/convert_diffusers20_original_sd.py"'
    )

    v1_models = [
        "runwayml/stable-diffusion-v1-5",
        "CompVis/stable-diffusion-v1-4",
    ]

    # check if v1 models
    if str(source_model_type) in v1_models:
        log.info("SD v1 model specified. Setting --v1 parameter")
        run_cmd += " --v1"
    else:
        log.info("SD v2 model specified. Setting --v2 parameter")
        run_cmd += " --v2"

    if not target_save_precision_type == "unspecified":
        run_cmd += f" --{target_save_precision_type}"

    if target_model_type == "diffuser" or target_model_type == "diffuser_safetensors":
        run_cmd += f' --reference_model="{source_model_type}"'

    if target_model_type == "diffuser_safetensors":
        run_cmd += " --use_safetensors"

    # Fix for stabilityAI diffusers format. When saving v2 models in Diffusers format in training scripts and conversion scripts,
    # it was found that the U-Net configuration is different from those of Hugging Face's stabilityai models (this repository is
    # "use_linear_projection": false, stabilityai is true). Please note that the weight shapes are different, so please be careful
    # when using the weight files directly.

    if unet_use_linear_projection:
        run_cmd += " --unet_use_linear_projection"

    run_cmd += f' "{source_model_input}"'

    if target_model_type == "diffuser" or target_model_type == "diffuser_safetensors":
        target_model_path = os.path.join(
            target_model_folder_input, target_model_name_input
        )
        run_cmd += f' "{target_model_path}"'
    else:
        target_model_path = os.path.join(
            target_model_folder_input,
            f"{target_model_name_input}.{target_model_type}",
        )
        run_cmd += f' "{target_model_path}"'

    log.info(run_cmd)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        rf"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"
    )

    # Run the command
    subprocess.run(run_cmd, env=env)


###
# Gradio UI
###


def gradio_convert_model_tab(headless=False):
    from .common_gui import create_refresh_button

    default_source_model = os.path.join(scriptdir, "outputs")
    default_target_folder = os.path.join(scriptdir, "outputs")
    current_source_model = default_source_model
    current_target_folder = default_target_folder

    def list_source_model(path):
        nonlocal current_source_model
        current_source_model = path
        return list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

    def list_target_folder(path):
        nonlocal current_target_folder
        current_target_folder = path
        return list(list_dirs(path))

    with gr.Tab("Convert model"):
        gr.Markdown(
            "This utility can be used to convert from one stable diffusion model format to another."
        )

        model_ext = gr.Textbox(value="*.safetensors *.ckpt", visible=False)
        model_ext_name = gr.Textbox(value="Model types", visible=False)

        with gr.Group(), gr.Row():
            with gr.Column(), gr.Row():
                source_model_input = gr.Dropdown(
                    label="Source model (path to source model folder of file to convert...)",
                    interactive=True,
                    choices=[""] + list_source_model(default_source_model),
                    value="",
                    allow_custom_value=True,
                )
                create_refresh_button(
                    source_model_input,
                    lambda: None,
                    lambda: {"choices": list_source_model(current_source_model)},
                    "open_folder_small",
                )
                button_source_model_dir = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not headless),
                )
                button_source_model_dir.click(
                    get_folder_path,
                    outputs=source_model_input,
                    show_progress=False,
                )

                button_source_model_file = gr.Button(
                    document_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not headless),
                )
                button_source_model_file.click(
                    get_file_path,
                    inputs=[source_model_input, model_ext, model_ext_name],
                    outputs=source_model_input,
                    show_progress=False,
                )

                source_model_input.change(
                    fn=lambda path: gr.Dropdown(choices=[""] + list_source_model(path)),
                    inputs=source_model_input,
                    outputs=source_model_input,
                    show_progress=False,
                )
            with gr.Column(), gr.Row():
                source_model_type = gr.Dropdown(
                    label="Source model type",
                    choices=[
                        "stabilityai/stable-diffusion-2-1-base",
                        "stabilityai/stable-diffusion-2-base",
                        "stabilityai/stable-diffusion-2-1",
                        "stabilityai/stable-diffusion-2",
                        "runwayml/stable-diffusion-v1-5",
                        "CompVis/stable-diffusion-v1-4",
                    ],
                )
        with gr.Group(), gr.Row():
            with gr.Column(), gr.Row():
                target_model_folder_input = gr.Dropdown(
                    label="Target model folder (path to target model folder of file name to create...)",
                    interactive=True,
                    choices=[""] + list_target_folder(default_target_folder),
                    value="",
                    allow_custom_value=True,
                )
                create_refresh_button(
                    target_model_folder_input,
                    lambda: None,
                    lambda: {"choices": list_target_folder(current_target_folder)},
                    "open_folder_small",
                )
                button_target_model_folder = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not headless),
                )
                button_target_model_folder.click(
                    get_folder_path,
                    outputs=target_model_folder_input,
                    show_progress=False,
                )

                target_model_folder_input.change(
                    fn=lambda path: gr.Dropdown(
                        choices=[""] + list_target_folder(path)
                    ),
                    inputs=target_model_folder_input,
                    outputs=target_model_folder_input,
                    show_progress=False,
                )

            with gr.Column(), gr.Row():
                target_model_name_input = gr.Textbox(
                    label="Target model name",
                    placeholder="target model name...",
                    interactive=True,
                )
        with gr.Row():
            target_model_type = gr.Dropdown(
                label="Target model type",
                choices=[
                    "diffuser",
                    "diffuser_safetensors",
                    "ckpt",
                    "safetensors",
                ],
            )
            target_save_precision_type = gr.Dropdown(
                label="Target model precision",
                choices=["unspecified", "fp16", "bf16", "float"],
                value="unspecified",
            )
            unet_use_linear_projection = gr.Checkbox(
                label="UNet linear projection",
                value=False,
                info="Enable for Hugging Face's stabilityai models",
            )

        convert_button = gr.Button("Convert model")

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
