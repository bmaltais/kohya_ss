import gradio as gr
import os
import subprocess
import sys
from .common_gui import (
    get_saveasfilename_path,
    get_file_path,
    scriptdir,
)
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

PYTHON = sys.executable


def convert_lcm(
    name,
    model_path,
    lora_scale,
    model_type
):
    run_cmd = fr'{PYTHON} "{scriptdir}/tools/lcm_convert.py"'
    # Construct the command to run the script
    run_cmd += f" --lora-scale {lora_scale}"
    run_cmd += f' --model "{model_path}"'
    run_cmd += f' --name "{name}"'
    
    if model_type == "SDXL":
        run_cmd += f" --sdxl"
    if model_type == "SSD-1B":
        run_cmd += f" --ssd-1b"

    log.info(run_cmd)

    env = os.environ.copy()
    env['PYTHONPATH'] = fr"{scriptdir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Run the command
    subprocess.run(run_cmd, shell=True, env=env)

    # Return a success message
    log.info("Done extracting...")


def gradio_convert_lcm_tab(headless=False):
    with gr.Tab("Convert to LCM"):
        gr.Markdown("This utility convert a model to an LCM model.")
        lora_ext = gr.Textbox(value="*.safetensors", visible=False)
        lora_ext_name = gr.Textbox(value="LCM model types", visible=False)
        model_ext = gr.Textbox(value="*.safetensors", visible=False)
        model_ext_name = gr.Textbox(value="Model types", visible=False)

        with gr.Row():
            model_path = gr.Textbox(
                label="Stable Diffusion model to convert to LCM",
                interactive=True,
            )
            button_model_path_file = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                visible=(not headless),
            )
            button_model_path_file.click(
                get_file_path,
                inputs=[model_path, model_ext, model_ext_name],
                outputs=model_path,
                show_progress=False,
            )

            name = gr.Textbox(
                label="Name of the new LCM model",
                placeholder="Path to the LCM file to create",
                interactive=True,
            )
            button_name = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                visible=(not headless),
            )
            button_name.click(
                get_saveasfilename_path,
                inputs=[name, lora_ext, lora_ext_name],
                outputs=name,
                show_progress=False,
            )

        with gr.Row():
            lora_scale = gr.Slider(
                label="Strength of the LCM",
                minimum=0.0,
                maximum=2.0,
                step=0.1,
                value=1.0,
                interactive=True,
            )
        # with gr.Row():
            # no_half = gr.Checkbox(label="Convert the new LCM model to FP32", value=False)
            model_type = gr.Dropdown(
                label="Model type", choices=["SD15", "SDXL", "SD-1B"], value="SD15"
            )

        extract_button = gr.Button("Extract LCM")

        extract_button.click(
            convert_lcm,
            inputs=[
                name,
                model_path,
                lora_scale,
                model_type
            ],
            show_progress=False,
        )
