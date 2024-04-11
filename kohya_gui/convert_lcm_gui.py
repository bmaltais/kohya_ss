import gradio as gr
import os
import subprocess
import sys
from .common_gui import (
    get_saveasfilename_path,
    get_file_path,
    scriptdir,
    list_files,
    create_refresh_button,
)
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

PYTHON = sys.executable


def convert_lcm(name, model_path, lora_scale, model_type):
    run_cmd = rf'"{PYTHON}" "{scriptdir}/tools/lcm_convert.py"'

    # Check if source model exist
    if not os.path.isfile(model_path):
        log.error("The provided DyLoRA model is not a file")
        return

    if os.path.dirname(name) == "":
        # only filename given. prepend dir
        name = os.path.join(os.path.dirname(model_path), name)
    if os.path.isdir(name):
        # only dir name given. set default lcm name
        name = os.path.join(name, "lcm.safetensors")
    if os.path.normpath(model_path) == os.path.normpath(name):
        # same path. silently ignore but rename output
        path, ext = os.path.splitext(save_to)
        save_to = f"{path}_lcm{ext}"

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
    env["PYTHONPATH"] = (
        rf"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"
    )

    # Run the command
    subprocess.run(run_cmd, env=env)

    # Return a success message
    log.info("Done extracting...")


def gradio_convert_lcm_tab(headless=False):
    current_model_dir = os.path.join(scriptdir, "outputs")
    current_save_dir = os.path.join(scriptdir, "outputs")

    def list_models(path):
        nonlocal current_model_dir
        current_model_dir = path
        return list(list_files(path, exts=[".safetensors"], all=True))

    def list_save_to(path):
        nonlocal current_save_dir
        current_save_dir = path
        return list(list_files(path, exts=[".safetensors"], all=True))

    with gr.Tab("Convert to LCM"):
        gr.Markdown("This utility convert a model to an LCM model.")
        lora_ext = gr.Textbox(value="*.safetensors", visible=False)
        lora_ext_name = gr.Textbox(value="LCM model types", visible=False)
        model_ext = gr.Textbox(value="*.safetensors", visible=False)
        model_ext_name = gr.Textbox(value="Model types", visible=False)

        with gr.Group(), gr.Row():
            model_path = gr.Dropdown(
                label="Stable Diffusion model to convert to LCM",
                interactive=True,
                choices=[""] + list_models(current_model_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                model_path,
                lambda: None,
                lambda: {"choices": list_models(current_model_dir)},
                "open_folder_small",
            )
            button_model_path_file = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_model_path_file.click(
                get_file_path,
                inputs=[model_path, model_ext, model_ext_name],
                outputs=model_path,
                show_progress=False,
            )

            name = gr.Dropdown(
                label="Name of the new LCM model",
                interactive=True,
                choices=[""] + list_save_to(current_save_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                name,
                lambda: None,
                lambda: {"choices": list_save_to(current_save_dir)},
                "open_folder_small",
            )
            button_name = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_name.click(
                get_saveasfilename_path,
                inputs=[name, lora_ext, lora_ext_name],
                outputs=name,
                show_progress=False,
            )
            model_path.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_models(path)),
                inputs=model_path,
                outputs=model_path,
                show_progress=False,
            )
            name.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_save_to(path)),
                inputs=name,
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
            model_type = gr.Radio(
                label="Model type", choices=["SD15", "SDXL", "SD-1B"], value="SD15"
            )

        extract_button = gr.Button("Extract LCM")

        extract_button.click(
            convert_lcm,
            inputs=[name, model_path, lora_scale, model_type],
            show_progress=False,
        )
