import gradio as gr
import subprocess
import os
import sys
from .common_gui import (
    get_file_path,
    scriptdir,
    list_files,
    create_refresh_button, setup_environment
)

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

PYTHON = sys.executable


def extract_dylora(
    model,
    save_to,
    unit,
):
    # Check for caption_text_input
    if model == "":
        log.info("Invalid DyLoRA model file")
        return

    # Check if source model exist
    if not os.path.isfile(model):
        log.info("The provided DyLoRA model is not a file")
        return

    if os.path.dirname(save_to) == "":
        # only filename given. prepend dir
        save_to = os.path.join(os.path.dirname(model), save_to)
    if os.path.isdir(save_to):
        # only dir name given. set default lora name
        save_to = os.path.join(save_to, "lora.safetensors")
    if os.path.normpath(model) == os.path.normpath(save_to):
        # same path. silently ignore but rename output
        path, ext = os.path.splitext(save_to)
        save_to = f"{path}_tmp{ext}"

    run_cmd = [
        rf"{PYTHON}",
        rf"{scriptdir}/sd-scripts/networks/extract_lora_from_dylora.py",
        "--save_to",
        rf"{save_to}",
        "--model",
        rf"{model}",
        "--unit",
        str(unit),
    ]

    env = setup_environment()

    # Reconstruct the safe command string for display
    command_to_run = " ".join(run_cmd)
    log.info(f"Executing command: {command_to_run}")

    # Run the command in the sd-scripts folder context
    subprocess.run(run_cmd, env=env, shell=False)

    log.info("Done extracting DyLoRA...")


###
# Gradio UI
###


def gradio_extract_dylora_tab(headless=False):
    current_model_dir = os.path.join(scriptdir, "outputs")
    current_save_dir = os.path.join(scriptdir, "outputs")

    with gr.Tab("Extract DyLoRA"):
        gr.Markdown("This utility can extract a DyLoRA network from a finetuned model.")
        lora_ext = gr.Textbox(value="*.safetensors *.pt", visible=False)
        lora_ext_name = gr.Textbox(value="LoRA model types", visible=False)

        def list_models(path):
            nonlocal current_model_dir
            current_model_dir = path
            return list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

        def list_save_to(path):
            nonlocal current_save_dir
            current_save_dir = path
            return list(list_files(path, exts=[".pt", ".safetensors"], all=True))

        with gr.Group(), gr.Row():
            model = gr.Dropdown(
                label="DyLoRA model (path to the DyLoRA model to extract from)",
                interactive=True,
                choices=[""] + list_models(current_model_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                model,
                lambda: None,
                lambda: {"choices": list_models(current_model_dir)},
                "open_folder_small",
            )
            button_model_file = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_model_file.click(
                get_file_path,
                inputs=[model, lora_ext, lora_ext_name],
                outputs=model,
                show_progress=False,
            )

            save_to = gr.Dropdown(
                label="Save to (path where to save the extracted LoRA model...)",
                interactive=True,
                choices=[""] + list_save_to(current_save_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                save_to,
                lambda: None,
                lambda: {"choices": list_save_to(current_save_dir)},
                "open_folder_small",
            )
            unit = gr.Slider(
                minimum=1,
                maximum=256,
                label="Network Dimension (Rank)",
                value=1,
                step=1,
                interactive=True,
            )

            model.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_models(path)),
                inputs=model,
                outputs=model,
                show_progress=False,
            )
            save_to.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_save_to(path)),
                inputs=save_to,
                outputs=save_to,
                show_progress=False,
            )

        extract_button = gr.Button("Extract LoRA model")

        extract_button.click(
            extract_dylora,
            inputs=[
                model,
                save_to,
                unit,
            ],
            show_progress=False,
        )
