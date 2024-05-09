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


def verify_lora(
    lora_model,
):
    # verify for caption_text_input
    if lora_model == "":
        log.info("Invalid model A file")
        return

    # verify if source model exist
    if not os.path.isfile(lora_model):
        log.info("The provided model A is not a file")
        return

    run_cmd = [
        rf"{PYTHON}",
        rf"{scriptdir}/sd-scripts/networks/check_lora_weights.py",
        rf"{lora_model}",
    ]
    # run_cmd = rf'"{PYTHON}" "{scriptdir}/sd-scripts/networks/check_lora_weights.py" "{lora_model}"'

    # Reconstruct the safe command string for display
    command_to_run = " ".join(run_cmd)
    log.info(f"Executing command: {command_to_run}")

    # Set the environment variable for the Python path
    env = setup_environment()

    # Run the command using subprocess.Popen for asynchronous handling
    process = subprocess.Popen(
        run_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    output, error = process.communicate()

    return (output.decode(), error.decode())


###
# Gradio UI
###


def gradio_verify_lora_tab(headless=False):
    current_model_dir = os.path.join(scriptdir, "outputs")

    def list_models(path):
        nonlocal current_model_dir
        current_model_dir = path
        return list(list_files(path, exts=[".pt", ".safetensors"], all=True))

    with gr.Tab("Verify LoRA"):
        gr.Markdown(
            "This utility can verify a LoRA network to make sure it is properly trained."
        )

        lora_ext = gr.Textbox(value="*.pt *.safetensors", visible=False)
        lora_ext_name = gr.Textbox(value="LoRA model types", visible=False)

        with gr.Group(), gr.Row():
            lora_model = gr.Dropdown(
                label="LoRA model (path to the LoRA model to verify)",
                interactive=True,
                choices=[""] + list_models(current_model_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                lora_model,
                lambda: None,
                lambda: {"choices": list_models(current_model_dir)},
                "open_folder_small",
            )
            button_lora_model_file = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_lora_model_file.click(
                get_file_path,
                inputs=[lora_model, lora_ext, lora_ext_name],
                outputs=lora_model,
                show_progress=False,
            )
            verify_button = gr.Button("Verify", variant="primary")

            lora_model.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_models(path)),
                inputs=lora_model,
                outputs=lora_model,
                show_progress=False,
            )

        lora_model_verif_output = gr.Textbox(
            label="Output",
            placeholder="Verification output",
            interactive=False,
            lines=1,
            max_lines=10,
        )

        lora_model_verif_error = gr.Textbox(
            label="Error",
            placeholder="Verification error",
            interactive=False,
            lines=1,
            max_lines=10,
        )

        verify_button.click(
            verify_lora,
            inputs=[
                lora_model,
            ],
            outputs=[lora_model_verif_output, lora_model_verif_error],
            show_progress=False,
        )
