import gradio as gr
import subprocess
import os
import sys
from .common_gui import (
    get_saveasfilename_path,
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


def resize_lora(
    model,
    new_rank,
    save_to,
    save_precision,
    device,
    dynamic_method,
    dynamic_param,
    verbose,
):
    # Check for caption_text_input
    if model == "":
        log.info("Invalid model file")
        return

    # Check if source model exist
    if not os.path.isfile(model):
        log.info("The provided model is not a file")
        return

    if dynamic_method == "sv_ratio":
        if float(dynamic_param) < 2:
            log.info(
                f"Dynamic parameter for {dynamic_method} need to be 2 or greater..."
            )
            return

    if dynamic_method == "sv_fro" or dynamic_method == "sv_cumulative":
        if float(dynamic_param) < 0 or float(dynamic_param) > 1:
            log.info(
                f"Dynamic parameter for {dynamic_method} need to be between 0 and 1..."
            )
            return

    # Check if save_to end with one of the defines extension. If not add .safetensors.
    if not save_to.endswith((".pt", ".safetensors")):
        save_to += ".safetensors"

    if device == "":
        device = "cuda"

    run_cmd = [
        rf"{PYTHON}",
        rf"{scriptdir}/sd-scripts/networks/resize_lora.py",
        "--save_precision",
        save_precision,
        "--save_to",
        rf"{save_to}",
        "--model",
        rf"{model}",
        "--new_rank",
        str(new_rank),
        "--device",
        device,
    ]

    # Conditional checks for dynamic parameters
    if dynamic_method != "None":
        run_cmd.append("--dynamic_method")
        run_cmd.append(dynamic_method)
        run_cmd.append("--dynamic_param")
        run_cmd.append(str(dynamic_param))

    # Check for verbosity
    if verbose:
        run_cmd.append("--verbose")

    env = setup_environment()

    # Reconstruct the safe command string for display
    command_to_run = " ".join(run_cmd)
    log.info(f"Executing command: {command_to_run}")

    # Run the command in the sd-scripts folder context
    subprocess.run(run_cmd, env=env)

    log.info("Done resizing...")


###
# Gradio UI
###


def gradio_resize_lora_tab(
    headless=False,
):
    current_model_dir = os.path.join(scriptdir, "outputs")
    current_save_dir = os.path.join(scriptdir, "outputs")

    def list_models(path):
        nonlocal current_model_dir
        current_model_dir = path
        return list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

    def list_save_to(path):
        nonlocal current_save_dir
        current_save_dir = path
        return list(list_files(path, exts=[".pt", ".safetensors"], all=True))

    with gr.Tab("Resize LoRA"):
        gr.Markdown("This utility can resize a LoRA.")

        lora_ext = gr.Textbox(value="*.safetensors *.pt", visible=False)
        lora_ext_name = gr.Textbox(value="LoRA model types", visible=False)

        with gr.Group(), gr.Row():
            model = gr.Dropdown(
                label="Source LoRA (path to the LoRA to resize)",
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
            button_lora_a_model_file = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_lora_a_model_file.click(
                get_file_path,
                inputs=[model, lora_ext, lora_ext_name],
                outputs=model,
                show_progress=False,
            )
            save_to = gr.Dropdown(
                label="Save to (path for the LoRA file to save...)",
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
            button_save_to = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_save_to.click(
                get_saveasfilename_path,
                inputs=[save_to, lora_ext, lora_ext_name],
                outputs=save_to,
                show_progress=False,
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
        with gr.Row():
            new_rank = gr.Slider(
                label="Desired LoRA rank",
                minimum=1,
                maximum=1024,
                step=1,
                value=4,
                interactive=True,
            )
            dynamic_method = gr.Radio(
                choices=["None", "sv_ratio", "sv_fro", "sv_cumulative"],
                value="sv_fro",
                label="Dynamic method",
                interactive=True,
            )
            dynamic_param = gr.Textbox(
                label="Dynamic parameter",
                value="0.9",
                interactive=True,
                placeholder="Value for the dynamic method selected.",
            )
        with gr.Row():

            verbose = gr.Checkbox(label="Verbose logging", value=True)
            save_precision = gr.Radio(
                label="Save precision",
                choices=["fp16", "bf16", "float"],
                value="fp16",
                interactive=True,
            )
            device = gr.Radio(
                label="Device",
                choices=[
                    "cpu",
                    "cuda",
                ],
                value="cuda",
                interactive=True,
            )

        convert_button = gr.Button("Resize model")

        convert_button.click(
            resize_lora,
            inputs=[
                model,
                new_rank,
                save_to,
                save_precision,
                device,
                dynamic_method,
                dynamic_param,
                verbose,
            ],
            show_progress=False,
        )
