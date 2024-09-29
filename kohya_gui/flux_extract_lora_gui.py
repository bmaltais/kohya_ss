import gradio as gr
import subprocess
import os
import sys
from .common_gui import (
    get_saveasfilename_path,
    get_file_path,
    scriptdir,
    list_files,
    create_refresh_button,
    setup_environment,
)
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

PYTHON = sys.executable


def extract_flux_lora(
    model_org,
    model_tuned,
    save_to,
    save_precision,
    dim,
    device,
    clamp_quantile,
    no_metadata,
    mem_eff_safe_open,
):
    # Check for required inputs
    if model_org == "" or model_tuned == "" or save_to == "":
        log.info(
            "Please provide all required inputs: original model, tuned model, and save path."
        )
        return

    # Check if source models exist
    if not os.path.isfile(model_org):
        log.info("The provided original model is not a file")
        return

    if not os.path.isfile(model_tuned):
        log.info("The provided tuned model is not a file")
        return

    # Prepare save path
    if os.path.dirname(save_to) == "":
        save_to = os.path.join(os.path.dirname(model_tuned), save_to)
    if os.path.isdir(save_to):
        save_to = os.path.join(save_to, "flux_lora.safetensors")
    if os.path.normpath(model_tuned) == os.path.normpath(save_to):
        path, ext = os.path.splitext(save_to)
        save_to = f"{path}_lora{ext}"

    run_cmd = [
        rf"{PYTHON}",
        rf"{scriptdir}/sd-scripts/networks/flux_extract_lora.py",
        "--model_org",
        rf"{model_org}",
        "--model_tuned",
        rf"{model_tuned}",
        "--save_to",
        rf"{save_to}",
        "--dim",
        str(dim),
        "--device",
        device,
        "--clamp_quantile",
        str(clamp_quantile),
    ]

    if save_precision:
        run_cmd.extend(["--save_precision", save_precision])

    if no_metadata:
        run_cmd.append("--no_metadata")

    if mem_eff_safe_open:
        run_cmd.append("--mem_eff_safe_open")

    env = setup_environment()

    # Reconstruct the safe command string for display
    command_to_run = " ".join(run_cmd)
    log.info(f"Executing command: {command_to_run}")

    # Run the command
    subprocess.run(run_cmd, env=env)


def gradio_flux_extract_lora_tab(headless=False):
    current_model_dir = os.path.join(scriptdir, "outputs")
    current_save_dir = os.path.join(scriptdir, "outputs")

    def list_models(path):
        return list(list_files(path, exts=[".safetensors"], all=True))

    with gr.Tab("Extract Flux LoRA"):
        gr.Markdown(
            "This utility can extract a LoRA network from a finetuned Flux model."
        )

        lora_ext = gr.Textbox(value="*.safetensors", visible=False)
        lora_ext_name = gr.Textbox(value="LoRA model types", visible=False)
        model_ext = gr.Textbox(value="*.safetensors", visible=False)
        model_ext_name = gr.Textbox(value="Model types", visible=False)

        with gr.Group(), gr.Row():
            model_org = gr.Dropdown(
                label="Original Flux model (path to the original model)",
                interactive=True,
                choices=[""] + list_models(current_model_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                model_org,
                lambda: None,
                lambda: {"choices": list_models(current_model_dir)},
                "open_folder_small",
            )
            button_model_org_file = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_model_org_file.click(
                get_file_path,
                inputs=[model_org, model_ext, model_ext_name],
                outputs=model_org,
                show_progress=False,
            )

            model_tuned = gr.Dropdown(
                label="Finetuned Flux model (path to the finetuned model to extract)",
                interactive=True,
                choices=[""] + list_models(current_model_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                model_tuned,
                lambda: None,
                lambda: {"choices": list_models(current_model_dir)},
                "open_folder_small",
            )
            button_model_tuned_file = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_model_tuned_file.click(
                get_file_path,
                inputs=[model_tuned, model_ext, model_ext_name],
                outputs=model_tuned,
                show_progress=False,
            )

        with gr.Group(), gr.Row():
            save_to = gr.Dropdown(
                label="Save to (path where to save the extracted LoRA model...)",
                interactive=True,
                choices=[""] + list_models(current_save_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                save_to,
                lambda: None,
                lambda: {"choices": list_models(current_save_dir)},
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

            save_precision = gr.Dropdown(
                label="Save precision",
                choices=["None", "float", "fp16", "bf16"],
                value="None",
                interactive=True,
            )

        with gr.Row():
            dim = gr.Slider(
                minimum=1,
                maximum=1024,
                label="Network Dimension (Rank)",
                value=4,
                step=1,
                interactive=True,
            )
            device = gr.Dropdown(
                label="Device",
                choices=["cpu", "cuda"],
                value="cuda",
                interactive=True,
            )
            clamp_quantile = gr.Slider(
                minimum=0,
                maximum=1,
                label="Clamp Quantile",
                value=0.99,
                step=0.01,
                interactive=True,
            )

        with gr.Row():
            no_metadata = gr.Checkbox(
                label="No metadata (do not save sai modelspec metadata)",
                value=False,
                interactive=True,
            )
            mem_eff_safe_open = gr.Checkbox(
                label="Memory efficient safe open (experimental feature)",
                value=False,
                interactive=True,
            )

        extract_button = gr.Button("Extract Flux LoRA model")

        extract_button.click(
            extract_flux_lora,
            inputs=[
                model_org,
                model_tuned,
                save_to,
                save_precision,
                dim,
                device,
                clamp_quantile,
                no_metadata,
                mem_eff_safe_open,
            ],
            show_progress=False,
        )

        model_org.change(
            fn=lambda path: gr.Dropdown(choices=[""] + list_models(path)),
            inputs=model_org,
            outputs=model_org,
            show_progress=False,
        )
        model_tuned.change(
            fn=lambda path: gr.Dropdown(choices=[""] + list_models(path)),
            inputs=model_tuned,
            outputs=model_tuned,
            show_progress=False,
        )
        save_to.change(
            fn=lambda path: gr.Dropdown(choices=[""] + list_models(path)),
            inputs=save_to,
            outputs=save_to,
            show_progress=False,
        )
