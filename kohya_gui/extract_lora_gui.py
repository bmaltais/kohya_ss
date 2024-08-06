import gradio as gr
import subprocess
import os
import sys
from .common_gui import (
    get_saveasfilename_path,
    get_file_path,
    is_file_writable,
    scriptdir,
    list_files,
    create_refresh_button, setup_environment
)

from .custom_logging import setup_logging
from .sd_modeltype import SDModelType

# Set up logging
log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

PYTHON = sys.executable


def extract_lora(
    model_tuned,
    model_org,
    save_to,
    save_precision,
    dim,
    v2,
    sdxl,
    conv_dim,
    clamp_quantile,
    min_diff,
    device,
    load_original_model_to,
    load_tuned_model_to,
    load_precision,
):
    # Check for caption_text_input
    if model_tuned == "":
        log.info("Invalid finetuned model file")
        return

    if model_org == "":
        log.info("Invalid base model file")
        return

    # Check if source model exist
    if not os.path.isfile(model_tuned):
        log.info("The provided finetuned model is not a file")
        return

    if not os.path.isfile(model_org):
        log.info("The provided base model is not a file")
        return

    if os.path.dirname(save_to) == "":
        # only filename given. prepend dir
        save_to = os.path.join(os.path.dirname(model_tuned), save_to)
    if os.path.isdir(save_to):
        # only dir name given. set default lora name
        save_to = os.path.join(save_to, "lora.safetensors")
    if os.path.normpath(model_tuned) == os.path.normpath(save_to):
        # same path. silently ignore but rename output
        path, ext = os.path.splitext(save_to)
        save_to = f"{path}_tmp{ext}"

    if not is_file_writable(save_to):
        return

    run_cmd = [
        rf"{PYTHON}",
        rf"{scriptdir}/sd-scripts/networks/extract_lora_from_models.py",
        "--load_precision",
        load_precision,
        "--save_precision",
        save_precision,
        "--save_to",
        rf"{save_to}",
        "--model_org",
        rf"{model_org}",
        "--model_tuned",
        rf"{model_tuned}",
        "--dim",
        str(dim),
        "--device",
        device,
        "--clamp_quantile",
        str(clamp_quantile),
        "--min_diff",
        str(min_diff),
    ]

    if conv_dim > 0:
        run_cmd.append("--conv_dim")
        run_cmd.append(str(conv_dim))

    if v2:
        run_cmd.append("--v2")

    if sdxl:
        run_cmd.append("--sdxl")
        run_cmd.append("--load_original_model_to")
        run_cmd.append(load_original_model_to)
        run_cmd.append("--load_tuned_model_to")
        run_cmd.append(load_tuned_model_to)

    env = setup_environment()

    # Reconstruct the safe command string for display
    command_to_run = " ".join(run_cmd)
    log.info(f"Executing command: {command_to_run}")

    # Run the command in the sd-scripts folder context
    subprocess.run(run_cmd, env=env)


###
# Gradio UI
###


def gradio_extract_lora_tab(
    headless=False,
):
    current_model_dir = os.path.join(scriptdir, "outputs")
    current_model_org_dir = os.path.join(scriptdir, "outputs")
    current_save_dir = os.path.join(scriptdir, "outputs")

    def list_models(path):
        nonlocal current_model_dir
        current_model_dir = path
        return list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

    def list_org_models(path):
        nonlocal current_model_org_dir
        current_model_org_dir = path
        return list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

    def list_save_to(path):
        nonlocal current_save_dir
        current_save_dir = path
        return list(list_files(path, exts=[".pt", ".safetensors"], all=True))

    def change_sdxl(sdxl):
        return gr.Dropdown(visible=sdxl), gr.Dropdown(visible=sdxl)

    with gr.Tab("Extract LoRA"):
        gr.Markdown("This utility can extract a LoRA network from a finetuned model.")
        lora_ext = gr.Textbox(value="*.safetensors *.pt", visible=False)
        lora_ext_name = gr.Textbox(value="LoRA model types", visible=False)
        model_ext = gr.Textbox(value="*.ckpt *.safetensors", visible=False)
        model_ext_name = gr.Textbox(value="Model types", visible=False)

        with gr.Group(), gr.Row():
            model_tuned = gr.Dropdown(
                label="Finetuned model (path to the finetuned model to extract)",
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
            load_tuned_model_to = gr.Radio(
                label="Load finetuned model to",
                choices=["cpu", "cuda", "cuda:0"],
                value="cpu",
                interactive=True,
                scale=1,
                info="only for SDXL",
                visible=False,
            )
            model_org = gr.Dropdown(
                label="Stable Diffusion base model (original model: ckpt or safetensors file)",
                interactive=True,
                choices=[""] + list_org_models(current_model_org_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                model_org,
                lambda: None,
                lambda: {"choices": list_org_models(current_model_org_dir)},
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
            load_original_model_to = gr.Dropdown(
                label="Load Stable Diffusion base model to",
                choices=["cpu", "cuda", "cuda:0"],
                value="cpu",
                interactive=True,
                scale=1,
                info="only for SDXL",
                visible=False,
            )
        with gr.Group(), gr.Row():
            save_to = gr.Dropdown(
                label="Save to (path where to save the extracted LoRA model...)",
                interactive=True,
                choices=[""] + list_save_to(current_save_dir),
                value="",
                allow_custom_value=True,
                scale=2,
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
            save_precision = gr.Radio(
                label="Save precision",
                choices=["fp16", "bf16", "float"],
                value="fp16",
                interactive=True,
                scale=1,
            )
            load_precision = gr.Radio(
                label="Load precision",
                choices=["fp16", "bf16", "float"],
                value="fp16",
                interactive=True,
                scale=1,
            )

            model_tuned.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_models(path)),
                inputs=model_tuned,
                outputs=model_tuned,
                show_progress=False,
            )
            model_org.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_org_models(path)),
                inputs=model_org,
                outputs=model_org,
                show_progress=False,
            )
            save_to.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_save_to(path)),
                inputs=save_to,
                outputs=save_to,
                show_progress=False,
            )
        with gr.Row():
            dim = gr.Slider(
                minimum=4,
                maximum=1024,
                label="Network Dimension (Rank)",
                value=128,
                step=1,
                interactive=True,
            )
            conv_dim = gr.Slider(
                minimum=0,
                maximum=1024,
                label="Conv Dimension (Rank)",
                value=128,
                step=1,
                interactive=True,
            )
            clamp_quantile = gr.Number(
                label="Clamp Quantile",
                value=0.99,
                minimum=0,
                maximum=1,
                step=0.001,
                interactive=True,
            )
            min_diff = gr.Number(
                label="Minimum difference",
                value=0.01,
                minimum=0,
                maximum=1,
                step=0.001,
                interactive=True,
            )
        with gr.Row():
            v2 = gr.Checkbox(label="v2", value=False, interactive=True)
            sdxl = gr.Checkbox(label="SDXL", value=False, interactive=True)
            device = gr.Radio(
                label="Device",
                choices=[
                    "cpu",
                    "cuda",
                ],
                value="cuda",
                interactive=True,
            )

            sdxl.change(
                change_sdxl,
                inputs=sdxl,
                outputs=[load_tuned_model_to, load_original_model_to],
            )

            #secondary event on model_tuned for auto-detection of v2/SDXL
            def change_modeltype_model_tuned(path):
                detect = SDModelType(path)
                v2 = gr.Checkbox(value=detect.Is_SD2())
                sdxl = gr.Checkbox(value=detect.Is_SDXL())
                return v2, sdxl

            model_tuned.change(
                change_modeltype_model_tuned,
                inputs=model_tuned,
                outputs=[v2, sdxl]
            )

        extract_button = gr.Button("Extract LoRA model")

        extract_button.click(
            extract_lora,
            inputs=[
                model_tuned,
                model_org,
                save_to,
                save_precision,
                dim,
                v2,
                sdxl,
                conv_dim,
                clamp_quantile,
                min_diff,
                device,
                load_original_model_to,
                load_tuned_model_to,
                load_precision,
            ],
            show_progress=False,
        )
