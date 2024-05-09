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


def extract_lycoris_locon(
    db_model,
    base_model,
    output_name,
    device,
    is_sdxl,
    is_v2,
    mode,
    linear_dim,
    conv_dim,
    linear_threshold,
    conv_threshold,
    linear_ratio,
    conv_ratio,
    linear_quantile,
    conv_quantile,
    use_sparse_bias,
    sparsity,
    disable_cp,
):
    # Check for caption_text_input
    if db_model == "":
        log.info("Invalid finetuned model file")
        return

    if base_model == "":
        log.info("Invalid base model file")
        return

    # Check if source model exist
    if not os.path.isfile(db_model):
        log.info("The provided finetuned model is not a file")
        return

    if not os.path.isfile(base_model):
        log.info("The provided base model is not a file")
        return

    if os.path.dirname(output_name) == "":
        # only filename given. prepend dir
        output_name = os.path.join(os.path.dirname(db_model), output_name)
    if os.path.isdir(output_name):
        # only dir name given. set default lora name
        output_name = os.path.join(output_name, "lora.safetensors")
    if os.path.normpath(db_model) == os.path.normpath(output_name):
        # same path. silently ignore but rename output
        path, ext = os.path.splitext(output_name)
        output_name = f"{path}_tmp{ext}"

    run_cmd = [fr'{PYTHON}', fr'{scriptdir}/tools/lycoris_locon_extract.py']

    if is_sdxl:
        run_cmd.append("--is_sdxl")
    if is_v2:
        run_cmd.append("--is_v2")

    # Adding required parameters
    run_cmd.append("--device")
    run_cmd.append(device)
    run_cmd.append("--mode")
    run_cmd.append(mode)
    run_cmd.append("--safetensors")

    # Handling conditional parameters based on mode
    if mode == "fixed":
        run_cmd.append("--linear_dim")
        run_cmd.append(str(linear_dim))
        run_cmd.append("--conv_dim")
        run_cmd.append(str(conv_dim))
    elif mode == "threshold":
        run_cmd.append("--linear_threshold")
        run_cmd.append(str(linear_threshold))
        run_cmd.append("--conv_threshold")
        run_cmd.append(str(conv_threshold))
    elif mode == "ratio":
        run_cmd.append("--linear_ratio")
        run_cmd.append(str(linear_ratio))
        run_cmd.append("--conv_ratio")
        run_cmd.append(str(conv_ratio))
    elif mode == "quantile":
        run_cmd.append("--linear_quantile")
        run_cmd.append(str(linear_quantile))
        run_cmd.append("--conv_quantile")
        run_cmd.append(str(conv_quantile))

    if use_sparse_bias:
        run_cmd.append("--use_sparse_bias")

    # Adding additional options
    run_cmd.append("--sparsity")
    run_cmd.append(str(sparsity))

    if disable_cp:
        run_cmd.append("--disable_cp")

    # Add paths
    run_cmd.append(fr"{base_model}")
    run_cmd.append(fr"{db_model}")
    run_cmd.append(fr"{output_name}")

    env = setup_environment()

    # Reconstruct the safe command string for display
    command_to_run = " ".join(run_cmd)
    log.info(f"Executing command: {command_to_run}")
            
    # Run the command in the sd-scripts folder context
    subprocess.run(run_cmd, env=env)


    log.info("Done extracting...")


###
# Gradio UI
###
# def update_mode(mode):
#     # 'fixed', 'threshold','ratio','quantile'
#     if mode == 'fixed':
#         return gr.Row(visible=True), gr.Row(visible=False), gr.Row(visible=False), gr.Row(visible=False)
#     if mode == 'threshold':
#         return gr.Row(visible=False), gr.Row(visible=True), gr.Row(visible=False), gr.Row(visible=False)
#     if mode == 'ratio':
#         return gr.Row(visible=False), gr.Row(visible=False), gr.Row(visible=True), gr.Row(visible=False)
#     if mode == 'threshold':
#         return gr.Row(visible=False), gr.Row(visible=False), gr.Row(visible=False), gr.Row(visible=True)


def update_mode(mode):
    # Create a list of possible mode values
    modes = ["fixed", "threshold", "ratio", "quantile"]

    # Initialize an empty list to store visibility updates
    updates = []

    # Iterate through the possible modes
    for m in modes:
        # Add a visibility update for each mode, setting it to True if the input mode matches the current mode in the loop
        updates.append(gr.Row(visible=(mode == m)))

    # Return the visibility updates as a tuple
    return tuple(updates)


def gradio_extract_lycoris_locon_tab(headless=False):

    current_model_dir = os.path.join(scriptdir, "outputs")
    current_base_model_dir = os.path.join(scriptdir, "outputs")
    current_save_dir = os.path.join(scriptdir, "outputs")

    def list_models(path):
        nonlocal current_model_dir
        current_model_dir = path
        return list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

    def list_base_models(path):
        nonlocal current_base_model_dir
        current_base_model_dir = path
        return list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

    def list_save_to(path):
        nonlocal current_save_dir
        current_save_dir = path
        return list(list_files(path, exts=[".safetensors"], all=True))

    with gr.Tab("Extract LyCORIS LoCon"):
        gr.Markdown(
            "This utility can extract a LyCORIS LoCon network from a finetuned model."
        )
        lora_ext = gr.Textbox(
            value="*.safetensors", visible=False
        )  # lora_ext = gr.Textbox(value='*.safetensors *.pt', visible=False)
        lora_ext_name = gr.Textbox(value="LoRA model types", visible=False)
        model_ext = gr.Textbox(value="*.safetensors *.ckpt", visible=False)
        model_ext_name = gr.Textbox(value="Model types", visible=False)

        with gr.Group(), gr.Row():
            db_model = gr.Dropdown(
                label="Finetuned model (path to the finetuned model to extract)",
                interactive=True,
                choices=[""] + list_models(current_model_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                db_model,
                lambda: None,
                lambda: {"choices": list_models(current_model_dir)},
                "open_folder_small",
            )
            button_db_model_file = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_db_model_file.click(
                get_file_path,
                inputs=[db_model, model_ext, model_ext_name],
                outputs=db_model,
                show_progress=False,
            )

            base_model = gr.Dropdown(
                label="Stable Diffusion base model (original model: ckpt or safetensors file)",
                choices=[""] + list_base_models(current_base_model_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                base_model,
                lambda: None,
                lambda: {"choices": list_base_models(current_base_model_dir)},
                "open_folder_small",
            )
            button_base_model_file = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_base_model_file.click(
                get_file_path,
                inputs=[base_model, model_ext, model_ext_name],
                outputs=base_model,
                show_progress=False,
            )
        with gr.Group(), gr.Row():
            output_name = gr.Dropdown(
                label="Save to (path where to save the extracted LoRA model...)",
                interactive=True,
                choices=[""] + list_save_to(current_save_dir),
                value="",
                allow_custom_value=True,
                scale=2,
            )
            create_refresh_button(
                output_name,
                lambda: None,
                lambda: {"choices": list_save_to(current_save_dir)},
                "open_folder_small",
            )
            button_output_name = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_output_name.click(
                get_saveasfilename_path,
                inputs=[output_name, lora_ext, lora_ext_name],
                outputs=output_name,
                show_progress=False,
            )
            device = gr.Radio(
                label="Device",
                choices=[
                    "cpu",
                    "cuda",
                ],
                value="cuda",
                interactive=True,
                scale=2,
            )

            db_model.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_models(path)),
                inputs=db_model,
                outputs=db_model,
                show_progress=False,
            )
            base_model.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_base_models(path)),
                inputs=base_model,
                outputs=base_model,
                show_progress=False,
            )
            output_name.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_save_to(path)),
                inputs=output_name,
                outputs=output_name,
                show_progress=False,
            )

            is_sdxl = gr.Checkbox(
                label="is SDXL", value=False, interactive=True, scale=1
            )

            is_v2 = gr.Checkbox(label="is v2", value=False, interactive=True, scale=1)
        with gr.Row():
            mode = gr.Radio(
                label="Mode",
                choices=["fixed", "full", "quantile", "ratio", "threshold"],
                value="fixed",
                interactive=True,
            )
        with gr.Row(visible=True) as fixed:
            linear_dim = gr.Slider(
                minimum=1,
                maximum=1024,
                label="Network Dimension",
                value=1,
                step=1,
                interactive=True,
                info="network dim for linear layer in fixed mode",
            )
            conv_dim = gr.Slider(
                minimum=1,
                maximum=1024,
                label="Conv Dimension",
                value=1,
                step=1,
                interactive=True,
                info="network dim for conv layer in fixed mode",
            )
        with gr.Row(visible=False) as threshold:
            linear_threshold = gr.Slider(
                minimum=0,
                maximum=1,
                label="Linear threshold",
                value=0.65,
                step=0.01,
                interactive=True,
                info="The higher the value, the smaller the file. Recommended starting value: 0.65",
            )
            conv_threshold = gr.Slider(
                minimum=0,
                maximum=1,
                label="Conv threshold",
                value=0.65,
                step=0.01,
                interactive=True,
                info="The higher the value, the smaller the file. Recommended starting value: 0.65",
            )
        with gr.Row(visible=False) as ratio:
            linear_ratio = gr.Slider(
                minimum=0,
                maximum=1,
                label="Linear ratio",
                value=0.75,
                step=0.01,
                interactive=True,
                info="The higher the value, the smaller the file. Recommended starting value: 0.75",
            )
            conv_ratio = gr.Slider(
                minimum=0,
                maximum=1,
                label="Conv ratio",
                value=0.75,
                step=0.01,
                interactive=True,
                info="The higher the value, the smaller the file. Recommended starting value: 0.75",
            )
        with gr.Row(visible=False) as quantile:
            linear_quantile = gr.Slider(
                minimum=0,
                maximum=1,
                label="Linear quantile",
                value=0.75,
                step=0.01,
                interactive=True,
                info="The higher the value, the larger the file. Recommended starting value: 0.75",
            )
            conv_quantile = gr.Slider(
                minimum=0,
                maximum=1,
                label="Conv quantile",
                value=0.75,
                step=0.01,
                interactive=True,
                info="The higher the value, the larger the file. Recommended starting value: 0.75",
            )
        with gr.Row():
            use_sparse_bias = gr.Checkbox(
                label="Use sparse biais", value=False, interactive=True
            )
            sparsity = gr.Slider(
                minimum=0,
                maximum=1,
                label="Sparsity",
                info="Sparsity for sparse bias",
                value=0.98,
                step=0.01,
                interactive=True,
            )
            disable_cp = gr.Checkbox(
                label="Disable CP decomposition", value=False, interactive=True
            )
        mode.change(
            update_mode,
            inputs=[mode],
            outputs=[
                fixed,
                threshold,
                ratio,
                quantile,
            ],
        )

        extract_button = gr.Button("Extract LyCORIS LoCon")

        extract_button.click(
            extract_lycoris_locon,
            inputs=[
                db_model,
                base_model,
                output_name,
                device,
                is_sdxl,
                is_v2,
                mode,
                linear_dim,
                conv_dim,
                linear_threshold,
                conv_threshold,
                linear_ratio,
                conv_ratio,
                linear_quantile,
                conv_quantile,
                use_sparse_bias,
                sparsity,
                disable_cp,
            ],
            show_progress=False,
        )
