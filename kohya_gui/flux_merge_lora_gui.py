# Standard library imports
import os
import subprocess
import sys
import json

# Third-party imports
import gradio as gr

# Local module imports
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


def check_model(model):
    if not model:
        return True
    if not os.path.isfile(model):
        log.info(f"The provided {model} is not a file")
        return False
    return True


def verify_conditions(flux_model, lora_models):
    lora_models_count = sum(1 for model in lora_models if model)
    if flux_model and lora_models_count >= 1:
        return True
    elif not flux_model and lora_models_count >= 2:
        return True
    return False


class GradioFluxMergeLoRaTab:
    def __init__(self, headless=False):
        self.headless = headless
        self.build_tab()

    def save_inputs_to_json(self, file_path, inputs):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(inputs, file)
        log.info(f"Saved inputs to {file_path}")

    def load_inputs_from_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            inputs = json.load(file)
        log.info(f"Loaded inputs from {file_path}")
        return inputs

    def build_tab(self):
        current_flux_model_dir = os.path.join(scriptdir, "outputs")
        current_save_dir = os.path.join(scriptdir, "outputs")
        current_lora_model_dir = current_flux_model_dir

        def list_flux_models(path):
            nonlocal current_flux_model_dir
            current_flux_model_dir = path
            return list(list_files(path, exts=[".safetensors"], all=True))

        def list_lora_models(path):
            nonlocal current_lora_model_dir
            current_lora_model_dir = path
            return list(list_files(path, exts=[".safetensors"], all=True))

        def list_save_to(path):
            nonlocal current_save_dir
            current_save_dir = path
            return list(list_files(path, exts=[".safetensors"], all=True))

        with gr.Tab("Merge FLUX LoRA"):
            gr.Markdown(
                "This utility can merge up to 4 LoRA into a FLUX model or alternatively merge up to 4 LoRA together."
            )

            lora_ext = gr.Textbox(value="*.safetensors", visible=False)
            lora_ext_name = gr.Textbox(value="LoRA model types", visible=False)
            flux_ext = gr.Textbox(value="*.safetensors", visible=False)
            flux_ext_name = gr.Textbox(value="FLUX model types", visible=False)

            with gr.Group(), gr.Row():
                flux_model = gr.Dropdown(
                    label="FLUX Model (Optional. FLUX model path, if you want to merge it with LoRA files via the 'concat' method)",
                    interactive=True,
                    choices=[""] + list_flux_models(current_flux_model_dir),
                    value="",
                    allow_custom_value=True,
                )
                create_refresh_button(
                    flux_model,
                    lambda: None,
                    lambda: {"choices": list_flux_models(current_flux_model_dir)},
                    "open_folder_small",
                )
                flux_model_file = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not self.headless),
                )
                flux_model_file.click(
                    get_file_path,
                    inputs=[flux_model, flux_ext, flux_ext_name],
                    outputs=flux_model,
                    show_progress=False,
                )

                flux_model.change(
                    fn=lambda path: gr.Dropdown(choices=[""] + list_flux_models(path)),
                    inputs=flux_model,
                    outputs=flux_model,
                    show_progress=False,
                )

            with gr.Group(), gr.Row():
                lora_a_model = gr.Dropdown(
                    label='LoRA model "A" (path to the LoRA A model)',
                    interactive=True,
                    choices=[""] + list_lora_models(current_lora_model_dir),
                    value="",
                    allow_custom_value=True,
                )
                create_refresh_button(
                    lora_a_model,
                    lambda: None,
                    lambda: {"choices": list_lora_models(current_lora_model_dir)},
                    "open_folder_small",
                )
                button_lora_a_model_file = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not self.headless),
                )
                button_lora_a_model_file.click(
                    get_file_path,
                    inputs=[lora_a_model, lora_ext, lora_ext_name],
                    outputs=lora_a_model,
                    show_progress=False,
                )

                lora_b_model = gr.Dropdown(
                    label='LoRA model "B" (path to the LoRA B model)',
                    interactive=True,
                    choices=[""] + list_lora_models(current_lora_model_dir),
                    value="",
                    allow_custom_value=True,
                )
                create_refresh_button(
                    lora_b_model,
                    lambda: None,
                    lambda: {"choices": list_lora_models(current_lora_model_dir)},
                    "open_folder_small",
                )
                button_lora_b_model_file = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not self.headless),
                )
                button_lora_b_model_file.click(
                    get_file_path,
                    inputs=[lora_b_model, lora_ext, lora_ext_name],
                    outputs=lora_b_model,
                    show_progress=False,
                )

                lora_a_model.change(
                    fn=lambda path: gr.Dropdown(choices=[""] + list_lora_models(path)),
                    inputs=lora_a_model,
                    outputs=lora_a_model,
                    show_progress=False,
                )
                lora_b_model.change(
                    fn=lambda path: gr.Dropdown(choices=[""] + list_lora_models(path)),
                    inputs=lora_b_model,
                    outputs=lora_b_model,
                    show_progress=False,
                )

            with gr.Row():
                ratio_a = gr.Slider(
                    label="Model A merge ratio (eg: 0.5 mean 50%)",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.0,
                    interactive=True,
                )

                ratio_b = gr.Slider(
                    label="Model B merge ratio (eg: 0.5 mean 50%)",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.0,
                    interactive=True,
                )

            with gr.Group(), gr.Row():
                lora_c_model = gr.Dropdown(
                    label='LoRA model "C" (path to the LoRA C model)',
                    interactive=True,
                    choices=[""] + list_lora_models(current_lora_model_dir),
                    value="",
                    allow_custom_value=True,
                )
                create_refresh_button(
                    lora_c_model,
                    lambda: None,
                    lambda: {"choices": list_lora_models(current_lora_model_dir)},
                    "open_folder_small",
                )
                button_lora_c_model_file = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not self.headless),
                )
                button_lora_c_model_file.click(
                    get_file_path,
                    inputs=[lora_c_model, lora_ext, lora_ext_name],
                    outputs=lora_c_model,
                    show_progress=False,
                )

                lora_d_model = gr.Dropdown(
                    label='LoRA model "D" (path to the LoRA D model)',
                    interactive=True,
                    choices=[""] + list_lora_models(current_lora_model_dir),
                    value="",
                    allow_custom_value=True,
                )
                create_refresh_button(
                    lora_d_model,
                    lambda: None,
                    lambda: {"choices": list_lora_models(current_lora_model_dir)},
                    "open_folder_small",
                )
                button_lora_d_model_file = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not self.headless),
                )
                button_lora_d_model_file.click(
                    get_file_path,
                    inputs=[lora_d_model, lora_ext, lora_ext_name],
                    outputs=lora_d_model,
                    show_progress=False,
                )
                lora_c_model.change(
                    fn=lambda path: gr.Dropdown(choices=[""] + list_lora_models(path)),
                    inputs=lora_c_model,
                    outputs=lora_c_model,
                    show_progress=False,
                )
                lora_d_model.change(
                    fn=lambda path: gr.Dropdown(choices=[""] + list_lora_models(path)),
                    inputs=lora_d_model,
                    outputs=lora_d_model,
                    show_progress=False,
                )

            with gr.Row():
                ratio_c = gr.Slider(
                    label="Model C merge ratio (eg: 0.5 mean 50%)",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.0,
                    interactive=True,
                )

                ratio_d = gr.Slider(
                    label="Model D merge ratio (eg: 0.5 mean 50%)",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.0,
                    interactive=True,
                )

            with gr.Group(), gr.Row():
                save_to = gr.Dropdown(
                    label="Save to (path for the file to save...)",
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
                    visible=(not self.headless),
                )
                button_save_to.click(
                    get_saveasfilename_path,
                    inputs=[save_to, lora_ext, lora_ext_name],
                    outputs=save_to,
                    show_progress=False,
                )
                precision = gr.Radio(
                    label="Merge precision",
                    choices=["float", "fp16", "bf16"],
                    value="float",
                    interactive=True,
                )
                save_precision = gr.Radio(
                    label="Save precision",
                    choices=["float", "fp16", "bf16", "fp8"],
                    value="fp16",
                    interactive=True,
                )

                save_to.change(
                    fn=lambda path: gr.Dropdown(choices=[""] + list_save_to(path)),
                    inputs=save_to,
                    outputs=save_to,
                    show_progress=False,
                )

            with gr.Row():
                loading_device = gr.Dropdown(
                    label="Loading device",
                    choices=["cpu", "cuda"],
                    value="cpu",
                    interactive=True,
                )
                working_device = gr.Dropdown(
                    label="Working device",
                    choices=["cpu", "cuda"],
                    value="cpu",
                    interactive=True,
                )

            with gr.Row():
                concat = gr.Checkbox(label="Concat LoRA", value=False)
                shuffle = gr.Checkbox(label="Shuffle LoRA weights", value=False)
                no_metadata = gr.Checkbox(label="Don't save metadata", value=False)
                diffusers  = gr.Checkbox(label="Diffusers LoRA", value=False)

            merge_button = gr.Button("Merge model")

            merge_button.click(
                self.merge_flux_lora,
                inputs=[
                    flux_model,
                    lora_a_model,
                    lora_b_model,
                    lora_c_model,
                    lora_d_model,
                    ratio_a,
                    ratio_b,
                    ratio_c,
                    ratio_d,
                    save_to,
                    precision,
                    save_precision,
                    loading_device,
                    working_device,
                    concat,
                    shuffle,
                    no_metadata,
                    diffusers,
                ],
                show_progress=False,
            )

    def merge_flux_lora(
        self,
        flux_model,
        lora_a_model,
        lora_b_model,
        lora_c_model,
        lora_d_model,
        ratio_a,
        ratio_b,
        ratio_c,
        ratio_d,
        save_to,
        precision,
        save_precision,
        loading_device,
        working_device,
        concat,
        shuffle,
        no_metadata,
        difffusers,
    ):
        log.info("Merge FLUX LoRA...")
        models = [
            lora_a_model,
            lora_b_model,
            lora_c_model,
            lora_d_model,
        ]
        lora_models = [model for model in models if model]
        ratios = [ratio for model, ratio in zip(models, [ratio_a, ratio_b, ratio_c, ratio_d]) if model]

        # if not verify_conditions(flux_model, lora_models):
        #     log.info(
        #         "Warning: Either provide at least one LoRA model along with the FLUX model or at least two LoRA models if no FLUX model is provided."
        #     )
        #     return

        for model in [flux_model] + lora_models:
            if not check_model(model):
                return

        run_cmd = [rf"{PYTHON}", rf"{scriptdir}/sd-scripts/networks/flux_merge_lora.py"]

        if flux_model:
            run_cmd.extend(["--flux_model", rf"{flux_model}"])

        run_cmd.extend([
            "--save_precision", save_precision,
            "--precision", precision,
            "--save_to", rf"{save_to}",
            "--loading_device", loading_device,
            "--working_device", working_device,
        ])

        if lora_models:
            run_cmd.append("--models")
            run_cmd.extend(lora_models)
            run_cmd.append("--ratios")
            run_cmd.extend(map(str, ratios))

        if concat:
            run_cmd.append("--concat")
        if shuffle:
            run_cmd.append("--shuffle")
        if no_metadata:
            run_cmd.append("--no_metadata")
        if difffusers:
            run_cmd.append("--diffusers")

        env = setup_environment()

        # Reconstruct the safe command string for display
        command_to_run = " ".join(run_cmd)
        log.info(f"Executing command: {command_to_run}")

        # Run the command in the sd-scripts folder context
        subprocess.run(run_cmd, env=env)

        log.info("Done merging...")