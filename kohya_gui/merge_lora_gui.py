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


def check_model(model):
    if not model:
        return True
    if not os.path.isfile(model):
        log.info(f"The provided {model} is not a file")
        return False
    return True


def verify_conditions(sd_model, lora_models):
    lora_models_count = sum(1 for model in lora_models if model)
    if sd_model and lora_models_count >= 1:
        return True
    elif not sd_model and lora_models_count >= 2:
        return True
    return False


class GradioMergeLoRaTab:
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
        current_sd_model_dir = os.path.join(scriptdir, "outputs")
        current_save_dir = os.path.join(scriptdir, "outputs")
        current_a_model_dir = current_sd_model_dir
        current_b_model_dir = current_sd_model_dir
        current_c_model_dir = current_sd_model_dir
        current_d_model_dir = current_sd_model_dir

        def list_sd_models(path):
            nonlocal current_sd_model_dir
            current_sd_model_dir = path
            return list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

        def list_a_models(path):
            nonlocal current_a_model_dir
            current_a_model_dir = path
            return list(list_files(path, exts=[".pt", ".safetensors"], all=True))

        def list_b_models(path):
            nonlocal current_b_model_dir
            current_b_model_dir = path
            return list(list_files(path, exts=[".pt", ".safetensors"], all=True))

        def list_c_models(path):
            nonlocal current_c_model_dir
            current_c_model_dir = path
            return list(list_files(path, exts=[".pt", ".safetensors"], all=True))

        def list_d_models(path):
            nonlocal current_d_model_dir
            current_d_model_dir = path
            return list(list_files(path, exts=[".pt", ".safetensors"], all=True))

        def list_save_to(path):
            nonlocal current_save_dir
            current_save_dir = path
            return list(list_files(path, exts=[".ckpt", ".safetensors"], all=True))

        with gr.Tab("Merge LoRA"):
            gr.Markdown(
                "This utility can merge up to 4 LoRA together or alternatively merge up to 4 LoRA into a SD checkpoint."
            )

            lora_ext = gr.Textbox(value="*.safetensors *.pt", visible=False)
            lora_ext_name = gr.Textbox(value="LoRA model types", visible=False)
            ckpt_ext = gr.Textbox(value="*.safetensors *.ckpt", visible=False)
            ckpt_ext_name = gr.Textbox(value="SD model types", visible=False)

            with gr.Group(), gr.Row():
                sd_model = gr.Dropdown(
                    label="SD Model (Optional. Stable Diffusion model path, if you want to merge it with LoRA files)",
                    interactive=True,
                    choices=[""] + list_sd_models(current_sd_model_dir),
                    value="",
                    allow_custom_value=True,
                )
                create_refresh_button(
                    sd_model,
                    lambda: None,
                    lambda: {"choices": list_sd_models(current_sd_model_dir)},
                    "open_folder_small",
                )
                sd_model_file = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not self.headless),
                )
                sd_model_file.click(
                    get_file_path,
                    inputs=[sd_model, ckpt_ext, ckpt_ext_name],
                    outputs=sd_model,
                    show_progress=False,
                )
                sdxl_model = gr.Checkbox(label="SDXL model", value=False)

                sd_model.change(
                    fn=lambda path: gr.Dropdown(choices=[""] + list_sd_models(path)),
                    inputs=sd_model,
                    outputs=sd_model,
                    show_progress=False,
                )

                #secondary event on sd_model for auto-detection of SDXL
                sd_model.change(
                    lambda path: gr.Checkbox(value=SDModelType(path).Is_SDXL()),
                    inputs=sd_model,
                    outputs=sdxl_model
                )

            with gr.Group(), gr.Row():
                lora_a_model = gr.Dropdown(
                    label='LoRA model "A" (path to the LoRA A model)',
                    interactive=True,
                    choices=[""] + list_a_models(current_a_model_dir),
                    value="",
                    allow_custom_value=True,
                )
                create_refresh_button(
                    lora_a_model,
                    lambda: None,
                    lambda: {"choices": list_a_models(current_a_model_dir)},
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
                    choices=[""] + list_b_models(current_b_model_dir),
                    value="",
                    allow_custom_value=True,
                )
                create_refresh_button(
                    lora_b_model,
                    lambda: None,
                    lambda: {"choices": list_b_models(current_b_model_dir)},
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
                    fn=lambda path: gr.Dropdown(choices=[""] + list_a_models(path)),
                    inputs=lora_a_model,
                    outputs=lora_a_model,
                    show_progress=False,
                )
                lora_b_model.change(
                    fn=lambda path: gr.Dropdown(choices=[""] + list_b_models(path)),
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
                    choices=[""] + list_c_models(current_c_model_dir),
                    value="",
                    allow_custom_value=True,
                )
                create_refresh_button(
                    lora_c_model,
                    lambda: None,
                    lambda: {"choices": list_c_models(current_c_model_dir)},
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
                    choices=[""] + list_d_models(current_d_model_dir),
                    value="",
                    allow_custom_value=True,
                )
                create_refresh_button(
                    lora_d_model,
                    lambda: None,
                    lambda: {"choices": list_d_models(current_d_model_dir)},
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
                    fn=lambda path: gr.Dropdown(choices=[""] + list_c_models(path)),
                    inputs=lora_c_model,
                    outputs=lora_c_model,
                    show_progress=False,
                )
                lora_d_model.change(
                    fn=lambda path: gr.Dropdown(choices=[""] + list_d_models(path)),
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
                    choices=[""] + list_save_to(current_d_model_dir),
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
                    choices=["fp16", "bf16", "float"],
                    value="float",
                    interactive=True,
                )
                save_precision = gr.Radio(
                    label="Save precision",
                    choices=["fp16", "bf16", "float"],
                    value="fp16",
                    interactive=True,
                )

                save_to.change(
                    fn=lambda path: gr.Dropdown(choices=[""] + list_save_to(path)),
                    inputs=save_to,
                    outputs=save_to,
                    show_progress=False,
                )

            merge_button = gr.Button("Merge model")

            merge_button.click(
                self.merge_lora,
                inputs=[
                    sd_model,
                    sdxl_model,
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
                ],
                show_progress=False,
            )

    def merge_lora(
        self,
        sd_model,
        sdxl_model,
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
    ):

        log.info("Merge model...")
        models = [
            sd_model,
            lora_a_model,
            lora_b_model,
            lora_c_model,
            lora_d_model,
        ]
        lora_models = models[1:]
        ratios = [ratio_a, ratio_b, ratio_c, ratio_d]

        if not verify_conditions(sd_model, lora_models):
            log.info(
                "Warning: Either provide at least one LoRa model along with the sd_model or at least two LoRa models if no sd_model is provided."
            )
            return

        for model in models:
            if not check_model(model):
                return

        if not sdxl_model:
            run_cmd = [rf"{PYTHON}", rf"{scriptdir}/sd-scripts/networks/merge_lora.py"]
        else:
            run_cmd = [
                rf"{PYTHON}",
                rf"{scriptdir}/sd-scripts/networks/sdxl_merge_lora.py",
            ]

        if sd_model:
            run_cmd.append("--sd_model")
            run_cmd.append(rf"{sd_model}")

        run_cmd.append("--save_precision")
        run_cmd.append(save_precision)
        run_cmd.append("--precision")
        run_cmd.append(precision)
        run_cmd.append("--save_to")
        run_cmd.append(rf"{save_to}")

        # Prepare model and ratios command as lists, including only non-empty models
        valid_models = [model for model in lora_models if model]
        valid_ratios = [ratios[i] for i, model in enumerate(lora_models) if model]

        if valid_models:
            run_cmd.append("--models")
            run_cmd.extend(valid_models)  # Each model is a separate argument
            run_cmd.append("--ratios")
            run_cmd.extend(
                map(str, valid_ratios)
            )  # Convert ratios to strings and include them as separate arguments

        env = setup_environment()

        # Reconstruct the safe command string for display
        command_to_run = " ".join(run_cmd)
        log.info(f"Executing command: {command_to_run}")

        # Run the command in the sd-scripts folder context
        subprocess.run(run_cmd, env=env)

        log.info("Done merging...")
