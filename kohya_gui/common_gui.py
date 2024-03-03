from tkinter import filedialog, Tk
from easygui import msgbox
import os
import re
import gradio as gr
import easygui
import shutil
import sys
import json

from .custom_logging import setup_logging
from datetime import datetime

# Set up logging
log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

scriptdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# define a list of substrings to search for v2 base models
V2_BASE_MODELS = [
    "stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned",
    "stabilityai/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-2-base",
]

# define a list of substrings to search for v_parameterization models
V_PARAMETERIZATION_MODELS = [
    "stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2",
]

# define a list of substrings to v1.x models
V1_MODELS = [
    "CompVis/stable-diffusion-v1-4",
    "runwayml/stable-diffusion-v1-5",
]

# define a list of substrings to search for SDXL base models
SDXL_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
]

# define a list of substrings to search for
ALL_PRESET_MODELS = V2_BASE_MODELS + V_PARAMETERIZATION_MODELS + V1_MODELS + SDXL_MODELS

ENV_EXCLUSION = ["COLAB_GPU", "RUNPOD_POD_ID"]


def check_if_model_exist(output_name, output_dir, save_model_as, headless=False):
    if headless:
        log.info(
            "Headless mode, skipping verification if model already exist... if model already exist it will be overwritten..."
        )
        return False

    if save_model_as in ["diffusers", "diffusers_safetendors"]:
        ckpt_folder = os.path.join(output_dir, output_name)
        if os.path.isdir(ckpt_folder):
            msg = f"A diffuser model with the same name {ckpt_folder} already exists. Do you want to overwrite it?"
            if not easygui.ynbox(msg, "Overwrite Existing Model?"):
                log.info("Aborting training due to existing model with same name...")
                return True
    elif save_model_as in ["ckpt", "safetensors"]:
        ckpt_file = os.path.join(output_dir, output_name + "." + save_model_as)
        if os.path.isfile(ckpt_file):
            msg = f"A model with the same file name {ckpt_file} already exists. Do you want to overwrite it?"
            if not easygui.ynbox(msg, "Overwrite Existing Model?"):
                log.info("Aborting training due to existing model with same name...")
                return True
    else:
        log.info(
            'Can\'t verify if existing model exist when save model is set a "same as source model", continuing to train model...'
        )
        return False

    return False


def output_message(msg="", title="", headless=False):
    if headless:
        log.info(msg)
    else:
        msgbox(msg=msg, title=title)


def update_my_data(my_data):
    # Update the optimizer based on the use_8bit_adam flag
    use_8bit_adam = my_data.get("use_8bit_adam", False)
    my_data.setdefault("optimizer", "AdamW8bit" if use_8bit_adam else "AdamW")

    # Update model_list to custom if empty or pretrained_model_name_or_path is not a preset model
    model_list = my_data.get("model_list", [])
    pretrained_model_name_or_path = my_data.get("pretrained_model_name_or_path", "")
    if not model_list or pretrained_model_name_or_path not in ALL_PRESET_MODELS:
        my_data["model_list"] = "custom"

    # Convert values to int if they are strings
    for key in ["epoch", "save_every_n_epochs", "lr_warmup"]:
        value = my_data.get(key, 0)
        if isinstance(value, str) and value.strip().isdigit():
            my_data[key] = int(value)
        elif not value:
            my_data[key] = 0

    # Convert values to float if they are strings
    for key in ["noise_offset", "learning_rate", "text_encoder_lr", "unet_lr"]:
        value = my_data.get(key, 0)
        if isinstance(value, str) and value.strip().isdigit():
            my_data[key] = float(value)
        elif not value:
            my_data[key] = 0

    # Update LoRA_type if it is set to LoCon
    if my_data.get("LoRA_type", "Standard") == "LoCon":
        my_data["LoRA_type"] = "LyCORIS/LoCon"

    # Update model save choices due to changes for LoRA and TI training
    if "save_model_as" in my_data:
        if (
            my_data.get("LoRA_type") or my_data.get("num_vectors_per_token")
        ) and my_data.get("save_model_as") not in ["safetensors", "ckpt"]:
            message = "Updating save_model_as to safetensors because the current value in the config file is no longer applicable to {}"
            if my_data.get("LoRA_type"):
                log.info(message.format("LoRA"))
            if my_data.get("num_vectors_per_token"):
                log.info(message.format("TI"))
            my_data["save_model_as"] = "safetensors"

    # Update xformers if it is set to True and is a boolean
    xformers_value = my_data.get("xformers", None)
    if isinstance(xformers_value, bool):
        if xformers_value:
            my_data["xformers"] = "xformers"
        else:
            my_data["xformers"] = "none"

    return my_data


def get_dir_and_file(file_path):
    dir_path, file_name = os.path.split(file_path)
    return (dir_path, file_name)


def get_file_path(
    file_path="", default_extension=".json", extension_name="Config files"
):
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != "darwin":
        current_file_path = file_path
        # log.info(f'current file path: {current_file_path}')

        initial_dir, initial_file = get_dir_and_file(file_path)

        # Create a hidden Tkinter root window
        root = Tk()
        root.wm_attributes("-topmost", 1)
        root.withdraw()

        # Show the open file dialog and get the selected file path
        file_path = filedialog.askopenfilename(
            filetypes=(
                (extension_name, f"*{default_extension}"),
                ("All files", "*.*"),
            ),
            defaultextension=default_extension,
            initialfile=initial_file,
            initialdir=initial_dir,
        )

        # Destroy the hidden root window
        root.destroy()

        # If no file is selected, use the current file path
        if not file_path:
            file_path = current_file_path
        current_file_path = file_path
        # log.info(f'current file path: {current_file_path}')

    return file_path


def get_any_file_path(file_path=""):
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != "darwin":
        current_file_path = file_path
        # log.info(f'current file path: {current_file_path}')

        initial_dir, initial_file = get_dir_and_file(file_path)

        root = Tk()
        root.wm_attributes("-topmost", 1)
        root.withdraw()
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            initialfile=initial_file,
        )
        root.destroy()

        if file_path == "":
            file_path = current_file_path

    return file_path


def get_folder_path(folder_path=""):
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != "darwin":
        current_folder_path = folder_path

        initial_dir, initial_file = get_dir_and_file(folder_path)

        root = Tk()
        root.wm_attributes("-topmost", 1)
        root.withdraw()
        folder_path = filedialog.askdirectory(initialdir=initial_dir)
        root.destroy()

        if folder_path == "":
            folder_path = current_folder_path

    return folder_path


def get_saveasfile_path(
    file_path="", defaultextension=".json", extension_name="Config files"
):
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != "darwin":
        current_file_path = file_path
        # log.info(f'current file path: {current_file_path}')

        initial_dir, initial_file = get_dir_and_file(file_path)

        root = Tk()
        root.wm_attributes("-topmost", 1)
        root.withdraw()
        save_file_path = filedialog.asksaveasfile(
            filetypes=(
                (f"{extension_name}", f"{defaultextension}"),
                ("All files", "*"),
            ),
            defaultextension=defaultextension,
            initialdir=initial_dir,
            initialfile=initial_file,
        )
        root.destroy()

        # log.info(save_file_path)

        if save_file_path == None:
            file_path = current_file_path
        else:
            log.info(save_file_path.name)
            file_path = save_file_path.name

        # log.info(file_path)

    return file_path


def get_saveasfilename_path(
    file_path="", extensions="*", extension_name="Config files"
):
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != "darwin":
        current_file_path = file_path
        # log.info(f'current file path: {current_file_path}')

        initial_dir, initial_file = get_dir_and_file(file_path)

        root = Tk()
        root.wm_attributes("-topmost", 1)
        root.withdraw()
        save_file_path = filedialog.asksaveasfilename(
            filetypes=(
                (f"{extension_name}", f"{extensions}"),
                ("All files", "*"),
            ),
            defaultextension=extensions,
            initialdir=initial_dir,
            initialfile=initial_file,
        )
        root.destroy()

        if save_file_path == "":
            file_path = current_file_path
        else:
            # log.info(save_file_path)
            file_path = save_file_path

    return file_path


def add_pre_postfix(
    folder: str = "",
    prefix: str = "",
    postfix: str = "",
    caption_file_ext: str = ".caption",
) -> None:
    """
    Add prefix and/or postfix to the content of caption files within a folder.
    If no caption files are found, create one with the requested prefix and/or postfix.

    Args:
        folder (str): Path to the folder containing caption files.
        prefix (str, optional): Prefix to add to the content of the caption files.
        postfix (str, optional): Postfix to add to the content of the caption files.
        caption_file_ext (str, optional): Extension of the caption files.
    """

    if prefix == "" and postfix == "":
        return

    image_extensions = (".jpg", ".jpeg", ".png", ".webp")
    image_files = [
        f for f in os.listdir(folder) if f.lower().endswith(image_extensions)
    ]

    for image_file in image_files:
        caption_file_name = os.path.splitext(image_file)[0] + caption_file_ext
        caption_file_path = os.path.join(folder, caption_file_name)

        if not os.path.exists(caption_file_path):
            with open(caption_file_path, "w", encoding="utf8") as f:
                separator = " " if prefix and postfix else ""
                f.write(f"{prefix}{separator}{postfix}")
        else:
            with open(caption_file_path, "r+", encoding="utf8") as f:
                content = f.read()
                content = content.rstrip()
                f.seek(0, 0)

                prefix_separator = " " if prefix else ""
                postfix_separator = " " if postfix else ""
                f.write(
                    f"{prefix}{prefix_separator}{content}{postfix_separator}{postfix}"
                )


def has_ext_files(folder_path: str, file_extension: str) -> bool:
    """
    Check if there are any files with the specified extension in the given folder.

    Args:
        folder_path (str): Path to the folder containing files.
        file_extension (str): Extension of the files to look for.

    Returns:
        bool: True if files with the specified extension are found, False otherwise.
    """
    for file in os.listdir(folder_path):
        if file.endswith(file_extension):
            return True
    return False


def find_replace(
    folder_path: str = "",
    caption_file_ext: str = ".caption",
    search_text: str = "",
    replace_text: str = "",
) -> None:
    """
    Find and replace text in caption files within a folder.

    Args:
        folder_path (str, optional): Path to the folder containing caption files.
        caption_file_ext (str, optional): Extension of the caption files.
        search_text (str, optional): Text to search for in the caption files.
        replace_text (str, optional): Text to replace the search text with.
    """
    log.info("Running caption find/replace")

    if not has_ext_files(folder_path, caption_file_ext):
        msgbox(
            f"No files with extension {caption_file_ext} were found in {folder_path}..."
        )
        return

    if search_text == "":
        return

    caption_files = [f for f in os.listdir(folder_path) if f.endswith(caption_file_ext)]

    for caption_file in caption_files:
        with open(os.path.join(folder_path, caption_file), "r", errors="ignore") as f:
            content = f.read()

        content = content.replace(search_text, replace_text)

        with open(os.path.join(folder_path, caption_file), "w") as f:
            f.write(content)


def color_aug_changed(color_aug):
    if color_aug:
        msgbox(
            'Disabling "Cache latent" because "Color augmentation" has been selected...'
        )
        return gr.Checkbox.update(value=False, interactive=False)
    else:
        return gr.Checkbox.update(value=True, interactive=True)


def save_inference_file(output_dir, v2, v_parameterization, output_name):
    # List all files in the directory
    files = os.listdir(output_dir)

    # Iterate over the list of files
    for file in files:
        # Check if the file starts with the value of output_name
        if file.startswith(output_name):
            # Check if it is a file or a directory
            if os.path.isfile(os.path.join(output_dir, file)):
                # Split the file name and extension
                file_name, ext = os.path.splitext(file)

                # Copy the v2-inference-v.yaml file to the current file, with a .yaml extension
                if v2 and v_parameterization:
                    log.info(
                        f"Saving v2-inference-v.yaml as {output_dir}/{file_name}.yaml"
                    )
                    shutil.copy(
                        fr"{scriptdir}/v2_inference/v2-inference-v.yaml",
                        f"{output_dir}/{file_name}.yaml",
                    )
                elif v2:
                    log.info(
                        f"Saving v2-inference.yaml as {output_dir}/{file_name}.yaml"
                    )
                    shutil.copy(
                        fr"{scriptdir}/v2_inference/v2-inference.yaml",
                        f"{output_dir}/{file_name}.yaml",
                    )


def set_pretrained_model_name_or_path_input(
    model_list,
    pretrained_model_name_or_path,
    pretrained_model_name_or_path_file,
    pretrained_model_name_or_path_folder,
    v2,
    v_parameterization,
    sdxl,
):
    # Check if the given model_list is in the list of SDXL models
    if str(model_list) in SDXL_MODELS:
        log.info("SDXL model selected. Setting sdxl parameters")
        v2 = gr.Checkbox(value=False, visible=False)
        v_parameterization = gr.Checkbox(value=False, visible=False)
        sdxl = gr.Checkbox(value=True, visible=False)
        pretrained_model_name_or_path = gr.Textbox(
            value=str(model_list), visible=False
        )
        pretrained_model_name_or_path_file = gr.Button(visible=False)
        pretrained_model_name_or_path_folder = gr.Button(visible=False)
        return (
            model_list,
            pretrained_model_name_or_path,
            pretrained_model_name_or_path_file,
            pretrained_model_name_or_path_folder,
            v2,
            v_parameterization,
            sdxl,
        )

    # Check if the given model_list is in the list of V2 base models
    if str(model_list) in V2_BASE_MODELS:
        log.info("SD v2 base model selected. Setting --v2 parameter")
        v2 = gr.Checkbox(value=True, visible=False)
        v_parameterization = gr.Checkbox(value=False, visible=False)
        sdxl = gr.Checkbox(value=False, visible=False)
        pretrained_model_name_or_path = gr.Textbox(
            value=str(model_list), visible=False
        )
        pretrained_model_name_or_path_file = gr.Button(visible=False)
        pretrained_model_name_or_path_folder = gr.Button(visible=False)
        return (
            model_list,
            pretrained_model_name_or_path,
            pretrained_model_name_or_path_file,
            pretrained_model_name_or_path_folder,
            v2,
            v_parameterization,
            sdxl,
        )

    # Check if the given model_list is in the list of V parameterization models
    if str(model_list) in V_PARAMETERIZATION_MODELS:
        log.info(
            "SD v2 model selected. Setting --v2 and --v_parameterization parameters"
        )
        v2 = gr.Checkbox(value=True, visible=False)
        v_parameterization = gr.Checkbox(value=True, visible=False)
        sdxl = gr.Checkbox(value=False, visible=False)
        pretrained_model_name_or_path = gr.Textbox(
            value=str(model_list), visible=False
        )
        pretrained_model_name_or_path_file = gr.Button(visible=False)
        pretrained_model_name_or_path_folder = gr.Button(visible=False)
        return (
            model_list,
            pretrained_model_name_or_path,
            pretrained_model_name_or_path_file,
            pretrained_model_name_or_path_folder,
            v2,
            v_parameterization,
            sdxl,
        )

    # Check if the given model_list is in the list of V1 models
    if str(model_list) in V1_MODELS:
        log.info(f"{model_list} model selected.")
        v2 = gr.Checkbox(value=False, visible=False)
        v_parameterization = gr.Checkbox(value=False, visible=False)
        sdxl = gr.Checkbox(value=False, visible=False)
        pretrained_model_name_or_path = gr.Textbox(
            value=str(model_list), visible=False
        )
        pretrained_model_name_or_path_file = gr.Button(visible=False)
        pretrained_model_name_or_path_folder = gr.Button(visible=False)
        return (
            model_list,
            pretrained_model_name_or_path,
            pretrained_model_name_or_path_file,
            pretrained_model_name_or_path_folder,
            v2,
            v_parameterization,
            sdxl,
        )

    # Check if the model_list is set to 'custom'
    if model_list == "custom":
        v2 = gr.Checkbox(visible=True)
        v_parameterization = gr.Checkbox(visible=True)
        sdxl = gr.Checkbox(visible=True)
        pretrained_model_name_or_path = gr.Textbox(visible=True)
        pretrained_model_name_or_path_file = gr.Button(visible=True)
        pretrained_model_name_or_path_folder = gr.Button(visible=True)
        return (
            model_list,
            pretrained_model_name_or_path,
            pretrained_model_name_or_path_file,
            pretrained_model_name_or_path_folder,
            v2,
            v_parameterization,
            sdxl,
        )


###
### Gradio common GUI section
###


def get_pretrained_model_name_or_path_file(model_list, pretrained_model_name_or_path):
    pretrained_model_name_or_path = get_any_file_path(pretrained_model_name_or_path)
    # set_model_list(model_list, pretrained_model_name_or_path)


def get_int_or_default(kwargs, key, default_value=0):
    value = kwargs.get(key, default_value)
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        return int(value)
    elif isinstance(value, float):
        return int(value)
    else:
        log.info(
            f"{key} is not an int, float or a string, setting value to {default_value}"
        )
        return default_value


def get_float_or_default(kwargs, key, default_value=0.0):
    value = kwargs.get(key, default_value)
    if isinstance(value, float):
        return value
    elif isinstance(value, int):
        return float(value)
    elif isinstance(value, str):
        return float(value)
    else:
        log.info(
            f"{key} is not an int, float or a string, setting value to {default_value}"
        )
        return default_value


def get_str_or_default(kwargs, key, default_value=""):
    value = kwargs.get(key, default_value)
    if isinstance(value, str):
        return value
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, str):
        return str(value)
    else:
        return default_value


# def run_cmd_training(**kwargs):
#     run_cmd = ""

#     lr_scheduler = kwargs.get("lr_scheduler", "")
#     if lr_scheduler:
#         run_cmd += f' --lr_scheduler="{lr_scheduler}"'

#     lr_warmup_steps = kwargs.get("lr_warmup_steps", "")
#     if lr_warmup_steps:
#         if lr_scheduler == "constant":
#             log.info("Can't use LR warmup with LR Scheduler constant... ignoring...")
#         else:
#             run_cmd += f' --lr_warmup_steps="{lr_warmup_steps}"'

#     train_batch_size = kwargs.get("train_batch_size", "")
#     if train_batch_size:
#         run_cmd += f' --train_batch_size="{train_batch_size}"'

#     max_train_steps = kwargs.get("max_train_steps", "")
#     if max_train_steps:
#         run_cmd += f' --max_train_steps="{max_train_steps}"'

#     save_every_n_epochs = kwargs.get("save_every_n_epochs")
#     if save_every_n_epochs:
#         run_cmd += f' --save_every_n_epochs="{int(save_every_n_epochs)}"'

#     mixed_precision = kwargs.get("mixed_precision", "")
#     if mixed_precision:
#         run_cmd += f' --mixed_precision="{mixed_precision}"'

#     save_precision = kwargs.get("save_precision", "")
#     if save_precision:
#         run_cmd += f' --save_precision="{save_precision}"'

#     seed = kwargs.get("seed", "")
#     if seed != "":
#         run_cmd += f' --seed="{seed}"'

#     caption_extension = kwargs.get("caption_extension", "")
#     if caption_extension:
#         run_cmd += f' --caption_extension="{caption_extension}"'

#     cache_latents = kwargs.get("cache_latents")
#     if cache_latents:
#         run_cmd += " --cache_latents"

#     cache_latents_to_disk = kwargs.get("cache_latents_to_disk")
#     if cache_latents_to_disk:
#         run_cmd += " --cache_latents_to_disk"

#     optimizer_type = kwargs.get("optimizer", "AdamW")
#     run_cmd += f' --optimizer_type="{optimizer_type}"'

#     optimizer_args = kwargs.get("optimizer_args", "")
#     if optimizer_args != "":
#         run_cmd += f" --optimizer_args {optimizer_args}"

#     lr_scheduler_args = kwargs.get("lr_scheduler_args", "")
#     if lr_scheduler_args != "":
#         run_cmd += f" --lr_scheduler_args {lr_scheduler_args}"

#     max_grad_norm = kwargs.get("max_grad_norm", "")
#     if max_grad_norm != "":
#         run_cmd += f' --max_grad_norm="{max_grad_norm}"'

#     return run_cmd


def run_cmd_advanced_training(**kwargs):
    run_cmd = ""

    additional_parameters = kwargs.get("additional_parameters")
    if additional_parameters:
        run_cmd += f" {additional_parameters}"

    block_lr = kwargs.get("block_lr")
    if block_lr:
        run_cmd += f' --block_lr="{block_lr}"'

    bucket_no_upscale = kwargs.get("bucket_no_upscale")
    if bucket_no_upscale:
        run_cmd += " --bucket_no_upscale"

    bucket_reso_steps = kwargs.get("bucket_reso_steps")
    if bucket_reso_steps:
        run_cmd += f" --bucket_reso_steps={int(bucket_reso_steps)}"

    cache_latents = kwargs.get("cache_latents")
    if cache_latents:
        run_cmd += " --cache_latents"

    cache_latents_to_disk = kwargs.get("cache_latents_to_disk")
    if cache_latents_to_disk:
        run_cmd += " --cache_latents_to_disk"

    cache_text_encoder_outputs = kwargs.get("cache_text_encoder_outputs")
    if cache_text_encoder_outputs:
        run_cmd += " --cache_text_encoder_outputs"

    caption_dropout_every_n_epochs = kwargs.get("caption_dropout_every_n_epochs")
    if caption_dropout_every_n_epochs and int(caption_dropout_every_n_epochs) > 0:
        run_cmd += (
            f' --caption_dropout_every_n_epochs="{int(caption_dropout_every_n_epochs)}"'
        )

    caption_dropout_rate = kwargs.get("caption_dropout_rate")
    if caption_dropout_rate and float(caption_dropout_rate) > 0:
        run_cmd += f' --caption_dropout_rate="{caption_dropout_rate}"'

    caption_extension = kwargs.get("caption_extension")
    if caption_extension:
        run_cmd += f' --caption_extension="{caption_extension}"'

    clip_skip = kwargs.get("clip_skip")
    if clip_skip and int(clip_skip) > 1:
        run_cmd += f" --clip_skip={int(clip_skip)}"

    color_aug = kwargs.get("color_aug")
    if color_aug:
        run_cmd += " --color_aug"

    dataset_repeats = kwargs.get("dataset_repeats")
    if dataset_repeats:
        run_cmd += f' --dataset_repeats="{dataset_repeats}"'

    debiased_estimation_loss = kwargs.get("debiased_estimation_loss")
    if debiased_estimation_loss:
        run_cmd += " --debiased_estimation_loss"

    dim_from_weights = kwargs.get("dim_from_weights")
    if dim_from_weights and kwargs.get(
        "lora_network_weights"
    ):  # Only if lora_network_weights is true
        run_cmd += f" --dim_from_weights"

    enable_bucket = kwargs.get("enable_bucket")
    if enable_bucket:
        min_bucket_reso = kwargs.get("min_bucket_reso")
        max_bucket_reso = kwargs.get("max_bucket_reso")
        if min_bucket_reso and max_bucket_reso:
            run_cmd += f" --enable_bucket --min_bucket_reso={min_bucket_reso} --max_bucket_reso={max_bucket_reso}"

    in_json = kwargs.get("in_json")
    if in_json:
        run_cmd += f' --in_json="{in_json}"'

    flip_aug = kwargs.get("flip_aug")
    if flip_aug:
        run_cmd += " --flip_aug"

    fp8_base = kwargs.get("fp8_base")
    if fp8_base:
        run_cmd += " --fp8_base"

    full_bf16 = kwargs.get("full_bf16")
    if full_bf16:
        run_cmd += " --full_bf16"

    full_fp16 = kwargs.get("full_fp16")
    if full_fp16:
        run_cmd += " --full_fp16"

    gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps")
    if gradient_accumulation_steps and int(gradient_accumulation_steps) > 1:
        run_cmd += f" --gradient_accumulation_steps={int(gradient_accumulation_steps)}"

    gradient_checkpointing = kwargs.get("gradient_checkpointing")
    if gradient_checkpointing:
        run_cmd += " --gradient_checkpointing"

    keep_tokens = kwargs.get("keep_tokens")
    if keep_tokens and int(keep_tokens) > 0:
        run_cmd += f' --keep_tokens="{int(keep_tokens)}"'

    learning_rate = kwargs.get("learning_rate")
    if learning_rate:
        run_cmd += f' --learning_rate="{learning_rate}"'

    learning_rate_te = kwargs.get("learning_rate_te")
    if learning_rate_te:
        run_cmd += f' --learning_rate_te="{learning_rate_te}"'

    learning_rate_te1 = kwargs.get("learning_rate_te1")
    if learning_rate_te1:
        run_cmd += f' --learning_rate_te1="{learning_rate_te1}"'

    learning_rate_te2 = kwargs.get("learning_rate_te2")
    if learning_rate_te2:
        run_cmd += f' --learning_rate_te2="{learning_rate_te2}"'

    logging_dir = kwargs.get("logging_dir")
    if logging_dir:
        if logging_dir.startswith('"') and logging_dir.endswith('"'):
            logging_dir = logging_dir[1:-1]
        if os.path.exists(logging_dir):
            run_cmd += fr' --logging_dir="{logging_dir}"'

    lora_network_weights = kwargs.get("lora_network_weights")
    if lora_network_weights:
        run_cmd += f' --network_weights="{lora_network_weights}"' # Yes, the parameter is now called network_weights instead of lora_network_weights

    lr_scheduler = kwargs.get("lr_scheduler")
    if lr_scheduler:
        run_cmd += f' --lr_scheduler="{lr_scheduler}"'

    lr_scheduler_args = kwargs.get("lr_scheduler_args")
    if lr_scheduler_args and lr_scheduler_args != "":
        run_cmd += f" --lr_scheduler_args {lr_scheduler_args}"

    lr_scheduler_num_cycles = kwargs.get("lr_scheduler_num_cycles")
    if lr_scheduler_num_cycles and not lr_scheduler_num_cycles == "":
        run_cmd += f' --lr_scheduler_num_cycles="{lr_scheduler_num_cycles}"'
    else:
        epoch = kwargs.get("epoch")
        if epoch:
            run_cmd += f' --lr_scheduler_num_cycles="{epoch}"'

    lr_scheduler_power = kwargs.get("lr_scheduler_power")
    if lr_scheduler_power and not lr_scheduler_power == "":
        run_cmd += f' --lr_scheduler_power="{lr_scheduler_power}"'

    lr_warmup_steps = kwargs.get("lr_warmup_steps")
    if lr_warmup_steps:
        if lr_scheduler == "constant":
            log.info("Can't use LR warmup with LR Scheduler constant... ignoring...")
        else:
            run_cmd += f' --lr_warmup_steps="{lr_warmup_steps}"'

    gpu_ids = kwargs.get("gpu_ids")
    if gpu_ids:
        run_cmd += f' --gpu_ids="{gpu_ids}"'

    max_data_loader_n_workers = kwargs.get("max_data_loader_n_workers")
    if max_data_loader_n_workers and not max_data_loader_n_workers == "":
        run_cmd += f' --max_data_loader_n_workers="{max_data_loader_n_workers}"'

    max_grad_norm = kwargs.get("max_grad_norm")
    if max_grad_norm and max_grad_norm != "":
        run_cmd += f' --max_grad_norm="{max_grad_norm}"'

    max_resolution = kwargs.get("max_resolution")
    if max_resolution:
        run_cmd += f' --resolution="{max_resolution}"'

    max_timestep = kwargs.get("max_timestep")
    if max_timestep and int(max_timestep) < 1000:
        run_cmd += f" --max_timestep={int(max_timestep)}"

    max_token_length = kwargs.get("max_token_length")
    if max_token_length and int(max_token_length) > 75:
        run_cmd += f" --max_token_length={int(max_token_length)}"

    max_train_epochs = kwargs.get("max_train_epochs")
    if max_train_epochs and not max_train_epochs == "":
        run_cmd += f" --max_train_epochs={max_train_epochs}"

    max_train_steps = kwargs.get("max_train_steps")
    if max_train_steps:
        run_cmd += f' --max_train_steps="{max_train_steps}"'

    mem_eff_attn = kwargs.get("mem_eff_attn")
    if mem_eff_attn:
        run_cmd += " --mem_eff_attn"

    min_snr_gamma = kwargs.get("min_snr_gamma")
    if min_snr_gamma and int(min_snr_gamma) >= 1:
        run_cmd += f" --min_snr_gamma={int(min_snr_gamma)}"

    min_timestep = kwargs.get("min_timestep")
    if min_timestep and int(min_timestep) > 0:
        run_cmd += f" --min_timestep={int(min_timestep)}"

    mixed_precision = kwargs.get("mixed_precision")
    if mixed_precision:
        run_cmd += f' --mixed_precision="{mixed_precision}"'

    multi_gpu = kwargs.get("multi_gpu")
    if multi_gpu:
        run_cmd += " --multi_gpu"

    network_alpha = kwargs.get("network_alpha")
    if network_alpha:
        run_cmd += f' --network_alpha="{network_alpha}"'

    network_args = kwargs.get("network_args")
    if network_args and len(network_args):
        run_cmd += f" --network_args{network_args}"

    network_dim = kwargs.get("network_dim")
    if network_dim:
        run_cmd += f" --network_dim={network_dim}"

    network_dropout = kwargs.get("network_dropout")
    if network_dropout and network_dropout > 0.0:
        run_cmd += f" --network_dropout={network_dropout}"

    network_module = kwargs.get("network_module")
    if network_module:
        run_cmd += f" --network_module={network_module}"

    network_train_text_encoder_only = kwargs.get("network_train_text_encoder_only")
    if network_train_text_encoder_only:
        run_cmd += " --network_train_text_encoder_only"

    network_train_unet_only = kwargs.get("network_train_unet_only")
    if network_train_unet_only:
        run_cmd += " --network_train_unet_only"

    no_half_vae = kwargs.get("no_half_vae")
    if no_half_vae:
        run_cmd += " --no_half_vae"

    no_token_padding = kwargs.get("no_token_padding")
    if no_token_padding:
        run_cmd += " --no_token_padding"

    noise_offset_type = kwargs.get("noise_offset_type")
    if noise_offset_type and noise_offset_type == "Original":
        noise_offset = kwargs.get("noise_offset")
        if noise_offset and float(noise_offset) > 0:
            run_cmd += f" --noise_offset={float(noise_offset)}"

        adaptive_noise_scale = kwargs.get("adaptive_noise_scale")
        if (
            adaptive_noise_scale
            and float(adaptive_noise_scale) != 0
            and float(noise_offset) > 0
        ):
            run_cmd += f" --adaptive_noise_scale={float(adaptive_noise_scale)}"
    elif noise_offset_type and noise_offset_type == "Multires":
        multires_noise_iterations = kwargs.get("multires_noise_iterations")
        if int(multires_noise_iterations) > 0:
            run_cmd += (
                f' --multires_noise_iterations="{int(multires_noise_iterations)}"'
            )

        multires_noise_discount = kwargs.get("multires_noise_discount")
        if multires_noise_discount and float(multires_noise_discount) > 0:
            run_cmd += f' --multires_noise_discount="{float(multires_noise_discount)}"'

    num_machines = kwargs.get("num_machines")
    if num_machines and int(num_machines) > 1:
        run_cmd += f" --num_machines={int(num_machines)}"

    num_processes = kwargs.get("num_processes")
    if num_processes and int(num_processes) > 1:
        run_cmd += f" --num_processes={int(num_processes)}"

    num_cpu_threads_per_process = kwargs.get("num_cpu_threads_per_process")
    if num_cpu_threads_per_process and int(num_cpu_threads_per_process) > 1:
        run_cmd += f" --num_cpu_threads_per_process={int(num_cpu_threads_per_process)}"

    optimizer_args = kwargs.get("optimizer_args")
    if optimizer_args and optimizer_args != "":
        run_cmd += f" --optimizer_args {optimizer_args}"

    optimizer_type = kwargs.get("optimizer")
    if optimizer_type:
        run_cmd += f' --optimizer_type="{optimizer_type}"'

    output_dir = kwargs.get("output_dir")
    if output_dir:
        if output_dir.startswith('"') and output_dir.endswith('"'):
            output_dir = output_dir[1:-1]
        if os.path.exists(output_dir):
            run_cmd += fr' --output_dir="{output_dir}"'

    output_name = kwargs.get("output_name")
    if output_name and not output_name == "":
        run_cmd += f' --output_name="{output_name}"'

    persistent_data_loader_workers = kwargs.get("persistent_data_loader_workers")
    if persistent_data_loader_workers:
        run_cmd += " --persistent_data_loader_workers"

    pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path")
    if pretrained_model_name_or_path:
        run_cmd += f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'

    prior_loss_weight = kwargs.get("prior_loss_weight")
    if prior_loss_weight and not float(prior_loss_weight) == 1.0:
        run_cmd += f" --prior_loss_weight={prior_loss_weight}"

    random_crop = kwargs.get("random_crop")
    if random_crop:
        run_cmd += " --random_crop"

    reg_data_dir = kwargs.get("reg_data_dir")
    if reg_data_dir and len(reg_data_dir):
        if reg_data_dir.startswith('"') and reg_data_dir.endswith('"'):
            reg_data_dir = reg_data_dir[1:-1]
        if os.path.isdir(reg_data_dir):
            run_cmd += fr' --reg_data_dir="{reg_data_dir}"'

    resume = kwargs.get("resume")
    if resume:
        run_cmd += f' --resume="{resume}"'

    save_every_n_epochs = kwargs.get("save_every_n_epochs")
    if save_every_n_epochs:
        run_cmd += f' --save_every_n_epochs="{int(save_every_n_epochs)}"'

    save_every_n_steps = kwargs.get("save_every_n_steps")
    if save_every_n_steps and int(save_every_n_steps) > 0:
        run_cmd += f' --save_every_n_steps="{int(save_every_n_steps)}"'

    save_last_n_steps = kwargs.get("save_last_n_steps")
    if save_last_n_steps and int(save_last_n_steps) > 0:
        run_cmd += f' --save_last_n_steps="{int(save_last_n_steps)}"'

    save_last_n_steps_state = kwargs.get("save_last_n_steps_state")
    if save_last_n_steps_state and int(save_last_n_steps_state) > 0:
        run_cmd += f' --save_last_n_steps_state="{int(save_last_n_steps_state)}"'

    save_model_as = kwargs.get("save_model_as")
    if save_model_as and not save_model_as == "same as source model":
        run_cmd += f" --save_model_as={save_model_as}"

    save_precision = kwargs.get("save_precision")
    if save_precision:
        run_cmd += f' --save_precision="{save_precision}"'

    save_state = kwargs.get("save_state")
    if save_state:
        run_cmd += " --save_state"

    scale_v_pred_loss_like_noise_pred = kwargs.get("scale_v_pred_loss_like_noise_pred")
    if scale_v_pred_loss_like_noise_pred:
        run_cmd += " --scale_v_pred_loss_like_noise_pred"

    scale_weight_norms = kwargs.get("scale_weight_norms")
    if scale_weight_norms and scale_weight_norms > 0.0:
        run_cmd += f' --scale_weight_norms="{scale_weight_norms}"'

    seed = kwargs.get("seed")
    if seed and seed != "":
        run_cmd += f' --seed="{seed}"'

    shuffle_caption = kwargs.get("shuffle_caption")
    if shuffle_caption:
        run_cmd += " --shuffle_caption"

    stop_text_encoder_training = kwargs.get("stop_text_encoder_training")
    if stop_text_encoder_training and stop_text_encoder_training > 0:
        run_cmd += f' --stop_text_encoder_training="{stop_text_encoder_training}"'

    text_encoder_lr = kwargs.get("text_encoder_lr")
    if text_encoder_lr and (float(text_encoder_lr) > 0):
        run_cmd += f" --text_encoder_lr={text_encoder_lr}"

    train_batch_size = kwargs.get("train_batch_size")
    if train_batch_size:
        run_cmd += f' --train_batch_size="{train_batch_size}"'

    training_comment = kwargs.get("training_comment")
    if training_comment and len(training_comment):
        run_cmd += f' --training_comment="{training_comment}"'

    train_data_dir = kwargs.get("train_data_dir")
    if train_data_dir:
        if train_data_dir.startswith('"') and train_data_dir.endswith('"'):
            train_data_dir = train_data_dir[1:-1]
        if os.path.exists(train_data_dir):
            run_cmd += fr' --train_data_dir="{train_data_dir}"'

    train_text_encoder = kwargs.get("train_text_encoder")
    if train_text_encoder:
        run_cmd += " --train_text_encoder"

    unet_lr = kwargs.get("unet_lr")
    if unet_lr and (float(unet_lr) > 0):
        run_cmd += f" --unet_lr={unet_lr}"

    use_wandb = kwargs.get("use_wandb")
    if use_wandb:
        run_cmd += " --log_with wandb"

    v_parameterization = kwargs.get("v_parameterization")
    if v_parameterization:
        run_cmd += " --v_parameterization"

    v_pred_like_loss = kwargs.get("v_pred_like_loss")
    if v_pred_like_loss and float(v_pred_like_loss) > 0:
        run_cmd += f' --v_pred_like_loss="{float(v_pred_like_loss)}"'

    v2 = kwargs.get("v2")
    if v2:
        run_cmd += " --v2"

    vae = kwargs.get("vae")
    if vae and not vae == "":
        run_cmd += f' --vae="{vae}"'

    vae_batch_size = kwargs.get("vae_batch_size")
    if vae_batch_size and int(vae_batch_size) > 0:
        run_cmd += f' --vae_batch_size="{int(vae_batch_size)}"'

    wandb_api_key = kwargs.get("wandb_api_key")
    if wandb_api_key:
        run_cmd += f' --wandb_api_key="{wandb_api_key}"'

    weighted_captions = kwargs.get("weighted_captions")
    if weighted_captions:
        run_cmd += " --weighted_captions"

    xformers = kwargs.get("xformers")
    if xformers and xformers == "xformers":
        run_cmd += " --xformers"
    elif xformers and xformers == "sdpa":
        run_cmd += " --sdpa"

    return run_cmd


def verify_image_folder_pattern(folder_path):
    false_response = True  # temporarily set to true to prevent stopping training in case of false positive
    true_response = True

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        log.error(
            f"The provided path '{folder_path}' is not a valid folder. Please follow the folder structure documentation found at docs\image_folder_structure.md ..."
        )
        return false_response

    # Create a regular expression pattern to match the required sub-folder names
    # The pattern should start with one or more digits (\d+) followed by an underscore (_)
    # After the underscore, it should match one or more word characters (\w+), which can be letters, numbers, or underscores
    # Example of a valid pattern matching name: 123_example_folder
    pattern = r"^\d+_\w+"

    # Get the list of sub-folders in the directory
    subfolders = [
        os.path.join(folder_path, subfolder)
        for subfolder in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, subfolder))
    ]

    # Check the pattern of each sub-folder
    matching_subfolders = [
        subfolder
        for subfolder in subfolders
        if re.match(pattern, os.path.basename(subfolder))
    ]

    # Print non-matching sub-folders
    non_matching_subfolders = set(subfolders) - set(matching_subfolders)
    if non_matching_subfolders:
        log.error(
            f"The following folders do not match the required pattern <number>_<text>: {', '.join(non_matching_subfolders)}"
        )
        log.error(
            f"Please follow the folder structure documentation found at docs\image_folder_structure.md ..."
        )
        return false_response

    # Check if no sub-folders exist
    if not matching_subfolders:
        log.error(
            f"No image folders found in {folder_path}. Please follow the folder structure documentation found at docs\image_folder_structure.md ..."
        )
        return false_response

    log.info(f"Valid image folder names found in: {folder_path}")
    return true_response


def SaveConfigFile(
    parameters,
    file_path: str,
    exclusion=["file_path", "save_as", "headless", "print_only"],
):
    # Return the values of the variables as a dictionary
    variables = {
        name: value
        for name, value in sorted(parameters, key=lambda x: x[0])
        if name not in exclusion
    }

    # Save the data to the selected file
    with open(file_path, "w") as file:
        json.dump(variables, file, indent=2)


def save_to_file(content):
    logs_directory = "logs"
    file_path = os.path.join(logs_directory, "print_command.txt")

    try:
        # Create the 'logs' directory if it does not exist
        if not os.path.exists(logs_directory):
            os.makedirs(logs_directory)

        with open(file_path, "a") as file:
            file.write(content + "\n")
    except IOError as e:
        print(f"Error: Could not write to file - {e}")
    except OSError as e:
        print(f"Error: Could not create 'logs' directory - {e}")


def check_duplicate_filenames(
    folder_path, image_extension=[".gif", ".png", ".jpg", ".jpeg", ".webp"]
):
    log.info("Checking for duplicate image filenames in training data directory...")
    for root, dirs, files in os.walk(folder_path):
        filenames = {}
        for file in files:
            filename, extension = os.path.splitext(file)
            if extension.lower() in image_extension:
                full_path = os.path.join(root, file)
                if filename in filenames:
                    existing_path = filenames[filename]
                    if existing_path != full_path:
                        print(
                            f"Warning: Same filename '{filename}' with different image extension found. This will cause training issues. Rename one of the file."
                        )
                        print(f"Existing file: {existing_path}")
                        print(f"Current file: {full_path}")
                else:
                    filenames[filename] = full_path


def is_file_writable(file_path):
    if not os.path.exists(file_path):
        # print(f"File '{file_path}' does not exist.")
        return True

    try:
        log.warning(f"File '{file_path}' already exist... it will be overwritten...")
        # Check if the file can be opened in write mode (which implies it's not open by another process)
        with open(file_path, "a"):
            pass
        return True
    except IOError:
        log.warning(f"File '{file_path}' can't be written to...")
        return False
