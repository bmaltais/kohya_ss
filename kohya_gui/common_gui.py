try:
    from tkinter import filedialog, Tk
except ImportError:
    pass
from easygui import msgbox, ynbox
from typing import Optional
from .custom_logging import setup_logging
from .sd_modeltype import SDModelType

import os
import re
import gradio as gr
import sys
import shlex
import json
import math
import shutil
import toml

# Set up logging
log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

scriptdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

if os.name == "nt":
    scriptdir = scriptdir.replace("\\", "/")

# insert sd-scripts path into PYTHONPATH
sys.path.insert(0, os.path.join(scriptdir, "sd-scripts"))

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


def get_executable_path(executable_name: str = None) -> str:
    """
    Retrieve and sanitize the path to an executable in the system's PATH.

    Args:
    executable_name (str): The name of the executable to find.

    Returns:
    str: The full, sanitized path to the executable if found, otherwise an empty string.
    """
    if executable_name:
        executable_path = shutil.which(executable_name)
        if executable_path:
            # Replace backslashes with forward slashes on Windows
            # if os.name == "nt":
            #     executable_path = executable_path.replace("\\", "/")
            return executable_path
        else:
            return ""  # Return empty string if the executable is not found
    else:
        return ""  # Return empty string if no executable name is provided


def calculate_max_train_steps(
    total_steps: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    epoch: int,
    reg_factor: int,
):
    return int(
        math.ceil(
            float(total_steps)
            / int(train_batch_size)
            / int(gradient_accumulation_steps)
            * int(epoch)
            * int(reg_factor)
        )
    )


def check_if_model_exist(
    output_name: str, output_dir: str, save_model_as: str, headless: bool = False
) -> bool:
    """
    Checks if a model with the same name already exists and prompts the user to overwrite it if it does.

    Parameters:
    output_name (str): The name of the output model.
    output_dir (str): The directory where the model is saved.
    save_model_as (str): The format to save the model as.
    headless (bool, optional): If True, skips the verification and returns False. Defaults to False.

    Returns:
    bool: True if the model already exists and the user chooses not to overwrite it, otherwise False.
    """
    if headless:
        log.info(
            "Headless mode, skipping verification if model already exist... if model already exist it will be overwritten..."
        )
        return False

    if save_model_as in ["diffusers", "diffusers_safetendors"]:
        ckpt_folder = os.path.join(output_dir, output_name)
        if os.path.isdir(ckpt_folder):
            msg = f"A diffuser model with the same name {ckpt_folder} already exists. Do you want to overwrite it?"
            if not ynbox(msg, "Overwrite Existing Model?"):
                log.info("Aborting training due to existing model with same name...")
                return True
    elif save_model_as in ["ckpt", "safetensors"]:
        ckpt_file = os.path.join(output_dir, output_name + "." + save_model_as)
        if os.path.isfile(ckpt_file):
            msg = f"A model with the same file name {ckpt_file} already exists. Do you want to overwrite it?"
            if not ynbox(msg, "Overwrite Existing Model?"):
                log.info("Aborting training due to existing model with same name...")
                return True
    else:
        log.info(
            'Can\'t verify if existing model exist when save model is set as "same as source model", continuing to train model...'
        )
        return False

    return False


def output_message(msg: str = "", title: str = "", headless: bool = False) -> None:
    """
    Outputs a message to the user, either in a message box or in the log.

    Parameters:
    msg (str, optional): The message to be displayed. Defaults to an empty string.
    title (str, optional): The title of the message box. Defaults to an empty string.
    headless (bool, optional): If True, the message is logged instead of displayed in a message box. Defaults to False.

    Returns:
    None
    """
    if headless:
        log.info(msg)
    else:
        msgbox(msg=msg, title=title)


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    """
    Creates a refresh button that can be used to update UI components.

    Parameters:
    refresh_component (list or object): The UI component(s) to be refreshed.
    refresh_method (callable): The method to be called when the button is clicked.
    refreshed_args (dict or callable): The arguments to be passed to the refresh method.
    elem_id (str): The ID of the button element.

    Returns:
    gr.Button: The configured refresh button.
    """
    # Converts refresh_component into a list for uniform processing. If it's already a list, keep it the same.
    refresh_components = (
        refresh_component
        if isinstance(refresh_component, list)
        else [refresh_component]
    )

    # Initialize label to None. This will store the label of the first component with a non-None label, if any.
    label = None
    # Iterate over each component to find the first non-None label and assign it to 'label'.
    for comp in refresh_components:
        label = getattr(comp, "label", None)
        if label is not None:
            break

    # Define the refresh function that will be triggered upon clicking the refresh button.
    def refresh():
        # Invoke the refresh_method, which is intended to perform the refresh operation.
        refresh_method()
        # Determine the arguments for the refresh: call refreshed_args if it's callable, otherwise use it directly.
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        # For each key-value pair in args, update the corresponding properties of each component.
        for k, v in args.items():
            for comp in refresh_components:
                setattr(comp, k, v)

        # Use gr.update to refresh the UI components. If multiple components are present, update each; else, update only the first.
        return (
            [gr.Dropdown(**(args or {})) for _ in refresh_components]
            if len(refresh_components) > 1
            else gr.Dropdown(**(args or {}))
        )

    # Create a refresh button with the specified label (via refresh_symbol), ID, and classes.
    # 'refresh_symbol' should be defined outside this function or passed as an argument, representing the button's label or icon.
    refresh_button = gr.Button(
        value=refresh_symbol, elem_id=elem_id, elem_classes=["tool"]
    )
    # Configure the button to invoke the refresh function.
    refresh_button.click(fn=refresh, inputs=[], outputs=refresh_components)
    # Return the configured refresh button to be used in the UI.
    return refresh_button


def list_dirs(path):
    if path is None or path == "None" or path == "":
        return

    if not os.path.exists(path):
        path = os.path.dirname(path)
        if not os.path.exists(path):
            return

    if not os.path.isdir(path):
        path = os.path.dirname(path)

    def natural_sort_key(s, regex=re.compile("([0-9]+)")):
        return [
            int(text) if text.isdigit() else text.lower() for text in regex.split(s)
        ]

    subdirs = [
        (item, os.path.join(path, item))
        for item in os.listdir(path)
        if os.path.isdir(os.path.join(path, item))
    ]
    subdirs = [
        filename
        for item, filename in subdirs
        if item[0] != "." and item not in ["__pycache__"]
    ]
    subdirs = sorted(subdirs, key=natural_sort_key)
    if os.path.dirname(path) != "":
        dirs = [os.path.dirname(path), path] + subdirs
    else:
        dirs = [path] + subdirs

    if os.sep == "\\":
        dirs = [d.replace("\\", "/") for d in dirs]
    for d in dirs:
        yield d


def list_files(path, exts=None, all=False):
    if path is None or path == "None" or path == "":
        return

    if not os.path.exists(path):
        path = os.path.dirname(path)
        if not os.path.exists(path):
            return

    if not os.path.isdir(path):
        path = os.path.dirname(path)

    files = [
        (item, os.path.join(path, item))
        for item in os.listdir(path)
        if all or os.path.isfile(os.path.join(path, item))
    ]
    files = [
        filename
        for item, filename in files
        if item[0] != "." and item not in ["__pycache__"]
    ]
    exts = set(exts) if exts is not None else None

    def natural_sort_key(s, regex=re.compile("([0-9]+)")):
        return [
            int(text) if text.isdigit() else text.lower() for text in regex.split(s)
        ]

    files = sorted(files, key=natural_sort_key)
    if os.path.dirname(path) != "":
        files = [os.path.dirname(path), path] + files
    else:
        files = [path] + files

    if os.sep == "\\":
        files = [d.replace("\\", "/") for d in files]

    for filename in files:
        if exts is not None:
            if os.path.isdir(filename):
                yield filename
            _, ext = os.path.splitext(filename)
            if ext.lower() not in exts:
                continue
            yield filename
        else:
            yield filename


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
    for key in [
        "clip_skip",
        "epoch",
        "gradient_accumulation_steps",
        "keep_tokens",
        "lr_warmup",
        "max_data_loader_n_workers",
        "max_train_epochs",
        "save_every_n_epochs",
        "seed",
    ]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = int(value)
            except ValueError:
                # Handle the case where the string is not a valid float
                my_data[key] = int(0)

    # Convert values to int if they are strings
    for key in ["lr_scheduler_num_cycles"]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = int(value)
            except ValueError:
                # Handle the case where the string is not a valid float
                my_data[key] = int(1)

    for key in [
        "max_train_steps",
        "caption_dropout_every_n_epochs"
    ]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = int(value)
            except ValueError:
                # Handle the case where the string is not a valid float
                my_data[key] = int(0)

    # Convert values to int if they are strings
    for key in ["max_token_length"]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = int(value)
            except ValueError:
                # Handle the case where the string is not a valid float
                my_data[key] = int(75)

    # Convert values to float if they are strings, correctly handling float representations
    for key in [
        "adaptive_noise_scale",
        "noise_offset",
        "learning_rate",
        "text_encoder_lr",
        "unet_lr",
    ]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = float(value)
            except ValueError:
                # Handle the case where the string is not a valid float
                my_data[key] = float(0.0)

    # Convert values to float if they are strings, correctly handling float representations
    for key in ["lr_scheduler_power"]:
        value = my_data.get(key)
        if value is not None:
            try:
                my_data[key] = float(value)
            except ValueError:
                # Handle the case where the string is not a valid float
                my_data[key] = float(1.0)

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

    # Convert use_wandb to log_with="wandb" if it is set to True
    for key in ["use_wandb"]:
        value = my_data.get(key)
        if value is not None:
            try:
                if value == "True":
                    my_data["log_with"] = "wandb"
            except ValueError:
                # Handle the case where the string is not a valid float
                pass

        my_data.pop(key, None)

    # Replace the lora_network_weights key with network_weights keeping the original value
    for key in ["lora_network_weights"]:
        value = my_data.get(key)  # Get original value
        if value is not None:  # Check if the key exists in the dictionary
            my_data["network_weights"] = value
            my_data.pop(key, None)

    return my_data


def get_dir_and_file(file_path):
    dir_path, file_name = os.path.split(file_path)
    return (dir_path, file_name)


def get_file_path(
    file_path="", default_extension=".json", extension_name="Config files"
):
    """
    Opens a file dialog to select a file, allowing the user to navigate and choose a file with a specific extension.
    If no file is selected, returns the initially provided file path or an empty string if not provided.
    This function is conditioned to skip the file dialog on macOS or if specific environment variables are present,
    indicating a possible automated environment where a dialog cannot be displayed.

    Parameters:
    - file_path (str): The initial file path or an empty string by default. Used as the fallback if no file is selected.
    - default_extension (str): The default file extension (e.g., ".json") for the file dialog.
    - extension_name (str): The display name for the type of files being selected (e.g., "Config files").

    Returns:
    - str: The path of the file selected by the user, or the initial `file_path` if no selection is made.

    Raises:
    - TypeError: If `file_path`, `default_extension`, or `extension_name` are not strings.

    Note:
    - The function checks the `ENV_EXCLUSION` list against environment variables to determine if the file dialog should be skipped, aiming to prevent its appearance during automated operations.
    - The dialog will also be skipped on macOS (`sys.platform != "darwin"`) as a specific behavior adjustment.
    """
    # Validate parameter types
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    if not isinstance(default_extension, str):
        raise TypeError("default_extension must be a string")
    if not isinstance(extension_name, str):
        raise TypeError("extension_name must be a string")

    # Environment and platform check to decide on showing the file dialog
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != "darwin":
        current_file_path = file_path  # Backup in case no file is selected

        initial_dir, initial_file = get_dir_and_file(
            file_path
        )  # Decompose file path for dialog setup

        # Initialize a hidden Tkinter window for the file dialog
        root = Tk()
        root.wm_attributes("-topmost", 1)  # Ensure the dialog is topmost
        root.withdraw()  # Hide the root window to show only the dialog

        # Open the file dialog and capture the selected file path
        file_path = filedialog.askopenfilename(
            filetypes=((extension_name, f"*{default_extension}"), ("All files", "*.*")),
            defaultextension=default_extension,
            initialfile=initial_file,
            initialdir=initial_dir,
        )

        root.destroy()  # Cleanup by destroying the Tkinter root window

        # Fallback to the initial path if no selection is made
        if not file_path:
            file_path = current_file_path

    # Return the selected or fallback file path
    return file_path


def get_any_file_path(file_path: str = "") -> str:
    """
    Opens a file dialog to select any file, allowing the user to navigate and choose a file.
    If no file is selected, returns the initially provided file path or an empty string if not provided.
    This function is conditioned to skip the file dialog on macOS or if specific environment variables are present,
    indicating a possible automated environment where a dialog cannot be displayed.

    Parameters:
    - file_path (str): The initial file path or an empty string by default. Used as the fallback if no file is selected.

    Returns:
    - str: The path of the file selected by the user, or the initial `file_path` if no selection is made.

    Raises:
    - TypeError: If `file_path` is not a string.
    - EnvironmentError: If there's an issue accessing environment variables.
    - RuntimeError: If there's an issue initializing the file dialog.

    Note:
    - The function checks the `ENV_EXCLUSION` list against environment variables to determine if the file dialog should be skipped, aiming to prevent its appearance during automated operations.
    - The dialog will also be skipped on macOS (`sys.platform != "darwin"`) as a specific behavior adjustment.
    """
    # Validate parameter type
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    try:
        # Check for environment variable conditions
        if (
            not any(var in os.environ for var in ENV_EXCLUSION)
            and sys.platform != "darwin"
        ):
            current_file_path: str = file_path

            initial_dir, initial_file = get_dir_and_file(file_path)

            # Initialize a hidden Tkinter window for the file dialog
            root = Tk()
            root.wm_attributes("-topmost", 1)
            root.withdraw()

            try:
                # Open the file dialog and capture the selected file path
                file_path = filedialog.askopenfilename(
                    initialdir=initial_dir,
                    initialfile=initial_file,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to open file dialog: {e}")
            finally:
                root.destroy()

            # Fallback to the initial path if no selection is made
            if not file_path:
                file_path = current_file_path
    except KeyError as e:
        raise EnvironmentError(f"Failed to access environment variables: {e}")

    # Return the selected or fallback file path
    return file_path


def get_folder_path(folder_path: str = "") -> str:
    """
    Opens a folder dialog to select a folder, allowing the user to navigate and choose a folder.
    If no folder is selected, returns the initially provided folder path or an empty string if not provided.
    This function is conditioned to skip the folder dialog on macOS or if specific environment variables are present,
    indicating a possible automated environment where a dialog cannot be displayed.

    Parameters:
    - folder_path (str): The initial folder path or an empty string by default. Used as the fallback if no folder is selected.

    Returns:
    - str: The path of the folder selected by the user, or the initial `folder_path` if no selection is made.

    Raises:
    - TypeError: If `folder_path` is not a string.
    - EnvironmentError: If there's an issue accessing environment variables.
    - RuntimeError: If there's an issue initializing the folder dialog.

    Note:
    - The function checks the `ENV_EXCLUSION` list against environment variables to determine if the folder dialog should be skipped, aiming to prevent its appearance during automated operations.
    - The dialog will also be skipped on macOS (`sys.platform != "darwin"`) as a specific behavior adjustment.
    """
    # Validate parameter type
    if not isinstance(folder_path, str):
        raise TypeError("folder_path must be a string")

    try:
        # Check for environment variable conditions
        if any(var in os.environ for var in ENV_EXCLUSION) or sys.platform == "darwin":
            return folder_path or ""

        root = Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        selected_folder = filedialog.askdirectory(initialdir=folder_path or ".")
        root.destroy()
        return selected_folder or folder_path
    except Exception as e:
        raise RuntimeError(f"Error initializing folder dialog: {e}") from e


def get_saveasfile_path(
    file_path: str = "",
    defaultextension: str = ".json",
    extension_name: str = "Config files",
) -> str:
    # Check if the current environment is not macOS and if the environment variables do not match the exclusion list
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != "darwin":
        # Store the initial file path to use as a fallback in case no file is selected
        current_file_path = file_path

        # Logging the current file path for debugging purposes; helps in tracking the flow of file selection
        # log.info(f'current file path: {current_file_path}')

        # Split the file path into directory and file name for setting the file dialog start location and filename
        initial_dir, initial_file = get_dir_and_file(file_path)

        # Initialize a hidden Tkinter window to act as the parent for the file dialog, ensuring it appears on top
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
        # Close the Tkinter root window to clean up the UI
        root.destroy()

        # Logging the save file path for auditing purposes; useful in confirming the user's file choice
        # log.info(save_file_path)

        # Default to the current file path if no file is selected, ensuring there's always a valid file path
        if save_file_path == None:
            file_path = current_file_path
        else:
            # Log the selected file name for transparency and tracking user actions
            # log.info(save_file_path.name)

            # Update the file path with the user-selected file name, facilitating the save operation
            file_path = save_file_path.name

        # Log the final file path for verification, ensuring the intended file is being used
        # log.info(file_path)

    # Return the final file path, either the user-selected file or the fallback path
    return file_path


def get_saveasfilename_path(
    file_path: str = "",
    extensions: str = "*",
    extension_name: str = "Config files",
) -> str:
    """
    Opens a file dialog to select a file name for saving, allowing the user to specify a file name and location.
    If no file is selected, returns the initially provided file path or an empty string if not provided.
    This function is conditioned to skip the file dialog on macOS or if specific environment variables are present,
    indicating a possible automated environment where a dialog cannot be displayed.

    Parameters:
    - file_path (str): The initial file path or an empty string by default. Used as the fallback if no file is selected.
    - extensions (str): The file extensions to filter the file dialog by. Defaults to "*" for all files.
    - extension_name (str): The name to display for the file extensions in the file dialog. Defaults to "Config files".

    Returns:
    - str: The path of the file selected by the user, or the initial `file_path` if no selection is made.

    Raises:
    - TypeError: If `file_path` is not a string.
    - EnvironmentError: If there's an issue accessing environment variables.
    - RuntimeError: If there's an issue initializing the file dialog.

    Note:
    - The function checks the `ENV_EXCLUSION` list against environment variables to determine if the file dialog should be skipped, aiming to prevent its appearance during automated operations.
    - The dialog will also be skipped on macOS (`sys.platform == "darwin"`) as a specific behavior adjustment.
    """
    # Check if the current environment is not macOS and if the environment variables do not match the exclusion list
    if not any(var in os.environ for var in ENV_EXCLUSION) and sys.platform != "darwin":
        # Store the initial file path to use as a fallback in case no file is selected
        current_file_path: str = file_path
        # log.info(f'current file path: {current_file_path}')

        # Split the file path into directory and file name for setting the file dialog start location and filename
        initial_dir, initial_file = get_dir_and_file(file_path)

        # Initialize a hidden Tkinter window to act as the parent for the file dialog, ensuring it appears on top
        root = Tk()
        root.wm_attributes("-topmost", 1)
        root.withdraw()
        # Open the file dialog and capture the selected file path
        save_file_path = filedialog.asksaveasfilename(
            filetypes=(
                (f"{extension_name}", f"{extensions}"),
                ("All files", "*"),
            ),
            defaultextension=extensions,
            initialdir=initial_dir,
            initialfile=initial_file,
        )
        # Close the Tkinter root window to clean up the UI
        root.destroy()

        # Default to the current file path if no file is selected, ensuring there's always a valid file path
        if save_file_path == "":
            file_path = current_file_path
        else:
            # Logging the save file path for auditing purposes; useful in confirming the user's file choice
            # log.info(save_file_path)
            # Update the file path with the user-selected file name, facilitating the save operation
            file_path = save_file_path

    # Return the final file path, either the user-selected file or the fallback path
    return file_path


def add_pre_postfix(
    folder: str = "",
    prefix: str = "",
    postfix: str = "",
    caption_file_ext: str = ".caption",
    recursive: bool = False,
) -> None:
    """
    Add prefix and/or postfix to the content of caption files within a folder.
    If no caption files are found, create one with the requested prefix and/or postfix.

    Args:
        folder (str): Path to the folder containing caption files.
        prefix (str, optional): Prefix to add to the content of the caption files.
        postfix (str, optional): Postfix to add to the content of the caption files.
        caption_file_ext (str, optional): Extension of the caption files.
        recursive (bool, optional): Whether to search for caption files recursively.
    """
    # If neither prefix nor postfix is provided, return early
    if prefix == "" and postfix == "":
        return

    # Define the image file extensions to filter
    image_extensions = (".jpg", ".jpeg", ".png", ".webp")

    # If recursive is true, list all image files in the folder and its subfolders
    if recursive:
        image_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
    else:
        # List all image files in the folder
        image_files = [
            f for f in os.listdir(folder) if f.lower().endswith(image_extensions)
        ]

    # Iterate over the list of image files
    for image_file in image_files:
        # Construct the caption file name by appending the caption file extension to the image file name
        caption_file_name = f"{os.path.splitext(image_file)[0]}{caption_file_ext}"
        # Construct the full path to the caption file
        caption_file_path = os.path.join(folder, caption_file_name)

        # Check if the caption file does not exist
        if not os.path.exists(caption_file_path):
            # Create a new caption file with the specified prefix and/or postfix
            try:
                with open(caption_file_path, "w", encoding="utf-8") as f:
                    # Determine the separator based on whether both prefix and postfix are provided
                    separator = " " if prefix and postfix else ""
                    f.write(f"{prefix}{separator}{postfix}")
            except Exception as e:
                log.error(f"Error writing to file {caption_file_path}: {e}")
        else:
            # Open the existing caption file for reading and writing
            try:
                with open(caption_file_path, "r+", encoding="utf-8") as f:
                    # Read the content of the caption file, stripping any trailing whitespace
                    content = f.read().rstrip()
                    # Move the file pointer to the beginning of the file
                    f.seek(0, 0)

                    # Determine the separator based on whether only prefix is provided
                    prefix_separator = " " if prefix else ""
                    # Determine the separator based on whether only postfix is provided
                    postfix_separator = " " if postfix else ""
                    # Write the updated content to the caption file, adding prefix and/or postfix
                    f.write(
                        f"{prefix}{prefix_separator}{content}{postfix_separator}{postfix}"
                    )
            except Exception as e:
                log.error(f"Error writing to file {caption_file_path}: {e}")


def has_ext_files(folder_path: str, file_extension: str) -> bool:
    """
    Determines whether any files within a specified folder have a given file extension.

    This function iterates through each file in the specified folder and checks if
    its extension matches the provided file_extension argument. The search is case-sensitive
    and expects file_extension to include the dot ('.') if applicable (e.g., '.txt').

    Args:
        folder_path (str): The absolute or relative path to the folder to search within.
        file_extension (str): The file extension to search for, including the dot ('.') if applicable.

    Returns:
        bool: True if at least one file with the specified extension is found, False otherwise.
    """
    # Iterate directly over files in the specified folder path
    for file in os.listdir(folder_path):
        # Return True at the first occurrence of a file with the specified extension
        if file.endswith(file_extension):
            return True

    # If no file with the specified extension is found, return False
    return False


def find_replace(
    folder_path: str = "",
    caption_file_ext: str = ".caption",
    search_text: str = "",
    replace_text: str = "",
) -> None:
    """
    Efficiently finds and replaces specified text across all caption files in a given folder.

    This function iterates through each caption file matching the specified extension within the given folder path, replacing all occurrences of the search text with the replacement text. It ensures that the operation only proceeds if the search text is provided and there are caption files to process.

    Args:
        folder_path (str, optional): The directory path where caption files are located. Defaults to an empty string, which implies the current directory.
        caption_file_ext (str, optional): The file extension for caption files. Defaults to ".caption".
        search_text (str, optional): The text to search for within the caption files. Defaults to an empty string.
        replace_text (str, optional): The text to use as a replacement. Defaults to an empty string.
    """
    # Log the start of the caption find/replace operation
    log.info("Running caption find/replace")

    # Validate the presence of caption files and the search text
    if not search_text or not has_ext_files(folder_path, caption_file_ext):
        # Display a message box indicating no files were found
        msgbox(
            f"No files with extension {caption_file_ext} were found in {folder_path}..."
        )
        log.warning(
            "No files with extension {caption_file_ext} were found in {folder_path}..."
        )
        # Exit the function early
        return

    # Check if the caption file extension is one of the supported extensions
    if caption_file_ext not in [".caption", ".txt", ".txt2", ".cap"]:
        log.error(
            f"Unsupported file extension {caption_file_ext} for caption files. Please use .caption, .txt, .txt2, or .cap."
        )
        # Exit the function early
        return

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        log.error(f"The provided path '{folder_path}' is not a valid folder.")
        return

    # List all caption files in the folder
    try:
        caption_files = [
            f for f in os.listdir(folder_path) if f.endswith(caption_file_ext)
        ]
    except Exception as e:
        log.error(f"Error accessing folder {folder_path}: {e}")
        return

    # Iterate over the list of caption files
    for caption_file in caption_files:
        # Construct the full path for each caption file
        file_path = os.path.join(folder_path, caption_file)
        # Read and replace text
        try:
            with open(file_path, "r", errors="ignore", encoding="utf-8") as f:
                content = f.read().replace(search_text, replace_text)

            # Write the updated content back to the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            log.error(f"Error processing file {file_path}: {e}")


def color_aug_changed(color_aug):
    """
    Handles the change in color augmentation checkbox.

    This function is called when the color augmentation checkbox is toggled.
    If color augmentation is enabled, it disables the cache latent checkbox
    and returns a new checkbox with the value set to False and interactive set to False.
    If color augmentation is disabled, it returns a new checkbox with interactive set to True.

    Args:
        color_aug (bool): The new state of the color augmentation checkbox.

    Returns:
        gr.Checkbox: A new checkbox with the appropriate settings based on the color augmentation state.
    """
    # If color augmentation is enabled, disable cache latent and return a new checkbox
    if color_aug:
        msgbox(
            'Disabling "Cache latent" because "Color augmentation" has been selected...'
        )
        return gr.Checkbox(value=False, interactive=False)
    # If color augmentation is disabled, return a new checkbox with interactive set to True
    else:
        return gr.Checkbox(interactive=True)


def set_pretrained_model_name_or_path_input(
    pretrained_model_name_or_path, refresh_method=None
):
    """
    Sets the pretrained model name or path input based on the model type.

    This function checks the type of the pretrained model and sets the appropriate
    parameters for the model. It also handles the case where the model list is
    set to 'custom' and a refresh method is provided.

    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model.
        refresh_method (callable, optional): A function to refresh the model list.

    Returns:
        tuple: A tuple containing the Dropdown widget, v2 checkbox, v_parameterization checkbox,
               and sdxl checkbox.
    """
    # Check if the given pretrained_model_name_or_path is in the list of SDXL models
    if pretrained_model_name_or_path in SDXL_MODELS:
        log.info("SDXL model selected. Setting sdxl parameters")
        v2 = gr.Checkbox(value=False, visible=False)
        v_parameterization = gr.Checkbox(value=False, visible=False)
        sdxl = gr.Checkbox(value=True, visible=False)
        sd3 = gr.Checkbox(value=False, visible=False)
        flux1 = gr.Checkbox(value=False, visible=False)
        return (
            gr.Dropdown(),
            v2,
            v_parameterization,
            sdxl,
            sd3,
            flux1,
        )

    # Check if the given pretrained_model_name_or_path is in the list of V2 base models
    if pretrained_model_name_or_path in V2_BASE_MODELS:
        log.info("SD v2 base model selected. Setting --v2 parameter")
        v2 = gr.Checkbox(value=True, visible=False)
        v_parameterization = gr.Checkbox(value=False, visible=False)
        sdxl = gr.Checkbox(value=False, visible=False)
        sd3 = gr.Checkbox(value=False, visible=False)
        flux1 = gr.Checkbox(value=False, visible=False)
        return (
            gr.Dropdown(),
            v2,
            v_parameterization,
            sdxl,
            sd3,
            flux1,
        )

    # Check if the given pretrained_model_name_or_path is in the list of V parameterization models
    if pretrained_model_name_or_path in V_PARAMETERIZATION_MODELS:
        log.info(
            "SD v2 model selected. Setting --v2 and --v_parameterization parameters"
        )
        v2 = gr.Checkbox(value=True, visible=False)
        v_parameterization = gr.Checkbox(value=True, visible=False)
        sdxl = gr.Checkbox(value=False, visible=False)
        sd3 = gr.Checkbox(value=False, visible=False)
        flux1 = gr.Checkbox(value=False, visible=False)
        return (
            gr.Dropdown(),
            v2,
            v_parameterization,
            sdxl,
            sd3,
            flux1,
        )

    # Check if the given pretrained_model_name_or_path is in the list of V1 models
    if pretrained_model_name_or_path in V1_MODELS:
        log.info(f"{pretrained_model_name_or_path} model selected.")
        v2 = gr.Checkbox(value=False, visible=False)
        v_parameterization = gr.Checkbox(value=False, visible=False)
        sdxl = gr.Checkbox(value=False, visible=False)
        sd3 = gr.Checkbox(value=False, visible=False)
        flux1 = gr.Checkbox(value=False, visible=False)
        return (
            gr.Dropdown(),
            v2,
            v_parameterization,
            sdxl,
            sd3,
            flux1,
        )

    # Check if the model_list is set to 'custom'
    v2 = gr.Checkbox(visible=True)
    v_parameterization = gr.Checkbox(visible=True)
    sdxl = gr.Checkbox(visible=True)
    sd3 = gr.Checkbox(visible=True)
    flux1 = gr.Checkbox(visible=True)

    # Auto-detect model type if safetensors file path is given
    if pretrained_model_name_or_path.lower().endswith(".safetensors"):
        detect = SDModelType(pretrained_model_name_or_path)
        v2 = gr.Checkbox(value=detect.Is_SD2(), visible=True)
        sdxl = gr.Checkbox(value=detect.Is_SDXL(), visible=True)
        sd3 = gr.Checkbox(value=detect.Is_SD3(), visible=True)
        flux1 = gr.Checkbox(value=detect.Is_FLUX1(), visible=True)
        #TODO: v_parameterization

    # If a refresh method is provided, use it to update the choices for the Dropdown widget
    if refresh_method is not None:
        args = dict(
            choices=refresh_method(pretrained_model_name_or_path),
        )
    else:
        args = {}
    return (
        gr.Dropdown(**args),
        v2,
        v_parameterization,
        sdxl,
        sd3,
        flux1,
    )


###
### Gradio common GUI section
###


def get_int_or_default(kwargs, key, default_value=0):
    """
    Retrieves an integer value from the provided kwargs dictionary based on the given key. If the key is not found,
    or the value cannot be converted to an integer, a default value is returned.

    Args:
        kwargs (dict): A dictionary of keyword arguments.
        key (str): The key to retrieve from the kwargs dictionary.
        default_value (int, optional): The default value to return if the key is not found or the value is not an integer.

    Returns:
        int: The integer value if found and valid, otherwise the default value.
    """
    # Try to retrieve the value for the specified key from the kwargs.
    # Use the provided default_value if the key does not exist.
    value = kwargs.get(key, default_value)
    try:
        # Try to convert the value to a integer. This should works for int,
        # and strings that represent a valid floating-point number.
        return int(value)
    except (ValueError, TypeError):
        # If the conversion fails (for example, the value is a string that cannot
        # be converted to an integer), log the issue and return the provided default_value.
        log.info(
            f"{key} is not an int or cannot be converted to int, setting value to {default_value}"
        )
        return default_value


def get_float_or_default(kwargs, key, default_value=0.0):
    """
    Retrieves a float value from the provided kwargs dictionary based on the given key. If the key is not found,
    or the value cannot be converted to a float, a default value is returned.

    This function attempts to convert the value to a float, which works for integers, floats, and strings that
    represent valid floating-point numbers. If the conversion fails, the issue is logged, and the provided
    default_value is returned.

    Args:
        kwargs (dict): A dictionary of keyword arguments.
        key (str): The key to retrieve from the kwargs dictionary.
        default_value (float, optional): The default value to return if the key is not found or the value is not a float.

    Returns:
        float: The float value if found and valid, otherwise the default value.
    """
    # Try to retrieve the value for the specified key from the kwargs.
    # Use the provided default_value if the key does not exist.
    value = kwargs.get(key, default_value)

    try:
        # Try to convert the value to a float. This should works for int, float,
        # and strings that represent a valid floating-point number.
        return float(value)
    except ValueError:
        # If the conversion fails (for example, the value is a string that cannot
        # be converted to a float), log the issue and return the provided default_value.
        log.info(
            f"{key} is not an int, float or a valid string for conversion, setting value to {default_value}"
        )
        return default_value


def get_str_or_default(kwargs, key, default_value=""):
    """
    Retrieves a string value from the provided kwargs dictionary based on the given key. If the key is not found,
    or the value is not a string, a default value is returned.

    Args:
        kwargs (dict): A dictionary of keyword arguments.
        key (str): The key to retrieve from the kwargs dictionary.
        default_value (str, optional): The default value to return if the key is not found or the value is not a string.

    Returns:
        str: The string value if found and valid, otherwise the default value.
    """
    # Try to retrieve the value for the specified key from the kwargs.
    # Use the provided default_value if the key does not exist.
    value = kwargs.get(key, default_value)

    # Check if the retrieved value is already a string.
    if isinstance(value, str):
        return value
    else:
        # If the value is not a string (e.g., int, float, or any other type),
        # convert it to a string and return the converted value.
        return str(value)


def run_cmd_advanced_training(run_cmd: list = [], **kwargs):
    """
    This function, run_cmd_advanced_training, dynamically constructs a command line string for advanced training
    configurations based on provided keyword arguments (kwargs). Each argument represents a different training parameter
    or flag that can be used to customize the training process. The function checks for the presence and validity of
    arguments, appending them to the command line string with appropriate formatting.

    Purpose
        The primary purpose of this function is to enable flexible and customizable training configurations for machine
        learning models. It allows users to specify a wide range of parameters and flags that control various aspects of
        the training process, such as learning rates, batch sizes, augmentation options, precision settings, and many more.

    Args:
        kwargs (dict): A variable number of keyword arguments that represent different training parameters or flags.
                       Each argument has a specific expected data type and format, which the function checks before
                       appending to the command line string.

    Returns:
        str: A command line string constructed based on the provided keyword arguments. This string includes the base
             command and additional parameters and flags tailored to the user's specifications for the training process
    """
    if "additional_parameters" in kwargs and kwargs["additional_parameters"] != "":
        additional_parameters = kwargs["additional_parameters"].replace('"', "")
        for arg in additional_parameters.split():
            run_cmd.append(shlex.quote(arg))

    if "max_data_loader_n_workers" in kwargs:
        max_data_loader_n_workers = kwargs.get("max_data_loader_n_workers")
        if max_data_loader_n_workers != "":
            run_cmd.append("--max_data_loader_n_workers")
            run_cmd.append(str(max_data_loader_n_workers))

    return run_cmd


def verify_image_folder_pattern(folder_path: str) -> bool:
    """
    Verify the image folder pattern in the given folder path.

    Args:
        folder_path (str): The path to the folder containing image folders.

    Returns:
        bool: True if the image folder pattern is valid, False otherwise.
    """
    # Initialize the return value to True
    return_value = True

    # Log the start of the verification process
    log.info(f"Verifying image folder pattern of {folder_path}...")

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        # Log an error message if the folder does not exist
        log.error(
            f"...the provided path '{folder_path}' is not a valid folder. "
            "Please follow the folder structure documentation found at docs\image_folder_structure.md ..."
        )
        # Return False to indicate that the folder pattern is not valid
        return False

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
        # Log an error message if any sub-folders do not match the pattern
        log.error(
            f"...the following folders do not match the required pattern <number>_<text>: {', '.join(non_matching_subfolders)}"
        )
        # Log an error message suggesting to follow the folder structure documentation
        log.error(
            f"...please follow the folder structure documentation found at docs\image_folder_structure.md ..."
        )
        # Return False to indicate that the folder pattern is not valid
        return False

    # Check if no sub-folders exist
    if not matching_subfolders:
        # Log an error message if no image folders are found
        log.error(
            f"...no image folders found in {folder_path}. "
            "Please follow the folder structure documentation found at docs\image_folder_structure.md ..."
        )
        # Return False to indicate that the folder pattern is not valid
        return False

    # Log the successful verification
    log.info(f"...valid")
    # Return True to indicate that the folder pattern is valid
    return return_value


def SaveConfigFile(
    parameters,
    file_path: str,
    exclusion: list = ["file_path", "save_as", "headless", "print_only"],
) -> None:
    """
    Saves the configuration parameters to a JSON file, excluding specified keys.

    This function iterates over a dictionary of parameters, filters out keys listed
    in the `exclusion` list, and saves the remaining parameters to a JSON file
    specified by `file_path`.

    Args:
        parameters (dict): Dictionary containing the configuration parameters.
        file_path (str): Path to the file where the filtered parameters should be saved.
        exclusion (list): List of keys to exclude from saving. Defaults to ["file_path", "save_as", "headless", "print_only"].
    """
    # Return the values of the variables as a dictionary
    variables = {
        name: value
        for name, value in sorted(parameters, key=lambda x: x[0])
        if name not in exclusion
    }

    # Check if the folder path for the file_path is valid
    # Extrach folder path
    folder_path = os.path.dirname(file_path)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If not, create the folder
        os.makedirs(os.path.dirname(folder_path))
        log.info(f"Creating folder {folder_path} for the configuration file...")

    # Save the data to the specified JSON file
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(variables, file, indent=2)


def save_to_file(content):
    """
    Appends the given content to a file named 'print_command.txt' within a 'logs' directory.

    This function checks for the existence of a 'logs' directory and creates it if
    it doesn't exist. Then, it appends the provided content along with a newline character
    to the 'print_command.txt' file within this directory.

    Args:
        content (str): The content to be saved to the file.
    """
    logs_directory = "logs"
    file_path = os.path.join(logs_directory, "print_command.txt")

    # Ensure the 'logs' directory exists
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    # Append content to the specified file
    try:
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(content + "\n")
    except IOError as e:
        print(f"Error: Could not write to file - {e}")
    except OSError as e:
        print(f"Error: Could not create 'logs' directory - {e}")


def check_duplicate_filenames(
    folder_path: str,
    image_extension: list = [".gif", ".png", ".jpg", ".jpeg", ".webp"],
) -> None:
    """
    Checks for duplicate image filenames in a given folder path.

    This function walks through the directory structure of the given folder path,
    and logs a warning if it finds files with the same name but different image extensions.
    This can lead to issues during training if not handled properly.

    Args:
        folder_path (str): The path to the folder containing image files.
        image_extension (list, optional): List of image file extensions to consider.
            Defaults to [".gif", ".png", ".jpg", ".jpeg", ".webp"].
    """
    # Initialize a flag to track if duplicates are found
    duplicate = False

    # Log the start of the duplicate check
    log.info(
        f"Checking for duplicate image filenames in training data directory {folder_path}..."
    )

    # Walk through the directory structure
    for root, dirs, files in os.walk(folder_path):
        # Initialize a dictionary to store filenames and their paths
        filenames = {}

        # Process each file in the current directory
        for file in files:
            # Split the filename and extension
            filename, extension = os.path.splitext(file)

            # Check if the extension is in the list of image extensions
            if extension.lower() in image_extension:
                # Construct the full path to the file
                full_path = os.path.join(root, file)

                # Check if the filename is already in the dictionary
                if filename in filenames:
                    # If it is, compare the existing path with the current path
                    existing_path = filenames[filename]
                    if existing_path != full_path:
                        # Log a warning if the paths are different
                        log.warning(
                            f"...same filename '{filename}' with different image extension found. This will cause training issues. Rename one of the file."
                        )
                        log.warning(f"  Existing file: {existing_path}")
                        log.warning(f"  Current file: {full_path}")

                        # Set the duplicate flag to True
                        duplicate = True
                else:
                    # If not, add the filename and path to the dictionary
                    filenames[filename] = full_path

    # If no duplicates were found, log a message indicating validation
    if not duplicate:
        log.info("...valid")


def validate_file_path(file_path: str) -> bool:
    if file_path == "":
        return True
    msg = f"Validating {file_path} existence..."
    if not os.path.isfile(file_path):
        log.error(f"{msg} FAILED: does not exist")
        return False
    log.info(f"{msg} SUCCESS")
    return True


def validate_folder_path(
    folder_path: str,
    can_be_written_to: bool = False,
    create_if_not_exists: bool = False,
) -> bool:
    if folder_path == "":
        return True
    msg = f"Validating {folder_path} existence{' and writability' if can_be_written_to else ''}..."
    if not os.path.isdir(folder_path):
        if create_if_not_exists:
            os.makedirs(folder_path)
            log.info(f"{msg} SUCCESS")
            return True
        else:
            log.error(f"{msg} FAILED: does not exist")
            return False
    if can_be_written_to and not os.access(folder_path, os.W_OK):
        log.error(f"{msg} FAILED: is not writable.")
        return False
    log.info(f"{msg} SUCCESS")
    return True


def validate_toml_file(file_path: str) -> bool:
    if file_path == "":
        return True
    msg = f"Validating toml {file_path} existence and validity..."
    if not os.path.isfile(file_path):
        log.error(f"{msg} FAILED: does not exist")
        return False

    try:
        toml.load(file_path)
    except:
        log.error(f"{msg} FAILED: is not a valid toml file.")
        return False
    log.info(f"{msg} SUCCESS")
    return True


def validate_model_path(pretrained_model_name_or_path: str) -> bool:
    """
    Validates the pretrained model name or path against Hugging Face models or local paths.

    Args:
        pretrained_model_name_or_path (str): The pretrained model name or path to validate.

    Returns:
        bool: True if the path is a valid Hugging Face model or exists locally; False otherwise.
    """
    from .class_source_model import default_models

    msg = f"Validating {pretrained_model_name_or_path} existence..."

    # Check if it matches the Hugging Face model pattern
    if re.match(r"^[\w-]+\/[\w-]+$", pretrained_model_name_or_path):
        log.info(f"{msg} SKIPPING: huggingface.co model")
    elif pretrained_model_name_or_path in default_models:
        log.info(f"{msg} SUCCESS")
    else:
        # If not one of the default models, check if it's a valid local path
        if not validate_file_path(
            pretrained_model_name_or_path
        ) and not validate_folder_path(pretrained_model_name_or_path):
            log.info(f"{msg} FAILURE: not a valid file or folder")
            return False
    return True


def is_file_writable(file_path: str) -> bool:
    """
    Checks if a file is writable.

    Args:
        file_path (str): The path to the file to be checked.

    Returns:
        bool: True if the file is writable, False otherwise.
    """
    # If the file does not exist, it is considered writable
    if not os.path.exists(file_path):
        return True

    try:
        # Attempt to open the file in append mode to check if it can be written to
        with open(file_path, "a", encoding="utf-8"):
            pass
        # If the file can be opened, it is considered writable
        return True
    except IOError as e:
        # If an IOError occurs, the file cannot be written to
        log.info(f"Error: {e}. File '{file_path}' is not writable.")
        return False


def print_command_and_toml(run_cmd, tmpfilename):
    log.warning(
        "Here is the trainer command as a reference. It will not be executed:\n"
    )
    # Reconstruct the safe command string for display
    command_to_run = " ".join(run_cmd)

    print(command_to_run)
    print("")

    log.info(f"Showing toml config file: {tmpfilename}")
    print("")
    with open(tmpfilename, "r", encoding="utf-8") as toml_file:
        log.info(toml_file.read())
    log.info(f"end of toml config file: {tmpfilename}")

    save_to_file(command_to_run)


def validate_args_setting(input_string):
    # Regex pattern to handle multiple conditions:
    # - Empty string is valid
    # - Single or multiple key/value pairs with exactly one space between pairs
    # - No spaces around '=' and no spaces within keys or values
    pattern = r"^(\S+=\S+)( \S+=\S+)*$|^$"
    if re.match(pattern, input_string):
        return True
    else:
        log.info(f"'{input_string}' is not a valid settings string.")
        log.info(
            "A valid settings string must consist of one or more key/value pairs formatted as key=value, with no spaces around the equals sign or within the value. Multiple pairs should be separated by a space."
        )
        return False


def setup_environment():
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        rf"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"

    if os.name == "nt":
        env["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

    return env
