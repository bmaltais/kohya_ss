from tkinter import filedialog, Tk
from easygui import msgbox, ynbox
from typing import Optional
from .custom_logging import setup_logging
from .class_command_executor import CommandExecutor

import os
import re
import gradio as gr
import sys
import shlex
import json
import math
import shutil
import time

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
            if os.name == "nt":
                executable_path = executable_path.replace("\\", "/")
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
    for key in ["epoch", "save_every_n_epochs", "lr_warmup"]:
        value = my_data.get(key, 0)
        if isinstance(value, str) and value.strip().isdigit():
            my_data[key] = int(value)
        elif not value:
            my_data[key] = 0

    # Convert values to float if they are strings, correctly handling float representations
    for key in ["noise_offset", "learning_rate", "text_encoder_lr", "unet_lr"]:
        value = my_data.get(key, 0)
        if isinstance(value, str):
            try:
                my_data[key] = float(value)
            except ValueError:
                # Handle the case where the string is not a valid float
                my_data[key] = 0
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
    # Enforce that the provided extension is one of .caption, .cap, .txt
    if caption_file_ext not in (".caption", ".cap", ".txt"):
        log.error("Invalid caption file extension. Must be on of .caption, .cap, .txt")
        return

    # If neither prefix nor postfix is provided, return early
    if prefix == "" and postfix == "":
        return

    # Define the image file extensions to filter
    image_extensions = (".jpg", ".jpeg", ".png", ".webp")

    # List all image files in the folder
    image_files = [
        f for f in os.listdir(folder) if f.lower().endswith(image_extensions)
    ]

    # Iterate over the list of image files
    for image_file in image_files:
        # Construct the caption file name by appending the caption file extension to the image file name
        caption_file_name = os.path.splitext(image_file)[0] + caption_file_ext
        # Construct the full path to the caption file
        caption_file_path = os.path.join(folder, caption_file_name)

        # Check if the caption file does not exist
        if not os.path.exists(caption_file_path):
            # Create a new caption file with the specified prefix and/or postfix
            with open(caption_file_path, "w", encoding="utf8") as f:
                # Determine the separator based on whether both prefix and postfix are provided
                separator = " " if prefix and postfix else ""
                f.write(f"{prefix}{separator}{postfix}")
        else:
            # Open the existing caption file for reading and writing
            with open(caption_file_path, "r+", encoding="utf8") as f:
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
            with open(file_path, "r", errors="ignore") as f:
                content = f.read().replace(search_text, replace_text)

            # Write the updated content back to the file
            with open(file_path, "w") as f:
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
        return (
            gr.Dropdown(),
            v2,
            v_parameterization,
            sdxl,
        )

    # Check if the given pretrained_model_name_or_path is in the list of V2 base models
    if pretrained_model_name_or_path in V2_BASE_MODELS:
        log.info("SD v2 base model selected. Setting --v2 parameter")
        v2 = gr.Checkbox(value=True, visible=False)
        v_parameterization = gr.Checkbox(value=False, visible=False)
        sdxl = gr.Checkbox(value=False, visible=False)
        return (
            gr.Dropdown(),
            v2,
            v_parameterization,
            sdxl,
        )

    # Check if the given pretrained_model_name_or_path is in the list of V parameterization models
    if pretrained_model_name_or_path in V_PARAMETERIZATION_MODELS:
        log.info(
            "SD v2 model selected. Setting --v2 and --v_parameterization parameters"
        )
        v2 = gr.Checkbox(value=True, visible=False)
        v_parameterization = gr.Checkbox(value=True, visible=False)
        sdxl = gr.Checkbox(value=False, visible=False)
        return (
            gr.Dropdown(),
            v2,
            v_parameterization,
            sdxl,
        )

    # Check if the given pretrained_model_name_or_path is in the list of V1 models
    if pretrained_model_name_or_path in V1_MODELS:
        log.info(f"{pretrained_model_name_or_path} model selected.")
        v2 = gr.Checkbox(value=False, visible=False)
        v_parameterization = gr.Checkbox(value=False, visible=False)
        sdxl = gr.Checkbox(value=False, visible=False)
        return (
            gr.Dropdown(),
            v2,
            v_parameterization,
            sdxl,
        )

    # Check if the model_list is set to 'custom'
    v2 = gr.Checkbox(visible=True)
    v_parameterization = gr.Checkbox(visible=True)
    sdxl = gr.Checkbox(visible=True)

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

    if "block_lr" in kwargs and kwargs["block_lr"] != "":
        run_cmd.append("--block_lr")
        run_cmd.append(shlex.quote(kwargs["block_lr"]))

    if kwargs.get("bucket_no_upscale"):
        run_cmd.append("--bucket_no_upscale")

    if "bucket_reso_steps" in kwargs:
        run_cmd.append("--bucket_reso_steps")
        run_cmd.append(str(int(kwargs["bucket_reso_steps"])))

    if kwargs.get("cache_latents"):
        run_cmd.append("--cache_latents")

    if kwargs.get("cache_latents_to_disk"):
        run_cmd.append("--cache_latents_to_disk")

    if kwargs.get("cache_text_encoder_outputs"):
        run_cmd.append("--cache_text_encoder_outputs")

    if (
        "caption_dropout_every_n_epochs" in kwargs
        and int(kwargs["caption_dropout_every_n_epochs"]) > 0
    ):
        run_cmd.append("--caption_dropout_every_n_epochs")
        run_cmd.append(str(int(kwargs["caption_dropout_every_n_epochs"])))

        caption_dropout_rate = kwargs.get("caption_dropout_rate")
        if caption_dropout_rate and float(caption_dropout_rate) > 0:
            run_cmd.append("--caption_dropout_rate")
            run_cmd.append(str(caption_dropout_rate))

        caption_extension = kwargs.get("caption_extension")
        if caption_extension:
            run_cmd.append("--caption_extension")
            run_cmd.append(shlex.quote(caption_extension))

        clip_skip = kwargs.get("clip_skip")
        if clip_skip and int(clip_skip) > 1:
            run_cmd.append("--clip_skip")
            run_cmd.append(str(clip_skip))

        color_aug = kwargs.get("color_aug")
        if color_aug:
            run_cmd.append("--color_aug")

        dataset_config = kwargs.get("dataset_config")
        if dataset_config:
            dataset_config = os.path.abspath(os.path.normpath(dataset_config))
            if os.name == "nt":
                dataset_config = dataset_config.replace("\\", "/")
            run_cmd.append("--dataset_config")
            run_cmd.append(dataset_config)

    dataset_repeats = kwargs.get("dataset_repeats")
    if dataset_repeats:
        run_cmd.append("--dataset_repeats")
        run_cmd.append(shlex.quote(dataset_repeats))

    debiased_estimation_loss = kwargs.get("debiased_estimation_loss")
    if debiased_estimation_loss:
        run_cmd.append("--debiased_estimation_loss")

    dim_from_weights = kwargs.get("dim_from_weights")
    if dim_from_weights and kwargs.get(
        "lora_network_weights"
    ):  # Only if lora_network_weights is true
        run_cmd.append("--dim_from_weights")

    # Check if enable_bucket is true and both min_bucket_reso and max_bucket_reso are provided as part of the kwargs
    if (
        kwargs.get("enable_bucket")
        and "min_bucket_reso" in kwargs
        and "max_bucket_reso" in kwargs
    ):
        run_cmd.append("--enable_bucket")
        run_cmd.append("--min_bucket_reso")
        run_cmd.append(str(kwargs["min_bucket_reso"]))  # Ensure it is a string
        run_cmd.append("--max_bucket_reso")
        run_cmd.append(str(kwargs["max_bucket_reso"]))  # Ensure it is a string

    in_json = kwargs.get("in_json")
    if in_json:
        run_cmd.append("--in_json")
        run_cmd.append(shlex.quote(in_json))

    flip_aug = kwargs.get("flip_aug")
    if flip_aug:
        run_cmd.append("--flip_aug")

    fp8_base = kwargs.get("fp8_base")
    if fp8_base:
        run_cmd.append("--fp8_base")

    full_bf16 = kwargs.get("full_bf16")
    if full_bf16:
        run_cmd.append("--full_bf16")

    full_fp16 = kwargs.get("full_fp16")
    if full_fp16:
        run_cmd.append("--full_fp16")

    if (
        "gradient_accumulation_steps" in kwargs
        and int(kwargs["gradient_accumulation_steps"]) > 1
    ):
        run_cmd.append("--gradient_accumulation_steps")
        run_cmd.append(str(int(kwargs["gradient_accumulation_steps"])))

    if kwargs.get("gradient_checkpointing"):
        run_cmd.append("--gradient_checkpointing")

    if "huber_c" in kwargs:
        run_cmd.append("--huber_c")
        run_cmd.append(str(kwargs.get("huber_c")))

    if "huber_schedule" in kwargs:
        run_cmd.append("--huber_schedule")
        run_cmd.append(shlex.quote(kwargs.get("huber_schedule")))

    if kwargs.get("ip_noise_gamma"):
        if float(kwargs["ip_noise_gamma"]) > 0:
            run_cmd.append("--ip_noise_gamma")
            run_cmd.append(str(kwargs["ip_noise_gamma"]))

    if kwargs.get("ip_noise_gamma_random_strength"):
        if kwargs["ip_noise_gamma_random_strength"]:
            run_cmd.append("--ip_noise_gamma_random_strength")

    if "keep_tokens" in kwargs and int(kwargs["keep_tokens"]) > 0:
        run_cmd.append("--keep_tokens")
        run_cmd.append(str(int(kwargs["keep_tokens"])))

    if "learning_rate" in kwargs:
        run_cmd.append("--learning_rate")
        run_cmd.append(str(float(kwargs["learning_rate"])))

    if "learning_rate_te" in kwargs:
        if kwargs["learning_rate_te"] == 0:
            run_cmd.append("--learning_rate_te")
            run_cmd.append("0")
        else:
            run_cmd.append("--learning_rate_te")
            run_cmd.append(str(float(kwargs["learning_rate_te"])))

    if "learning_rate_te1" in kwargs:
        if kwargs["learning_rate_te1"] == 0:
            run_cmd.append("--learning_rate_te1")
            run_cmd.append("0")
        else:
            run_cmd.append("--learning_rate_te1")
            run_cmd.append(str(float(kwargs["learning_rate_te1"])))

    if "learning_rate_te2" in kwargs:
        if kwargs["learning_rate_te2"] == 0:
            run_cmd.append("--learning_rate_te2")
            run_cmd.append("0")
        else:
            run_cmd.append("--learning_rate_te2")
            run_cmd.append(str(float(kwargs["learning_rate_te2"])))

    if logging_dir := kwargs.get("logging_dir"):
        if logging_dir.startswith('"') and logging_dir.endswith('"'):
            logging_dir = logging_dir[1:-1]
        if os.path.exists(logging_dir):
            logging_dir = os.path.abspath(os.path.normpath(logging_dir))
            if os.name == "nt":
                logging_dir = logging_dir.replace("\\", "/")
            run_cmd.append("--logging_dir")
            run_cmd.append(logging_dir)

    log_tracker_name = kwargs.get("log_tracker_name")
    if log_tracker_name:
        run_cmd.append(rf"--log_tracker_name")
        run_cmd.append(f"{shlex.quote(log_tracker_name)}")

    log_tracker_config = kwargs.get("log_tracker_config")
    if log_tracker_config:
        if log_tracker_config.startswith('"') and log_tracker_config.endswith('"'):
            log_tracker_config = log_tracker_config[1:-1]
        if os.path.exists(log_tracker_config):
            log_tracker_config = os.path.abspath(os.path.normpath(log_tracker_config))
            if os.name == "nt":
                log_tracker_config = log_tracker_config.replace("\\", "/")
            run_cmd.append(f"--log_tracker_config")
            run_cmd.append(log_tracker_config)

    lora_network_weights = kwargs.get("lora_network_weights")
    if lora_network_weights:
        run_cmd.append(f"--network_weights")
        run_cmd.append(
            f"{shlex.quote(lora_network_weights)}"
        )  # Yes, the parameter is now called network_weights instead of lora_network_weights

    if "loss_type" in kwargs:
        run_cmd.append("--loss_type")
        run_cmd.append(shlex.quote(kwargs["loss_type"]))

    lr_scheduler = kwargs.get("lr_scheduler")
    if lr_scheduler:
        run_cmd.append("--lr_scheduler")
        run_cmd.append(shlex.quote(lr_scheduler))

    lr_scheduler_args = kwargs.get("lr_scheduler_args")
    if lr_scheduler_args and lr_scheduler_args != "":
        lr_scheduler_args = lr_scheduler_args.replace('"', "")
        args = lr_scheduler_args.split()
        run_cmd.append("--lr_scheduler_args")
        for arg in args:
            run_cmd.append(shlex.quote(arg))

    lr_scheduler_num_cycles = kwargs.get("lr_scheduler_num_cycles")
    if lr_scheduler_num_cycles and lr_scheduler_num_cycles != "":
        run_cmd.append("--lr_scheduler_num_cycles")
        run_cmd.append(str(lr_scheduler_num_cycles))
    else:
        epoch = kwargs.get("epoch")
        if epoch:
            run_cmd.append("--lr_scheduler_num_cycles")
            run_cmd.append(str(epoch))

    lr_scheduler_power = kwargs.get("lr_scheduler_power")
    if lr_scheduler_power and lr_scheduler_power != "":
        run_cmd.append("--lr_scheduler_power")
        run_cmd.append(str(lr_scheduler_power))

    lr_warmup_steps = kwargs.get("lr_warmup_steps")
    if lr_warmup_steps and lr_scheduler != "constant":
        run_cmd.append("--lr_warmup_steps")
        run_cmd.append(str(lr_warmup_steps))
    else:
        log.info("Can't use LR warmup with LR Scheduler constant... ignoring...")

    if "masked_loss" in kwargs and kwargs.get(
        "masked_loss"
    ):  # Test if the value is true as it could be false
        run_cmd.append("--masked_loss")

    if "max_data_loader_n_workers" in kwargs:
        max_data_loader_n_workers = kwargs.get("max_data_loader_n_workers")
        if max_data_loader_n_workers != "":
            run_cmd.append("--max_data_loader_n_workers")
            run_cmd.append(str(max_data_loader_n_workers))

    if "max_grad_norm" in kwargs and kwargs["max_grad_norm"] != "":
        run_cmd.append("--max_grad_norm")
        run_cmd.append(str(float(kwargs["max_grad_norm"])))

    if "max_resolution" in kwargs:
        run_cmd.append("--resolution")
        run_cmd.append(shlex.quote(kwargs["max_resolution"]))

    if "max_timestep" in kwargs and int(kwargs["max_timestep"]) < 1000:
        run_cmd.append("--max_timestep")
        run_cmd.append(str(int(kwargs["max_timestep"])))

    if "max_token_length" in kwargs and int(kwargs["max_token_length"]) > 75:
        run_cmd.append("--max_token_length")
        run_cmd.append(str(int(kwargs["max_token_length"])))

    if "max_train_epochs" in kwargs and kwargs["max_train_epochs"] != "":
        run_cmd.append("--max_train_epochs")
        run_cmd.append(str(int(kwargs["max_train_epochs"])))

    if "max_train_steps" in kwargs and kwargs["max_train_steps"] != "":
        run_cmd.append("--max_train_steps")
        run_cmd.append(str(int(kwargs["max_train_steps"])))

    if "mem_eff_attn" in kwargs and kwargs.get("mem_eff_attn"):
        run_cmd.append("--mem_eff_attn")

    if "min_snr_gamma" in kwargs and int(kwargs["min_snr_gamma"]) >= 1:
        run_cmd.append("--min_snr_gamma")
        run_cmd.append(str(int(kwargs["min_snr_gamma"])))

    if "min_timestep" in kwargs and int(kwargs["min_timestep"]) > -1:
        run_cmd.append("--min_timestep")
        run_cmd.append(str(int(kwargs["min_timestep"])))

    if "mixed_precision" in kwargs:
        run_cmd.append("--mixed_precision")
        run_cmd.append(shlex.quote(kwargs["mixed_precision"]))

    if "network_alpha" in kwargs:
        run_cmd.append("--network_alpha")
        run_cmd.append(str(int(kwargs["network_alpha"])))

    if "network_args" in kwargs and kwargs["network_args"]:
        network_args = kwargs["network_args"].replace('"', "")
        args = network_args.split()
        run_cmd.append("--network_args")
        run_cmd.extend(args)  # Adds all args to the command list directly

    if "network_dim" in kwargs:
        run_cmd.append("--network_dim")
        run_cmd.append(str(int(kwargs["network_dim"])))

    if "network_dropout" in kwargs and float(kwargs["network_dropout"]) > 0.0:
        run_cmd.append("--network_dropout")
        run_cmd.append(str(kwargs["network_dropout"]))

    if "network_module" in kwargs and kwargs["network_module"]:
        run_cmd.append("--network_module")
        run_cmd.append(shlex.quote(kwargs["network_module"]))

    if (
        "network_train_text_encoder_only" in kwargs
        and kwargs["network_train_text_encoder_only"]
    ):
        run_cmd.append("--network_train_text_encoder_only")

    if "network_train_unet_only" in kwargs and kwargs["network_train_unet_only"]:
        run_cmd.append("--network_train_unet_only")

    if "no_half_vae" in kwargs and kwargs["no_half_vae"]:
        run_cmd.append("--no_half_vae")

    if "no_token_padding" in kwargs and kwargs["no_token_padding"]:
        run_cmd.append("--no_token_padding")

    if "noise_offset_type" in kwargs:
        noise_offset_type = kwargs["noise_offset_type"]

        if noise_offset_type == "Original":
            if "noise_offset" in kwargs and float(kwargs.get("noise_offset", 0)):
                run_cmd.append("--noise_offset")
                run_cmd.append(str(float(kwargs["noise_offset"])))

            if (
                "adaptive_noise_scale" in kwargs
                and float(kwargs.get("adaptive_noise_scale", 0)) != 0
            ):
                adaptive_noise_scale = float(kwargs["adaptive_noise_scale"])
                if (
                    adaptive_noise_scale > 0
                    and float(kwargs.get("noise_offset", 0)) > 0
                ):
                    run_cmd.append("--adaptive_noise_scale")
                    run_cmd.append(str(adaptive_noise_scale))

            if "noise_offset_random_strength" in kwargs and kwargs.get(
                "noise_offset_random_strength"
            ):
                run_cmd.append("--noise_offset_random_strength")

        elif noise_offset_type == "Multires":
            if (
                "multires_noise_iterations" in kwargs
                and int(kwargs.get("multires_noise_iterations", 0)) > 0
            ):
                run_cmd.append("--multires_noise_iterations")
                run_cmd.append(str(int(kwargs["multires_noise_iterations"])))

            if (
                "multires_noise_discount" in kwargs
                and float(kwargs.get("multires_noise_discount", 0)) > 0
            ):
                run_cmd.append("--multires_noise_discount")
                run_cmd.append(str(float(kwargs["multires_noise_discount"])))

    if "optimizer_args" in kwargs and kwargs.get("optimizer_args"):
        run_cmd.append("--optimizer_args")
        optimizer_args = kwargs["optimizer_args"].replace('"', "")
        args = optimizer_args.split()
        for arg in args:
            run_cmd.append(shlex.quote(arg))

    if "optimizer" in kwargs:
        run_cmd.append("--optimizer_type")
        run_cmd.append(shlex.quote(kwargs["optimizer"]))

    if "output_dir" in kwargs and kwargs["output_dir"]:
        output_dir = kwargs["output_dir"].strip(
            '"'
        )  # Remove surrounding quotes if present
        if os.path.exists(output_dir):
            output_dir = os.path.abspath(os.path.normpath(output_dir))
            if os.name == "nt":
                output_dir = output_dir.replace("\\", "/")
            run_cmd.append("--output_dir")
            run_cmd.append(output_dir)

    if "output_name" in kwargs and kwargs["output_name"]:
        run_cmd.append("--output_name")
        run_cmd.append(shlex.quote(kwargs["output_name"]))

    if (
        "persistent_data_loader_workers" in kwargs
        and kwargs["persistent_data_loader_workers"]
    ):
        run_cmd.append("--persistent_data_loader_workers")

    if (
        "pretrained_model_name_or_path" in kwargs
        and kwargs["pretrained_model_name_or_path"]
    ):
        path = kwargs["pretrained_model_name_or_path"]
        if os.name == "nt":
            path = path.replace("\\", "/")
        run_cmd.append("--pretrained_model_name_or_path")
        run_cmd.append(path)

    if "prior_loss_weight" in kwargs and float(kwargs["prior_loss_weight"]) != 1.0:
        run_cmd.append("--prior_loss_weight")
        run_cmd.append(shlex.quote(str(float(kwargs["prior_loss_weight"]))))

    if "random_crop" in kwargs and kwargs["random_crop"]:
        run_cmd.append("--random_crop")

    if "reg_data_dir" in kwargs and kwargs["reg_data_dir"]:
        reg_data_dir = kwargs["reg_data_dir"].strip(
            '"'
        )  # Remove surrounding quotes if present
        if os.path.isdir(reg_data_dir):
            reg_data_dir = os.path.abspath(os.path.normpath(reg_data_dir))
            if os.name == "nt":
                reg_data_dir = reg_data_dir.replace("\\", "/")
            run_cmd.append("--reg_data_dir")
            run_cmd.append(reg_data_dir)

    if "resume" in kwargs and kwargs["resume"]:
        run_cmd.append("--resume")
        run_cmd.append(shlex.quote(kwargs["resume"]))

    if "save_every_n_epochs" in kwargs and int(kwargs["save_every_n_epochs"]) > 0:
        run_cmd.append("--save_every_n_epochs")
        run_cmd.append(str(int(kwargs["save_every_n_epochs"])))

    if "save_every_n_steps" in kwargs and int(kwargs["save_every_n_steps"]) > 0:
        run_cmd.append("--save_every_n_steps")
        run_cmd.append(str(int(kwargs["save_every_n_steps"])))

    if "save_last_n_steps" in kwargs and int(kwargs["save_last_n_steps"]) > 0:
        run_cmd.append("--save_last_n_steps")
        run_cmd.append(str(int(kwargs["save_last_n_steps"])))

    if (
        "save_last_n_steps_state" in kwargs
        and int(kwargs["save_last_n_steps_state"]) > 0
    ):
        run_cmd.append("--save_last_n_steps_state")
        run_cmd.append(str(int(kwargs["save_last_n_steps_state"])))

    if "save_model_as" in kwargs and kwargs["save_model_as"] != "same as source model":
        run_cmd.append("--save_model_as")
        run_cmd.append(kwargs["save_model_as"])

    if "save_precision" in kwargs:
        run_cmd.append("--save_precision")
        run_cmd.append(kwargs["save_precision"])

    if "save_state" in kwargs and kwargs["save_state"]:
        run_cmd.append("--save_state")

    if "save_state_on_train_end" in kwargs and kwargs["save_state_on_train_end"]:
        run_cmd.append("--save_state_on_train_end")

    if (
        "scale_v_pred_loss_like_noise_pred" in kwargs
        and kwargs["scale_v_pred_loss_like_noise_pred"]
    ):
        run_cmd.append("--scale_v_pred_loss_like_noise_pred")

    if "scale_weight_norms" in kwargs and kwargs["scale_weight_norms"] > 0.0:
        run_cmd.append("--scale_weight_norms")
        run_cmd.append(str(float(kwargs["scale_weight_norms"])))

    if "seed" in kwargs and kwargs["seed"] != "":
        run_cmd.append("--seed")
        run_cmd.append(str(int(kwargs["seed"])))

    if "shuffle_caption" in kwargs and kwargs["shuffle_caption"]:
        run_cmd.append("--shuffle_caption")

    if (
        "stop_text_encoder_training" in kwargs
        and kwargs["stop_text_encoder_training"] > 0
    ):
        run_cmd.append("--stop_text_encoder_training")
        run_cmd.append(str(int(kwargs["stop_text_encoder_training"])))

    # Text encoder learning rate
    if "text_encoder_lr" in kwargs and float(kwargs["text_encoder_lr"]) > 0:
        run_cmd.append("--text_encoder_lr")
        run_cmd.append(str(kwargs["text_encoder_lr"]))

    # Training batch size
    if "train_batch_size" in kwargs:
        run_cmd.append("--train_batch_size")
        run_cmd.append(str(int(kwargs["train_batch_size"])))

    # Training comment
    if training_comment := kwargs.get("training_comment"):
        run_cmd.append("--training_comment")
        run_cmd.append(shlex.quote(training_comment))

    # Train data directory
    if train_data_dir := kwargs.get("train_data_dir"):
        train_data_dir = train_data_dir.strip('"')
        if os.path.exists(train_data_dir):
            train_data_dir = os.path.abspath(os.path.normpath(train_data_dir))
            train_data_dir = (
                train_data_dir.replace("\\", "/") if os.name == "nt" else train_data_dir
            )
            run_cmd.append("--train_data_dir")
            run_cmd.append(train_data_dir)

    # Train text encoder
    if kwargs.get("train_text_encoder"):
        run_cmd.append("--train_text_encoder")

    # Unet learning rate
    if "unet_lr" in kwargs and float(kwargs["unet_lr"]) > 0:
        run_cmd.append("--unet_lr")
        run_cmd.append(str(kwargs["unet_lr"]))

    # Use Weights and Biases logging
    if kwargs.get("use_wandb"):
        run_cmd.append("--log_with")
        run_cmd.append("wandb")

    # V parameterization
    if kwargs.get("v_parameterization"):
        run_cmd.append("--v_parameterization")

    # V predicted likelihood loss
    if "v_pred_like_loss" in kwargs and float(kwargs["v_pred_like_loss"]) > 0:
        run_cmd.append("--v_pred_like_loss")
        run_cmd.append(str(float(kwargs["v_pred_like_loss"])))

    # Version 2
    if kwargs.get("v2"):
        run_cmd.append("--v2")

    # VAE path
    if vae := kwargs.get("vae"):
        vae = (
            vae
            if os.path.exists(vae)
            else os.path.join("models", "VAE", vae).replace(os.sep, "/")
        )
        if os.path.exists(vae):
            vae = (
                os.path.abspath(os.path.normpath(vae)).replace("\\", "/")
                if os.name == "nt"
                else vae
            )
            run_cmd.append("--vae")
            run_cmd.append(vae)

    # VAE batch size
    if "vae_batch_size" in kwargs and int(kwargs["vae_batch_size"]) > 0:
        run_cmd.append("--vae_batch_size")
        run_cmd.append(str(int(kwargs["vae_batch_size"])))

    # Weights and Biases API key
    if wandb_api_key := kwargs.get("wandb_api_key"):
        run_cmd.append("--wandb_api_key")
        run_cmd.append(shlex.quote(wandb_api_key))

    # Weights and Biases run name
    if wandb_run_name := kwargs.get("wandb_run_name"):
        run_cmd.append("--wandb_run_name")
        run_cmd.append(shlex.quote(wandb_run_name))

    # Weighted captions
    if kwargs.get("weighted_captions"):
        run_cmd.append("--weighted_captions")

    # Xformers
    if xformers := kwargs.get("xformers"):
        run_cmd.append("--xformers" if xformers == "xformers" else "--sdpa")

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
    with open(file_path, "w") as file:
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
        with open(file_path, "a") as file:
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


def validate_paths(headless: bool = False, **kwargs: Optional[str]) -> bool:
    """
    Validates the existence of specified paths and patterns for model training configurations.

    This function checks for the existence of various directory paths and files provided as keyword arguments,
    including model paths, data directories, output directories, and more. It leverages predefined default
    models for validation and ensures directory creation if necessary.

    Args:
        headless (bool): A flag indicating if the function should run without requiring user input.
        **kwargs (Optional[str]): Keyword arguments that represent various path configurations,
                                  including but not limited to `pretrained_model_name_or_path`, `train_data_dir`,
                                  and more.

    Returns:
        bool: True if all specified paths are valid or have been successfully created; False otherwise.
    """

    def validate_path(
        path: Optional[str], path_type: str, create_if_missing: bool = False
    ) -> bool:
        """
        Validates the existence of a path. If the path does not exist and `create_if_missing` is True,
        attempts to create the directory.

        Args:
            path (Optional[str]): The path to validate.
            path_type (str): Description of the path type for logging purposes.
            create_if_missing (bool): Whether to create the directory if it does not exist.

        Returns:
            bool: True if the path is valid or has been created; False otherwise.
        """
        if path:
            log.info(f"Validating {path_type} path {path} existence...")
            if os.path.exists(path):
                log.info("...valid")
            else:
                if create_if_missing:
                    try:
                        os.makedirs(path, exist_ok=True)
                        log.info(f"...created folder at {path}")
                        return True
                    except Exception as e:
                        log.error(f"...failed to create {path_type} folder: {e}")
                        return False
                else:
                    log.error(
                        f"...{path_type} path '{path}' is missing or does not exist"
                    )
                    return False
        else:
            log.info(f"{path_type} not specified, skipping validation")
        return True

    # Validates the model name or path against default models or existence as a local path
    if not validate_model_path(kwargs.get("pretrained_model_name_or_path")):
        return False

    # Validates the existence of specified directories or files, and creates them if necessary
    for key, value in kwargs.items():
        if key in ["output_dir", "logging_dir"]:
            if not validate_path(value, key, create_if_missing=True):
                return False
        elif key in ["vae"]:
            # Check if it matches the Hugging Face model pattern
            if re.match(r"^[\w-]+\/[\w-]+$", value):
                log.info("Checking vae... huggingface.co model, skipping validation")
            else:
                if not validate_path(value, key):
                    return False
        else:
            if key not in ["pretrained_model_name_or_path"]:
                if not validate_path(value, key):
                    return False

    return True


def validate_model_path(pretrained_model_name_or_path: Optional[str]) -> bool:
    """
    Validates the pretrained model name or path against Hugging Face models or local paths.

    Args:
        pretrained_model_name_or_path (Optional[str]): The pretrained model name or path to validate.

    Returns:
        bool: True if the path is a valid Hugging Face model or exists locally; False otherwise.
    """
    from .class_source_model import default_models

    if pretrained_model_name_or_path:
        log.info(
            f"Validating model file or folder path {pretrained_model_name_or_path} existence..."
        )

        # Check if it matches the Hugging Face model pattern
        if re.match(r"^[\w-]+\/[\w-]+$", pretrained_model_name_or_path):
            log.info("...huggingface.co model, skipping validation")
        elif pretrained_model_name_or_path not in default_models:
            # If not one of the default models, check if it's a valid local path
            if not os.path.exists(pretrained_model_name_or_path):
                log.error(
                    f"...source model path '{pretrained_model_name_or_path}' is missing or does not exist"
                )
                return False
            else:
                log.info("...valid")
        else:
            log.info("...valid")
    else:
        log.info("Model name or path not specified, skipping validation")
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
        with open(file_path, "a"):
            pass
        # If the file can be opened, it is considered writable
        return True
    except IOError:
        # If an IOError occurs, the file cannot be written to
        return False
