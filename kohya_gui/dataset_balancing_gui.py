import os
import re
import gradio as gr
from easygui import msgbox, boolbox
from .common_gui import get_folder_path, scriptdir, list_dirs, create_refresh_button

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


import os
import re
import logging as log
from easygui import msgbox

def dataset_balancing(concept_repeats, folder, insecure):

    if not concept_repeats > 0:
        # Display an error message if the total number of repeats is not a valid integer
        msgbox("Please enter a valid integer for the total number of repeats.")
        return

    concept_repeats = int(concept_repeats)

    # Check if folder exist
    if folder == "" or not os.path.isdir(folder):
        msgbox("Please enter a valid folder for balancing.")
        return

    pattern = re.compile(r"^\d+_.+$")

    # Iterate over the subdirectories in the selected folder
    for subdir in os.listdir(folder):
        if pattern.match(subdir) or insecure:
            # Calculate the number of repeats for the current subdirectory
            # Get a list of all the files in the folder
            files = os.listdir(os.path.join(folder, subdir))

            # Filter the list to include only image files
            image_files = [
                f
                for f in files
                if f.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
            ]

            # Count the number of image files
            images = len(image_files)

            if images == 0:
                log.info(
                    f"No images of type .jpg, .jpeg, .png, .gif, .webp were found in {os.listdir(os.path.join(folder, subdir))}"
                )

            # Check if the subdirectory name starts with a number inside braces,
            # indicating that the repeats value should be multiplied
            match = re.match(r"^\{(\d+\.?\d*)\}", subdir)
            if match:
                # Multiply the repeats value by the number inside the braces
                if not images == 0:
                    repeats = max(
                        1,
                        round(concept_repeats / images * float(match.group(1))),
                    )
                else:
                    repeats = 0
                subdir = subdir[match.end() :]
            else:
                if not images == 0:
                    repeats = max(1, round(concept_repeats / images))
                else:
                    repeats = 0

            # Check if the subdirectory name already has a number at the beginning
            match = re.match(r"^\d+_", subdir)
            if match:
                # Replace the existing number with the new number
                old_name = os.path.join(folder, subdir)
                new_name = os.path.join(folder, f"{repeats}_{subdir[match.end():]}")
            else:
                # Add the new number at the beginning of the name
                old_name = os.path.join(folder, subdir)
                new_name = os.path.join(folder, f"{repeats}_{subdir}")

            # Check if the new folder name already exists
            if os.path.exists(new_name):
                log.warning(f"Destination folder {new_name} already exists. Skipping...")
            else:
                os.rename(old_name, new_name)
        else:
            log.info(
                f"Skipping folder {subdir} because it does not match kohya_ss expected syntax..."
            )

    msgbox("Dataset balancing completed...")



def warning(insecure):
    if insecure:
        if boolbox(
            f"WARNING!!! You have asked to rename non kohya_ss <num>_<text> folders...\n\nAre you sure you want to do that?",
            choices=("Yes, I like danger", "No, get me out of here"),
        ):
            return True
        else:
            return False


def gradio_dataset_balancing_tab(headless=False):

    current_dataset_dir = os.path.join(scriptdir, "data")

    with gr.Tab("Dreambooth/LoRA Dataset balancing"):
        gr.Markdown(
            "This utility will ensure that each concept folder in the dataset folder is used equally during the training process of the dreambooth machine learning model, regardless of the number of images in each folder. It will do this by renaming the concept folders to indicate the number of times they should be repeated during training."
        )
        gr.Markdown(
            "WARNING! The use of this utility on the wrong folder can lead to unexpected folder renaming!!!"
        )
        with gr.Group(), gr.Row():

            def list_dataset_dirs(path):
                nonlocal current_dataset_dir
                current_dataset_dir = path
                return list(list_dirs(path))

            select_dataset_folder_input = gr.Dropdown(
                label="Dataset folder (folder containing the concepts folders to balance...)",
                interactive=True,
                choices=[""] + list_dataset_dirs(current_dataset_dir),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                select_dataset_folder_input,
                lambda: None,
                lambda: {"choices": list_dataset_dirs(current_dataset_dir)},
                "open_folder_small",
            )
            select_dataset_folder_button = gr.Button(
                "📂",
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            select_dataset_folder_button.click(
                get_folder_path,
                outputs=select_dataset_folder_input,
                show_progress=False,
            )

            total_repeats_number = gr.Number(
                value=1000,
                interactive=True,
                label="Training steps per concept per epoch",
            )
            select_dataset_folder_input.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_dataset_dirs(path)),
                inputs=select_dataset_folder_input,
                outputs=select_dataset_folder_input,
                show_progress=False,
            )

        with gr.Accordion("Advanced options", open=False):
            insecure = gr.Checkbox(
                value=False,
                label="DANGER!!! -- Insecure folder renaming -- DANGER!!!",
            )
            insecure.change(warning, inputs=insecure, outputs=insecure)
        balance_button = gr.Button("Balance dataset")
        balance_button.click(
            dataset_balancing,
            inputs=[
                total_repeats_number,
                select_dataset_folder_input,
                insecure,
            ],
            show_progress=False,
        )
