import os
import re
import gradio as gr
from easygui import msgbox
from .common_gui import get_folder_path

# def select_folder():
#     # Open a file dialog to select a directory
#     folder = filedialog.askdirectory()

#     # Update the GUI to display the selected folder
#     selected_folder_label.config(text=folder)


def dataset_balancing(concept_repeats, folder):
    if not concept_repeats > 0:
        # Display an error message if the total number of repeats is not a valid integer
        msgbox('Please enter a valid integer for the total number of repeats.')
        return

    concept_repeats = int(concept_repeats)

    # Check if folder exist
    if folder == '' or not os.path.isdir(folder):
        msgbox('Please enter a valid folder for balancing.')
        return

    # Iterate over the subdirectories in the selected folder
    for subdir in os.listdir(folder):
        # Calculate the number of repeats for the current subdirectory
        # Get a list of all the files in the folder
        files = os.listdir(os.path.join(folder, subdir))

        # Filter the list to include only image files
        image_files = [
            f
            for f in files
            if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))
        ]

        # Count the number of image files
        images = len(image_files)

        # Check if the subdirectory name starts with a number inside braces,
        # indicating that the repeats value should be multiplied
        match = re.match(r'^\{(\d+\.?\d*)\}', subdir)
        if match:
            # Multiply the repeats value by the number inside the braces
            repeats = max(
                1, round(concept_repeats / images * float(match.group(1)))
            )
            subdir = subdir[match.end() :]
        else:
            repeats = max(1, round(concept_repeats / images))

        # Check if the subdirectory name already has a number at the beginning
        match = re.match(r'^\d+_', subdir)
        if match:
            # Replace the existing number with the new number
            old_name = os.path.join(folder, subdir)
            new_name = os.path.join(
                folder, f'{repeats}_{subdir[match.end():]}'
            )
        else:
            # Add the new number at the beginning of the name
            old_name = os.path.join(folder, subdir)
            new_name = os.path.join(folder, f'{repeats}_{subdir}')

        os.rename(old_name, new_name)

    msgbox('Dataset balancing completed...')


def gradio_dataset_balancing_tab():
    with gr.Tab('Dataset balancing'):
        gr.Markdown('This utility will ensure that each concept folder in the dataset folder is used equally during the training process of the dreambooth machine learning model, regardless of the number of images in each folder. It will do this by renaming the concept folders to indicate the number of times they should be repeated during training.')
        with gr.Row():
            select_dataset_folder_input = gr.Textbox(label="Dataset folder",
                placeholder='Folder containing the concepts folders to balance...',
                interactive=True,
            )

            select_dataset_folder_button = gr.Button(
                '📂', elem_id='open_folder_small'
            )
            select_dataset_folder_button.click(
                get_folder_path, outputs=select_dataset_folder_input
            )

            total_repeats_number = gr.Number(
                value=1000,
                min=1,
                interactive=True,
                label='Training steps per concept per epoch',
            )
        balance_button = gr.Button('Balance dataset')
        balance_button.click(
            dataset_balancing,
            inputs=[total_repeats_number, select_dataset_folder_input],
        )
