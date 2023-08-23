import gradio as gr
from easygui import diropenbox, msgbox
from .common_gui import get_folder_path
import shutil
import os

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()


def copy_info_to_Folders_tab(training_folder):
    img_folder = os.path.join(training_folder, 'img')
    if os.path.exists(os.path.join(training_folder, 'reg')):
        reg_folder = os.path.join(training_folder, 'reg')
    else:
        reg_folder = ''
    model_folder = os.path.join(training_folder, 'model')
    log_folder = os.path.join(training_folder, 'log')

    return img_folder, reg_folder, model_folder, log_folder


def dreambooth_folder_preparation(
    util_training_images_dir_input,
    util_training_images_repeat_input,
    util_instance_prompt_input,
    util_regularization_images_dir_input,
    util_regularization_images_repeat_input,
    util_class_prompt_input,
    util_training_dir_output,
):

    # Check if the input variables are empty
    if not len(util_training_dir_output):
        log.info(
            "Destination training directory is missing... can't perform the required task..."
        )
        return
    else:
        # Create the util_training_dir_output directory if it doesn't exist
        os.makedirs(util_training_dir_output, exist_ok=True)

    # Check for instance prompt
    if util_instance_prompt_input == '':
        msgbox('Instance prompt missing...')
        return

    # Check for class prompt
    if util_class_prompt_input == '':
        msgbox('Class prompt missing...')
        return

    # Create the training_dir path
    if util_training_images_dir_input == '':
        log.info(
            "Training images directory is missing... can't perform the required task..."
        )
        return
    else:
        training_dir = os.path.join(
            util_training_dir_output,
            f'img/{int(util_training_images_repeat_input)}_{util_instance_prompt_input} {util_class_prompt_input}',
        )

        # Remove folders if they exist
        if os.path.exists(training_dir):
            log.info(f'Removing existing directory {training_dir}...')
            shutil.rmtree(training_dir)

        # Copy the training images to their respective directories
        log.info(f'Copy {util_training_images_dir_input} to {training_dir}...')
        shutil.copytree(util_training_images_dir_input, training_dir)

    if not util_regularization_images_dir_input == '':
        # Create the regularization_dir path
        if not util_regularization_images_repeat_input > 0:
            log.info(
                'Repeats is missing... not copying regularisation images...'
            )
        else:
            regularization_dir = os.path.join(
                util_training_dir_output,
                f'reg/{int(util_regularization_images_repeat_input)}_{util_class_prompt_input}',
            )

            # Remove folders if they exist
            if os.path.exists(regularization_dir):
                log.info(
                    f'Removing existing directory {regularization_dir}...'
                )
                shutil.rmtree(regularization_dir)

            # Copy the regularisation images to their respective directories
            log.info(
                f'Copy {util_regularization_images_dir_input} to {regularization_dir}...'
            )
            shutil.copytree(
                util_regularization_images_dir_input, regularization_dir
            )
    else:
        log.info(
            'Regularization images directory is missing... not copying regularisation images...'
        )

    # create log and model folder
    # Check if the log folder exists and create it if it doesn't
    if not os.path.exists(os.path.join(util_training_dir_output, 'log')):
        os.makedirs(os.path.join(util_training_dir_output, 'log'))

    # Check if the model folder exists and create it if it doesn't
    if not os.path.exists(os.path.join(util_training_dir_output, 'model')):
        os.makedirs(os.path.join(util_training_dir_output, 'model'))

    log.info(
        f'Done creating kohya_ss training folder structure at {util_training_dir_output}...'
    )


def gradio_dreambooth_folder_creation_tab(
    train_data_dir_input=gr.Textbox(),
    reg_data_dir_input=gr.Textbox(),
    output_dir_input=gr.Textbox(),
    logging_dir_input=gr.Textbox(),
    headless=False,
):
    with gr.Tab('Dreambooth/LoRA Folder preparation'):
        gr.Markdown(
            'This utility will create the necessary folder structure for the training images and optional regularization images needed for the kohys_ss Dreambooth/LoRA method to function correctly.'
        )
        with gr.Row():
            util_instance_prompt_input = gr.Textbox(
                label='Instance prompt',
                placeholder='Eg: asd',
                interactive=True,
            )
            util_class_prompt_input = gr.Textbox(
                label='Class prompt',
                placeholder='Eg: person',
                interactive=True,
            )
        with gr.Row():
            util_training_images_dir_input = gr.Textbox(
                label='Training images',
                placeholder='Directory containing the training images',
                interactive=True,
            )
            button_util_training_images_dir_input = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            button_util_training_images_dir_input.click(
                get_folder_path,
                outputs=util_training_images_dir_input,
                show_progress=False,
            )
            util_training_images_repeat_input = gr.Number(
                label='Repeats',
                value=40,
                interactive=True,
                elem_id='number_input',
            )
        with gr.Row():
            util_regularization_images_dir_input = gr.Textbox(
                label='Regularisation images',
                placeholder='(Optional) Directory containing the regularisation images',
                interactive=True,
            )
            button_util_regularization_images_dir_input = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            button_util_regularization_images_dir_input.click(
                get_folder_path,
                outputs=util_regularization_images_dir_input,
                show_progress=False,
            )
            util_regularization_images_repeat_input = gr.Number(
                label='Repeats',
                value=1,
                interactive=True,
                elem_id='number_input',
            )
        with gr.Row():
            util_training_dir_output = gr.Textbox(
                label='Destination training directory',
                placeholder='Directory where formatted training and regularisation folders will be placed',
                interactive=True,
            )
            button_util_training_dir_output = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            button_util_training_dir_output.click(
                get_folder_path, outputs=util_training_dir_output
            )
        button_prepare_training_data = gr.Button('Prepare training data')
        button_prepare_training_data.click(
            dreambooth_folder_preparation,
            inputs=[
                util_training_images_dir_input,
                util_training_images_repeat_input,
                util_instance_prompt_input,
                util_regularization_images_dir_input,
                util_regularization_images_repeat_input,
                util_class_prompt_input,
                util_training_dir_output,
            ],
            show_progress=False,
        )
        button_copy_info_to_Folders_tab = gr.Button('Copy info to Folders Tab')
        button_copy_info_to_Folders_tab.click(
            copy_info_to_Folders_tab,
            inputs=[util_training_dir_output],
            outputs=[
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ],
            show_progress=False,
        )
