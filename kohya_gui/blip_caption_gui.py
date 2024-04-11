import gradio as gr
from easygui import msgbox
import subprocess
import os
import sys
from .common_gui import get_folder_path, add_pre_postfix, scriptdir, list_dirs
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

PYTHON = sys.executable


def caption_images(
    train_data_dir: str,
    caption_file_ext: str,
    batch_size: int,
    num_beams: int,
    top_p: float,
    max_length: int,
    min_length: int,
    beam_search: bool,
    prefix: str = "",
    postfix: str = "",
) -> None:
    """
    Automatically generates captions for images in the specified directory using the BLIP model.

    This function prepares and executes a command-line script to process images in batches, applying advanced
    NLP techniques for caption generation. It supports customization of the captioning process through various
    parameters like batch size, beam search, and more. Optionally, prefixes and postfixes can be added to captions.


    Args:
        train_data_dir (str): The directory containing the images to be captioned.
        caption_file_ext (str): The extension for the caption files.
        batch_size (int): The batch size for the captioning process.
        num_beams (int): The number of beams to use in the captioning process.
        top_p (float): The top p value to use in the captioning process.
        max_length (int): The maximum length of the captions.
        min_length (int): The minimum length of the captions.
        beam_search (bool): Whether to use beam search in the captioning process.
        prefix (str): The prefix to add to the captions.
        postfix (str): The postfix to add to the captions.
    """
    # Check if the image folder is provided
    if not train_data_dir:
        msgbox("Image folder is missing...")
        return

    # Check if the caption file extension is provided
    if not caption_file_ext:
        msgbox("Please provide an extension for the caption files.")
        return

    log.info(f"Captioning files in {train_data_dir}...")

    # Construct the command to run
    run_cmd = rf'"{PYTHON}" "{scriptdir}/sd-scripts/finetune/make_captions.py"'
    run_cmd += f' --batch_size="{int(batch_size)}"'
    run_cmd += f' --num_beams="{int(num_beams)}"'
    run_cmd += f' --top_p="{top_p}"'
    run_cmd += f' --max_length="{int(max_length)}"'
    run_cmd += f' --min_length="{int(min_length)}"'
    if beam_search:
        run_cmd += f" --beam_search"
    if caption_file_ext:
        run_cmd += f' --caption_extension="{caption_file_ext}"'
    run_cmd += f' "{train_data_dir}"'
    run_cmd += f' --caption_weights="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"'

    log.info(run_cmd)

    # Set up the environment
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"
    )

    # Run the command in the sd-scripts folder context
    subprocess.run(run_cmd, env=env, cwd=f"{scriptdir}/sd-scripts")

    # Add prefix and postfix
    add_pre_postfix(
        folder=train_data_dir,
        caption_file_ext=caption_file_ext,
        prefix=prefix,
        postfix=postfix,
    )

    log.info("...captioning done")


###
# Gradio UI
###


def gradio_blip_caption_gui_tab(headless=False, default_train_dir=None):
    from .common_gui import create_refresh_button

    default_train_dir = (
        default_train_dir
        if default_train_dir is not None
        else os.path.join(scriptdir, "data")
    )
    current_train_dir = default_train_dir

    def list_train_dirs(path):
        nonlocal current_train_dir
        current_train_dir = path
        return list(list_dirs(path))

    with gr.Tab("BLIP Captioning"):
        gr.Markdown(
            "This utility uses BLIP to caption files for each image in a folder."
        )
        with gr.Group(), gr.Row():
            train_data_dir = gr.Dropdown(
                label="Image folder to caption (containing the images to caption)",
                choices=[""] + list_train_dirs(default_train_dir),
                value="",
                interactive=True,
                allow_custom_value=True,
            )
            create_refresh_button(
                train_data_dir,
                lambda: None,
                lambda: {"choices": list_train_dirs(current_train_dir)},
                "open_folder_small",
            )
            button_train_data_dir_input = gr.Button(
                "📂",
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_train_data_dir_input.click(
                get_folder_path,
                outputs=train_data_dir,
                show_progress=False,
            )
        with gr.Row():
            caption_file_ext = gr.Textbox(
                label="Caption file extension",
                placeholder="Extension for caption file (e.g., .caption, .txt)",
                value=".txt",
                interactive=True,
            )

            prefix = gr.Textbox(
                label="Prefix to add to BLIP caption",
                placeholder="(Optional)",
                interactive=True,
            )

            postfix = gr.Textbox(
                label="Postfix to add to BLIP caption",
                placeholder="(Optional)",
                interactive=True,
            )

            batch_size = gr.Number(value=1, label="Batch size", interactive=True)

        with gr.Row():
            beam_search = gr.Checkbox(
                label="Use beam search", interactive=True, value=True
            )
            num_beams = gr.Number(value=1, label="Number of beams", interactive=True)
            top_p = gr.Number(value=0.9, label="Top p", interactive=True)
            max_length = gr.Number(value=75, label="Max length", interactive=True)
            min_length = gr.Number(value=5, label="Min length", interactive=True)

        caption_button = gr.Button("Caption images")

        caption_button.click(
            caption_images,
            inputs=[
                train_data_dir,
                caption_file_ext,
                batch_size,
                num_beams,
                top_p,
                max_length,
                min_length,
                beam_search,
                prefix,
                postfix,
            ],
            show_progress=False,
        )

        train_data_dir.change(
            fn=lambda path: gr.Dropdown(choices=[""] + list_train_dirs(path)),
            inputs=train_data_dir,
            outputs=train_data_dir,
            show_progress=False,
        )
