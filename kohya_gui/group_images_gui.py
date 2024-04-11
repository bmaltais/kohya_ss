import gradio as gr
from easygui import msgbox
import subprocess
from .common_gui import get_folder_path, scriptdir, list_dirs
import os
import sys

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

PYTHON = sys.executable


def group_images(
    input_folder,
    output_folder,
    group_size,
    include_subfolders,
    do_not_copy_other_files,
    generate_captions,
    caption_ext,
):
    if input_folder == "":
        msgbox("Input folder is missing...")
        return

    if output_folder == "":
        msgbox("Please provide an output folder.")
        return

    log.info(f"Grouping images in {input_folder}...")

    run_cmd = rf'"{PYTHON}" "{scriptdir}/tools/group_images.py"'
    run_cmd += f' "{input_folder}"'
    run_cmd += f' "{output_folder}"'
    run_cmd += f" {(group_size)}"
    if include_subfolders:
        run_cmd += f" --include_subfolders"
    if do_not_copy_other_files:
        run_cmd += f" --do_not_copy_other_files"
    if generate_captions:
        run_cmd += f" --caption"
        if caption_ext:
            run_cmd += f" --caption_ext={caption_ext}"

    log.info(run_cmd)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        rf"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"
    )

    # Run the command
    subprocess.run(run_cmd, env=env)

    log.info("...grouping done")


def gradio_group_images_gui_tab(headless=False):
    from .common_gui import create_refresh_button

    current_input_folder = os.path.join(scriptdir, "data")
    current_output_folder = os.path.join(scriptdir, "data")

    def list_input_dirs(path):
        nonlocal current_input_folder
        current_input_folder = path
        return list(list_dirs(path))

    def list_output_dirs(path):
        nonlocal current_output_folder
        current_output_folder = path
        return list(list_dirs(path))

    with gr.Tab("Group Images"):
        gr.Markdown(
            "This utility will group images in a folder based on their aspect ratio."
        )

        with gr.Group(), gr.Row():
            input_folder = gr.Dropdown(
                label="Input folder (containing the images to group)",
                interactive=True,
                choices=[""] + list_input_dirs(current_input_folder),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                input_folder,
                lambda: None,
                lambda: {"choices": list_input_dirs(current_input_folder)},
                "open_folder_small",
            )
            button_input_folder = gr.Button(
                "📂",
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_input_folder.click(
                get_folder_path,
                outputs=input_folder,
                show_progress=False,
            )

            output_folder = gr.Dropdown(
                label="Output folder (where the grouped images will be stored)",
                interactive=True,
                choices=[""] + list_output_dirs(current_output_folder),
                value="",
                allow_custom_value=True,
            )
            create_refresh_button(
                output_folder,
                lambda: None,
                lambda: {"choices": list_output_dirs(current_output_folder)},
                "open_folder_small",
            )
            button_output_folder = gr.Button(
                "📂",
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_output_folder.click(
                get_folder_path,
                outputs=output_folder,
                show_progress=False,
            )

            input_folder.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_input_dirs(path)),
                inputs=input_folder,
                outputs=input_folder,
                show_progress=False,
            )
            output_folder.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_output_dirs(path)),
                inputs=output_folder,
                outputs=output_folder,
                show_progress=False,
            )
        with gr.Row():
            group_size = gr.Slider(
                label="Group size",
                info="Number of images to group together",
                value=4,
                minimum=1,
                maximum=64,
                step=1,
                interactive=True,
            )

            include_subfolders = gr.Checkbox(
                label="Include Subfolders",
                value=False,
                info="Include images in subfolders as well",
            )

            do_not_copy_other_files = gr.Checkbox(
                label="Do not copy other files",
                value=False,
                info="Do not copy other files in the input folder to the output folder",
            )

            generate_captions = gr.Checkbox(
                label="Generate Captions",
                value=False,
                info="Generate caption files for the grouped images based on their folder name",
            )

            caption_ext = gr.Dropdown(
                label="Caption file extension",
                choices=[".cap", ".caption", ".txt"],
                value=".txt",
                interactive=True,
            )

        group_images_button = gr.Button("Group images")

        group_images_button.click(
            group_images,
            inputs=[
                input_folder,
                output_folder,
                group_size,
                include_subfolders,
                do_not_copy_other_files,
                generate_captions,
                caption_ext,
            ],
            show_progress=False,
        )
