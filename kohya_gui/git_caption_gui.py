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
    train_data_dir,
    caption_ext,
    batch_size,
    max_data_loader_n_workers,
    max_length,
    model_id,
    prefix,
    postfix,
    use_shell: bool = False,
):
    # Check for images_dir_input
    if train_data_dir == "":
        msgbox("Image folder is missing...")
        return

    if caption_ext == "":
        msgbox("Please provide an extension for the caption files.")
        return

    log.info(f"GIT captioning files in {train_data_dir}...")

    run_cmd = [PYTHON, f"{scriptdir}/sd-scripts/finetune/make_captions_by_git.py"]

    # Add --model_id if provided
    if model_id != "":
        run_cmd.append("--model_id")
        run_cmd.append(model_id)

    # Add other arguments with their values
    run_cmd.append("--batch_size")
    run_cmd.append(str(batch_size))

    run_cmd.append("--max_data_loader_n_workers")
    run_cmd.append(str(max_data_loader_n_workers))

    run_cmd.append("--max_length")
    run_cmd.append(str(max_length))

    # Add --caption_extension if provided
    if caption_ext != "":
        run_cmd.append("--caption_extension")
        run_cmd.append(caption_ext)

    # Add the directory containing the training data
    run_cmd.append(train_data_dir)

    # Log the command
    log.info(" ".join(run_cmd))

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        rf"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    # Adding an example of an environment variable that might be relevant
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Run the command
    subprocess.run(run_cmd, env=env, shell=use_shell)

    # Add prefix and postfix
    add_pre_postfix(
        folder=train_data_dir,
        caption_file_ext=caption_ext,
        prefix=prefix,
        postfix=postfix,
    )

    log.info("...captioning done")


###
# Gradio UI
###


def gradio_git_caption_gui_tab(
    headless=False, default_train_dir=None, use_shell: bool = False
):
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

    with gr.Tab("GIT Captioning"):
        gr.Markdown(
            "This utility will use GIT to caption files for each images in a folder."
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
            caption_ext = gr.Dropdown(
                label="Caption file extension",
                choices=[".cap", ".caption", ".txt"],
                value=".txt",
                interactive=True,
            )

            prefix = gr.Textbox(
                label="Prefix to add to GIT caption",
                placeholder="(Optional)",
                interactive=True,
            )

            postfix = gr.Textbox(
                label="Postfix to add to GIT caption",
                placeholder="(Optional)",
                interactive=True,
            )

            batch_size = gr.Number(value=1, label="Batch size", interactive=True)

        with gr.Row():
            max_data_loader_n_workers = gr.Number(
                value=2, label="Number of workers", interactive=True
            )
            max_length = gr.Number(value=75, label="Max length", interactive=True)
            model_id = gr.Textbox(
                label="Model",
                placeholder="(Optional) model id for GIT in Hugging Face",
                interactive=True,
            )

        caption_button = gr.Button("Caption images")

        caption_button.click(
            caption_images,
            inputs=[
                train_data_dir,
                caption_ext,
                batch_size,
                max_data_loader_n_workers,
                max_length,
                model_id,
                prefix,
                postfix,
                gr.Checkbox(value=use_shell, visible=False),
            ],
            show_progress=False,
        )

        train_data_dir.change(
            fn=lambda path: gr.Dropdown(choices=[""] + list_train_dirs(path)),
            inputs=train_data_dir,
            outputs=train_data_dir,
            show_progress=False,
        )
