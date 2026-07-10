import gradio as gr
from .common_gui import get_folder_path, scriptdir, list_dirs, create_refresh_button
import os
import shutil
import tempfile
from .class_gui_config import KohyaSSGUIConfig

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


def copy_info_to_Folders_tab(training_folder):
    img_folder = gr.Dropdown(value=os.path.join(training_folder, "img"))
    if os.path.exists(os.path.join(training_folder, "reg")):
        reg_folder = gr.Dropdown(value=os.path.join(training_folder, "reg"))
    else:
        reg_folder = gr.Dropdown(value="")
    model_folder = gr.Dropdown(value=os.path.join(training_folder, "model"))
    log_folder = gr.Dropdown(value=os.path.join(training_folder, "log"))

    return img_folder, reg_folder, model_folder, log_folder


def _resolve_path(path: str) -> str:
    """Absolute, normalized path (resolves .. and symlinks when possible)."""
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _paths_equal(a: str, b: str) -> bool:
    return _resolve_path(a) == _resolve_path(b)


def _is_under(path: str, parent: str) -> bool:
    """True if path is strictly inside parent (not equal)."""
    path_r = _resolve_path(path)
    parent_r = _resolve_path(parent)
    if path_r == parent_r:
        return False
    try:
        common = os.path.commonpath([path_r, parent_r])
    except ValueError:
        # Different drives on Windows
        return False
    return common == parent_r


def _validate_copy_pair(src: str, dst: str, label: str) -> str | None:
    """Return an error message if src→dst is unsafe, else None."""
    if not src or not str(src).strip():
        return f"Error: {label} source directory is missing."
    src = str(src).strip()
    if not os.path.isdir(src):
        return f"Error: {label} source does not exist or is not a directory: {src}"

    if _paths_equal(src, dst):
        return (
            f"Error: refused — {label} source and destination are the same path "
            f"({_resolve_path(src)}). This would delete your only copy. "
            f"Point the {label} source at your original folder, not the prepared "
            "target folder under the destination."
        )

    # Destination under source → recursive copy bomb (#1761)
    if _is_under(dst, src):
        return (
            f"Error: refused — {label} destination is inside the source folder "
            f"(source={_resolve_path(src)}, dest={_resolve_path(dst)}). "
            f"Choose a destination outside the {label.lower()} source directory."
        )

    # Source under destination → replacing dest would wipe source
    if _is_under(src, dst):
        return (
            f"Error: refused — {label} source is inside the destination folder "
            f"(source={_resolve_path(src)}, dest={_resolve_path(dst)}). "
            "That would delete the source when replacing the destination."
        )

    return None


def _safe_copytree_replace(src: str, dst: str) -> None:
    """Copy src directory to dst. If dst exists, replace via backup rename after a full staged copy.

    Never deletes src. Existing dst is renamed aside only after the staged copy
    succeeds; on failure to install the new tree, the backup is restored.
    """
    src_r = _resolve_path(src)
    dst_abs = os.path.abspath(dst)
    parent = os.path.dirname(dst_abs)
    os.makedirs(parent, exist_ok=True)

    if _paths_equal(src_r, dst_abs):
        raise RuntimeError(
            f"Refusing to replace destination that equals source: {dst_abs}"
        )

    staging_root = tempfile.mkdtemp(prefix=".kohya_prep_", dir=parent)
    staged = os.path.join(staging_root, "data")
    backup_old = None
    try:
        shutil.copytree(src_r, staged)

        if os.path.isdir(dst_abs):
            # Same-filesystem rename keeps previous dest until new tree is installed
            backup_old = dst_abs + ".kohya_prep_bak"
            if os.path.exists(backup_old):
                shutil.rmtree(backup_old)
            os.rename(dst_abs, backup_old)

        try:
            try:
                os.rename(staged, dst_abs)
            except OSError:
                shutil.move(staged, dst_abs)
        except Exception:
            if backup_old is not None and os.path.isdir(backup_old):
                if not os.path.exists(dst_abs):
                    os.rename(backup_old, dst_abs)
                backup_old = None
            raise

        if backup_old is not None and os.path.isdir(backup_old):
            shutil.rmtree(backup_old)
            backup_old = None
    finally:
        if os.path.isdir(staging_root):
            shutil.rmtree(staging_root, ignore_errors=True)
        if backup_old is not None and os.path.isdir(backup_old):
            log.error(
                "Folder preparation left a backup at %s; restore manually if needed.",
                backup_old,
            )


def dreambooth_folder_preparation(
    util_training_images_dir_input,
    util_training_images_repeat_input,
    util_instance_prompt_input,
    util_regularization_images_dir_input,
    util_regularization_images_repeat_input,
    util_class_prompt_input,
    util_training_dir_output,
):
    """Create kohya training folder layout by copying images.

    Returns a status string for the Gradio UI on every path (success or refusal).
    """
    # Destination required
    if not util_training_dir_output or not str(util_training_dir_output).strip():
        msg = "Error: Destination training directory is missing... can't perform the required task..."
        log.info(msg)
        return msg

    util_training_dir_output = str(util_training_dir_output).strip()

    # Normalize prompts (trailing newlines/spaces break Windows paths — #1541)
    instance_prompt = (util_instance_prompt_input or "").strip()
    class_prompt = (util_class_prompt_input or "").strip()

    if instance_prompt == "":
        msg = "Error: Instance prompt missing..."
        log.error(msg)
        return msg

    if class_prompt == "":
        msg = "Error: Class prompt missing..."
        log.error(msg)
        return msg

    if (
        not util_training_images_dir_input
        or not str(util_training_images_dir_input).strip()
    ):
        msg = "Error: Training images directory is missing... can't perform the required task..."
        log.info(msg)
        return msg

    util_training_images_dir_input = str(util_training_images_dir_input).strip()

    try:
        train_repeats = int(util_training_images_repeat_input)
    except (TypeError, ValueError):
        msg = f"Error: Invalid training images repeats value: {util_training_images_repeat_input!r}"
        log.error(msg)
        return msg

    training_dir = os.path.join(
        util_training_dir_output,
        f"img/{train_repeats}_{instance_prompt} {class_prompt}",
    )

    # Optional regularization — resolve target early for preflight validation
    do_reg = bool(
        util_regularization_images_dir_input
        and str(util_regularization_images_dir_input).strip()
    )
    regularization_dir = None
    reg_src = None
    reg_repeats = 0
    if do_reg:
        reg_src = str(util_regularization_images_dir_input).strip()
        try:
            reg_repeats = int(util_regularization_images_repeat_input)
        except (TypeError, ValueError):
            reg_repeats = 0
        if reg_repeats <= 0:
            log.info("Repeats is missing... not copying regularisation images...")
            do_reg = False
        else:
            regularization_dir = os.path.join(
                util_training_dir_output,
                f"reg/{reg_repeats}_{class_prompt}",
            )

    # --- Preflight: validate before any mutation ---
    err = _validate_copy_pair(
        util_training_images_dir_input, training_dir, "Training images"
    )
    if err:
        log.error(err)
        return err

    if do_reg and regularization_dir is not None:
        err = _validate_copy_pair(reg_src, regularization_dir, "Regularisation images")
        if err:
            log.error(err)
            return err

    # Create destination root only after validation
    os.makedirs(util_training_dir_output, exist_ok=True)

    try:
        log.info(f"Copy {util_training_images_dir_input} to {training_dir}...")
        _safe_copytree_replace(util_training_images_dir_input, training_dir)

        if do_reg and regularization_dir is not None:
            log.info(f"Copy {reg_src} to {regularization_dir}...")
            _safe_copytree_replace(reg_src, regularization_dir)
        elif not (
            util_regularization_images_dir_input
            and str(util_regularization_images_dir_input).strip()
        ):
            log.info(
                "Regularization images directory is missing... not copying regularisation images..."
            )

        # log and model folders
        os.makedirs(os.path.join(util_training_dir_output, "log"), exist_ok=True)
        os.makedirs(os.path.join(util_training_dir_output, "model"), exist_ok=True)

    except Exception as exc:
        msg = f"Error: folder preparation failed: {exc}"
        log.error(msg)
        return msg

    msg = f"Done creating kohya_ss training folder structure at {util_training_dir_output}..."
    log.info(msg)
    return msg


def gradio_dreambooth_folder_creation_tab(
    config: KohyaSSGUIConfig,
    train_data_dir_input=gr.Dropdown(),
    reg_data_dir_input=gr.Dropdown(),
    output_dir_input=gr.Dropdown(),
    logging_dir_input=gr.Dropdown(),
    headless=False,
):

    current_train_data_dir = os.path.join(scriptdir, "data")
    current_reg_data_dir = os.path.join(scriptdir, "data")
    current_train_output_dir = os.path.join(scriptdir, "data")

    with gr.Tab("Dreambooth/LoRA Folder preparation"):
        gr.Markdown(
            "This utility will create the necessary folder structure for the training images and optional regularization images needed for the kohys_ss Dreambooth/LoRA method to function correctly."
        )
        with gr.Row():
            util_instance_prompt_input = gr.Textbox(
                label="Instance prompt",
                placeholder="Eg: asd",
                interactive=True,
                value=config.get(key="dataset_preparation.instance_prompt", default=""),
            )
            util_class_prompt_input = gr.Textbox(
                label="Class prompt",
                placeholder="Eg: person",
                interactive=True,
                value=config.get(key="dataset_preparation.class_prompt", default=""),
            )
        with gr.Group(), gr.Row():

            def list_train_data_dirs(path):
                nonlocal current_train_data_dir
                current_train_data_dir = path
                return list(list_dirs(path))

            util_training_images_dir_input = gr.Dropdown(
                label="Training images (directory containing the training images)",
                interactive=True,
                choices=[
                    config.get(key="dataset_preparation.images_folder", default="")
                ]
                + list_train_data_dirs(current_train_data_dir),
                value=config.get(key="dataset_preparation.images_folder", default=""),
                allow_custom_value=True,
            )
            create_refresh_button(
                util_training_images_dir_input,
                lambda: None,
                lambda: {"choices": list_train_data_dirs(current_train_data_dir)},
                "open_folder_small",
            )
            button_util_training_images_dir_input = gr.Button(
                "📂",
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_util_training_images_dir_input.click(
                get_folder_path,
                outputs=util_training_images_dir_input,
                show_progress=False,
            )
            util_training_images_repeat_input = gr.Number(
                label="Repeats",
                value=config.get(
                    key="dataset_preparation.util_training_images_repeat_input",
                    default=40,
                ),
                interactive=True,
                elem_id="number_input",
            )
            util_training_images_dir_input.change(
                fn=lambda path: gr.Dropdown(
                    choices=[
                        config.get(key="dataset_preparation.images_folder", default="")
                    ]
                    + list_train_data_dirs(path)
                ),
                inputs=util_training_images_dir_input,
                outputs=util_training_images_dir_input,
                show_progress=False,
            )

        with gr.Group(), gr.Row():

            def list_reg_data_dirs(path):
                nonlocal current_reg_data_dir
                current_reg_data_dir = path
                return list(list_dirs(path))

            util_regularization_images_dir_input = gr.Dropdown(
                label="Regularisation images (Optional. directory containing the regularisation images)",
                interactive=True,
                choices=[
                    config.get(key="dataset_preparation.reg_images_folder", default="")
                ]
                + list_reg_data_dirs(current_reg_data_dir),
                value=config.get(
                    key="dataset_preparation.reg_images_folder", default=""
                ),
                allow_custom_value=True,
            )
            create_refresh_button(
                util_regularization_images_dir_input,
                lambda: None,
                lambda: {"choices": list_reg_data_dirs(current_reg_data_dir)},
                "open_folder_small",
            )
            button_util_regularization_images_dir_input = gr.Button(
                "📂",
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_util_regularization_images_dir_input.click(
                get_folder_path,
                outputs=util_regularization_images_dir_input,
                show_progress=False,
            )
            util_regularization_images_repeat_input = gr.Number(
                label="Repeats",
                value=config.get(
                    key="dataset_preparation.util_regularization_images_repeat_input",
                    default=1,
                ),
                interactive=True,
                elem_id="number_input",
            )
            util_regularization_images_dir_input.change(
                fn=lambda path: gr.Dropdown(choices=[""] + list_reg_data_dirs(path)),
                inputs=util_regularization_images_dir_input,
                outputs=util_regularization_images_dir_input,
                show_progress=False,
            )
        with gr.Group(), gr.Row():

            def list_train_output_dirs(path):
                nonlocal current_train_output_dir
                current_train_output_dir = path
                return list(list_dirs(path))

            util_training_dir_output = gr.Dropdown(
                label="Destination training directory (where formatted training and regularisation folders will be placed)",
                interactive=True,
                choices=[config.get(key="train_data_dir", default="")]
                + list_train_output_dirs(current_train_output_dir),
                value=config.get(key="train_data_dir", default=""),
                allow_custom_value=True,
            )
            create_refresh_button(
                util_training_dir_output,
                lambda: None,
                lambda: {"choices": list_train_output_dirs(current_train_output_dir)},
                "open_folder_small",
            )
            button_util_training_dir_output = gr.Button(
                "📂",
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_util_training_dir_output.click(
                get_folder_path, outputs=util_training_dir_output
            )
            util_training_dir_output.change(
                fn=lambda path: gr.Dropdown(
                    choices=[config.get(key="train_data_dir", default="")]
                    + list_train_output_dirs(path)
                ),
                inputs=util_training_dir_output,
                outputs=util_training_dir_output,
                show_progress=False,
            )
        button_prepare_training_data = gr.Button("Prepare training data")
        prepare_status = gr.Textbox(
            label="Preparation status",
            interactive=False,
            lines=3,
            placeholder="Status messages appear here after you run Prepare training data.",
        )
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
            outputs=prepare_status,
            show_progress=True,
        )

        button_copy_info_to_Folders_tab = gr.Button("Copy info to respective fields")
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
