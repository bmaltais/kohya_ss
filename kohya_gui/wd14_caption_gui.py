import gradio as gr
import os
import re
import subprocess
import sys

from .common_gui import (
    get_folder_path,
    add_pre_postfix,
    scriptdir,
    list_dirs,
    setup_environment,
)
from .class_gui_config import KohyaSSGUIConfig
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()
old_onnx_value = True

# Same pattern as BLIP captioning: run the tagger with the venv Python.
# accelerate launch is unnecessary for single-process ONNX tagging and pulls
# TensorFlow via TensorBoard (#3422 / #3577 Phase 1).
PYTHON = sys.executable

# Repo IDs exposed in the UI (include changelog models such as eva02 v3).
WD14_REPO_IDS = [
    "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "SmilingWolf/wd-v1-4-vit-tagger-v2",
    "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "SmilingWolf/wd-v1-4-moat-tagger-v2",
    "SmilingWolf/wd-swinv2-tagger-v3",
    "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-convnext-tagger-v3",
    "SmilingWolf/wd-eva02-large-tagger-v3",
]

DEFAULT_WD14_REPO_ID = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"

# HF hub style: org/name or org/name/subdir segments only (no path traversal).
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")


def sanitize_hf_repo_id(repo_id: str) -> str | None:
    """Return repo_id if it is a safe Hugging Face id, else None.

    Rejects empty values, ``..``, absolute paths, and characters outside the
    normal ``org/name`` (optional subdir) pattern used by the tagger cache.
    """
    if not repo_id or not isinstance(repo_id, str):
        return None
    cleaned = repo_id.strip()
    if not cleaned or ".." in cleaned or cleaned.startswith(("/", "\\")):
        return None
    if not _HF_REPO_ID_RE.fullmatch(cleaned):
        return None
    return cleaned


def resolve_wd14_model_dir(repo_id: str) -> str | None:
    """Resolve the local model cache dir under the project, or None if unsafe.

    Guarantees the resolved path stays under ``{scriptdir}/wd14_tagger_model``.
    """
    safe_id = sanitize_hf_repo_id(repo_id)
    if safe_id is None:
        return None
    base = os.path.realpath(os.path.join(scriptdir, "wd14_tagger_model"))
    candidate = os.path.realpath(os.path.join(base, safe_id.replace("/", "_")))
    # Ensure candidate is base or a subpath of base (path-injection guard).
    try:
        common = os.path.commonpath([base, candidate])
    except ValueError:
        return None
    if common != base:
        return None
    return candidate


def validate_train_data_dir(train_data_dir: str) -> str | None:
    """Return absolute image folder path if it exists as a directory."""
    if not train_data_dir or not isinstance(train_data_dir, str):
        return None
    path = os.path.abspath(os.path.expanduser(train_data_dir.strip()))
    if not os.path.isdir(path):
        return None
    return path


def check_keras_backend_ready() -> str | None:
    """Return an actionable error if the non-ONNX (Keras/TF) path cannot run.

    ONNX captioning does not need TensorFlow. Call only when onnx is False.
    """
    try:
        import numpy as np
    except ImportError:
        return (
            "NumPy is not installed. Enable ONNX (recommended) or install "
            'NumPy 1.x: pip install "numpy>=1.26,<2"'
        )

    major = int(str(np.__version__).split(".", 1)[0])
    if major >= 2:
        return (
            f"NumPy {np.__version__} is installed; the Keras/TF WD14 path needs "
            "NumPy 1.x. Enable ONNX (recommended) or run: "
            'pip install "numpy>=1.26,<2"'
        )

    try:
        import tensorflow  # noqa: F401
    except Exception as exc:  # import can fail for many ABI reasons
        return (
            f"TensorFlow import failed ({exc}). Enable ONNX (recommended) or "
            "fix TensorFlow with NumPy 1.x before using the Keras path."
        )
    return None


def caption_images(
    train_data_dir: str,
    caption_extension: str,
    batch_size: int,
    general_threshold: float,
    character_threshold: float,
    repo_id: str,
    recursive: bool,
    max_data_loader_n_workers: int,
    debug: bool,
    undesired_tags: str,
    frequency_tags: bool,
    always_first_tags: str,
    caption_postfix: str,
    onnx: bool,
    append_tags: bool,
    force_download: bool,
    caption_separator: str,
    tag_replacement: bool,
    character_tag_expand: str,
    use_rating_tags: bool,
    use_rating_tags_as_last_tag: bool,
    use_quality_tags: bool,
    use_quality_tags_as_last_tag: bool,
    character_tags_first: bool,
    remove_underscore: bool,
    thresh: float,
) -> None:
    # Check for images_dir_input
    if train_data_dir == "":
        log.info("Image folder is missing...")
        return

    validated_train_dir = validate_train_data_dir(train_data_dir)
    if validated_train_dir is None:
        log.error(f"Image folder is missing or not a directory: {train_data_dir!r}")
        return
    train_data_dir = validated_train_dir

    if caption_extension == "":
        log.info("Please provide an extension for the caption files.")
        return

    # Normalize extension (users may type "txt" without a leading dot)
    caption_extension = caption_extension.strip()
    if caption_extension and not caption_extension.startswith("."):
        caption_extension = f".{caption_extension}"

    if not onnx:
        keras_error = check_keras_backend_ready()
        if keras_error:
            log.error(keras_error)
            return

    safe_repo_id = sanitize_hf_repo_id(repo_id)
    if safe_repo_id is None:
        log.error(
            f"Invalid Hugging Face repo id (refusing path-unsafe value): {repo_id!r}"
        )
        return
    repo_id = safe_repo_id

    model_dir = resolve_wd14_model_dir(repo_id)
    if model_dir is None:
        log.error(f"Could not resolve a safe model cache path for {repo_id!r}")
        return
    if not os.path.exists(model_dir):
        force_download = True

    log.info(f"Captioning files in {train_data_dir}...")
    # Fixed script path under the project tree (not user-controlled).
    tagger_script = os.path.join(
        scriptdir, "sd-scripts", "finetune", "tag_images_by_wd14_tagger.py"
    )
    run_cmd = [
        PYTHON,
        tagger_script,
    ]

    # Prefix/postfix are applied after a successful run via add_pre_postfix
    # (not --always_first_tags) so they are not double-applied and postfix works.

    if append_tags:
        run_cmd.append("--append_tags")
    run_cmd.append("--batch_size")
    run_cmd.append(str(int(batch_size)))
    run_cmd.append("--caption_extension")
    run_cmd.append(caption_extension)
    run_cmd.append("--caption_separator")
    run_cmd.append(caption_separator)

    if character_tag_expand:
        run_cmd.append("--character_tag_expand")
    if not character_threshold == 0.35:
        run_cmd.append("--character_threshold")
        run_cmd.append(str(character_threshold))
    if debug:
        run_cmd.append("--debug")
    if force_download:
        run_cmd.append("--force_download")
    if frequency_tags:
        run_cmd.append("--frequency_tags")
    if not general_threshold == 0.35:
        run_cmd.append("--general_threshold")
        run_cmd.append(str(general_threshold))
    run_cmd.append("--max_data_loader_n_workers")
    run_cmd.append(str(int(max_data_loader_n_workers)))

    if onnx:
        run_cmd.append("--onnx")
    if recursive:
        run_cmd.append("--recursive")
    if remove_underscore:
        run_cmd.append("--remove_underscore")
    run_cmd.append("--repo_id")
    run_cmd.append(repo_id)
    if not tag_replacement == "":
        run_cmd.append("--tag_replacement")
        run_cmd.append(tag_replacement)
    if not thresh == 0.35:
        run_cmd.append("--thresh")
        run_cmd.append(str(thresh))
    if not undesired_tags == "":
        run_cmd.append("--undesired_tags")
        run_cmd.append(undesired_tags)
    if use_rating_tags:
        run_cmd.append("--use_rating_tags")
    if use_rating_tags_as_last_tag:
        run_cmd.append("--use_rating_tags_as_last_tag")
    if use_quality_tags:
        run_cmd.append("--use_quality_tags")
    if use_quality_tags_as_last_tag:
        run_cmd.append("--use_quality_tags_as_last_tag")
    if character_tags_first:
        run_cmd.append("--character_tags_first")

    # Add the validated image directory (absolute path, isdir-checked above).
    run_cmd.append(train_data_dir)

    env = setup_environment()

    # Reconstruct the safe command string for display
    command_to_run = " ".join(run_cmd)
    log.info(f"Executing command: {command_to_run}")

    # Run with project root as cwd so the default relative model dir
    # (wd14_tagger_model/) stays at the repo root. PYTHONPATH from
    # setup_environment() already includes sd-scripts for imports.
    # shell=False + argv list: no shell metacharacter expansion.
    result = subprocess.run(
        run_cmd,
        env=env,
        shell=False,
        cwd=scriptdir,
    )

    if result.returncode != 0:
        log.error(
            f"WD14 captioning failed with exit code {result.returncode}. "
            "Caption files may be incomplete; prefix/postfix was not applied."
        )
        return

    # Prefix/postfix only after a successful tagger run (#2569, #3577 Phase 2)
    add_pre_postfix(
        folder=train_data_dir,
        caption_file_ext=caption_extension,
        prefix=always_first_tags or "",
        postfix=caption_postfix or "",
        recursive=recursive,
    )

    log.info("...captioning done")


###
# Gradio UI
###


def gradio_wd14_caption_gui_tab(
    headless=False,
    default_train_dir=None,
    config: KohyaSSGUIConfig = {},
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

    with gr.Tab("WD14 Captioning"):
        gr.Markdown(
            "This utility uses WD14 to caption images in a folder.\n\n"
            "**ONNX (recommended, default):** runs via onnxruntime and does not "
            "require a working TensorFlow install. Disable only if you need the "
            "Keras path; that path needs TensorFlow and NumPy 1.x.\n\n"
            "**Prefix / postfix** are applied after a successful tagger run "
            "(not via the tagger CLI), so they work for both ONNX and Keras."
        )

        # Input Settings
        with gr.Group(), gr.Row():
            train_data_dir = gr.Dropdown(
                label="Image folder to caption (containing the images to caption)",
                choices=[config.get("wd14_caption.train_data_dir", "")]
                + list_train_dirs(default_train_dir),
                value=config.get("wd14_caption.train_data_dir", ""),
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

            repo_id = gr.Dropdown(
                label="Repo ID",
                choices=WD14_REPO_IDS,
                value=config.get("wd14_caption.repo_id", DEFAULT_WD14_REPO_ID),
                show_label="Repo id for wd14 tagger on Hugging Face",
                allow_custom_value=True,
                info="Includes v2/v3 and eva02 large v3. Custom HF repo IDs allowed.",
            )

            force_download = gr.Checkbox(
                label="Force model re-download",
                value=config.get("wd14_caption.force_download", False),
                info="Useful to force model re download when switching to onnx",
            )

        with gr.Row():

            caption_extension = gr.Dropdown(
                label="Caption file extension",
                choices=[".cap", ".caption", ".txt", ".wd14.txt"],
                value=config.get("wd14_caption.caption_extension", ".txt"),
                interactive=True,
                allow_custom_value=True,
                info="Type any extension for multi-tool workflows (e.g. .wd14.txt).",
            )

            caption_separator = gr.Textbox(
                label="Caption Separator",
                value=config.get("wd14_caption.caption_separator", ", "),
                interactive=True,
            )

        with gr.Row():

            tag_replacement = gr.Textbox(
                label="Tag replacement",
                info=r"tag replacement in the format of `source1,target1;source2,target2; ...`. Escape `,` and `;` with `\`. e.g. `tag1,tag2;tag3,tag4`",
                value=config.get("wd14_caption.tag_replacement", ""),
                interactive=True,
            )

            character_tag_expand = gr.Checkbox(
                label="Character tag expand",
                info="expand tag tail parenthesis to another tag for character tags. `chara_name_(series)` becomes `chara_name, series`",
                value=config.get("wd14_caption.character_tag_expand", False),
                interactive=True,
            )

        undesired_tags = gr.Textbox(
            label="Undesired tags",
            placeholder="(Optional) Separate `undesired_tags` with comma `(,)` if you want to remove multiple tags, e.g. `1girl,solo,smile`.",
            interactive=True,
            value=config.get("wd14_caption.undesired_tags", ""),
        )

        with gr.Row():
            always_first_tags = gr.Textbox(
                label="Prefix to add to WD14 caption",
                info="Applied after tagging succeeds (GUI post-process). e.g.: 1girl, 1boy",
                placeholder="(Optional)",
                interactive=True,
                value=config.get("wd14_caption.always_first_tags", ""),
            )
            caption_postfix = gr.Textbox(
                label="Postfix to add to WD14 caption",
                info="Applied after tagging succeeds (GUI post-process).",
                placeholder="(Optional)",
                interactive=True,
                value=config.get("wd14_caption.caption_postfix", ""),
            )

        with gr.Row():
            onnx = gr.Checkbox(
                label="Use ONNX (recommended)",
                value=config.get("wd14_caption.onnx", True),
                interactive=True,
                info=(
                    "Recommended. Uses onnxruntime (no TensorFlow). "
                    "Turn off only for the Keras/TF backend (needs TensorFlow + NumPy 1.x)."
                ),
            )
            append_tags = gr.Checkbox(
                label="Append TAGs",
                value=config.get("wd14_caption.append_tags", False),
                interactive=True,
                info="This option appends the tags to the existing tags, instead of replacing them.",
            )

            use_rating_tags = gr.Checkbox(
                label="Use rating tags",
                value=config.get("wd14_caption.use_rating_tags", False),
                interactive=True,
                info="Adds rating tags as the first tag",
            )

            use_rating_tags_as_last_tag = gr.Checkbox(
                label="Use rating tags as last tag",
                value=config.get("wd14_caption.use_rating_tags_as_last_tag", False),
                interactive=True,
                info="Adds rating tags as the last tag",
            )

        with gr.Row():
            use_quality_tags = gr.Checkbox(
                label="Use quality tags",
                value=config.get("wd14_caption.use_quality_tags", False),
                interactive=True,
                info="Adds quality tags as the first tag",
            )
            use_quality_tags_as_last_tag = gr.Checkbox(
                label="Use quality tags as last tag",
                value=config.get("wd14_caption.use_quality_tags_as_last_tag", False),
                interactive=True,
                info="Adds quality tags as the last tag",
            )
            character_tags_first = gr.Checkbox(
                label="Character tags first",
                value=config.get("wd14_caption.character_tags_first", False),
                interactive=True,
                info="Always insert character tags before general tags",
            )

        with gr.Row():
            recursive = gr.Checkbox(
                label="Recursive",
                value=config.get("wd14_caption.recursive", False),
                info="Tag subfolders images as well",
            )
            remove_underscore = gr.Checkbox(
                label="Remove underscore",
                value=config.get("wd14_caption.remove_underscore", True),
                info="replace underscores with spaces in the output tags",
            )

            debug = gr.Checkbox(
                label="Debug",
                value=config.get("wd14_caption.debug", True),
                info="Debug mode",
            )
            frequency_tags = gr.Checkbox(
                label="Show tags frequency",
                value=config.get("wd14_caption.frequency_tags", True),
                info="Show frequency of tags for images.",
            )

        with gr.Row():
            thresh = gr.Slider(
                value=config.get("wd14_caption.thresh", 0.35),
                label="Threshold",
                info="threshold of confidence to add a tag",
                minimum=0,
                maximum=1,
                step=0.05,
            )

            general_threshold = gr.Slider(
                value=config.get("wd14_caption.general_threshold", 0.35),
                label="General threshold",
                info="Adjust `general_threshold` for pruning tags (less tags, less flexible)",
                minimum=0,
                maximum=1,
                step=0.05,
            )
            character_threshold = gr.Slider(
                value=config.get("wd14_caption.character_threshold", 0.35),
                label="Character threshold",
                minimum=0,
                maximum=1,
                step=0.05,
            )

        # Advanced Settings
        with gr.Row():
            batch_size = gr.Number(
                value=config.get("wd14_caption.batch_size", 1),
                label="Batch size",
                interactive=True,
            )

            max_data_loader_n_workers = gr.Number(
                value=config.get("wd14_caption.max_data_loader_n_workers", 2),
                label="Max dataloader workers",
                interactive=True,
            )

        def repo_id_changes(repo_id, onnx):
            global old_onnx_value

            if "-v3" in repo_id:
                old_onnx_value = onnx
                return gr.Checkbox(value=True, interactive=False)
            else:
                return gr.Checkbox(value=old_onnx_value, interactive=True)

        repo_id.change(repo_id_changes, inputs=[repo_id, onnx], outputs=[onnx])

        caption_button = gr.Button("Caption images")

        caption_button.click(
            caption_images,
            inputs=[
                train_data_dir,
                caption_extension,
                batch_size,
                general_threshold,
                character_threshold,
                repo_id,
                recursive,
                max_data_loader_n_workers,
                debug,
                undesired_tags,
                frequency_tags,
                always_first_tags,
                caption_postfix,
                onnx,
                append_tags,
                force_download,
                caption_separator,
                tag_replacement,
                character_tag_expand,
                use_rating_tags,
                use_rating_tags_as_last_tag,
                use_quality_tags,
                use_quality_tags_as_last_tag,
                character_tags_first,
                remove_underscore,
                thresh,
            ],
            show_progress=False,
        )

        train_data_dir.change(
            fn=lambda path: gr.Dropdown(choices=[""] + list_train_dirs(path)),
            inputs=train_data_dir,
            outputs=train_data_dir,
            show_progress=False,
        )
