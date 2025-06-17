import gradio as gr
import subprocess
from .common_gui import (
    get_folder_path,
    add_pre_postfix,
    scriptdir,
    list_dirs,
    get_executable_path, setup_environment,
)
from .class_gui_config import KohyaSSGUIConfig
import os

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()
old_onnx_value = True


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
    onnx: bool,
    append_tags: bool,
    force_download: bool,
    caption_separator: str,
    tag_replacement: bool,
    character_tag_expand: str,
    use_rating_tags: bool,
    use_rating_tags_as_last_tag: bool,
    remove_underscore: bool,
    thresh: float,
) -> None:
    # Check for images_dir_input
    if train_data_dir == "":
        log.info("Image folder is missing...")
        return

    if caption_extension == "":
        log.info("Please provide an extension for the caption files.")
        return

    repo_id_converted = repo_id.replace("/", "_")
    if not os.path.exists(f"./wd14_tagger_model/{repo_id_converted}"):
        force_download = True

    log.info(f"Captioning files in {train_data_dir}...")
    run_cmd = [
        rf'{get_executable_path("accelerate")}',
        "launch",
        rf"{scriptdir}/sd-scripts/finetune/tag_images_by_wd14_tagger.py",
    ]

    # Uncomment and modify if needed
    # if always_first_tags != "":
    #     run_cmd.append('--always_first_tags')
    #     run_cmd.append(always_first_tags)

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

    # Add the directory containing the training data
    run_cmd.append(rf"{train_data_dir}")

    env = setup_environment()

    # Reconstruct the safe command string for display
    command_to_run = " ".join(run_cmd)
    log.info(f"Executing command: {command_to_run}")

    # Run the command in the sd-scripts folder context
    subprocess.run(run_cmd, env=env)

    # Add prefix and postfix
    add_pre_postfix(
        folder=train_data_dir,
        caption_file_ext=caption_extension,
        prefix=always_first_tags,
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
            "This utility will use WD14 to caption files for each images in a folder."
        )

        # Input Settings
        # with gr.Section('Input Settings'):
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
                choices=[
                    "SmilingWolf/wd-v1-4-convnext-tagger-v2",
                    "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
                    "SmilingWolf/wd-v1-4-vit-tagger-v2",
                    "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
                    "SmilingWolf/wd-v1-4-moat-tagger-v2",
                    "SmilingWolf/wd-swinv2-tagger-v3",
                    "SmilingWolf/wd-vit-tagger-v3",
                    "SmilingWolf/wd-convnext-tagger-v3",
                ],
                value=config.get(
                    "wd14_caption.repo_id", "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
                ),
                show_label="Repo id for wd14 tagger on Hugging Face",
            )

            force_download = gr.Checkbox(
                label="Force model re-download",
                value=config.get("wd14_caption.force_download", False),
                info="Useful to force model re download when switching to onnx",
            )

        with gr.Row():

            caption_extension = gr.Dropdown(
                label="Caption file extension",
                choices=[".cap", ".caption", ".txt"],
                value=".txt",
                interactive=True,
                allow_custom_value=True,
            )

            caption_separator = gr.Textbox(
                label="Caption Separator",
                value=config.get("wd14_caption.caption_separator", ", "),
                interactive=True,
            )

        with gr.Row():

            tag_replacement = gr.Textbox(
                label="Tag replacement",
                info="tag replacement in the format of `source1,target1;source2,target2; ...`. Escape `,` and `;` with `\`. e.g. `tag1,tag2;tag3,tag4`",
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
                info="comma-separated list of tags to always put at the beginning, e.g.: 1girl, 1boy, ",
                placeholder="(Optional)",
                interactive=True,
                value=config.get("wd14_caption.always_first_tags", ""),
            )

        with gr.Row():
            onnx = gr.Checkbox(
                label="Use onnx",
                value=config.get("wd14_caption.onnx", True),
                interactive=True,
                info="https://github.com/onnx/onnx",
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
                onnx,
                append_tags,
                force_download,
                caption_separator,
                tag_replacement,
                character_tag_expand,
                use_rating_tags,
                use_rating_tags_as_last_tag,
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
