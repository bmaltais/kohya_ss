import gradio as gr
from easygui import msgbox
import subprocess
from .common_gui import get_folder_path, add_pre_postfix, scriptdir, list_dirs
import os

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


def caption_images(
    train_data_dir,
    caption_extension,
    batch_size,
    general_threshold,
    character_threshold,
    replace_underscores,
    model,
    recursive,
    max_data_loader_n_workers,
    debug,
    undesired_tags,
    frequency_tags,
    prefix,
    postfix,
    onnx,
    append_tags,
    force_download,
    caption_separator
):
    # Check for images_dir_input
    if train_data_dir == '':
        msgbox('Image folder is missing...')
        return

    if caption_extension == '':
        msgbox('Please provide an extension for the caption files.')
        return

    log.info(f'Captioning files in {train_data_dir}...')
    run_cmd = fr'accelerate launch "{scriptdir}/sd-scripts/finetune/tag_images_by_wd14_tagger.py"'
    run_cmd += f' --batch_size={int(batch_size)}'
    run_cmd += f' --general_threshold={general_threshold}'
    run_cmd += f' --character_threshold={character_threshold}'
    run_cmd += f' --caption_extension="{caption_extension}"'
    run_cmd += f' --caption_separator="{caption_separator}"'
    run_cmd += f' --model="{model}"'
    run_cmd += (
        f' --max_data_loader_n_workers="{int(max_data_loader_n_workers)}"'
    )

    if recursive:
        run_cmd += f' --recursive'
    if debug:
        run_cmd += f' --debug'
    if replace_underscores:
        run_cmd += f' --remove_underscore'
    if frequency_tags:
        run_cmd += f' --frequency_tags'
    if onnx:
        run_cmd += f' --onnx'
    if append_tags:
        run_cmd += f' --append_tags'
    if force_download:
        run_cmd += f' --force_download'

    if not undesired_tags == '':
        run_cmd += f' --undesired_tags="{undesired_tags}"'
    run_cmd += fr' "{train_data_dir}"'

    log.info(run_cmd)

    env = os.environ.copy()
    env['PYTHONPATH'] = fr"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Run the command
    subprocess.run(run_cmd, shell=True, env=env)

    # Add prefix and postfix
    add_pre_postfix(
        folder=train_data_dir,
        caption_file_ext=caption_extension,
        prefix=prefix,
        postfix=postfix,
    )

    log.info('...captioning done')


###
# Gradio UI
###


def gradio_wd14_caption_gui_tab(headless=False, default_train_dir=None):
    from .common_gui import create_refresh_button

    default_train_dir = default_train_dir if default_train_dir is not None else os.path.join(scriptdir, "data")
    current_train_dir = default_train_dir

    def list_train_dirs(path):
        current_train_dir = path
        return list(list_dirs(path))

    with gr.Tab('WD14 Captioning'):
        gr.Markdown(
            'This utility will use WD14 to caption files for each images in a folder.'
        )

        # Input Settings
        # with gr.Section('Input Settings'):
        with gr.Group(), gr.Row():
            train_data_dir = gr.Dropdown(
                label='Image folder to caption (containing the images to caption)',
                choices=list_train_dirs(default_train_dir),
                value="",
                interactive=True,
                allow_custom_value=True,
            )
            create_refresh_button(train_data_dir, lambda: None, lambda: {"choices": list_train_dir(current_train_dir)},"open_folder_small")
            button_train_data_dir_input = gr.Button(
                '📂', elem_id='open_folder_small', elem_classes=['tool'], visible=(not headless)
            )
            button_train_data_dir_input.click(
                get_folder_path,
                outputs=train_data_dir,
                show_progress=False,
            )

            caption_extension = gr.Textbox(
                label='Caption file extension',
                placeholder='Extention for caption file. eg: .caption, .txt',
                value='.txt',
                interactive=True,
            )

            caption_separator = gr.Textbox(
                label='Caption Separator',
                value=',',
                interactive=True,
            )

        undesired_tags = gr.Textbox(
            label='Undesired tags',
            placeholder='(Optional) Separate `undesired_tags` with comma `(,)` if you want to remove multiple tags, e.g. `1girl,solo,smile`.',
            interactive=True,
        )

        with gr.Row():
            prefix = gr.Textbox(
                label='Prefix to add to WD14 caption',
                placeholder='(Optional)',
                interactive=True,
            )

            postfix = gr.Textbox(
                label='Postfix to add to WD14 caption',
                placeholder='(Optional)',
                interactive=True,
            )

        with gr.Row():
            onnx = gr.Checkbox(
                label='Use onnx',
                value=False,
                interactive=True,
                info="https://github.com/onnx/onnx"
            )
            append_tags = gr.Checkbox(
                label='Append TAGs',
                value=False,
                interactive=True,
                info="This option appends the tags to the existing tags, instead of replacing them."
            )

        with gr.Row():
            replace_underscores = gr.Checkbox(
                label='Replace underscores in filenames with spaces',
                value=True,
                interactive=True,
            )
            recursive = gr.Checkbox(
                label='Recursive',
                value=False,
                info='Tag subfolders images as well',
            )

            debug = gr.Checkbox(
                label='Verbose logging',
                value=True,
                info='Debug while tagging, it will print your image file with general tags and character tags.',
            )
            frequency_tags = gr.Checkbox(
                label='Show tags frequency',
                value=True,
                info='Show frequency of tags for images.',
            )

        # Model Settings
        with gr.Row():
            model = gr.Dropdown(
                label='Model',
                choices=[
                    'SmilingWolf/wd-v1-4-convnext-tagger-v2',
                    'SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
                    'SmilingWolf/wd-v1-4-vit-tagger-v2',
                    'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
                    'SmilingWolf/wd-v1-4-moat-tagger-v2',
                ],
                value='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
            )

            force_download = gr.Checkbox(
                label='Force model re-download',
                value=False,
                info='Useful to force model re download when switching to onnx',
            )

            general_threshold = gr.Slider(
                value=0.35,
                label='General threshold',
                info='Adjust `general_threshold` for pruning tags (less tags, less flexible)',
                minimum=0,
                maximum=1,
                step=0.05,
            )
            character_threshold = gr.Slider(
                value=0.35,
                label='Character threshold',
                info='useful if you want to train with character',
                minimum=0,
                maximum=1,
                step=0.05,
            )

        # Advanced Settings
        with gr.Row():
            batch_size = gr.Number(
                value=8, label='Batch size', interactive=True
            )

            max_data_loader_n_workers = gr.Number(
                value=2, label='Max dataloader workers', interactive=True
            )

        caption_button = gr.Button('Caption images')

        caption_button.click(
            caption_images,
            inputs=[
                train_data_dir,
                caption_extension,
                batch_size,
                general_threshold,
                character_threshold,
                replace_underscores,
                model,
                recursive,
                max_data_loader_n_workers,
                debug,
                undesired_tags,
                frequency_tags,
                prefix,
                postfix,
                onnx,
                append_tags,
                force_download,
                caption_separator
            ],
            show_progress=False,
        )

        train_data_dir.change(
            fn=lambda path: gr.Dropdown().update(choices=list_train_dirs(path)),
            inputs=train_data_dir,
            outputs=train_data_dir,
            show_progress=False,
        )
