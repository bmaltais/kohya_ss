import gradio as gr
from easygui import msgbox, boolbox
from .common_gui import get_folder_path
from math import ceil
import os
import re

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

IMAGES_TO_SHOW = 5
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
auto_save = True


def _get_caption_path(image_file, images_dir, caption_ext):
    """
    Returns the expected path of a caption file for a given image path
    """
    caption_file_name = os.path.splitext(image_file)[0] + caption_ext
    caption_file_path = os.path.join(images_dir, caption_file_name)
    return caption_file_path


def _get_quick_tags(quick_tags_text):
    """
    Gets a list of tags from the quick tags text box
    """
    quick_tags = [t.strip() for t in quick_tags_text.split(',') if t.strip()]
    quick_tags_set = set(quick_tags)
    return quick_tags, quick_tags_set


def _get_tag_checkbox_updates(caption, quick_tags, quick_tags_set):
    """
    Updates a list of caption checkboxes to show possible tags and tags
    already included in the caption
    """
    caption_tags_have = [c.strip() for c in caption.split(',') if c.strip()]
    caption_tags_unique = [
        t for t in caption_tags_have if t not in quick_tags_set
    ]
    caption_tags_all = quick_tags + caption_tags_unique
    return gr.CheckboxGroup.update(
        choices=caption_tags_all, value=caption_tags_have
    )


def paginate(page, max_page, page_change):
    return int(max(min(page + page_change, max_page), 1))


def save_caption(caption, caption_ext, image_file, images_dir):
    caption_path = _get_caption_path(image_file, images_dir, caption_ext)
    with open(caption_path, 'w+', encoding='utf8') as f:
        f.write(caption)

    log.info(f'Wrote captions to {caption_path}')


def update_quick_tags(quick_tags_text, *image_caption_texts):
    quick_tags, quick_tags_set = _get_quick_tags(quick_tags_text)
    return [
        _get_tag_checkbox_updates(caption, quick_tags, quick_tags_set)
        for caption in image_caption_texts
    ]


def update_image_caption(
    quick_tags_text, caption, image_file, images_dir, caption_ext, auto_save
):
    if auto_save:
        save_caption(caption, caption_ext, image_file, images_dir)

    quick_tags, quick_tags_set = _get_quick_tags(quick_tags_text)
    return _get_tag_checkbox_updates(caption, quick_tags, quick_tags_set)


def update_image_tags(
    quick_tags_text,
    selected_tags,
    image_file,
    images_dir,
    caption_ext,
    auto_save,
):
    # Try to determine order by quick tags
    quick_tags, quick_tags_set = _get_quick_tags(quick_tags_text)
    selected_tags_set = set(selected_tags)

    output_tags = [t for t in quick_tags if t in selected_tags_set] + [
        t for t in selected_tags if t not in quick_tags_set
    ]
    caption = ', '.join(output_tags)

    if auto_save:
        save_caption(caption, caption_ext, image_file, images_dir)

    return caption


def import_tags_from_captions(
    images_dir, caption_ext, quick_tags_text, ignore_load_tags_word_count
):
    """
    Scans images directory for all available captions and loads all tags
    under a specified word count into the quick tags box
    """

    def empty_return():
        return gr.Text.update()

    # Check for images_dir
    if not images_dir:
        msgbox('Image folder is missing...')
        return empty_return()

    if not os.path.exists(images_dir):
        msgbox('Image folder does not exist...')
        return empty_return()

    if not caption_ext:
        msgbox('Please provide an extension for the caption files.')
        return empty_return()

    if quick_tags_text:
        if not boolbox(
            f'Are you sure you wish to overwrite the current quick tags?',
            choices=('Yes', 'No'),
        ):
            return empty_return()

    images_list = os.listdir(images_dir)
    image_files = [
        f for f in images_list if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    # Use a set for lookup but store order with list
    tags = []
    tags_set = set()
    for image_file in image_files:
        caption_file_path = _get_caption_path(
            image_file, images_dir, caption_ext
        )
        if os.path.exists(caption_file_path):
            with open(caption_file_path, 'r', encoding='utf8') as f:
                caption = f.read()
                for tag in caption.split(','):
                    tag = tag.strip()
                    tag_key = tag.lower()
                    if not tag_key in tags_set:
                        # Ignore extra spaces
                        total_words = len(re.findall(r'\s+', tag)) + 1
                        if total_words <= ignore_load_tags_word_count:
                            tags.append(tag)
                            tags_set.add(tag_key)

    return ', '.join(tags)


def load_images(images_dir, caption_ext, loaded_images_dir, page, max_page):
    """
    Triggered to load a new set of images from the folder to caption
    This loads in the total expected image counts to be used by pagination
    before running update_images
    """

    def empty_return():
        return [loaded_images_dir, page, max_page]

    # Check for images_dir
    if not images_dir:
        msgbox('Image folder is missing...')
        return empty_return()

    if not os.path.exists(images_dir):
        msgbox('Image folder does not exist...')
        return empty_return()

    if not caption_ext:
        msgbox('Please provide an extension for the caption files.')
        return empty_return()

    # Load Images
    images_list = os.listdir(images_dir)
    total_images = len(
        [True for f in images_list if f.lower().endswith(IMAGE_EXTENSIONS)]
    )
    return [images_dir, 1, ceil(total_images / IMAGES_TO_SHOW)]


def update_images(
    images_dir,
    caption_ext,
    quick_tags_text,
    page,
):
    """
    Updates the displayed images and captions from the current page and
    image directory
    """

    # Load Images
    images_list = os.listdir(images_dir)
    image_files = [
        f for f in images_list if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    # Quick tags
    quick_tags, quick_tags_set = _get_quick_tags(quick_tags_text or '')

    # Display Images
    rows = []
    image_paths = []
    captions = []
    tag_checkbox_groups = []

    start_index = (int(page) - 1) * IMAGES_TO_SHOW
    for i in range(IMAGES_TO_SHOW):
        image_index = start_index + i
        show_row = image_index < len(image_files)

        image_path = None
        caption = ''
        tag_checkboxes = None
        if show_row:
            image_file = image_files[image_index]
            image_path = os.path.join(images_dir, image_file)

            caption_file_path = _get_caption_path(
                image_file, images_dir, caption_ext
            )
            if os.path.exists(caption_file_path):
                with open(caption_file_path, 'r', encoding='utf8') as f:
                    caption = f.read()

        tag_checkboxes = _get_tag_checkbox_updates(
            caption, quick_tags, quick_tags_set
        )
        rows.append(gr.Row.update(visible=show_row))
        image_paths.append(image_path)
        captions.append(caption)
        tag_checkbox_groups.append(tag_checkboxes)

    return (
        rows
        + image_paths
        + image_paths
        + captions
        + tag_checkbox_groups
        + [gr.Row.update(visible=True), gr.Row.update(visible=True)]
    )


# Gradio UI
def gradio_manual_caption_gui_tab(headless=False):
    with gr.Tab('Manual Captioning'):
        gr.Markdown(
            'This utility allows quick captioning and tagging of images.'
        )
        page = gr.Number(-1, visible=False)
        max_page = gr.Number(1, visible=False)
        loaded_images_dir = gr.Text(visible=False)
        with gr.Row():
            images_dir = gr.Textbox(
                label='Image folder to caption',
                placeholder='Directory containing the images to caption',
                interactive=True,
            )
            folder_button = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            folder_button.click(
                get_folder_path,
                outputs=images_dir,
                show_progress=False,
            )
            load_images_button = gr.Button('Load 💾', elem_id='open_folder')
            caption_ext = gr.Textbox(
                label='Caption file extension',
                placeholder='Extension for caption file. eg: .caption, .txt',
                value='.txt',
                interactive=True,
            )
            auto_save = gr.Checkbox(
                label='Autosave', info='Options', value=True, interactive=True
            )

        # Caption Section
        with gr.Row():
            quick_tags_text = gr.Textbox(
                label='Quick Tags',
                placeholder='Comma separated list of tags',
                interactive=True,
            )
            import_tags_button = gr.Button('Import 📄', elem_id='open_folder')
            ignore_load_tags_word_count = gr.Slider(
                minimum=1,
                maximum=100,
                value=3,
                step=1,
                label='Ignore Imported Tags Above Word Count',
                interactive=True,
            )

        # Next/Prev section generator
        def render_pagination():
            gr.Button('< Prev', elem_id='open_folder').click(
                paginate,
                inputs=[page, max_page, gr.Number(-1, visible=False)],
                outputs=[page],
            )
            page_count = gr.Label('Page 1', label='Page')
            gr.Button('Next >', elem_id='open_folder').click(
                paginate,
                inputs=[page, max_page, gr.Number(1, visible=False)],
                outputs=[page],
            )
            return page_count

        with gr.Row(visible=False) as pagination_row1:
            page_count1 = render_pagination()

        # Images section
        image_rows = []
        image_files = []
        image_images = []
        image_caption_texts = []
        image_tag_checks = []
        save_buttons = []
        for _ in range(IMAGES_TO_SHOW):
            with gr.Row(visible=False) as row:
                image_file = gr.Text(visible=False)
                image_files.append(image_file)
                image_image = gr.Image(type='filepath')
                image_images.append(image_image)
                image_caption_text = gr.TextArea(
                    label='Captions',
                    placeholder='Input captions',
                    interactive=True,
                )
                image_caption_texts.append(image_caption_text)
                tag_checkboxes = gr.CheckboxGroup(
                    [], label='Tags', interactive=True
                )
                save_button = gr.Button(
                    '💾', elem_id='open_folder_small', visible=False
                )
                save_buttons.append(save_button)

                # Caption text change
                image_caption_text.input(
                    update_image_caption,
                    inputs=[
                        quick_tags_text,
                        image_caption_text,
                        image_file,
                        loaded_images_dir,
                        caption_ext,
                        auto_save,
                    ],
                    outputs=tag_checkboxes,
                )

                # Quick tag check
                tag_checkboxes.input(
                    update_image_tags,
                    inputs=[
                        quick_tags_text,
                        tag_checkboxes,
                        image_file,
                        loaded_images_dir,
                        caption_ext,
                        auto_save,
                    ],
                    outputs=[image_caption_text],
                )

                # Save Button
                save_button.click(
                    save_caption,
                    inputs=[
                        image_caption_text,
                        caption_ext,
                        image_file,
                        images_dir,
                    ],
                )

                image_tag_checks.append(tag_checkboxes)
                image_rows.append(row)

        # Next/Prev Section
        with gr.Row(visible=False) as pagination_row2:
            page_count2 = render_pagination()

        # Quick tag text update
        quick_tags_text.change(
            update_quick_tags,
            inputs=[quick_tags_text] + image_caption_texts,
            outputs=image_tag_checks,
        )

        # Import tags button
        import_tags_button.click(
            import_tags_from_captions,
            inputs=[
                loaded_images_dir,
                caption_ext,
                quick_tags_text,
                ignore_load_tags_word_count,
            ],
            outputs=quick_tags_text,
        )

        # Load Images button
        load_images_button.click(
            load_images,
            inputs=[
                images_dir,
                caption_ext,
                loaded_images_dir,
                page,
                max_page,
            ],
            outputs=[loaded_images_dir, page, max_page],
        )

        # Update images shown when the update key changes
        # This allows us to trigger a change from multiple
        # sources (page, image_dir)
        image_update_key = gr.Text(visible=False)
        image_update_key.change(
            update_images,
            inputs=[loaded_images_dir, caption_ext, quick_tags_text, page],
            outputs=image_rows
            + image_files
            + image_images
            + image_caption_texts
            + image_tag_checks
            + [pagination_row1, pagination_row2],
            show_progress=False,
        )
        # Update the key on page and image dir change
        listener_kwargs = {
            'fn': lambda p, i: f'{p}-{i}',
            'inputs': [page, loaded_images_dir],
            'outputs': image_update_key,
        }
        page.change(**listener_kwargs)
        loaded_images_dir.change(**listener_kwargs)

        # Save buttons visibility
        # (on auto-save on/off)
        auto_save.change(
            lambda auto_save: [gr.Button.update(visible=not auto_save)]
            * IMAGES_TO_SHOW,
            inputs=auto_save,
            outputs=save_buttons,
        )

        # Page Count
        page.change(
            lambda page, max_page: [f'Page {int(page)} / {int(max_page)}'] * 2,
            inputs=[page, max_page],
            outputs=[page_count1, page_count2],
            show_progress=False,
        )
