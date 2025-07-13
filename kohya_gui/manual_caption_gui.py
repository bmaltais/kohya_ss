import gradio as gr
from easygui import msgbox, boolbox
from .common_gui import get_folder_path, scriptdir, list_dirs
from math import ceil
import os
import re

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

IMAGES_TO_SHOW = 5
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
auto_save = True


def _get_caption_path(image_file, images_dir, caption_ext):
    """
    Returns the expected path of a caption file for a given image path
    """
    if not image_file:
        return None
    base_name = os.path.basename(image_file)
    caption_file_name = os.path.splitext(base_name)[0] + caption_ext
    caption_file_path = os.path.join(images_dir, caption_file_name)
    return caption_file_path


def _get_quick_tags(quick_tags_text):
    """
    Gets a list of tags from the quick tags text box
    """
    quick_tags = [t.strip() for t in quick_tags_text.split(",") if t.strip()]
    quick_tags_set = set(tag.lower() for tag in quick_tags)
    return quick_tags, quick_tags_set


def _get_tag_checkbox_updates(caption, quick_tags, quick_tags_set):
    """
    Updates a list of caption checkboxes to show possible tags and tags
    already included in the caption
    """
    caption_tags_have = [c.strip() for c in caption.split(",") if c.strip()]
    caption_tags_unique = [t for t in caption_tags_have if t.lower() not in quick_tags_set]
    caption_tags_all = quick_tags + caption_tags_unique
    return gr.CheckboxGroup(choices=caption_tags_all, value=caption_tags_have)


def paginate_go(page, max_page):
    try:
        page_num = int(page)
        return paginate(page_num, max_page, 0)
    except (ValueError, TypeError):
        gr.Warning(f"Invalid page number: {page}")
        return gr.update()


def paginate(page, max_page, page_change):
    return int(max(min(int(page) + page_change, max_page), 1))


def save_caption(caption, caption_ext, image_file, images_dir):
    caption_path = _get_caption_path(image_file, images_dir, caption_ext)
    if caption_path:
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption)
        log.info(f"Wrote captions to {caption_path}")
        return gr.Markdown(f"💾 Caption saved to `{caption_path}`", visible=True)
    return gr.Markdown(visible=False)


def update_quick_tags(quick_tags_text, *image_caption_texts):
    quick_tags, quick_tags_set = _get_quick_tags(quick_tags_text)
    return [
        _get_tag_checkbox_updates(caption, quick_tags, quick_tags_set)
        for caption in image_caption_texts
    ]


def update_image_caption(
    quick_tags_text, caption, image_file, images_dir, caption_ext, auto_save_is_checked
):
    if auto_save_is_checked:
        save_caption(caption, caption_ext, image_file, images_dir)

    quick_tags, quick_tags_set = _get_quick_tags(quick_tags_text)
    return _get_tag_checkbox_updates(caption, quick_tags, quick_tags_set)


def update_image_tags(
    quick_tags_text,
    selected_tags,
    image_file,
    images_dir,
    caption_ext,
    auto_save_is_checked,
):
    # Try to determine order by quick tags
    quick_tags, quick_tags_set = _get_quick_tags(quick_tags_text)
    selected_tags_set = set(selected_tags)

    output_tags = [t for t in quick_tags if t in selected_tags_set] + [
        t for t in selected_tags if t not in quick_tags_set
    ]
    caption = ", ".join(output_tags)

    if auto_save_is_checked:
        save_caption(caption, caption_ext, image_file, images_dir)

    return caption


def import_tags_from_captions(
    images_dir, caption_ext, quick_tags_text, ignore_load_tags_word_count
):
    if not images_dir or not os.path.exists(images_dir):
        gr.Warning("Image folder is not set or does not exist. Please load images first.")
        return gr.update()

    if not caption_ext:
        gr.Warning("Please provide an extension for the caption files.")
        return gr.update()

    if quick_tags_text:
        if not boolbox("Are you sure you wish to overwrite the current quick tags?", choices=("Yes", "No")):
            return gr.update()

    tags = []
    tags_set = set()
    for entry in os.scandir(images_dir):
        if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS):
            caption_file_path = _get_caption_path(entry.name, images_dir, caption_ext)
            if os.path.exists(caption_file_path):
                with open(caption_file_path, "r", encoding="utf-8") as f:
                    caption = f.read()
                    for tag in caption.split(","):
                        tag = tag.strip()
                        if not tag:
                            continue
                        tag_key = tag.lower()
                        if tag_key not in tags_set:
                            if tag.count(" ") + 1 <= ignore_load_tags_word_count:
                                tags.append(tag)
                                tags_set.add(tag_key)

    gr.Info(f"Imported {len(tags)} unique tags.")
    return ", ".join(sorted(tags, key=str.lower))


def load_images(images_dir, caption_ext):
    def error_message(msg):
        gr.Warning(msg)
        return [None, None, 1, 1, gr.Markdown(f"⚠️ {msg}", visible=True)]

    if not images_dir or not os.path.exists(images_dir):
        return error_message("Image folder is missing or does not exist.")
    if not caption_ext:
        return error_message("Please provide an extension for the caption files.")

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTENSIONS)])

    if not image_files:
        return error_message("No images found in the folder.")

    total_images = len(image_files)
    max_pages = ceil(total_images / IMAGES_TO_SHOW)

    info = f"✅ Loaded {total_images} images. {max_pages} pages total."
    gr.Info(info)

    return [image_files, images_dir, 1, max_pages, gr.Markdown(info, visible=True)]


def update_images(
    image_files,
    images_dir,
    caption_ext,
    quick_tags_text,
    page,
):
    if not image_files or not images_dir:
        empty_row = gr.Row(visible=False)
        return [empty_row] * (IMAGES_TO_SHOW * 4 + 2)

    quick_tags, quick_tags_set = _get_quick_tags(quick_tags_text or "")

    outputs = []
    start_index = (int(page) - 1) * IMAGES_TO_SHOW

    rows_update, files_update, paths_update, captions, tags_checks = [], [], [], [], []

    for i in range(IMAGES_TO_SHOW):
        image_index = start_index + i
        is_visible = image_index < len(image_files)

        rows_update.append(gr.Row(visible=is_visible))

        image_file, image_path, caption = None, None, ""
        if is_visible:
            image_file = image_files[image_index]
            image_path = os.path.join(images_dir, image_file)
            caption_file_path = _get_caption_path(image_file, images_dir, caption_ext)
            if caption_file_path and os.path.exists(caption_file_path):
                with open(caption_file_path, "r", encoding="utf-8") as f:
                    caption = f.read()

        files_update.append(image_file)
        paths_update.append(image_path)
        captions.append(caption)
        tags_checks.append(_get_tag_checkbox_updates(caption, quick_tags, quick_tags_set))

    outputs.extend(rows_update)
    outputs.extend(files_update)
    outputs.extend(paths_update)
    outputs.extend(captions)
    outputs.extend(tags_checks)
    outputs.extend([gr.Row(visible=True), gr.Row(visible=True)])

    return outputs


# Gradio UI
def gradio_manual_caption_gui_tab(headless=False, default_images_dir=None):
    from .common_gui import create_refresh_button

    default_images_dir = default_images_dir or os.path.join(scriptdir, "data")

    def update_dir_list(path):
        return gr.Dropdown(choices=[""] + list(list_dirs(path)))

    def render_pagination_with_logic(page, max_page):
        with gr.Row(visible=False) as pagination_row:
            gr.Button("◀ Prev").click(paginate, inputs=[page, max_page, gr.Number(-1, visible=False)], outputs=[page])
            page_count = gr.Text("Page 1 / 1", show_label=False, interactive=False, text_align="center")
            page_goto_text = gr.Textbox(show_label=False, placeholder="Go to page...", container=False, scale=1)
            gr.Button("Next ▶").click(paginate, inputs=[page, max_page, gr.Number(1, visible=False)], outputs=[page])
            page_goto_text.submit(paginate_go, inputs=[page_goto_text, max_page], outputs=[page])
        return pagination_row, page_count

    with gr.Tab("Manual Captioning"):
        gr.Markdown("This utility allows quick captioning and tagging of images.")

        image_files_state = gr.State([])

        info_box = gr.Markdown(visible=False)
        page = gr.State(value=1)
        max_page = gr.Number(value=1, visible=False)
        loaded_images_dir = gr.Text(visible=False)

        with gr.Group():
            with gr.Row():
                images_dir = gr.Dropdown(label="Image folder to caption", choices=[""] + list(list_dirs(default_images_dir)), value="", interactive=True, allow_custom_value=True)
                create_refresh_button(images_dir, lambda: None, lambda: {"choices": list(list_dirs(images_dir.value or default_images_dir))}, "open_folder_small")
                gr.Button("📂", elem_id="open_folder_small", elem_classes=["tool"], visible=not headless).click(get_folder_path, outputs=images_dir, show_progress=False)


            with gr.Row():
                caption_ext = gr.Dropdown(label="Caption file extension", choices=[".cap", ".caption", ".txt"], value=".txt", interactive=True, allow_custom_value=True)
                auto_save = gr.Checkbox(label="Autosave", value=True, interactive=True)
                
            with gr.Row():
                load_images_button = gr.Button("Load Images", variant="primary")

            images_dir.change(update_dir_list, inputs=images_dir, outputs=images_dir, show_progress=False)

        with gr.Group():
            quick_tags_text = gr.Textbox(label="Quick Tags", placeholder="Comma separated list of tags", interactive=True)
            with gr.Row():
                ignore_load_tags_word_count = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Ignore Imported Tags Above Word Count", interactive=True, scale=2)
            
            with gr.Row():
                import_tags_button = gr.Button("Import tags from captions", scale=1)

        pagination_row1, page_count1 = render_pagination_with_logic(page, max_page)

        image_rows, image_files, image_images, image_caption_texts, image_tag_checks, save_buttons = [], [], [], [], [], []
        for i in range(IMAGES_TO_SHOW):
            with gr.Row(visible=False) as row:
                image_file = gr.Text(visible=False)
                with gr.Column():
                    image_image = gr.Image(type="filepath", label=f"Image {i+1}")
                with gr.Column(scale=2):
                    image_caption_text = gr.TextArea(label="Captions", placeholder="Input captions for image", interactive=True)
                    tag_checkboxes = gr.CheckboxGroup([], label="Tags", interactive=True)
                with gr.Column(min_width=40):
                    save_button = gr.Button("💾", elem_id="save_button", visible=False)

                image_rows.append(row)
                image_files.append(image_file)
                image_images.append(image_image)
                image_caption_texts.append(image_caption_text)
                image_tag_checks.append(tag_checkboxes)
                save_buttons.append(save_button)

                image_caption_text.input(update_image_caption, inputs=[quick_tags_text, image_caption_text, image_file, loaded_images_dir, caption_ext, auto_save], outputs=tag_checkboxes)
                tag_checkboxes.input(update_image_tags, inputs=[quick_tags_text, tag_checkboxes, image_file, loaded_images_dir, caption_ext, auto_save], outputs=[image_caption_text])
                save_button.click(save_caption, inputs=[image_caption_text, caption_ext, image_file, loaded_images_dir], outputs=info_box)

        pagination_row2, page_count2 = render_pagination_with_logic(page, max_page)

        quick_tags_text.change(update_quick_tags, inputs=[quick_tags_text] + image_caption_texts, outputs=image_tag_checks)
        import_tags_button.click(import_tags_from_captions, inputs=[loaded_images_dir, caption_ext, quick_tags_text, ignore_load_tags_word_count], outputs=quick_tags_text)

        load_images_outputs = [image_files_state, loaded_images_dir, page, max_page, info_box]
        load_images_button.click(load_images, inputs=[images_dir, caption_ext], outputs=load_images_outputs)

        update_trigger_inputs = [image_files_state, loaded_images_dir, caption_ext, quick_tags_text, page]
        update_outputs = (
            image_rows + image_files + image_images +
            image_caption_texts + image_tag_checks + [pagination_row1, pagination_row2]
        )

        page.change(update_images, inputs=update_trigger_inputs, outputs=update_outputs, show_progress=False)
        image_files_state.change(update_images, inputs=update_trigger_inputs, outputs=update_outputs, show_progress=False)

        auto_save.change(lambda is_auto_save: [gr.Button(visible=not is_auto_save)] * IMAGES_TO_SHOW, inputs=auto_save, outputs=save_buttons)
        page.change(lambda p, m: (f"Page {int(p)} / {int(m)}", f"Page {int(p)} / {int(m)}"), inputs=[page, max_page], outputs=[page_count1, page_count2], show_progress=False)
        image_files_state.change(lambda p, m: (f"Page {int(p)} / {int(m)}", f"Page {int(p)} / {int(m)}"), inputs=[page, max_page], outputs=[page_count1, page_count2], show_progress=False)
        max_page.change(lambda p, m: (f"Page {int(p)} / {int(m)}", f"Page {int(p)} / {int(m)}"), inputs=[page, max_page], outputs=[page_count1, page_count2], show_progress=False)
