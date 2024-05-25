from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import gradio as gr
import os

from .common_gui import get_folder_path, scriptdir, list_dirs
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


def load_model():
    # Set the device to GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the BLIP2 processor
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    # Initialize the BLIP2 model
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )

    # Move the model to the specified device
    model.to(device)

    return processor, model, device


def get_images_in_directory(directory_path):
    """
    Returns a list of image file paths found in the provided directory path.

    Parameters:
    - directory_path: A string representing the path to the directory to search for images.

    Returns:
    - A list of strings, where each string is the full path to an image file found in the specified directory.
    """
    import os

    # List of common image file extensions to look for
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]

    # Generate a list of image file paths in the directory
    image_files = [
        # constructs the full path to the file
        os.path.join(directory_path, file)
        # lists all files and directories in the given path
        for file in os.listdir(directory_path)
        # gets the file extension in lowercase
        if os.path.splitext(file)[1].lower() in image_extensions
    ]

    # Return the list of image file paths
    return image_files


def generate_caption(
    file_list,
    processor,
    model,
    device,
    caption_file_ext=".txt",
    num_beams=5,
    repetition_penalty=1.5,
    length_penalty=1.2,
    max_new_tokens=40,
    min_new_tokens=20,
    do_sample=True,
    temperature=1.0,
    top_p=0.0,
):
    """
    Fetches and processes each image in file_list, generates captions based on the image, and writes the generated captions to a file.

    Parameters:
    - file_list: A list of file paths pointing to the images to be captioned.
    - processor: The preprocessor for the BLIP2 model.
    - model: The BLIP2 model to be used for generating captions.
    - device: The device on which the computation is performed.
    - extension: The extension for the output text files.
    - num_beams: Number of beams for beam search. Default: 5.
    - repetition_penalty: Penalty for repeating tokens. Default: 1.5.
    - length_penalty: Penalty for sentence length. Default: 1.2.
    - max_new_tokens: Maximum number of new tokens to generate. Default: 40.
    - min_new_tokens: Minimum number of new tokens to generate. Default: 20.
    """
    for file_path in file_list:
        image = Image.open(file_path)

        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        if top_p == 0.0:
            generated_ids = model.generate(
                **inputs,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
            )
        else:
            generated_ids = model.generate(
                **inputs,
                do_sample=do_sample,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
            )

        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # Construct the output file path by replacing the original file extension with the specified extension
        output_file_path = os.path.splitext(file_path)[0] + caption_file_ext

        # Write the generated text to the output file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(generated_text)

        # Log the image file path with a message about the fact that the caption was generated
        log.info(f"{file_path} caption was generated")


def caption_images_beam_search(
    directory_path,
    num_beams,
    repetition_penalty,
    length_penalty,
    min_new_tokens,
    max_new_tokens,
    caption_file_ext,
):
    """
    Captions all images in the specified directory using the provided prompt.

    Parameters:
    - directory_path: A string representing the path to the directory containing the images to be captioned.
    """
    log.info("BLIP2 captionning beam...")

    if not os.path.isdir(directory_path):
        log.error(f"Directory {directory_path} does not exist.")
        return

    processor, model, device = load_model()
    image_files = get_images_in_directory(directory_path)
    generate_caption(
        file_list=image_files,
        processor=processor,
        model=model,
        device=device,
        num_beams=int(num_beams),
        repetition_penalty=float(repetition_penalty),
        length_penalty=length_penalty,
        min_new_tokens=int(min_new_tokens),
        max_new_tokens=int(max_new_tokens),
        caption_file_ext=caption_file_ext,
    )


def caption_images_nucleus(
    directory_path,
    do_sample,
    temperature,
    top_p,
    min_new_tokens,
    max_new_tokens,
    caption_file_ext,
):
    """
    Captions all images in the specified directory using the provided prompt.

    Parameters:
    - directory_path: A string representing the path to the directory containing the images to be captioned.
    """
    log.info("BLIP2 captionning nucleus...")

    if not os.path.isdir(directory_path):
        log.error(f"Directory {directory_path} does not exist.")
        return

    processor, model, device = load_model()
    image_files = get_images_in_directory(directory_path)
    generate_caption(
        file_list=image_files,
        processor=processor,
        model=model,
        device=device,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        min_new_tokens=int(min_new_tokens),
        max_new_tokens=int(max_new_tokens),
        caption_file_ext=caption_file_ext,
    )


def gradio_blip2_caption_gui_tab(headless=False, directory_path=None):
    from .common_gui import create_refresh_button

    directory_path = (
        directory_path
        if directory_path is not None
        else os.path.join(scriptdir, "data")
    )
    current_train_dir = directory_path

    def list_train_dirs(path):
        nonlocal current_train_dir
        current_train_dir = path
        return list(list_dirs(path))

    with gr.Tab("BLIP2 Captioning"):
        gr.Markdown(
            "This utility uses BLIP2 to caption files for each image in a folder."
        )

        with gr.Group(), gr.Row():
            directory_path_dir = gr.Dropdown(
                label="Image folder to caption (containing the images to caption)",
                choices=[""] + list_train_dirs(directory_path),
                value="",
                interactive=True,
                allow_custom_value=True,
            )
            create_refresh_button(
                directory_path_dir,
                lambda: None,
                lambda: {"choices": list_train_dirs(current_train_dir)},
                "open_folder_small",
            )
            button_directory_path_dir_input = gr.Button(
                "📂",
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            button_directory_path_dir_input.click(
                get_folder_path,
                outputs=directory_path_dir,
                show_progress=False,
            )
        with gr.Group(), gr.Row():
            min_new_tokens = gr.Number(
                value=20,
                label="Min new tokens",
                interactive=True,
                step=1,
                minimum=5,
                maximum=300,
            )
            max_new_tokens = gr.Number(
                value=40,
                label="Max new tokens",
                interactive=True,
                step=1,
                minimum=5,
                maximum=300,
            )
            caption_file_ext = gr.Textbox(
                label="Caption file extension",
                placeholder="Extension for caption file (e.g., .caption, .txt)",
                value=".txt",
                interactive=True,
            )

        with gr.Row():
            with gr.Tab("Beam search"):
                with gr.Row():
                    num_beams = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=16,
                        step=1,
                        interactive=True,
                        label="Number of beams",
                    )

                    len_penalty = gr.Slider(
                        minimum=-1.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.2,
                        interactive=True,
                        label="Length Penalty",
                        info="increase for longer sequence",
                    )

                    rep_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=5.0,
                        value=1.5,
                        step=0.5,
                        interactive=True,
                        label="Repeat Penalty",
                        info="larger value prevents repetition",
                    )

                caption_button_beam = gr.Button(
                    value="Caption images", interactive=True, variant="primary"
                )
                caption_button_beam.click(
                    caption_images_beam_search,
                    inputs=[
                        directory_path_dir,
                        num_beams,
                        rep_penalty,
                        len_penalty,
                        min_new_tokens,
                        max_new_tokens,
                        caption_file_ext,
                    ],
                )
            with gr.Tab("Nucleus sampling"):
                with gr.Row():
                    do_sample = gr.Checkbox(label="Sample", value=True)
                    
                    temperature = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=1.0,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                        info="used with nucleus sampling",
                    )

                    top_p = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.9,
                        step=0.1,
                        interactive=True,
                        label="Top_p",
                    )

                caption_button_nucleus = gr.Button(
                    value="Caption images", interactive=True, variant="primary"
                )
                caption_button_nucleus.click(
                    caption_images_nucleus,
                    inputs=[
                        directory_path_dir,
                        do_sample,
                        temperature,
                        top_p,
                        min_new_tokens,
                        max_new_tokens,
                        caption_file_ext,
                    ],
                )
