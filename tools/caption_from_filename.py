# Proposed by https://github.com/kainatquaderee
import os
import argparse
import logging
from pathlib import Path

def is_image_file(filename, image_extensions):
    """Check if a file is an image file based on its extension."""
    return Path(filename).suffix.lower() in image_extensions

def create_text_file(image_filename, output_directory, text_extension):
    """Create a text file with the same name as the image file."""
    # Extract prompt from filename
    prompt = Path(image_filename).stem

    # Construct path for the output text file
    text_file_path = Path(output_directory) / (prompt + text_extension)
    try:

        # Write prompt to text file
        with open(text_file_path, 'w') as text_file:
            text_file.write(prompt)

        logging.info(f"Text file created: {text_file_path}")

        return 1

    except IOError as e:
        logging.error(f"Failed to write to {text_file_path}: {e}")
        return 0

def main(image_directory, output_directory, image_extension, text_extension):
    # If no output directory is provided, use the image directory
    if not output_directory:
        output_directory = image_directory

    # Ensure the output directory exists, create it if necessary
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Initialize a counter for the number of text files created
    text_files_created = 0

    # Iterate through files in the directory
    for image_filename in Path(image_directory).iterdir():
        # Check if the file is an image
        if is_image_file(image_filename, image_extension):
            # Create a text file with the same name as the image file and increment the counter if successful
            text_files_created += create_text_file(image_filename, output_directory, text_extension)

    # Report if no text files were created
    if text_files_created == 0:
        logging.info("No image matching extensions were found in the specified directory. No caption files were created.")
    else:
        logging.info(f"{text_files_created} text files created successfully.")

def create_gui(image_directory, output_directory, image_extension, text_extension):
    try:
        import gradio
        import gradio.blocks as blocks
    except ImportError:
        print("gradio module is not installed. Please install it to use the GUI.")
        exit(1)
    
    """Create a Gradio interface for the caption creation process."""
    with gradio.Blocks() as demo:
        gradio.Markdown("## Caption From Filename")
        with gradio.Row():
            with gradio.Column():
                image_dir = gradio.Textbox(label="Image Directory", value=image_directory)
                output_dir = gradio.Textbox(label="Output Directory", value=output_directory)
                image_ext = gradio.Textbox(label="Image Extensions", value=" ".join(image_extension))
                text_ext = gradio.Textbox(label="Text Extension", value=text_extension)
                run_button = gradio.Button("Run")
            with gradio.Column():
                output = gradio.Textbox(label="Output", placeholder="Output will be displayed here...", lines=10, max_lines=10)
        run_button.click(main, inputs=[image_dir, output_dir, image_ext, text_ext], outputs=output)
    demo.launch()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Generate caption files from image filenames.')

    # Add arguments for the image directory, output directory, and file extension
    parser.add_argument('image_directory', help='Directory containing the image files.')
    parser.add_argument('--output_directory', help='Optional: Output directory where text files will be saved. If not provided, the files will be saved in the same directory as the images.')
    parser.add_argument('--image_extension', nargs='+', default=['.jpg', '.jpeg', '.png', '.webp', '.bmp'], help='Extension(s) for the image files. Defaults to common image extensions .jpg, .jpeg, .png, .webp, .bmp.')
    parser.add_argument('--text_extension', default='.txt', help='Extension for the output text files. Defaults to .txt.')
    parser.add_argument('--gui', action='store_true', help='Launch a Gradio interface for the caption creation process.')

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.gui:
        create_gui(args.image_directory, args.output_directory, args.image_extension, args.text_extension)
    else:
        main(args.image_directory, args.output_directory, args.image_extension, args.text_extension)
