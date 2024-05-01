# This script will create the caption text files in the specified folder using the specified file pattern and caption text.
#
# eg: python caption.py D:\some\folder\location "*.png, *.jpg, *.webp" "some caption text"

import argparse
import os
import logging
from pathlib import Path

def create_caption_files(image_folder: Path, file_pattern: str, caption_text: str, caption_file_ext: str, overwrite: bool):
    # Split the file patterns string and remove whitespace from each extension
    patterns = [pattern.strip() for pattern in file_pattern.split(",")]

    # Use the glob method to match the file pattern
    for pattern in patterns:
        files = image_folder.glob(pattern)

        # Iterate over the matched files
        for file in files:
            # Check if a text file with the same name as the current file exists in the folder
            txt_file = file.with_suffix(caption_file_ext)
            if not txt_file.exists() or overwrite:
                txt_file.write_text(caption_text)
                logging.info(f"Caption file created: {txt_file}")
                
def writable_dir(target_path):
    """ Check if a path is a valid directory and that it can be written to. """
    path = Path(target_path)
    if path.is_dir():
        if os.access(path, os.W_OK):
            return path
        else:
            raise argparse.ArgumentTypeError(f"Directory '{path}' is not writable.")
    else:
        raise argparse.ArgumentTypeError(f"Directory '{path}' does not exist.")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", type=writable_dir, help="The folder where the image files are located")
    parser.add_argument("--file_pattern", type=str, default="*.png, *.jpg, *.jpeg, *.webp", help="the pattern to match the image file names")
    parser.add_argument("--caption_file_ext", type=str, default=".caption", help="the caption file extension.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="whether to overwrite existing caption files")

    # Create a mutually exclusive group for the caption_text and caption_file arguments
    caption_group = parser.add_mutually_exclusive_group(required=True)
    caption_group.add_argument("--caption_text", type=str, help="the text to include in the caption files")
    caption_group.add_argument("--caption_file", type=argparse.FileType("r"), help="the file containing the text to include in the caption files")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create the caption files
    create_caption_files(args.image_folder, args.file_pattern, args.caption_text, args.caption_file_ext, args.overwrite)

if __name__ == "__main__":
    main()