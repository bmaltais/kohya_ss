# This script will create the caption text files in the specified folder using the specified file pattern and caption text.
#
# eg: python caption.py D:\some\folder\location "*.png, *.jpg, *.webp" "some caption text"

import argparse
# import glob
# import os
from pathlib import Path

def create_caption_files(image_folder: str, file_pattern: str, caption_text: str, caption_file_ext: str, overwrite: bool):
    # Split the file patterns string and strip whitespace from each pattern
    patterns = [pattern.strip() for pattern in file_pattern.split(",")]

    # Create a Path object for the image folder
    folder = Path(image_folder)

    # Iterate over the file patterns
    for pattern in patterns:
        # Use the glob method to match the file patterns
        files = folder.glob(pattern)

        # Iterate over the matched files
        for file in files:
            # Check if a text file with the same name as the current file exists in the folder
            txt_file = file.with_suffix(caption_file_ext)
            if not txt_file.exists() or overwrite:
                # Create a text file with the caption text in the folder, if it does not already exist
                # or if the overwrite argument is True
                with open(txt_file, "w") as f:
                    f.write(caption_text)

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", type=str, help="the folder where the image files are located")
    parser.add_argument("--file_pattern", type=str, default="*.png, *.jpg, *.jpeg, *.webp", help="the pattern to match the image file names")
    parser.add_argument("--caption_file_ext", type=str, default=".caption", help="the caption file extension.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="whether to overwrite existing caption files")

    # Create a mutually exclusive group for the caption_text and caption_file arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--caption_text", type=str, help="the text to include in the caption files")
    group.add_argument("--caption_file", type=argparse.FileType("r"), help="the file containing the text to include in the caption files")

    # Parse the command-line arguments
    args = parser.parse_args()
    image_folder = args.image_folder
    file_pattern = args.file_pattern
    caption_file_ext = args.caption_file_ext
    overwrite = args.overwrite

    # Get the caption text from either the caption_text or caption_file argument
    if args.caption_text:
        caption_text = args.caption_text
    elif args.caption_file:
        caption_text = args.caption_file.read()

    # Create a Path object for the image folder
    folder = Path(image_folder)

    # Check if the image folder exists and is a directory
    if not folder.is_dir():
        raise ValueError(f"{image_folder} is not a valid directory.")
        
    # Create the caption files
    create_caption_files(image_folder, file_pattern, caption_text, caption_file_ext, overwrite)

if __name__ == "__main__":
    main()