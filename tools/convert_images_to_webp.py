import argparse
import glob
import os
from pathlib import Path
from PIL import Image


def main():
    # Define the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str,
                        help="the directory containing the images to be converted")
    parser.add_argument("--in_ext", type=str, default="webp",
                        help="the input file extension")
    parser.add_argument("--delete_originals", action="store_true",
                        help="whether to delete the original files after conversion")

    # Parse the command-line arguments
    args = parser.parse_args()
    directory = args.directory
    in_ext = args.in_ext
    delete_originals = args.delete_originals

    # Set the output file extension to .webp
    out_ext = "webp"

    # Create the file pattern string using the input file extension
    file_pattern = f"*.{in_ext}"

    # Get the list of files in the directory that match the file pattern
    files = glob.glob(os.path.join(directory, file_pattern))

    # Iterate over the list of files
    for file in files:
        # Open the image file
        img = Image.open(file)

        # Create a new file path with the output file extension
        new_path = Path(file).with_suffix(f".{out_ext}")
        print(new_path)

        # Check if the output file already exists
        if new_path.exists():
            # Skip the conversion if the output file already exists
            print(f"Skipping {file} because {new_path} already exists")
            continue

        # Save the image to the new file as lossless
        img.save(new_path, lossless=True)

        # Optionally, delete the original file
        if delete_originals:
            os.remove(file)


if __name__ == "__main__":
    main()
