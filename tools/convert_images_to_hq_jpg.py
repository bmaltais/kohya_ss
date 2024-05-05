import argparse
import glob
import os
from pathlib import Path
from PIL import Image


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

def main(directory, in_ext, quality, delete_originals):
    out_ext = "jpg"

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

        # Check if the output file already exists
        if new_path.exists():
            # Skip the conversion if the output file already exists
            print(f"Skipping {file} because {new_path} already exists")
            continue

        # Save the image to the new file as high-quality JPEG
        img.save(new_path, quality=quality, optimize=True)

        # Optionally, delete the original file
        if delete_originals:
            os.remove(file)


if __name__ == "__main__":
    # Define the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=writable_dir,
                        help="the directory containing the images to be converted")
    parser.add_argument("--in_ext", type=str, default="webp",
                        help="the input file extension")
    parser.add_argument("--quality", type=int, default=95,
                        help="the JPEG quality (0-100)")
    parser.add_argument("--delete_originals", action="store_true",
                        help="whether to delete the original files after conversion")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    main(directory=args.directory, in_ext=args.in_ext, quality=args.quality, delete_originals=args.delete_originals)
