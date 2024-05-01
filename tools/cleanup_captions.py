import os
import argparse
import logging
from pathlib import Path

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
    
def main(folder_path:Path, extension:str, keywords:set=None):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(extension):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "r") as f:
                    text = f.read()
                # extract tags from text and split into a list using comma as the delimiter
                tags = [tag.strip() for tag in text.split(",")]
                # remove the specified keywords from the tags list
                if keywords:
                    tags = [tag for tag in tags if tag not in keywords]
                # remove empty or whitespace-only tags
                tags = [tag for tag in tags if tag.strip() != ""]
                # join the tags back into a comma-separated string and write back to the file
                with open(file_path, "w") as f:
                    f.write(", ".join(tags))
                logging.info(f"Processed {file_name}")
            except Exception as e:
                logging.error(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Remove specified keywords from all text files in a directory.")
    parser.add_argument("folder_path", type=writable_dir, help="path to directory containing text files")
    parser.add_argument("-e", "--extension", type=str, default=".txt", help="file extension of text files to be processed (default: .txt)")
    parser.add_argument("-k", "--keywords", type=str, nargs="*", help="Optional: list of keywords to be removed from text files. If not provided, the default list will be used.")
    args = parser.parse_args()

    folder_path = args.folder_path
    extension = args.extension
    keywords = set(args.keywords) if args.keywords else set(["1girl", "solo", "blue eyes", "brown eyes", "blonde hair", "black hair", "realistic", "red lips", "lips", "artist name", "makeup", "realistic","brown hair", "dark skin", 
                "dark-skinned female", "medium breasts", "breasts", "1boy"])

    main(folder_path, extension, keywords)
