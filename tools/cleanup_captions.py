import os
import argparse

parser = argparse.ArgumentParser(description="Remove specified keywords from all text files in a directory.")
parser.add_argument("folder_path", type=str, help="path to directory containing text files")
parser.add_argument("-e", "--extension", type=str, default=".txt", help="file extension of text files to be processed (default: .txt)")
args = parser.parse_args()

folder_path = args.folder_path
extension = args.extension
keywords = ["1girl", "solo", "blue eyes", "brown eyes", "blonde hair", "black hair", "realistic", "red lips", "lips", "artist name", "makeup", "realistic","brown hair", "dark skin", 
            "dark-skinned female", "medium breasts", "breasts", "1boy"]

for file_name in os.listdir(folder_path):
    if file_name.endswith(extension):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as f:
            text = f.read()
        # extract tags from text and split into a list using comma as the delimiter
        tags = [tag.strip() for tag in text.split(",")]
        # remove the specified keywords from the tags list
        tags = [tag for tag in tags if tag not in keywords]
        # remove empty or whitespace-only tags
        tags = [tag for tag in tags if tag.strip() != ""]
        # join the tags back into a comma-separated string and write back to the file
        with open(file_path, "w") as f:
            f.write(", ".join(tags))