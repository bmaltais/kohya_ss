import os
import argparse

# Define the command line arguments
parser = argparse.ArgumentParser(description='Rename files in a folder')
parser.add_argument('folder', metavar='folder', type=str, help='the folder containing the files to rename')

# Parse the arguments
args = parser.parse_args()

# Get the list of files in the folder
files = os.listdir(args.folder)

# Loop through each file in the folder
for file in files:
    # Check if the file has the expected format
    if file.endswith('-0000.png'):
        # Get the new file name
        new_file_name = file[:-9] + '.mask'
        # Rename the file
        os.rename(os.path.join(args.folder, file), os.path.join(args.folder, new_file_name))
