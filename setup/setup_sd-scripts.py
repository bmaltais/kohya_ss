import os
import sys
import setup_common

# Get the absolute path of the current file's directory (Kohua_SS project directory)
project_directory = os.path.dirname(os.path.abspath(__file__))

# Check if the "setup" directory is present in the project_directory
if "setup" in project_directory:
    # If the "setup" directory is present, move one level up to the parent directory
    project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project directory to the beginning of the Python search path
sys.path.insert(0, project_directory)

from kohya_gui.custom_logging import setup_logging

# Set up logging
log = setup_logging()

def main():
    # Read the tag version from the file
    tag_version = setup_common.read_tag_version_from_file(".sd-scripts-release")
    
    setup_common.clone_or_checkout(
        "https://github.com/kohya-ss/sd-scripts.git", tag_version, "sd-scripts"
    )

if __name__ == '__main__':
    main()
