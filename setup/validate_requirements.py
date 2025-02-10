import os
import sys
import shutil
import argparse
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

def check_path_with_space():
    # Get the current working directory
    cwd = os.getcwd()

    # Check if the current working directory contains a space
    if " " in cwd:
        log.error("The path in which this python code is executed contain one or many spaces. This is not supported for running kohya_ss GUI.")
        log.error("Please move the repo to a path without spaces, delete the venv folder and run setup.sh again.")
        log.error("The current working directory is: " + cwd)
        exit(1)
        
def main():
    setup_common.check_repo_version()
    
    check_path_with_space()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Validate that requirements are satisfied.'
    )
    parser.add_argument(
        '-r',
        '--requirements',
        type=str,
        help='Path to the requirements file.',
    )
    parser.add_argument('--debug', action='store_true', help='Debug on')
    args = parser.parse_args()
    
    setup_common.update_submodule()
    
    if not setup_common.check_python_version() or not setup_common.check_torch():
        exit(1)
    
    if args.requirements:
        setup_common.install_requirements(args.requirements, check_no_verify_flag=True)
    else:
        setup_common.install_requirements('requirements_pytorch_windows.txt', check_no_verify_flag=True)
        setup_common.install_requirements('requirements_windows.txt', check_no_verify_flag=True)

if __name__ == '__main__':
    main()
