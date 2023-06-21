import os
import re
import sys
import shutil
import argparse
from setup.setup_windows import install, check_repo_version

# Get the absolute path of the current file's directory (Kohua_SS project directory)
project_directory = os.path.dirname(os.path.abspath(__file__))

# Check if the "setup" directory is present in the project_directory
if "setup" in project_directory:
    # If the "setup" directory is present, move one level up to the parent directory
    project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project directory to the beginning of the Python search path
sys.path.insert(0, project_directory)

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()


def check_torch():
    # Check for nVidia toolkit or AMD toolkit
    if shutil.which('nvidia-smi') is not None or os.path.exists(
        os.path.join(
            os.environ.get('SystemRoot') or r'C:\Windows',
            'System32',
            'nvidia-smi.exe',
        )
    ):
        log.info('nVidia toolkit detected')
    elif shutil.which('rocminfo') is not None or os.path.exists(
        '/opt/rocm/bin/rocminfo'
    ):
        log.info('AMD toolkit detected')
    else:
        log.info('Using CPU-only Torch')

    try:
        import torch

        log.info(f'Torch {torch.__version__}')

        # Check if CUDA is available
        if not torch.cuda.is_available():
            log.warning('Torch reports CUDA not available')
        else:
            if torch.version.cuda:
                # Log nVidia CUDA and cuDNN versions
                log.info(
                    f'Torch backend: nVidia CUDA {torch.version.cuda} cuDNN {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"}'
                )
            elif torch.version.hip:
                # Log AMD ROCm HIP version
                log.info(f'Torch backend: AMD ROCm HIP {torch.version.hip}')
            else:
                log.warning('Unknown Torch backend')

            # Log information about detected GPUs
            for device in [
                torch.cuda.device(i) for i in range(torch.cuda.device_count())
            ]:
                log.info(
                    f'Torch detected GPU: {torch.cuda.get_device_name(device)} VRAM {round(torch.cuda.get_device_properties(device).total_memory / 1024 / 1024)} Arch {torch.cuda.get_device_capability(device)} Cores {torch.cuda.get_device_properties(device).multi_processor_count}'
                )
                return int(torch.__version__[0])
    except Exception as e:
        log.error(f'Could not load torch: {e}')
        sys.exit(1)


def install_requirements(requirements_file):
    log.info('Verifying requirements')
    with open(requirements_file, 'r', encoding='utf8') as f:
        # Read lines from the requirements file, strip whitespace, and filter out empty lines, comments, and lines starting with '.'
        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip() != ''
            and not line.startswith('#')
            and line is not None
            and not line.startswith('.')
        ]

        # Iterate over each line and install the requirements
        for line in lines:
            # Remove brackets and their contents from the line using regular expressions
            # eg diffusers[torch]==0.10.2 becomes diffusers==0.10.2
            package_name = re.sub(r'\[.*?\]', '', line)
            install(line, package_name)


def main():
    check_repo_version()
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

    if not args.requirements:
        # Check Torch
        if check_torch() == 1:
            install_requirements('requirements_windows_torch1.txt')
        else:
            install_requirements('requirements_windows_torch2.txt')
    else:
        install_requirements(args.requirements)


if __name__ == '__main__':
    main()
