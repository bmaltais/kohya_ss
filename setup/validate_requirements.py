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

def check_torch():
    # Check for toolkit
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
    elif (shutil.which('sycl-ls') is not None
    or os.environ.get('ONEAPI_ROOT') is not None
    or os.path.exists('/opt/intel/oneapi')):
        log.info('Intel OneAPI toolkit detected')
    else:
        log.info('Using CPU-only Torch')

    try:
        import torch
        try:
            # Import IPEX / XPU support
            import intel_extension_for_pytorch as ipex
        except Exception:
            pass
        log.info(f'Torch {torch.__version__}')

        if torch.cuda.is_available():
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
        # Check if XPU is available
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            # Log Intel IPEX version
            log.info(f'Torch backend: Intel IPEX {ipex.__version__}')
            for device in [
                torch.xpu.device(i) for i in range(torch.xpu.device_count())
            ]:
                log.info(
                    f'Torch detected GPU: {torch.xpu.get_device_name(device)} VRAM {round(torch.xpu.get_device_properties(device).total_memory / 1024 / 1024)} Compute Units {torch.xpu.get_device_properties(device).max_compute_units}'
                )
        else:
            log.warning('Torch reports GPU not available')
        
        return int(torch.__version__[0])
    except Exception as e:
        log.error(f'Could not load torch: {e}')
        sys.exit(1)
        
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

    torch_ver = check_torch()
    
    if not setup_common.check_python_version():
        exit(1)
    
    if args.requirements:
        setup_common.install_requirements(args.requirements, check_no_verify_flag=True)
    else:
        setup_common.install_requirements('requirements_pytorch_windows.txt', check_no_verify_flag=True)
        setup_common.install_requirements('requirements_windows.txt', check_no_verify_flag=True)

if __name__ == '__main__':
    main()
