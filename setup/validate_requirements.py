import os
import re
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
    elif (shutil.which('sycl-ls') is not None
    or os.environ.get('ONEAPI_ROOT') is not None
    or os.path.exists('/opt/intel/oneapi')):
        log.info('Intel OneAPI toolkit detected')
    else:
        log.info('Using CPU-only Torch')

    try:
        import torch
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                from library.ipex import ipex_init
                ipex_init()
        except Exception:
            pass
        log.info(f'Torch {torch.__version__}')

        # Check if CUDA is available
        if not torch.cuda.is_available():
            log.warning('Torch reports CUDA not available')
        else:
            if torch.version.cuda:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    # Log Intel IPEX OneAPI version
                    log.info(f'Torch backend: Intel IPEX {ipex.__version__}')
                else:
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
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    log.info(
                        f'Torch detected GPU: {torch.xpu.get_device_name(device)} VRAM {round(torch.xpu.get_device_properties(device).total_memory / 1024 / 1024)} Compute Units {torch.xpu.get_device_properties(device).max_compute_units}'
                    )
                else:
                    log.info(
                        f'Torch detected GPU: {torch.cuda.get_device_name(device)} VRAM {round(torch.cuda.get_device_properties(device).total_memory / 1024 / 1024)} Arch {torch.cuda.get_device_capability(device)} Cores {torch.cuda.get_device_properties(device).multi_processor_count}'
                    )
                return int(torch.__version__[0])
    except Exception as e:
        log.error(f'Could not load torch: {e}')
        sys.exit(1)


def main():
    setup_common.check_repo_version()
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

    torch_ver = check_torch()
    
    if args.requirements:
        setup_common.install_requirements(args.requirements, check_no_verify_flag=True)
    else:
        setup_common.install_requirements('requirements_windows_torch2.txt', check_no_verify_flag=True)


if __name__ == '__main__':
    main()
