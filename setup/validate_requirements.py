import os
import sys
import shutil
import argparse
import setup_common

# Get the absolute path of the current file's directory (Kohua_SS project directory)
project_directory = (
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if "setup" in os.path.dirname(os.path.abspath(__file__))
    else os.path.dirname(os.path.abspath(__file__))
)

# Add the project directory to the beginning of the Python search path
sys.path.insert(0, project_directory)

from kohya_gui.custom_logging import setup_logging

# Set up logging
log = setup_logging()
log.debug(f"Project directory set to: {project_directory}")

def check_path_with_space():
    """Check if the current working directory contains a space."""
    cwd = os.getcwd()
    log.debug(f"Current working directory: {cwd}")
    if " " in cwd:
        # Log an error if the current working directory contains spaces
        log.error(
            "The path in which this python code is executed contains one or many spaces. This is not supported for running kohya_ss GUI."
        )
        log.error(
            "Please move the repo to a path without spaces, delete the venv folder, and run setup.sh again."
        )
        log.error(f"The current working directory is: {cwd}")
        raise RuntimeError("Invalid path: contains spaces.")

def detect_toolkit():
    """Detect the available toolkit (NVIDIA, AMD, or Intel) and log the information."""
    log.debug("Detecting available toolkit...")
    # Check for NVIDIA toolkit by looking for nvidia-smi executable
    if shutil.which("nvidia-smi") or os.path.exists(
        os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"
        )
    ):
        log.debug("nVidia toolkit detected")
        return "nVidia"
    # Check for AMD toolkit by looking for rocminfo executable
    elif shutil.which("rocminfo") or os.path.exists("/opt/rocm/bin/rocminfo"):
        log.debug("AMD toolkit detected")
        return "AMD"
    # Check for Intel toolkit by looking for SYCL or OneAPI indicators
    elif (
        shutil.which("sycl-ls")
        or os.environ.get("ONEAPI_ROOT")
        or os.path.exists("/opt/intel/oneapi")
    ):
        log.debug("Intel toolkit detected")
        return "Intel"
    # Default to CPU if no toolkit is detected
    else:
        log.debug("No specific GPU toolkit detected, defaulting to CPU")
        return "CPU"

def check_torch():
    """Check if torch is available and log the relevant information."""
    # Detect the available toolkit (e.g., NVIDIA, AMD, Intel, or CPU)
    toolkit = detect_toolkit()
    log.info(f"{toolkit} toolkit detected")

    try:
        # Import PyTorch
        log.debug("Importing PyTorch...")
        import torch

        ipex = None
        # Attempt to import Intel Extension for PyTorch if Intel toolkit is detected
        if toolkit == "Intel":
            try:
                log.debug("Attempting to import Intel Extension for PyTorch (IPEX)...")
                import intel_extension_for_pytorch as ipex
                log.debug("Intel Extension for PyTorch (IPEX) imported successfully")
            except ImportError:
                log.warning("Intel Extension for PyTorch (IPEX) not found.")
        
        # Log the PyTorch version
        log.info(f"Torch {torch.__version__}")

        # Check if CUDA (NVIDIA GPU) is available
        if torch.cuda.is_available():
            log.debug("CUDA is available, logging CUDA info...")
            log_cuda_info(torch)
        # Check if XPU (Intel GPU) is available
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            log.debug("XPU is available, logging XPU info...")
            log_xpu_info(torch, ipex)
        # Log a warning if no GPU is available
        elif hasattr(torch, "mps") and torch.mps.is_available():
            log.info("MPS is available, logging MPS info...")
            log_mps_info(torch)
        else:
            log.warning("Torch reports GPU not available")

        # Return the major version of PyTorch
        return int(torch.__version__[0])
    except ImportError as e:
        # Log an error if PyTorch cannot be loaded
        log.error(f"Could not load torch: {e}")
        sys.exit(1)
    except Exception as e:
        # Log an unexpected error
        log.error(f"Unexpected error while checking torch: {e}")
        sys.exit(1)

def log_cuda_info(torch):
    """Log information about CUDA-enabled GPUs."""
    # Log the CUDA and cuDNN versions if available
    if torch.version.cuda:
        log.info(
            f'Torch backend: nVidia CUDA {torch.version.cuda} cuDNN {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"}'
        )
    # Log the ROCm HIP version if using AMD GPU
    elif torch.version.hip:
        log.info(f"Torch backend: AMD ROCm HIP {torch.version.hip}")
    else:
        log.warning("Unknown Torch backend")

    # Log information about each detected CUDA-enabled GPU
    for device in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device)
        log.info(
            f"Torch detected GPU: {props.name} VRAM {round(props.total_memory / 1024 / 1024)}MB Arch {props.major}.{props.minor} Cores {props.multi_processor_count}"
        )

def log_mps_info(torch):
    """Log information about Apple Silicone (MPS)"""
    max_recommended_mem = round(torch.mps.recommended_max_memory() / 1024**2)
    log.info(
        f"Torch detected Apple MPS: {max_recommended_mem}MB Unified Memory Available"
    )
    log.warning('MPS support is still experimental, proceed with caution.')


def log_xpu_info(torch, ipex):
    """Log information about Intel XPU-enabled GPUs."""
    # Log the Intel Extension for PyTorch (IPEX) version if available
    if ipex:
        log.info(f"Torch backend: Intel IPEX {ipex.__version__}")
    # Log information about each detected XPU-enabled GPU
    for device in range(torch.xpu.device_count()):
        props = torch.xpu.get_device_properties(device)
        log.info(
            f"Torch detected GPU: {props.name} VRAM {round(props.total_memory / 1024 / 1024)}MB Compute Units {props.max_compute_units}"
        )

def main():
    # Check the repository version to ensure compatibility
    log.debug("Checking repository version...")
    setup_common.check_repo_version()
    # Check if the current path contains spaces, which are not supported
    log.debug("Checking if the current path contains spaces...")
    check_path_with_space()

    # Parse command line arguments
    log.debug("Parsing command line arguments...")
    parser = argparse.ArgumentParser(
        description="Validate that requirements are satisfied."
    )
    parser.add_argument(
        "-r", "--requirements", type=str, help="Path to the requirements file."
    )
    parser.add_argument("--debug", action="store_true", help="Debug on")
    args = parser.parse_args()

    # Update git submodules if necessary
    log.debug("Updating git submodules...")
    setup_common.update_submodule()

    # Check if PyTorch is installed and log relevant information
    log.debug("Checking if PyTorch is installed...")
    torch_ver = check_torch()

    # Check if the Python version is compatible
    log.debug("Checking Python version...")
    if not setup_common.check_python_version():
        sys.exit(1)

    # Install required packages from the specified requirements file
    requirements_file = args.requirements or "requirements_pytorch_windows.txt"
    log.debug(f"Installing requirements from: {requirements_file}")
    setup_common.install_requirements_inbulk(
        requirements_file, show_stdout=True, 
        # optional_parm="--index-url https://download.pytorch.org/whl/cu124"
    )
    
    # setup_common.install_requirements(requirements_file, check_no_verify_flag=True)
    
    # log.debug("Installing additional requirements from: requirements_windows.txt")
    # setup_common.install_requirements(
    #     "requirements_windows.txt", check_no_verify_flag=True
    # )

if __name__ == "__main__":
    log.debug("Starting main function...")
    main()
    log.debug("Main function finished.")
