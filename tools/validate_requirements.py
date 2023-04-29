import os
import sys
import pkg_resources
import argparse
import shutil
import logging
import time

log = logging.getLogger("sd")

# setup console and file logging
def setup_logging(clean=False):
    try:
        if clean and os.path.isfile('setup.log'):
            os.remove('setup.log')
        time.sleep(0.1) # prevent race condition
    except:
        pass
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(pathname)s | %(message)s', filename='setup.log', filemode='a', encoding='utf-8', force=True)
    from rich.theme import Theme
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.pretty import install as pretty_install
    from rich.traceback import install as traceback_install
    console = Console(log_time=True, log_time_format='%H:%M:%S-%f', theme=Theme({
        "traceback.border": "black",
        "traceback.border.syntax_error": "black",
        "inspect.value.border": "black",
    }))
    pretty_install(console=console)
    traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False, suppress=[])
    rh = RichHandler(show_time=True, omit_repeated_times=False, show_level=True, show_path=False, markup=False, rich_tracebacks=True, log_time_format='%H:%M:%S-%f', level=logging.DEBUG if args.debug else logging.INFO, console=console)
    rh.set_name(logging.DEBUG if args.debug else logging.INFO)
    log.addHandler(rh)

def check_torch():
    if shutil.which('nvidia-smi') is not None or os.path.exists(os.path.join(os.environ.get('SystemRoot') or r'C:\Windows', 'System32', 'nvidia-smi.exe')):
        log.info('nVidia toolkit detected')
    elif shutil.which('rocminfo') is not None or os.path.exists('/opt/rocm/bin/rocminfo'):
        log.info('AMD toolkit detected')
    else:
        log.info('Using CPU-only Torch')
    try:
        import torch
        log.info(f'Torch {torch.__version__}')
        if not torch.cuda.is_available():
            log.warning("Torch repoorts CUDA not available")
        else:
            if torch.version.cuda:
                log.info(f'Torch backend: nVidia CUDA {torch.version.cuda} cuDNN {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"}')
            elif torch.version.hip:
                log.info(f'Torch backend: AMD ROCm HIP {torch.version.hip}')
            else:
                log.warning('Unknown Torch backend')
            for device in [torch.cuda.device(i) for i in range(torch.cuda.device_count())]:
                log.info(f'Torch detected GPU: {torch.cuda.get_device_name(device)} VRAM {round(torch.cuda.get_device_properties(device).total_memory / 1024 / 1024)} Arch {torch.cuda.get_device_capability(device)} Cores {torch.cuda.get_device_properties(device).multi_processor_count}')
    except Exception as e:
        log.error(f'Could not load torch: {e}')
        exit(1)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Validate that requirements are satisfied.")
parser.add_argument('-r', '--requirements', type=str, default='requirements.txt', help="Path to the requirements file.")
parser.add_argument(
        '--debug', action='store_true', help='Debug on'
    )
args = parser.parse_args()

setup_logging()
check_torch()

log.info("Validating that requirements are satisfied.")

# Load the requirements from the specified requirements file
with open(args.requirements) as f:
    requirements = f.readlines()

# Check each requirement against the installed packages
missing_requirements = []
wrong_version_requirements = []
for requirement in requirements:
    requirement = requirement.strip()
    if requirement == ".":
        # Skip the current requirement if it is a dot (.)
        continue
    try:
        pkg_resources.require(requirement)
    except pkg_resources.DistributionNotFound:
        # Check if the requirement contains a VCS URL
        if "@" in requirement:
            # If it does, split the requirement into two parts: the package name and the VCS URL
            package_name, vcs_url = requirement.split("@", 1)
            # Use pip to install the package from the VCS URL
            os.system(f"pip install -e {vcs_url}")
            # Try to require the package again
            try:
                pkg_resources.require(package_name)
            except pkg_resources.DistributionNotFound:
                missing_requirements.append(requirement)
        else:
            missing_requirements.append(requirement)
    except pkg_resources.VersionConflict as e:
        wrong_version_requirements.append((requirement, str(e.req), e.dist.version))

# If there are any missing or wrong version requirements, print an error message and exit with a non-zero exit code
if missing_requirements or wrong_version_requirements:
    if missing_requirements:
        log.info("Error: The following packages are missing:")
        for requirement in missing_requirements:
            log.info(f" - {requirement}")
    if wrong_version_requirements:
        log.info("Error: The following packages have the wrong version:")
        for requirement, expected_version, actual_version in wrong_version_requirements:
            log.info(f" - {requirement} (expected version {expected_version}, found version {actual_version})")
    upgrade_script = "upgrade.ps1" if os.name == "nt" else "upgrade.sh"
    log.info(f"\nRun \033[33m{upgrade_script}\033[0m or \033[33mpip install -U -r {args.requirements}\033[0m to resolve the missing requirements listed above...")

    sys.exit(1)

# All requirements satisfied
log.info("All requirements satisfied.")
sys.exit(0)
