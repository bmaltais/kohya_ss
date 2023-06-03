import os
import sys
import shutil
import time
import re
import argparse
import pkg_resources
from packaging.requirements import Requirement
from packaging.markers import default_environment

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

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
            log.warning("Torch reports CUDA not available")
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
        sys.exit(1)


def validate_requirements(requirements_file):
    log.info("Validating that requirements are satisfied.")
    
    with open(requirements_file) as f:
        requirements = f.readlines()

    missing_requirements = []
    wrong_version_requirements = []
    url_requirement_pattern = re.compile(r"(?P<url>https?://.+);?\s?(?P<marker>.+)?")

    for requirement in requirements:
        requirement = requirement.strip()
        
        if requirement == ".":
            # Skip the current requirement if it is a dot (.)
            continue
        
        url_match = url_requirement_pattern.match(requirement)
        
        if url_match:
            if url_match.group("marker"):
                marker = url_match.group("marker")
                parsed_marker = Marker(marker)
                
                if not parsed_marker.evaluate(default_environment()):
                    continue
            
            requirement = url_match.group("url")

        try:
            parsed_req = Requirement(requirement)
            
            if parsed_req.marker and not parsed_req.marker.evaluate(default_environment()):
                continue
            
            pkg_resources.require(str(parsed_req))
        except ValueError:
            # This block will handle URL-based requirements
            pass
        except pkg_resources.DistributionNotFound:
            missing_requirements.append(requirement)
        except pkg_resources.VersionConflict as e:
            wrong_version_requirements.append((requirement, str(e.req), e.dist.version))

    return missing_requirements, wrong_version_requirements


def print_error_messages(missing_requirements, wrong_version_requirements, requirements_file):
    if missing_requirements or wrong_version_requirements:
        if missing_requirements:
            log.info("Error: The following packages are missing:")
            
            for requirement in missing_requirements:
                log.info(f" - {requirement}")
        
        if wrong_version_requirements:
            log.info("Error: The following packages have the wrong version:")
            
            for requirement, expected_version, actual_version in wrong_version_requirements:
                log.info(f" - {requirement} (expected version {expected_version}, found version {actual_version})")
        
        upgrade_script = "upgrade.bat" if os.name == "nt" else "upgrade.sh"
        log.info(f"\nRun {upgrade_script} to resolve the missing requirements listed above...")
        sys.exit(1)

    log.info("All requirements satisfied.")
    sys.exit(0)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Validate that requirements are satisfied.")
    parser.add_argument('-r', '--requirements', type=str, default='requirements.txt', help="Path to the requirements file.")
    parser.add_argument('--debug', action='store_true', help='Debug on')
    args = parser.parse_args()

    # Check Torch
    check_torch()

    # Validate requirements
    missing_requirements, wrong_version_requirements = validate_requirements(args.requirements)

    # Print error messages if there are missing or wrong version requirements
    print_error_messages(missing_requirements, wrong_version_requirements, args.requirements)


if __name__ == "__main__":
    main()
