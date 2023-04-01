import os
import sys
import pkg_resources
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Validate that requirements are satisfied.")
parser.add_argument('-r', '--requirements', type=str, default='requirements.txt', help="Path to the requirements file.")
args = parser.parse_args()

print("Validating that requirements are satisfied.")

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
        print("Error: The following packages are missing:")
        for requirement in missing_requirements:
            print(f" - {requirement}")
    if wrong_version_requirements:
        print("Error: The following packages have the wrong version:")
        for requirement, expected_version, actual_version in wrong_version_requirements:
            print(f" - {requirement} (expected version {expected_version}, found version {actual_version})")
    upgrade_script = "upgrade.ps1" if os.name == "nt" else "upgrade.sh"
    print(f"\nRun \033[33m{upgrade_script}\033[0m or \033[33mpip install -U -r {args.requirements}\033[0m to resolve the missing requirements listed above...")

    sys.exit(1)

# All requirements satisfied
print("All requirements satisfied.")
sys.exit(0)
