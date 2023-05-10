import os
import sys
import pkg_resources
import argparse
from packaging.requirements import Requirement
from packaging.markers import default_environment
import re

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
        
        # Check if the requirement has an environment marker and if it evaluates to False
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
