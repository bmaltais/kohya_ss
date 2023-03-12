import sys
import pkg_resources

print("Validating that requirements are satisfied.")

# Load the requirements from the requirements.txt file
with open('requirements.txt') as f:
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
    print('\nRun \033[33mupgrade.ps1\033[0m or \033[33mpip install -U -r requirements.txt\033[0m to resolve the missing requirements listed above...')

    sys.exit(1)

# All requirements satisfied
print("All requirements satisfied.")
sys.exit(0)
