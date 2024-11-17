import os
import sys
import logging
import shutil
import datetime
import subprocess
import re
import pkg_resources

log = logging.getLogger("sd")

# Constants
MIN_PYTHON_VERSION = (3, 10, 9)
MAX_PYTHON_VERSION = (3, 13, 0)
LOG_DIR = "../logs/setup/"
LOG_LEVEL = "INFO" # Set to "INFO" or "WARNING" for less verbose logging


def check_python_version():
    """
    Check if the current Python version is within the acceptable range.
    Returns:
        bool: True if the current Python version is valid, False otherwise.
    """
    log.debug("Checking Python version...")
    try:
        current_version = sys.version_info
        log.info(f"Python version is {sys.version}")

        if not (MIN_PYTHON_VERSION <= current_version < MAX_PYTHON_VERSION):
            log.error(
                f"The current version of python ({sys.version}) is not supported."
            )
            log.error("The Python version must be >= 3.10.9 and < 3.13.0.")
            return False
        return True
    except Exception as e:
        log.error(f"Failed to verify Python version. Error: {e}")
        return False


def update_submodule(quiet=True):
    """
    Ensure the submodule is initialized and updated.
    """
    log.debug("Updating submodule...")
    git_command = ["git", "submodule", "update", "--init", "--recursive"]
    if quiet:
        git_command.append("--quiet")

    try:
        subprocess.run(git_command, check=True)
        log.info("Submodule initialized and updated.")
    except subprocess.CalledProcessError as e:
        log.error(f"Error during Git operation: {e}")
    except FileNotFoundError as e:
        log.error(e)


def clone_or_checkout(repo_url, branch_or_tag, directory_name):
    """
    Clone a repo or checkout a specific branch or tag if the repo already exists.
    """
    log.debug(
        f"Cloning or checking out repository: {repo_url}, branch/tag: {branch_or_tag}, directory: {directory_name}"
    )
    original_dir = os.getcwd()
    try:
        if not os.path.exists(directory_name):
            run_cmd = [
                "git",
                "clone",
                "--branch",
                branch_or_tag,
                "--single-branch",
                "--quiet",
                repo_url,
                directory_name,
            ]
            log.debug(f"Cloning repository: {run_cmd}")
            subprocess.run(run_cmd, check=True)
            log.info(f"Successfully cloned {repo_url} ({branch_or_tag})")
        else:
            os.chdir(directory_name)
            log.debug("Fetching all branches and tags...")
            subprocess.run(["git", "fetch", "--all", "--quiet"], check=True)
            subprocess.run(
                ["git", "config", "advice.detachedHead", "false"], check=True
            )

            current_branch_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
            )
            target_branch_hash = (
                subprocess.check_output(["git", "rev-parse", branch_or_tag])
                .strip()
                .decode()
            )

            if current_branch_hash != target_branch_hash:
                log.debug(f"Checking out branch/tag: {branch_or_tag}")
                subprocess.run(
                    ["git", "checkout", branch_or_tag, "--quiet"], check=True
                )
                log.info(f"Checked out {branch_or_tag} successfully.")
            else:
                log.info(f"Already at required branch/tag: {branch_or_tag}")
    except subprocess.CalledProcessError as e:
        log.error(f"Error during Git operation: {e}")
    finally:
        os.chdir(original_dir)


def setup_logging():
    """
    Set up logging to file and console.
    """
    log.debug("Setting up logging...")

    from rich.theme import Theme
    from rich.logging import RichHandler
    from rich.console import Console

    console = Console(
        log_time=True,
        log_time_format="%H:%M:%S-%f",
        theme=Theme({"traceback.border": "black", "inspect.value.border": "black"}),
    )
    current_datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(
        os.path.dirname(__file__), f"{LOG_DIR}kohya_ss_gui_{current_datetime_str}.log"
    )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(message)s",
        filename=log_file,
        filemode="a",
        encoding="utf-8",
        force=True,
    )
    log_level = os.getenv("LOG_LEVEL", LOG_LEVEL).upper()
    log.setLevel(getattr(logging, log_level, logging.DEBUG))
    rich_handler = RichHandler(console=console)

    # Replace existing handlers with the rich handler
    log.handlers.clear()
    log.addHandler(rich_handler)
    log.debug("Logging setup complete.")


def install_requirements_inbulk(
    requirements_file, show_stdout=True, optional_parm="", upgrade=False
):
    log.debug(f"Installing requirements in bulk from: {requirements_file}")
    if not os.path.exists(requirements_file):
        log.error(f"Could not find the requirements file in {requirements_file}.")
        return

    log.info(f"Installing/Validating requirements from {requirements_file}...")

    # Build the command as a list
    cmd = ["pip", "install", "-r", requirements_file]
    if upgrade:
        cmd.append("--upgrade")
    if not show_stdout:
        cmd.append("--quiet")
    if optional_parm:
        cmd.extend(optional_parm.split())

    try:
        # Run the command and filter output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        for line in process.stdout:
            if "Requirement already satisfied" not in line:
                log.info(line.strip()) if show_stdout else None

        # Capture and log any errors
        _, stderr = process.communicate()
        if process.returncode != 0:
            log.error(f"Failed to install requirements: {stderr.strip()}")

    except subprocess.CalledProcessError as e:
        log.error(f"An error occurred while installing requirements: {e}")


def configure_accelerate(run_accelerate=False):
    log.debug("Configuring accelerate...")
    from pathlib import Path

    def env_var_exists(var_name):
        return var_name in os.environ and os.environ[var_name] != ""

    log.info("Configuring accelerate...")

    source_accelerate_config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "config_files",
        "accelerate",
        "default_config.yaml",
    )

    if not os.path.exists(source_accelerate_config_file):
        log.warning(
            f"Could not find the accelerate configuration file in {source_accelerate_config_file}."
        )
        if run_accelerate:
            log.debug("Running accelerate configuration command...")
            run_cmd([sys.executable, "-m", "accelerate", "config"])
        else:
            log.warning(
                "Please configure accelerate manually by running the option in the menu."
            )
        return

    log.debug(f"Source accelerate config location: {source_accelerate_config_file}")

    target_config_location = None

    env_vars = {
        "HF_HOME": Path(os.environ.get("HF_HOME", "")),
        "LOCALAPPDATA": Path(
            os.environ.get("LOCALAPPDATA", ""),
            "huggingface",
            "accelerate",
            "default_config.yaml",
        ),
        "USERPROFILE": Path(
            os.environ.get("USERPROFILE", ""),
            ".cache",
            "huggingface",
            "accelerate",
            "default_config.yaml",
        ),
    }

    for var, path in env_vars.items():
        if env_var_exists(var):
            target_config_location = path
            break

    log.debug(f"Target config location: {target_config_location}")

    if target_config_location:
        if not target_config_location.is_file():
            log.debug(
                f"Creating target config directory: {target_config_location.parent}"
            )
            target_config_location.parent.mkdir(parents=True, exist_ok=True)
            log.debug(
                f"Copying config file to target location: {target_config_location}"
            )
            shutil.copyfile(source_accelerate_config_file, target_config_location)
            log.info(f"Copied accelerate config file to: {target_config_location}")
        elif run_accelerate:
            log.debug("Running accelerate configuration command...")
            run_cmd([sys.executable, "-m", "accelerate", "config"])
        else:
            log.warning(
                "Could not automatically configure accelerate. Please manually configure accelerate with the option in the menu or with: accelerate config."
            )
    elif run_accelerate:
        log.debug("Running accelerate configuration command...")
        run_cmd([sys.executable, "-m", "accelerate", "config"])
    else:
        log.warning(
            "Could not automatically configure accelerate. Please manually configure accelerate with the option in the menu or with: accelerate config."
        )


def check_torch():
    log.debug("Checking Torch installation...")
    #
    # This function was adapted from code written by vladimandic: https://github.com/vladimandic/automatic/commits/master
    #

    # Check for toolkit
    if shutil.which("nvidia-smi") is not None or os.path.exists(
        os.path.join(
            os.environ.get("SystemRoot") or r"C:\Windows",
            "System32",
            "nvidia-smi.exe",
        )
    ):
        log.info("nVidia toolkit detected")
    elif shutil.which("rocminfo") is not None or os.path.exists(
        "/opt/rocm/bin/rocminfo"
    ):
        log.info("AMD toolkit detected")
    elif (
        shutil.which("sycl-ls") is not None
        or os.environ.get("ONEAPI_ROOT") is not None
        or os.path.exists("/opt/intel/oneapi")
    ):
        log.info("Intel OneAPI toolkit detected")
    else:
        log.info("Using CPU-only Torch")

    try:
        import torch

        log.debug("Torch module imported successfully.")
        try:
            # Import IPEX / XPU support
            import intel_extension_for_pytorch as ipex

            log.debug("Intel extension for PyTorch imported successfully.")
        except Exception as e:
            log.warning(f"Failed to import intel_extension_for_pytorch: {e}")
        log.info(f"Torch {torch.__version__}")

        if torch.cuda.is_available():
            if torch.version.cuda:
                # Log nVidia CUDA and cuDNN versions
                log.info(
                    f'Torch backend: nVidia CUDA {torch.version.cuda} cuDNN {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"}'
                )
            elif torch.version.hip:
                # Log AMD ROCm HIP version
                log.info(f"Torch backend: AMD ROCm HIP {torch.version.hip}")
            else:
                log.warning("Unknown Torch backend")

            # Log information about detected GPUs
            for device in [
                torch.cuda.device(i) for i in range(torch.cuda.device_count())
            ]:
                log.info(
                    f"Torch detected GPU: {torch.cuda.get_device_name(device)} VRAM {round(torch.cuda.get_device_properties(device).total_memory / 1024 / 1024)} Arch {torch.cuda.get_device_capability(device)} Cores {torch.cuda.get_device_properties(device).multi_processor_count}"
                )
        # Check if XPU is available
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            # Log Intel IPEX version
            log.info(f"Torch backend: Intel IPEX {ipex.__version__}")
            for device in [
                torch.xpu.device(i) for i in range(torch.xpu.device_count())
            ]:
                log.info(
                    f"Torch detected GPU: {torch.xpu.get_device_name(device)} VRAM {round(torch.xpu.get_device_properties(device).total_memory / 1024 / 1024)} Compute Units {torch.xpu.get_device_properties(device).max_compute_units}"
                )
        else:
            log.warning("Torch reports GPU not available")

        return int(torch.__version__[0])
    except Exception as e:
        log.error(f"Could not load torch: {e}")
        return 0


# report current version of code
def check_repo_version():
    """
    This function checks the version of the repository by reading the contents of a file named '.release'
    in the current directory. If the file exists, it reads the release version from the file and logs it.
    If the file does not exist, it logs a debug message indicating that the release could not be read.
    """
    log.debug("Checking repository version...")
    if os.path.exists(".release"):
        try:
            with open(os.path.join("./.release"), "r", encoding="utf8") as file:
                release = file.read()

            log.info(f"Kohya_ss GUI version: {release}")
        except Exception as e:
            log.error(f"Could not read release: {e}")
    else:
        log.debug("Could not read release...")


# execute git command
def git(arg: str, folder: str = None, ignore: bool = False):
    """
    Executes a Git command with the specified arguments.

    This function is designed to run Git commands and handle their output.
    It can be used to execute Git commands in a specific folder or the current directory.
    If an error occurs during the Git operation and the 'ignore' flag is not set,
    it logs the error message and the Git output for debugging purposes.

    Parameters:
    - arg: A string containing the Git command arguments.
    - folder: An optional string specifying the folder where the Git command should be executed.
               If not provided, the current directory is used.
    - ignore: A boolean flag indicating whether to ignore errors during the Git operation.
               If set to True, errors will not be logged.

    Note:
    This function was adapted from code written by vladimandic: https://github.com/vladimandic/automatic/commits/master
    """
    log.debug(f"Running git command: git {arg} in folder: {folder or '.'}")
    result = subprocess.run(
        ["git", arg],
        check=False,
        shell=True,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=folder or ".",
    )
    txt = result.stdout.decode(encoding="utf8", errors="ignore")
    if len(result.stderr) > 0:
        txt += ("\n" if len(txt) > 0 else "") + result.stderr.decode(
            encoding="utf8", errors="ignore"
        )
    txt = txt.strip()
    if result.returncode != 0 and not ignore:
        global errors
        errors += 1
        log.error(f"Error running git: {folder} / {arg}")
        if "or stash them" in txt:
            log.error(f"Local changes detected: check log for details...")
        log.debug(f"Git output: {txt}")


def pip(arg: str, ignore: bool = False, quiet: bool = False, show_stdout: bool = False):
    """
    Executes a pip command with the specified arguments.

    This function is designed to run pip commands and handle their output.
    It can be used to install, upgrade, or uninstall packages using pip.
    If an error occurs during the pip operation and the 'ignore' flag is not set,
    it logs the error message and the pip output for debugging purposes.

    Parameters:
    - arg: A string containing the pip command arguments.
    - ignore: A boolean flag indicating whether to ignore errors during the pip operation.
               If set to True, errors will not be logged.
    - quiet: A boolean flag indicating whether to suppress the output of the pip command.
              If set to True, the function will not log any output.
    - show_stdout: A boolean flag indicating whether to display the pip command's output
                    to the console. If set to True, the function will print the output
                    to the console.

    Returns:
    - The output of the pip command as a string, or None if the 'show_stdout' flag is set.
    """
    log.debug(f"Running pip command: {arg}")
    if not quiet:
        log.info(
            f'Installing package: {arg.replace("install", "").replace("--upgrade", "").replace("--no-deps", "").replace("--force", "").replace("  ", " ").strip()}'
        )
    pip_cmd = [rf"{sys.executable}", "-m", "pip"] + arg.split(" ")
    log.debug(f"Running pip: {pip_cmd}")
    if show_stdout:
        subprocess.run(pip_cmd, shell=False, check=False, env=os.environ)
    else:
        result = subprocess.run(
            pip_cmd,
            shell=False,
            check=False,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        txt = result.stdout.decode(encoding="utf8", errors="ignore")
        if len(result.stderr) > 0:
            txt += ("\n" if len(txt) > 0 else "") + result.stderr.decode(
                encoding="utf8", errors="ignore"
            )
        txt = txt.strip()
        if result.returncode != 0 and not ignore:
            log.error(f"Error running pip: {arg}")
            log.error(f"Pip output: {txt}")
        return txt


def installed(package, friendly: str = None):
    """
    Checks if the specified package(s) are installed with the correct version.
    This function can handle package specifications with or without version constraints,
    and can also filter out command-line options and URLs when a 'friendly' string is provided.

    Parameters:
    - package: A string that specifies one or more packages with optional version constraints.
    - friendly: An optional string used to provide a cleaner version of the package string
                that excludes command-line options and URLs.

    Returns:
    - True if all specified packages are installed with the correct versions, False otherwise.

    Note:
    This function was adapted from code written by vladimandic.
    """
    log.debug(f"Checking if package is installed: {package}")
    # Remove any optional features specified in brackets (e.g., "package[option]==version" becomes "package==version")
    package = re.sub(r"\[.*?\]", "", package)

    try:
        if friendly:
            # If a 'friendly' version of the package string is provided, split it into components
            pkgs = friendly.split()

            # Filter out command-line options and URLs from the package specification
            pkgs = [
                p for p in package.split() if not p.startswith("--") and "://" not in p
            ]
        else:
            # Split the package string into components, excluding '-' and '=' prefixed items
            pkgs = [
                p
                for p in package.split()
                if not p.startswith("-") and not p.startswith("=")
            ]
            # For each package component, extract the package name, excluding any URLs
            pkgs = [p.split("/")[-1] for p in pkgs]

        for pkg in pkgs:
            # Parse the package name and version based on the version specifier used
            if ">=" in pkg:
                pkg_name, pkg_version = [x.strip() for x in pkg.split(">=")]
            elif "==" in pkg:
                pkg_name, pkg_version = [x.strip() for x in pkg.split("==")]
            else:
                pkg_name, pkg_version = pkg.strip(), None

            # Attempt to find the installed package by its name
            spec = pkg_resources.working_set.by_key.get(pkg_name, None)
            if spec is None:
                # Try again with lowercase name
                spec = pkg_resources.working_set.by_key.get(pkg_name.lower(), None)
            if spec is None:
                # Try replacing underscores with dashes
                spec = pkg_resources.working_set.by_key.get(
                    pkg_name.replace("_", "-"), None
                )

            if spec is not None:
                # Package is found, check version
                version = pkg_resources.get_distribution(pkg_name).version
                log.debug(f"Package version found: {pkg_name} {version}")

                if pkg_version is not None:
                    # Verify if the installed version meets the specified constraints
                    if ">=" in pkg:
                        ok = version >= pkg_version
                    else:
                        ok = version == pkg_version

                    if not ok:
                        # Version mismatch, log warning and return False
                        log.warning(
                            f"Package wrong version: {pkg_name} {version} required {pkg_version}"
                        )
                        return False
            else:
                # Package not found, log debug message and return False
                log.debug(f"Package version not found: {pkg_name}")
                return False

        # All specified packages are installed with the correct versions
        return True
    except ModuleNotFoundError:
        # One or more packages are not installed, log debug message and return False
        log.debug(f"Package not installed: {pkgs}")
        return False


# install package using pip if not already installed
def install(
    package,
    friendly: str = None,
    ignore: bool = False,
    reinstall: bool = False,
    show_stdout: bool = False,
):
    """
    Installs or upgrades a Python package using pip, with options to ignode errors,
    reinstall packages, and display outputs.

    Parameters:
    - package (str): The name of the package to be installed or upgraded. Can include
      version specifiers. Anything after a '#' in the package name will be ignored.
    - friendly (str, optional): A more user-friendly name for the package, used for
      logging or user interface purposes. Defaults to None.
    - ignore (bool, optional): If True, any errors encountered during the installation
      will be ignored. Defaults to False.
    - reinstall (bool, optional): If True, forces the reinstallation of the package
      even if it's already installed. This also disables any quick install checks. Defaults to False.
    - show_stdout (bool, optional): If True, displays the standard output from the pip
      command to the console. Useful for debugging. Defaults to False.

    Returns:
    None. The function performs operations that affect the environment but does not return
    any value.

    Note:
    If `reinstall` is True, it disables any mechanism that allows for skipping installations
    when the package is already present, forcing a fresh install.
    """
    log.debug(f"Installing package: {package}")
    # Remove anything after '#' in the package variable
    package = package.split("#")[0].strip()

    if reinstall:
        global quick_allowed  # pylint: disable=global-statement
        quick_allowed = False
    if reinstall or not installed(package, friendly):
        pip(f"install --upgrade {package}", ignore=ignore, show_stdout=show_stdout)


def process_requirements_line(line, show_stdout: bool = False):
    log.debug(f"Processing requirements line: {line}")
    # Remove brackets and their contents from the line using regular expressions
    # e.g., diffusers[torch]==0.10.2 becomes diffusers==0.10.2
    package_name = re.sub(r"\[.*?\]", "", line)
    install(line, package_name, show_stdout=show_stdout)


def install_requirements(
    requirements_file, check_no_verify_flag=False, show_stdout: bool = False
):
    """
    Install or verify modules from a requirements file.

    Parameters:
    - requirements_file (str): Path to the requirements file.
    - check_no_verify_flag (bool): If True, verify modules installation status without installing.
    - show_stdout (bool): If True, show the standard output of the installation process.
    """
    log.debug(f"Installing requirements from file: {requirements_file}")
    action = "Verifying" if check_no_verify_flag else "Installing"
    log.info(f"{action} modules from {requirements_file}...")

    with open(requirements_file, "r", encoding="utf8") as f:
        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#") and "no_verify" not in line
        ]

    for line in lines:
        if line.startswith("-r"):
            included_file = line[2:].strip()
            log.debug(f"Processing included requirements file: {included_file}")
            install_requirements(
                included_file,
                check_no_verify_flag=check_no_verify_flag,
                show_stdout=show_stdout,
            )
        else:
            process_requirements_line(line, show_stdout=show_stdout)


def ensure_base_requirements():
    try:
        import rich  # pylint: disable=unused-import
    except ImportError:
        install("--upgrade rich", "rich")

    try:
        import packaging
    except ImportError:
        install("packaging")


def run_cmd(run_cmd):
    """
    Execute a command using subprocess.
    """
    log.debug(f"Running command: {run_cmd}")
    try:
        subprocess.run(run_cmd, shell=True, check=True, env=os.environ)
        log.debug(f"Command executed successfully: {run_cmd}")
    except subprocess.CalledProcessError as e:
        log.error(f"Error occurred while running command: {run_cmd}")
        log.error(f"Error: {e}")


def clear_screen():
    """
    Clear the terminal screen.
    """
    log.debug("Attempting to clear the terminal screen")
    try:
        os.system("cls" if os.name == "nt" else "clear")
        log.info("Terminal screen cleared successfully")
    except Exception as e:
        log.error("Error occurred while clearing the terminal screen")
        log.error(f"Error: {e}")
