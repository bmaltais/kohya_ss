import argparse
import errno
import logging
import importlib
import mimetypes
import os
import pkgutil
from getpass import getpass
import platform
import re
import shutil
import site
import subprocess
import sys
import stat
import tempfile
import time
import zipfile
from pathlib import Path


# This enables programmatically installing pip packages
def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def check_and_import(module_name, package_name=None, alias=None):
    if package_name is None:
        package_name = module_name

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logging.debug(f"Installing {package_name}...")
        install_package(package_name)
        module = importlib.import_module(module_name)

    if alias:
        sys.modules[alias] = module

    return module


base64 = check_and_import('base64')
requests = check_and_import('requests')
yaml = check_and_import('yaml', 'PyYAML')
tqdm_module = check_and_import("tqdm", "tqdm")
tqdm_progress = tqdm_module.tqdm

# Set the package versions at the beginning of the script to make them easy to modify as needed.
TENSORFLOW_VERSION = "2.12.0"
TENSORFLOW_MACOS_VERSION = "2.12.0"
TENSORFLOW_METAL_VERSION = "0.8.0"


def find_config_file(config_file_locations):
    for location in config_file_locations:
        abs_location = os.path.abspath(location)
        if os.path.isfile(abs_location):
            return abs_location
    return None


def load_config(_config_file):
    config_locations = []

    if _config_file is not None:
        config_locations.append(_config_file)

    if sys.platform == "win32":
        config_locations.extend([
            os.path.join(os.environ.get("APPDATA", ""), "kohya_ss", "config_files", "installation",
                         "install_config.yaml"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "kohya_ss", "install_config.yaml")
        ])

    config_locations.extend([
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "install_config.yaml"),
        os.path.join(os.environ.get("USERPROFILE", ""), "kohya_ss", "install_config.yaml"),
        os.path.join(os.environ.get("HOME", ""), ".kohya_ss", "install_config.yaml"),
    ])

    _config_file = ""
    for location in config_locations:
        if os.path.isfile(os.path.abspath(location)):
            _config_file = location
            break

    if _config_file:
        with open(_config_file, "r") as f:
            _config_data = yaml.safe_load(f)
    else:
        _config_data = None

    return _config_data


def parse_file_arg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", "--file", dest="config_file", default=None,
                        help="Path to the configuration file.")
    _args, _ = parser.parse_known_args()
    if _args.config_file is not None:
        logging.debug(f"Configuration file specified by command line: {os.path.abspath(_args.file)}")
        return os.path.abspath(_args.file)
    else:
        return None


def normalize_paths(_args, default_args):
    def is_valid_path(value):
        try:
            path = os.path.abspath(value)
            return os.path.exists(path)
        except TypeError:
            return False

    for arg in default_args:
        default_value = arg["default"]
        if isinstance(default_value, str) and is_valid_path(default_value):
            arg_name = arg["long"][2:].replace("-", "_")
            path_value = getattr(_args, arg_name)
            if path_value:
                setattr(_args, arg_name, os.path.abspath(path_value))


# This custom action was added so that the v option could be used Windows-style with integers (-v 3) setting the
# verbosity and Unix style (-vvv).
class CountOccurrencesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, int):
            # If value is an integer, set verbosity level to that value
            setattr(namespace, self.dest, values)
        elif isinstance(values, str):
            # If value is a string, check if it's a single integer
            try:
                count = int(values)
                setattr(namespace, self.dest, count)
            except ValueError:
                # If value is not a single integer, check if it's a valid verbosity string
                if not bool(re.search('[^v]', values)):
                    # We add a single v because .count starts at zero and returns v - 1.
                    count = (values + 'v').count('v')
                    setattr(namespace, self.dest, count)
                else:
                    logging.error('Invalid verbosity level')
                    exit(1)
        else:
            # If value is not a string or integer, raise an error
            raise argparse.ArgumentTypeError('Invalid verbosity level')

        # Check if verbosity level is a non-negative integer
        if getattr(namespace, self.dest) < 0:
            logging.error('Verbosity level must be a positive integer')
            exit(1)


def parse_args(_config_data):
    parser = argparse.ArgumentParser(
        description="Launcher script for Kohya_SS. This script helps you configure, install, and launch the Kohya_SS "
                    "application.",
        epilog="""Examples:
    Switch to the dev branch:
    python launcher.py --branch dev

    Point to a custom installation directory
    python launcher.py --dir /path/to/kohya_ss
    
    Update to the latest stable mainline installation
    python launcher.py --dir /path/to/kohya_ss --update

    Bypass all environment checks except Python dependency validation and launch the GUI:
    python launcher.py --exclude-setup""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Define the default arguments first. The spacing is purely for readability.
    default_args = [
        {"short": "-b", "long": "--branch", "default": "master", "type": str,
         "help": "Select which branch of kohya to check out on new installs."},

        {"short": "-d", "long": "--dir", "default": os.path.expanduser("~/kohya_ss"), "type": str,
         "help": "The full path you want kohya_ss installed to."},

        {"short": "-f", "long": "--file", "default": "install_config.yaml", "type": str,
         "help": "Configuration file with installation options."},

        {"short": "-g", "long": "--git-repo", "default": "https://github.com/bmaltais/kohya_ss.git", "type": str,
         "help": "You can optionally provide a git repo to check out. Useful for custom forks."},

        {"short": "-i", "long": "--interactive", "default": False, "type": bool,
         "help": "Interactively configure accelerate instead of using value config file."},

        {"short": "-n", "long": "--no-setup", "default": False, "type": bool,
         "help": "Skip setup operations and launch the GUI."},

        {"short": "-p", "long": "--public", "default": False, "type": bool,
         "help": "Expose public URL in runpod mode. Won't have an effect in other modes."},

        {"short": "-r", "long": "--runpod", "default": False, "type": bool,
         "help": "Forces a runpod installation. Useful if detection fails for any reason."},

        {"short": "-s", "long": "--skip-space-check", "default": False, "type": bool,
         "help": "Skip the 10Gb minimum storage space check."},

        {"short": "-u", "long": "--update", "default": False, "type": bool,
         "help": "Update kohya_ss with specified branch, repo, or latest stable if git's unavailable."},

        {"short": "-v", "long": "--verbosity", "default": 0,
         "help": "Increase verbosity levels. Use multiple times (e.g., -vvv) or specify number (e.g., -v 4).",
         "action": CountOccurrencesAction},

        {"short": "-x", "long": "--exclude-setup", "default": False, "type": bool,
         "help": "Skip all setup steps and only validate python requirements then launch GUI."},

        {"short": "", "long": "--listen", "default": "127.0.0.1", "type": str,
         "help": "IP to listen on for connections to Gradio."},

        {"short": "", "long": "--username", "default": "", "type": str, "help": "Username for authentication."},

        {"short": "", "long": "--password", "default": "", "type": str, "help": "Password for authentication."},

        {"short": "", "long": "--server-port", "default": 0, "type": str,
         "help": "Port to run the server listener on."},

        {"short": "", "long": "--inbrowser", "default": False, "type": bool, "help": "Open in browser."},

        {"short": "", "long": "--share", "default": False, "type": bool, "help": "Share the gradio UI."},
    ]

    # Update the default arguments with values from the config file
    if _config_data:
        if "setup_arguments" in _config_data:
            for arg in _config_data["setup_arguments"]:
                name = arg["name"]
                value = arg["value"]
                description = arg["description"]
                for default_arg in default_args:
                    if f'--{name.lower()}' == default_arg["long"]:
                        default_arg["default"] = value
                        default_arg["help"] = description
        if "kohya_gui_arguments" in _config_data:
            for arg in _config_data["kohya_gui_arguments"]:
                name = arg["name"]
                value = arg["value"]
                description = arg["description"]
                for default_arg in default_args:
                    if f'--{name.lower()}' == default_arg["long"]:
                        default_arg["default"] = value
                        default_arg["help"] = description

    # Add arguments to the parser with updated default values
    for arg in default_args:
        short_opt = arg["short"]
        long_opt = arg["long"]
        default_value = arg["default"]
        arg_type = arg.get("type", str)
        help_text = arg.get("help", None)
        custom_action = arg.get("action", None)

        if custom_action:
            if short_opt:
                parser.add_argument(short_opt, long_opt, dest=None, action=custom_action, default=default_value,
                                    help=help_text)
            else:
                parser.add_argument(long_opt, dest=long_opt[2:].replace("-", "_"), action=custom_action,
                                    default=default_value, help=help_text)
        elif isinstance(default_value, bool):
            action = 'store_true' if default_value is False else 'store_false'
            if short_opt:
                parser.add_argument(short_opt, long_opt, dest=long_opt[2:], action=action, default=default_value,
                                    help=help_text)
            else:
                parser.add_argument(long_opt, dest=long_opt[2:].replace("-", "_"), default=default_value, type=arg_type,
                                    help=help_text)
        else:
            if short_opt:
                parser.add_argument(short_opt, long_opt, dest=long_opt[2:], default=default_value, type=arg_type,
                                    help=help_text)
            else:
                parser.add_argument(long_opt, dest=long_opt[2:].replace("-", "_"), default=default_value, type=arg_type,
                                    help=help_text)

    _args = parser.parse_args()

    # Normalize paths to ensure absolute paths
    normalize_paths(_args, default_args)
    return _args


def env_var_exists(var_name):
    return var_name in os.environ and os.environ[var_name] != ""


def get_default_dir(runpod, script_dir):
    os_type = platform.system()
    if os_type == "Linux":
        if runpod:
            default_dir = "/workspace/kohya_ss"
        elif os.path.isdir(os.path.join(script_dir, ".git")):
            default_dir = script_dir
        elif os.access("/opt", os.W_OK):
            default_dir = "/opt/kohya_ss"
        elif env_var_exists("HOME"):
            default_dir = os.path.join(os.environ["HOME"], "kohya_ss")
        else:
            default_dir = os.getcwd()
    else:
        if os.path.isdir(os.path.join(script_dir, ".git")):
            default_dir = script_dir
        elif env_var_exists("HOME"):
            default_dir = os.path.join(os.environ["HOME"], "kohya_ss")
        else:
            default_dir = os.getcwd()
    return default_dir


def get_venv_directory():
    # Get the current environment path
    env_path = sys.prefix

    # Get the site-packages directory
    _site_packages_dir = site.getsitepackages()[0]

    # Return the environment path and site-packages path
    return env_path, _site_packages_dir


def check_and_create_install_folder(parent_dir, _dir):
    if os.access(parent_dir, os.W_OK) and not os.path.isdir(_dir):
        logging.info(f"Creating install folder {_dir}.")
        os.makedirs(_dir)

    if not os.access(_dir, os.W_OK):
        logging.error(f"We cannot write to {_dir}.")
        logging.info("Please ensure the install directory is accurate and you have the correct permissions.")
        exit(1)


def size_available(_dir, parent_dir):
    folder = None
    if os.path.isdir(_dir):
        folder = _dir
    elif os.path.isdir(parent_dir):
        folder = parent_dir
    else:
        path_parts = os.path.split(_dir)
        if path_parts[0] and os.path.isdir(path_parts[0]):
            folder = path_parts[0]

    if not folder:
        logging.info("We are assuming a root drive install for space-checking purposes.")
        folder = os.path.abspath(os.sep)

    free_space_in_bytes = shutil.disk_usage(folder).free
    free_space_in_gb = free_space_in_bytes / (1024 * 1024 * 1024)
    return free_space_in_gb


def check_storage_space(_dir, parent_dir, space_check=True):
    if space_check:
        if size_available(_dir, parent_dir) < 10:
            logging.info("You have less than 10Gb of free space. This installation may fail.")
            msg_timeout = 10  # In seconds
            message = "Continuing in..."
            logging.info("Press control-c to cancel the installation.")

            for i in range(msg_timeout, -1, -1):
                print(f"\r{message} {i}s.", end="")
                time.sleep(1)


def create_symlinks(symlink, target_file):
    logging.info("Checking symlinks now.")
    # Next line checks for valid symlink
    if os.path.islink(symlink):
        # Check if the linked file exists and points to the expected file
        if os.path.exists(symlink) and os.path.realpath(symlink) == target_file:
            logging.debug(f"{os.path.basename(symlink)} symlink looks fine. Skipping.")
        else:
            if os.path.isfile(target_file):
                logging.warning(f"Broken symlink detected. Recreating {os.path.basename(symlink)}.")
                os.remove(symlink)
                os.symlink(target_file, symlink)
            else:
                logging.error(f"{target_file} does not exist. Nothing to link.")
    else:
        logging.info(f"Linking {os.path.basename(symlink)}.")
        os.symlink(target_file, symlink)


def setup_file_links(_site_packages_dir, runpod):
    if os_info.family == "Windows":
        bitsandbytes_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitsandbytes_windows")
        bitsandbytes_dest = os.path.join(_site_packages_dir, "bitsandbytes")
        bitsandbytes_cuda_dest = os.path.join(_site_packages_dir, "bitsandbytes", "cuda_setup")

        if os.path.exists(bitsandbytes_source):
            # Create destination directories if they don't exist
            try:
                os.makedirs(bitsandbytes_dest, exist_ok=True)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            try:
                os.makedirs(bitsandbytes_cuda_dest, exist_ok=True)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            # Copy .dll files
            for file in os.listdir(bitsandbytes_source):
                if file.endswith(".dll"):
                    shutil.copy(os.path.join(bitsandbytes_source, file), bitsandbytes_dest)

            # Copy cextension.py
            shutil.copy(os.path.join(bitsandbytes_source, "cextension.py"),
                        os.path.join(bitsandbytes_dest, "cextension.py"))

            # Copy main.py
            shutil.copy(os.path.join(bitsandbytes_source, "main.py"), os.path.join(bitsandbytes_cuda_dest, "main.py"))

    if runpod and in_container:
        # Symlink paths
        libnvinfer_plugin_symlink = os.path.join(_site_packages_dir, "tensorrt", "libnvinfer_plugin.so.7")
        libnvinfer_symlink = os.path.join(_site_packages_dir, "tensorrt", "libnvinfer.so.7")
        libcudart_symlink = os.path.join(_site_packages_dir, "nvidia", "cuda_runtime", "lib", "libcudart.so.11.0")

        # Target file paths
        libnvinfer_plugin_target = os.path.join(_site_packages_dir, "tensorrt", "libnvinfer_plugin.so.8")
        libnvinfer_target = os.path.join(_site_packages_dir, "tensorrt", "libnvinfer.so.8")
        libcudart_target = os.path.join(_site_packages_dir, "nvidia", "cuda_runtime", "lib", "libcudart.so.12")

        logging.info("Checking symlinks now.")
        create_symlinks(libnvinfer_plugin_symlink, libnvinfer_plugin_target)
        create_symlinks(libnvinfer_symlink, libnvinfer_target)
        create_symlinks(libcudart_symlink, libcudart_target)

        tensorrt_dir = os.path.join(_site_packages_dir, "tensorrt")
        if os.path.isdir(tensorrt_dir):
            os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{tensorrt_dir}"
        else:
            logging.warning(f"{tensorrt_dir} not found; not linking library.")

        cuda_runtime_dir = os.path.join(_site_packages_dir, "nvidia", "cuda_runtime", "lib")
        if os.path.isdir(cuda_runtime_dir):
            os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{cuda_runtime_dir}"
        else:
            logging.warning(f"{cuda_runtime_dir} not found; not linking library.")


def in_container():
    cgroup_path = "/proc/1/cgroup"

    if not os.path.isfile(cgroup_path):
        return False

    with open(cgroup_path, "r") as cgroup_file:
        content = cgroup_file.read()

    container_indicators = [
        r':cpuset:/(docker|kubepods)',
        r':/docker/',
        r':cpuset:/docker/buildkit',
        r':/system.slice/docker-',
        r':/system.slice/containerd-',
        r':/system.slice/rkt-',
        r':/system.slice/run-',
        r':/system.slice/pod-',
    ]

    if any(re.search(pattern, content) for pattern in container_indicators) or os.path.exists('/.dockerenv'):
        return True

    return False


class GitAuthenticationError(Exception):
    pass


def is_git_installed():
    git_commands = ["git"]
    if sys.platform == "win32":
        git_commands.append("git.exe")

    for git_command in git_commands:
        try:
            subprocess.run([git_command, "--version"], check=True, capture_output=True)
            logging.debug("Git found.")
            return True
        except FileNotFoundError:
            logging.debug("Git not found.")
            return False

    logging.warning("Git not found.")
    return False


def run_git_command(_args, cwd=None, timeout=10, username=None, password=None):
    env = os.environ.copy()
    if username and password:
        # Create a temporary file for the GIT_ASKPASS script
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as askpass_file:
            askpass_file.write(f"#!/usr/bin/env python\nimport sys\nprint('{password}')\n")
            askpass_file_path = askpass_file.name

        env["GIT_ASKPASS"] = askpass_file_path
        env["GIT_USERNAME"] = username
        env["GIT_PASSWORD"] = password

    try:
        result = subprocess.run(["git"] + _args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
                                timeout=timeout, env=env)
        return result.stdout.decode("utf-8").strip()
    except subprocess.TimeoutExpired:
        logging.error("Git command timed out.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e}")
        logging.debug(f"stdout: {e.stdout.decode('utf-8')}")
        logging.debug(f"stderr: {e.stderr.decode('utf-8')}")
        raise Exception("Git failed exiting script to be on the safe side.")
    finally:
        if username and password:
            if os.path.exists(askpass_file_path):
                os.remove(askpass_file_path)


def get_latest_tag(git_repo):
    repo_name = git_repo.split("/")[-1].rstrip(".git")
    owner = git_repo.split("/")[-2]

    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/releases/latest"
    response = requests.get(api_url)

    if response.status_code != 200:
        raise Exception(f"Failed to get the latest release: {response.status_code}")

    data = response.json()
    return data["tag_name"]


def update_kohya_ss(_dir, git_repo, branch, update):
    success = False
    logging.debug(f"Update: {update}")
    logging.debug(f".git folder path: {os.path.join(_dir, '.git')}")
    logging.debug(f"Items detected in _dir: {os.listdir(_dir)}")
    logging.debug(f".git detected: {'.git' in os.listdir(_dir)}")

    def git_operations_with_credentials(username=None, password=None):
        nonlocal success
        if len(os.listdir(_dir)) == 0 or (
                len(os.listdir(_dir)) == 1 and os.path.exists(os.path.join(_dir, "venv"))):
            logging.debug("git clone operation entered.")
            run_git_command(["clone", "-b", branch, git_repo, _dir], username=username,
                            password=password if password else None)
            success = True
        elif update and ".git" in os.listdir(_dir):
            logging.debug("git pull operation entered.")
            run_git_command(["pull"], cwd=_dir, username=username,
                            password=password if password else None)
            success = True
        elif len(os.listdir(_dir)) > 1:
            logging.info(f"A git repo was detected at {_dir}, but update was not enabled. "
                         f"Skipping updating folder contents.")
            success = False
        else:
            logging.error("Git operations failed.")
            success = False

    git_installed = is_git_installed()
    max_attempts = 4
    attempt = 0
    username = None
    password = None
    success = False

    try:
        logging.debug(f"_dir contents: {os.listdir(_dir)}")
        logging.debug(f"branch: {branch}")
        logging.debug(f"git_repo: {git_repo}")
        if git_installed:
            if os.path.exists(_dir) and os.path.isdir(_dir):
                while attempt < max_attempts:
                    try:
                        git_operations_with_credentials(username, password)
                        break
                    except GitAuthenticationError as e:
                        if attempt < max_attempts - 1:
                            logging.warning(f"Git authentication failed: {e}")
                            logging.info(f"Attempting to authenticate to {git_repo}.")
                            logging.info("Please enter your Git credentials:")
                            username = input("Username: ")
                            password = getpass.getpass("Password: ")
                            attempt += 1
                        else:
                            raise e
        else:
            raise Exception("Git not installed.")
    except Exception as e:
        logging.warning(f"Failed to clone or update the repository using git: {e}")

    if not success:
        # Check if the directory is empty or contains only a "venv" folder, the branch is "master",
        # and the Git repository URL starts with "https://github.com/bmaltais/kohya_ss" or the update flag is specified.
        # If all conditions are met, we try to download the latest tag as a zip for installation.
        # We only overwrite the files we download. Otherwise, skip the installation.
        if (update or len(os.listdir(_dir)) == 0 or (
                len(os.listdir(_dir)) == 1 and os.path.exists(os.path.join(_dir, "venv")))) and \
                (not branch or branch == "master") and (
                not git_repo or git_repo.startswith("https://github.com/bmaltais/kohya_ss")):

            # Download the latest release as a zip file from the default repository
            try:
                # Download the repo as a zip file
                # Remove .git extension if present
                git_repo = git_repo.rstrip('.git')
                download_url = git_repo.rstrip("/") + f"/archive/refs/tags/{get_latest_tag(git_repo)}.zip"
                auth = (username, password) if username and password else None
                logging.info(f"Attempting to download from: {download_url}")
                response = requests.get(download_url, auth=auth)

                if response.status_code != 200:
                    raise Exception(f"Failed to download the repository: {response.status_code}")

                # Get the file size from the 'Content-Length' header
                file_size = int(response.headers.get("Content-Length", 0))

                # Create a progress bar
                progress_bar = tqdm_progress(total=file_size, unit="B", unit_scale=True, desc="Downloading")

                # Save the zip file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False) as temp_zip:
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_zip.write(chunk)
                        progress_bar.update(len(chunk))  # Update the progress bar
                    temp_zip.close()
                    logging.debug(f"Zip file downloaded to: {temp_zip.name}")
                    progress_bar.close()

                    # Extract the zip file to a temporary directory
                    with zipfile.ZipFile(temp_zip.name, "r") as zip_ref:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            zip_ref.extractall(temp_dir)
                            logging.debug(f"Zip file extracted to: {temp_dir}")

                            # Get the actual extracted folder name
                            extracted_folder = os.path.join(temp_dir, os.listdir(temp_dir)[0])

                            for root, _, files in os.walk(extracted_folder):
                                rel_path = os.path.relpath(root, extracted_folder)
                                target_dir = os.path.join(_dir, rel_path)

                                if not os.path.exists(target_dir):
                                    os.makedirs(target_dir)

                                for file in files:
                                    src_file = os.path.join(root, file)
                                    dst_file = os.path.join(target_dir, file)
                                    shutil.move(src_file, dst_file)
                                    logging.debug(f"Moved file: {src_file} to {dst_file}")

                            # Clean up the extracted folder
                            shutil.rmtree(extracted_folder)
                            logging.debug(f"Cleaned up extracted folder: {extracted_folder}")

                # Remove the temporary zip file
                os.remove(temp_zip.name)
                logging.debug(f"Removed temporary zip file: {temp_zip.name}")
                success = True

            except Exception as e:
                logging.warning(f"Failed to download the latest release: {e}")

        elif update is True and not git_repo.startswith("https://github.com/bmaltais/kohya_ss"):
            logging.info("Sorry, we only support zip file updates for master branch on "
                         "github.com/bmaltais/kohya_ss")
            success = False

        else:
            logging.error("We could not download the latest release via git or zip file.")
            success = False

    return success


class OSInfo:
    def __init__(self):
        self.name = "Unknown"
        self.family = "Unknown"
        self.version = "Unknown"
        self.detect_os()

    def detect_os(self):
        system = platform.system()
        if system == "Windows":
            self.name = "Windows"
            self.family = "Windows"
            self.version = platform.version()

        elif system == "Darwin":
            self.name = "macOS"
            self.family = "macOS"
            self.version = "Unknown"

            try:
                with open("/System/Library/CoreServices/SystemVersion.plist", "r") as f:
                    content = f.read()
                    version_match = re.search(r"<string>([\d.]+)</string>", content)
                    if version_match:
                        self.version = version_match.group(1)
            except Exception as e:
                logging.error(f"Error reading /System/Library/CoreServices/SystemVersion.plist: {e}")

        elif system == "Linux":
            if os.path.exists("/etc/os-release"):
                try:
                    with open("/etc/os-release", "r") as f:
                        content = f.read()
                        self.name = re.search(r'ID="?([^"\n]+)', content).group(1)
                        self.family = re.search(r'ID_LIKE="?([^"\n]+)', content).group(1)
                        self.version = re.search(r'VERSION="?([^"\n]+)', content).group(1)
                except Exception as e:
                    logging.error(f"Error reading /etc/os-release: {e}")

            elif os.path.exists("/etc/redhat-release"):
                try:
                    with open("/etc/redhat-release", "r") as f:
                        content = f.read()
                        match = re.search(r'([^ ]+) release ([^ ]+)', content)
                        if match:
                            self.name = match.group(1)
                            self.family = "RedHat"
                            self.version = match.group(2)
                except Exception as e:
                    logging.error(f"Error reading /etc/redhat-release: {e}")

            if self.name == "Unknown":
                try:
                    uname = subprocess.getoutput("uname -a")
                    if "Ubuntu" in uname:
                        self.name = "Ubuntu"
                        self.family = "Ubuntu"
                    elif "Debian" in uname:
                        self.name = "Debian"
                        self.family = "Debian"
                    elif "Red Hat" in uname or "CentOS" in uname:
                        self.name = "RedHat"
                        self.family = "RedHat"
                    elif "Fedora" in uname:
                        self.name = "Fedora"
                        self.family = "Fedora"
                    elif "SUSE" in uname:
                        self.name = "openSUSE"
                        self.family = "SUSE"
                    elif "Arch" in uname:
                        self.name = "Arch"
                        self.family = "Arch"
                    else:
                        self.name = "Generic Linux"
                        self.family = "Generic Linux"
                except Exception as e:
                    logging.error(f"Error executing uname command: {e}")
                    self.name = "Generic Linux"
                    self.family = "Generic Linux"
            return {
                "name": self.name,
                "family": self.family,
                "version": self.version
            }


def get_os_info():
    return OSInfo()


def brew_install_tensorflow_deps(verbosity=1):
    brew_install_cmd = "brew install icu4c xz zlib bzip2 lz4 lzo openssl readline sqlite libyaml libiconv libarchive " \
                       "libffi libxml2"

    def brew_installed():
        if os_info.family == "Darwin":
            try:
                subprocess.run(["brew", "-v"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            except subprocess.CalledProcessError:
                return False

        if not os_info.family == "Darwin":
            logging.debug("Non-macOS detected. Skipping brew installation of dependencies.")
            return True
        else:
            if not brew_installed():
                logging.error("Homebrew not found. Please install Homebrew before running this script.")
                return False
            stdout_setting = subprocess.PIPE if verbosity >= 3 else subprocess.DEVNULL
            stderr_setting = subprocess.PIPE if verbosity >= 1 else subprocess.DEVNULL

            try:
                logging.info("Installing Homebrew packages...")
                result = subprocess.run(brew_install_cmd.split(), stdout=stdout_setting, stderr=stderr_setting)
                result.check_returncode()
                if verbosity >= 3:
                    logging.debug(result.stdout.decode('utf-8'))
                logging.info("Homebrew packages installed successfully.")
                return True
            except subprocess.CalledProcessError as e:
                if verbosity >= 1:
                    logging.error(e.stderr.decode('utf-8'))
                return False


def check_permissions(_dir):
    venv_directory = os.path.join(_dir, "venv")
    extensions_to_check = (".py", ".exe", ".elf")

    for root, dirs, files in os.walk(venv_directory):
        # Skip site-packages directory
        if root.startswith(os.path.join(venv_directory, "Lib", "site-packages")):
            continue

        if root.startswith(os.path.join(venv_directory, "share", "doc")):
            continue

        for file in files:
            file_path = os.path.join(root, file)
            current_permissions = os.stat(file_path).st_mode

            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type == 'application/x-executable' or file.endswith(extensions_to_check):
                required_permissions = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | \
                                       stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | \
                                       stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH
            else:
                required_permissions = stat.S_IRUSR | stat.S_IWUSR | \
                                       stat.S_IRGRP | stat.S_IWGRP | \
                                       stat.S_IROTH | stat.S_IWOTH

            missing_permissions = required_permissions & ~current_permissions
            if missing_permissions:
                logging.debug(f"Missing permissions on file: {file_path}")

                try:
                    os.chmod(file_path, current_permissions | missing_permissions)
                    logging.debug(f"Fixed permissions for file: {file_path}")
                except PermissionError as e:
                    logging.error(f"Unable to fix permissions for file: {file_path}")
                    logging.error(f"Error: {str(e)}")
                    return False
    return True


def find_python_binary():
    possible_binaries = ["python3.10", "python3.9", "python3.8", "python3.7", "python3", "python"]

    if os_info.family == "Windows":
        possible_binaries = [binary + ".exe" for binary in possible_binaries] + possible_binaries

    for binary in possible_binaries:
        if shutil.which(binary):
            try:
                version_output = subprocess.check_output([binary, "--version"], stderr=subprocess.STDOUT).decode(
                    "utf-8")
                version_parts = version_output.strip().split(" ")[1].split(".")
                major, minor = int(version_parts[0]), int(version_parts[1])

                if major == 3 and minor >= 10:
                    return binary

            except (subprocess.CalledProcessError, IndexError, ValueError):
                continue

    return None


def install_python_dependencies(_dir, runpod):
    # Update pip
    logging.info("Checking for pip updates before Python operations.")
    if args.verbosity >= 2:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])

    # Install python dependencies
    logging.info("Installing python dependencies. This could take a long time as it downloads some large files.")

    # Set the paths for the built-in requirements and temporary requirements files
    requirements_path = os.path.join(_dir, "requirements.txt")
    logging.debug(f"Found requirements.txt at: {requirements_path}")
    if os.path.exists(requirements_path):
        temp_requirements = tempfile.NamedTemporaryFile(delete=False, mode="w+")
        try:
            found_comment = False
            with open(requirements_path, "r") as original_file:
                for line in original_file:
                    # Skip comments and empty lines
                    if line.strip().startswith("#") or not line.strip():
                        continue

                    logging.debug(f"Processing line: {line.strip()}")
                    if found_comment:
                        line = line.replace(".", _dir)
                        logging.debug(f"Replaced . with: {line}")
                        found_comment = False
                    elif re.search(r"#.*kohya_ss.*library", line):
                        logging.debug(f"Found kohya_ss library comment in line: {line.strip()}")
                        found_comment = True
                        continue
                    else:
                        logging.debug(f"Processing line without any conditions: {line.strip()}")

                    logging.debug(f"Installing: {line.strip()}")
                    temp_requirements.write(line)

                # Append the appropriate packages based on the conditionals
                if runpod:
                    temp_requirements.write("tensorrt\n")

                if os_info.family == "Darwin":
                    if platform.machine() == "arm64":
                        temp_requirements.write(f"tensorflow-macos=={TENSORFLOW_MACOS_VERSION}\n")
                        temp_requirements.write(f"tensorflow-metal=={TENSORFLOW_METAL_VERSION}\n")
                    elif platform.machine() == "x86_64":
                        temp_requirements.write(f"tensorflow=={TENSORFLOW_VERSION}\n")
                elif os_info.family == "Windows":
                    torch_installed = "torch" in [pkg.name.lower() for pkg in pkgutil.iter_modules()]
                    torchvision_installed = "torchvision" in [pkg.name.lower() for pkg in pkgutil.iter_modules()]
                    if not (torch_installed and torchvision_installed):
                        logging.info("Installing torch and torchvision packages")
                        if args.verbosity < 3:
                            subprocess.run(["pip", "install", "torch==1.12.1+cu116", "torchvision==0.13.1+cu116",
                                            "--extra-index-url", "https://download.pytorch.org/whl/cu116", "--quiet"])
                        else:
                            subprocess.run(["pip", "install", "torch==1.12.1+cu116", "torchvision==0.13.1+cu116",
                                            "--extra-index-url", "https://download.pytorch.org/whl/cu116"])

                if os_info.family == "Darwin":
                    macos_requirements_path = os.path.join(_dir, "requirements_macos.txt")
                    if os.path.exists(macos_requirements_path):
                        with open(macos_requirements_path, "r") as macos_req_file:
                            for line in macos_req_file:
                                # Skip comments and empty lines
                                if line.strip().startswith("#") or not line.strip():
                                    continue

                                logging.debug(f"Appending macOS requirement: {line.strip()}")
                                temp_requirements.write(line)

        finally:
            temp_requirements.flush()
            temp_requirements.close()

        logging.info("Installing required packages...")
        if args.verbosity >= 3:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "-r", temp_requirements.name])
        else:
            # Count the number of packages in the temporary requirements file
            with open(temp_requirements.name, "r") as f:
                num_packages = sum(1 for line in f if line.strip())

            with open(temp_requirements.name, "r") as f:
                for line in tqdm_progress(f, total=num_packages, desc="Installing packages", unit="package"):
                    package = line.strip()
                    if package:
                        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", package])

        # Delete the temporary requirements file
        logging.debug(f"Removing {temp_requirements.name}")
        if os.path.exists(temp_requirements.name):
            os.remove(temp_requirements.name)


def configure_accelerate(interactive):
    source_accelerate_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_files",
                                                 "accelerate", "default_config.yaml")
    logging.debug(f"Source accelerate config location: {source_accelerate_config_file}")

    if interactive:
        os.system("accelerate config")
    else:
        target_config_location = None

        if env_var_exists("HF_HOME"):
            target_config_location = Path(os.environ["HF_HOME"], "accelerate", "default_config.yaml")
        elif env_var_exists("XDG_CACHE_HOME"):
            target_config_location = Path(os.environ["XDG_CACHE_HOME"], "huggingface",
                                          "accelerate", "default_config.yaml")
        elif env_var_exists("HOME"):
            target_config_location = Path(os.environ["HOME"], ".cache", "huggingface",
                                          "accelerate", "default_config.yaml")

        if target_config_location:
            if not target_config_location.is_file():
                target_config_location.parent.mkdir(parents=True, exist_ok=True)
                logging.debug(f"Target accelerate config location: {target_config_location}")
                shutil.copyfile(source_accelerate_config_file, target_config_location)
                logging.debug(f"Copied accelerate config file to: {target_config_location}")
        else:
            logging.info("Could not place the accelerate configuration file. Please configure manually.")
            os.system("accelerate config")


def launch_kohya_gui(_args):
    if not in_container():
        _venv_path = os.path.join(_args.dir, "venv")
        kohya_gui_path = os.path.join(_args.dir, "kohya_gui.py")

        if not os.path.exists(_venv_path):
            logging.info("Error: Virtual environment not found")
            sys.exit(1)

        python_executable = os.path.join(_venv_path, "bin", "python") if sys.platform != "win32" else os.path.join(
            _venv_path, "Scripts", "python.exe")

        if not os.path.exists(python_executable):
            logging.info("Error: Python executable not found in the virtual environment")
            sys.exit(1)
    else:
        python_executable = sys.executable
        kohya_gui_path = os.path.join(_args.dir, "kohya_gui.py")

    cmd = [
        venv_python_bin, os.path.join(_args.dir, "kohya_gui.py"),
        "--listen", "127.0.0.1",
        "--server_port", "7861",
    ]

    if _args.username:
        cmd.extend(["--username", _args.username])

    if _args.password:
        cmd.extend(["--password", _args.password])

    try:
        logging.debug(f"Launching kohya_gui.py with Python bin: {venv_python_bin}")
        logging.debug(f"Running kohya_gui.py as: {cmd}")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logging.info("Process terminated by the user. Exiting...")
        sys.exit(0)


def main(_args=None):
    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        logging.info("Error: This script requires Python 3.10.")
        sys.exit(1)

    # Get the directory where the script is located
    script_directory = os.path.dirname(os.path.realpath(__file__))

    # Read config file or use defaults
    _config_file = _args.file if _args.file else os.path.join(script_directory,
                                                              "/config_files/installation"
                                                              "/install_config.yaml")
    config = load_config(_config_file)

    # Check for DIR in command line arguments, config file, or use the default
    if _args.dir:
        _dir = _args.dir
    elif config and 'Dir' in config['arguments']:
        _dir = config['arguments']['Dir']['default']
    else:
        _dir = get_default_dir(_args.runpod, script_directory)

    if not getattr(_args, "git-repo") or not _dir or not getattr(_args, "branch"):
        logging.info(
            "Error: gitRepo, Branch, and Dir must have a value. Please provide values in the config file or through "
            "command line arguments.")
        sys.exit(1)

    # Define the directories relative to the install directory needed for install and launch
    parent_dir = os.path.dirname(_dir)

    # The main logic will go here after the sanity checks.
    check_and_create_install_folder(parent_dir, _dir)
    check_storage_space(getattr(_args, "skip-space-check"), _args.dir, parent_dir)
    if update_kohya_ss(_args.dir, getattr(_args, "git-repo"), _args.branch, _args.update):
        if brew_install_tensorflow_deps(_args.verbosity):
            install_python_dependencies(_dir, _args.runpod)
            setup_file_links(site_packages_dir, _args.runpod)
            configure_accelerate(args.interactive)
            launch_kohya_gui(args)


if __name__ == "__main__":
    config_file = parse_file_arg()
    config_data = load_config(config_file)
    args = parse_args(config_data)

    # Initialize log_level with a default value
    log_level = logging.ERROR

    # Set logging level based on the verbosity count
    if args.verbosity == 0:
        log_level = logging.INFO
    elif args.verbosity == 1:
        log_level = logging.ERROR
    elif args.verbosity == 2:
        log_level = logging.WARNING
    elif args.verbosity >= 3:
        log_level = logging.DEBUG

    # Configure logging
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    # Use logging in the script like so
    # logging.debug("This is a debug message.")
    # logging.info("This is an info message.")
    # logging.warning("This is a warning message.")
    # logging.error("This is an error message.")
    os_info = get_os_info()

    # Store the original sys.executable value
    original_sys_executable = sys.executable

    # Print all arguments and their values in verbose 3 mode
    if args.verbosity >= 3:
        for k, v in args.__dict__.items():
            logging.debug(f"{k}: {v}")

    # Following check disabled as PyCharm can't detect it's being used in a subprocess
    # noinspection PyUnusedLocal
    python_bin = None
    venv_python_bin = None

    # Check if python3 or python3.10 binary exists
    python_bin = find_python_binary()
    if not python_bin:
        logging.error("Valid python3 or python3.10 binary not found.")
        logging.error("Cannot proceed with the python steps.")
        exit(1)

    # Create and activate virtual environment if not in container environment
    if not in_container():
        logging.info("Switching to virtual Python environment.")
        venv_path = os.path.join(args.dir, "venv")
        subprocess.run([python_bin, "-m", "venv", venv_path])

        # Check the virtual environment for permissions issues
        check_permissions(args.dir)

        # Activate the virtual environment
        venv_bin_dir = os.path.join(venv_path, "bin") if os.name != "nt" else os.path.join(venv_path, "Scripts")
        venv_python_bin = os.path.join(venv_bin_dir, python_bin)
        sys.executable = os.path.join(venv_python_bin)
        logging.debug(f"Python sys.executable: {sys.executable}")
        logging.debug(f"venv_path: {venv_path}")
        logging.debug(f"venv_bin_dir: {venv_bin_dir}")
        logging.debug(f"python_bin: {python_bin}")
        logging.debug(f"venv_python_bin: {venv_python_bin}")
        site_packages_dir = os.path.join(venv_path, "Lib", "site-packages")
    else:
        logging.info("In container, skipping virtual environment.")
        venv_python_bin = python_bin
        python_executable_dir = os.path.dirname(python_bin)
        if os.name == "Windows":
            site_packages_dir = os.path.join(python_executable_dir, "Lib", "site-packages")
        else:
            site_packages_dir = os.path.join(python_executable_dir, "..", "lib", "python" + sys.version[:3],
                                             "site-packages")

    if getattr(args, 'no-setup'):
        launch_kohya_gui(args)
        exit(0)
    else:
        main(args)
