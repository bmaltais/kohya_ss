#!/usr/bin/env python3

import argparse
import errno
import importlib
import logging
import mimetypes
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import redirect_stderr
from datetime import datetime
from getpass import getpass
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse


# This enables programmatically installing pip packages
def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


# noinspection GrazieInspection
def check_and_import(module_name, package_name=None, imports=None):
    """
    Check if a module is installed, and if not, install it and then import it.
    This function also allows importing specific objects from the module and
    adding them to sys.modules with a custom alias if needed.

    Parameters:
    module_name (str): The name of the module to be imported.
    package_name (str, optional): The name of the package to be installed if the module
                                   is not found. Defaults to the same as module_name.
    imports (list of tuples, optional): A list of objects to import from the module.
                                        Each tuple should contain the object name as its
                                        first element, and an optional alias as its second
                                        element. If no alias is provided, the object will
                                        be added to sys.modules using its original name.
                                        Defaults to None.

    Returns:
    module: The imported module.
    """
    if package_name is None:
        package_name = module_name

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        print(f"Installing {package_name}...")
        install_package(package_name)
        module = importlib.import_module(module_name)

    if imports:
        for obj_name, alias in imports:
            obj = getattr(module, obj_name)
            if alias:
                sys.modules[alias] = obj
            else:
                sys.modules[obj_name] = obj

    return module


base64 = check_and_import('base64')
requests = check_and_import('requests')
yaml = check_and_import('yaml', 'PyYAML')
git = check_and_import("git", "gitpython", imports=[("Repo", None), ("GitCommandError", None)])
# noinspection SpellCheckingInspection
tqdm_module = check_and_import("tqdm", "tqdm")
# noinspection SpellCheckingInspection
tqdm_progress = tqdm_module.tqdm

# Set the package versions at the beginning of the script to make them easy to modify as needed.
# These are used to control the Tensorflow versions for macOS.
# macOS x86_64
TENSORFLOW_VERSION = "2.12.0"

# macOS Apple Silicon
TENSORFLOW_MACOS_VERSION = "2.12.0"
TENSORFLOW_METAL_VERSION = "0.8.0"

# These are used to control Torch V1 version and source
TORCH_VERSION_1 = "1.12.1+cu116"
TORCHVISION_VERSION_1 = "0.13.1+cu116"
TORCH_INDEX_URL_1 = "https://download.pytorch.org/whl/cu116"
# noinspection SpellCheckingInspection
XFORMERS_VERSION_1 = "0.0.14"

# These are used to control Torch V2 + dependencies versions and sources
TORCH_VERSION_2 = "2.0.0+cu118"
TORCHVISION_VERSION_2 = "0.15.1+cu118"
TORCH_INDEX_URL_2 = "https://download.pytorch.org/whl/cu118"
TRITON_URL_2 = "https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl"
# noinspection SpellCheckingInspection
XFORMERS_VERSION_2 = "0.0.17"


# noinspection DuplicatedCode
def find_config_file(config_file_locations):
    for location in config_file_locations:
        abs_location = os.path.abspath(location)
        if os.path.isfile(abs_location):
            return abs_location
    return None


# noinspection DuplicatedCode,SpellCheckingInspection
def load_config(_config_file=None):
    # Define config file locations
    if sys.platform == "win32":
        config_file_locations = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_files", "installation",
                         "install_config.yml"),
            os.path.join(os.environ.get("USERPROFILE", ""), ".kohya_ss", "install_config.yml"),
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "install_config.yml")
        ]
    else:
        config_file_locations = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_files", "installation",
                         "install_config.yml"),
            os.path.join(os.environ.get("HOME", ""), ".kohya_ss", "install_config.yml"),
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "install_config.yml"),
        ]

    # Load and merge default config files
    _config_data = {}
    for location in config_file_locations:
        try:
            with open(location, 'r') as f:
                file_config_data = yaml.safe_load(f)
                if file_config_data:
                    _config_data = {**file_config_data, **_config_data}
        except FileNotFoundError:
            pass

    # Load and merge user-specified config file
    if _config_file is not None:
        try:
            with open(_config_file, 'r') as f:
                file_config_data = yaml.safe_load(f)
                if file_config_data:
                    _config_data = {**file_config_data, **_config_data}
        except FileNotFoundError:
            pass

    return _config_data if _config_data else None


# noinspection DuplicatedCode
def parse_file_arg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", "--file", dest="config_file", default=None,
                        help="Path to the configuration file.")
    _args, _ = parser.parse_known_args()
    if _args.config_file is not None:
        # print(f"Configuration file specified by command line: {os.path.abspath(_args.config_file)}")
        return os.path.abspath(_args.config_file)
    else:
        return None


def normalize_paths(_args, default_args):
    for arg in default_args:
        arg_name = arg["long"][2:].replace("-", "_")
        default_value = arg["default"]
        is_path = arg.get("is_path", False)
        if is_path and isinstance(default_value, str):
            path_value = getattr(_args, arg_name, None)
            if path_value and isinstance(path_value, str):
                expanded_path_value = os.path.expanduser(path_value)
                if not os.path.isabs(expanded_path_value):
                    expanded_path_value = os.path.abspath(expanded_path_value)
                setattr(_args, arg_name, expanded_path_value)


# This custom action was added so that the v option could be used Windows-style with integers (-v 3) setting the
# verbosity and Unix style (-vvv).
# noinspection DuplicatedCode
class CountOccurrencesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
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

        # Check if verbosity level is a non-negative integer
        if getattr(namespace, self.dest) < 0:
            logging.error('Verbosity level must be a positive integer')
            exit(1)


# This custom action checks if the provided --torch-version value is valid.
# Valid versions are listed in valid_torch_versions.
class CheckTorchVersionAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        valid_torch_versions = [0, 1, 2]
        if int(values) not in valid_torch_versions:
            parser.error(f'\nInvalid value for --torch-version: {values}. '
                         f'Valid values are {", ".join(map(str, valid_torch_versions))}.')
        setattr(namespace, self.dest, int(values))


# noinspection SpellCheckingInspection
def parse_args(_config_data):
    # Define the default arguments first. The spacing is purely for readability.
    default_args = [
        {"short": "-b", "long": "--branch", "default": "master", "type": str,
         "help": "Select which branch of kohya to check out on new installs."},

        {"short": "-d", "long": "--dir", "default": os.path.dirname(os.path.realpath(__file__)), "type": str,
         "help": "The full path you want kohya_ss installed to.", "is_path": True},

        {"short": "-f", "long": "--file", "default": "install_config.yml", "type": str,
         "help": "Configuration file with installation options.", "is_path": True},

        {"short": "-g", "long": "--git-repo", "default": "https://github.com/bmaltais/kohya_ss.git", "type": str,
         "help": "You can optionally provide a git repo to check out. Useful for custom forks."},

        {"short": "", "long": "--headless", "default": False, "type": bool,
         "help": "Headless mode will not display the native windowing toolkit. Useful for remote deployments."},

        {"short": "-i", "long": "--interactive", "default": False, "type": bool,
         "help": "Interactively configure accelerate instead of using value config file."},

        {"short": "-l", "long": "--log-dir", "default": None, "type": str,
         "help": "Override the default log directory.", "is_path": True},

        {"short": "-n", "long": "--no-setup", "default": False, "type": bool,
         "help": "Skip setup operations and launch the GUI."},

        {"short": "-r", "long": "--repair", "default": False, "type": bool,
         "help": "This runs the installation repair operations. These could take a few minutes to run."},

        {"short": None, "long": "--setup-only", "default": False, "type": bool,
         "help": "Do not launch GUI. Only conduct setup operations."},

        {"short": "-s", "long": "--skip-space-check", "default": False, "type": bool,
         "help": "Skip the 10Gb minimum storage space check."},

        {"short": "-t", "long": "--torch-version", "default": 0, "type": int,
         "help": "Configure the major version of Torch.", "action": CheckTorchVersionAction},

        {"short": "-u", "long": "--update", "default": False, "type": bool,
         "help": "Update kohya_ss with specified branch, repo, or latest stable if git's unavailable."},

        {"short": "-v", "long": "--verbosity", "default": '0', "type": str,
         "help": "Increase verbosity levels. Use multiple times (e.g., -vvv) or specify number (e.g., -v 4).",
         "action": CountOccurrencesAction},

        {"short": None, "long": "--listen", "default": "127.0.0.1", "type": str,
         "help": "IP to listen on for connections to Gradio."},

        {"short": "", "long": "--username", "default": "", "type": str, "help": "Username for authentication."},

        {"short": "", "long": "--password", "default": "", "type": str, "help": "Password for authentication."},

        {"short": "", "long": "--server-port", "default": 0, "type": str,
         "help": "The port number the GUI server should use."},

        {"short": "", "long": "--inbrowser", "default": False, "type": bool, "help": "Open in browser."},

        {"short": "", "long": "--share", "default": False, "type": bool, "help": "Share the gradio UI."},
    ]

    # noinspection DuplicatedCode
    def generate_usage(_default_args):
        """
        This function generates nicer usage string for the command line arguments in the form of [ -s | --long VAR ].
        :param _default_args: List of default argument dictionaries
        :return: Usage string
        """
        usage = "usage: launcher.py "
        for _arg in _default_args:
            _arg_type = _arg.get("type", str)
            _arg_type = _arg_type.__name__.upper()  # Get the name of the type and convert to upper case
            _short_opt = _arg["short"]
            _long_opt = _arg["long"]
            if _short_opt:
                usage += f'[{_short_opt} | {_long_opt} {_arg_type if _arg_type != "BOOL" else ""}] '
            else:
                usage += f'[{_long_opt} {_arg_type if _arg_type != "BOOL" else ""}] '
        return usage

    # usage is generated dynamically here
    parser = argparse.ArgumentParser(
        description="Launcher script for Kohya_SS. This script helps you configure, install, and launch the Kohya_SS "
                    "application.",
        usage=generate_usage(default_args),
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
                parser.add_argument(short_opt, long_opt, dest=None, action=custom_action, nargs='?',
                                    default=default_value,
                                    type=str, help=help_text)
            else:
                parser.add_argument(long_opt, dest=long_opt[2:].replace("-", "_"), action=custom_action, nargs='?',
                                    default=default_value, type=str, help=help_text)

        elif isinstance(default_value, bool):
            action = 'store_true' if default_value is False else 'store_false'
            if short_opt:
                parser.add_argument(short_opt, long_opt, dest=long_opt[2:], action=action, default=default_value,
                                    help=help_text)
            else:
                parser.add_argument(long_opt, dest=long_opt[2:].replace("-", "_"), action=action, default=default_value,
                                    help=help_text)
        else:
            if short_opt:
                parser.add_argument(short_opt, long_opt, dest=long_opt[2:], default=default_value, type=arg_type,
                                    help=help_text)
            else:
                parser.add_argument(long_opt, dest=long_opt[2:].replace("-", "_"), default=default_value, type=arg_type,
                                    help=help_text)

    _args = parser.parse_args()
    _args.verbosity = int(_args.verbosity)

    # Normalize paths to ensure absolute paths
    normalize_paths(_args, default_args)

    # Replace the placeholder with the script directory
    for arg, value in vars(_args).items():
        if arg == 'dir' and '_CURRENT_SCRIPT_DIR_' in value:
            script_directory = os.path.dirname(os.path.realpath(__file__))
            setattr(_args, arg, script_directory)
    return _args


def env_var_exists(var_name):
    return var_name in os.environ and os.environ[var_name] != ""


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
        if size_available(_dir, parent_dir) < 15:
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
        if os.path.isfile(target_file):
            logging.info(f"Linking {os.path.basename(symlink)}.")
            os.symlink(target_file, symlink)
        else:
            logging.warning(f"{target_file} does not exist. Nothing to link.")


# noinspection SpellCheckingInspection
def post_pip_prepare_environment_files_and_folders(_site_packages_dir, _dir):
    if in_container:
        # Symlink paths for libnvinfer.so.7 and libnvinfer_plugin.so.7
        libnvinfer_symlink = "/usr/lib/x86_64-linux-gnu/libnvinfer.so.7"
        libnvinfer_plugin_symlink = "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7"

        # Target file paths for libnvinfer.so.7 and libnvinfer_plugin.so.7
        libnvinfer_target = "/usr/lib/x86_64-linux-gnu/libnvinfer.so"
        libnvinfer_plugin_target = "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so"

        create_symlinks(libnvinfer_symlink, libnvinfer_target)
        create_symlinks(libnvinfer_plugin_symlink, libnvinfer_plugin_target)

        # Symlink paths
        libnvinfer_plugin_symlink = os.path.join(_site_packages_dir, "tensorrt", "libnvinfer_plugin.so.8")
        libnvinfer_symlink = os.path.join(_site_packages_dir, "tensorrt", "libnvinfer.so.7")
        libcudart_symlink = os.path.join(_site_packages_dir, "nvidia", "cuda_runtime", "lib", "libcudart.so.11.0")

        # Target file paths
        libnvinfer_plugin_target = os.path.join(_site_packages_dir, "tensorrt", "libnvinfer_plugin.so.7")
        libnvinfer_target = os.path.join(_site_packages_dir, "tensorrt", "libnvinfer.so.8")
        libcudart_target = os.path.join(_site_packages_dir, "nvidia", "cuda_runtime", "lib", "libcudart.so.12")

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


# noinspection SpellCheckingInspection
def in_container():
    cgroup_path = "/proc/1/cgroup"
    sched_path = "/proc/1/sched"

    if not os.path.isfile(cgroup_path) or not os.path.isfile(sched_path):
        return False

    with open(cgroup_path, "r") as cgroup_file:
        content = cgroup_file.read()

    with open(sched_path, "r") as sched_file:
        sched_content = sched_file.readline()

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

    if (any(re.search(pattern, content) for pattern in container_indicators) or
            os.path.exists('/.dockerenv') or
            not sched_content.startswith("systemd (1")):
        return True

    return False


class GitAuthenticationError(Exception):
    def __init__(self):
        logging.critical(f"Authentication error in git.")
        super().__init__(f"Authentication error in git.")


class UpdateSkippedException(Exception):
    pass


class UncommittedChangesException(Exception):
    def __init__(self, local_git_repo):
        logging.critical(f"Uncommitted changes in {local_git_repo}")
        super().__init__(f"Uncommitted changes in {local_git_repo}")


def is_git_installed():
    git_commands = ["git"]
    if os_info.family == "Windows":
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


def run_git_command(repo, _args, username=None, password=None):
    if username and password:
        repo.git.update_environment(GIT_USERNAME=username, GIT_PASSWORD=password)

    try:
        git_cmd = _args
        logging.debug(f"Executing git command: {git_cmd}")
        result = repo.git.execute(git_cmd)

        if result:
            logging.debug(f"Git command completed successfully. stdout: {result}")
            return result, None
        else:
            logging.error(f"Git command failed.")
            return None, "Git command failed."

    except git.GitCommandError as _e:
        logging.error(f"Git command failed with error code {_e.status}: {_e.stderr.strip()}")
        return None, _e.stderr.strip()

    except Exception as _e:
        logging.error(f"Unexpected error occurred during git operations: {str(_e)}")
        return None, str(_e)


def get_latest_tag(git_repo):
    repo_name = git_repo.split("/")[-1].rstrip(".git")
    owner = git_repo.split("/")[-2]

    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/releases/latest"
    response = requests.get(api_url)

    if response.status_code != 200:
        raise Exception(f"Failed to get the latest release: {response.status_code}")

    data = response.json()
    return data["tag_name"]


def find_ssh_private_key_path(git_repo):
    def get_git_config_ssh_key(_host):
        try:
            output = subprocess.check_output(["git", "config", f"host.{_host}.identityFile"], universal_newlines=True)
            key_path = output.strip()
            if os.path.exists(key_path):
                logging.debug(f"Found SSH key in Git config for host '{_host}': {key_path}")
                return key_path
        except subprocess.CalledProcessError:
            pass
        return None

    def find_key_in_ssh_folder():
        home = os.path.expanduser("~")
        ssh_folder = os.path.join(home, ".ssh")
        common_key_names = ["id_rsa", "id_ecdsa", "id_ed25519"]

        for file_name in os.listdir(ssh_folder):
            full_path = os.path.join(ssh_folder, file_name)
            if file_name in common_key_names and os.path.isfile(full_path):
                logging.debug(f"Found SSH key in .ssh folder: {full_path}")
                return full_path

        logging.debug(f"No common SSH key names found in .ssh folder: {ssh_folder}")
        return None

    parsed_git_url = urlparse(git_repo)
    host = parsed_git_url.hostname

    # Check Git configuration for a specified key
    git_config_key = get_git_config_ssh_key(host)
    if git_config_key:
        return git_config_key

    # Look for common key names in the .ssh folder
    key_in_ssh_folder = find_key_in_ssh_folder()
    if key_in_ssh_folder:
        return key_in_ssh_folder

    logging.debug(f"No SSH key found for git repo: {git_repo}")
    return None


class GitProgressPrinter(git.remote.RemoteProgress):
    def __init__(self, operation, repo_name, local_folder, total=None):
        super().__init__()
        self.operation = operation
        self.repo_name = repo_name
        self.local_folder = local_folder
        self.total = total
        self.received_objects = 0

    # noinspection PyUnusedLocal
    def update(self, op_code, cur_count, max_count=None, message="", binary=False):
        if self.total is None and max_count:
            self.total = max_count
            logging.critical(
                f"{self.operation.capitalize()} {self.repo_name} to {self.local_folder}\n Total objects: {self.total}")

        if cur_count > self.received_objects:
            self.received_objects = cur_count
            print(f"Received {self.received_objects}/{self.total} objects", end='\r')


# noinspection SpellCheckingInspection
def update_kohya_ss(_dir, git_repo, branch, update, repair=None):
    logging.debug(f"Update: {update}")
    logging.debug(f"Items detected in _dir: {os.listdir(_dir)}")
    logging.debug(f".git detected: {'.git' in os.listdir(_dir)}")

    def has_uncommitted_changes(local_git_repo):
        try:
            git_status_output = local_git_repo.git.status("--porcelain")
        except git.GitCommandError as e_:
            logging.error(f"Error while checking for uncommitted changes: {e_}")
            git_status_output = None

        if git_status_output is not None and git_status_output.strip():
            logging.warning(f"Uncommitted changes detected: {git_status_output.strip()}")

        return len(git_status_output) > 0 if git_status_output is not None else False

    def git_operations_with_credentials(_dir, _git_repo, _branch, _update, _username=None, _password=None):
        git_folder_present = os.path.exists(os.path.join(_dir, ".git"))
        venv_folder_present = os.path.exists(os.path.join(_dir, "venv"))

        parsed_git_url = urlparse(_git_repo)

        if parsed_git_url.scheme == 'https':
            # For HTTPS, set the username and password
            git_credentials = {'username': _username, 'password': _password}
        elif parsed_git_url.scheme == 'ssh':
            # For SSH, we will find the default key.
            private_key_path = find_ssh_private_key_path(_git_repo)
            env = os.environ.copy()
            env["GIT_SSH_COMMAND"] = f"ssh -i {private_key_path} -o StrictHostKeyChecking=no -F /dev/null"
            git_credentials = {'env': env}
        elif os.path.isdir(_git_repo):
            # For local folder paths, normalize to an absolute path
            local_repo_path = os.path.abspath(_git_repo)
            git_credentials = {'local_repo_path': local_repo_path}
        else:
            raise ValueError(
                "Invalid Git URL scheme or local folder path. Only 'https', 'ssh', and local folder paths are "
                "supported.")

        _success = False
        _error = None

        try:
            if git_folder_present and not _update:
                logging.info(f"A git repo was detected at {_dir}, but update was not enabled. "
                             f"Skipping updating folder contents.")
                raise UpdateSkippedException()
            elif git_folder_present and (_update or repair):
                local_git_repo = git.Repo(_dir)

                if has_uncommitted_changes(local_git_repo):
                    raise UncommittedChangesException(local_git_repo)

                logging.debug("git pull operation entered.")

                pull_progress = GitProgressPrinter(operation="pull", repo_name=_git_repo, local_folder=_dir)

                if 'local_repo_path' in git_credentials:
                    remote_name = "local_repo"
                    if remote_name not in [remote.name for remote in local_git_repo.remotes]:
                        local_git_repo.create_remote(remote_name, git_credentials['local_repo_path'])
                    local_git_repo.remotes[remote_name].fetch(progress=pull_progress)
                    local_git_repo.git.merge('FETCH_HEAD')
                    local_git_repo.delete_remote(remote_name)
                else:
                    local_git_repo.remotes.origin.pull(progress=pull_progress, **git_credentials)

                _success = True
                return _success, _error

            elif not git_folder_present:
                if len(os.listdir(_dir)) in (0, 1, 2) or \
                        (len(os.listdir(_dir)) == 2 and 'venv' in os.listdir(_dir) and 'logs' in os.listdir(_dir)):
                    tmp_venv_path = None
                    tmp_logs_path = None
                    if venv_folder_present:
                        tmp_venv_path = os.path.join(tempfile.mkdtemp(), "venv")
                        try:
                            shutil.move(os.path.join(_dir, "venv"), tmp_venv_path)
                        except Exception as e_:
                            logging.error(f"Failed to move venv folder: {e_}")
                            return False, str(e_)

                    logs_folder_present = os.path.exists(os.path.join(_dir, "logs"))
                    # Find the log FileHandler
                    logger = logging.getLogger()
                    file_handler = None
                    for _handler in logger.handlers:
                        if isinstance(_handler, logging.FileHandler):
                            file_handler = _handler
                            break

                    if logs_folder_present and file_handler is not None:
                        tmp_logs_path = os.path.join(tempfile.mkdtemp(), "logs")
                        try:
                            # Close the current log file handler and remove it from the logger
                            logging.debug("Closing log file handler and moving it to {tmp_logs_path}.")
                            file_handler.close()
                            logger.removeHandler(file_handler)
                            shutil.move(os.path.join(_dir, "logs"), tmp_logs_path)
                        except Exception as e_:
                            logging.error(f"Failed to move logs folder: {e_}")
                            return False, str(e_)

                    try:
                        logging.debug("git clone operation entered.")
                        progress_printer = GitProgressPrinter("clone", _git_repo, _dir)

                        if os.path.isdir(_git_repo):
                            _git_repo = os.path.abspath(_git_repo)
                            git_credentials = {'url': _git_repo}

                        if 'url' in git_credentials:
                            git.Repo.clone_from(git_credentials['url'], _dir, branch=_branch, depth=1,
                                                progress=progress_printer)
                        elif 'env' in git_credentials:
                            git.Repo.clone_from(_git_repo, _dir, branch=_branch, depth=1, progress=progress_printer,
                                                env=git_credentials['env'])
                        else:
                            git.Repo.clone_from(_git_repo, _dir, branch=_branch, depth=1, progress=progress_printer,
                                                username=git_credentials['username'],
                                                password=git_credentials['password'])
                    except git.GitCommandError as e_:
                        logging.error(f"Git clone operation failed: {e_}")
                        return False, str(e_)

                    if venv_folder_present and tmp_venv_path is not None:
                        try:
                            shutil.move(tmp_venv_path, os.path.join(_dir, "venv"))
                        except Exception as e_:
                            logging.error(f"Failed to move venv folder back: {e_}")
                            return False, str(e_)

                    if logs_folder_present and tmp_logs_path is not None:
                        try:
                            shutil.move(tmp_logs_path, os.path.join(_dir, "logs"))

                            # After moving the log folder back to its original position
                            # Reopen the log file and file handler to resume original log
                            file_handler = logging.FileHandler(log_file, mode='a')
                            file_handler.setFormatter(CustomFormatter())
                            logger.addHandler(file_handler)
                            logging.debug(
                                "Reopened log file handler and moved it back to {os.path.join(_dir, 'logs')}.")
                        except Exception as e_:
                            logging.error(f"Failed to move logs folder back: {e_}")
                            return False, str(e_)

                    _success = True
                    return _success, _error
                elif (not git_folder_present or venv_folder_present) and (len(os.listdir(_dir)) > 2) and (not update):
                    logging.critical("We have detected a current non-git installation, but --update flag not used. "
                                     "Skipping git clone operation.")
                    return False, "Non-git installation detected, update flag not used"
                else:
                    return False, "Detected non-git installation and the folder has data. Skipping git operations to " \
                                  "avoid data corruption."

        except git.GitCommandError as e_:
            logging.warning(f"Git command error: {e_}")
            _success = False
            if "Authentication failed" in str(e_):
                _error = "authentication_error"
            else:
                _error = "unknown_error"
            return _success, _error
        except UpdateSkippedException:
            _success = True
            _error = "update_skipped"
            return _success, _error
        except UncommittedChangesException:
            _success = False
            _error = "uncommitted_changes"
            return _success, _error

    git_installed = is_git_installed()
    max_attempts = 4
    username = None
    password = None
    success = False

    try:
        if git_installed:
            if os.path.exists(_dir) and os.path.isdir(_dir):
                for _attempt in range(max_attempts):
                    success, error_type = git_operations_with_credentials(_dir, git_repo, branch,
                                                                          update, username, password)
                    if success:
                        break
                    elif error_type == "uncommitted_changes":
                        # If there are uncommitted changes, break the loop and proceed with the fallback method
                        if update or repair:
                            # If we are in update or repair mode, we should continue. We might still be able to rescue
                            # the installation if it was pip or another downstream component that broke.
                            logging.critical("Uncommitted_changes detected in the repo. Skipping repo update.")
                            success = True
                            return success
                        success = False
                        return success
                    elif error_type == "authentication_error":
                        # Prompt for new credentials in case of authentication error
                        logging.info("Please enter your Git credentials:")
                        username = input("Username: ")
                        password = getpass("Password: ")
                    elif error_type == "update_skipped":
                        success = True
                        return success
                    else:
                        # For unexpected errors, break the loop and proceed with the fallback method
                        logging.warning(
                            "An unexpected error occurred during git operations. Proceeding with the fallback method.")
                        break
        else:
            raise Exception("Git not installed.")
    except Exception as _e:
        logging.warning(f"Failed to clone or update the repository using git: {_e}")

    if not success:
        # Check if the directory is empty or contains only a "venv" folder, the branch is "master",
        # and the Git repository URL starts with "https://github.com/bmaltais/kohya_ss" or the update flag is specified.
        # If all conditions are met, we try to download the latest tag as a zip for installation.
        # We only overwrite the files we download. Otherwise, skip the installation.
        if (update or len(os.listdir(_dir)) == 0 or
            (len(os.listdir(_dir)) in [1, 2] and os.path.exists(os.path.join(_dir, "venv")) and os.path.exists(
                os.path.join(_dir, "logs")))) and \
                (not branch or branch != "master") and (
                not git_repo or git_repo.startswith("https://github.com/bmaltais/kohya_ss")):

            # Download the latest release as a zip file from the default repository
            try:
                # Download the repo as a zip file
                # Remove .git extension if present
                git_repo = git_repo.rstrip('.git')
                download_url = git_repo.rstrip("/") + f"/archive/refs/tags/{get_latest_tag(git_repo)}.zip"
                auth = (username, password) if username and password else None
                logging.critical(f"Attempting to download from: {download_url}")
                response = requests.get(download_url, auth=auth)
                logging.debug(f"Zip download response: {response.status_code}, {response.text}")

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
                    if args.versbosity < 3:
                        write_to_log(f"Zip file downloaded to: {temp_zip.name}")
                    progress_bar.close()

                    # Extract the zip file to a temporary directory
                    with zipfile.ZipFile(temp_zip.name, "r") as zip_ref:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            zip_ref.extractall(temp_dir)
                            logging.debug(f"Zip file extracted to: {temp_dir}")
                            if args.versbosity < 3:
                                write_to_log(f"Zip file extracted to: {temp_dir}")

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

            except Exception as _e:
                logging.warning(f"Failed to download the latest release: {_e}")

        elif not git_repo.startswith("https://github.com/bmaltais/kohya_ss") or branch != "master":
            logging.info("Sorry, we only support zip file updates for master branch on "
                         "https://github.com/bmaltais/kohya_ss")

            # If we have a valid installation, but we're unable to update the entire project, we still want to
            # continue and update all the pip components.
            if (len(os.listdir(_dir)) > 2 and os.path.exists(os.path.join(_dir, "pyproject.toml")) and
                    os.path.exists(os.path.join(_dir, "kohya_gui.py"))):
                success = True
            else:
                success = False
        elif len(os.listdir(_dir)) > 1:
            logging.critical("Non-git installation detected, but --update flag not used. Skipping release zip file "
                             "download attempt.")
            success = True
        else:
            logging.error("We could not download the latest release via git or zip file.")
            success = False

    logging.debug(f"Kohya Update success: {success}")
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
            except Exception as _e:
                logging.error(f"Error reading /System/Library/CoreServices/SystemVersion.plist: {_e}")

        elif system == "Linux":
            if os.path.exists("/etc/os-release"):
                try:
                    with open("/etc/os-release", "r") as f:
                        content = f.read()
                        self.name = re.search(r'ID="?([^"\n]+)', content).group(1)
                        self.family = re.search(r'ID_LIKE="?([^"\n]+)', content).group(1)
                        self.version = re.search(r'VERSION="?([^"\n]+)', content).group(1)
                except Exception as _e:
                    logging.error(f"Error reading /etc/os-release: {_e}")

            elif os.path.exists("/etc/redhat-release"):
                try:
                    with open("/etc/redhat-release", "r") as f:
                        content = f.read()
                        match = re.search(r'([^ ]+) release ([^ ]+)', content)
                        if match:
                            self.name = match.group(1)
                            self.family = "RedHat"
                            self.version = match.group(2)
                except Exception as _e:
                    logging.error(f"Error reading /etc/redhat-release: {_e}")

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
                except Exception as _e:
                    logging.error(f"Error executing uname command: {_e}")
                    self.name = "Generic Linux"
                    self.family = "Generic Linux"
            return {
                "name": self.name,
                "family": self.family,
                "version": self.version
            }
        elif system == "FreeBSD":
            self.name = "FreeBSD"
            self.family = "FreeBSD"
            self.version = "Unknown"

            # Try the `uname -r` command first to get FreeBSD version number
            try:
                self.version = subprocess.getoutput("uname -r")
            except Exception as _e:
                logging.warning(f"Error executing uname command: {_e}")

                # If `uname -r` fails, try using platform.release()
                try:
                    self.version = platform.release()
                except Exception as _e:
                    logging.error(f"Error using platform.release(): {_e}")
                    self.version = "Unknown"


def get_os_info():
    return OSInfo()


def brew_install_tensorflow_deps(verbosity=1):
    # noinspection SpellCheckingInspection
    brew_install_cmd = "brew install llvm cctools ld64"

    def brew_installed():
        if os_info.family == "macOS":
            try:
                subprocess.run(["brew", "-v"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            except subprocess.CalledProcessError:
                return False

    if not os_info.family == "macOS":
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
        except subprocess.CalledProcessError as _e:
            if verbosity >= 1:
                logging.error(_e.stderr.decode('utf-8'))
            return False


def check_permissions(_dir):
    venv_directory = os.path.join(_dir, "venv")

    for root, dirs, files in os.walk(venv_directory):
        # Skip site-packages directory
        if root.startswith(os.path.join(venv_directory, "Lib", "site-packages")):
            continue

        if root.startswith(os.path.join(venv_directory, "share", "doc")):
            continue

        for file in files:
            file_path = os.path.join(root, file)

            try:
                file_stat = os.stat(file_path)
            except PermissionError:
                logging.debug(f"Unable to get permissions for file: {file_path}")
                continue

            current_permissions = file_stat.st_mode

            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type == 'application/x-executable' or mime_type == 'application/octet-stream':
                required_permissions = 0o755
            else:
                required_permissions = 0o644

            missing_permissions = required_permissions & ~current_permissions
            if missing_permissions:
                logging.debug(f"Missing permissions on file: {file_path}")

                try:
                    os.chmod(file_path, current_permissions | missing_permissions)
                    logging.debug(f"Fixed permissions for file: {file_path}")
                except PermissionError as _e:
                    folder_name, file_name = os.path.split(file_path)

                    if "bin" not in folder_name and "python" not in file_name:
                        logging.debug(f"Unable to fix permissions for file: {file_path}")
                    pass
                except Exception as _e:
                    logging.debug(f"{str(_e)}")
                    pass

    return True


# noinspection SpellCheckingInspection
def pre_pip_prepare_environment_files_and_folders(_dir, _site_packages_dir):
    if os_info.family == "Windows":
        bitsandbytes_site_packages_dir = os.path.join(_site_packages_dir, "bitsandbytes")

        # Remove directories if they exist
        if os.path.exists(bitsandbytes_site_packages_dir):
            logging.debug(f"Removing folder {bitsandbytes_site_packages_dir}")
            shutil.rmtree(bitsandbytes_site_packages_dir)

        bitsandbytes_source = os.path.join(_dir, "bitsandbytes_windows")
        bitsandbytes_dest = os.path.join(_site_packages_dir, "bitsandbytes")
        bitsandbytes_cuda_dest = os.path.join(_site_packages_dir, "bitsandbytes", "cuda_setup")

        if os.path.exists(bitsandbytes_source):
            # Create destination directories if they don't exist
            try:
                os.makedirs(bitsandbytes_dest, exist_ok=True)
            except OSError as _e:
                if _e.errno != errno.EEXIST:
                    raise
            try:
                os.makedirs(bitsandbytes_cuda_dest, exist_ok=True)
            except OSError as _e:
                if _e.errno != errno.EEXIST:
                    raise

            # Copy .dll files
            for file in os.listdir(bitsandbytes_source):
                if file.endswith(".dll"):
                    shutil.copy(os.path.join(bitsandbytes_source, file), bitsandbytes_dest)
                    logging.debug(f"Copying {os.path.join(bitsandbytes_source, file)}"
                                  f"to {os.path.join(bitsandbytes_dest, file)}")

            # Copy cextension.py
            shutil.copy(os.path.join(bitsandbytes_source, "cextension.py"),
                        os.path.join(bitsandbytes_dest, "cextension.py"))
            logging.debug(f"Copying {os.path.join(bitsandbytes_source, 'cextension.py')}")

            # Copy main.py
            shutil.copy(os.path.join(bitsandbytes_source, "main.py"), os.path.join(bitsandbytes_cuda_dest, "main.py"))
            logging.debug(f"Copying {os.path.join(bitsandbytes_source, 'main.py')} to "
                          f"{os.path.join(bitsandbytes_cuda_dest, 'main.py')}")


# noinspection DuplicatedCode
def find_python_binary():
    possible_binaries = ["python3.10", "python310", "python3", "python"]

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
                    logging.debug(f"Python binary found: {binary}")
                    return binary

            except (subprocess.CalledProcessError, IndexError, ValueError):
                continue

    return None


# noinspection SpellCheckingInspection
def safe_subprocess_run(_command, stop_flag=None, interactive=False,
                        capture_output=False, text=False, bufsize=1):
    stdout_option = None
    stderr_option = None

    if not interactive:
        stdout_option = subprocess.PIPE
        stderr_option = subprocess.STDOUT

    with subprocess.Popen(_command, stdout=stdout_option, stderr=stderr_option,
                          bufsize=bufsize, universal_newlines=text) as p:
        output = ""
        if interactive:
            if stop_flag and stop_flag.is_set():
                p.terminate()
        else:
            output_lines = []
            for _line in p.stdout:
                if stop_flag and stop_flag.is_set():
                    p.terminate()
                    break
                output_lines.append(_line)
                if args.verbosity >= 3:
                    if isinstance(_line, bytes):
                        _line = _line.decode('utf-8')
                    logging.debug(_line.strip())
            if capture_output:
                output = "".join(output_lines)

        # Wait for process to terminate to get the return code
        time.sleep(0.5)
        p.wait()
        if p.returncode != 0:
            logging.error(f"Command '{' '.join(_command)}' returned non-zero exit status {p.returncode}")
            raise subprocess.CalledProcessError(p.returncode, _command, output=output)

    if capture_output:
        completed_process = subprocess.CompletedProcess(args=_command, returncode=p.returncode, stdout=output)
        return completed_process


def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor


def spinner_task(stop_event):
    cursor = spinning_cursor()
    while not stop_event.is_set():
        sys.stdout.write(next(cursor))
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\b')
        sys.stdout.flush()


class PackageManager:
    """
    PackageManager is a class that represents the installed packages in the environment.
    It provides methods to access package version information, query packages by version,
    and check for the presence of specific packages.
    """

    def __init__(self, packages: Dict[str, str]):
        self._packages = packages

    @classmethod
    def from_installed_packages(cls) -> "PackageManager":
        packages = {}
        output = safe_subprocess_run(["pip", "freeze"], capture_output=True, text=True)
        for line in output.stdout.strip().split("\n"):
            parts = line.split("==")
            if len(parts) == 2:
                name, version = parts
                packages[name.lower()] = version
        return cls(packages)

    def add_package(self, name: str, version: str):
        self._packages[name.lower()] = version

    def get_version(self, package_name: str) -> str:
        return self._packages.get(package_name.lower())

    def get_packages_by_version(self, version: str) -> List[str]:
        return [name for name, pkg_version in self._packages.items() if pkg_version == version]

    def contains(self, package_name: str) -> bool:
        return package_name.lower() in self._packages

    def get_installed_package_names(self) -> List[str]:
        return list(self._packages.keys())


# noinspection SpellCheckingInspection,GrazieInspection
def install_python_dependencies(_dir, torch_version, update=False, repair=False,
                                interactive=False, _log_dir=None):
    # Initialize package manager class first to be used in many Python operations
    package_manager = PackageManager.from_installed_packages()

    # noinspection SpellCheckingInspection,DuplicatedCode
    def install_or_repair_torch(_update, _repair, _interactive, _torch_version):
        logging.debug("Looking for pre-existing torch packages.")

        # PackageManager functionality demo:
        if package_manager.contains('torch'):
            logging.debug(f"Package Manager functions: ", )
            logging.debug(f"Torch version: {package_manager.get_version('torch')}", )
            logging.debug(f"Packages with version 1.12.1+cu116: "
                          f"{package_manager.get_packages_by_version('1.12.1+cu116')}")
            logging.debug(f"Is Torch in the package list? {package_manager.contains('torch')}")
        else:
            logging.debug("Torch installation not detected.")

        logging.debug(f"Torch Version to install: {_torch_version}")

        if _repair or _interactive or (package_manager.get_version("torch") is not None
                                       and package_manager.get_version("torch")[0] != _torch_version):

            package_names = ["xformers", "torch", "torchvision", "triton", "bitsandbytes"]
            logging.debug(f"Looking for {', '.join(package_names)} packages.")
            installed_package_names = [pkg for pkg in package_names if package_manager.contains(pkg)]

            if installed_package_names:
                logging.debug(f"Uninstalling {', '.join(installed_package_names)} packages.")
            else:
                logging.debug(f"None of the following packages were found installed: {', '.join(package_names)}.")

            if installed_package_names:
                logging.info(f"Uninstalling {', '.join(installed_package_names)} packages.")
                if args.verbosity < 3:
                    _bar_format = "{desc:30}: {n_fmt}/{total_fmt} {bar} {elapsed}/{remaining}"
                    with tqdm_progress(total=len(installed_package_names), desc="Uninstalling packages",
                                       unit="package", unit_scale=False, dynamic_ncols=True,
                                       bar_format=_bar_format, mininterval=0, miniters=0) as _progress_bar:
                        for idx, _package_name in enumerate(installed_package_names):
                            _progress_bar.set_description(f"Uninstalling {_package_name}")
                            try:
                                subprocess.run(
                                    [sys.executable, "-m", "pip", "uninstall", "--quiet", "-y", _package_name],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                write_to_log(f"Uninstalling {_package_name}.")
                            except subprocess.CalledProcessError as e_:
                                if args.verbosity > 3:
                                    logging.debug(
                                        f"Failed to uninstall {_package_name}. Error code {e_.returncode}")
                            finally:
                                _progress_bar.update(1)
                else:
                    for _package_name in installed_package_names:
                        _result = subprocess.run(
                            [sys.executable, "-m", "pip", "uninstall", "-y", _package_name],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                        _output = _result.stdout.decode() + _result.stderr.decode()
                        logging.debug(_output)

        # Install/Reinstall Torch and Torchvision if one is missing or update/repair is flagged.
        if not (package_manager.contains('torch') or package_manager.contains('torchvision')) or \
                (_update or _repair or _interactive) \
                or (package_manager.get_version("torch") is not None
                    and package_manager.get_version("torch")[0] != _torch_version):

            # Create a flag file for the Torch version
            torch_flag_file = os.path.join(_log_dir, "status", ".torch_version")

            if _interactive:
                while True:
                    input_torch_version = input("Choose Torch version: (1) V1, (2) V2: ")
                    if input_torch_version == "1":
                        _TORCH_VERSION = TORCH_VERSION_1
                        _TORCHVISION_VERSION = TORCHVISION_VERSION_1
                        _TORCH_INDEX_URL = TORCH_INDEX_URL_1
                        break
                    elif input_torch_version == "2":
                        _TORCH_VERSION = TORCH_VERSION_2
                        _TORCHVISION_VERSION = TORCHVISION_VERSION_2
                        _TORCH_INDEX_URL = TORCH_INDEX_URL_2
                        break
                    else:
                        print("Invalid choice. Please enter 1 for Torch V1 or 2 for Torch V2.")
            else:
                # Read the torch version from the flag file if it exists and the input _torch_version is 0
                if _torch_version == 0 and os.path.exists(torch_flag_file):
                    with open(torch_flag_file, 'r') as _f:
                        _torch_version = int(_f.read().strip())
                elif _torch_version in (0, 1):
                    _TORCH_VERSION = TORCH_VERSION_1
                    _TORCHVISION_VERSION = TORCHVISION_VERSION_1
                    _TORCH_INDEX_URL = TORCH_INDEX_URL_1
                elif _torch_version == 2:
                    _TORCH_VERSION = TORCH_VERSION_2
                    _TORCHVISION_VERSION = TORCHVISION_VERSION_2
                    _TORCH_INDEX_URL = TORCH_INDEX_URL_2
                else:
                    # This should not be hit, but it's better to have a stable fallback if invalid data makes it here
                    _TORCH_VERSION = TORCH_VERSION_1
                    _TORCHVISION_VERSION = TORCHVISION_VERSION_1
                    _TORCH_INDEX_URL = TORCH_INDEX_URL_1

            identifier = ""
            xformers_sys = ""

            if os_info.family == "Windows":
                xformers_sys = "win_amd64"
                identifier = "f"
            if platform.system() == "Linux":
                xformers_sys = "linux_x86_64"
                identifier = "linux"

            xformers_url_1 = f"https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases" \
                             f"/download/{identifier}/xformers-{XFORMERS_VERSION_1}." \
                             f"dev0-cp310-cp310-{xformers_sys}.whl"

            install_commands = [
                [sys.executable, "-m", "pip", "install", f"torch=={_TORCH_VERSION}",
                 "--extra-index-url", f"{_TORCH_INDEX_URL}", "--quiet"],
                [sys.executable, "-m", "pip", "install", f"torchvision=={_TORCHVISION_VERSION}",
                 "--extra-index-url", f"{_TORCH_INDEX_URL}", "--quiet"],
                [sys.executable, "-m", "pip", "install", "--upgrade",
                 f"{xformers_url_1}", "--quiet"]
            ]

            if _torch_version == 2:
                install_commands = [
                    [sys.executable, "-m", "pip", "install", f"torch=={_TORCH_VERSION}",
                     "--extra-index-url", f"{_TORCH_INDEX_URL}", "--quiet"],
                    [sys.executable, "-m", "pip", "install", f"torchvision=={_TORCHVISION_VERSION}",
                     "--extra-index-url", f"{_TORCH_INDEX_URL}", "--quiet"],
                    [sys.executable, "-m", "pip", "install", "--upgrade",
                     f"xformers=={XFORMERS_VERSION_2}", "--quiet"],
                ]

                if os_info.family == "Windows":
                    install_commands.extend([[sys.executable, "-m", "pip", "install", f"{TRITON_URL_2}", "--quiet"]])
                elif platform.system() == "Linux":
                    install_commands.extend([[sys.executable, "-m", "pip",
                                              "install", "triton==2.0.0.post1", "--quiet"]])

            if _torch_version in (0, 1):
                logging.critical(
                    "Installing torch and related packages. These download large models. Please have "
                    "patience.")
            else:
                logging.critical(
                    "Installing torch v2 and related packages. These download large models that can "
                    "take up to an hour to download. Please have patience.")

            if args.verbosity < 3:
                stop_event = multiprocessing.Event()
                spinner_process = multiprocessing.Process()
                try:
                    for idx, _command in enumerate(install_commands):
                        try:
                            # Find the index of "install" in the command and assume the package name
                            # is the next argument
                            install_index = _command.index("install")
                            _package_name = _command[install_index + 1]
                            if _package_name.startswith(f"{TRITON_URL_2}"):
                                _package_name = "Triton"
                            if _package_name == "--upgrade":
                                _package_name = _command[install_index + 2]

                            # Extract version specifier if present
                            if '==' in _package_name:
                                _package_name, _package_version = _package_name.split('==')
                            else:
                                _package_version = None

                        except ValueError:
                            _package_name = "unknown package"
                            _package_version = None

                        if _package_version:
                            msg = f"Installing {_package_name} ({_package_version}): ({idx + 1}/" \
                                  f"{len(install_commands)})..."
                        else:
                            msg = f"Installing {_package_name}: ({idx + 1}/{len(install_commands)})..."

                        print(msg, end="")
                        sys.stdout.flush()

                        # Restart and recreate a new spinner process for each package installation
                        stop_event.clear()
                        spinner_process = multiprocessing.Process(target=spinner_task,
                                                                  args=(stop_event,))
                        spinner_process.start()

                        # Run the command that needs to have the spinner
                        try:
                            safe_subprocess_run(_command, stop_flag=stop_event, bufsize=-1)
                            version = package_manager.get_version(_package_name)
                            if version:
                                package_manager.add_package(_package_name, version)
                        except subprocess.CalledProcessError as e_:
                            # Stop the spinner on the way out
                            stop_event.set()
                            spinner_process.join()

                            print(
                                f"\nError occurred during the installation of {_package_name}. "
                                f"Exit code: {e_.returncode}")
                            logging.debug(
                                f"Error occurred during the installation of {_package_name}. "
                                f"Exit code: {e_.returncode}")
                            exit(1)

                        # Stop the spinner on the way out
                        stop_event.set()
                        spinner_process.join()
                        print("\b Done")

                    # After the installation process is complete and successful,
                    # update the torch version in the flag file
                    os.makedirs(os.path.join(_log_dir, "status"), exist_ok=True)
                    with open(torch_flag_file, 'w') as _f:
                        _f.write(str(_torch_version))
                except KeyboardInterrupt:
                    stop_event.set()
                    if spinner_process:
                        spinner_process.join()
                    print("\nInstallation was interrupted by user.")
                    exit(130)

            else:
                commands = [
                    [sys.executable, "-m", "pip", "install", f"torch=={_TORCH_VERSION}",
                     f"torchvision=={_TORCHVISION_VERSION}",
                     "--extra-index-url", f"{_TORCH_INDEX_URL}"],
                ]

                if _torch_version == '2':
                    commands.extend([
                        [sys.executable, "-m", "pip", "install", f"{TRITON_URL_2}"],
                        [sys.executable, "-m", "pip", "install", "--upgrade",
                         f"xformers=={XFORMERS_VERSION_2}"],
                    ])

                for idx, _command in enumerate(commands):
                    _package_name = "unknown package"
                    try:
                        # Find the index of "install" in the command and assume the package name
                        # is the next argument
                        install_index = _command.index("install")
                        _package_name = _command[install_index + 1]
                        if _package_name.startswith(f"{TRITON_URL_2}"):
                            _package_name = "Triton"
                        if _package_name == "--upgrade":
                            _package_name = _command[install_index + 2]

                        # Remove version specifier if present
                        _package_name = _package_name.split('==')[0]
                    except ValueError:
                        pass

                    try:
                        safe_subprocess_run(_command, bufsize=-1)
                        # After successful installation, add the package to the package manager
                        version_index = _command.index("==")  # Assuming version is specified in the command
                        version = _command[version_index + 1]
                        package_manager.add_package(_package_name, version)
                    except subprocess.CalledProcessError as e_:
                        print(
                            f"\nError occurred during the installation of {_package_name}. "
                            f"Exit code: {e_.returncode}")
                        logging.debug(
                            f"Error occurred during the installation of {_package_name}. "
                            f"Exit code: {e_.returncode}")
                        exit(1)

                # After the installation process is complete and successful,
                # update the torch version in the flag file
                os.makedirs(os.path.join(_log_dir, "status"), exist_ok=True)
                with open(torch_flag_file, 'w') as _f:
                    _f.write(str(_torch_version))

    def censor_local_path(_package_name):
        if os.path.isfile(_package_name) or os.path.isdir(_package_name):
            return "<Censored local path to maintain user privacy>"
        return _package_name

    # Name of the flag file
    flag_file = os.path.join(_log_dir, "status", ".pip_operations_done")

    logging.debug(f"Pip update flag: {update}")
    logging.debug(f"Pip repair flag: {repair}")

    # Check for the existence of the flag file
    if os.path.exists(flag_file) and not (update or repair or interactive) and package_manager.contains("torch") and \
            (package_manager.get_version("torch") is not None and
             package_manager.get_version("torch")[0] == torch_version):
        logging.critical("--update or --repair not specified. Skipping pip installations and repairs.")
        return

    # Check the conditions for running the install_or_repair_torch function:
    # 1. The system is Windows or Linux
    # 2. One of the following is true:
    #    a. The script is in repair or interactive mode
    #    b. Torch is not installed
    #    c. There is a version mismatch between the specified torch_version
    #       and the major version of the installed torch
    should_install_or_repair_torch = (
            (os_info.family == "Windows" or platform.system() == "Linux") and
            ((repair or interactive) or
             package_manager.get_version("torch") is None or
             int(package_manager.get_version("torch").split('.')[0]) != torch_version
             )
    )

    # Check the conditions for running the code block:
    # 1. The flag_file does not exist
    # 2. The flag_file exists, and either the update or repair flag is set
    if not os.path.exists(flag_file) or (os.path.exists(flag_file) and (update or repair)):
        try:
            # Update pip
            logging.critical("Checking for pip updates before Python operations.")

            try:
                if args.verbosity >= 2:
                    safe_subprocess_run(
                        [sys.executable, "-m", "pip", "install", "--upgrade", "--no-warn-script-location", "pip"],
                        bufsize=-1)
                else:
                    safe_subprocess_run([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
                                         "--no-warn-script-location", "pip"], bufsize=-1)
            except subprocess.CalledProcessError as _e:
                logging.error(f"An error occurred during pip operations: {str(_e)}")
                exit(1)

            # Install python dependencies
            logging.critical("Beginning kohya_ss dependencies installation.")

            # Set the paths for the built-in requirements and temporary requirements files
            requirements_path = os.path.join(_dir, "requirements.txt")
            logging.debug(f"Found requirements.txt at: {requirements_path}")
            if os.path.exists(requirements_path):
                temp_requirements = tempfile.NamedTemporaryFile(delete=False, mode="w+")
                try:
                    found_comment = False
                    with open(requirements_path, "r") as original_file:
                        for line in original_file:
                            logging.debug(f"Processing line: {line.strip()}")
                            if found_comment:
                                line = line.replace(".", _dir)
                                logging.debug(f"Replaced . with: Kohya_SS library path.")
                                found_comment = False
                            elif re.search(r"#.*kohya_ss.*library.*", line):
                                logging.debug(f"Found kohya_ss library comment in line: {line.strip()}")
                                found_comment = True
                            else:
                                logging.debug(f"Processing line without any conditions: {line.strip()}")

                            # Skip comments and empty lines
                            if (line.strip().startswith("#") or not line.strip()) or found_comment:
                                continue

                            if not found_comment:
                                logging.debug(f"Installing: {line.strip()}")
                            else:
                                logging.debug(f"Installing: Kohya_SS library.")
                            temp_requirements.write(line)

                        if os_info.family == "macOS":
                            if platform.machine() == "arm64":
                                temp_requirements.write(f"tensorflow-macos=={TENSORFLOW_MACOS_VERSION}\n")
                                temp_requirements.write(f"tensorflow-metal=={TENSORFLOW_METAL_VERSION}\n")
                            elif platform.machine() == "x86_64":
                                temp_requirements.write(f"tensorflow=={TENSORFLOW_VERSION}\n")

                        if should_install_or_repair_torch:
                            install_or_repair_torch(update, repair, interactive, torch_version)

                        # This is where we merge the macOS requirements.txt into the temporary requirements.txt
                        if os_info.family == "macOS":
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

                logging.debug("requirements.txt successfully processed and merged.")
                if args.verbosity >= 3:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--upgrade", "--use-pep517",
                         "-r", temp_requirements.name],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    output = result.stdout.decode() + result.stderr.decode()

                    for line in output.splitlines():
                        print(f"DEBUG: {line}")
                        censored_line = censor_local_path(line)
                        write_to_log(f"DEBUG: {censored_line}")
                else:
                    logging.critical("Please be patient. It may take time for the requirements to be "
                                     "collected before showing progress.")

                    bar_format = "{desc:30}: {n_fmt}/{total_fmt} {bar} {elapsed}/{remaining}"

                    with open(temp_requirements.name, "r") as f:
                        num_packages = sum(1 for line in f if line.strip())
                        with tqdm_progress(total=num_packages, desc="Installing packages",
                                           unit="package", unit_scale=False, dynamic_ncols=True,
                                           bar_format=bar_format, mininterval=0, miniters=0) as progress_bar:
                            f.seek(0)
                            for package in f:
                                package_name = package.strip().split('==')[0]

                                progress_bar.set_description(f"Installing {package_name}")
                                command = [sys.executable, "-m", "pip", "install", "--upgrade",
                                           "--use-pep517", "--no-warn-script-location", package]

                                if args.verbosity < 3:
                                    command.append("--quiet")
                                    try:
                                        safe_subprocess_run(command, bufsize=-1)
                                        censored_package_name = censor_local_path(package_name)
                                        write_to_log(f"Installing {censored_package_name}.")
                                    except subprocess.CalledProcessError as _e:
                                        logging.error(f"An error occurred during pip operations: {str(_e)}")
                                        exit(1)
                                else:
                                    try:
                                        safe_subprocess_run(command)
                                    except subprocess.CalledProcessError as _e:
                                        output = _e.output
                                        logging.debug(output)
                                        logging.error(f"An error occurred during pip operations: {str(_e)}")
                                        exit(1)
                                progress_bar.update(1)

                # Get the scripts directory based on the operating system
                if os_info.family == "Windows":
                    scripts_dir = os.path.join(Path(sys.executable).parent, 'Scripts')
                else:
                    scripts_dir = os.path.join(Path(sys.executable).parent, 'bin')

                # Add the scripts directory to the PATH
                os.environ['PATH'] = scripts_dir + os.pathsep + os.environ['PATH']

                # Delete the temporary requirements file
                logging.debug(f"Removing {temp_requirements.name}")
                if os.path.exists(temp_requirements.name):
                    os.remove(temp_requirements.name)

            os.makedirs(os.path.join(_log_dir, "status"), exist_ok=True)
            with open(flag_file, 'w') as f:
                f.write('Pip operations done on: ' + str(datetime.now()))

            logging.info("Pip operations completed successfully.")

        except Exception as _e:
            # Handle exceptions appropriately
            logging.error("An error occurred during pip operations: %s", str(_e))
            # You may choose to re-raise the exception if the error is critical
            # raise

    elif should_install_or_repair_torch:
        install_or_repair_torch(update, repair, interactive, torch_version)


# noinspection SpellCheckingInspection
def configure_accelerate(interactive):
    def configure_accelerate_manually(_interactive):
        if _interactive:
            try:
                logging.critical("Manually configure accelerate: ")
                safe_subprocess_run(["accelerate", "config"], interactive=True)
            except FileNotFoundError:
                logging.error(
                    "The 'accelerate' command was not found. Please ensure it is installed and available in your PATH.")
                exit(1)
            except subprocess.CalledProcessError as e_:
                logging.error(f"Accelerate config failed with error code {e_.returncode}")
                exit(1)

    if os_info.family == "macOS" and platform.machine() == "arm64":
        source_accelerate_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_files",
                                                     "accelerate", "macos_config.yaml")
    else:
        source_accelerate_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_files",
                                                     "accelerate", "default_config.yaml")

    logging.debug(f"Source accelerate config location: {source_accelerate_config_file}")

    configure_accelerate_manually(interactive)

    if not interactive:
        target_config_location = None

        if os_info.family == "Windows":
            logging.debug(
                f"Environment variables: HF_HOME: {os.environ.get('HF_HOME')}, "
                f"LOCALAPPDATA: {os.environ.get('LOCALAPPDATA')}, "
                f"USERPROFILE: {os.environ.get('USERPROFILE')}")
            if env_var_exists("HF_HOME"):
                target_config_location = Path(os.environ["HF_HOME"], "accelerate", "default_config.yaml")
            elif env_var_exists("LOCALAPPDATA"):
                target_config_location = Path(os.environ["LOCALAPPDATA"], "huggingface",
                                              "accelerate", "default_config.yaml")
            elif env_var_exists("USERPROFILE"):
                target_config_location = Path(os.environ["USERPROFILE"], ".cache", "huggingface",
                                              "accelerate", "default_config.yaml")
        else:
            if env_var_exists("HF_HOME"):
                target_config_location = Path(os.environ["HF_HOME"], "accelerate", "default_config.yaml")
            elif env_var_exists("XDG_CACHE_HOME"):
                target_config_location = Path(os.environ["XDG_CACHE_HOME"], "huggingface",
                                              "accelerate", "default_config.yaml")
            elif env_var_exists("HOME"):
                target_config_location = Path(os.environ["HOME"], ".cache", "huggingface",
                                              "accelerate", "default_config.yaml")

        logging.debug(f"Target config location: {target_config_location}")

        if target_config_location:
            if not target_config_location.is_file():
                target_config_location.parent.mkdir(parents=True, exist_ok=True)
                logging.debug(f"Target accelerate config location: {target_config_location}")
                shutil.copyfile(source_accelerate_config_file, target_config_location)
                logging.debug(f"Copied accelerate config file to: {target_config_location}")
        else:
            logging.info("Could not place the accelerate configuration file. Please configure manually.")
            configure_accelerate_manually(_interactive=True)


# noinspection SpellCheckingInspection
def launch_kohya_gui(_args):
    if not in_container():
        _venv_path = os.path.join(_args.dir, "venv")

        if not os.path.exists(_venv_path):
            logging.info("Error: Virtual environment not found")
            sys.exit(1)

        python_executable = os.path.join(_venv_path, "bin", "python") if sys.platform != "win32" else os.path.join(
            _venv_path, "Scripts", "python.exe")

        if not os.path.exists(python_executable):
            logging.info("Error: Python executable not found in the virtual environment")
            sys.exit(1)

    kohya_gui_file = os.path.join(_args.dir, "kohya_gui.py")
    logging.debug(f"kohya_gui.py expected at {kohya_gui_file}")

    if os.path.exists(kohya_gui_file):
        cmd = [
            venv_python_bin, os.path.join(kohya_gui_file)
        ]

        if _args.headless:
            cmd.extend(["--headless"])

        if _args.inbrowser:
            cmd.extend(["--inbrowser"])

        if _args.listen:
            cmd.extend(["--listen", _args.listen])

        if _args.password:
            cmd.extend(["--password", _args.password])

        if str(_args.server_port) is not None and str(_args.server_port) != "":
            cmd.extend(["--server-port", str(_args.server_port)])

        if _args.share:
            cmd.extend(["--share"])

        if _args.username:
            cmd.extend(["--username", _args.username])

        if _args.verbosity > 0:
            cmd.extend(["--verbosity", str(_args.verbosity)])

        try:
            logging.debug(f"Launching kohya_gui.py with Python bin: {venv_python_bin}")
            logging.debug(f"Running kohya_gui.py as: {cmd}")
            print("\n")
            logging.critical("Running kohya_gui.py now. Press control+c in this terminal to shutdown the software.")
            subprocess.run(cmd, check=True)
            pass
        except KeyboardInterrupt:
            with open(os.devnull, 'w') as null_file, redirect_stderr(null_file):
                logging.info("Process terminated by the user. Exiting...")
    else:
        logging.critical("kohya_gui.py not found. Can't run the software.")
        exit(1)


def main(_args=None):
    if not getattr(_args, "git-repo") or not _args.dir or not getattr(_args, "branch"):
        logging.info(
            "Error: gitRepo, Branch, and Dir must have a value. Please provide values in the config file or through "
            "command line arguments.")
        exit(1)

    # Define the directories relative to the installation directory needed for install and launch
    parent_dir = os.path.dirname(_args.dir)

    # The main logic will go here after the sanity checks.
    check_and_create_install_folder(parent_dir, _args.dir)
    check_storage_space(getattr(_args, "skip-space-check"), _args.dir, parent_dir)
    if update_kohya_ss(_args.dir, getattr(_args, "git-repo"), _args.branch, _args.update, _args.repair):
        pre_pip_prepare_environment_files_and_folders(_args.dir, site_packages_dir)
        if brew_install_tensorflow_deps(_args.verbosity):
            install_python_dependencies(_args.dir, _args.torch_version, _args.update, _args.repair,
                                        _args.interactive, getattr(_args, "log-dir"))
            post_pip_prepare_environment_files_and_folders(site_packages_dir, _args.dir)
            configure_accelerate(_args.interactive)
            if not getattr(_args, 'setup_only'):
                launch_kohya_gui(_args)
            else:
                logging.critical(f"Installation to {_args.dir} is complete.")
                exit(0)


def get_logs_dir(_args):
    if getattr(_args, "log-dir"):
        _logs_dir = os.path.abspath(os.path.expanduser(getattr(_args, "log-dir")))
    else:
        _logs_dir = os.path.abspath(os.path.join(_args.dir, "logs"))

    os.makedirs(_logs_dir, exist_ok=True)
    return _logs_dir


# noinspection DuplicatedCode
def write_to_log(message, _log_file=None):
    if _log_file is None:
        # Get the log file from the existing logging handlers
        for _handler in logging.getLogger().handlers:
            if isinstance(_handler, logging.FileHandler):
                _log_file = _handler.baseFilename
                # Ensure the handler has flushed all its output before we write to the file directly
                _handler.flush()
                break
        else:
            raise ValueError("No log file found in the logging handlers.")

    formatted_message = "LOG: " + message
    with open(_log_file, 'a') as f:
        f.write(formatted_message + '\n')


# noinspection DuplicatedCode,SpellCheckingInspection
class CustomFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(fmt='%(levelname)s: %(message)s')

    def format(self, record):
        if record.levelno == logging.CRITICAL:
            return f"{record.getMessage()}"
        else:
            return f"{record.levelname}: {record.getMessage()}"

    @staticmethod
    def generate_log_filename(_logs_dir):
        now = datetime.now()
        current_date_str = now.strftime("%Y-%m-%d")  # Just the date part
        current_time_str = now.strftime("%H%M")  # Time in 24-hour format

        # Create a subdirectory for the current date
        date_subdir = os.path.join(_logs_dir, current_date_str)
        os.makedirs(date_subdir, exist_ok=True)

        log_level_name = logging.getLevelName(log_level).lower()
        if log_level == logging.ERROR:
            log_filename = f"launcher_{current_time_str}.log"
        else:
            log_filename = f"launcher_{current_time_str}_{log_level_name}.log"
        log_filepath = os.path.join(date_subdir, log_filename)

        return log_filepath


if __name__ == "__main__":
    try:
        # noinspection DuplicatedCode
        config_file = parse_file_arg()
        config_data = load_config(config_file)
        args = parse_args(config_data)

        # Initialize log_level with a default value
        log_level = logging.ERROR

        # Set logging level based on the verbosity count
        # print(f"Verbosity: {args.verbosity}")
        if args.verbosity == 0:
            log_level = logging.ERROR
        elif args.verbosity == 1:
            log_level = logging.WARNING
        elif args.verbosity == 2:
            log_level = logging.INFO
        elif args.verbosity >= 3:
            log_level = logging.DEBUG

        # Configure logging
        # noinspection SpellCheckingInspection
        setattr(args, "log-dir", os.path.abspath(get_logs_dir(args)))
        log_file = CustomFormatter.generate_log_filename(getattr(args, "log-dir"))
        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter())

        log_file_handler = logging.FileHandler(log_file, mode='a')
        # noinspection SpellCheckingInspection
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s',
                            handlers=[logging.StreamHandler(),
                                      log_file_handler])

        logging.getLogger().setLevel(log_level)

        # Replace 'root' with an empty string in the logger name
        for handler in logging.getLogger().handlers:
            handler.setFormatter(CustomFormatter())

        # We print the log location to inform users, but avoid leaking their personal info in logs
        log_dir = getattr(args, "log-dir")
        print(f"Logs will be stored in: {log_dir}")

        # Use logging in the script like so (in order of log levels).
        # write_to_log("This goes to log file, but does not display in terminal.")
        # logging.critical("This will always display.")
        # logging.error("This is an error message.")
        # logging.warning("This is a warning message.")
        # logging.info("This is an info message.")
        # logging.debug("This is a debug message.")

        if getattr(args, 'no-setup') is True and args.setup_only:
            logging.critical("Setup Only and No Setup options are mutually exclusive.")
            logging.critical("Please run with only one of those options or none of them.")
            exit(1)

        os_info = get_os_info()

        # Store the original sys.executable value
        original_sys_executable = sys.executable

        # Print all arguments and their values in verbose 3 mode
        if args.verbosity >= 3:
            for k, v in args.__dict__.items():
                logging.debug(f"{k}: {v}")

        logging.critical("Beginning launcher.py.")

        # Following check disabled as PyCharm can't detect it's being used in a subprocess
        # noinspection PyUnusedLocal
        venv_python_bin = None

        # Check if python3 or python3.10 binary exists
        # noinspection DuplicatedCode
        python_bin = find_python_binary()
        if not python_bin:
            logging.error("Valid python3 or python3.10 binary not found.")
            logging.error("Cannot proceed with the python steps.")
            exit(1)

        if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
            logging.info("Error: This script requires Python 3.10.")
            logging.debug(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
            sys.exit(1)

        # Create and activate virtual environment if not in container environment
        if not in_container():
            logging.critical("Switching to virtual Python environment.")
            venv_path = os.path.join(args.dir, "venv")
            subprocess.run([python_bin, "-m", "venv", venv_path])

            # Check the virtual environment for permissions issues
            check_permissions(venv_path)

            # Activate the virtual environment
            venv_bin_dir = os.path.join(venv_path, "bin") if os.name != "nt" else os.path.join(venv_path, "Scripts")
            os.environ['PATH'] = venv_bin_dir + os.pathsep + os.environ['PATH']
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
            if os_info.family == "Windows":
                site_packages_dir = os.path.join(python_executable_dir, "Lib", "site-packages")
            else:
                site_packages_dir = os.path.join(python_executable_dir, "..", "lib", "python" + sys.version[:3],
                                                 "site-packages")

        if getattr(args, 'no-setup') and not getattr(args, 'setup_only'):
            launch_kohya_gui(args)
            exit(0)
        else:
            main(args)
    except KeyboardInterrupt:
        logging.error("User interrupted process. Exiting!")
        exit(130)
    except Exception as e:
        logging.error("Launcher.py had to exit due to an error during the main installation.")
        logging.error(str(e))
        exit(1)
