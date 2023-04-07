import argparse
import errno
import logging
import importlib
import mimetypes
import os
import platform
import re
import shutil
import site
import subprocess
import sys
import stat
import tempfile
import time
from pathlib import Path
from importlib.metadata import Distribution, PackageNotFoundError, version


# This enables programmatically installing pip packages
def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def check_and_import(module_name, package_name=None):
    if package_name is None:
        package_name = module_name

    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"Installing {package_name}...")
        install_package(package_name)
        return importlib.import_module(module_name)


base64 = check_and_import('base64')
requests = check_and_import('requests')
yaml = check_and_import('yaml', 'PyYAML')

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


def parse_args(_config_data):
    parser = argparse.ArgumentParser(
        description="Launcher script for Kohya_SS. This script helps you configure, install, and launch the Kohya_SS "
                    "application.",
        epilog="""Examples:
    Switch to the dev branch:
    python launcher.py --branch dev

    Point to a custom installation directory, but skip any git operations:
    python launcher.py --dir /path/to/kohya_ss --no-git-update

    Bypass all environment checks except Python dependency validation and launch the GUI:
    python launcher.py --exclude-setup""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Define the default arguments first
    default_args = [
        {"short": "-b", "long": "--branch", "default": "master", "type": str},
        {"short": "-d", "long": "--dir", "default": os.path.expanduser("~/kohya_ss"), "type": str},
        {"short": "-f", "long": "--file", "default": "install_config.yaml", "type": str},
        {"short": "-g", "long": "--git-repo", "default": "https://github.com/bmaltais/kohya_ss.git", "type": str},
        {"short": "-i", "long": "--interactive", "default": False, "type": bool},
        {"short": "-n", "long": "--no-git-update", "default": False, "type": bool},
        {"short": "-p", "long": "--public", "default": False, "type": bool},
        {"short": "-r", "long": "--runpod", "default": False, "type": bool},
        {"short": "-s", "long": "--skip-space-check", "default": False, "type": bool},
        {"short": "-v", "long": "--verbosity", "default": 0, "type": int},
        {"short": "-x", "long": "--exclude-setup", "default": False, "type": bool},
        {"short": "", "long": "--listen", "default": "127.0.0.1", "type": str},
        {"short": "", "long": "--username", "default": "", "type": str},
        {"short": "", "long": "--password", "default": "", "type": str},
        {"short": "", "long": "--server-port", "default": 0, "type": str},
        {"short": "", "long": "--inbrowser", "default": False, "type": bool},
        {"short": "", "long": "--share", "default": False, "type": bool},
    ]

    # Update the default arguments with values from the config file
    if _config_data:
        if "arguments" in _config_data:
            for arg in _config_data["arguments"]:
                name = arg["name"]
                value = arg["value"]
                for default_arg in default_args:
                    if f'--{name.lower()}' == default_arg["long"]:
                        default_arg["default"] = value
        if "kohya_gui_arguments" in _config_data:
            for arg in _config_data["kohya_gui_arguments"]:
                name = arg["name"]
                value = arg["value"]
                for default_arg in default_args:
                    if f'--{name.lower()}' == default_arg["long"]:
                        default_arg["default"] = value

    # Add arguments to the parser with updated default values
    for arg in default_args:
        short_opt = arg["short"]
        long_opt = arg["long"]
        default_value = arg["default"]
        arg_type = arg.get("type", str)

        if isinstance(default_value, bool):
            action = 'store_true' if default_value is False else 'store_false'
            if short_opt:
                parser.add_argument(short_opt, long_opt, dest=long_opt[2:], action=action, default=default_value)
            else:
                parser.add_argument(long_opt, dest=long_opt[2:].replace("-", "_"), default=default_value, type=arg_type)
        else:
            if short_opt:
                parser.add_argument(short_opt, long_opt, dest=long_opt[2:], default=default_value, type=arg_type)
            else:
                parser.add_argument(long_opt, dest=long_opt[2:].replace("-", "_"), default=default_value, type=arg_type)

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
    site_packages_dir = site.getsitepackages()[0]

    # Return the environment path and site-packages path
    return env_path, site_packages_dir


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
    print("Checking symlinks now.")
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
                print(f"{target_file} does not exist. Nothing to link.")
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


class GitCredentialError(Exception):
    pass


class GitCredentialNotStoredError(GitCredentialError):
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


def run_git_command(_args, cwd=None, timeout=10):
    try:
        result = subprocess.run(["git"] + _args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
                                timeout=timeout)
        return result.stdout.decode("utf-8").strip()
    except subprocess.TimeoutExpired:
        logging.error("Git command timed out.")
        exit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e}")
        logging.debug(f"stdout: {e.stdout.decode('utf-8')}")
        logging.debug(f"stderr: {e.stderr.decode('utf-8')}")
        raise Exception("Git failed exiting script to be on the safe side.")


def get_stored_git_credentials(url):
    try:
        output = run_git_command(["credential", "fill"], cwd=None)
        return _parse_git_credential_output(output)
    except GitCredentialError as e:
        raise GitCredentialNotStoredError(f"Failed to get stored Git credentials: {e}")


def _parse_git_credential_output(output):
    username = None
    password = None
    for line in output.split("\n"):
        key, value = line.split("=", 1)
        if key == "username":
            username = value
        elif key == "password":
            password = value
    if username is None or password is None:
        raise GitCredentialNotStoredError("Failed to parse Git credential output.")
    return username, password


def add_credentials_to_url(url, username, password):
    url_parts = url.split("://", 1)
    if len(url_parts) == 1:
        return f"https://{username}:{password}@{url}"
    else:
        return f"{url_parts[0]}://{username}:{password}@{url_parts[1]}"


def get_latest_tag(git_repo):
    match = re.search(r'github.com/([^/]+)/([^/.]+)', git_repo)
    if match:
        owner, repo_name = match.groups()
    else:
        raise ValueError("Invalid GitHub repository URL")

    url = f"https://api.github.com/repos/{owner}/{repo_name}/tags"
    response = requests.get(url)
    response.raise_for_status()
    tags_data = response.json()
    latest_tag = tags_data[0]["name"]
    return latest_tag


def update_kohya_ss(_dir, git_repo, branch, parent_dir, no_git_update):
    def clone_and_switch(_dir, _git_repo, _branch, _parent_dir, _username=None, _password=None):
        logging.info(f"Cloning and switching to {_git_repo}:{_branch}")

        # Download the repo as a zip file
        # Remove .git extension if present
        _git_repo = _git_repo.rstrip('.git')
        _download_url = _git_repo.rstrip("/") + f"/archive/refs/tags/{get_latest_tag(_git_repo)}.zip"
        auth = (_username, _password) if _username and _password else None
        logging.info(f"Attempting to download from: {_download_url}")
        response = requests.get(_download_url, auth=auth)

        if response.status_code != 200:
            raise Exception(f"Failed to download the repository: {response.status_code}")

        # Save the zip file to a temporary location
        temp_zip = tempfile.NamedTemporaryFile(delete=False)
        for chunk in response.iter_content(chunk_size=8192):
            temp_zip.write(chunk)
        temp_zip.close()

        # Extract the zip file to the parent directory
        with zipfile.ZipFile(temp_zip.name, "r") as zip_ref:
            zip_ref.extractall(_parent_dir)
        os.unlink(temp_zip.name)

        # Rename the extracted folder to the desired folder name
        extracted_folder = os.path.join(_parent_dir, f"{os.path.basename(_dir)}-{_branch}")
        if os.path.exists(_dir):
            shutil.rmtree(_dir)
        shutil.move(extracted_folder, _dir)

        run_git_command(["git", "-C", _dir, "init"])
        run_git_command(["git", "-C", _dir, "remote", "add", "origin", _git_repo])
        run_git_command(["git", "-C", _dir, "fetch"])
        run_git_command(["git", "-C", _dir, "checkout", _branch])

    if not no_git_update:
        git_installed = is_git_installed()

        try:
            if git_installed:
                logging.debug(f"Git installed: {git_installed}")
                if os.path.exists(_dir) and os.path.isdir(_dir):
                    logging.debug(
                        f"Items counted in {_dir}: {len(os.listdir(_dir))}")

                    if len(os.listdir(_dir)) > 1:
                        logging.error(f"The destination path {_dir} already exists and is not an empty directory. "
                                      f"Git will not clone in this situation. Please use a different path or clear "
                                      f"the existing directory before running the script.")
                        exit(1)
                    elif len(os.listdir(_dir)) == 1 and os.path.exists(venv_path):
                        logging.debug(
                            f"Moving {os.path.join(_dir, 'venv')} to {os.path.join(tempfile.gettempdir(), 'venv')}")
                        shutil.move(os.path.join(_dir, "venv"), os.path.join(tempfile.gettempdir(), "venv"))
                        run_git_command(["clone", "-b", branch, git_repo, _dir])
                    else:
                        run_git_command(["clone", "-b", branch, git_repo, _dir])

                if os.path.exists(os.path.join(tempfile.gettempdir(), "venv")):
                    shutil.move(os.path.join(tempfile.gettempdir(), "venv"), os.path.join(_dir, "venv"))
            else:
                raise Exception("Git not installed.")
        except Exception as e:
            logging.warning(f"Failed to clone the repository using git: {e}")

            # Download the latest release as a zip file from the default repository
            try:
                download_url = git_repo.rstrip("/") + f"/archive/refs/tags/{get_latest_tag(git_repo)}.zip"
                clone_and_switch(_dir, download_url, branch, parent_dir)
            except Exception as e:
                logging.warning(f"Failed to download the latest release: {e}")
                try:
                    username, password = get_stored_git_credentials(git_repo)
                    git_repo_with_credentials = add_credentials_to_url(git_repo, username, password)
                except GitCredentialNotStoredError:
                    logging.warning("No stored Git credentials found.")
                    git_repo_with_credentials = git_repo
                    username, password = None, None
                except GitCredentialError as e:
                    logging.warning(f"Unable to get Git credentials: {e}")
                    git_repo_with_credentials = git_repo
                    username, password = None, None

                clone_and_switch(_dir, git_repo_with_credentials, branch, parent_dir, username, password)
    else:
        logging.warning("Skipping git operations.")


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
                    print(f"Error executing uname command: {e}")
                    self.name = "Generic Linux"
                    self.family = "Generic Linux"
            return {
                "name": self.name,
                "family": self.family,
                "version": self.version
            }


def get_os_info():
    return OSInfo()


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
                logging.info(f"Missing permissions on file: {file_path}")

                try:
                    os.chmod(file_path, current_permissions | missing_permissions)
                    logging.info(f"Fixed permissions for file: {file_path}")
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


def validate_requirements(requirements_file='requirements.txt'):
    logging.info("Validating that requirements are satisfied.")

    with open(requirements_file) as f:
        requirements = f.readlines()

    missing_requirements = []
    wrong_version_requirements = []

    for requirement in requirements:
        requirement = requirement.strip()
        if requirement == ".":
            continue

        try:
            split_req = re.split(r'[=<>!~]+', requirement)
            if len(split_req) >= 2:
                pkg_name, pkg_version = split_req[:2]
            else:
                pkg_name = split_req[0]
                pkg_version = None

            installed_version = version(pkg_name)
            if pkg_version and installed_version != pkg_version:
                wrong_version_requirements.append((requirement, pkg_version, installed_version))
        except PackageNotFoundError:
            if "@" in requirement:
                package_name, vcs_url = requirement.split("@", 1)
                os.system(f"pip install -e {vcs_url}")
                try:
                    version(package_name)
                except PackageNotFoundError:
                    missing_requirements.append(requirement)
            else:
                missing_requirements.append(requirement)

    if missing_requirements or wrong_version_requirements:
        if missing_requirements:
            logging.error("The following packages are missing:")
            for requirement in missing_requirements:
                logging.error(f" - {requirement}")
        if wrong_version_requirements:
            logging.error("The following packages have the wrong version:")
            for requirement, expected_version, actual_version in wrong_version_requirements:
                logging.error(f" - {requirement} (expected version {expected_version}, found version {actual_version})")
        upgrade_script = "upgrade.ps1" if os.name == "nt" else "upgrade.sh"
        logging.error(
            f"\nRun {upgrade_script} or pip install -U -r {requirements_file} to resolve the missing requirements listed above...")

        return False

    logging.info("All requirements satisfied.")
    return True


def install_python_dependencies(_dir, runpod):
    # Update pip
    logging.info("Checking for pip updates before Python operations.")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install python dependencies
    logging.info("Installing python dependencies. This could take a few minutes as it downloads files.")

    # Set the paths for the built-in requirements and temporary requirements files
    requirements_path = os.path.join(_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        temp_requirements = tempfile.NamedTemporaryFile(delete=False, mode="wt")

        with open(requirements_path, "r") as original_file, open(temp_requirements.name, "w") as temp_file:
            for line in original_file:
                if "#.*kohya_ss.*library" in line:
                    line = line.replace(".", _dir)
                temp_file.write(line)

            # Append the appropriate packages based on the conditionals
            if runpod:
                temp_file.write("tensorrt\n")

            if os_info.family == "Darwin":
                if platform.machine() == "arm64":
                    temp_file.write(f"tensorflow-macos=={TENSORFLOW_MACOS_VERSION}\n")
                    temp_file.write(f"tensorflow-metal=={TENSORFLOW_METAL_VERSION}\n")
                elif platform.machine() == "x86_64":
                    temp_file.write(f"tensorflow=={TENSORFLOW_VERSION}\n")

        if os.path.exists(temp_requirements.name):
            if not validate_requirements(temp_requirements.name):
                # Install the packages from the temporary requirements file
                pip_install_args = [sys.executable, "-m", "pip", "install", "--use-pep517", "--upgrade", "-r",
                                    temp_requirements.name]
                if args.verbosity <= 1:
                    pip_install_args.insert(4, "--quiet")
                subprocess.run(pip_install_args, check=True)

            subprocess.run(pip_install_args, check=True, stderr=subprocess.PIPE)
        else:
            logging.error(f"Unable to locate {temp_requirements.name}")
            exit(1)
    else:
        logging.error(f"Unable to locate requirements.txt in {_dir}")
        exit(1)

    logging.debug("Removing the temp requirements file.")
    if os.path.isfile(temp_requirements.name):
        # Close the temporary file and remove
        temp_requirements.close()
        os.remove(temp_requirements.name)

    # Only exit the virtual environment if we aren't in a container.
    # This is because we never entered one at the beginning of the function if container detected.
    if not in_container():
        logging.debug("Exiting Python virtual environment.")

        # Reset sys.executable to its original value
        sys.executable = original_sys_executable
        logging.debug(f"sys.executable reset to: {sys.executable}")


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

    if _args.username:
        cmd.extend(["--password", _args.username])

    logging.debug(f"Launching kohya_gui.py with Python bin: {venv_python_bin}")
    logging.debug(f"Running kohya_gui.py as: {cmd}")
    subprocess.run(cmd, check=True)


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
    update_kohya_ss(_args.dir, getattr(_args, "git-repo"), _args.branch, parent_dir, getattr(_args, "no-git-update"))
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

    if getattr(args, 'exclude-setup'):
        launch_kohya_gui(args)
        exit(0)
    else:
        main(args)
