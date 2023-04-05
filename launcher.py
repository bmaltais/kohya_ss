import argparse
import logging
import os
import platform
import re
import shutil
import site
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Set the package versions at the beginning of the script to make them easy to modify as needed.
TENSORFLOW_VERSION = "2.12.0"
TENSORFLOW_MACOS_VERSION = "2.12.0"
TENSORFLOW_METAL_VERSION = "0.8.0"


def load_config(_config_file):
    if os.path.isfile(_config_file):
        with open(_config_file, "r") as f:
            _config_data = yaml.safe_load(f)
    else:
        _config_data = None
    return _config_data


def parse_file_arg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", "--file", dest="config_file", default="install_config.yaml",
                        help="Path to the configuration file.")
    _args, _ = parser.parse_known_args()
    return args.config_file


def parse_args(_config_data):
    parser = argparse.ArgumentParser(description="Launcher script for Kohya_SS.")

    if _config_data:
        for arg in _config_data["arguments"]:
            name = arg["name"]
            short = arg["short"]
            long = arg["long"]
            description = arg["description"]
            default = arg["default"]

            if isinstance(default, bool):
                parser.add_argument(short, long, dest=name, action="store_true", help=description)
            else:
                parser.add_argument(short, long, dest=name, default=default, help=description)
    else:
        parser.add_argument("-b", "--branch", dest="branch", default="master",
                            help="Select which branch of kohya to check out on new installs.")
        parser.add_argument("-d", "--dir", dest="dir", default=os.path.expanduser("~/kohya_ss"),
                            help="The full path you want kohya_ss installed to.")
        parser.add_argument("-f", "--file", dest="config_file", default="install_config.yaml",
                            help="Path to the configuration file.")
        parser.add_argument("-g", "--git-repo", dest="gitRepo", default="https://github.com/kohya/kohya_ss.git",
                            help="You can optionally provide a git repo to check out. Useful "
                                 "for custom forks.")
        parser.add_argument("-i", "--interactive", dest="interactive", action="store_true",
                            help="Interactively configure accelerate instead of using default config file.")
        parser.add_argument("-n", "--no-git-update", dest="gitUpdate", action="store_true",
                            help="Do not update kohya_ss repo. No git pull or clone operations.")
        parser.add_argument("-p", "--public", dest="public", action="store_true",
                            help="Expose public URL in runpod mode. Won't have an effect in other modes.")
        parser.add_argument("-r", "--runpod", dest="runpod", action="store_true",
                            help="Forces a runpod installation. Useful if detection fails for any reason.")
        parser.add_argument("-s", "--skip-space-check", dest="spaceCheck", action="store_true",
                            help="Skip the 10Gb minimum storage space check.")
        parser.add_argument("-v", "--verbosity", action="count", default=0, help="Increase verbosity levels up to 3.")
        parser.add_argument("-x", "--exclude-setup", dest="skip_setup", action="store_false",
                            help="Skip all setup steps and only validate python requirements then launch GUI.")

        # Now the kohya_gui.py arguments to be passed through
        parser.add_argument('--gui-listen', type=str, default='127.0.0.1',
                            help='IP to listen on for connections to Gradio')
        parser.add_argument('--gui-username', type=str, default='', help='Username for authentication')
        parser.add_argument('--gui-password', type=str, default='', help='Password for authentication')
        parser.add_argument('--gui-server-port', type=int, default=0, help='Port to run the server listener on')
        parser.add_argument('--gui-inbrowser', action='store_true', help='Open in browser')
        parser.add_argument('--gui-share', action='store_true', help='Share the gradio UI')

    _args = parser.parse_args()
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
        print(f"Creating install folder {_dir}.")
        os.makedirs(_dir)

    if not os.access(_dir, os.W_OK):
        print(f"We cannot write to {_dir}.")
        print("Please ensure the install directory is accurate and you have the correct permissions.")
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
        print("We are assuming a root drive install for space-checking purposes.")
        folder = os.path.abspath(os.sep)

    free_space_in_bytes = shutil.disk_usage(folder).free
    free_space_in_gb = free_space_in_bytes / (1024 * 1024 * 1024)
    return free_space_in_gb


def check_storage_space(_dir, parent_dir, space_check=True):
    if space_check:
        if size_available(_dir, parent_dir) < 10:
            print("You have less than 10Gb of free space. This installation may fail.")
            msg_timeout = 10  # In seconds
            message = "Continuing in..."
            print("Press control-c to cancel the installation.")

            for i in range(msg_timeout, -1, -1):
                print(f"\r{message} {i}s.", end="")
                time.sleep(1)


def create_symlinks(symlink, target_file):
    print("Checking symlinks now.")
    # Next line checks for valid symlink
    if os.path.islink(symlink):
        # Check if the linked file exists and points to the expected file
        if os.path.exists(symlink) and os.path.realpath(symlink) == target_file:
            print(f"{os.path.basename(symlink)} symlink looks fine. Skipping.")
        else:
            if os.path.isfile(target_file):
                print(f"Broken symlink detected. Recreating {os.path.basename(symlink)}.")
                os.remove(symlink)
                os.symlink(target_file, symlink)
            else:
                print(f"{target_file} does not exist. Nothing to link.")
    else:
        print(f"Linking {os.path.basename(symlink)}.")
        os.symlink(target_file, symlink)


def setup_file_links(site_packages_dir, runpod):
    if os_info.family == "Windows":
        bitsandbytes_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitsandbytes_windows")
        bitsandbytes_dest = os.path.join(site_packages_dir, "bitsandbytes")
        bitsandbytes_cuda_dest = os.path.join(bitsandbytes_dest, "cuda_setup")

        if os.path.exists(bitsandbytes_source):
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
        libnvinfer_plugin_symlink = os.path.join(site_packages_dir, "tensorrt", "libnvinfer_plugin.so.7")
        libnvinfer_symlink = os.path.join(site_packages_dir, "tensorrt", "libnvinfer.so.7")
        libcudart_symlink = os.path.join(site_packages_dir, "nvidia", "cuda_runtime", "lib", "libcudart.so.11.0")

        # Target file paths
        libnvinfer_plugin_target = os.path.join(site_packages_dir, "tensorrt", "libnvinfer_plugin.so.8")
        libnvinfer_target = os.path.join(site_packages_dir, "tensorrt", "libnvinfer.so.8")
        libcudart_target = os.path.join(site_packages_dir, "nvidia", "cuda_runtime", "lib", "libcudart.so.12")

        print("Checking symlinks now.")
        create_symlinks(libnvinfer_plugin_symlink, libnvinfer_plugin_target)
        create_symlinks(libnvinfer_symlink, libnvinfer_target)
        create_symlinks(libcudart_symlink, libcudart_target)

        tensorrt_dir = os.path.join(site_packages_dir, "tensorrt")
        if os.path.isdir(tensorrt_dir):
            os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{tensorrt_dir}"
        else:
            print(f"{tensorrt_dir} not found; not linking library.")

        cuda_runtime_dir = os.path.join(site_packages_dir, "nvidia", "cuda_runtime", "lib")
        if os.path.isdir(cuda_runtime_dir):
            os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{cuda_runtime_dir}"
        else:
            print(f"{cuda_runtime_dir} not found; not linking library.")


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


def update_kohya_ss(_dir, git_repo, branch, parent_dir, git_update):
    if git_update:
        if shutil.which("git"):
            # First, we make sure there are no changes that need to be made in git, so no work is lost.
            git_status = subprocess.run(["git", "-C", _dir, "status", "--porcelain=v1"], capture_output=True, text=True)
            if git_status.stdout.strip():
                print(f"These files need to be committed or discarded:")
                print(git_status.stdout)
                print(f"There are changes that need to be committed or discarded in the repo in {_dir}.")
                print(f"Commit those changes or run this script with -n to skip git operations entirely.")
                exit(1)

            print(f"Attempting to clone {git_repo}.")
            if not os.path.exists(os.path.join(_dir, ".git")):
                print(f"Cloning and switching to {git_repo}:{branch}")
                subprocess.run(["git", "-C", parent_dir, "clone", "-b", branch, git_repo, os.path.basename(_dir)])
                subprocess.run(["git", "-C", _dir, "switch", branch])
            else:
                print("git repo detected. Attempting to update repository instead.")
                print(f"Updating: {git_repo}")
                subprocess.run(["git", "-C", _dir, "pull", git_repo, branch])
                git_switch = subprocess.run(["git", "-C", _dir, "switch", branch], capture_output=True)
                if git_switch.returncode != 0:
                    print(f"Branch {branch} did not exist. Creating it.")
                    subprocess.run(["git", "-C", _dir, "switch", "-c", branch])
        else:
            print("You need to install git.")
            print("Rerun this after installing git or run this script with -n to skip the git operations.")
    else:
        print("Skipping git operations.")


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
                print(f"Error reading /System/Library/CoreServices/SystemVersion.plist: {e}")

        elif system == "Linux":
            if os.path.exists("/etc/os-release"):
                try:
                    with open("/etc/os-release", "r") as f:
                        content = f.read()
                        self.name = re.search(r'ID="?([^"\n]+)', content).group(1)
                        self.family = re.search(r'ID_LIKE="?([^"\n]+)', content).group(1)
                        self.version = re.search(r'VERSION="?([^"\n]+)', content).group(1)
                except Exception as e:
                    print(f"Error reading /etc/os-release: {e}")

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
                    print(f"Error reading /etc/redhat-release: {e}")

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


def install_python_dependencies(_dir, runpod, script_dir):
    # Following check disabled as PyCharm can't detect it's being used in a subprocess
    # noinspection PyUnusedLocal
    python_bin = None
    venv_python_bin = None

    # Check if python3 or python3.10 binary exists
    if shutil.which("python3"):
        python_bin = "python3"
    elif shutil.which("python3.10"):
        python_bin = "python3.10"
    else:
        print("Valid python3 or python3.10 binary not found.")
        print("Cannot proceed with the python steps.")
        return 1

    # Create and activate virtual environment if not in container environment
    if not in_container:
        print("Switching to virtual Python environment.")
        venv_path = os.path.join(_dir, "venv")
        subprocess.run([python_bin, "-m", "venv", venv_path])

        # Activate the virtual environment
        venv_bin_dir = os.path.join(venv_path, "bin") if os.name != "nt" else os.path.join(venv_path, "Scripts")
        venv_python_bin = os.path.join(venv_bin_dir, python_bin)

    # Update pip
    print("Checking for pip updates before Python operations.")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install python dependencies
    print("Installing python dependencies. This could take a few minutes as it downloads files.")
    if os_info.family == "Windows":
        subprocess.run([venv_python_bin, "-m", "pip", "install", "torch==1.12.1+cu116", "torchvision==0.13.1+cu116",
                        "--extra-index-url", "https://download.pytorch.org/whl/cu116"])
        subprocess.run([venv_python_bin, "-m", "pip", "install", "--use-pep517", "--upgrade", "-r", "requirements.txt"])
        subprocess.run([venv_python_bin, "-m", "pip", "install", "-U", "-I", "--no-deps",
                        "https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14"
                        ".dev0-cp310-cp310-win_amd64.whl"])
    elif platform.system() == "Linux":
        subprocess.run([venv_python_bin, "-m", "pip", "install", "torch==1.12.1+cu116", "torchvision==0.13.1+cu116",
                        "--extra-index-url", "https://download.pytorch.org/whl/cu116"])
        subprocess.run([venv_python_bin, "-m", "pip", "install", "-U", "-I", "--no-deps",
                        "https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/downloadlinux/xformers-0.0"
                        ".14.dev0-cp310-cp310-linux_x86_64.whl"])
    elif os_info.family == "Darwin":
        subprocess.run([venv_python_bin, "-m", "pip", "install", "torch==2.0.0", "torchvision==0.15.1", "-f",
                        "https://download.pytorch.org/whl/cpu/torch_stable.html"])

    if runpod:
        print("Installing tenssort.")
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorrt"])

    # Set the paths for the built-in requirements and temporary requirements files
    requirements_path = os.path.join(_dir, "requirements.txt")
    tmp_requirements_path = os.path.join(script_dir, "requirements_tmp.txt")
    # Copy the original requirements.txt and make the kohya_ss lib a dynamic location
    with (open(requirements_path), "r") as original_file, \
            open(tmp_requirements_path, "w") as temp_file:
        for line in original_file:
            if "#.*kohya_ss.*library" in line:
                line = line.replace(".", script_dir)
            temp_file.write(line)

    # Check if the OS is macOS, then determine if M1+ or Intel CPU
    # and append the appropriate packages to the requirements.txt file
    if os_info.family == "Darwin":
        with open(tmp_requirements_path, "a") as temp_file:
            # Check if the processor is Apple Silicon (arm64)
            if platform.machine() == "arm64":
                temp_file.write(f"tensorflow-macos=={TENSORFLOW_MACOS_VERSION}\n")
                temp_file.write(f"tensorflow-metal=={TENSORFLOW_METAL_VERSION}\n")
            # Check if the processor is Intel (x86_64)
            elif platform.machine() == "x86_64":
                temp_file.write(f"tensorflow=={TENSORFLOW_VERSION}\n")

    # Install the packages from the temporary requirements file
    pip_install_args = [sys.executable, "-m", "pip", "install", "--use-pep517", "--upgrade", "-r",
                        tmp_requirements_path]
    if args.verbosity <= 1:
        pip_install_args.insert(4, "--quiet")

    subprocess.run(pip_install_args, check=True, stderr=subprocess.PIPE)

    print("Removing the temp requirements file.")
    if os.path.isfile(tmp_requirements_path):
        os.remove(tmp_requirements_path)

    # Only exit the virtual environment if we aren't in a container.
    # This is because we never entered one at the beginning of the function if container detected.
    if not in_container():
        print("Exiting Python virtual environment.")
        sys.exit(0)


def configure_accelerate(interactive, source_config_file):
    print(f"Source accelerate config location: {source_config_file}")

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
                print(f"Target accelerate config location: {target_config_location}")
                shutil.copyfile(source_config_file, target_config_location)
                print(f"Copied accelerate config file to: {target_config_location}")
        else:
            print("Could not place the accelerate configuration file. Please configure manually.")
            os.system("accelerate config")


def launch_kohya_gui(_args):
    if not in_container():
        venv_path = os.path.join(_args.dir, "venv")
        kohya_gui_path = os.path.join(_args.dir, "kohya_gui.py")

        if not os.path.exists(venv_path):
            print("Error: Virtual environment not found")
            sys.exit(1)

        python_executable = os.path.join(venv_path, "bin", "python") if sys.platform != "win32" else os.path.join(
            venv_path, "Scripts", "python.exe")

        if not os.path.exists(python_executable):
            print("Error: Python executable not found in the virtual environment")
            sys.exit(1)
    else:
        python_executable = sys.executable
        kohya_gui_path = os.path.join(_args.dir, "kohya_gui.py")

    cmd = [
        python_executable,
        kohya_gui_path,
        "--listen", _args.gui_listen,
        "--username", _args.gui_username,
        "--password", _args.gui_password,
        "--server_port", str(_args.gui_server_port),
        "--inbrowser" if _args.gui_inbrowser else "",
        "--share" if _args.gui_share else "",
    ]

    subprocess.run(cmd, check=True)


def main(_args=None):
    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        print("Error: This script requires Python 3.10.")
        sys.exit(1)

    # Get the directory where the script is located
    script_directory = os.path.dirname(os.path.realpath(__file__))

    # Read config file or use defaults
    _config_file = _args.config_file if _args.config_file else os.path.join(script_directory,
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

    if not _args.GitRepo or not _dir or not _args.Branch:
        print(
            "Error: gitRepo, Branch, and Dir must have a value. Please provide values in the config file or through "
            "command line arguments.")
        sys.exit(1)

    # Define the directories relative to the install directory needed for install and launch
    parent_dir = os.path.dirname(_dir)
    venv_dir, site_packages_dir = get_venv_directory()

    # The main logic will go here after the sanity checks.
    check_and_create_install_folder(parent_dir, _dir)
    check_storage_space(_args.spaceCheck, _args.dir, parent_dir)
    update_kohya_ss(_dir, _args.git_repo, _args.branch, parent_dir, _args.gitUpdate)
    install_python_dependencies(_dir, _args.runpod, script_directory)
    setup_file_links(site_packages_dir, _args.runpod)
    configure_accelerate(args.interactive, args.config_file)
    launch_kohya_gui(args)


if __name__ == "__main__":
    config_file = parse_file_arg()
    config_data = load_config(config_file)
    args = parse_args(config_data)

    # Initialize log_level with a default value
    log_level = logging.ERROR

    # Set logging level based on the verbosity count
    if args.verbose == 0:
        log_level = logging.ERROR
    elif args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose == 2:
        log_level = logging.WARNING
    elif args.verbose >= 3:
        log_level = logging.DEBUG

    # Configure logging
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    # Use logging in the script
    # logging.debug("This is a debug message.")
    # logging.info("This is an info message.")
    # logging.warning("This is a warning message.")
    # logging.error("This is an error message.")

    os_info = get_os_info()
    print("OS name:", os_info.name)
    print("OS family:", os_info.family)
    print("OS version:", os_info.version)

    if args.verbose >= 3:
        for k, v in args.__dict__.items():
            logging.debug(f"{k}: {v}")

    if args.skip_setup:
        launch_kohya_gui(args)
        exit(0)
    else:
        main(args)
