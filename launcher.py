import argparse
import shutil
import sys
import yaml
import os
import subprocess
import platform
import logging

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
        parser.add_argument("-n", "--no-git-update", dest="noGitUpdate", action="store_true",
                            help="Do not update kohya_ss repo. No git pull or clone operations.")
        parser.add_argument("-p", "--public", dest="public", action="store_true",
                            help="Expose public URL in runpod mode. Won't have an effect in other modes.")
        parser.add_argument("-r", "--runpod", dest="runpod", action="store_true",
                            help="Forces a runpod installation. Useful if detection fails for any reason.")
        parser.add_argument("-s", "--skip-space-check", dest="skipSpaceCheck", action="store_true",
                            help="Skip the 10Gb minimum storage space check.")
        parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity levels up to 3.")

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


def install_python_dependencies(DIR, RUNPOD, VERBOSITY):
    # Check if python3 or python3.10 binary exists
    python_bin = None
    if shutil.which("python3"):
        python_bin = "python3"
    elif shutil.which("python3.10"):
        python_bin = "python3.10"
    else:
        print("Valid python3 or python3.10 binary not found.")
        print("Cannot proceed with the python steps.")
        return 1

    # Create virtual environment
    print("Switching to virtual Python environment.")
    venv_path = os.path.join(DIR, "venv")
    subprocess.run([python_bin, "-m", "venv", venv_path])

    # Activate the virtual environment
    venv_activate = os.path.join(venv_path, "bin", "activate_this.py")
    exec(open(venv_activate).read(), {'__file__': venv_activate})

    # Update pip
    print("Checking for pip updates before Python operations.")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install python dependencies
    print("Installing python dependencies. This could take a few minutes as it downloads files.")
    if platform.system() == "Linux":
        subprocess.run([sys.executable, "-m", "pip", "install", "torch==1.12.1+cu116", "torchvision==0.13.1+cu116",
                        "--extra-index-url", "https://download.pytorch.org/whl/cu116"])
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "-I", "--no-deps",
                        "https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/downloadlinux/xformers-0.0"
                        ".14.dev0-cp310-cp310-linux_x86_64.whl"])

    elif platform.system() == "Darwin":
        subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.0.0", "torchvision==0.15.1", "-f",
                        "https://download.pytorch.org/whl/cpu/torch_stable.html"])

    if RUNPOD:
        print("Installing tenssort.")
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorrt"])

    requirements_path = os.path.join(DIR, "requirements.txt")
    # Set the path for the temporary requirements file
    tmp_requirements_path = os.path.join(script_directory, "requirements_tmp.txt")
    # Copy the original requirements.txt and make the kohya_ss lib a dynamic location
    with open(os.path.join(script_directory, "requirements.txt"), "r") as original_file, \
            open(tmp_requirements_path, "w") as temp_file:
        for line in original_file:
            if "#.*kohya_ss.*library" in line:
                line = line.replace(".", script_directory)
            temp_file.write(line)

    # Check if the OS is macOS, then determine if M1+ or Intel CPU
    # and append the appropriate packages to the requirements.txt file
    if platform.system() == "Darwin":
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
    if VERBOSITY == 2:
        pip_install_args.insert(4, "--quiet")

    subprocess.run(pip_install_args, check=True, stderr=subprocess.PIPE)

    print("Removing the temp requirements file.")
    if os.path.isfile(tmp_requirements_path):
        os.remove(tmp_requirements_path)

    print("Exiting Python virtual environment.")
    sys.exit(0)


def main(_args=None):
    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        print("Error: This script requires Python 3.10.")
        sys.exit(1)

    # Get the directory where the script is located
    script_directory = os.path.dirname(os.path.realpath(__file__))

    # Read config file or use defaults
    _config_file = _args.config_file if _args.config_file else os.path.join(script_directory, "install_config.yaml")
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

    # The main logic will go here after the sanity checks.
    install_python_dependencies(_dir, _args.runpod, _args.verbosity)


if __name__ == "__main__":
    config_file = parse_file_arg()
    config_data = load_config(config_file)
    args = parse_args(config_data)

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

    for k, v in args.__dict__.items():
        logging.debug(f"{k}: {v}")

    main(args)
