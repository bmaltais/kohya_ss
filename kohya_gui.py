#!/usr/bin/env python3
# noinspection DuplicatedCode

import argparse
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime

import gradio as gr
import yaml

from dreambooth_gui import dreambooth_tab
from finetune_gui import finetune_tab
from library.extract_lora_from_dylora_gui import gradio_extract_dylora_tab
from library.extract_lora_gui import gradio_extract_lora_tab
from library.extract_lycoris_locon_gui import gradio_extract_lycoris_locon_tab
from library.merge_lora_gui import gradio_merge_lora_tab
from library.merge_lycoris_gui import gradio_merge_lycoris_tab
from library.resize_lora_gui import gradio_resize_lora_tab
from library.utilities import utilities_tab
from lora_gui import lora_tab
from textual_inversion_gui import ti_tab


# noinspection PyPep8Naming
def UI(**kwargs):
    headless = kwargs.get('headless', False)
    logging.debug(f'Headless mode: {headless}.')

    css = ''
    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            logging.debug('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Blocks(css=css, title='Kohya_ss GUI')

    with interface:
        with gr.Tab('Dreambooth'):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = dreambooth_tab(headless=headless)
        with gr.Tab('Dreambooth LoRA'):
            lora_tab(headless=headless)
        with gr.Tab('Dreambooth TI'):
            ti_tab(headless=headless)
        with gr.Tab('Finetune'):
            finetune_tab(headless=headless)
        with gr.Tab('Utilities'):
            utilities_tab(
                train_data_dir_input=train_data_dir_input,
                reg_data_dir_input=reg_data_dir_input,
                output_dir_input=output_dir_input,
                logging_dir_input=logging_dir_input,
                enable_copy_info_button=True,
                headless=headless,
            )
            gradio_extract_dylora_tab(headless=headless)
            gradio_extract_lora_tab(headless=headless)
            gradio_extract_lycoris_locon_tab(headless=headless)
            gradio_merge_lora_tab(headless=headless)
            gradio_merge_lycoris_tab(headless=headless)
            gradio_resize_lora_tab(headless=headless)

    # Show the interface
    launch_kwargs = {}
    username = kwargs.get('username')
    password = kwargs.get('password')
    server_port = kwargs.get('server_port', 0)
    inbrowser = kwargs.get('inbrowser', False)
    share = kwargs.get('share', False)
    server_name = kwargs.get('listen')

    launch_kwargs['server_name'] = server_name
    if username and password:
        launch_kwargs['auth'] = (username, password)
    if server_port > 0:
        launch_kwargs['server_port'] = server_port
    if inbrowser:
        launch_kwargs['inbrowser'] = inbrowser
    if share:
        launch_kwargs['share'] = share
    interface.launch(**launch_kwargs)


def run_command(command):
    try:
        return subprocess.check_output(command, shell=True).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return "Command failed, possibly due to lack of administrative privileges."
    except FileNotFoundError:
        return "Command not found."
    except Exception as e:
        return f"Command failed due to an unknown error: {str(e)}"


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    from shutil import which
    return which(name) is not None


def get_cpu_manufacturer():
    cpu_info = platform.processor().lower()
    if 'intel' in cpu_info:
        return 'Intel'
    elif 'amd' in cpu_info:
        return 'AMD'
    elif 'arm' in cpu_info:
        return 'ARM'
    else:
        return "Could not obtain CPU manufacturer."


# noinspection SpellCheckingInspection
def check_gpu_vram(os_type):
    nvidia_smi = shutil.which("nvidia-smi")

    if os_type == "Windows" and nvidia_smi is not None:
        try:
            output = subprocess.check_output([nvidia_smi, '--query-gpu=memory.total,name', '--format=csv'])
            output = output.decode('utf-8').strip().split('\n')[1:]
            gpu_info = [line.split(', ') for line in output]
            gpu_vram, gpu_name = gpu_info[0]
            gpu_vram = gpu_vram.replace(' MiB', '')
            return gpu_vram, gpu_name, int(gpu_vram) < 8000
        except (subprocess.CalledProcessError, IndexError):
            return "N/A", "N/A", False

    elif os_type == "Linux":
        try:
            # Get GPU info
            output = subprocess.check_output(['lspci', '-v']).decode().strip()
            gpu_info = re.search(r'VGA compatible controller: (.*) \(rev', output).group(1)

            # Get VRAM
            if "NVIDIA" in gpu_info and nvidia_smi is not None:
                output = subprocess.check_output(
                    [nvidia_smi, '--query-gpu=memory.total', '--format=csv,noheader,nounits']).decode().strip()
                gpu_vram = output
            else:
                gpu_vram = re.search(r'Memory at [a-f0-9]{8} ([a-f0-9]{8})', output)
                if gpu_vram is not None:
                    gpu_vram = int(gpu_vram.group(1), 16) // 1024 // 1024
                else:
                    output = subprocess.check_output(['glxinfo', '-B']).decode().strip()
                    gpu_vram = re.search(r'Video memory: (\d+)', output)
                    if gpu_vram is not None:
                        gpu_vram = gpu_vram.group(1)
                    else:
                        gpu_vram = "N/A"

            return gpu_vram, gpu_info, False if gpu_vram == "N/A" else int(gpu_vram) < 8000
        except (subprocess.CalledProcessError, IndexError):
            return "N/A", "N/A", False

    elif os_type == "Darwin":
        try:
            # Get GPU info
            output = subprocess.check_output(['system_profiler', 'SPDisplaysDataType']).decode().strip()
            gpu_info = re.search('Chipset Model: (.*)', output).group(1).strip()

            # Get VRAM
            output = subprocess.check_output(['system_profiler', 'SPDisplaysDataType']).decode().strip()
            gpu_vram = re.search(r'VRAM \(Total\): (\d+)', output).group(1).strip()

            return gpu_vram, gpu_info, int(gpu_vram) < 8000
        except (subprocess.CalledProcessError, IndexError):
            return "N/A", "N/A", False

    else:
        return "N/A", "N/A", False


# noinspection PyDictCreation,SpellCheckingInspection
def debug_system_info():
    os_type = platform.system()

    # Get OS, CPU and GPU information
    _system_info = {}

    # Get System and Python information
    _system_info['OS'] = platform.system()
    _system_info['Release'] = platform.release()
    _system_info['Version'] = platform.version()
    _system_info['Processor'] = platform.processor()
    _system_info['CPU Arch'] = platform.machine()
    _system_info['Python Version'] = platform.python_version()
    _system_info['Python Implementation'] = platform.python_implementation()
    _system_info['Python Compiler'] = platform.python_compiler()
    _system_info['GPU VRAM'], _system_info['GPU'], gpu_vram_warning = check_gpu_vram(os_type)

    if gpu_vram_warning:
        logging.critical("\nIf you have less than 8Gb of VRAM, you may see performance issues.\n")

    # Get CPU manufacturer
    try:
        if os_type == "Windows":
            output = subprocess.check_output(['wmic', 'cpu', 'get', 'Manufacturer'],
                                             stderr=subprocess.STDOUT).decode().strip()
            _system_info['CPU Manufacturer'] = output.split('\n')[1].strip()
        elif os_type == "Linux":
            output = subprocess.check_output(['lscpu']).decode().strip()
            cpu_info = re.search('Vendor ID:(.*)', output)
            if cpu_info is not None:
                _system_info['CPU Manufacturer'] = cpu_info.group(1).strip()
            else:  # Try another way to get the CPU info
                output = subprocess.check_output(['cat', '/proc/cpuinfo']).decode().strip()
                cpu_info = re.search('vendor_id\\s+: (.*)', output)
                if cpu_info is not None:
                    _system_info['CPU Manufacturer'] = cpu_info.group(1).strip()
                else:
                    _system_info['CPU Manufacturer'] = "Unknown"
        elif os_type == "Darwin":
            output = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.vendor']).decode().strip()
            if output:
                _system_info['CPU Manufacturer'] = output
            else:
                output = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
                _system_info['CPU Manufacturer'] = output if output else "Unknown"
        else:
            _system_info['CPU Manufacturer'] = "Unknown"
    except Exception as e:
        _system_info['CPU Manufacturer'] = "Could not get system information: " + str(e)

    # Get GPU information
    try:
        if _system_info['OS'] == "Windows":
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                                 stderr=subprocess.STDOUT).decode().strip()
                _system_info['GPU'] = output
            except (FileNotFoundError, subprocess.CalledProcessError):
                output = subprocess.check_output(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                                                 stderr=subprocess.STDOUT).decode().strip()
                _system_info['GPU'] = output.split('\n')[1].strip()
        elif _system_info['OS'] == "Linux":
            output = subprocess.check_output(['lspci']).decode().strip()
            for line in output.split('\n'):
                if 'VGA compatible controller' in line:
                    _system_info['GPU'] = line.split(':')[2].strip().split('[')[0].strip()
                    break
        elif _system_info['OS'] == "Darwin":
            output = subprocess.check_output(['system_profiler', 'SPDisplaysDataType'],
                                             stderr=subprocess.STDOUT).decode().strip()
            _system_info['GPU'] = re.search('Chipset Model:(.*)', output).group(1).strip()
        else:
            _system_info['GPU'] = "Unknown"
    except Exception as e:
        _system_info['GPU'] = "Could not get information: " + str(e)

    # Check for virtual environment
    _system_info['Virtual Environment'] = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and
                                                                          sys.base_prefix != sys.prefix)
    return _system_info


# noinspection DuplicatedCode
def find_config_file(config_file_locations):
    for location in config_file_locations:
        abs_location = os.path.abspath(location)
        if os.path.isfile(abs_location):
            return abs_location
    return None


def load_config(_config_file=None):
    # Define config file locations
    if sys.platform == "win32":
        config_file_locations = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_files", "installation",
                         "install_config.yml"),
            os.path.join(os.environ.get("USERPROFILE", ""),
                         ".kohya_ss", "install_config.yml"),
            os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "install_config.yml")
        ]
    else:
        config_file_locations = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_files", "installation",
                         "install_config.yml"),
            os.path.join(os.environ.get("HOME", ""),
                         ".kohya_ss", "install_config.yml"),
            os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "install_config.yml"),
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
                setattr(_args, arg_name, os.path.abspath(expanded_path_value))


def parse_args(_config_data):
    # Define the default arguments first. The spacing is purely for readability.
    default_args = [
        {"short": "-f", "long": "--file", "default": "install_config.yml", "type": str,
         "help": "Configuration file with installation options.", "is_path": True},

        {"short": "-l", "long": "--log-dir", "default": None, "type": str,
         "help": "Override the default log directory.", "is_path": True},

        {"short": "-p", "long": "--public", "default": False, "type": bool,
         "help": "Expose public URL in runpod mode. Won't have an effect in other modes."},

        {"short": "-v", "long": "--verbosity", "default": '0', "type": str,
         "help": "Increase verbosity levels. Use multiple times (e.g., -vvv) or specify number (e.g., -v 4).",
         "action": CountOccurrencesAction},

        {"short": "", "long": "--headless", "default": False, "type": bool,
         "help": "Headless mode will not display the native windowing toolkit. Useful for remote deployments."},

        {"short": None, "long": "--listen", "default": "127.0.0.1", "type": str,
         "help": "IP to listen on for connections to Gradio."},

        {"short": "", "long": "--username", "default": "",
         "type": str, "help": "Username for authentication."},

        {"short": "", "long": "--password", "default": "",
         "type": str, "help": "Password for authentication."},

        {"short": "", "long": "--server-port", "default": 0, "type": int,
         "help": "The port number the GUI server should use."},

        {"short": "", "long": "--inbrowser", "default": False,
         "type": bool, "help": "Open in browser."},

        {"short": "", "long": "--share", "default": False,
         "type": bool, "help": "Share the gradio UI."},
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
            # Get the name of the type and convert to upper case
            _arg_type = _arg_type.__name__.upper()
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


# This custom action was added so that the v option could be used Windows-style with integers (-v 3) setting the
# verbosity and Unix style (-vvv).
# noinspection DuplicatedCode
class CountOccurrencesAction(argparse.Action):
    def __call__(self, _parser, namespace, values, option_string=None):
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


def get_logs_dir(_args):
    if getattr(_args, "log-dir"):
        os.path.expanduser(getattr(_args, "log-dir"))
        _logs_dir = os.path.abspath(getattr(_args, "log-dir"))
    else:
        _logs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")

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
            log_filename = f"kohya_ss_{current_time_str}.log"
        else:
            log_filename = f"kohya_ss_{current_time_str}_{log_level_name}.log"
        log_filepath = os.path.join(date_subdir, log_filename)

        return log_filepath


# noinspection DuplicatedCode
def find_python_binary():
    possible_binaries = ["python3.10", "python310", "python3", "python"]

    if sys.platform == 'win32':
        possible_binaries = [
                                binary + ".exe" for binary in possible_binaries] + possible_binaries

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


if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)
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
    args.log_dir = os.path.abspath(get_logs_dir(args))
    log_file = CustomFormatter.generate_log_filename(args.log_dir)
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())

    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_file, mode='w')])
    logging.getLogger().setLevel(log_level)

    # Replace 'root' with an empty string in the logger name
    for handler in logging.getLogger().handlers:
        handler.setFormatter(CustomFormatter())

    print(f"Logs will be stored in: {args.log_dir}")

    # Check if python3 or python3.10 binary exists
    # noinspection DuplicatedCode
    python_bin = find_python_binary()
    if not python_bin:
        logging.error("Valid python3 or python3.10 binary not found.")
        logging.error("Cannot proceed with the python steps.")
        exit(1)

    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        logging.info("Error: This script requires Python 3.10.")
        logging.debug(
            f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)

    if args.verbosity >= 3:
        # Get system information
        system_info = debug_system_info()

        logging.debug(
            "\nOS: '%s'"
            "\nCPU Arch: '%s'"
            "\nCPU Manufacturer: '%s'"
            "\nGPU: '%s'"
            "\nGPU VRAM: '%s'"
            "\nRelease: '%s'"
            "\nVersion: '%s'"
            "\nPython Version: '%s'"
            "\nPython Implementation: '%s'"
            "\nPython Compiler: '%s'"
            "\nVirtual Environment: '%s'",
            system_info['OS'],
            system_info['CPU Arch'],
            system_info['CPU Manufacturer'],
            system_info['GPU'],
            system_info['GPU VRAM'],
            system_info['Release'],
            system_info['Version'],
            system_info['Python Version'],
            system_info['Python Implementation'],
            system_info['Python Compiler'],
            system_info['Virtual Environment']
        )
    else:
        _, _, vram_warning = check_gpu_vram(platform.system())
        if vram_warning:
            logging.critical("We detected less than 8Gb of VRAM. You may see performance issues.")

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
        headless=args.headless,
    )
