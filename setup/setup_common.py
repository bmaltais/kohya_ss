import subprocess
import os
import re
import sys
import logging
import shutil
import datetime
import pkg_resources

errors = 0  # Define the 'errors' variable before using it
log = logging.getLogger('sd')

def check_python_version():
    """
    Check if the current Python version is within the acceptable range.
    
    Returns:
    bool: True if the current Python version is valid, False otherwise.
    """
    min_version = (3, 10, 9)
    max_version = (3, 11, 0)
    
    from packaging import version
    
    try:
        current_version = sys.version_info
        log.info(f"Python version is {sys.version}")
        
        if not (min_version <= current_version < max_version):
            log.error(f"The current version of python ({current_version}) is not appropriate to run Kohya_ss GUI")
            log.error("The python version needs to be greater or equal to 3.10.9 and less than 3.11.0")
            return False
        return True
    except Exception as e:
        log.error(f"Failed to verify Python version. Error: {e}")
        return False

def update_submodule(quiet=True):
    """
    Ensure the submodule is initialized and updated.
    
    This function uses the Git command line interface to initialize and update 
    the specified submodule recursively. Errors during the Git operation
    or if Git is not found are caught and logged.
    
    Parameters:
    - quiet: If True, suppresses the output of the Git command.
    """
    git_command = ["git", "submodule", "update", "--init", "--recursive"]
    
    if quiet:
        git_command.append("--quiet")
        
    try:
        # Initialize and update the submodule
        subprocess.run(git_command, check=True)
        log.info("Submodule initialized and updated.")
        
    except subprocess.CalledProcessError as e:
        # Log the error if the Git operation fails
        log.error(f"Error during Git operation: {e}")
    except FileNotFoundError as e:
        # Log the error if the file is not found
        log.error(e)

# def read_tag_version_from_file(file_path):
#     """
#     Read the tag version from a given file.

#     Parameters:
#     - file_path: The path to the file containing the tag version.

#     Returns:
#     The tag version as a string.
#     """
#     with open(file_path, 'r') as file:
#         # Read the first line and strip whitespace
#         tag_version = file.readline().strip()
#     return tag_version

def clone_or_checkout(repo_url, branch_or_tag, directory_name):
    """
    Clone a repo or checkout a specific branch or tag if the repo already exists.
    For branches, it updates to the latest version before checking out.
    Suppresses detached HEAD advice for tags or specific commits.
    Restores the original working directory after operations.

    Parameters:
    - repo_url: The URL of the Git repository.
    - branch_or_tag: The name of the branch or tag to clone or checkout.
    - directory_name: The name of the directory to clone into or where the repo already exists.
    """
    original_dir = os.getcwd()  # Store the original directory
    try:
        if not os.path.exists(directory_name):
            # Directory does not exist, clone the repo quietly
            
            # Construct the command as a string for logging
            # run_cmd = f"git clone --branch {branch_or_tag} --single-branch --quiet {repo_url} {directory_name}"
            run_cmd = ["git", "clone", "--branch", branch_or_tag, "--single-branch", "--quiet", repo_url, directory_name]


            # Log the command
            log.debug(run_cmd)
            
            # Run the command
            process = subprocess.Popen(
                run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output, error = process.communicate()
            
            if error and not error.startswith("Note: switching to"):
                log.warning(error)
            else:
                log.info(f"Successfully cloned sd-scripts {branch_or_tag}")
            
        else:
            os.chdir(directory_name)
            subprocess.run(["git", "fetch", "--all", "--quiet"], check=True)
            subprocess.run(["git", "config", "advice.detachedHead", "false"], check=True)
            
            # Get the current branch or commit hash
            current_branch_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
            tag_branch_hash = subprocess.check_output(["git", "rev-parse", branch_or_tag]).strip().decode()
            
            if current_branch_hash != tag_branch_hash:
                run_cmd = f"git checkout {branch_or_tag} --quiet"
                # Log the command
                log.debug(run_cmd)
                
                # Execute the checkout command
                process = subprocess.Popen(run_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output, error = process.communicate()
                
                if error:
                    log.warning(error.decode())
                else:
                    log.info(f"Checked out sd-scripts {branch_or_tag} successfully.")
            else:
                log.info(f"Current branch of sd-scripts is already at the required release {branch_or_tag}.")
    except subprocess.CalledProcessError as e:
        log.error(f"Error during Git operation: {e}")
    finally:
        os.chdir(original_dir)  # Restore the original directory

# setup console and file logging
def setup_logging(clean=False):
    #
    # This function was adapted from code written by vladimandic: https://github.com/vladmandic/automatic/commits/master
    #

    from rich.theme import Theme
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.pretty import install as pretty_install
    from rich.traceback import install as traceback_install

    console = Console(
        log_time=True,
        log_time_format='%H:%M:%S-%f',
        theme=Theme(
            {
                'traceback.border': 'black',
                'traceback.border.syntax_error': 'black',
                'inspect.value.border': 'black',
            }
        ),
    )
    # logging.getLogger("urllib3").setLevel(logging.ERROR)
    # logging.getLogger("httpx").setLevel(logging.ERROR)

    current_datetime = datetime.datetime.now()
    current_datetime_str = current_datetime.strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(
        os.path.dirname(__file__),
        f'../logs/setup/kohya_ss_gui_{current_datetime_str}.log',
    )

    # Create directories if they don't exist
    log_directory = os.path.dirname(log_file)
    os.makedirs(log_directory, exist_ok=True)

    level = logging.INFO
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(message)s',
        filename=log_file,
        filemode='a',
        encoding='utf-8',
        force=True,
    )
    log.setLevel(
        logging.DEBUG
    )   # log to file is always at level debug for facility `sd`
    pretty_install(console=console)
    traceback_install(
        console=console,
        extra_lines=1,
        width=console.width,
        word_wrap=False,
        indent_guides=False,
        suppress=[],
    )
    rh = RichHandler(
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format='%H:%M:%S-%f',
        level=level,
        console=console,
    )
    rh.set_name(level)
    while log.hasHandlers() and len(log.handlers) > 0:
        log.removeHandler(log.handlers[0])
    log.addHandler(rh)


def install_requirements_inbulk(requirements_file, show_stdout=True, optional_parm="", upgrade = False):
    if not os.path.exists(requirements_file):
        log.error(f'Could not find the requirements file in {requirements_file}.')
        return

    log.info(f'Installing requirements from {requirements_file}...')

    if upgrade:
        optional_parm += " -U"

    if show_stdout:
        run_cmd(f'pip install -r {requirements_file} {optional_parm}')
    else:
        run_cmd(f'pip install -r {requirements_file} {optional_parm} --quiet')
    log.info(f'Requirements from {requirements_file} installed.')
    


def configure_accelerate(run_accelerate=False):
    #
    # This function was taken and adapted from code written by jstayco
    #

    from pathlib import Path

    def env_var_exists(var_name):
        return var_name in os.environ and os.environ[var_name] != ''

    log.info('Configuring accelerate...')
    
    source_accelerate_config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        'config_files',
        'accelerate',
        'default_config.yaml',
    )

    if not os.path.exists(source_accelerate_config_file):
        if run_accelerate:
            run_cmd('accelerate config')
        else:
            log.warning(
                f'Could not find the accelerate configuration file in {source_accelerate_config_file}. Please configure accelerate manually by runningthe option in the menu.'
            )
    
    log.debug(
        f'Source accelerate config location: {source_accelerate_config_file}'
    )

    target_config_location = None

    log.debug(
        f"Environment variables: HF_HOME: {os.environ.get('HF_HOME')}, "
        f"LOCALAPPDATA: {os.environ.get('LOCALAPPDATA')}, "
        f"USERPROFILE: {os.environ.get('USERPROFILE')}"
    )
    if env_var_exists('HF_HOME'):
        target_config_location = Path(
            os.environ['HF_HOME'], 'accelerate', 'default_config.yaml'
        )
    elif env_var_exists('LOCALAPPDATA'):
        target_config_location = Path(
            os.environ['LOCALAPPDATA'],
            'huggingface',
            'accelerate',
            'default_config.yaml',
        )
    elif env_var_exists('USERPROFILE'):
        target_config_location = Path(
            os.environ['USERPROFILE'],
            '.cache',
            'huggingface',
            'accelerate',
            'default_config.yaml',
        )

    log.debug(f'Target config location: {target_config_location}')

    if target_config_location:
        if not target_config_location.is_file():
            target_config_location.parent.mkdir(parents=True, exist_ok=True)
            log.debug(
                f'Target accelerate config location: {target_config_location}'
            )
            shutil.copyfile(
                source_accelerate_config_file, target_config_location
            )
            log.info(
                f'Copied accelerate config file to: {target_config_location}'
            )
        else:
            if run_accelerate:
                run_cmd('accelerate config')
            else:
                log.warning(
                    'Could not automatically configure accelerate. Please manually configure accelerate with the option in the menu or with: accelerate config.'
                )
    else:
        if run_accelerate:
            run_cmd('accelerate config')
        else:
            log.warning(
                'Could not automatically configure accelerate. Please manually configure accelerate with the option in the menu or with: accelerate config.'
            )


def check_torch():
    #
    # This function was adapted from code written by vladimandic: https://github.com/vladmandic/automatic/commits/master
    #

    # Check for toolkit
    if shutil.which('nvidia-smi') is not None or os.path.exists(
        os.path.join(
            os.environ.get('SystemRoot') or r'C:\Windows',
            'System32',
            'nvidia-smi.exe',
        )
    ):
        log.info('nVidia toolkit detected')
    elif shutil.which('rocminfo') is not None or os.path.exists(
        '/opt/rocm/bin/rocminfo'
    ):
        log.info('AMD toolkit detected')
    elif (shutil.which('sycl-ls') is not None
    or os.environ.get('ONEAPI_ROOT') is not None
    or os.path.exists('/opt/intel/oneapi')):
        log.info('Intel OneAPI toolkit detected')
    else:
        log.info('Using CPU-only Torch')

    try:
        import torch
        try:
            # Import IPEX / XPU support
            import intel_extension_for_pytorch as ipex
        except Exception:
            pass
        log.info(f'Torch {torch.__version__}')

        if torch.cuda.is_available():
            if torch.version.cuda:
                # Log nVidia CUDA and cuDNN versions
                log.info(
                    f'Torch backend: nVidia CUDA {torch.version.cuda} cuDNN {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"}'
                )
            elif torch.version.hip:
                # Log AMD ROCm HIP version
                log.info(f'Torch backend: AMD ROCm HIP {torch.version.hip}')
            else:
                log.warning('Unknown Torch backend')

            # Log information about detected GPUs
            for device in [
                torch.cuda.device(i) for i in range(torch.cuda.device_count())
            ]:
                log.info(
                    f'Torch detected GPU: {torch.cuda.get_device_name(device)} VRAM {round(torch.cuda.get_device_properties(device).total_memory / 1024 / 1024)} Arch {torch.cuda.get_device_capability(device)} Cores {torch.cuda.get_device_properties(device).multi_processor_count}'
                )
        # Check if XPU is available
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            # Log Intel IPEX version
            log.info(f'Torch backend: Intel IPEX {ipex.__version__}')
            for device in [
                torch.xpu.device(i) for i in range(torch.xpu.device_count())
            ]:
                log.info(
                    f'Torch detected GPU: {torch.xpu.get_device_name(device)} VRAM {round(torch.xpu.get_device_properties(device).total_memory / 1024 / 1024)} Compute Units {torch.xpu.get_device_properties(device).max_compute_units}'
                )
        else:
            log.warning('Torch reports GPU not available')
        
        return int(torch.__version__[0])
    except Exception as e:
        # log.warning(f'Could not load torch: {e}')
        return 0


# report current version of code
def check_repo_version():
    """
    This function checks the version of the repository by reading the contents of a file named '.release'
    in the current directory. If the file exists, it reads the release version from the file and logs it.
    If the file does not exist, it logs a debug message indicating that the release could not be read.
    """
    if os.path.exists('.release'):
        try:
            with open(os.path.join('./.release'), 'r', encoding='utf8') as file:
                release= file.read()
            
            log.info(f'Kohya_ss GUI version: {release}')
        except Exception as e:
            log.error(f'Could not read release: {e}')
    else:
        log.debug('Could not read release...')
    
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
    This function was adapted from code written by vladimandic: https://github.com/vladmandic/automatic/commits/master
    """
    
    # git_cmd = os.environ.get('GIT', "git")
    result = subprocess.run(["git", arg], check=False, shell=True, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=folder or '.')
    txt = result.stdout.decode(encoding="utf8", errors="ignore")
    if len(result.stderr) > 0:
        txt += ('\n' if len(txt) > 0 else '') + result.stderr.decode(encoding="utf8", errors="ignore")
    txt = txt.strip()
    if result.returncode != 0 and not ignore:
        global errors
        errors += 1
        log.error(f'Error running git: {folder} / {arg}')
        if 'or stash them' in txt:
            log.error(f'Local changes detected: check log for details...')
        log.debug(f'Git output: {txt}')


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
    # arg = arg.replace('>=', '==')
    if not quiet:
        log.info(f'Installing package: {arg.replace("install", "").replace("--upgrade", "").replace("--no-deps", "").replace("--force", "").replace("  ", " ").strip()}')
    pip_cmd = [fr"{sys.executable}", "-m", "pip"] + arg.split(" ")
    log.debug(f"Running pip: {pip_cmd}")
    if show_stdout:
        subprocess.run(pip_cmd, shell=False, check=False, env=os.environ)
    else:
        result = subprocess.run(pip_cmd, shell=False, check=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        txt = result.stdout.decode(encoding="utf8", errors="ignore")
        if len(result.stderr) > 0:
            txt += ('\n' if len(txt) > 0 else '') + result.stderr.decode(encoding="utf8", errors="ignore")
        txt = txt.strip()
        if result.returncode != 0 and not ignore:
            global errors # pylint: disable=global-statement
            errors += 1
            log.error(f'Error running pip: {arg}')
            log.error(f'Pip output: {txt}')
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
    
    # Remove any optional features specified in brackets (e.g., "package[option]==version" becomes "package==version")
    package = re.sub(r'\[.*?\]', '', package)

    try:
        if friendly:
            # If a 'friendly' version of the package string is provided, split it into components
            pkgs = friendly.split()
            
            # Filter out command-line options and URLs from the package specification
            pkgs = [
                p
                for p in package.split()
                if not p.startswith('--') and "://" not in p
            ]
        else:
            # Split the package string into components, excluding '-' and '=' prefixed items
            pkgs = [
                p
                for p in package.split()
                if not p.startswith('-') and not p.startswith('=')
            ]
            # For each package component, extract the package name, excluding any URLs
            pkgs = [
                p.split('/')[-1] for p in pkgs
            ]

        for pkg in pkgs:
            # Parse the package name and version based on the version specifier used
            if '>=' in pkg:
                pkg_name, pkg_version = [x.strip() for x in pkg.split('>=')]
            elif '==' in pkg:
                pkg_name, pkg_version = [x.strip() for x in pkg.split('==')]
            else:
                pkg_name, pkg_version = pkg.strip(), None

            # Attempt to find the installed package by its name
            spec = pkg_resources.working_set.by_key.get(pkg_name, None)
            if spec is None:
                # Try again with lowercase name
                spec = pkg_resources.working_set.by_key.get(pkg_name.lower(), None)
            if spec is None:
                # Try replacing underscores with dashes
                spec = pkg_resources.working_set.by_key.get(pkg_name.replace('_', '-'), None)

            if spec is not None:
                # Package is found, check version
                version = pkg_resources.get_distribution(pkg_name).version
                log.debug(f'Package version found: {pkg_name} {version}')

                if pkg_version is not None:
                    # Verify if the installed version meets the specified constraints
                    if '>=' in pkg:
                        ok = version >= pkg_version
                    else:
                        ok = version == pkg_version

                    if not ok:
                        # Version mismatch, log warning and return False
                        log.warning(f'Package wrong version: {pkg_name} {version} required {pkg_version}')
                        return False
            else:
                # Package not found, log debug message and return False
                log.debug(f'Package version not found: {pkg_name}')
                return False

        # All specified packages are installed with the correct versions
        return True
    except ModuleNotFoundError:
        # One or more packages are not installed, log debug message and return False
        log.debug(f'Package not installed: {pkgs}')
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
    # Remove anything after '#' in the package variable
    package = package.split('#')[0].strip()

    if reinstall:
        global quick_allowed   # pylint: disable=global-statement
        quick_allowed = False
    if reinstall or not installed(package, friendly):
        pip(f'install --upgrade {package}', ignore=ignore, show_stdout=show_stdout)


def process_requirements_line(line, show_stdout: bool = False):
    # Remove brackets and their contents from the line using regular expressions
    # e.g., diffusers[torch]==0.10.2 becomes diffusers==0.10.2
    package_name = re.sub(r'\[.*?\]', '', line)
    install(line, package_name, show_stdout=show_stdout)


def install_requirements(requirements_file, check_no_verify_flag=False, show_stdout: bool = False):
    if check_no_verify_flag:
        log.info(f'Verifying modules installation status from {requirements_file}...')
    else:
        log.info(f'Installing modules from {requirements_file}...')
    with open(requirements_file, 'r', encoding='utf8') as f:
        # Read lines from the requirements file, strip whitespace, and filter out empty lines, comments, and lines starting with '.'
        if check_no_verify_flag:
            lines = [
                line.strip()
                for line in f.readlines()
                if line.strip() != ''
                and not line.startswith('#')
                and line is not None
                and 'no_verify' not in line
            ]
        else:
            lines = [
                line.strip()
                for line in f.readlines()
                if line.strip() != ''
                and not line.startswith('#')
                and line is not None
            ]

        # Iterate over each line and install the requirements
        for line in lines:
            # Check if the line starts with '-r' to include another requirements file
            if line.startswith('-r'):
                # Get the path to the included requirements file
                included_file = line[2:].strip()
                # Expand the included requirements file recursively
                install_requirements(included_file, check_no_verify_flag=check_no_verify_flag, show_stdout=show_stdout)
            else:
                process_requirements_line(line, show_stdout=show_stdout)


def ensure_base_requirements():
    try:
        import rich   # pylint: disable=unused-import
    except ImportError:
        install('--upgrade rich', 'rich')
        
    try:
        import packaging
    except ImportError:
        install('packaging')


def run_cmd(run_cmd):
    try:
        subprocess.run(run_cmd, shell=True, check=False, env=os.environ)
    except subprocess.CalledProcessError as e:
        log.error(f'Error occurred while running command: {run_cmd}')
        log.error(f'Error: {e}')


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def write_to_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
    except IOError as e:
        print(f'Error occurred while writing to file: {file_path}')
        print(f'Error: {e}')


def clear_screen():
    # Check the current operating system to execute the correct clear screen command
    if os.name == 'nt':  # If the operating system is Windows
        os.system('cls')
    else:  # If the operating system is Linux or Mac
        os.system('clear')

