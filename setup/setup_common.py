import subprocess
import os
import re
import sys
import filecmp
import logging
import shutil
import sysconfig
import datetime
import platform
import pkg_resources

errors = 0  # Define the 'errors' variable before using it
log = logging.getLogger('sd')

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

    # Check for nVidia toolkit or AMD toolkit
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
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                from library.ipex import ipex_init
                ipex_init()
                os.environ.setdefault('NEOReadDebugKeys', '1')
                os.environ.setdefault('ClDeviceGlobalMemSizeAvailablePercent', '100')
        except Exception:
            pass
        log.info(f'Torch {torch.__version__}')

        # Check if CUDA is available
        if not torch.cuda.is_available():
            log.warning('Torch reports CUDA not available')
        else:
            if torch.version.cuda:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    # Log Intel IPEX OneAPI version
                    log.info(f'Torch backend: Intel IPEX OneAPI {ipex.__version__}')
                else:
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
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    log.info(
                        f'Torch detected GPU: {torch.xpu.get_device_name(device)} VRAM {round(torch.xpu.get_device_properties(device).total_memory / 1024 / 1024)} Compute Units {torch.xpu.get_device_properties(device).max_compute_units}'
                    )
                else:
                    log.info(
                        f'Torch detected GPU: {torch.cuda.get_device_name(device)} VRAM {round(torch.cuda.get_device_properties(device).total_memory / 1024 / 1024)} Arch {torch.cuda.get_device_capability(device)} Cores {torch.cuda.get_device_properties(device).multi_processor_count}'
                    )
                return int(torch.__version__[0])
    except Exception as e:
        # log.warning(f'Could not load torch: {e}')
        return 0


# report current version of code
def check_repo_version(): # pylint: disable=unused-argument
    if os.path.exists('.release'):
        with open(os.path.join('./.release'), 'r', encoding='utf8') as file:
            release= file.read()
        
        log.info(f'Version: {release}')
    else:
        log.debug('Could not read release...')
    
# execute git command
def git(arg: str, folder: str = None, ignore: bool = False):
    #
    # This function was adapted from code written by vladimandic: https://github.com/vladmandic/automatic/commits/master
    #
    
    git_cmd = os.environ.get('GIT', "git")
    result = subprocess.run(f'"{git_cmd}" {arg}', check=False, shell=True, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=folder or '.')
    txt = result.stdout.decode(encoding="utf8", errors="ignore")
    if len(result.stderr) > 0:
        txt += ('\n' if len(txt) > 0 else '') + result.stderr.decode(encoding="utf8", errors="ignore")
    txt = txt.strip()
    if result.returncode != 0 and not ignore:
        global errors # pylint: disable=global-statement
        errors += 1
        log.error(f'Error running git: {folder} / {arg}')
        if 'or stash them' in txt:
            log.error(f'Local changes detected: check log for details...')
        log.debug(f'Git output: {txt}')


def pip(arg: str, ignore: bool = False, quiet: bool = False, show_stdout: bool = False):
    # arg = arg.replace('>=', '==')
    if not quiet:
        log.info(f'Installing package: {arg.replace("install", "").replace("--upgrade", "").replace("--no-deps", "").replace("--force", "").replace("  ", " ").strip()}')
    log.debug(f"Running pip: {arg}")
    if show_stdout:
        subprocess.run(f'"{sys.executable}" -m pip {arg}', shell=True, check=False, env=os.environ)
    else:
        result = subprocess.run(f'"{sys.executable}" -m pip {arg}', shell=True, check=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        txt = result.stdout.decode(encoding="utf8", errors="ignore")
        if len(result.stderr) > 0:
            txt += ('\n' if len(txt) > 0 else '') + result.stderr.decode(encoding="utf8", errors="ignore")
        txt = txt.strip()
        if result.returncode != 0 and not ignore:
            global errors # pylint: disable=global-statement
            errors += 1
            log.error(f'Error running pip: {arg}')
            log.debug(f'Pip output: {txt}')
        return txt


def installed(package, friendly: str = None):
    #
    # This function was adapted from code written by vladimandic: https://github.com/vladmandic/automatic/commits/master
    #
    
    # Remove brackets and their contents from the line using regular expressions
    # e.g., diffusers[torch]==0.10.2 becomes diffusers==0.10.2
    package = re.sub(r'\[.*?\]', '', package)

    try:
        if friendly:
            pkgs = friendly.split()
        else:
            pkgs = [
                p
                for p in package.split()
                if not p.startswith('-') and not p.startswith('=')
            ]
            pkgs = [
                p.split('/')[-1] for p in pkgs
            ]   # get only package name if installing from URL
        
        for pkg in pkgs:
            if '>=' in pkg:
                pkg_name, pkg_version = [x.strip() for x in pkg.split('>=')]
            elif '==' in pkg:
                pkg_name, pkg_version = [x.strip() for x in pkg.split('==')]
            else:
                pkg_name, pkg_version = pkg.strip(), None

            spec = pkg_resources.working_set.by_key.get(pkg_name, None)
            if spec is None:
                spec = pkg_resources.working_set.by_key.get(pkg_name.lower(), None)
            if spec is None:
                spec = pkg_resources.working_set.by_key.get(pkg_name.replace('_', '-'), None)

            if spec is not None:
                version = pkg_resources.get_distribution(pkg_name).version
                log.debug(f'Package version found: {pkg_name} {version}')

                if pkg_version is not None:
                    if '>=' in pkg:
                        ok = version >= pkg_version
                    else:
                        ok = version == pkg_version

                    if not ok:
                        log.warning(f'Package wrong version: {pkg_name} {version} required {pkg_version}')
                        return False
            else:
                log.debug(f'Package version not found: {pkg_name}')
                return False

        return True
    except ModuleNotFoundError:
        log.debug(f'Package not installed: {pkgs}')
        return False


# install package using pip if not already installed
def install(
    #
    # This function was adapted from code written by vladimandic: https://github.com/vladmandic/automatic/commits/master
    #
    package,
    friendly: str = None,
    ignore: bool = False,
    reinstall: bool = False,
    show_stdout: bool = False,
):
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


def run_cmd(run_cmd):
    try:
        subprocess.run(run_cmd, shell=True, check=False, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f'Error occurred while running command: {run_cmd}')
        print(f'Error: {e}')


# check python version
def check_python(ignore=True, skip_git=False):
    #
    # This function was adapted from code written by vladimandic: https://github.com/vladmandic/automatic/commits/master
    #

    supported_minors = [9, 10]
    log.info(f'Python {platform.python_version()} on {platform.system()}')
    if not (
        int(sys.version_info.major) == 3
        and int(sys.version_info.minor) in supported_minors
    ):
        log.error(
            f'Incompatible Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} required 3.{supported_minors}'
        )
        if not ignore:
            sys.exit(1)
    if not skip_git:
        git_cmd = os.environ.get('GIT', 'git')
        if shutil.which(git_cmd) is None:
            log.error('Git not found')
            if not ignore:
                sys.exit(1)
    else:
        git_version = git('--version', folder=None, ignore=False)
        log.debug(f'Git {git_version.replace("git version", "").strip()}')


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

