import subprocess
import os
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

# ANSI escape code for yellow color
YELLOW = '\033[93m'

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


def configure_accelerate():
    #
    # This function was taken and adapted from code written by jstayco
    #

    from pathlib import Path

    def env_var_exists(var_name):
        return var_name in os.environ and os.environ[var_name] != ''

    source_accelerate_config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'config_files',
        'accelerate',
        'default_config.yaml',
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
        log.info('Configured accelerate...')
        if not target_config_location.is_file():
            target_config_location.parent.mkdir(parents=True, exist_ok=True)
            log.debug(
                f'Target accelerate config location: {target_config_location}'
            )
            shutil.copyfile(
                source_accelerate_config_file, target_config_location
            )
            log.debug(
                f'Copied accelerate config file to: {target_config_location}'
            )
    else:
        log.info(
            'Could not place the accelerate configuration file. Please configure manually with: accelerate config.'
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
    else:
        log.info('Using CPU-only Torch')

    try:
        import torch

        log.info(f'Torch {torch.__version__}')

        # Check if CUDA is available
        if not torch.cuda.is_available():
            log.warning('Torch reports CUDA not available')
        else:
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
                return int(torch.__version__[0])
    except Exception as e:
        log.error(f'Could not load torch: {e}')
        sys.exit(1)


def cudann_install():
    cudnn_src = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..\cudnn_windows'
    )
    cudnn_dest = os.path.join(sysconfig.get_paths()['purelib'], 'torch', 'lib')

    log.info(f'Checking for CUDNN files in {cudnn_dest}...')
    if os.path.exists(cudnn_src):
        if os.path.exists(cudnn_dest):
            # check for different files
            filecmp.clear_cache()
            for file in os.listdir(cudnn_src):
                src_file = os.path.join(cudnn_src, file)
                dest_file = os.path.join(cudnn_dest, file)
                # if dest file exists, check if it's different
                if os.path.exists(dest_file):
                    shutil.copy2(src_file, cudnn_dest)
            log.info('Copied CUDNN 8.6 files to destination')
    else:
        log.error(f'Installation Failed: "{cudnn_src}" could not be found. ')


def pip(
    #
    # This function was adapted from code written by vladimandic: https://github.com/vladmandic/automatic/commits/master
    #
    arg: str,
    ignore: bool = False,
    quiet: bool = False,
    reinstall: bool = False,
):
    arg = arg.replace('>=', '==')
    uninstall = arg.startswith(
        'uninstall'
    )  # Check if the argument is for uninstalling

    if not quiet:
        package_name = (
            arg.replace('uninstall', '')
            .replace('--ignore-installed', '')
            .replace('--no-cache-dir', '')
            .replace('--upgrade', '')
            .replace('--no-deps', '')
            .replace('--force', '')
            .replace('-I', '')
            .replace('-U', '')
            .replace('  ', ' ')
            .replace('install', '')
            .replace('-y', '')
            .strip()
        )

        if uninstall:
            log.info(f'Uninstalling package: {package_name}')
        else:
            action = 'Reinstalling' if reinstall else 'Installing'
            log.info(f'{action} package: {package_name}')

    log.debug(f'Running pip: {arg}')
    result = subprocess.run(
        f'"{sys.executable}" -m pip {arg}',
        shell=True,
        check=False,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    txt = result.stdout.decode(encoding='utf8', errors='ignore')
    if len(result.stderr) > 0:
        txt += ('\n' if len(txt) > 0 else '') + result.stderr.decode(
            encoding='utf8', errors='ignore'
        )
    txt = txt.strip()
    if result.returncode != 0 and not ignore:
        global errors  # pylint: disable=global-statement
        errors += 1
        if uninstall:
            log.error(f'Error running pip uninstall: {arg}')
        else:
            log.error(f'Error running pip install: {arg}')
        log.debug(f'Pip output: {txt}')
    return txt


def installed(package, friendly: str = None):
    #
    # This function was adapted from code written by vladimandic: https://github.com/vladmandic/automatic/commits/master
    #

    ok = True
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
            ]   # get only package name if installing from url
        for pkg in pkgs:
            if '>=' in pkg:
                p = pkg.split('>=')
            elif '==' in pkg:
                p = pkg.split('==')
            else:
                p = [pkg]
            spec = pkg_resources.working_set.by_key.get(
                p[0], None
            )   # more reliable than importlib
            if spec is None:
                spec = pkg_resources.working_set.by_key.get(
                    p[0].lower(), None
                )   # check name variations
            if spec is None:
                spec = pkg_resources.working_set.by_key.get(
                    p[0].replace('_', '-'), None
                )   # check name variations
            ok = ok and spec is not None
            if ok:
                version = pkg_resources.get_distribution(p[0]).version
                log.debug(f'Package version found: {p[0]} {version}')
                if len(p) > 1:
                    ok = ok and version == p[1]
                    if not ok:
                        log.warning(
                            f'Package wrong version: {p[0]} {version} required {p[1]}'
                        )
            else:
                log.debug(f'Package version not found: {p[0]}')
        return ok
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
):
    if reinstall:
        global quick_allowed   # pylint: disable=global-statement
        quick_allowed = False
    if reinstall or not installed(package, friendly):
        pip(f'install --upgrade {package}', ignore=ignore, reinstall=reinstall)


# uninstall package using pip
def uninstall(package, ignore: bool = False):
    pip(f'{package}', ignore=ignore)


def ensure_base_requirements():
    try:
        import rich   # pylint: disable=unused-import
    except ImportError:
        install('rich', 'rich')


def run_cmd(run_cmd):
    try:
        subprocess.run(run_cmd, check=True)
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


def install_requirements(requirements_file):
    #
    # This function was adapted from code written by vladimandic: https://github.com/vladmandic/automatic/commits/master
    #

    log.info('Verifying requirements')
    with open(requirements_file, 'r', encoding='utf8') as f:
        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip() != ''
            and not line.startswith('#')
            and line is not None
        ]
        for line in lines:
            install(line)


def write_to_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
    except IOError as e:
        print(f'Error occurred while writing to file: {file_path}')
        print(f'Error: {e}')


def sync_bits_and_bytes_files():
    import filecmp

    """
    Check for "different" bitsandbytes Files and copy only if necessary.
    This function is specific for Windows OS.
    """

    # Only execute on Windows
    if os.name != 'nt':
        print('This function is only applicable to Windows OS.')
        return

    try:
        log.info(f'Copying bitsandbytes files...')
        # Define source and destination directories
        source_dir = os.path.join(os.getcwd(), 'bitsandbytes_windows')

        dest_dir_base = os.path.join(
            sysconfig.get_paths()['purelib'], 'bitsandbytes'
        )

        # Clear file comparison cache
        filecmp.clear_cache()

        # Iterate over each file in source directory
        for file in os.listdir(source_dir):
            source_file_path = os.path.join(source_dir, file)

            # Decide the destination directory based on file name
            if file in ('main.py', 'paths.py'):
                dest_dir = os.path.join(dest_dir_base, 'cuda_setup')
            else:
                dest_dir = dest_dir_base

            dest_file_path = os.path.join(dest_dir, file)

            # Compare the source file with the destination file
            if os.path.exists(dest_file_path) and filecmp.cmp(
                source_file_path, dest_file_path
            ):
                log.debug(
                    f'Skipping {source_file_path} as it already exists in {dest_dir}'
                )
            else:
                # Copy file from source to destination, maintaining original file's metadata
                log.debug(f'Copy {source_file_path} to {dest_dir}')
                shutil.copy2(source_file_path, dest_dir)

    except FileNotFoundError as fnf_error:
        log.error(f'File not found error: {fnf_error}')
    except PermissionError as perm_error:
        log.error(f'Permission error: {perm_error}')
    except Exception as e:
        log.error(f'An unexpected error occurred: {e}')


def cleanup_venv():
    log.info(f'Cleaning up all modules from the venv...')
    subprocess.run(
        f'"{sys.executable}" -m pip freeze > uninstall.txt',
        shell=True,
        check=False,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with open('uninstall.txt', 'r', encoding='utf8') as f:
        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip() != ''
            and not line.startswith('#')
            and line is not None
        ]
        for line in lines:
            log.info(f'Uninstalling: {line}')
            subprocess.run(
                f'"{sys.executable}" -m pip uninstall -y --no-cache-dir {line}',
                shell=True,
                check=False,
                env=os.environ,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )


def install_kohya_ss_torch1():
    check_python()

    # Upgrade pip if needed
    install('--upgrade pip')

    reinstall = False

    if check_torch() != 1:
        uninstall(
            f'uninstall -y --no-cache-dir xformers torchvision torch tensorflow  tensorflow-estimator tensorflow-intel tensorflow-io-gcs-filesystem triton'
        )
        reinstall = True

    install(
        'torch==1.12.1+cu116 torchvision==0.13.1+cu116 --index-url https://download.pytorch.org/whl/cu116',
        'torch torchvision',
        reinstall=reinstall,
    )
    install(
        'https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl -U -I --no-deps',
        'xformers-0.0.14',
        reinstall=reinstall,
    )
    install_requirements('requirements_windows_torch1.txt')
    sync_bits_and_bytes_files()
    configure_accelerate()
    # run_cmd(f'accelerate config')


def install_kohya_ss_torch2():
    check_python()

    # Upgrade pip if needed
    install('--upgrade pip')

    reinstall = False

    if check_torch() != 2:
        uninstall(
            f'uninstall -y --no-cache-dir xformers torchvision torch tensorflow tensorflow-estimator tensorflow-intel tensorflow-io-gcs-filesystem'
        )
        reinstall = True

    install(
        'torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118',
        'torch torchvision',
        reinstall=reinstall,
    )
    install_requirements('requirements_windows_torch2.txt')
    # install('https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl', 'triton', reinstall=reinstall)
    sync_bits_and_bytes_files()
    configure_accelerate()
    # run_cmd(f'accelerate config')


def clear_screen():
    # Check the current operating system to execute the correct clear screen command
    if os.name == 'nt':  # If the operating system is Windows
        os.system('cls')
    else:  # If the operating system is Linux or Mac
        os.system('clear')


def main_menu():
    clear_screen()
    while True:
        print('\nKohya_ss GUI setup menu:\n')
        print('0. Cleanup the venv')
        print('1. Install kohya_ss gui')
        print('2. Install cudann files')
        print('3. Start Kohya_ss GUI in browser')
        print('4. Quit')

        choice = input('\nEnter your choice: ')
        print('')

        if choice == '0':
            confirmation = input(
                f'{YELLOW}Are you sure you want to delete all Python modules installed in the current venv? (y/n): \033[0m'
            )
            if confirmation.lower() == 'y':
                cleanup_venv()
            else:
                print('Cleanup canceled.')
        elif choice == '1':
            while True:
                print('1. Torch 1')
                print('2. Torch 2')
                print('3. Cancel')
                choice_torch = input('\nEnter your choice: ')
                print('')

                if choice_torch == 1:
                    install_kohya_ss_torch1()
                    break
                elif choice_torch == '2':
                    install_kohya_ss_torch2()
                    break
                elif choice_torch == '3':
                    break
                else:
                    print('Invalid choice. Please enter a number between 1-3.')
        elif choice == '2':
            cudann_install()
        elif choice == '3':
            subprocess.Popen('start cmd /c .\gui.bat --inbrowser', shell=True)
        elif choice == '4':
            print('Quitting the program.')
            break
        else:
            print('Invalid choice. Please enter a number between 0-3.')


if __name__ == '__main__':
    ensure_base_requirements()
    setup_logging()
    main_menu()
