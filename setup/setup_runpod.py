import argparse
import logging
import setup_common
import os
import shutil

errors = 0  # Define the 'errors' variable before using it
log = logging.getLogger('sd')

# ANSI escape code for yellow color
YELLOW = '\033[93m'
RESET_COLOR = '\033[0m'

def configure_accelerate():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = "/root/.cache/huggingface/accelerate"
    
    log.info("Configuring accelerate...")
    os.makedirs(cache_dir, exist_ok=True)

    config_file_src = os.path.join(script_dir, "config_files", "accelerate", "runpod.yaml")
    config_file_dest = os.path.join(cache_dir, "default_config.yaml")
    shutil.copyfile(config_file_src, config_file_dest)


def setup_environment():
    # Get the directory the script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Install tk and python3.10-venv
    log.info("Install tk and python3.10-venv...")
    subprocess.run(['apt', 'update', '-y'])
    subprocess.run(['apt', 'install', '-y', 'python3-tk', 'python3.10-venv'])

    # Check if the venv folder doesn't exist
    venv_dir = os.path.join(script_dir, 'venv')
    if not os.path.exists(venv_dir):
        log.info("Creating venv...")
        subprocess.run(['python3', '-m', 'venv', venv_dir])

    # Activate the virtual environment
    log.info("Activate venv...")
    activate_script = os.path.join(venv_dir, 'bin', 'activate')
    activate_command = f'source "{activate_script}" || exit 1'
    subprocess.run(activate_command, shell=True, executable='/bin/bash')


def main_menu(platform_requirements_file):
    log.info("Installing python dependencies. This could take a few minutes as it downloads files.")
    log.info("If this operation ever runs too long, you can rerun this script in verbose mode to check.")

    setup_common.check_repo_version()
    # setup_common.check_python()

    # Upgrade pip if needed
    setup_common.install('pip')
    setup_common.install_requirements(platform_requirements_file, check_no_verify_flag=False, show_stdout=True)
    configure_accelerate()


if __name__ == '__main__':
    setup_common.ensure_base_requirements()
    setup_common.setup_logging()
    if not setup_common.check_python_version():
        exit(1)
    
    setup_common.update_submodule()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform-requirements-file', dest='platform_requirements_file', default='requirements_runpod.txt', help='Path to the platform-specific requirements file')
    args = parser.parse_args()
    
    main_menu(args.platform_requirements_file)
