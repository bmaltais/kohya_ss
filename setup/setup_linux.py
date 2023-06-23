import argparse
import logging
import setup_common

errors = 0  # Define the 'errors' variable before using it
log = logging.getLogger('sd')

# ANSI escape code for yellow color
YELLOW = '\033[93m'
RESET_COLOR = '\033[0m'


def install_kohya_ss(platform_requirements_file):
    setup_common.check_repo_version()
    setup_common.check_python()

    # Upgrade pip if needed
    setup_common.install('--upgrade pip')
    setup_common.install_requirements(platform_requirements_file, check_no_verify_flag=False)
    setup_common.configure_accelerate(run_accelerate=False)
    # run_cmd(f'accelerate config')


def main_menu(platform_requirements_file):
    log.info("Installing python dependencies. This could take a few minutes as it downloads files.")
    log.info("If this operation ever runs too long, you can rerun this script in verbose mode to check.")
    install_kohya_ss(platform_requirements_file)


if __name__ == '__main__':
    setup_common.ensure_base_requirements()
    setup_common.setup_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform-requirements-file', dest='platform_requirements_file', default='requirements_linux.txt', help='Path to the platform-specific requirements file')
    args = parser.parse_args()
    
    main_menu(args.platform_requirements_file)
