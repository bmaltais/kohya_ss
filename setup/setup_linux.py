import argparse
import logging
import setup_common

errors = 0  # Define the 'errors' variable before using it
log = logging.getLogger('sd')

# ANSI escape code for yellow color
YELLOW = '\033[93m'
RESET_COLOR = '\033[0m'


def main_menu(platform_requirements_file, show_stdout: bool = False, no_run_accelerate: bool = False):
    log.info("Installing python dependencies. This could take a few minutes as it downloads files.")
    log.info("If this operation ever runs too long, you can rerun this script in verbose mode to check.")
    
    setup_common.check_repo_version()
    # setup_common.check_python()

    # Upgrade pip if needed
    setup_common.install('pip')
    setup_common.install_requirements(platform_requirements_file, check_no_verify_flag=False, show_stdout=show_stdout)
    if not no_run_accelerate:
        setup_common.configure_accelerate(run_accelerate=False)


if __name__ == '__main__':
    setup_common.ensure_base_requirements()
    setup_common.setup_logging()
    if not setup_common.check_python_version():
        exit(1)
    
    setup_common.update_submodule()
    
    # setup_common.clone_or_checkout(
    #     "https://github.com/kohya-ss/sd-scripts.git", tag_version, "sd-scripts"
    # )

    parser = argparse.ArgumentParser()
    parser.add_argument('--platform-requirements-file', dest='platform_requirements_file', default='requirements_linux.txt', help='Path to the platform-specific requirements file')
    parser.add_argument('--show_stdout', dest='show_stdout', action='store_true', help='Whether to show stdout during installation')
    parser.add_argument('--no_run_accelerate', dest='no_run_accelerate', action='store_true', help='Whether to not run accelerate config')
    args = parser.parse_args()

    main_menu(args.platform_requirements_file, show_stdout=args.show_stdout, no_run_accelerate=args.no_run_accelerate)
