import subprocess
import os
import filecmp
import logging
import shutil
import sysconfig
import setup_common
import argparse

errors = 0  # Define the 'errors' variable before using it
log = logging.getLogger("sd")

# ANSI escape code for yellow color
YELLOW = "\033[93m"
RESET_COLOR = "\033[0m"


def cudnn_install():
    log.info("Installing nvidia-cudnn-cu11 8.9.5.29...")
    setup_common.install(
        "--upgrade nvidia-cudnn-cu11==8.9.5.29",
        "nvidia-cudnn-cu11 8.9.5.29",
        reinstall=True,
    )
    
    # Original path with "..\\venv"
    original_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..\\venv\\Lib\\site-packages\\nvidia\\cudnn\\bin")
    # Normalize the path to resolve "..\\venv"
    cudnn_src = os.path.abspath(original_path)
    cudnn_dest = os.path.join(sysconfig.get_paths()["purelib"], "torch", "lib")

    log.info(f"Copying CUDNN files from {cudnn_src} to {cudnn_dest}...")
    if os.path.exists(cudnn_src):
        if os.path.exists(cudnn_dest):
            # check for different files
            filecmp.clear_cache()
            for file in os.listdir(cudnn_src):
                if file.lower().endswith('.dll'):  # Check if the file is a .dll file
                    src_file = os.path.join(cudnn_src, file)
                    dest_file = os.path.join(cudnn_dest, file)
                    # if dest file exists, check if it's different
                    if os.path.exists(dest_file):
                        if not filecmp.cmp(src_file, dest_file, shallow=False):
                            shutil.copy2(src_file, cudnn_dest)
                    else:
                        shutil.copy2(src_file, cudnn_dest)
            log.info("Copied CUDNN .dll files to destination")
        else:
            log.warning(f"Destination directory {cudnn_dest} does not exist")
    else:
        log.error(f'Installation Failed: "{cudnn_src}" could not be found.')


def sync_bits_and_bytes_files():
    import filecmp

    """
    Check for "different" bitsandbytes Files and copy only if necessary.
    This function is specific for Windows OS.
    """

    # Only execute on Windows
    if os.name != "nt":
        print("This function is only applicable to Windows OS.")
        return

    try:
        log.info(f"Copying bitsandbytes files...")
        # Define source and destination directories
        source_dir = os.path.join(os.getcwd(), "bitsandbytes_windows")

        dest_dir_base = os.path.join(sysconfig.get_paths()["purelib"], "bitsandbytes")

        # Clear file comparison cache
        filecmp.clear_cache()

        # Iterate over each file in source directory
        for file in os.listdir(source_dir):
            source_file_path = os.path.join(source_dir, file)

            # Decide the destination directory based on file name
            if file in ("main.py", "paths.py"):
                dest_dir = os.path.join(dest_dir_base, "cuda_setup")
            else:
                dest_dir = dest_dir_base

            dest_file_path = os.path.join(dest_dir, file)

            # Compare the source file with the destination file
            if os.path.exists(dest_file_path) and filecmp.cmp(
                source_file_path, dest_file_path
            ):
                log.debug(
                    f"Skipping {source_file_path} as it already exists in {dest_dir}"
                )
            else:
                # Copy file from source to destination, maintaining original file's metadata
                log.debug(f"Copy {source_file_path} to {dest_dir}")
                shutil.copy2(source_file_path, dest_dir)

    except FileNotFoundError as fnf_error:
        log.error(f"File not found error: {fnf_error}")
    except PermissionError as perm_error:
        log.error(f"Permission error: {perm_error}")
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}")


def install_kohya_ss_torch2(headless: bool = False):
    setup_common.check_repo_version()
    if not setup_common.check_python_version():
        exit(1)
        
    setup_common.update_submodule()
    
    setup_common.install("pip")

    setup_common.install_requirements(
        "requirements_windows_torch2.txt", check_no_verify_flag=False
    )
    
    setup_common.configure_accelerate(run_accelerate=not headless) # False if headless is True and vice versa


def install_bitsandbytes_0_35_0():
    log.info("Installing bitsandbytes 0.35.0...")
    setup_common.install(
        "--upgrade bitsandbytes==0.35.0", "bitsandbytes 0.35.0", reinstall=True
    )
    sync_bits_and_bytes_files()


def install_bitsandbytes_0_40_1():
    log.info("Installing bitsandbytes 0.40.1...")
    setup_common.install(
        "--upgrade https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.40.1.post1-py3-none-win_amd64.whl",
        "bitsandbytes 0.40.1",
        reinstall=True,
    )


def install_bitsandbytes_0_41_1():
    log.info("Installing bitsandbytes 0.41.1...")
    setup_common.install(
        "--upgrade https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl",
        "bitsandbytes 0.41.1",
        reinstall=True,
    )

def install_bitsandbytes_0_41_2():
    log.info("Installing bitsandbytes 0.41.2...")
    setup_common.install(
        "--upgrade https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl",
        "bitsandbytes 0.41.2",
        reinstall=True,
    )

def main_menu(headless: bool = False):
    if headless:
        install_kohya_ss_torch2(headless=headless)
    else:
        setup_common.clear_screen()
        while True:
            print("\nKohya_ss GUI setup menu:\n")
            print("1. Install kohya_ss gui")
            print("2. (Optional) Install cudnn files (if you want to use latest supported cudnn version)")
            print("3. (Optional) Install specific bitsandbytes versions")
            print("4. (Optional) Manually configure accelerate")
            print("5. (Optional) Start Kohya_ss GUI in browser")
            print("6. Quit")

            choice = input("\nEnter your choice: ")
            print("")

            if choice == "1":
                install_kohya_ss_torch2()
            elif choice == "2":
                cudnn_install()
            elif choice == "3":
                while True:
                    print("1. (Optional) Force installation of bitsandbytes 0.35.0")
                    print(
                        "2. (Optional) Force installation of bitsandbytes 0.40.1 for new optimizer options support and pre-bugfix results"
                    )
                    print(
                        "3. (Optional) Force installation of bitsandbytes 0.41.1 for new optimizer options support"
                    )
                    print(
                        "4. (Recommended) Force installation of bitsandbytes 0.41.2 for new optimizer options support"
                    )
                    print(
                        "5. (Danger) Install bitsandbytes-windows (this package has been reported to cause issues for most... avoid...)"
                    )
                    print("6. Exit")
                    choice_torch = input("\nEnter your choice: ")
                    print("")

                    if choice_torch == "1":
                        install_bitsandbytes_0_35_0()
                        break
                    elif choice_torch == "2":
                        install_bitsandbytes_0_40_1()
                        break
                    elif choice_torch == "3":
                        install_bitsandbytes_0_41_1()
                        break
                    elif choice_torch == "4":
                        install_bitsandbytes_0_41_2()
                        break
                    elif choice_torch == "5":
                        setup_common.install(
                            "--upgrade bitsandbytes-windows", reinstall=True
                        )
                        break
                    elif choice_torch == "6":
                        break
                    else:
                        print("Invalid choice. Please enter a number between 1-3.")
            elif choice == "4":
                setup_common.run_cmd("accelerate config")
            elif choice == "5":
                subprocess.Popen(
                    "start cmd /k .\gui.bat --inbrowser", shell=True
                )  # /k keep the terminal open on quit. /c would close the terminal instead
            elif choice == "6":
                print("Quitting the program.")
                break
            else:
                print("Invalid choice. Please enter a number between 1-5.")


if __name__ == "__main__":
    setup_common.ensure_base_requirements()
    setup_common.setup_logging()
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Your Script Description")
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')

    # Parse arguments
    args = parser.parse_args()
    
    main_menu(headless=args.headless)
