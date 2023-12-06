import subprocess
import os
import filecmp
import logging
import shutil
import sysconfig
import setup_common

errors = 0  # Define the 'errors' variable before using it
log = logging.getLogger("sd")

# ANSI escape code for yellow color
YELLOW = "\033[93m"
RESET_COLOR = "\033[0m"


def cudann_install():
    cudnn_src = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..\cudnn_windows"
    )
    cudnn_dest = os.path.join(sysconfig.get_paths()["purelib"], "torch", "lib")

    log.info(f"Checking for CUDNN files in {cudnn_dest}...")
    if os.path.exists(cudnn_src):
        if os.path.exists(cudnn_dest):
            # check for different files
            filecmp.clear_cache()
            for file in os.listdir(cudnn_src):
                src_file = os.path.join(cudnn_src, file)
                dest_file = os.path.join(cudnn_dest, file)
                # if dest file exists, check if it's different
                if os.path.exists(dest_file):
                    if not filecmp.cmp(src_file, dest_file, shallow=False):
                        shutil.copy2(src_file, cudnn_dest)
                else:
                    shutil.copy2(src_file, cudnn_dest)
            log.info("Copied CUDNN 8.6 files to destination")
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


def install_kohya_ss_torch2():
    setup_common.check_repo_version()
    setup_common.check_python()

    # Upgrade pip if needed
    setup_common.install("--upgrade pip")

    setup_common.install_requirements(
        "requirements_windows_torch2.txt", check_no_verify_flag=False
    )

    sync_bits_and_bytes_files()
    setup_common.configure_accelerate(run_accelerate=True)

    # run_cmd(f'accelerate config')


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


def main_menu():
    setup_common.clear_screen()
    while True:
        print("\nKohya_ss GUI setup menu:\n")
        print("1. Install kohya_ss gui")
        print("2. (Optional) Install cudann files (avoid unless you really need it)")
        print("3. (Optional) Install specific bitsandbytes versions")
        print("4. (Optional) Manually configure accelerate")
        print("5. (Optional) Start Kohya_ss GUI in browser")
        print("6. Quit")

        choice = input("\nEnter your choice: ")
        print("")

        if choice == "1":
            install_kohya_ss_torch2()
        elif choice == "2":
            cudann_install()
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
                    "4. (Danger) Install bitsandbytes-windows (this package has been reported to cause issues for most... avoid...)"
                )
                print("5. Cancel")
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
                    setup_common.install(
                        "--upgrade bitsandbytes-windows", reinstall=True
                    )
                    break
                elif choice_torch == "5":
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
    main_menu()
