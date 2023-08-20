import os
import sysconfig
import filecmp
import shutil

def sync_bits_and_bytes_files():
    """
    Check for "different" bitsandbytes Files and copy only if necessary.
    This function is specific for Windows OS.
    """
    
    # Only execute on Windows
    if os.name != "nt":
        print("This function is only applicable to Windows OS.")
        return

    try:
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

            # Copy file from source to destination, maintaining original file's metadata
            print(f'Copy {source_file_path} to {dest_dir}')
            shutil.copy2(source_file_path, dest_dir)

    except FileNotFoundError as fnf_error:
        print(f"File not found error: {fnf_error}")
    except PermissionError as perm_error:
        print(f"Permission error: {perm_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    sync_bits_and_bytes_files()