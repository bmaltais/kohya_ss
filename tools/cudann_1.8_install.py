import filecmp
import os
import shutil
import sys
import sysconfig

# Check for "different" B&B Files and copy only if necessary
if os.name == "nt":
    python = sys.executable
    cudnn_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..\cudnn_windows")
    cudnn_dest = os.path.join(sysconfig.get_paths()["purelib"], "torch", "lib")
    
    print(f"Checking for CUDNN files in {cudnn_dest}")
    if os.path.exists(cudnn_src):
        if os.path.exists(cudnn_dest):
            # check for different files
            filecmp.clear_cache()
            for file in os.listdir(cudnn_src):
                src_file = os.path.join(cudnn_src, file)
                dest_file = os.path.join(cudnn_dest, file)
                #if dest file exists, check if it's different
                if os.path.exists(dest_file):
                    shutil.copy2(src_file, cudnn_dest)
            print("Copied CUDNN 8.6 files to destination")
    else: 
        print(f"Installation Failed: \"{cudnn_src}\" could not be found. ")
