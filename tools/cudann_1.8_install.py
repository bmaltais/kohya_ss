import filecmp
import importlib.util
import os
import shutil
import sys
import sysconfig
import subprocess
from pathlib import Path
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../requirements.txt")

def run(command, desc=None, errdesc=None, custom_env=None):
    if desc is not None:
        print(desc)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")

def check_versions():
    global req_file
    reqs = open(req_file, 'r')
    lines = reqs.readlines()
    reqs_dict = {}
    for line in lines:
        splits = line.split("==")
        if len(splits) == 2:
            key = splits[0]
            if "torch" not in key:
                if "diffusers" in key:
                    key = "diffusers"
                reqs_dict[key] = splits[1].replace("\n", "").strip()
    if os.name == "nt":
        reqs_dict["torch"] = "1.12.1+cu116"
        reqs_dict["torchvision"] = "0.13.1+cu116"

    checks = ["xformers","bitsandbytes", "diffusers", "transformers", "torch", "torchvision"]
    for check in checks:
        check_ver = "N/A"
        status = "[ ]"
        try:
            check_available = importlib.util.find_spec(check) is not None
            if check_available:
                check_ver = importlib_metadata.version(check)
                if check in reqs_dict:
                    req_version = reqs_dict[check]
                    if str(check_ver) == str(req_version):
                        status = "[+]"
                    else:
                        status = "[!]"
        except importlib_metadata.PackageNotFoundError:
            check_available = False
        if not check_available:
            status = "[!]"
            print(f"{status} {check} NOT installed.")
            if check == 'xformers':
                x_cmd = "-U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl"
                print(f"Installing xformers with: pip install {x_cmd}")
                run(f"pip install {x_cmd}", desc="Installing xformers")

        else:
            print(f"{status} {check} version {check_ver} installed.")

base_dir = os.path.dirname(os.path.realpath(__file__))
#repo = git.Repo(base_dir)
#revision = repo.rev_parse("HEAD")
#print(f"Dreambooth revision is {revision}")
check_versions()
# Check for "different" B&B Files and copy only if necessary
if os.name == "nt":
    python = sys.executable
    bnb_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..\bitsandbytes_windows")
    bnb_dest = os.path.join(sysconfig.get_paths()["purelib"], "bitsandbytes")
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

            