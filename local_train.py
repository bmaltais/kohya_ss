import os
import subprocess

def run_command(command):
    """Runs a shell command and prints its output."""
    print(f"Running command: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Command failed with return code {process.returncode}: {command}")

def main():
    print("Starting local training setup...")

    # Target directory for cloning and operations, can be made configurable
    base_dir = "kohya_local_setup"
    os.makedirs(base_dir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(base_dir)

    print(f"Working directory: {os.getcwd()}")

    # 1. Install dependencies (This should ideally be handled by gui-uv.sh using requirements.txt)
    # For now, we'll list them. These were in the notebook:
    # dadaptation==3.1 diffusers[torch]==0.17.1 easygui==0.98.3 einops==0.6.0
    # fairscale==0.4.13 ftfy==6.1.1 gradio==3.36.1 huggingface-hub==0.14.1
    # lion-pytorch==0.0.6 lycoris_lora==1.8.0.dev6 open-clip-torch==2.20.0
    # prodigyopt==1.0 pytorch-lightning==1.9.0 safetensors==0.3.1 timm==0.6.12
    # tk==0.1.0 transformers==4.30.2 voluptuous==0.13.1 wandb==0.15.0
    # xformers==0.0.20 omegaconf
    print("Ensure all dependencies are installed via requirements.txt by gui-uv.sh")

    # 2. Clone bitsandbytes
    if not os.path.exists("bitsandbytes"):
        run_command("git clone -b 0.41.0 https://github.com/TimDettmers/bitsandbytes")
    else:
        print("bitsandbytes directory already exists. Skipping clone.")

    # 3. Build and install bitsandbytes
    # Note: This might need specific CUDA versions. The notebook used CUDA_VERSION=118
    # This part is tricky for a generic local setup and might need user intervention or
    # pre-compiled versions if not handled by the main gui-uv.sh environment.
    # For now, we are replicating the notebook's steps.
    # It's better if bitsandbytes is installed as a pip package if possible.
    print("Building and installing bitsandbytes...")
    os.chdir("bitsandbytes")
    run_command("make cuda11x") # This assumes CUDA 11.8 development toolkit is available
    run_command("python setup.py install")
    os.chdir("..") # Back to base_dir

    # 4. Clone kohya_ss
    if not os.path.exists("kohya_ss"):
        run_command("git clone -b v1.0 https://github.com/camenduru/kohya_ss")
    else:
        print("kohya_ss directory already exists. Skipping clone.")
        # Optionally, add a git pull here to update
        # os.chdir("kohya_ss")
        # run_command("git pull")
        # os.chdir("..")


    # 5. Launch Kohya GUI
    print("Launching Kohya GUI...")
    os.chdir("kohya_ss")
    # The notebook uses --share --headless.
    # --share might not be needed or desired for local use.
    # --headless might be for Colab's environment. For local GUI, we might not want headless.
    # We need to check how gui-uv.sh expects to launch this.
    # For now, using the notebook's command but this will likely need adjustment.
    try:
        run_command("python kohya_gui.py --headless")
    except Exception as e:
        print(f"Error launching Kohya GUI: {e}")
        print("Please ensure all dependencies, including CUDA and xformers, are correctly installed.")
    finally:
        os.chdir(original_dir) # Change back to the original directory

if __name__ == "__main__":
    main()
