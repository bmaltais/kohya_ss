import platform
import subprocess
import os

# Get system information
system = platform.system()
release = platform.release()
version = platform.version()
machine = platform.machine()
processor = platform.processor()

# Print system information
print("System Information:")
print(f"System: {system}, Release: {release}, Version: {version}, Machine: {machine}, Processor: {processor}")

# Get Python information
python_version = platform.python_version()
python_implementation = platform.python_implementation()
python_compiler = platform.python_compiler()

# Print Python information
print("\nPython Information:")
print(f"Version: {python_version}, Implementation: {python_implementation}, Compiler: {python_compiler}")

# Get virtual environment information
venv = os.environ.get('VIRTUAL_ENV', None)

# Print virtual environment information
if venv:
    print("\nVirtual Environment Information:")
    print(f"Path: {venv}")
else:
    print("\nVirtual Environment Information:")
    print("Not running inside a virtual environment.")

# Get GPU information (requires nvidia-smi to be installed)
try:
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'])
    output = output.decode('utf-8').strip().split('\n')[1:]
    gpu_info = [line.split(', ') for line in output]
    gpu_name, gpu_vram = gpu_info[0]
    gpu_vram = gpu_vram.replace(' MiB', '')
    gpu_vram_warning = int(gpu_vram) < 8000
except (subprocess.CalledProcessError, FileNotFoundError):
    gpu_name, gpu_vram = "N/A", "N/A"
    gpu_vram_warning = False

# Print GPU information
print("\nGPU Information:")
print(f"Name: {gpu_name}, VRAM: {gpu_vram} MiB")

# Print VRAM warning if necessary
if gpu_vram_warning:
    print('\033[33mWarning: GPU VRAM is less than 8GB and will likely result in proper operations.\033[0m')

print(' ')
