"""
extract factors the build is dependent on:
[X] compute capability
    [ ] TODO: Q - What if we have multiple GPUs of different makes?
- CUDA version
- Software:
    - CPU-only: only CPU quantization functions (no optimizer, no matrix multipl)
    - CuBLAS-LT: full-build 8-bit optimizer
    - no CuBLAS-LT: no 8-bit matrix multiplication (`nomatmul`)

evaluation:
    - if paths faulty, return meaningful error
    - else:
        - determine CUDA version
        - determine capabilities
        - based on that set the default path
"""

import ctypes as ct
import os
import errno
import torch
import platform
from warnings import warn
from itertools import product

from pathlib import Path
from typing import Set, Union
from .env_vars import get_potentially_lib_path_containing_env_vars

# these are the most common libs names
# libcudart.so is missing by default for a conda install with PyTorch 2.0 and instead
# we have libcudart.so.11.0 which causes a lot of errors before
# not sure if libcudart.so.12.0 exists in pytorch installs, but it does not hurt
CUDA_RUNTIME_LIBS: list = ["libcudart.so", 'libcudart.so.11.0', 'libcudart.so.12.0']

# this is a order list of backup paths to search CUDA in, if it cannot be found in the main environmental paths
backup_paths = []


IS_WINDOWS_PLATFORM: bool = (platform.system()=="Windows")
PATH_COLLECTION_SEPARATOR: str = ":" if not IS_WINDOWS_PLATFORM else ";"
CUDA_RUNTIME_LIBS: list = ["libcudart.so", 'libcudart.so.11.0', 'libcudart.so.12.0'] if not IS_WINDOWS_PLATFORM else ["cudart64_110.dll", "cudart64_120.dll", "cudart64_12.dll"]
backup_paths.append('$CONDA_PREFIX/lib/libcudart.so.11.0' if not IS_WINDOWS_PLATFORM else '%CONDA_PREFIX%\\lib\\cudart64_110.dll')
CUDA_SHARED_LIB_NAME: str = "libcuda.so" if not IS_WINDOWS_PLATFORM else f"{os.environ['SystemRoot']}\\System32\\nvcuda.dll"
SHARED_LIB_EXTENSION: str = ".so" if not IS_WINDOWS_PLATFORM else ".dll"
class CUDASetup:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def generate_instructions(self):
        if getattr(self, 'error', False): return
        print(self.error)
        self.error = True
        if self.cuda is None:
            self.add_log_entry('CUDA SETUP: Problem: The main issue seems to be that the main CUDA library was not detected.')
            self.add_log_entry('CUDA SETUP: Solution 1): Your paths are probably not up-to-date. You can update them via: sudo ldconfig.')
            self.add_log_entry('CUDA SETUP: Solution 2): If you do not have sudo rights, you can do the following:')
            self.add_log_entry('CUDA SETUP: Solution 2a): Find the cuda library via: find / -name libcuda.so 2>/dev/null')
            self.add_log_entry('CUDA SETUP: Solution 2b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_2a')
            self.add_log_entry('CUDA SETUP: Solution 2c): For a permanent solution add the export from 2b into your .bashrc file, located at ~/.bashrc')
            return

        if self.cudart_path is None:
            self.add_log_entry('CUDA SETUP: Problem: The main issue seems to be that the main CUDA runtime library was not detected.')
            self.add_log_entry('CUDA SETUP: Solution 1: To solve the issue the libcudart.so location needs to be added to the LD_LIBRARY_PATH variable')
            self.add_log_entry('CUDA SETUP: Solution 1a): Find the cuda runtime library via: find / -name libcudart.so 2>/dev/null')
            self.add_log_entry('CUDA SETUP: Solution 1b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_1a')
            self.add_log_entry('CUDA SETUP: Solution 1c): For a permanent solution add the export from 1b into your .bashrc file, located at ~/.bashrc')
            self.add_log_entry('CUDA SETUP: Solution 2: If no library was found in step 1a) you need to install CUDA.')
            self.add_log_entry('CUDA SETUP: Solution 2a): Download CUDA install script: wget https://github.com/TimDettmers/bitsandbytes/blob/main/cuda_install.sh')
            self.add_log_entry('CUDA SETUP: Solution 2b): Install desired CUDA version to desired location. The syntax is bash cuda_install.sh CUDA_VERSION PATH_TO_INSTALL_INTO.')
            self.add_log_entry('CUDA SETUP: Solution 2b): For example, "bash cuda_install.sh 113 ~/local/" will download CUDA 11.3 and install into the folder ~/local')
            return

        make_cmd = f'CUDA_VERSION={self.cuda_version_string}'
        if len(self.cuda_version_string) < 3:
            make_cmd += ' make cuda92'
        elif self.cuda_version_string == '110':
            make_cmd += ' make cuda110'
        elif self.cuda_version_string[:2] == '11' and int(self.cuda_version_string[2]) > 0:
            make_cmd += ' make cuda11x'
        elif self.cuda_version_string == '100':
            self.add_log_entry('CUDA SETUP: CUDA 10.0 not supported. Please use a different CUDA version.')
            self.add_log_entry('CUDA SETUP: Before you try again running bitsandbytes, make sure old CUDA 10.0 versions are uninstalled and removed from $LD_LIBRARY_PATH variables.')
            return


        has_cublaslt = is_cublasLt_compatible(self.cc)
        if not has_cublaslt:
            make_cmd += '_nomatmul'

        self.add_log_entry('CUDA SETUP: Something unexpected happened. Please compile from source:')
        self.add_log_entry('git clone git@github.com:TimDettmers/bitsandbytes.git')
        self.add_log_entry('cd bitsandbytes')
        self.add_log_entry(make_cmd)
        self.add_log_entry('python setup.py install')

    def initialize(self):
        if not getattr(self, 'initialized', False):
            self.has_printed = False
            self.lib = None
            self.initialized = False
            self.error = False

    def run_cuda_setup(self):
        self.initialized = True
        self.cuda_setup_log = []

        binary_name, cudart_path, cuda, cc, cuda_version_string = evaluate_cuda_setup()
        self.cudart_path = cudart_path
        self.cuda = cuda
        self.cc = cc
        self.cuda_version_string = cuda_version_string

        package_dir = Path(__file__).parent.parent
        binary_path = package_dir / binary_name

        print('bin', binary_path)

        try:
            if not binary_path.exists():
                self.add_log_entry(f"CUDA SETUP: Required library version not found: {binary_name}. Maybe you need to compile it from source?")
                legacy_binary_name = "libbitsandbytes_cpu" + SHARED_LIB_EXTENSION
                self.add_log_entry(f"CUDA SETUP: Defaulting to {legacy_binary_name}...")
                binary_path = package_dir / legacy_binary_name
                if not binary_path.exists() or torch.cuda.is_available():
                    self.add_log_entry('')
                    self.add_log_entry('='*48 + 'ERROR' + '='*37)
                    self.add_log_entry('CUDA SETUP: CUDA detection failed! Possible reasons:')
                    self.add_log_entry('1. CUDA driver not installed')
                    self.add_log_entry('2. CUDA not installed')
                    self.add_log_entry('3. You have multiple conflicting CUDA libraries')
                    self.add_log_entry('4. Required library not pre-compiled for this bitsandbytes release!')
                    self.add_log_entry('CUDA SETUP: If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION` for example, `make CUDA_VERSION=113`.')
                    self.add_log_entry('CUDA SETUP: The CUDA version for the compile might depend on your conda install. Inspect CUDA version via `conda list | grep cuda`.')
                    self.add_log_entry('='*80)
                    self.add_log_entry('')
                    self.generate_instructions()
                    raise Exception('CUDA SETUP: Setup Failed!')
                self.lib = ct.cdll.LoadLibrary(str(binary_path))
            else:
                self.add_log_entry(f"CUDA SETUP: Loading binary {binary_path}...")
                self.lib = ct.cdll.LoadLibrary(str(binary_path))
        except Exception as ex:
            self.add_log_entry(str(ex))

    def add_log_entry(self, msg, is_warning=False):
        self.cuda_setup_log.append((msg, is_warning))

    def print_log_stack(self):
        for msg, is_warning in self.cuda_setup_log:
            if is_warning:
                warn(msg)
            else:
                print(msg)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


def is_cublasLt_compatible(cc):
    has_cublaslt = False
    if cc is not None:
        cc_major, cc_minor = cc.split('.')
        if int(cc_major) < 7 or (int(cc_major) == 7 and int(cc_minor) < 5):
            CUDASetup.get_instance().add_log_entry("WARNING: Compute capability < 7.5 detected! Only slow 8-bit matmul is supported for your GPU!", is_warning=True)
        else:
            has_cublaslt = True
    return has_cublaslt

def extract_candidate_paths(paths_list_candidate: str) -> Set[Path]:
    return {Path(ld_path) for ld_path in paths_list_candidate.split(PATH_COLLECTION_SEPARATOR) if ld_path}


def remove_non_existent_dirs(candidate_paths: Set[Path]) -> Set[Path]:
    existent_directories: Set[Path] = set()
    for path in candidate_paths:
        try:
            if path.exists():
                existent_directories.add(path)
        except OSError as exc:
            if exc.errno != errno.ENAMETOOLONG:
                raise exc

    non_existent_directories: Set[Path] = candidate_paths - existent_directories
    if non_existent_directories:
        CUDASetup.get_instance().add_log_entry("WARNING: The following directories listed in your path were found to "
            f"be non-existent: {non_existent_directories}", is_warning=True)

    return existent_directories


def get_cuda_runtime_lib_paths(candidate_paths: Set[Path]) -> Set[Path]:
    paths = set()
    for libname in CUDA_RUNTIME_LIBS:
        for path in candidate_paths:
            if (path / libname).is_file():
                paths.add(path / libname)
    return paths


def resolve_paths_list(paths_list_candidate: str) -> Set[Path]:
    """
    Searches a given environmental var for the CUDA runtime library,
    i.e. `libcudart.so`.
    """
    return remove_non_existent_dirs(extract_candidate_paths(paths_list_candidate))


def find_cuda_lib_in(paths_list_candidate: str) -> Set[Path]:
    return get_cuda_runtime_lib_paths(
        resolve_paths_list(paths_list_candidate)
    )


def warn_in_case_of_duplicates(results_paths: Set[Path]) -> None:
    if len(results_paths) > 1:
        warning_msg = (
            f"Found duplicate {CUDA_RUNTIME_LIBS} files: {results_paths}.. "
            "We'll flip a coin and try one of these, in order to fail forward.\n"
            "Either way, this might cause trouble in the future:\n"
            "If you get `CUDA error: invalid device function` errors, the above "
            "might be the cause and the solution is to make sure only one "
            f"{CUDA_RUNTIME_LIBS} in the paths that we search based on your env.")
        CUDASetup.get_instance().add_log_entry(warning_msg, is_warning=True)


def determine_cuda_runtime_lib_path() -> Union[Path, None]:
    """
        Searches for a cuda installations, in the following order of priority:
            1. active conda env
            2. LD_LIBRARY_PATH
            3. any other env vars, while ignoring those that
                - are known to be unrelated (see `bnb.cuda_setup.env_vars.to_be_ignored`)
                - don't contain the path separator `/`

        If multiple libraries are found in part 3, we optimistically try one,
        while giving a warning message.
    """
    candidate_env_vars = get_potentially_lib_path_containing_env_vars()

    if "CONDA_PREFIX" in candidate_env_vars:
        conda_libs_path = Path(candidate_env_vars["CONDA_PREFIX"]) / "bin"

        conda_cuda_libs = find_cuda_lib_in(str(conda_libs_path))
        warn_in_case_of_duplicates(conda_cuda_libs)

        if conda_cuda_libs:
            return next(iter(conda_cuda_libs))
        
        conda_libs_path = Path(candidate_env_vars["CONDA_PREFIX"]) / "lib"

        conda_cuda_libs = find_cuda_lib_in(str(conda_libs_path))
        warn_in_case_of_duplicates(conda_cuda_libs)

        if conda_cuda_libs:
            return next(iter(conda_cuda_libs))
        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["CONDA_PREFIX"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)
        
    if "CUDA_PATH" in candidate_env_vars:
        ld_cuda_libs_path = Path(candidate_env_vars["CUDA_PATH"]) / "bin"
    
        lib_ld_cuda_libs = find_cuda_lib_in(str(ld_cuda_libs_path))
        warn_in_case_of_duplicates(lib_ld_cuda_libs)

        if lib_ld_cuda_libs:
            return next(iter(lib_ld_cuda_libs))

        ld_cuda_libs_path = Path(candidate_env_vars["CUDA_PATH"]) / "lib"
    
        lib_ld_cuda_libs = find_cuda_lib_in(str(ld_cuda_libs_path))
        warn_in_case_of_duplicates(lib_ld_cuda_libs)

        if lib_ld_cuda_libs:
            return next(iter(lib_ld_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["CUDA_PATH"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)
        
    if "CUDA_HOME" in candidate_env_vars:
        ld_cuda_libs_path = Path(candidate_env_vars["CUDA_HOME"]) / "bin"
    
        lib_ld_cuda_libs = find_cuda_lib_in(str(ld_cuda_libs_path))
        warn_in_case_of_duplicates(lib_ld_cuda_libs)

        if lib_ld_cuda_libs:
            return next(iter(lib_ld_cuda_libs))
        
        ld_cuda_libs_path = Path(candidate_env_vars["CUDA_HOME"]) / "lib"
    
        lib_ld_cuda_libs = find_cuda_lib_in(str(ld_cuda_libs_path))
        warn_in_case_of_duplicates(lib_ld_cuda_libs)

        if lib_ld_cuda_libs:
            return next(iter(lib_ld_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["CUDA_HOME"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)

    if "LD_LIBRARY_PATH" in candidate_env_vars:
        lib_ld_cuda_libs = find_cuda_lib_in(candidate_env_vars["LD_LIBRARY_PATH"])
        warn_in_case_of_duplicates(lib_ld_cuda_libs)

        if lib_ld_cuda_libs:
            return next(iter(lib_ld_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["LD_LIBRARY_PATH"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)
        
    if "PATH" in candidate_env_vars:
        lib_ld_cuda_libs = find_cuda_lib_in(candidate_env_vars["PATH"])
        warn_in_case_of_duplicates(lib_ld_cuda_libs)

        if lib_ld_cuda_libs:
            return next(iter(lib_ld_cuda_libs))

        CUDASetup.get_instance().add_log_entry(f'{candidate_env_vars["PATH"]} did not contain '
            f'{CUDA_RUNTIME_LIBS} as expected! Searching further paths...', is_warning=True)
        
    remaining_candidate_env_vars = {
        env_var: value for env_var, value in candidate_env_vars.items()
        if env_var not in {"CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "PATH"}
    }

    cuda_runtime_libs = set()
    for env_var, value in remaining_candidate_env_vars.items():
        cuda_runtime_libs.update(find_cuda_lib_in(value))

    if len(cuda_runtime_libs) == 0:
        CUDASetup.get_instance().add_log_entry('CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching in backup paths...')
        cuda_runtime_libs.update(find_cuda_lib_in('/usr/local/cuda/lib64'))

    warn_in_case_of_duplicates(cuda_runtime_libs)

    return next(iter(cuda_runtime_libs)) if cuda_runtime_libs else None


def check_cuda_result(cuda, result_val):
    # 3. Check for CUDA errors
    if result_val != 0:
        error_str = ct.c_char_p()
        cuda.cuGetErrorString(result_val, ct.byref(error_str))
        if error_str.value is not None:
            CUDASetup.get_instance().add_log_entry(f"CUDA exception! Error code: {error_str.value.decode()}")
        else:
            CUDASetup.get_instance().add_log_entry(f"Unknown CUDA exception! Please check your CUDA install. It might also be that your GPU is too old.")


# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION
def get_cuda_version(cuda, cudart_path):
    if cuda is None: return None

    try:
        cudart = ct.CDLL(str(cudart_path))
    except OSError:
        CUDASetup.get_instance().add_log_entry(f'ERROR: libcudart.so could not be read from path: {cudart_path}!')
        return None

    version = ct.c_int()
    try:
        check_cuda_result(cuda, cudart.cudaRuntimeGetVersion(ct.byref(version)))
    except AttributeError as e:
        CUDASetup.get_instance().add_log_entry(f'ERROR: {str(e)}')
        CUDASetup.get_instance().add_log_entry(f'CUDA SETUP: libcudart.so path is {cudart_path}')
        CUDASetup.get_instance().add_log_entry(f'CUDA SETUP: Is seems that your cuda installation is not in your path. See https://github.com/TimDettmers/bitsandbytes/issues/85 for more information.')
    version = int(version.value)
    major = version//1000
    minor = (version-(major*1000))//10

    if major < 11:
        CUDASetup.get_instance().add_log_entry('CUDA SETUP: CUDA version lower than 11 are currently not supported for LLM.int8(). You will be only to use 8-bit optimizers and quantization routines!!')

    return f'{major}{minor}'


def get_cuda_lib_handle():
    # 1. find libcuda.so library (GPU driver) (/usr/lib)
    try:
        cuda = ct.CDLL(CUDA_SHARED_LIB_NAME)
    except OSError:
        CUDASetup.get_instance().add_log_entry('CUDA SETUP: WARNING! libcuda.so not found! Do you have a CUDA driver installed? If you are on a cluster, make sure you are on a CUDA machine!')
        return None
    check_cuda_result(cuda, cuda.cuInit(0))

    return cuda


def get_compute_capabilities(cuda):
    """
    1. find libcuda.so library (GPU driver) (/usr/lib)
       init_device -> init variables -> call function by reference
    2. call extern C function to determine CC
       (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html)
    3. Check for CUDA errors
       https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
    # bits taken from https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
    """

    nGpus = ct.c_int()
    cc_major = ct.c_int()
    cc_minor = ct.c_int()

    device = ct.c_int()

    check_cuda_result(cuda, cuda.cuDeviceGetCount(ct.byref(nGpus)))
    ccs = []
    for i in range(nGpus.value):
        check_cuda_result(cuda, cuda.cuDeviceGet(ct.byref(device), i))
        ref_major = ct.byref(cc_major)
        ref_minor = ct.byref(cc_minor)
        # 2. call extern C function to determine CC
        check_cuda_result(cuda, cuda.cuDeviceComputeCapability(ref_major, ref_minor, device))
        ccs.append(f"{cc_major.value}.{cc_minor.value}")

    return ccs


# def get_compute_capability()-> Union[List[str, ...], None]: # FIXME: error
def get_compute_capability(cuda):
    """
    Extracts the highest compute capbility from all available GPUs, as compute
    capabilities are downwards compatible. If no GPUs are detected, it returns
    None.
    """
    if cuda is None: return None

    # TODO: handle different compute capabilities; for now, take the max
    ccs = get_compute_capabilities(cuda)
    if ccs: return ccs[-1]


def evaluate_cuda_setup():
    if 'BITSANDBYTES_NOWELCOME' not in os.environ or str(os.environ['BITSANDBYTES_NOWELCOME']) == '0':
        print('')
        print('='*35 + 'BUG REPORT' + '='*35)
        print(('Welcome to bitsandbytes. For bug reports, please run\n\npython -m bitsandbytes\n\n'),
              ('and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues'))
        print('='*80)
        return 'libbitsandbytes_cuda118.dll', None, None, None, None
    if not torch.cuda.is_available(): return 'libbitsandbytes_cpu'+SHARED_LIB_EXTENSION, None, None, None, None

    cuda_setup = CUDASetup.get_instance()
    cudart_path = determine_cuda_runtime_lib_path()
    cuda = get_cuda_lib_handle()
    cc = get_compute_capability(cuda)
    cuda_version_string = get_cuda_version(cuda, cudart_path)

    failure = False
    if cudart_path is None:
        failure = True
        cuda_setup.add_log_entry("WARNING: No libcudart.so found! Install CUDA or the cudatoolkit package (anaconda)!", is_warning=True)
    else:
        cuda_setup.add_log_entry(f"CUDA SETUP: CUDA runtime path found: {cudart_path}")

    if cc == '' or cc is None:
        failure = True
        cuda_setup.add_log_entry("WARNING: No GPU detected! Check your CUDA paths. Proceeding to load CPU-only library...", is_warning=True)
    else:
        cuda_setup.add_log_entry(f"CUDA SETUP: Highest compute capability among GPUs detected: {cc}")

    if cuda is None:
        failure = True
    else:
        cuda_setup.add_log_entry(f'CUDA SETUP: Detected CUDA version {cuda_version_string}')

    # 7.5 is the minimum CC vor cublaslt
    has_cublaslt = is_cublasLt_compatible(cc)

    # TODO:
    # (1) CUDA missing cases (no CUDA installed by CUDA driver (nvidia-smi accessible)
    # (2) Multiple CUDA versions installed

    # we use ls -l instead of nvcc to determine the cuda version
    # since most installations will have the libcudart.so installed, but not the compiler

    if failure:
        binary_name = "libbitsandbytes_cpu" + SHARED_LIB_EXTENSION
    elif has_cublaslt:
        binary_name = f"libbitsandbytes_cuda{cuda_version_string}" + SHARED_LIB_EXTENSION
    else:
        "if not has_cublaslt (CC < 7.5), then we have to choose  _nocublaslt"
        binary_name = f"libbitsandbytes_cuda{cuda_version_string}_nocublaslt" +  SHARED_LIB_EXTENSION

    return binary_name, cudart_path, cuda, cc, cuda_version_string
