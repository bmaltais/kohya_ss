import functools
import gc

import torch

try:
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    HAS_MPS = torch.backends.mps.is_available()
except Exception:
    HAS_MPS = False


def clean_memory():
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
    if HAS_MPS:
        torch.mps.empty_cache()


@functools.lru_cache(maxsize=None)
def get_preferred_device() -> torch.device:
    if HAS_CUDA:
        device = torch.device("cuda")
    elif HAS_MPS:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"get_preferred_device() -> {device}")
    return device
