import gc

import torch


def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
