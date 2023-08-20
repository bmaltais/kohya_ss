import os, sys
sys.path.insert(0, os.getcwd())
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_model", help="The model you want to merge with loha",
        default='', type=str
    )
    parser.add_argument(
        "lycoris_model", help="the lyco model you want to merge into sd model",
        default='', type=str
    )
    parser.add_argument(
        "output_name", help="the output model",
        default='./out.pt', type=str
    )
    parser.add_argument(
        "--is_v2", help="Your base model is sd v2 or not",
        default=False, action="store_true"
    )
    parser.add_argument(
        "--device", help="Which device you want to use to merge the weight",
        default='cpu', type=str
    )
    parser.add_argument(
        "--dtype", help='dtype to save',
        default='float', type=str
    )
    parser.add_argument(
        "--weight", help='weight for the lyco model to merge',
        default='1.0', type=float
    )
    return parser.parse_args()
ARGS = get_args()


from lycoris.utils import merge
from lycoris.kohya.model_utils import (
    load_models_from_stable_diffusion_checkpoint,
    save_stable_diffusion_checkpoint,
    load_file
)

import torch


def main():
    base = load_models_from_stable_diffusion_checkpoint(ARGS.is_v2, ARGS.base_model)
    if ARGS.lycoris_model.rsplit('.', 1)[-1] == 'safetensors':
        lyco = load_file(ARGS.lycoris_model)
    else:
        lyco = torch.load(ARGS.lycoris_model)
    
    dtype_str = ARGS.dtype.replace('fp', 'float').replace('bf', 'bfloat')
    dtype = {
        'float': torch.float,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'bfloat': torch.bfloat16,
        'bfloat16': torch.bfloat16,
    }.get(dtype_str, None)
    if dtype is None:
        raise ValueError(f'Cannot Find the dtype "{dtype}"')
    
    merge(
        base,
        lyco,
        ARGS.weight,
        ARGS.device
    )
    
    save_stable_diffusion_checkpoint(
        ARGS.is_v2, ARGS.output_name, 
        base[0], base[2], 
        None, 0, 0, dtype, 
        base[1]
    )


if __name__ == '__main__':
    main()