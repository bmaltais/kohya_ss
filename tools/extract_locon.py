import os, sys
sys.path.insert(0, os.getcwd())
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_model", help="The model which use it to train the dreambooth model",
        default='', type=str
    )
    parser.add_argument(
        "db_model", help="the dreambooth model you want to extract the locon",
        default='', type=str
    )
    parser.add_argument(
        "output_name", help="the output model",
        default='./out.pt', type=str
    )
    parser.add_argument(
        "--is_v2", help="Your base/db model is sd v2 or not",
        default=False, action="store_true"
    )
    parser.add_argument(
        "--device", help="Which device you want to use to extract the locon",
        default='cpu', type=str
    )
    parser.add_argument(
        "--mode", 
        help=(
            'extraction mode, can be "fixed", "threshold", "ratio", "quantile". '
            'If not "fixed", network_dim and conv_dim will be ignored'
        ),
        default='fixed', type=str
    )
    parser.add_argument(
        "--safetensors", help='use safetensors to save locon model',
        default=True, action="store_true"
    )
    parser.add_argument(
        "--linear_dim", help="network dim for linear layer in fixed mode",
        default=1, type=int
    )
    parser.add_argument(
        "--conv_dim", help="network dim for conv layer in fixed mode",
        default=1, type=int
    )
    parser.add_argument(
        "--linear_threshold", help="singular value threshold for linear layer in threshold mode",
        default=0., type=float
    )
    parser.add_argument(
        "--conv_threshold", help="singular value threshold for conv layer in threshold mode",
        default=0., type=float
    )
    parser.add_argument(
        "--linear_ratio", help="singular ratio for linear layer in ratio mode",
        default=0., type=float
    )
    parser.add_argument(
        "--conv_ratio", help="singular ratio for conv layer in ratio mode",
        default=0., type=float
    )
    parser.add_argument(
        "--linear_quantile", help="singular value quantile for linear layer quantile mode",
        default=1., type=float
    )
    parser.add_argument(
        "--conv_quantile", help="singular value quantile for conv layer quantile mode",
        default=1., type=float
    )
    parser.add_argument(
        "--use_sparse_bias", help="enable sparse bias",
        default=False, action="store_true"
    )
    parser.add_argument(
        "--sparsity", help="sparsity for sparse bias",
        default=0.98, type=float
    )
    parser.add_argument(
        "--disable_cp", help="don't use cp decomposition",
        default=False, action="store_true"
    )
    return parser.parse_args()
ARGS = get_args()


from lycoris.utils import extract_diff
from lycoris.kohya.model_utils import load_models_from_stable_diffusion_checkpoint

import torch
from safetensors.torch import save_file


def main():
    args = ARGS
    base = load_models_from_stable_diffusion_checkpoint(args.is_v2, args.base_model)
    db = load_models_from_stable_diffusion_checkpoint(args.is_v2, args.db_model)
    
    linear_mode_param = {
        'fixed': args.linear_dim,
        'threshold': args.linear_threshold,
        'ratio': args.linear_ratio,
        'quantile': args.linear_quantile,
    }[args.mode]
    conv_mode_param = {
        'fixed': args.conv_dim,
        'threshold': args.conv_threshold,
        'ratio': args.conv_ratio,
        'quantile': args.conv_quantile,
    }[args.mode]
    
    state_dict = extract_diff(
        base, db,
        args.mode,
        linear_mode_param, conv_mode_param,
        args.device, 
        args.use_sparse_bias, args.sparsity,
        not args.disable_cp
    )
    
    if args.safetensors:
        save_file(state_dict, args.output_name)
    else:
        torch.save(state_dict, args.output_name)


if __name__ == '__main__':
    main()