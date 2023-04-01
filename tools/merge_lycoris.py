import os
import sys
import argparse
import torch
from lycoris.utils import merge_loha, merge_locon
from lycoris.kohya_model_utils import (
    load_models_from_stable_diffusion_checkpoint,
    save_stable_diffusion_checkpoint,
    load_file
)
import gradio as gr


def merge_models(base_model, lycoris_model, output_name, is_v2, device, dtype, weight):
    base = load_models_from_stable_diffusion_checkpoint(is_v2, base_model)
    if lycoris_model.rsplit('.', 1)[-1] == 'safetensors':
        lyco = load_file(lycoris_model)
    else:
        lyco = torch.load(lycoris_model)

    algo = None
    for key in lyco:
        if 'hada' in key:
            algo = 'loha'
            break
        elif 'lora_up' in key:
            algo = 'lora'
            break
    else:
        raise NotImplementedError('Cannot find the algo for this lycoris model file.')

    dtype_str = dtype.replace('fp', 'float').replace('bf', 'bfloat')
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

    if algo == 'loha':
        merge_loha(base, lyco, weight, device)
    elif algo == 'lora':
        merge_locon(base, lyco, weight, device)

    save_stable_diffusion_checkpoint(
        is_v2, output_name,
        base[0], base[2],
        None, 0, 0, dtype,
        base[1]
    )

    return output_name


def main():
    iface = gr.Interface(
        fn=merge_models,
        inputs=[
            gr.inputs.Textbox(label="Base Model Path"),
            gr.inputs.Textbox(label="Lycoris Model Path"),
            gr.inputs.Textbox(label="Output Model Path", default='./out.pt'),
            gr.inputs.Checkbox(label="Is base model SD V2?", default=False),
            gr.inputs.Textbox(label="Device", default='cpu'),
            gr.inputs.Dropdown(choices=['float', 'float16', 'float32', 'float64', 'bfloat', 'bfloat16'], label="Dtype", default='float'),
            gr.inputs.Number(label="Weight", default=1.0)
        ],
        outputs=gr.outputs.Textbox(label="Merged Model Path"),
        title="Model Merger",
        description="Merge Lycoris and Stable Diffusion models",
    )

    iface.launch()


if __name__ == '__main__':
    main()
