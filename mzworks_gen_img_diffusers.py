import os
import argparse
import time
import random

import torch
from diffusers import StableDiffusionPipeline
from networks.lora import create_network_from_weights
from safetensors.torch import load_file

DEVICE = "mps"


class CPrompt(object):
    prompt: str = ""
    height: int = None
    width: int = None
    num_inference_steps: float = None
    guidance_scale: float = None
    negative_prompt: str = None
    seed: int = None
    valid: bool = False

    def __init__(self, args: argparse.Namespace):
        if args.H is not None:
            self.height = args.H
        if args.W is not None:
            self.width = args.W
        if args.steps is not None:
            self.num_inference_steps = args.steps
        if args.scale is not None:
            self.guidance_scale = args.scale

    def load(self, prompt: str) -> bool:
        found_pos = prompt.find("#")
        if found_pos > -1:
            prompt = prompt[:found_pos]

        list_txt_block = [v.strip() for v in prompt.strip().split("--")]

        if len(list_txt_block[0]) == 0:
            return False

        self.prompt = list_txt_block[0]

        for txt_block in list_txt_block[1:]:
            match txt_block[0]:
                case "n":
                    self.negative_prompt = txt_block[1:].strip()
                case "d":
                    self.seed = int(txt_block[1:])
                case "s":
                    self.num_inference_steps = int(txt_block[1:])
                case "l":
                    self.guidance_scale = float(txt_block[1:])

        return True

    def dict(self) -> dict:
        dict_result = {"prompt": self.prompt}

        if self.negative_prompt is not None:
            dict_result["negative_prompt"] = self.negative_prompt

        if self.seed is None:
            self.seed = random.randint(0, 0x7FFFFFFF)
        dict_result["generator"] = torch.Generator().manual_seed(self.seed)

        if self.num_inference_steps is not None:
            dict_result["num_inference_steps"] = self.num_inference_steps

        if self.guidance_scale is not None:
            dict_result["guidance_scale"] = self.guidance_scale

        return dict_result


def prompt_from_file(args: argparse.Namespace) -> list[CPrompt]:
    list_result = []
    with open(args.from_file, "r") as rf:
        for text in rf.readlines():
            o = CPrompt(args)
            if o.load(text) is True:
                list_result.append(o)

    return list_result


def skip_safety_checker(images, **kwargs):
    return images, [False]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--ckpt", type=str)
    # parser.add_argument("--clip_skip", type=int)
    parser.add_argument("--from_file", type=str)
    # parser.add_argument("--images_per_prompt", type=int)
    # parser.add_argument("--max_embeddings_multiples", type=int)
    parser.add_argument("--n_iter", type=int, default=None)
    # parser.add_argument("--network_module", type=str)
    parser.add_argument("--network_mul", type=float)
    parser.add_argument("--network_weights", type=str)
    parser.add_argument("--outdir", type=str)
    # parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--W", type=int, default=None)

    args = parser.parse_args()

    print("Create StableDiffusionPipeline")
    print("--ckpt   {:s}".format(args.ckpt))

    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float32
    ).to(DEVICE)

    pipe.safety_checker = skip_safety_checker

    vae = pipe.vae
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    print("--network_weights {:s}".format(args.network_weights))
    print("--network_mul {:f}".format(args.network_mul))

    sd = load_file(args.network_weights)
    network1, sd = create_network_from_weights(
        args.network_mul, None, vae, text_encoder, unet, sd
    )
    network1.apply_to(text_encoder, unet)
    network1.load_state_dict(sd)
    network1.to(DEVICE, dtype=torch.float32)

    print("--outdir {:s}".format(args.outdir))
    print("--from_file {:s}".format(args.from_file))

    os.makedirs(args.outdir, exist_ok=True)

    for idx, o_prompt in enumerate(prompt_from_file(args)):
        image = pipe(**o_prompt.dict()).images[0]

        filename = "im_{:s}_{:03d}_{:d}.png".format(
            time.strftime("%Y%m%d%H%M%S", time.localtime()), idx, o_prompt.seed
        )

        image.save(os.path.join(args.outdir, filename))


if __name__ == "__main__":
    main()
