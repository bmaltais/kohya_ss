# some parts are modified from Diffusers library (Apache License 2.0)

import math
from types import SimpleNamespace
from typing import Any, Optional
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import sdxl_original_unet
from library.sdxl_model_util import convert_sdxl_unet_state_dict_to_diffusers, convert_diffusers_unet_state_dict_to_sdxl


class ControlNetConditioningEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        dims = [16, 32, 96, 256]

        self.conv_in = nn.Conv2d(3, dims[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])

        for i in range(len(dims) - 1):
            channel_in = dims[i]
            channel_out = dims[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = nn.Conv2d(dims[-1], 320, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)  # zero module weight
        nn.init.zeros_(self.conv_out.bias)  # zero module bias

    def forward(self, x):
        x = self.conv_in(x)
        x = F.silu(x)
        for block in self.blocks:
            x = block(x)
            x = F.silu(x)
        x = self.conv_out(x)
        return x


class SdxlControlNet(sdxl_original_unet.SdxlUNet2DConditionModel):
    def __init__(self, multiplier: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.multiplier = multiplier

        # remove unet layers
        self.output_blocks = nn.ModuleList([])
        del self.out

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding()

        dims = [320, 320, 320, 320, 640, 640, 640, 1280, 1280]
        self.controlnet_down_blocks = nn.ModuleList([])
        for dim in dims:
            self.controlnet_down_blocks.append(nn.Conv2d(dim, dim, kernel_size=1))
            nn.init.zeros_(self.controlnet_down_blocks[-1].weight)  # zero module weight
            nn.init.zeros_(self.controlnet_down_blocks[-1].bias)  # zero module bias

        self.controlnet_mid_block = nn.Conv2d(1280, 1280, kernel_size=1)
        nn.init.zeros_(self.controlnet_mid_block.weight)  # zero module weight
        nn.init.zeros_(self.controlnet_mid_block.bias)  # zero module bias

    def init_from_unet(self, unet: sdxl_original_unet.SdxlUNet2DConditionModel):
        unet_sd = unet.state_dict()
        unet_sd = {k: v for k, v in unet_sd.items() if not k.startswith("out")}
        sd = super().state_dict()
        sd.update(unet_sd)
        info = super().load_state_dict(sd, strict=True, assign=True)
        return info

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = True) -> Any:
        # convert state_dict to SAI format
        unet_sd = {}
        for k in list(state_dict.keys()):
            if not k.startswith("controlnet_"):
                unet_sd[k] = state_dict.pop(k)
        unet_sd = convert_diffusers_unet_state_dict_to_sdxl(unet_sd)
        state_dict.update(unet_sd)
        super().load_state_dict(state_dict, strict=strict, assign=assign)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # convert state_dict to Diffusers format
        state_dict = super().state_dict(destination, prefix, keep_vars)
        control_net_sd = {}
        for k in list(state_dict.keys()):
            if k.startswith("controlnet_"):
                control_net_sd[k] = state_dict.pop(k)
        state_dict = convert_sdxl_unet_state_dict_to_diffusers(state_dict)
        state_dict.update(control_net_sd)
        return state_dict

    def forward(
        self,
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        cond_image: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # broadcast timesteps to batch dimension
        timesteps = timesteps.expand(x.shape[0])

        t_emb = sdxl_original_unet.get_timestep_embedding(timesteps, self.model_channels, downscale_freq_shift=0)
        t_emb = t_emb.to(x.dtype)
        emb = self.time_embed(t_emb)

        assert x.shape[0] == y.shape[0], f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
        assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} != {y.dtype}"
        emb = emb + self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                if isinstance(layer, sdxl_original_unet.ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, sdxl_original_unet.Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        h = x
        multiplier = self.multiplier if self.multiplier is not None else 1.0
        hs = []
        for i, module in enumerate(self.input_blocks):
            h = call_module(module, h, emb, context)
            if i == 0:
                h = self.controlnet_cond_embedding(cond_image) + h
            hs.append(self.controlnet_down_blocks[i](h) * multiplier)

        h = call_module(self.middle_block, h, emb, context)
        h = self.controlnet_mid_block(h) * multiplier

        return hs, h


class SdxlControlledUNet(sdxl_original_unet.SdxlUNet2DConditionModel):
    """
    This class is for training purpose only.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, timesteps=None, context=None, y=None, input_resi_add=None, mid_add=None, **kwargs):
        # broadcast timesteps to batch dimension
        timesteps = timesteps.expand(x.shape[0])

        hs = []
        t_emb = sdxl_original_unet.get_timestep_embedding(timesteps, self.model_channels, downscale_freq_shift=0)
        t_emb = t_emb.to(x.dtype)
        emb = self.time_embed(t_emb)

        assert x.shape[0] == y.shape[0], f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
        assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} != {y.dtype}"
        emb = emb + self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                if isinstance(layer, sdxl_original_unet.ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, sdxl_original_unet.Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        h = x
        for module in self.input_blocks:
            h = call_module(module, h, emb, context)
            hs.append(h)

        h = call_module(self.middle_block, h, emb, context)
        h = h + mid_add

        for module in self.output_blocks:
            resi = hs.pop() + input_resi_add.pop()
            h = torch.cat([h, resi], dim=1)
            h = call_module(module, h, emb, context)

        h = h.type(x.dtype)
        h = call_module(self.out, h, emb, context)

        return h


if __name__ == "__main__":
    import time

    logger.info("create unet")
    unet = SdxlControlledUNet()
    unet.to("cuda", torch.bfloat16)
    unet.set_use_sdpa(True)
    unet.set_gradient_checkpointing(True)
    unet.train()

    logger.info("create control_net")
    control_net = SdxlControlNet()
    control_net.to("cuda")
    control_net.set_use_sdpa(True)
    control_net.set_gradient_checkpointing(True)
    control_net.train()

    logger.info("Initialize control_net from unet")
    control_net.init_from_unet(unet)

    unet.requires_grad_(False)
    control_net.requires_grad_(True)

    # 使用メモリ量確認用の疑似学習ループ
    logger.info("preparing optimizer")

    # optimizer = torch.optim.SGD(unet.parameters(), lr=1e-3, nesterov=True, momentum=0.9) # not working

    import bitsandbytes

    optimizer = bitsandbytes.adam.Adam8bit(control_net.parameters(), lr=1e-3)  # not working
    # optimizer = bitsandbytes.optim.RMSprop8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2
    # optimizer=bitsandbytes.optim.Adagrad8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2

    # import transformers
    # optimizer = transformers.optimization.Adafactor(unet.parameters(), relative_step=True)  # working at 22.2GB with torch2

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    logger.info("start training")
    steps = 10
    batch_size = 1

    for step in range(steps):
        logger.info(f"step {step}")
        if step == 1:
            time_start = time.perf_counter()

        x = torch.randn(batch_size, 4, 128, 128).cuda()  # 1024x1024
        t = torch.randint(low=0, high=1000, size=(batch_size,), device="cuda")
        txt = torch.randn(batch_size, 77, 2048).cuda()
        vector = torch.randn(batch_size, sdxl_original_unet.ADM_IN_CHANNELS).cuda()
        cond_img = torch.rand(batch_size, 3, 1024, 1024).cuda()

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            input_resi_add, mid_add = control_net(x, t, txt, vector, cond_img)
            output = unet(x, t, txt, vector, input_resi_add, mid_add)
            target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    time_end = time.perf_counter()
    logger.info(f"elapsed time: {time_end - time_start} [sec] for last {steps - 1} steps")

    logger.info("finish training")
    sd = control_net.state_dict()

    from safetensors.torch import save_file

    save_file(sd, r"E:\Work\SD\Tmp\sdxl\ctrl\control_net.safetensors")
