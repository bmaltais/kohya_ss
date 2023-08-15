import os
from typing import Optional, List, Type
import torch
from networks.lora import LoRAModule, LoRANetwork
from library import sdxl_original_unet


SKIP_OUTPUT_BLOCKS = False
SKIP_CONV2D = False


class LoRAModuleControlNet(LoRAModule):
    def __init__(self, depth, cond_emb_dim, name, org_module, multiplier, lora_dim, alpha, dropout=None):
        super().__init__(name, org_module, multiplier, lora_dim, alpha, dropout=dropout)
        self.is_conv2d = org_module.__class__.__name__ == "Conv2d"

        # adjust channels of conditioning image to LoRA channels
        ch = 2 ** (depth - 1) * cond_emb_dim
        if self.is_conv2d:
            self.conditioning = torch.nn.Conv2d(ch, lora_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.conditioning = torch.nn.Linear(ch, lora_dim)
        torch.nn.init.zeros_(self.conditioning.weight)  # zero conv/linear layer

        self.depth = depth
        self.cond_emb_dim = cond_emb_dim
        self.cond_emb = None

    def set_control(self, cond_emb):
        self.cond_emb = cond_emb

    def forward(self, x):
        # conditioning image embs -> LoRA channels
        cx = self.cond_emb
        if not self.is_conv2d:
            # b,c,h,w -> b,h*w,c
            n, c, h, w = cx.shape
            cx = cx.view(n, c, h * w).permute(0, 2, 1)
        # print(f"C {self.lora_name}, x.shape={x.shape}, cx.shape={cx.shape}, weight.shape={self.conditioning.weight.shape}")
        cx = self.conditioning(cx)

        # LoRA
        # print(f"C {self.lora_name}, x.shape={x.shape}, cx.shape={cx.shape}")
        lx = self.lora_down(x)

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # add conditioning
        lx = lx + cx

        lx = self.lora_up(lx)

        x = self.org_forward(x) + lx * self.multiplier * self.scale
        return x


class LoRAControlNet(torch.nn.Module):
    def __init__(
        self,
        unet: sdxl_original_unet.SdxlUNet2DConditionModel,
        cond_emb_dim: int = 16,
        lora_dim: int = 16,
        alpha: float = 1,
        dropout: Optional[float] = None,
        varbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        # self.unets = [unet]

        def create_modules(
            root_module: torch.nn.Module,
            target_replace_modules: List[torch.nn.Module],
            module_class: Type[object],
        ) -> List[torch.nn.Module]:
            prefix = LoRANetwork.LORA_PREFIX_UNET

            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"

                        if is_linear or (is_conv2d and not SKIP_CONV2D):
                            # block index to depth: depth is using to calculate conditioning size and channels
                            block_name, index1, index2 = (name + "." + child_name).split(".")[:3]
                            index1 = int(index1)
                            if block_name == "input_blocks":
                                depth = 1 if index1 <= 2 else (2 if index1 <= 5 else 3)
                            elif block_name == "middle_block":
                                depth = 3
                            elif block_name == "output_blocks":
                                if SKIP_OUTPUT_BLOCKS:
                                    continue
                                depth = 3 if index1 <= 2 else (2 if index1 <= 5 else 1)
                                if int(index2) >= 2:
                                    depth -= 1
                            else:
                                raise NotImplementedError()

                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            # skip time emb or clip emb
                            if "emb_layers" in lora_name or ("attn2" in lora_name and ("to_k" in lora_name or "to_v" in lora_name)):
                                continue

                            lora = module_class(
                                depth,
                                cond_emb_dim,
                                lora_name,
                                child_module,
                                1.0,
                                lora_dim,
                                alpha,
                                dropout=dropout,
                            )
                            loras.append(lora)
            return loras

        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE + LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        # create module instances
        self.unet_loras: List[LoRAModuleControlNet] = create_modules(unet, target_modules, LoRAModuleControlNet)
        print(f"create ControlNet LoRA for U-Net: {len(self.unet_loras)} modules.")

        # stem for conditioning image
        self.cond_stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, cond_emb_dim, kernel_size=4, stride=4, padding=0),
            torch.nn.ReLU(inplace=True),
        )

        # embs for each depth
        self.cond_block0 = torch.nn.Sequential(
            torch.nn.Conv2d(cond_emb_dim, cond_emb_dim, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.cond_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(cond_emb_dim, cond_emb_dim * 2, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.cond_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(cond_emb_dim * 2, cond_emb_dim * 4, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.cond_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(cond_emb_dim * 4, cond_emb_dim * 8, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    # forawrdでなくset_controlに入れてもやはり動かない
    def forward(self, x):
        cx = self.cond_stem(x)
        cx = self.cond_block0(cx)
        c0 = cx
        cx = self.cond_block1(cx)
        c1 = cx
        cx = self.cond_block2(cx)
        c2 = cx
        cx = self.cond_block3(cx)
        c3 = cx
        return c0, c1, c2, c3

    def set_control(self, cond_embs):
        for lora in self.unet_loras:
            lora.set_control(cond_embs[lora.depth - 1])

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self):
        print("applying LoRA for U-Net...")
        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    # マージできるかどうかを返す
    def is_mergeable(self):
        return False

    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        raise NotImplementedError()

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_optimizer_params(self):
        self.requires_grad_(True)
        return self.parameters()

    def prepare_grad_etc(self):
        self.requires_grad_(True)

    def on_epoch_start(self):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)


if __name__ == "__main__":
    # test shape etc
    print("create unet")
    unet = sdxl_original_unet.SdxlUNet2DConditionModel()
    unet.to("cuda")  # , dtype=torch.float16)

    print("create LoRA controlnet")
    control_net = LoRAControlNet(unet, 16, 32, 1)
    control_net.apply_to()
    control_net.to("cuda")

    # print(controlnet)
    # input()

    # print number of parameters
    print("number of parameters", sum(p.numel() for p in control_net.parameters() if p.requires_grad))

    unet.set_use_memory_efficient_attention(True, False)
    unet.set_gradient_checkpointing(True)
    unet.train()  # for gradient checkpointing

    control_net.train()

    # # visualize
    # import torchviz
    # print("run visualize")
    # controlnet.set_control(conditioning_image)
    # output = unet(x, t, ctx, y)
    # print("make_dot")
    # image = torchviz.make_dot(output, params=dict(controlnet.named_parameters()))
    # print("render")
    # image.format = "svg" # "png"
    # image.render("NeuralNet")
    # input()

    import bitsandbytes

    optimizer = bitsandbytes.adam.Adam8bit(control_net.prepare_optimizer_params(), 1e-3)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print("start training")
    steps = 10

    for step in range(steps):
        print(f"step {step}")

        batch_size = 1
        conditioning_image = torch.rand(batch_size, 3, 1024, 1024).cuda() * 2.0 - 1.0
        x = torch.randn(batch_size, 4, 128, 128).cuda()
        t = torch.randint(low=0, high=10, size=(batch_size,)).cuda()
        ctx = torch.randn(batch_size, 77, 2048).cuda()
        y = torch.randn(batch_size, sdxl_original_unet.ADM_IN_CHANNELS).cuda()

        with torch.cuda.amp.autocast(enabled=True):
            cond_embs = control_net(conditioning_image)
            control_net.set_control(cond_embs)
            output = unet(x, t, ctx, y)
            target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
