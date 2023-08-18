import os
from typing import Optional, List, Type
import torch
from networks.lora import LoRAModule, LoRANetwork
from library import sdxl_original_unet


# input_blocksに適用するかどうか / if True, input_blocks are not applied
SKIP_INPUT_BLOCKS = False

# output_blocksに適用するかどうか / if True, output_blocks are not applied
SKIP_OUTPUT_BLOCKS = True

# conv2dに適用するかどうか / if True, conv2d are not applied
SKIP_CONV2D = False

# transformer_blocksのみに適用するかどうか。Trueの場合、ResBlockには適用されない
# if True, only transformer_blocks are applied, and ResBlocks are not applied
TRANSFORMER_ONLY = True  # if True, SKIP_CONV2D is ignored because conv2d is not used in transformer_blocks

# Trueならattn1とattn2にのみ適用し、ffなどには適用しない / if True, apply only to attn1 and attn2, not to ff etc.
ATTN1_2_ONLY = True

# Trueならattn1やffなどにのみ適用し、attn2などには適用しない / if True, apply only to attn1 and ff, not to attn2
# ATTN1_2_ONLYと同時にTrueにできない / cannot be True at the same time as ATTN1_2_ONLY
ATTN1_ETC_ONLY = False  # True

# transformer_blocksの最大インデックス。Noneなら全てのtransformer_blocksに適用
# max index of transformer_blocks. if None, apply to all transformer_blocks
TRANSFORMER_MAX_BLOCK_INDEX = None


class LoRAModuleControlNet(LoRAModule):
    def __init__(self, depth, cond_emb_dim, name, org_module, multiplier, lora_dim, alpha, dropout=None):
        super().__init__(name, org_module, multiplier, lora_dim, alpha, dropout=dropout)
        self.is_conv2d = org_module.__class__.__name__ == "Conv2d"
        self.cond_emb_dim = cond_emb_dim

        # conditioning1は、conditioning image embeddingを、各LoRA的モジュールでさらに学習する。ここはtimestepごとに呼ばれない
        # それぞれのモジュールで異なる表現を学習することを期待している
        # conditioning1 learns conditioning image embedding in each LoRA-like module. this is not called for each timestep
        # we expect to learn different representations in each module

        # conditioning2は、conditioning1の出力とLoRAの出力を結合し、LoRAの出力に加算する。timestepごとに呼ばれる
        # conditioning image embeddingとU-Netの出力を合わせて学ぶことで、conditioningに応じたU-Netの調整を行う
        # conditioning2 combines the output of conditioning1 and the output of LoRA, and adds it to the output of LoRA. this is called for each timestep
        # by learning the output of conditioning image embedding and U-Net together, we adjust U-Net according to conditioning

        if self.is_conv2d:
            self.conditioning1 = torch.nn.Sequential(
                torch.nn.Conv2d(cond_emb_dim, cond_emb_dim, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(cond_emb_dim, cond_emb_dim, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
            self.conditioning2 = torch.nn.Sequential(
                torch.nn.Conv2d(lora_dim + cond_emb_dim, cond_emb_dim, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(cond_emb_dim, lora_dim, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
        else:
            self.conditioning1 = torch.nn.Sequential(
                torch.nn.Linear(cond_emb_dim, cond_emb_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(cond_emb_dim, cond_emb_dim),
                torch.nn.ReLU(inplace=True),
            )
            self.conditioning2 = torch.nn.Sequential(
                torch.nn.Linear(lora_dim + cond_emb_dim, cond_emb_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(cond_emb_dim, lora_dim),
                torch.nn.ReLU(inplace=True),
            )

        # Zero-Convにするならコメントを外す / uncomment if you want to use Zero-Conv
        # torch.nn.init.zeros_(self.conditioning2[-2].weight)  # zero conv

        self.depth = depth  # 1~3
        self.cond_emb = None
        self.batch_cond_only = False  # Trueなら推論時のcondにのみ適用する / if True, apply only to cond at inference
        self.use_zeros_for_batch_uncond = False  # Trueならuncondのconditioningを0にする / if True, set uncond conditioning to 0

    def set_cond_embs(self, cond_embs_4d, cond_embs_3d):
        r"""
        中でモデルを呼び出すので必要ならwith torch.no_grad()で囲む
        / call the model inside, so if necessary, surround it with torch.no_grad()
        """
        # conv2dとlinearでshapeが違うので必要な方を選択 / select the required one because the shape is different for conv2d and linear
        cond_embs = cond_embs_4d if self.is_conv2d else cond_embs_3d

        cond_emb = cond_embs[self.depth - 1]

        # timestepごとに呼ばれないので、あらかじめ計算しておく / it is not called for each timestep, so calculate it in advance
        self.cond_emb = self.conditioning1(cond_emb)

    def set_batch_cond_only(self, cond_only, zeros):
        self.batch_cond_only = cond_only
        self.use_zeros_for_batch_uncond = zeros

    def forward(self, x):
        if self.cond_emb is None:
            return self.org_forward(x)

        # LoRA-Down
        lx = x
        if self.batch_cond_only:
            lx = lx[1::2]  # cond only in inference

        lx = self.lora_down(lx)

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # conditioning image embeddingを結合 / combine conditioning image embedding
        cx = self.cond_emb

        if not self.batch_cond_only and lx.shape[0] // 2 == cx.shape[0]:  # inference only
            cx = cx.repeat(2, 1, 1, 1) if self.is_conv2d else cx.repeat(2, 1, 1)
            if self.use_zeros_for_batch_uncond:
                cx[0::2] = 0.0  # uncond is zero
        # print(f"C {self.lora_name}, lx.shape={lx.shape}, cx.shape={cx.shape}")

        # 加算ではなくchannel方向に結合することで、うまいこと混ぜてくれることを期待している
        # we expect that it will mix well by combining in the channel direction instead of adding
        cx = torch.cat([cx, lx], dim=1 if self.is_conv2d else 2)
        cx = self.conditioning2(cx)

        lx = lx + cx  # lxはresidual的に加算される / lx is added residually

        # LoRA-Up
        lx = self.lora_up(lx)

        # call original module
        x = self.org_forward(x)

        # add LoRA
        if self.batch_cond_only:
            x[1::2] += lx * self.multiplier * self.scale
        else:
            x += lx * self.multiplier * self.scale

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
                            # block indexからdepthを計算: depthはconditioningのサイズやチャネルを計算するのに使う
                            # block index to depth: depth is using to calculate conditioning size and channels
                            block_name, index1, index2 = (name + "." + child_name).split(".")[:3]
                            index1 = int(index1)
                            if block_name == "input_blocks":
                                if SKIP_INPUT_BLOCKS:
                                    continue
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

                            if TRANSFORMER_MAX_BLOCK_INDEX is not None:
                                p = lora_name.find("transformer_blocks")
                                if p >= 0:
                                    tf_index = int(lora_name[p:].split("_")[2])
                                    if tf_index > TRANSFORMER_MAX_BLOCK_INDEX:
                                        continue

                            #  time embは適用外とする
                            # attn2のconditioning (CLIPからの入力) はshapeが違うので適用できない
                            # time emb is not applied
                            # attn2 conditioning (input from CLIP) cannot be applied because the shape is different
                            if "emb_layers" in lora_name or ("attn2" in lora_name and ("to_k" in lora_name or "to_v" in lora_name)):
                                continue

                            if ATTN1_2_ONLY:
                                if not ("attn1" in lora_name or "attn2" in lora_name):
                                    continue

                            if ATTN1_ETC_ONLY:
                                if "proj_out" in lora_name:
                                    pass
                                elif "attn1" in lora_name and ("to_k" in lora_name or "to_v" in lora_name or "to_out" in lora_name):
                                    pass
                                elif "ff_net_2" in lora_name:
                                    pass
                                else:
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

        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
        if not TRANSFORMER_ONLY:
            target_modules = target_modules + LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        # create module instances
        self.unet_loras: List[LoRAModuleControlNet] = create_modules(unet, target_modules, LoRAModuleControlNet)
        print(f"create ControlNet LoRA for U-Net: {len(self.unet_loras)} modules.")

        # conditioning image embedding

        # control画像そのままではLoRA的モジュールの入力にはサイズもチャネルも扱いにくいので、
        # 適切な潜在空間に変換する。ここでは、conditioning image embeddingと呼ぶ
        # ただcontrol画像自体にはあまり情報量はないので、conditioning image embeddingはわりと小さくてよいはず
        # また、conditioning image embeddingは、各LoRA的モジュールでさらに個別に学習する
        # depthに応じて3つのサイズを用意する

        # conditioning image embedding is converted to an appropriate latent space
        # because the size and channels of the input to the LoRA-like module are difficult to handle
        # we call it conditioning image embedding
        # however, the control image itself does not have much information, so the conditioning image embedding should be small
        # conditioning image embedding is also learned individually in each LoRA-like module
        # prepare three sizes according to depth

        self.cond_block0 = torch.nn.Sequential(
            torch.nn.Conv2d(3, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0),  #  to latent (from VAE) size
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.cond_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(cond_emb_dim, cond_emb_dim, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(cond_emb_dim, cond_emb_dim, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.cond_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(cond_emb_dim, cond_emb_dim, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.cond_block0(x)
        x0 = x
        x = self.cond_block1(x)
        x1 = x
        x = self.cond_block2(x)
        x2 = x

        x_3d = []  # for Linear
        for x0 in [x0, x1, x2]:
            # b,c,h,w -> b,h*w,c
            n, c, h, w = x0.shape
            x0 = x0.view(n, c, h * w).permute(0, 2, 1)
            x_3d.append(x0)

        return [x0, x1, x2], x_3d

    def set_cond_embs(self, cond_embs_4d, cond_embs_3d):
        r"""
        中でモデルを呼び出すので必要ならwith torch.no_grad()で囲む
        / call the model inside, so if necessary, surround it with torch.no_grad()
        """
        for lora in self.unet_loras:
            lora.set_cond_embs(cond_embs_4d, cond_embs_3d)

    def set_batch_cond_only(self, cond_only, zeros):
        for lora in self.unet_loras:
            lora.set_batch_cond_only(cond_only, zeros)

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
    # デバッグ用 / for debug

    # これを指定しないとエラーが出てcond_blockが学習できない / if not specified, an error occurs and cond_block cannot be learned
    sdxl_original_unet.USE_REENTRANT = False

    # test shape etc
    print("create unet")
    unet = sdxl_original_unet.SdxlUNet2DConditionModel()
    unet.to("cuda").to(torch.float16)

    print("create LoRA controlnet")
    control_net = LoRAControlNet(unet, 64, 32, 1)
    control_net.apply_to()
    control_net.to("cuda")

    print(control_net)

    # print number of parameters
    print("number of parameters", sum(p.numel() for p in control_net.parameters() if p.requires_grad))

    input()

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
    # image.render("NeuralNet") # すごく時間がかかるので注意 / be careful because it takes a long time
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
            cond_embs_4d, cond_embs_3d = control_net(conditioning_image)
            control_net.set_cond_embs(cond_embs_4d, cond_embs_3d)

            output = unet(x, t, ctx, y)
            target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
