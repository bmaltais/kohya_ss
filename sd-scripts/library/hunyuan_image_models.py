# Original work: https://github.com/Tencent-Hunyuan/HunyuanImage-2.1
# Re-implemented for license compliance for sd-scripts.

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from library import custom_offloading_utils
from library.attention import AttentionParams
from library.fp8_optimization_utils import apply_fp8_monkey_patch
from library.lora_utils import load_safetensors_with_lora_and_fp8
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


from library.hunyuan_image_modules import (
    SingleTokenRefiner,
    ByT5Mapper,
    PatchEmbed2D,
    TimestepEmbedder,
    MMDoubleStreamBlock,
    MMSingleStreamBlock,
    FinalLayer,
)
from library.hunyuan_image_utils import get_nd_rotary_pos_embed

FP8_OPTIMIZATION_TARGET_KEYS = ["double_blocks", "single_blocks"]
# FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "_mod", "_emb"]  # , "modulation"
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "_emb"]  # , "modulation", "_mod"

# full exclude 24.2GB
# norm and _emb 19.7GB
# fp8 cast 19.7GB


# region DiT Model
class HYImageDiffusionTransformer(nn.Module):
    """
    HunyuanImage-2.1 Diffusion Transformer.

    A multimodal transformer for image generation with text conditioning,
    featuring separate double-stream and single-stream processing blocks.

    Args:
        attn_mode: Attention implementation mode ("torch" or "sageattn").
    """

    def __init__(self, attn_mode: str = "torch", split_attn: bool = False):
        super().__init__()

        # Fixed architecture parameters for HunyuanImage-2.1
        self.patch_size = [1, 1]  # 1x1 patch size (no spatial downsampling)
        self.in_channels = 64  # Input latent channels
        self.out_channels = 64  # Output latent channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = False  # Guidance embedding disabled
        self.rope_dim_list = [64, 64]  # RoPE dimensions for 2D positional encoding
        self.rope_theta = 256  # RoPE frequency scaling
        self.use_attention_mask = True
        self.text_projection = "single_refiner"
        self.hidden_size = 3584  # Model dimension
        self.heads_num = 28  # Number of attention heads

        # Architecture configuration
        mm_double_blocks_depth = 20  # Double-stream transformer blocks
        mm_single_blocks_depth = 40  # Single-stream transformer blocks
        mlp_width_ratio = 4  # MLP expansion ratio
        text_states_dim = 3584  # Text encoder output dimension
        guidance_embed = False  # No guidance embedding

        # Layer configuration
        mlp_act_type: str = "gelu_tanh"  # MLP activation function
        qkv_bias: bool = True  # Use bias in QKV projections
        qk_norm: bool = True  # Apply QK normalization
        qk_norm_type: str = "rms"  # RMS normalization type

        self.attn_mode = attn_mode
        self.split_attn = split_attn

        # ByT5 character-level text encoder mapping
        self.byt5_in = ByT5Mapper(in_dim=1472, out_dim=2048, hidden_dim=2048, out_dim1=self.hidden_size, use_residual=False)

        # Image latent patch embedding
        self.img_in = PatchEmbed2D(self.patch_size, self.in_channels, self.hidden_size)

        # Text token refinement with cross-attention
        self.txt_in = SingleTokenRefiner(text_states_dim, self.hidden_size, self.heads_num, depth=2)

        # Timestep embedding for diffusion process
        self.time_in = TimestepEmbedder(self.hidden_size, nn.SiLU)

        # MeanFlow not supported in this implementation
        self.time_r_in = None

        # Guidance embedding (disabled for non-distilled model)
        self.guidance_in = TimestepEmbedder(self.hidden_size, nn.SiLU) if guidance_embed else None

        # Double-stream blocks: separate image and text processing
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # Single-stream blocks: joint processing of concatenated features
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels, nn.SiLU)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.blocks_to_swap = None

        self.offloader_double = None
        self.offloader_single = None
        self.num_double_blocks = len(self.double_blocks)
        self.num_single_blocks = len(self.single_blocks)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

        for block in self.double_blocks + self.single_blocks:
            block.enable_gradient_checkpointing(cpu_offload=cpu_offload)

        print(f"HunyuanImage-2.1: Gradient checkpointing enabled. CPU offload: {cpu_offload}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

        for block in self.double_blocks + self.single_blocks:
            block.disable_gradient_checkpointing()

        print("HunyuanImage-2.1: Gradient checkpointing disabled.")

    def enable_block_swap(self, num_blocks: int, device: torch.device, supports_backward: bool = False):
        self.blocks_to_swap = num_blocks
        double_blocks_to_swap = num_blocks // 2
        single_blocks_to_swap = (num_blocks - double_blocks_to_swap) * 2

        assert double_blocks_to_swap <= self.num_double_blocks - 2 and single_blocks_to_swap <= self.num_single_blocks - 2, (
            f"Cannot swap more than {self.num_double_blocks - 2} double blocks and {self.num_single_blocks - 2} single blocks. "
            f"Requested {double_blocks_to_swap} double blocks and {single_blocks_to_swap} single blocks."
        )

        self.offloader_double = custom_offloading_utils.ModelOffloader(
            self.double_blocks, double_blocks_to_swap, device, supports_backward=supports_backward
        )
        self.offloader_single = custom_offloading_utils.ModelOffloader(
            self.single_blocks, single_blocks_to_swap, device, supports_backward=supports_backward
        )
        # , debug=True
        print(
            f"HunyuanImage-2.1: Block swap enabled. Swapping {num_blocks} blocks, double blocks: {double_blocks_to_swap}, single blocks: {single_blocks_to_swap}."
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader_double.set_forward_only(True)
            self.offloader_single.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print(f"HunyuanImage-2.1: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader_double.set_forward_only(False)
            self.offloader_single.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print(f"HunyuanImage-2.1: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            save_double_blocks = self.double_blocks
            save_single_blocks = self.single_blocks
            self.double_blocks = nn.ModuleList()
            self.single_blocks = nn.ModuleList()

        self.to(device)

        if self.blocks_to_swap:
            self.double_blocks = save_double_blocks
            self.single_blocks = save_single_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader_double.prepare_block_devices_before_forward(self.double_blocks)
        self.offloader_single.prepare_block_devices_before_forward(self.single_blocks)

    def get_rotary_pos_embed(self, rope_sizes):
        """
        Generate 2D rotary position embeddings for image tokens.

        Args:
            rope_sizes: Tuple of (height, width) for spatial dimensions.

        Returns:
            Tuple of (freqs_cos, freqs_sin) tensors for rotary position encoding.
        """
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(self.rope_dim_list, rope_sizes, theta=self.rope_theta)
        return freqs_cos, freqs_sin

    def reorder_txt_token(
        self, byt5_txt: torch.Tensor, txt: torch.Tensor, byt5_text_mask: torch.Tensor, text_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        Combine and reorder ByT5 character-level and word-level text embeddings.

        Concatenates valid tokens from both encoders and creates appropriate masks.

        Args:
            byt5_txt: ByT5 character-level embeddings [B, L1, D].
            txt: Word-level text embeddings [B, L2, D].
            byt5_text_mask: Valid token mask for ByT5 [B, L1].
            text_mask: Valid token mask for word tokens [B, L2].

        Returns:
            Tuple of (reordered_embeddings, combined_mask, sequence_lengths).
        """
        # Process each batch element separately to handle variable sequence lengths

        reorder_txt = []
        reorder_mask = []

        txt_lens = []
        for i in range(text_mask.shape[0]):
            byt5_text_mask_i = byt5_text_mask[i].bool()
            text_mask_i = text_mask[i].bool()
            byt5_text_length = byt5_text_mask_i.sum()
            text_length = text_mask_i.sum()
            assert byt5_text_length == byt5_text_mask_i[:byt5_text_length].sum()
            assert text_length == text_mask_i[:text_length].sum()

            byt5_txt_i = byt5_txt[i]
            txt_i = txt[i]
            reorder_txt_i = torch.cat(
                [byt5_txt_i[:byt5_text_length], txt_i[:text_length], byt5_txt_i[byt5_text_length:], txt_i[text_length:]], dim=0
            )

            reorder_mask_i = torch.zeros(
                byt5_text_mask_i.shape[0] + text_mask_i.shape[0], dtype=torch.bool, device=byt5_text_mask_i.device
            )
            reorder_mask_i[: byt5_text_length + text_length] = True

            reorder_txt.append(reorder_txt_i)
            reorder_mask.append(reorder_mask_i)
            txt_lens.append(byt5_text_length + text_length)

        reorder_txt = torch.stack(reorder_txt)
        reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)

        return reorder_txt, reorder_mask, txt_lens

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        byt5_text_states: Optional[torch.Tensor] = None,
        byt5_text_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb_cache: Optional[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the HunyuanImage diffusion transformer.

        Args:
            hidden_states: Input image latents [B, C, H, W].
            timestep: Diffusion timestep [B].
            text_states: Word-level text embeddings [B, L, D].
            encoder_attention_mask: Text attention mask [B, L].
            byt5_text_states: ByT5 character-level embeddings [B, L_byt5, D_byt5].
            byt5_text_mask: ByT5 attention mask [B, L_byt5].

        Returns:
            Tuple of (denoised_image, spatial_shape).
        """
        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        txt = text_states

        # Calculate spatial dimensions for rotary position embeddings
        _, _, oh, ow = x.shape
        th, tw = oh, ow  # Height and width (patch_size=[1,1] means no spatial downsampling)
        if rotary_pos_emb_cache is not None:
            if (th, tw) in rotary_pos_emb_cache:
                freqs_cis = rotary_pos_emb_cache[(th, tw)]
                freqs_cis = (freqs_cis[0].to(img.device), freqs_cis[1].to(img.device))
            else:
                freqs_cis = self.get_rotary_pos_embed((th, tw))
                rotary_pos_emb_cache[(th, tw)] = (freqs_cis[0].cpu(), freqs_cis[1].cpu())
        else:
            freqs_cis = self.get_rotary_pos_embed((th, tw))

        # Reshape image latents to sequence format: [B, C, H, W] -> [B, H*W, C]
        img = self.img_in(img)

        # Generate timestep conditioning vector
        vec = self.time_in(t)

        # MeanFlow and guidance embedding not used in this configuration

        # Process text tokens through refinement layers
        txt_attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, 0, text_mask)
        txt = self.txt_in(txt, t, txt_attn_params)

        # Integrate character-level ByT5 features with word-level tokens
        # Use variable length sequences with sequence lengths
        byt5_txt = self.byt5_in(byt5_text_states)
        txt, text_mask, txt_lens = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask)

        # Trim sequences to maximum length in the batch
        img_seq_len = img.shape[1]
        max_txt_len = max(txt_lens)
        txt = txt[:, :max_txt_len, :]
        text_mask = text_mask[:, :max_txt_len]

        attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, img_seq_len, text_mask)

        input_device = img.device

        # Process through double-stream blocks (separate image/text attention)
        for index, block in enumerate(self.double_blocks):
            if self.blocks_to_swap:
                self.offloader_double.wait_for_block(index)
            img, txt = block(img, txt, vec, freqs_cis, attn_params)
            if self.blocks_to_swap:
                self.offloader_double.submit_move_blocks(self.double_blocks, index)

        # Concatenate image and text tokens for joint processing
        x = torch.cat((img, txt), 1)

        # Process through single-stream blocks (joint attention)
        for index, block in enumerate(self.single_blocks):
            if self.blocks_to_swap:
                self.offloader_single.wait_for_block(index)
            x = block(x, vec, freqs_cis, attn_params)
            if self.blocks_to_swap:
                self.offloader_single.submit_move_blocks(self.single_blocks, index)

        x = x.to(input_device)
        vec = vec.to(input_device)

        img = x[:, :img_seq_len, ...]
        del x

        # Apply final projection to output space
        img = self.final_layer(img, vec)
        del vec

        # Reshape from sequence to spatial format: [B, L, C] -> [B, C, H, W]
        img = self.unpatchify_2d(img, th, tw)
        return img

    def unpatchify_2d(self, x, h, w):
        """
        Convert sequence format back to spatial image format.

        Args:
            x: Input tensor [B, H*W, C].
            h: Height dimension.
            w: Width dimension.

        Returns:
            Spatial tensor [B, C, H, W].
        """
        c = self.unpatchify_channels

        x = x.reshape(shape=(x.shape[0], h, w, c))
        imgs = x.permute(0, 3, 1, 2)
        return imgs


# endregion

# region Model Utils


def create_model(attn_mode: str, split_attn: bool, dtype: Optional[torch.dtype]) -> HYImageDiffusionTransformer:
    with init_empty_weights():
        model = HYImageDiffusionTransformer(attn_mode=attn_mode, split_attn=split_attn)
        if dtype is not None:
            model.to(dtype)
    return model


def load_hunyuan_image_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[Dict[str, torch.Tensor]] = None,
    lora_multipliers: Optional[list[float]] = None,
) -> HYImageDiffusionTransformer:
    """
    Load a HunyuanImage model from the specified checkpoint.

    Args:
        device (Union[str, torch.device]): Device for optimization or merging
        dit_path (str): Path to the DiT model checkpoint.
        attn_mode (str): Attention mode to use, e.g., "torch", "flash", etc.
        split_attn (bool): Whether to use split attention.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights.
            If None, it will be loaded as is (same as the state_dict) or scaled for fp8. if not None, model weights will be casted to this dtype.
        fp8_scaled (bool): Whether to use fp8 scaling for the model weights.
        lora_weights_list (Optional[Dict[str, torch.Tensor]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
    """
    # dit_weight_dtype is None for fp8_scaled
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    model = create_model(attn_mode, split_attn, dit_weight_dtype)

    # load model weights with dynamic fp8 optimization and LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
    )

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded DiT model from {dit_path}, info={info}")

    return model


# endregion
