# Original work: https://github.com/Tencent-Hunyuan/HunyuanImage-2.1
# Re-implemented for license compliance for sd-scripts.

from typing import Tuple, Callable
import torch
import torch.nn as nn
from einops import rearrange

from library import custom_offloading_utils
from library.attention import AttentionParams, attention
from library.hunyuan_image_utils import timestep_embedding, apply_rotary_emb, _to_tuple, apply_gate, modulate
from library.attention import attention

# region Modules


class ByT5Mapper(nn.Module):
    """
    Maps ByT5 character-level encoder outputs to transformer hidden space.

    Applies layer normalization, two MLP layers with GELU activation,
    and optional residual connection.

    Args:
        in_dim: Input dimension from ByT5 encoder (1472 for ByT5-large).
        out_dim: Intermediate dimension after first projection.
        hidden_dim: Hidden dimension for MLP layer.
        out_dim1: Final output dimension matching transformer hidden size.
        use_residual: Whether to add residual connection (requires in_dim == out_dim).
    """

    def __init__(self, in_dim, out_dim, hidden_dim, out_dim1, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim1)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        """
        Transform ByT5 embeddings to transformer space.

        Args:
            x: Input ByT5 embeddings [..., in_dim].

        Returns:
            Transformed embeddings [..., out_dim1].
        """
        residual = x if self.use_residual else None
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        x = self.fc3(x)
        if self.use_residual:
            x = x + residual
        return x


class PatchEmbed2D(nn.Module):
    """
    2D patch embedding layer for converting image latents to transformer tokens.

    Uses 2D convolution to project image patches to embedding space.
    For HunyuanImage-2.1, patch_size=[1,1] means no spatial downsampling.

    Args:
        patch_size: Spatial size of patches (int or tuple).
        in_chans: Number of input channels.
        embed_dim: Output embedding dimension.
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        self.norm = nn.Identity()  # No normalization layer used

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar diffusion timesteps into vector representations.

    Uses sinusoidal encoding followed by a two-layer MLP.

    Args:
        hidden_size: Output embedding dimension.
        act_layer: Activation function class (e.g., nn.SiLU).
        frequency_embedding_size: Dimension of sinusoidal encoding.
        max_period: Maximum period for sinusoidal frequencies.
        out_size: Output dimension (defaults to hidden_size).
    """

    def __init__(self, hidden_size, act_layer, frequency_embedding_size=256, max_period=10000, out_size=None):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True), act_layer(), nn.Linear(hidden_size, out_size, bias=True)
        )

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class TextProjection(nn.Module):
    """
    Projects text embeddings through a two-layer MLP.

    Used for context-aware representation computation in token refinement.

    Args:
        in_channels: Input feature dimension.
        hidden_size: Hidden and output dimension.
        act_layer: Activation function class.
    """

    def __init__(self, in_channels, hidden_size, act_layer):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_channels, out_features=hidden_size, bias=True)
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable activation and normalization.

    Standard two-layer MLP with optional dropout and intermediate normalization.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer dimension (defaults to in_channels).
        out_features: Output dimension (defaults to in_channels).
        act_layer: Activation function class.
        norm_layer: Optional normalization layer class.
        bias: Whether to use bias (can be bool or tuple for each layer).
        drop: Dropout rate (can be float or tuple for each layer).
        use_conv: Whether to use convolution instead of linear (not supported).
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        assert not use_conv, "Convolutional MLP not supported in this implementation."

        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        bias = _to_tuple(bias, 2)
        drop_probs = _to_tuple(drop, 2)

        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_channels) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_channels, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class IndividualTokenRefinerBlock(nn.Module):
    """
    Single transformer block for individual token refinement.

    Applies self-attention and MLP with adaptive layer normalization (AdaLN)
    conditioned on timestep and context information.

    Args:
        hidden_size: Model dimension.
        heads_num: Number of attention heads.
        mlp_width_ratio: MLP expansion ratio.
        mlp_drop_rate: MLP dropout rate.
        act_type: Activation function (only "silu" supported).
        qk_norm: QK normalization flag (must be False).
        qk_norm_type: QK normalization type (only "layer" supported).
        qkv_bias: Use bias in QKV projections.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
    ):
        super().__init__()
        assert qk_norm_type == "layer", "Only layer normalization supported for QK norm."
        assert act_type == "silu", "Only SiLU activation supported."
        assert not qk_norm, "QK normalization must be disabled."

        self.heads_num = heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.self_attn_q_norm = nn.Identity()
        self.self_attn_k_norm = nn.Identity()
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(in_channels=hidden_size, hidden_channels=mlp_hidden_dim, act_layer=nn.SiLU, drop=mlp_drop_rate)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_params: AttentionParams) -> torch.Tensor:
        """
        Apply self-attention and MLP with adaptive conditioning.

        Args:
            x: Input token embeddings [B, L, C].
            c: Combined conditioning vector [B, C].
            attn_params: Attention parameters including sequence lengths.

        Returns:
            Refined token embeddings [B, L, C].
        """
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        del norm_x
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        del qkv
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)
        qkv = [q, k, v]
        del q, k, v
        attn = attention(qkv, attn_params=attn_params)

        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)
        return x


class IndividualTokenRefiner(nn.Module):
    """
    Stack of token refinement blocks with self-attention.

    Processes tokens individually with adaptive layer normalization.

    Args:
        hidden_size: Model dimension.
        heads_num: Number of attention heads.
        depth: Number of refinement blocks.
        mlp_width_ratio: MLP expansion ratio.
        mlp_drop_rate: MLP dropout rate.
        act_type: Activation function type.
        qk_norm: QK normalization flag.
        qk_norm_type: QK normalization type.
        qkv_bias: Use bias in QKV projections.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        depth: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    act_type=act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.LongTensor, attn_params: AttentionParams) -> torch.Tensor:
        """
        Apply sequential token refinement.

        Args:
            x: Input token embeddings [B, L, C].
            c: Combined conditioning vector [B, C].
            attn_params: Attention parameters including sequence lengths.

        Returns:
            Refined token embeddings [B, L, C].
        """
        for block in self.blocks:
            x = block(x, c, attn_params)
        return x


class SingleTokenRefiner(nn.Module):
    """
    Text embedding refinement with timestep and context conditioning.

    Projects input text embeddings and applies self-attention refinement
    conditioned on diffusion timestep and aggregate text context.

    Args:
        in_channels: Input text embedding dimension.
        hidden_size: Transformer hidden dimension.
        heads_num: Number of attention heads.
        depth: Number of refinement blocks.
    """

    def __init__(self, in_channels: int, hidden_size: int, heads_num: int, depth: int):
        # Fixed architecture parameters for HunyuanImage-2.1
        mlp_drop_rate: float = 0.0  # No MLP dropout
        act_type: str = "silu"  # SiLU activation
        mlp_width_ratio: float = 4.0  # 4x MLP expansion
        qk_norm: bool = False  # No QK normalization
        qk_norm_type: str = "layer"  # Layer norm type (unused)
        qkv_bias: bool = True  # Use QKV bias

        super().__init__()
        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        act_layer = nn.SiLU
        self.t_embedder = TimestepEmbedder(hidden_size, act_layer)
        self.c_embedder = TextProjection(in_channels, hidden_size, act_layer)
        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
        )

    def forward(self, x: torch.Tensor, t: torch.LongTensor, attn_params: AttentionParams) -> torch.Tensor:
        """
        Refine text embeddings with timestep conditioning.

        Args:
            x: Input text embeddings [B, L, in_channels].
            t: Diffusion timestep [B].
            attn_params: Attention parameters including sequence lengths.

        Returns:
            Refined embeddings [B, L, hidden_size].
        """
        timestep_aware_representations = self.t_embedder(t)

        # Compute context-aware representations by averaging valid tokens
        txt_lens = attn_params.seqlens  # img_len is not used for SingleTokenRefiner
        context_aware_representations = torch.stack([x[i, : txt_lens[i]].mean(dim=0) for i in range(x.shape[0])], dim=0)  # [B, C]

        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations
        del timestep_aware_representations, context_aware_representations
        x = self.input_embedder(x)
        x = self.individual_token_refiner(x, c, attn_params)
        return x


class FinalLayer(nn.Module):
    """
    Final output projection layer with adaptive layer normalization.

    Projects transformer hidden states to output patch space with
    timestep-conditioned modulation.

    Args:
        hidden_size: Input hidden dimension.
        patch_size: Spatial patch size for output reshaping.
        out_channels: Number of output channels.
        act_layer: Activation function class.
    """

    def __init__(self, hidden_size, patch_size, out_channels, act_layer):
        super().__init__()

        # Layer normalization without learnable parameters
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        out_size = (patch_size[0] * patch_size[1]) * out_channels
        self.linear = nn.Linear(hidden_size, out_size, bias=True)

        # Adaptive layer normalization modulation
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        del shift, scale, c
        x = self.linear(x)
        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes input using RMS and applies learnable scaling.
    More efficient than LayerNorm as it doesn't compute mean.

    Args:
        dim: Input feature dimension.
        eps: Small value for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply RMS normalization.

        Args:
            x: Input tensor.

        Returns:
            RMS normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def reset_parameters(self):
        self.weight.fill_(1)

    def forward(self, x):
        """
        Apply RMSNorm with learnable scaling.

        Args:
            x: Input tensor.

        Returns:
            Normalized and scaled tensor.
        """
        output = self._norm(x.float()).type_as(x)
        del x
        # output = output * self.weight
        # fp8 support
        output = output * self.weight.to(output.dtype)
        return output


# kept for reference, not used in current implementation
# class LinearWarpforSingle(nn.Module):
#     """
#     Linear layer wrapper for concatenating and projecting two inputs.

#     Used in single-stream blocks to combine attention output with MLP features.

#     Args:
#         in_dim: Input dimension (sum of both input feature dimensions).
#         out_dim: Output dimension.
#         bias: Whether to use bias in linear projection.
#     """

#     def __init__(self, in_dim: int, out_dim: int, bias=False):
#         super().__init__()
#         self.fc = nn.Linear(in_dim, out_dim, bias=bias)

#     def forward(self, x, y):
#         """Concatenate inputs along feature dimension and project."""
#         x = torch.cat([x.contiguous(), y.contiguous()], dim=2).contiguous()
#         return self.fc(x)


class ModulateDiT(nn.Module):
    """
    Timestep conditioning modulation layer.

    Projects timestep embeddings to multiple modulation parameters
    for adaptive layer normalization.

    Args:
        hidden_size: Input conditioning dimension.
        factor: Number of modulation parameters to generate.
        act_layer: Activation function class.
    """

    def __init__(self, hidden_size: int, factor: int, act_layer: Callable):
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(hidden_size, factor * hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x))


class MMDoubleStreamBlock(nn.Module):
    """
    Multimodal double-stream transformer block.

    Processes image and text tokens separately with cross-modal attention.
    Each stream has its own normalization and MLP layers but shares
    attention computation for cross-modal interaction.

    Args:
        hidden_size: Model dimension.
        heads_num: Number of attention heads.
        mlp_width_ratio: MLP expansion ratio.
        mlp_act_type: MLP activation function (only "gelu_tanh" supported).
        qk_norm: QK normalization flag (must be True).
        qk_norm_type: QK normalization type (only "rms" supported).
        qkv_bias: Use bias in QKV projections.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
    ):
        super().__init__()

        assert mlp_act_type == "gelu_tanh", "Only GELU-tanh activation supported."
        assert qk_norm_type == "rms", "Only RMS normalization supported."
        assert qk_norm, "QK normalization must be enabled."

        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        # Image stream processing components
        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=nn.SiLU)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.img_attn_q_norm = RMSNorm(head_dim, eps=1e-6)
        self.img_attn_k_norm = RMSNorm(head_dim, eps=1e-6)
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"), bias=True)

        # Text stream processing components
        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=nn.SiLU)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.txt_attn_q_norm = RMSNorm(head_dim, eps=1e-6)
        self.txt_attn_k_norm = RMSNorm(head_dim, eps=1e-6)
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"), bias=True)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _forward(
        self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, freqs_cis: tuple = None, attn_params: AttentionParams = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract modulation parameters for image and text streams
        (img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate) = self.img_mod(vec).chunk(
            6, dim=-1
        )
        (txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.txt_mod(vec).chunk(
            6, dim=-1
        )

        # Process image stream for attention
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
        del img_mod1_shift, img_mod1_scale

        img_qkv = self.img_attn_qkv(img_modulated)
        del img_modulated
        img_q, img_k, img_v = img_qkv.chunk(3, dim=-1)
        del img_qkv

        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)

        # Apply QK-Norm if enabled
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply rotary position embeddings to image tokens
        if freqs_cis is not None:
            img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            del freqs_cis

        # Process text stream for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)

        txt_qkv = self.txt_attn_qkv(txt_modulated)
        del txt_modulated
        txt_q, txt_k, txt_v = txt_qkv.chunk(3, dim=-1)
        del txt_qkv

        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)

        # Apply QK-Norm if enabled
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Concatenate image and text tokens for joint attention
        img_seq_len = img.shape[1]
        q = torch.cat([img_q, txt_q], dim=1)
        del img_q, txt_q
        k = torch.cat([img_k, txt_k], dim=1)
        del img_k, txt_k
        v = torch.cat([img_v, txt_v], dim=1)
        del img_v, txt_v

        qkv = [q, k, v]
        del q, k, v
        attn = attention(qkv, attn_params=attn_params)
        del qkv

        # Split attention outputs back to separate streams
        img_attn, txt_attn = (attn[:, :img_seq_len].contiguous(), attn[:, img_seq_len:].contiguous())
        del attn

        # Apply attention projection and residual connection for image stream
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        del img_attn, img_mod1_gate

        # Apply MLP and residual connection for image stream
        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)),
            gate=img_mod2_gate,
        )
        del img_mod2_shift, img_mod2_scale, img_mod2_gate

        # Apply attention projection and residual connection for text stream
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        del txt_attn, txt_mod1_gate

        # Apply MLP and residual connection for text stream
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )
        del txt_mod2_shift, txt_mod2_scale, txt_mod2_gate

        return img, txt

    def forward(
        self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, freqs_cis: tuple = None, attn_params: AttentionParams = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.gradient_checkpointing and self.training:
            forward_fn = self._forward
            if self.cpu_offload_checkpointing:
                forward_fn = custom_offloading_utils.cpu_offload_wrapper(forward_fn, self.img_attn_qkv.weight.device)

            return torch.utils.checkpoint.checkpoint(forward_fn, img, txt, vec, freqs_cis, attn_params, use_reentrant=False)
        else:
            return self._forward(img, txt, vec, freqs_cis, attn_params)


class MMSingleStreamBlock(nn.Module):
    """
    Multimodal single-stream transformer block.

    Processes concatenated image and text tokens jointly with shared attention.
    Uses parallel linear layers for efficiency and applies RoPE only to image tokens.

    Args:
        hidden_size: Model dimension.
        heads_num: Number of attention heads.
        mlp_width_ratio: MLP expansion ratio.
        mlp_act_type: MLP activation function (only "gelu_tanh" supported).
        qk_norm: QK normalization flag (must be True).
        qk_norm_type: QK normalization type (only "rms" supported).
        qk_scale: Attention scaling factor (computed automatically if None).
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
    ):
        super().__init__()

        assert mlp_act_type == "gelu_tanh", "Only GELU-tanh activation supported."
        assert qk_norm_type == "rms", "Only RMS normalization supported."
        assert qk_norm, "QK normalization must be enabled."

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        # Parallel linear projections for efficiency
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim)

        # Combined output projection
        # self.linear2 = LinearWarpforSingle(hidden_size + mlp_hidden_dim, hidden_size, bias=True) # for reference
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size, bias=True)

        # QK normalization layers
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=nn.SiLU)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        attn_params: AttentionParams = None,
    ) -> torch.Tensor:
        # Extract modulation parameters
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)

        # Compute Q, K, V, and MLP input
        qkv_mlp = self.linear1(x_mod)
        del x_mod
        q, k, v, mlp = qkv_mlp.split([self.hidden_size, self.hidden_size, self.hidden_size, self.mlp_hidden_dim], dim=-1)
        del qkv_mlp

        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)

        # Apply QK-Norm if enabled
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Separate image and text tokens
        img_q, txt_q = q[:, : attn_params.img_len, :, :], q[:, attn_params.img_len :, :, :]
        del q
        img_k, txt_k = k[:, : attn_params.img_len, :, :], k[:, attn_params.img_len :, :, :]
        del k

        # Apply rotary position embeddings only to image tokens
        img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        del freqs_cis

        # Recombine and compute joint attention
        q = torch.cat([img_q, txt_q], dim=1)
        del img_q, txt_q
        k = torch.cat([img_k, txt_k], dim=1)
        del img_k, txt_k
        # v = torch.cat([img_v, txt_v], dim=1)
        # del img_v, txt_v
        qkv = [q, k, v]
        del q, k, v
        attn = attention(qkv, attn_params=attn_params)
        del qkv

        # Combine attention and MLP outputs, apply gating
        # output = self.linear2(attn, self.mlp_act(mlp))

        mlp = self.mlp_act(mlp)
        output = torch.cat([attn, mlp], dim=2).contiguous()
        del attn, mlp
        output = self.linear2(output)

        return x + apply_gate(output, gate=mod_gate)

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        attn_params: AttentionParams = None,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            forward_fn = self._forward
            if self.cpu_offload_checkpointing:
                forward_fn = custom_offloading_utils.create_cpu_offloading_wrapper(forward_fn, self.linear1.weight.device)

            return torch.utils.checkpoint.checkpoint(forward_fn, x, vec, freqs_cis, attn_params, use_reentrant=False)
        else:
            return self._forward(x, vec, freqs_cis, attn_params)


# endregion
