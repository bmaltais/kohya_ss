# Copyright Alpha VLLM/Lumina Image 2.0 and contributors
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F

from library import custom_offloading_utils

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    # flash_attn may not be available but it is not required
    pass

try:
    from sageattention import sageattn
except:
    pass

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except:
    import warnings

    warnings.warn("Cannot import apex RMSNorm, switch to vanilla implementation")

    #############################################################################
    #                                 RMSNorm                                   #
    #############################################################################

    class RMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            """
            Initialize the RMSNorm normalization layer.

            Args:
                dim (int): The dimension of the input tensor.
                eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

            Attributes:
                eps (float): A small value added to the denominator for numerical stability.
                weight (nn.Parameter): Learnable scaling parameter.

            """
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def _norm(self, x) -> Tensor:
            """
            Apply the RMSNorm normalization to the input tensor.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The normalized tensor.

            """
            return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x: Tensor):
            """
            Apply RMSNorm to the input tensor.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The normalized tensor.
            """
            x_dtype = x.dtype
            # To handle float8 we need to convert the tensor to float
            x = x.float()
            rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
            return ((x * rrms) * self.weight.float()).to(dtype=x_dtype)



@dataclass
class LuminaParams:
    """Parameters for Lumina model configuration"""

    patch_size: int = 2
    in_channels: int = 4
    dim: int = 4096
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 24
    n_kv_heads: int = 8
    multiple_of: int = 256
    axes_dims: List[int] = None
    axes_lens: List[int] = None
    qk_norm: bool = False
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    scaling_factor: float = 1.0
    cap_feat_dim: int = 32

    def __post_init__(self):
        if self.axes_dims is None:
            self.axes_dims = [36, 36, 36]
        if self.axes_lens is None:
            self.axes_lens = [300, 512, 512]

    @classmethod
    def get_2b_config(cls) -> "LuminaParams":
        """Returns the configuration for the 2B parameter model"""
        return cls(
            patch_size=2,
            in_channels=16,  # VAE channels
            dim=2304,
            n_layers=26,
            n_heads=24,
            n_kv_heads=8,
            axes_dims=[32, 32, 32],
            axes_lens=[300, 512, 512],
            qk_norm=True,
            cap_feat_dim=2304,  # Gemma 2 hidden_size
        )

    @classmethod
    def get_7b_config(cls) -> "LuminaParams":
        """Returns the configuration for the 7B parameter model"""
        return cls(
            patch_size=2,
            dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            axes_dims=[64, 64, 64],
            axes_lens=[300, 512, 512],
        )


class GradientCheckpointMixin(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = False

    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        else:
            return self._forward(*args, **kwargs)



def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


#############################################################################
#             Embedding Layers for Timesteps and Class Labels               #
#############################################################################


class TimestepEmbedder(GradientCheckpointMixin):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                hidden_size,
                bias=True,
            ),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def _forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


def to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, (list, tuple)):
        return [to_cuda(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, (list, tuple)):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


#############################################################################
#                               Core NextDiT Model                              #
#############################################################################


class JointAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        qk_norm: bool,
        use_flash_attn=False,
        use_sage_attn=False,
    ):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.
            qk_norm (bool): Whether to use normalization for queries and keys.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(
            dim,
            (n_heads + self.n_kv_heads + self.n_kv_heads) * self.head_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.qkv.weight)

        self.out = nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.out.weight)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        self.use_flash_attn = use_flash_attn
        self.use_sage_attn = use_sage_attn

        if use_sage_attn :
            self.attention_processor = self.sage_attn
        else:
            # self.attention_processor = xformers.ops.memory_efficient_attention
            self.attention_processor = F.scaled_dot_product_attention

    def set_attention_processor(self, attention_processor):
        self.attention_processor = attention_processor

    def get_attention_processor(self):
        return self.attention_processor

    def forward(
        self,
        x: Tensor,
        x_mask: Tensor,
        freqs_cis: Tensor,
    ) -> Tensor:
        """
        Args:
            x:
            x_mask:
            freqs_cis:
        """
        bsz, seqlen, _ = x.shape
        dtype = x.dtype

        xq, xk, xv = torch.split(
            self.qkv(x),
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xq = apply_rope(xq, freqs_cis=freqs_cis)
        xk = apply_rope(xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        softmax_scale = math.sqrt(1 / self.head_dim)

        if self.use_sage_attn:
            # Handle GQA (Grouped Query Attention) if needed
            n_rep = self.n_local_heads // self.n_local_kv_heads
            if n_rep >= 1:
                xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

            output = self.sage_attn(xq, xk, xv, x_mask, softmax_scale)
        elif self.use_flash_attn:
            output = self.flash_attn(xq, xk, xv, x_mask, softmax_scale)
        else:
            n_rep = self.n_local_heads // self.n_local_kv_heads
            if n_rep >= 1:
                xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

            output = (
                self.attention_processor(
                    xq.permute(0, 2, 1, 3),
                    xk.permute(0, 2, 1, 3),
                    xv.permute(0, 2, 1, 3),
                    attn_mask=x_mask.bool().view(bsz, 1, 1, seqlen).expand(-1, self.n_local_heads, seqlen, -1),
                    scale=softmax_scale,
                )
                .permute(0, 2, 1, 3)
                .to(dtype)
            )

        output = output.flatten(-2)
        return self.out(output)

    # copied from huggingface modeling_llama.py
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            return (
                indices,
                cu_seqlens,
                max_seqlen_in_batch,
            )

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.n_local_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def sage_attn(self, q: Tensor, k: Tensor, v: Tensor, x_mask: Tensor, softmax_scale: float):
        try:
            bsz = q.shape[0]
            seqlen = q.shape[1]

            # Transpose tensors to match SageAttention's expected format (HND layout)
            q_transposed = q.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
            k_transposed = k.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
            v_transposed = v.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
            
            # Handle masking for SageAttention
            # We need to filter out masked positions - this approach handles variable sequence lengths
            outputs = []
            for b in range(bsz):
                # Find valid token positions from the mask
                valid_indices = torch.nonzero(x_mask[b], as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    # If all tokens are masked, create a zero output
                    batch_output = torch.zeros(
                        seqlen, self.n_local_heads, self.head_dim, 
                        device=q.device, dtype=q.dtype
                    )
                else:
                    # Extract only valid tokens for this batch
                    batch_q = q_transposed[b, :, valid_indices, :]
                    batch_k = k_transposed[b, :, valid_indices, :]
                    batch_v = v_transposed[b, :, valid_indices, :]
                    
                    # Run SageAttention on valid tokens only
                    batch_output_valid = sageattn(
                        batch_q.unsqueeze(0),  # Add batch dimension back
                        batch_k.unsqueeze(0), 
                        batch_v.unsqueeze(0), 
                        tensor_layout="HND",
                        is_causal=False,
                        sm_scale=softmax_scale
                    )
                    
                    # Create output tensor with zeros for masked positions
                    batch_output = torch.zeros(
                        seqlen, self.n_local_heads, self.head_dim, 
                        device=q.device, dtype=q.dtype
                    )
                    # Place valid outputs back in the right positions
                    batch_output[valid_indices] = batch_output_valid.squeeze(0).permute(1, 0, 2)
                    
                outputs.append(batch_output)
            
            # Stack batch outputs and reshape to expected format
            output = torch.stack(outputs, dim=0)  # [batch, seq_len, heads, head_dim]
        except NameError as e:
            raise RuntimeError(
                f"Could not load Sage Attention. Please install https://github.com/thu-ml/SageAttention. / Sage Attention を読み込めませんでした。https://github.com/thu-ml/SageAttention をインストールしてください。 / {e}"
            )

        return output

    def flash_attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        x_mask: Tensor,
        softmax_scale,
    ) -> Tensor:
        bsz, seqlen, _, _ = q.shape

        try:
            # begin var_len flash attn
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(q, k, v, x_mask, seqlen)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=0.0,
                causal=False,
                softmax_scale=softmax_scale,
            )
            output = pad_input(attn_output_unpad, indices_q, bsz, seqlen)
            # end var_len_flash_attn

            return output
        except NameError as e:
            raise RuntimeError(
                f"Could not load flash attention. Please install flash_attn. / フラッシュアテンションを読み込めませんでした。flash_attn をインストールしてください。 / {e}"
            )


def apply_rope(
    x_in: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency
    tensor.

    This function applies rotary embeddings to the given query 'xq' and
    key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
    input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors
    contain rotary embeddings and are returned as real tensors.

    Args:
        x_in (torch.Tensor): Query or Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
            exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor
            and key tensor with rotary embeddings.
    """
    with torch.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)

    return x_out.type_as(x_in)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension. Defaults to None.

        """
        super().__init__()
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.w1.weight)
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.w2.weight)
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.w3.weight)

    # @torch.compile
    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class JointTransformerBlock(GradientCheckpointMixin):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        use_flash_attn=False,
        use_sage_attn=False,
    ) -> None:
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            dim (int): Embedding dimension of the input features.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int): Number of multiple of the hidden dimension.
            ffn_dim_multiplier (Optional[float]): Dimension multiplier for the
                feedforward layer.
            norm_eps (float): Epsilon value for normalization.
            qk_norm (bool): Whether to use normalization for queries and keys.
            modulation (bool): Whether to use modulation for the attention
                layer.
        """
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm, use_flash_attn=use_flash_attn, use_sage_attn=use_sage_attn)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    min(dim, 1024),
                    4 * dim,
                    bias=True,
                ),
            )
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

    def _forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        pe: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (Tensor): Input tensor.
            pe (Tensor): Rope position embedding.

        Returns:
            Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(
                    modulate(self.attention_norm1(x), scale_msa),
                    x_mask,
                    pe,
                )
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(
                    modulate(self.ffn_norm1(x), scale_mlp),
                )
            )
        else:
            assert adaln_input is None
            x = x + self.attention_norm2(
                self.attention(
                    self.attention_norm1(x),
                    x_mask,
                    pe,
                )
            )
            x = x + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )
            )
        return x


class FinalLayer(GradientCheckpointMixin):
    """
    The final layer of NextDiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        """
        Initialize the FinalLayer.

        Args:
            hidden_size (int): Hidden size of the input features.
            patch_size (int): Patch size of the input features.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(hidden_size, 1024),
                hidden_size,
                bias=True,
            ),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 10000.0,
        axes_dims: List[int] = [16, 56, 56],
        axes_lens: List[int] = [1, 512, 512],
    ):
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis = NextDiT.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)

    def __call__(self, ids: torch.Tensor):
        device = ids.device
        self.freqs_cis = [freqs_cis.to(ids.device) for freqs_cis in self.freqs_cis]
        result = []
        for i in range(len(self.axes_dims)):
            freqs = self.freqs_cis[i].to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            result.append(torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
        return torch.cat(result, dim=-1)


class NextDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_refiner_layers: int = 2,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        axes_dims: List[int] = [16, 56, 56],
        axes_lens: List[int] = [1, 512, 512],
        use_flash_attn=False,
        use_sage_attn=False,
    ) -> None:
        """
        Initialize the NextDiT model.

        Args:
            patch_size (int): Patch size of the input features.
            in_channels (int): Number of input channels.
            dim (int): Hidden size of the input features.
            n_layers (int): Number of Transformer layers.
            n_refiner_layers (int): Number of refiner layers.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int): Multiple of the hidden size.
            ffn_dim_multiplier (Optional[float]): Dimension multiplier for the
                feedforward layer.
            norm_eps (float): Epsilon value for normalization.
            qk_norm (bool): Whether to use query key normalization.
            cap_feat_dim (int): Dimension of the caption features.
            axes_dims (List[int]): List of dimensions for the axes.
            axes_lens (List[int]): List of lengths for the axes.
            use_flash_attn (bool): Whether to use Flash Attention.
            use_sage_attn (bool): Whether to use Sage Attention. Sage Attention only supports inference.

        Returns:
            None
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(
                cap_feat_dim,
                dim,
                bias=True,
            ),
        )

        nn.init.trunc_normal_(self.cap_embedder[1].weight, std=0.02)
        nn.init.zeros_(self.cap_embedder[1].bias)

        self.context_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
        )
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0.0)

        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )


        self.layers = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    use_flash_attn=use_flash_attn,
                    use_sage_attn=use_sage_attn,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.norm_final = RMSNorm(dim, eps=norm_eps)
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = RopeEmbedder(axes_dims=axes_dims, axes_lens=axes_lens)
        self.dim = dim
        self.n_heads = n_heads

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False # TODO: not yet supported
        self.blocks_to_swap = None # TODO: not yet supported

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

        self.t_embedder.enable_gradient_checkpointing()

        for block in self.layers + self.context_refiner + self.noise_refiner:
            block.enable_gradient_checkpointing(cpu_offload=cpu_offload)

        self.final_layer.enable_gradient_checkpointing()

        print(f"Lumina: Gradient checkpointing enabled. CPU offload: {cpu_offload}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

        self.t_embedder.disable_gradient_checkpointing()

        for block in self.layers + self.context_refiner + self.noise_refiner:
            block.disable_gradient_checkpointing()

        self.final_layer.disable_gradient_checkpointing()

        print("Lumina: Gradient checkpointing disabled.")

    def unpatchify(
        self,
        x: Tensor,
        width: int,
        height: int,
        encoder_seq_lengths: List[int],
        seq_lengths: List[int],
    ) -> Tensor:
        """
        Unpatchify the input tensor and embed the caption features.
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)

        Args:
            x (Tensor): Input tensor.
            width (int): Width of the input tensor.
            height (int): Height of the input tensor.
            encoder_seq_lengths (List[int]): List of encoder sequence lengths.
            seq_lengths (List[int]): List of sequence lengths

        Returns:
            output: (N, C, H, W)
        """
        pH = pW = self.patch_size

        output = []
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            output.append(
                x[i][encoder_seq_len:seq_len]
                .view(height // pH, width // pW, pH, pW, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )
        output = torch.stack(output, dim=0)

        return output

    def patchify_and_embed(
        self,
        x: Tensor,
        cap_feats: Tensor,
        cap_mask: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, List[int], List[int]]:
        """
        Patchify and embed the input image and caption features.

        Args:
            x: (N, C, H, W) image latents
            cap_feats: (N, C, D) caption features
            cap_mask: (N, C, D) caption attention mask
            t: (N), T timesteps

        Returns:
            Tuple[Tensor, Tensor, Tensor, List[int], List[int]]:

            return x, attention_mask, freqs_cis, l_effective_cap_len, seq_lengths
        """
        bsz, channels, height, width = x.shape
        pH = pW = self.patch_size
        device = x.device

        l_effective_cap_len = cap_mask.sum(dim=1).tolist()
        encoder_seq_len = cap_mask.shape[1]
        image_seq_len = (height // self.patch_size) * (width // self.patch_size)

        seq_lengths = [cap_seq_len + image_seq_len for cap_seq_len in l_effective_cap_len]
        max_seq_len = max(seq_lengths)

        position_ids = torch.zeros(bsz, max_seq_len, 3, dtype=torch.int32, device=device)

        for i, (cap_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            H_tokens, W_tokens = height // pH, width // pW

            position_ids[i, :cap_len, 0] = torch.arange(cap_len, dtype=torch.int32, device=device)
            position_ids[i, cap_len:seq_len, 0] = cap_len

            row_ids = torch.arange(H_tokens, dtype=torch.int32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
            col_ids = torch.arange(W_tokens, dtype=torch.int32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()

            position_ids[i, cap_len:seq_len, 1] = row_ids
            position_ids[i, cap_len:seq_len, 2] = col_ids

        # Get combined rotary embeddings
        freqs_cis = self.rope_embedder(position_ids)

        # Create separate rotary embeddings for captions and images
        cap_freqs_cis = torch.zeros(
            bsz,
            encoder_seq_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )
        img_freqs_cis = torch.zeros(
            bsz,
            image_seq_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )

        for i, (cap_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            cap_freqs_cis[i, :cap_len] = freqs_cis[i, :cap_len]
            img_freqs_cis[i, :image_seq_len] = freqs_cis[i, cap_len:seq_len]

        # Refine caption context
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis)

        x = x.view(bsz, channels, height // pH, pH, width // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)

        x_mask = torch.zeros(bsz, image_seq_len, dtype=torch.bool, device=device)
        for i in range(bsz):
            x[i, :image_seq_len] = x[i]
            x_mask[i, :image_seq_len] = True

        x = self.x_embedder(x)

        # Refine image context
        for layer in self.noise_refiner:
            x = layer(x, x_mask, img_freqs_cis, t)

        joint_hidden_states = torch.zeros(bsz, max_seq_len, self.dim, device=device, dtype=x.dtype)
        attention_mask = torch.zeros(bsz, max_seq_len, dtype=torch.bool, device=device)
        for i, (cap_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            attention_mask[i, :seq_len] = True
            joint_hidden_states[i, :cap_len] = cap_feats[i, :cap_len]
            joint_hidden_states[i, cap_len:seq_len] = x[i]

        x = joint_hidden_states

        return x, attention_mask, freqs_cis, l_effective_cap_len, seq_lengths

    def forward(self, x: Tensor, t: Tensor, cap_feats: Tensor, cap_mask: Tensor) -> Tensor:
        """
        Forward pass of NextDiT.
        Args:
            x: (N, C, H, W) image latents
            t: (N,) tensor of diffusion timesteps
            cap_feats: (N, L, D) caption features
            cap_mask: (N, L) caption attention mask

        Returns:
            x: (N, C, H, W) denoised latents
        """
        _, _, height, width = x.shape  # B, C, H, W
        t = self.t_embedder(t)  # (N, D)
        cap_feats = self.cap_embedder(cap_feats)  # (N, L, D)  # todo check if able to batchify w.o. redundant compute

        x, mask, freqs_cis, l_effective_cap_len, seq_lengths = self.patchify_and_embed(x, cap_feats, cap_mask, t)

        if not self.blocks_to_swap:
            for layer in self.layers:
                x = layer(x, mask, freqs_cis, t)
        else:
            for block_idx, layer in enumerate(self.layers):
                self.offloader_main.wait_for_block(block_idx)
                
                x = layer(x, mask, freqs_cis, t)
                
                self.offloader_main.submit_move_blocks(self.layers, block_idx)

        x = self.final_layer(x, t)
        x = self.unpatchify(x, width, height, l_effective_cap_len, seq_lengths)

        return x

    def forward_with_cfg(
        self,
        x: Tensor,
        t: Tensor,
        cap_feats: Tensor,
        cap_mask: Tensor,
        cfg_scale: float,
        cfg_trunc: float = 0.25,
        renorm_cfg: float = 1.0,
    ):
        """
        Forward pass of NextDiT, but also batches the unconditional forward pass
        for classifier-free guidance.
        """
        # # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        if t[0] < cfg_trunc:
            combined = torch.cat([half, half], dim=0)  # [2, 16, 128, 128]
            assert (
                cap_mask.shape[0] == combined.shape[0]
            ), f"caption attention mask shape: {cap_mask.shape[0]} latents shape: {combined.shape[0]}"
            model_out = self.forward(x, t, cap_feats, cap_mask)  # [2, 16, 128, 128]
            # For exact reproducibility reasons, we apply classifier-free guidance on only
            # three channels by default. The standard approach to cfg applies it to all channels.
            # This can be done by uncommenting the following line and commenting-out the line following that.
            eps, rest = (
                model_out[:, : self.in_channels],
                model_out[:, self.in_channels :],
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            if float(renorm_cfg) > 0.0:
                ori_pos_norm = torch.linalg.vector_norm(cond_eps, dim=tuple(range(1, len(cond_eps.shape))), keepdim=True)
                max_new_norm = ori_pos_norm * float(renorm_cfg)
                new_pos_norm = torch.linalg.vector_norm(half_eps, dim=tuple(range(1, len(half_eps.shape))), keepdim=True)
                if new_pos_norm >= max_new_norm:
                    half_eps = half_eps * (max_new_norm / new_pos_norm)
        else:
            combined = half
            model_out = self.forward(
                combined,
                t[: len(x) // 2],
                cap_feats[: len(x) // 2],
                cap_mask[: len(x) // 2],
            )
            eps, rest = (
                model_out[:, : self.in_channels],
                model_out[:, self.in_channels :],
            )
            half_eps = eps

        output = torch.cat([half_eps, half_eps], dim=0)
        return output

    @staticmethod
    def precompute_freqs_cis(
        dim: List[int],
        end: List[int],
        theta: float = 10000.0,
    ) -> List[Tensor]:
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (list): Dimension of the frequency tensor.
            end (list): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            List[torch.Tensor]: Precomputed frequency tensor with complex
                exponentials.
        """
        freqs_cis = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        for i, (d, e) in enumerate(zip(dim, end)):
            pos = torch.arange(e, dtype=freqs_dtype, device="cpu")
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=freqs_dtype, device="cpu") / d))
            freqs = torch.outer(pos, freqs)
            freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs)  # [S, D/2]
            freqs_cis.append(freqs_cis_i)

        return freqs_cis

    def parameter_count(self) -> int:
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            for param in module.parameters(recurse=False):
                total_params += param.numel()
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

    def get_checkpointing_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

    def enable_block_swap(self, blocks_to_swap: int, device: torch.device):
        """
        Enable block swapping to reduce memory usage during inference.
        
        Args:
            num_blocks (int): Number of blocks to swap between CPU and device
            device (torch.device): Device to use for computation
        """
        self.blocks_to_swap = blocks_to_swap
        
        # Calculate how many blocks to swap from main layers
        
        assert blocks_to_swap <= len(self.layers) - 2, (
            f"Cannot swap more than {len(self.layers) - 2} main blocks. "
            f"Requested {blocks_to_swap} blocks."
        )
        
        self.offloader_main = custom_offloading_utils.ModelOffloader(
            self.layers, blocks_to_swap, device, debug=False
        )

    def move_to_device_except_swap_blocks(self, device: torch.device):
        """
        Move the model to the device except for blocks that will be swapped.
        This reduces temporary memory usage during model loading.
        
        Args:
            device (torch.device): Device to move the model to
        """
        if self.blocks_to_swap:
            save_layers = self.layers
            self.layers = nn.ModuleList([])
            
        self.to(device)
            
        if self.blocks_to_swap:
            self.layers = save_layers

    def prepare_block_swap_before_forward(self):
        """
        Prepare blocks for swapping before forward pass.
        """
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        
        self.offloader_main.prepare_block_devices_before_forward(self.layers)


#############################################################################
#                                 NextDiT Configs                               #
#############################################################################


def NextDiT_2B_GQA_patch2_Adaln_Refiner(params: Optional[LuminaParams] = None, **kwargs):
    if params is None:
        params = LuminaParams.get_2b_config()

    return NextDiT(
        patch_size=params.patch_size,
        in_channels=params.in_channels,
        dim=params.dim,
        n_layers=params.n_layers,
        n_heads=params.n_heads,
        n_kv_heads=params.n_kv_heads,
        axes_dims=params.axes_dims,
        axes_lens=params.axes_lens,
        qk_norm=params.qk_norm,
        ffn_dim_multiplier=params.ffn_dim_multiplier,
        norm_eps=params.norm_eps,
        cap_feat_dim=params.cap_feat_dim,
        **kwargs,
    )


def NextDiT_3B_GQA_patch2_Adaln_Refiner(**kwargs):
    return NextDiT(
        patch_size=2,
        dim=2592,
        n_layers=30,
        n_heads=24,
        n_kv_heads=8,
        axes_dims=[36, 36, 36],
        axes_lens=[300, 512, 512],
        **kwargs,
    )


def NextDiT_4B_GQA_patch2_Adaln_Refiner(**kwargs):
    return NextDiT(
        patch_size=2,
        dim=2880,
        n_layers=32,
        n_heads=24,
        n_kv_heads=8,
        axes_dims=[40, 40, 40],
        axes_lens=[300, 512, 512],
        **kwargs,
    )


def NextDiT_7B_GQA_patch2_Adaln_Refiner(**kwargs):
    return NextDiT(
        patch_size=2,
        dim=3840,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        axes_dims=[40, 40, 40],
        axes_lens=[300, 512, 512],
        **kwargs,
    )
