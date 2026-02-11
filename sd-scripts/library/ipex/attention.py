import os
import torch
from functools import cache, wraps

# pylint: disable=protected-access, missing-function-docstring, line-too-long

# ARC GPUs can't allocate more than 4GB to a single block so we slice the attention layers

sdpa_slice_trigger_rate = float(os.environ.get('IPEX_SDPA_SLICE_TRIGGER_RATE', 1))
attention_slice_rate = float(os.environ.get('IPEX_ATTENTION_SLICE_RATE', 0.5))

# Find something divisible with the input_tokens
@cache
def find_split_size(original_size, slice_block_size, slice_rate=2):
    split_size = original_size
    while True:
        if (split_size * slice_block_size) <= slice_rate and original_size % split_size == 0:
            return split_size
        split_size = split_size - 1
        if split_size <= 1:
            return 1
    return split_size


# Find slice sizes for SDPA
@cache
def find_sdpa_slice_sizes(query_shape, key_shape, query_element_size, slice_rate=2, trigger_rate=3):
    batch_size, attn_heads, query_len, _ = query_shape
    _, _, key_len, _ = key_shape

    slice_batch_size = attn_heads * (query_len * key_len) * query_element_size / 1024 / 1024 / 1024

    split_batch_size = batch_size
    split_head_size = attn_heads
    split_query_size = query_len

    do_batch_split = False
    do_head_split = False
    do_query_split = False

    if batch_size * slice_batch_size >= trigger_rate:
        do_batch_split = True
        split_batch_size = find_split_size(batch_size, slice_batch_size, slice_rate=slice_rate)

        if split_batch_size * slice_batch_size > slice_rate:
            slice_head_size = split_batch_size * (query_len * key_len) * query_element_size / 1024 / 1024 / 1024
            do_head_split = True
            split_head_size = find_split_size(attn_heads, slice_head_size, slice_rate=slice_rate)

            if split_head_size * slice_head_size > slice_rate:
                slice_query_size = split_batch_size * split_head_size * (key_len) * query_element_size / 1024 / 1024 / 1024
                do_query_split = True
                split_query_size = find_split_size(query_len, slice_query_size, slice_rate=slice_rate)

    return do_batch_split, do_head_split, do_query_split, split_batch_size, split_head_size, split_query_size


original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@wraps(torch.nn.functional.scaled_dot_product_attention)
def dynamic_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    if query.device.type != "xpu":
        return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
    is_unsqueezed = False
    if query.dim() == 3:
        query = query.unsqueeze(0)
        is_unsqueezed = True
        if key.dim() == 3:
            key = key.unsqueeze(0)
        if value.dim() == 3:
            value = value.unsqueeze(0)
    do_batch_split, do_head_split, do_query_split, split_batch_size, split_head_size, split_query_size = find_sdpa_slice_sizes(query.shape, key.shape, query.element_size(), slice_rate=attention_slice_rate, trigger_rate=sdpa_slice_trigger_rate)

    # Slice SDPA
    if do_batch_split:
        batch_size, attn_heads, query_len, _ = query.shape
        _, _, _, head_dim = value.shape
        hidden_states = torch.zeros((batch_size, attn_heads, query_len, head_dim), device=query.device, dtype=query.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.expand((query.shape[0], query.shape[1], query.shape[2], key.shape[-2]))
        for ib in range(batch_size // split_batch_size):
            start_idx = ib * split_batch_size
            end_idx = (ib + 1) * split_batch_size
            if do_head_split:
                for ih in range(attn_heads // split_head_size): # pylint: disable=invalid-name
                    start_idx_h = ih * split_head_size
                    end_idx_h = (ih + 1) * split_head_size
                    if do_query_split:
                        for iq in range(query_len // split_query_size): # pylint: disable=invalid-name
                            start_idx_q = iq * split_query_size
                            end_idx_q = (iq + 1) * split_query_size
                            hidden_states[start_idx:end_idx, start_idx_h:end_idx_h, start_idx_q:end_idx_q, :] = original_scaled_dot_product_attention(
                                query[start_idx:end_idx, start_idx_h:end_idx_h, start_idx_q:end_idx_q, :],
                                key[start_idx:end_idx, start_idx_h:end_idx_h, :, :],
                                value[start_idx:end_idx, start_idx_h:end_idx_h, :, :],
                                attn_mask=attn_mask[start_idx:end_idx, start_idx_h:end_idx_h, start_idx_q:end_idx_q, :] if attn_mask is not None else attn_mask,
                                dropout_p=dropout_p, is_causal=is_causal, **kwargs
                            )
                    else:
                        hidden_states[start_idx:end_idx, start_idx_h:end_idx_h, :, :] = original_scaled_dot_product_attention(
                            query[start_idx:end_idx, start_idx_h:end_idx_h, :, :],
                            key[start_idx:end_idx, start_idx_h:end_idx_h, :, :],
                            value[start_idx:end_idx, start_idx_h:end_idx_h, :, :],
                            attn_mask=attn_mask[start_idx:end_idx, start_idx_h:end_idx_h, :, :] if attn_mask is not None else attn_mask,
                            dropout_p=dropout_p, is_causal=is_causal, **kwargs
                        )
            else:
                hidden_states[start_idx:end_idx, :, :, :] = original_scaled_dot_product_attention(
                    query[start_idx:end_idx, :, :, :],
                    key[start_idx:end_idx, :, :, :],
                    value[start_idx:end_idx, :, :, :],
                    attn_mask=attn_mask[start_idx:end_idx, :, :, :] if attn_mask is not None else attn_mask,
                    dropout_p=dropout_p, is_causal=is_causal, **kwargs
                )
        torch.xpu.synchronize(query.device)
    else:
        hidden_states = original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
    if is_unsqueezed:
        hidden_states = hidden_states.squeeze(0)
    return hidden_states
