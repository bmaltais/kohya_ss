import argparse
import json
import os
import time
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from library import sai_model_spec, model_util, sdxl_model_util
import lora
from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

MIN_SV = 1e-6

def index_sv_cumulative(S, target):
    original_sum = float(torch.sum(S))
    cumulative_sums = torch.cumsum(S, dim=0) / original_sum
    index = int(torch.searchsorted(cumulative_sums, target)) + 1
    index = max(1, min(index, len(S) - 1))
    return index

def index_sv_fro(S, target):
    S_squared = S.pow(2)
    S_fro_sq = float(torch.sum(S_squared))
    sum_S_squared = torch.cumsum(S_squared, dim=0) / S_fro_sq
    index = int(torch.searchsorted(sum_S_squared, target**2)) + 1
    index = max(1, min(index, len(S) - 1))
    return index

def index_sv_ratio(S, target):
    max_sv = S[0]
    min_sv = max_sv / target
    index = int(torch.sum(S > min_sv).item())
    index = max(1, min(index, len(S) - 1))
    return index

def index_sv_knee_improved(S, MIN_SV_KNEE=1e-8): # MIN_SV_KNEE can be same as global MIN_SV or specific
    """
    Determine rank using the knee point detection method with normalization.
    Normalizes singular values and their indices to the [0,1] range
    to make the knee detection scale-invariant.
    """
    n = len(S)
    if n < 3:  # Need at least 3 points to detect a knee
        return 1

    # S is expected to be sorted in descending order.
    s_max, s_min = S[0], S[-1]

    # Handle flat or nearly flat singular value spectrum
    if s_max - s_min < MIN_SV_KNEE:
        # If all singular values are almost the same, a knee is not well-defined.
        # Returning 1 is a conservative choice for low rank.
        # Alternatively, n // 2 or n - 1 could be chosen depending on desired behavior.
        return 1

    # Normalize singular values to [0, 1]
    # s_normalized[0] will be 1, s_normalized[n-1] will be 0
    s_normalized = (S - s_min) / (s_max - s_min)

    # Normalize indices to [0, 1]
    # x_normalized[0] will be 0, x_normalized[n-1] will be 1
    x_normalized = torch.linspace(0, 1, n, device=S.device, dtype=S.dtype)

    # The line in normalized space connects (x_norm[0], s_norm[0]) to (x_norm[n-1], s_norm[n-1])
    # This is (0, 1) to (1, 0).
    # The equation of the line passing through (0,1) and (1,0) is x + y - 1 = 0.
    # So, A=1, B=1, C=-1 for the line equation Ax + By + C = 0.
    
    # Calculate the perpendicular distance from each point (x_normalized[i], s_normalized[i]) to this line.
    # Distance = |A*x_i + B*y_i + C| / sqrt(A^2 + B^2)
    # Distance = |1*x_normalized + 1*s_normalized - 1| / sqrt(1^2 + 1^2)
    # Distance = |x_normalized + s_normalized - 1| / sqrt(2)
    
    # The sqrt(2) denominator is constant and doesn't affect argmax, so it can be omitted for finding the index.
    distances = (x_normalized + s_normalized - 1).abs()
    
    # Find the 0-based index of the point with the maximum distance.
    knee_index_0based = torch.argmax(distances).item()
    
    # Convert 0-based index to 1-based rank.
    rank = knee_index_0based + 1
    
    # Clamp rank similar to original: must be > 0 and <= n-1 (typical for rank reduction)
    # If knee_index_0based is n-1 (last point), rank becomes n. min(n, n-1) results in n-1.
    rank = max(1, min(rank, n - 1))
    
    return rank

def index_sv_cumulative_knee(S, min_sv_threshold=1e-8):
    """
    Determine rank using the knee point detection method on the normalized cumulative sum of singular values.
    This method identifies a point where adding more singular values contributes diminishingly to the total sum.
    """
    n = len(S)
    if n < 3:  # Need at least 3 points to detect a knee
        return 1

    s_sum = torch.sum(S)
    # If all singular values are zero or very small, return rank 1.
    if s_sum < min_sv_threshold:
        return 1
    
    # Calculate cumulative sum of singular values, normalized by the total sum.
    # y_values[0] = S[0]/s_sum, ..., y_values[n-1] = 1.0
    y_values = torch.cumsum(S, dim=0) / s_sum

    # Normalize these y_values (cumulative sums) to the range [0,1] for knee detection.
    y_min, y_max = y_values[0], y_values[n-1] # y_max is typically 1.0

    # If the normalized cumulative sum curve is very flat (e.g., S[0] captures almost all energy),
    # it implies the first few components are dominant.
    if y_max - y_min < min_sv_threshold: # Using min_sv_threshold here as a sensitivity for flatness
        # This condition means (S[0] + ... + S[n-1]) - S[0] is small relative to sum(S) if n>1
        # Effectively, S[1:] components are negligible.
        return 1 

    # y_norm[0] = 0, y_norm[n-1] = 1 (represents the normalized cumulative sum from start to end)
    y_norm = (y_values - y_min) / (y_max - y_min)
    
    # x_values are indices, normalized to [0, 1]
    # x_norm[0] = 0, ..., x_norm[n-1] = 1
    x_norm = torch.linspace(0, 1, n, device=S.device, dtype=S.dtype)

    # The "knee" is the point on the curve (x_norm[i], y_norm[i]) that is farthest
    # from the line connecting the start and end of this normalized curve.
    # In this normalized space, the line connects (0,0) to (1,1).
    # The equation of this line is Y = X, or X - Y = 0.
    # The distance from a point (x_i, y_i) to the line X - Y = 0 is |x_i - y_i| / sqrt(1^2 + (-1)^2).
    # We can maximize |x_i - y_i| (or |y_i - x_i|) as sqrt(2) is a constant factor.
    distances = (y_norm - x_norm).abs() # y_norm is expected to be >= x_norm for a concave cumulative curve.

    # Find the 0-based index of the point with the maximum distance.
    knee_index_0based = torch.argmax(distances).item()
    
    # Convert 0-based index to 1-based rank.
    rank = knee_index_0based + 1
    
    # Clamp rank to be between 1 and n-1 (as n elements give n-1 possible ranks for truncation).
    # A rank of n means no truncation. n-1 is the highest sensible rank for reduction.
    rank = max(1, min(rank, n - 1))
    
    return rank

def index_sv_rel_decrease(S, tau=0.1):
    """Determine rank based on relative decrease threshold."""
    if len(S) < 2:
        return 1
    
    # Compute ratios of consecutive singular values
    ratios = S[1:] / S[:-1]
    
    # Find the smallest k where ratio < tau
    for k in range(len(ratios)):
        if ratios[k] < tau:
            return max(1, k + 1)  # k + 1 because we want rank after the drop
    
    # If no drop below tau, return max rank
    return min(len(S), len(S) - 1)

def save_to_file(file_name, model, state_dict, dtype, metadata=None):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].to(dtype)
    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(model, file_name, metadata)
    else:
        torch.save(model, file_name)

def svd(
    model_org=None,
    model_tuned=None,
    save_to=None,
    dim=4,
    v2=None,
    sdxl=None,
    conv_dim=None,
    v_parameterization=None,
    device=None,
    save_precision=None,
    clamp_quantile=0.99,
    min_diff=0.01,
    no_metadata=False,
    load_precision=None,
    load_original_model_to=None,
    load_tuned_model_to=None,
    dynamic_method=None,
    dynamic_param=None,
    verbose=False,
):
    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    assert not (v2 and sdxl), "v2 and sdxl cannot be specified at the same time"
    v_parameterization = v2 if v_parameterization is None else v_parameterization

    load_dtype = str_to_dtype(load_precision) if load_precision else None
    save_dtype = str_to_dtype(save_precision) if save_precision else torch.float
    work_device = "cpu"

    # Load models
    if not sdxl:
        logger.info(f"Loading original SD model: {model_org}")
        text_encoder_o, _, unet_o = model_util.load_models_from_stable_diffusion_checkpoint(v2, model_org)
        text_encoders_o = [text_encoder_o]
        if load_dtype:
            text_encoder_o.to(load_dtype)
            unet_o.to(load_dtype)

        logger.info(f"Loading tuned SD model: {model_tuned}")
        text_encoder_t, _, unet_t = model_util.load_models_from_stable_diffusion_checkpoint(v2, model_tuned)
        text_encoders_t = [text_encoder_t]
        if load_dtype:
            text_encoder_t.to(load_dtype)
            unet_t.to(load_dtype)

        model_version = model_util.get_model_version_str_for_sd1_sd2(v2, v_parameterization)
    else:
        device_org = load_original_model_to or "cpu"
        device_tuned = load_tuned_model_to or "cpu"

        logger.info(f"Loading original SDXL model: {model_org}")
        text_encoder_o1, text_encoder_o2, _, unet_o, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
            sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, model_org, device_org
        )
        text_encoders_o = [text_encoder_o1, text_encoder_o2]
        if load_dtype:
            text_encoder_o1.to(load_dtype)
            text_encoder_o2.to(load_dtype)
            unet_o.to(load_dtype)

        logger.info(f"Loading tuned SDXL model: {model_tuned}")
        text_encoder_t1, text_encoder_t2, _, unet_t, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
            sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, model_tuned, device_tuned
        )
        text_encoders_t = [text_encoder_t1, text_encoder_t2]
        if load_dtype:
            text_encoder_t1.to(load_dtype)
            text_encoder_t2.to(load_dtype)
            unet_t.to(load_dtype)

        model_version = sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0

    # Create LoRA network
    kwargs = {"conv_dim": conv_dim, "conv_alpha": conv_dim} if conv_dim else {}
    
    # Define a small initial dimension for memory efficiency
    init_dim = 4  # Small value to minimize memory usage

    # Create LoRA networks with minimal dimension
    lora_network_o = lora.create_network(1.0, init_dim, init_dim, None, text_encoders_o, unet_o, **kwargs)
    lora_network_t = lora.create_network(1.0, init_dim, init_dim, None, text_encoders_t, unet_t, **kwargs)
    
    assert len(lora_network_o.text_encoder_loras) == len(lora_network_t.text_encoder_loras), "Model versions differ (SD1.x vs SD2.x)"

    # Compute differences
    diffs = {}
    text_encoder_different = False
    for lora_o, lora_t in zip(lora_network_o.text_encoder_loras, lora_network_t.text_encoder_loras):
        lora_name = lora_o.lora_name
        diff = lora_t.org_module.weight.to(work_device) - lora_o.org_module.weight.to(work_device)
        lora_o.org_module.weight = None
        lora_t.org_module.weight = None

        if not text_encoder_different and torch.max(torch.abs(diff)) > min_diff:
            text_encoder_different = True
            logger.info(f"Text encoder differs: max diff {torch.max(torch.abs(diff))} > {min_diff}")
        diffs[lora_name] = diff

    for text_encoder in text_encoders_t:
        del text_encoder

    if not text_encoder_different:
        logger.warning("Text encoders are identical. Extracting U-Net only.")
        lora_network_o.text_encoder_loras = []
        diffs.clear()

    for lora_o, lora_t in zip(lora_network_o.unet_loras, lora_network_t.unet_loras):
        lora_name = lora_o.lora_name
        diff = lora_t.org_module.weight.to(work_device) - lora_o.org_module.weight.to(work_device)
        lora_o.org_module.weight = None
        lora_t.org_module.weight = None
        diffs[lora_name] = diff

    del lora_network_t, unet_t

    # Filter relevant modules
    lora_names = set(lora.lora_name for lora in lora_network_o.text_encoder_loras + lora_network_o.unet_loras)

    # Extract and resize LoRA using SVD
    logger.info("Extracting and resizing LoRA via SVD")
    lora_weights = {}
    with torch.no_grad():
        for lora_name in tqdm(lora_names):
            mat = diffs[lora_name]
            if device:
                mat = mat.to(device)
            mat = mat.to(torch.float)

            conv2d = len(mat.size()) == 4
            kernel_size = mat.size()[2:4] if conv2d else None
            conv2d_3x3 = conv2d and kernel_size != (1, 1)
            out_dim, in_dim = mat.size()[0:2]

            if conv2d:
                mat = mat.flatten(start_dim=1) if conv2d_3x3 else mat.squeeze()

            U, S, Vh = torch.linalg.svd(mat)

            # Determine rank
            max_rank = dim if not conv2d_3x3 or conv_dim is None else conv_dim
            if dynamic_method:
                if S[0] <= MIN_SV:
                    rank = 1
                elif dynamic_method == "sv_ratio":
                    rank = index_sv_ratio(S, dynamic_param)
                elif dynamic_method == "sv_cumulative":
                    rank = index_sv_cumulative(S, dynamic_param)
                elif dynamic_method == "sv_fro":
                    rank = index_sv_fro(S, dynamic_param)
                elif dynamic_method == "sv_knee":
                    rank = index_sv_knee_improved(S, MIN_SV) # Pass MIN_SV or a specific threshold
                elif dynamic_method == "sv_cumulative_knee": # New method
                    rank = index_sv_cumulative_knee(S, MIN_SV) # Pass MIN_SV or a specific threshold
                elif dynamic_method == "sv_rel_decrease":
                    rank = index_sv_rel_decrease(S, dynamic_param)
                rank = min(rank, max_rank, in_dim, out_dim)
            else:
                rank = min(max_rank, in_dim, out_dim)

            # Truncate SVD components
            U = U[:, :rank] @ torch.diag(S[:rank])
            Vh = Vh[:rank, :]

            # Clamp values
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp_quantile)
            U = U.clamp(-hi_val, hi_val)
            Vh = Vh.clamp(-hi_val, hi_val)

            if conv2d:
                U = U.reshape(out_dim, rank, 1, 1)
                Vh = Vh.reshape(rank, in_dim, *kernel_size)

            U = U.to(work_device, dtype=save_dtype).contiguous()
            Vh = Vh.to(work_device, dtype=save_dtype).contiguous()
            lora_weights[lora_name] = (U, Vh)

            # Verbose output
            if verbose:
                s_sum = float(torch.sum(S))
                s_rank = float(torch.sum(S[:rank]))
                fro = float(torch.sqrt(torch.sum(S.pow(2))))
                fro_rank = float(torch.sqrt(torch.sum(S[:rank].pow(2))))
                ratio = S[0] / S[rank - 1] if rank > 1 else float('inf')
                logger.info(f"{lora_name:75} | sum(S) retained: {s_rank/s_sum:.1%}, fro retained: {fro_rank/fro:.1%}, max ratio: {ratio:.1f}, rank: {rank}")

    # Create state dict
    lora_sd = {}
    for lora_name, (up_weight, down_weight) in lora_weights.items():
        lora_sd[lora_name + ".lora_up.weight"] = up_weight
        lora_sd[lora_name + ".lora_down.weight"] = down_weight
        lora_sd[lora_name + ".alpha"] = torch.tensor(down_weight.size()[0], dtype=save_dtype)

    # Load and save LoRA
    lora_network_save, lora_sd = lora.create_network_from_weights(1.0, None, None, text_encoders_o, unet_o, weights_sd=lora_sd)
    lora_network_save.apply_to(text_encoders_o, unet_o)
    info = lora_network_save.load_state_dict(lora_sd)
    logger.info(f"Loaded extracted and resized LoRA weights: {info}")

    os.makedirs(os.path.dirname(save_to), exist_ok=True)

    # Metadata
    net_kwargs = {"conv_dim": str(conv_dim), "conv_alpha": str(float(conv_dim))} if conv_dim else {}
    metadata = {
        "ss_v2": str(v2),
        "ss_base_model_version": model_version,
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(dim) if not dynamic_method else "Dynamic",
        "ss_network_alpha": str(float(dim)) if not dynamic_method else "Dynamic",
        "ss_network_args": json.dumps(net_kwargs),
    }
    if not no_metadata:
        title = os.path.splitext(os.path.basename(save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(None, v2, v_parameterization, sdxl, True, False, time.time(), title=title)
        metadata.update(sai_metadata)

    save_to_file(save_to, lora_sd, lora_sd, save_dtype, metadata)
    logger.info(f"LoRA saved to: {save_to}")

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="Load Stable Diffusion v2.x model")
    parser.add_argument("--v_parameterization", action="store_true", help="Set v-parameterization metadata (defaults to v2)")
    parser.add_argument("--sdxl", action="store_true", help="Load Stable Diffusion SDXL base model")
    parser.add_argument("--load_precision", choices=[None, "float", "fp16", "bf16"], help="Precision for loading models")
    parser.add_argument("--save_precision", choices=[None, "float", "fp16", "bf16"], default=None, help="Precision for saving LoRA")
    parser.add_argument("--model_org", required=True, help="Original Stable Diffusion model (ckpt/safetensors)")
    parser.add_argument("--model_tuned", required=True, help="Tuned Stable Diffusion model (ckpt/safetensors)")
    parser.add_argument("--save_to", required=True, help="Output file name (ckpt/safetensors)")
    parser.add_argument("--dim", type=int, default=4, help="Max dimension (rank) of LoRA for linear layers")
    parser.add_argument("--conv_dim", type=int, help="Max dimension (rank) of LoRA for Conv2d-3x3")
    parser.add_argument("--device", default="cuda", help="Device for computation (e.g., cuda)")
    parser.add_argument("--clamp_quantile", type=float, default=0.99, help="Quantile for clamping weights")
    parser.add_argument("--min_diff", type=float, default=0.01, help="Minimum weight difference to extract")
    parser.add_argument("--no_metadata", action="store_true", help="Omit detailed metadata")
    parser.add_argument("--load_original_model_to", help="Device for original model (SDXL only)")
    parser.add_argument("--load_tuned_model_to", help="Device for tuned model (SDXL only)")
    parser.add_argument("--dynamic_param", type=float, help="Parameter for dynamic rank reduction")
    parser.add_argument("--verbose", action="store_true", help="Show detailed rank reduction info")
    parser.add_argument(
        "--dynamic_method", 
        choices=[None, "sv_ratio", "sv_fro", "sv_cumulative", "sv_knee", "sv_rel_decrease", "sv_cumulative_knee"],  # Added "sv_cumulative_knee"
        help="Dynamic rank reduction method"
    )
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if args.dynamic_method and not args.dynamic_param:
        raise ValueError("Dynamic method requires a dynamic parameter")
    svd(**vars(args))