import sys
import os

# 1. Add sd-scripts directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sd_scripts_dir_path = os.path.join(project_root, "sd-scripts")

if sd_scripts_dir_path not in sys.path:
    sys.path.insert(0, sd_scripts_dir_path)

# Now you can import from the library package and the networks package
try:
    # sai_model_spec REMOVED from here
    from library import model_util, sdxl_model_util
    from library.utils import setup_logging
    from networks import lora
except ImportError as e:
    print(f"Error importing from sd-scripts. Please check your sd-scripts folder structure.")
    print(f"Attempted to load from: {sd_scripts_dir_path}")
    print(f"Original error: {e}")
    print("Current sys.path relevant entries:")
    for p in sys.path:
        if "sd-scripts" in p or "kohya_ss" in p:
            print(p)
    raise

import argparse
import json
import time
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

setup_logging()
import logging
logger = logging.getLogger(__name__)

MIN_SV = 1e-6

# --- Singular Value Indexing Functions (Unchanged) ---
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

def index_sv_knee(S, MIN_SV_KNEE=1e-8):
    n = len(S)
    if n < 3: return 1
    s_max, s_min = S[0], S[-1]
    if s_max - s_min < MIN_SV_KNEE: return 1
    s_normalized = (S - s_min) / (s_max - s_min)
    x_normalized = torch.linspace(0, 1, n, device=S.device, dtype=S.dtype)
    distances = (x_normalized + s_normalized - 1).abs()
    knee_index_0based = torch.argmax(distances).item()
    rank = knee_index_0based + 1
    rank = max(1, min(rank, n - 1))
    return rank

def index_sv_cumulative_knee(S, min_sv_threshold=1e-8):
    n = len(S)
    if n < 3: return 1
    s_sum = torch.sum(S)
    if s_sum < min_sv_threshold: return 1
    y_values = torch.cumsum(S, dim=0) / s_sum
    y_min, y_max = y_values[0], y_values[n-1]
    if y_max - y_min < min_sv_threshold: return 1
    y_norm = (y_values - y_min) / (y_max - y_min)
    x_norm = torch.linspace(0, 1, n, device=S.device, dtype=S.dtype)
    distances = (y_norm - x_norm).abs()
    knee_index_0based = torch.argmax(distances).item()
    rank = knee_index_0based + 1
    rank = max(1, min(rank, n - 1))
    return rank

def index_sv_rel_decrease(S, tau=0.1):
    if len(S) < 2: return 1
    ratios = S[1:] / S[:-1]
    for k in range(len(ratios)):
        if ratios[k] < tau:
            return k + 1
    return len(S)

# --- Utility Functions ---
def _str_to_dtype(p):
    if p == "float": return torch.float
    if p == "fp16": return torch.float16
    if p == "bf16": return torch.bfloat16
    return None

def save_to_file(file_name, state_dict_to_save, dtype, metadata=None):
    # Make a copy to modify for dtype conversion if necessary
    state_dict_final = {}
    for key, value in state_dict_to_save.items():
        if isinstance(value, torch.Tensor) and dtype is not None:
            state_dict_final[key] = value.to(dtype)
        else:
            state_dict_final[key] = value # Handles non-tensors or when dtype is None

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(state_dict_final, file_name, metadata=metadata)
    else:
        torch.save(state_dict_final, file_name)

# --- NEW LOCAL METADATA FUNCTION ---
def _build_local_sai_metadata(title, creation_time, is_v2_flag, is_v_param_flag, is_sdxl_flag):
    """
    Creates a dictionary of SAI-like metadata based on the provided arguments,
    specifically for the context of this LoRA extraction script.
    Keys are aligned with common Civitai/SAI metadata practices.
    """
    metadata = {}
    metadata["ss_sd_model_name"] = str(title)
    metadata["ss_creation_time"] = str(int(creation_time)) # Original uses int timestamp

    # Determine base model version for SAI metadata
    if is_sdxl_flag:
        metadata["ss_base_model_version"] = "sdxl_v10" # Standard SAI identifier for SDXL 1.0
        metadata["ss_sdxl_model_version"] = "1.0"    # Specific SDXL version
        # In SAI context, SDXL is often implicitly v-parameterization,
        # but explicitly stating it if requested is good.
        if is_v_param_flag:
             metadata["ss_v_parameterization"] = "true"
    elif is_v2_flag:
        metadata["ss_base_model_version"] = "sd_v2" # Standard SAI identifier for SD 2.x
        if is_v_param_flag: # v-parameterization is key for some SD2.x models
            metadata["ss_v_parameterization"] = "true"
    else:
        metadata["ss_base_model_version"] = "sd_v1" # Standard SAI identifier for SD 1.x
        # v-parameterization is less common for SD1.x but can exist
        if is_v_param_flag:
            metadata["ss_v_parameterization"] = "true"
            
    # Other flags from the original sai_model_spec call were fixed (training_info=None, is_v_pred_like=False etc.)
    # or their effects are covered by the flags above.
    # We are only adding keys that are generally present and have meaningful values from the inputs.
    # Example: "ss_is_v_prediction_like": "false" could be added if is_v_pred_like was a param,
    # but it's false in the original call, so we omit it for brevity.
    return metadata

# --- Refactored Helper Functions ---
def _load_sd_model_components(model_path, is_v2, load_dtype_torch):
    logger.info(f"Loading SD model from: {model_path} (to CPU initially)")
    text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(is_v2, model_path)
    del vae
    text_encoders = [text_encoder]
    if load_dtype_torch:
        for te in text_encoders:
            te.to(load_dtype_torch)
        unet.to(load_dtype_torch)
    return text_encoders, unet

def _load_sdxl_model_components(model_path, target_device_override, load_dtype_torch):
    actual_load_device = target_device_override if target_device_override else "cpu"
    logger.info(f"Loading SDXL model from: {model_path} to device: {actual_load_device}")
    text_encoder1, text_encoder2, vae, unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
        sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, model_path, actual_load_device
    )
    del vae
    text_encoders = [text_encoder1, text_encoder2]
    if load_dtype_torch:
        for te in text_encoders:
            te.to(load_dtype_torch)
        unet.to(load_dtype_torch)
    return text_encoders, unet

def _calculate_module_diffs_and_check(module_loras_o, module_loras_t, diff_calc_device, min_diff_thresh, module_type_str):
    diffs_map = {}
    is_different_flag = False
    first_diff_logged = False

    for lora_o, lora_t in zip(module_loras_o, module_loras_t):
        lora_name = lora_o.lora_name
        if lora_o.org_module is None or lora_t.org_module is None or \
           not hasattr(lora_o.org_module, 'weight') or lora_o.org_module.weight is None or \
           not hasattr(lora_t.org_module, 'weight') or lora_t.org_module.weight is None:
            logger.warning(f"Skipping {lora_name} in {module_type_str} due to missing org_module or weight.")
            continue
        
        weight_o = lora_o.org_module.weight
        weight_t = lora_t.org_module.weight

        if str(weight_o.device) != str(diff_calc_device):
            weight_o = weight_o.to(diff_calc_device)
        if str(weight_t.device) != str(diff_calc_device):
            weight_t = weight_t.to(diff_calc_device)
            
        diff = weight_t - weight_o
        diffs_map[lora_name] = diff
        current_max_diff = torch.max(torch.abs(diff))
        if not is_different_flag and current_max_diff > min_diff_thresh:
            is_different_flag = True
            if not first_diff_logged:
                 logger.info(f"{module_type_str} '{lora_name}' differs: max diff {current_max_diff} > {min_diff_thresh}")
                 first_diff_logged = True
    return diffs_map, is_different_flag

def _determine_rank(S_values, dynamic_method_name, dynamic_param_value, max_rank_limit, 
                    module_eff_in_dim, module_eff_out_dim, min_sv_threshold=MIN_SV):
    if not S_values.numel() or S_values[0] <= min_sv_threshold: return 1
    rank = 0
    if dynamic_method_name == "sv_ratio": rank = index_sv_ratio(S_values, dynamic_param_value)
    elif dynamic_method_name == "sv_cumulative": rank = index_sv_cumulative(S_values, dynamic_param_value)
    elif dynamic_method_name == "sv_fro": rank = index_sv_fro(S_values, dynamic_param_value)
    elif dynamic_method_name == "sv_knee": rank = index_sv_knee(S_values, min_sv_threshold)
    elif dynamic_method_name == "sv_cumulative_knee": rank = index_sv_cumulative_knee(S_values, min_sv_threshold)
    elif dynamic_method_name == "sv_rel_decrease": rank = index_sv_rel_decrease(S_values, dynamic_param_value)
    else: rank = max_rank_limit 
    rank = min(rank, max_rank_limit, module_eff_in_dim, module_eff_out_dim, len(S_values))
    rank = max(1, rank)
    return rank

def _construct_lora_weights_from_svd_components(U_full, S_all_values, Vh_full, rank,
                                                clamp_quantile_val, is_conv2d, is_conv2d_3x3,
                                                conv_kernel_size,
                                                module_out_channels, module_in_channels,
                                                target_device_for_final_weights, target_dtype_for_final_weights):
    S_k = S_all_values[:rank]
    U_k = U_full[:, :rank]
    Vh_k = Vh_full[:rank, :]

    S_k_non_negative = torch.clamp(S_k, min=0.0)
    s_sqrt = torch.sqrt(S_k_non_negative)

    U_final = U_k * s_sqrt.unsqueeze(0)
    Vh_final = Vh_k * s_sqrt.unsqueeze(1)

    dist = torch.cat([U_final.flatten(), Vh_final.flatten()])
    hi_val = torch.quantile(dist, clamp_quantile_val)
    if hi_val == 0 and torch.max(torch.abs(dist)) > 1e-9:
         logger.debug(f"Clamping hi_val is zero for non-zero distribution. Max abs val: {torch.max(torch.abs(dist))}. Quantile: {clamp_quantile_val}")

    U_clamped = U_final.clamp(-hi_val, hi_val)
    Vh_clamped = Vh_final.clamp(-hi_val, hi_val)

    if is_conv2d:
        U_clamped = U_clamped.reshape(module_out_channels, rank, 1, 1)
        if is_conv2d_3x3:
            Vh_clamped = Vh_clamped.reshape(rank, module_in_channels, *conv_kernel_size)
        else: 
            Vh_clamped = Vh_clamped.reshape(rank, module_in_channels, 1, 1)

    U_clamped = U_clamped.to(target_device_for_final_weights, dtype=target_dtype_for_final_weights).contiguous()
    Vh_clamped = Vh_clamped.to(target_device_for_final_weights, dtype=target_dtype_for_final_weights).contiguous()
    return U_clamped, Vh_clamped

def _log_svd_stats(lora_module_name, S_all_values, rank_used, min_sv_for_calc=MIN_SV):
    if not S_all_values.numel():
        logger.info(f"{lora_module_name:75} | rank: {rank_used}, SVD not performed (empty singular values).")
        return
    
    S_cpu = S_all_values.to('cpu')
    s_sum_total = float(torch.sum(S_cpu))
    s_sum_rank = float(torch.sum(S_cpu[:rank_used]))
    fro_orig_total = float(torch.sqrt(torch.sum(S_cpu.pow(2))))
    fro_reconstructed_rank = float(torch.sqrt(torch.sum(S_cpu[:rank_used].pow(2))))
    ratio_sv = float('inf')
    if rank_used > 0 and S_cpu[rank_used - 1].abs() > min_sv_for_calc:
        ratio_sv = S_cpu[0] / S_cpu[rank_used - 1]
    
    sum_s_retained_percentage = (s_sum_rank / s_sum_total) if s_sum_total > min_sv_for_calc else 1.0
    fro_retained_percentage = (fro_reconstructed_rank / fro_orig_total) if fro_orig_total > min_sv_for_calc else 1.0

    logger.info(
        f"{lora_module_name:75} | rank: {rank_used}, "
        f"sum(S) retained: {sum_s_retained_percentage:.2%}, "
        f"Frobenius norm retained: {fro_retained_percentage:.2%}, "
        f"max_retained_sv/min_retained_sv ratio: {ratio_sv:.2f}"
    )

def _prepare_lora_metadata(output_path, is_v2_flag, kohya_base_model_version_str, network_conv_dim_val, 
                           use_dynamic_method_flag, network_dim_config_val, 
                           is_v_param_flag, is_sdxl_flag, skip_sai_meta):
    net_kwargs = {"conv_dim": str(network_conv_dim_val), "conv_alpha": str(float(network_conv_dim_val))} if network_conv_dim_val is not None else {}
    
    if use_dynamic_method_flag:
        network_dim_meta = "Dynamic"
        network_alpha_meta = "Dynamic" 
    else:
        network_dim_meta = str(network_dim_config_val)
        network_alpha_meta = str(float(network_dim_config_val))

    # Initial metadata using Kohya's conventions
    final_metadata = {
        "ss_v2": str(is_v2_flag), # Kohya's flag for v2 checkpoint type
        "ss_base_model_version": kohya_base_model_version_str, # Kohya's specific base model string
        "ss_network_module": "networks.lora",
        "ss_network_dim": network_dim_meta,
        "ss_network_alpha": network_alpha_meta,
        "ss_network_args": json.dumps(net_kwargs),
        "ss_lowram": "False", 
        "ss_num_train_images": "N/A", # This script doesn't involve training
    }

    if not skip_sai_meta:
        title = os.path.splitext(os.path.basename(output_path))[0]
        current_time = time.time()
        
        # Build SAI-like metadata using the local function
        sai_metadata_content = _build_local_sai_metadata(
            title=title,
            creation_time=current_time,
            is_v2_flag=is_v2_flag,       # Pass the script's v2 context
            is_v_param_flag=is_v_param_flag,
            is_sdxl_flag=is_sdxl_flag
        )
        # Update final_metadata. Keys from sai_metadata_content will overwrite
        # existing keys if they are the same (e.g., potentially 'ss_base_model_version').
        final_metadata.update(sai_metadata_content)
        
    return final_metadata

# --- Main SVD Function ---
def svd(
    model_org=None, model_tuned=None, save_to=None, dim=4, v2=None, sdxl=None,
    conv_dim=None, v_parameterization=None, device=None, save_precision=None,
    clamp_quantile=0.99, min_diff=0.01, no_metadata=False, load_precision=None,
    load_original_model_to=None, load_tuned_model_to=None,
    dynamic_method=None, dynamic_param=None, verbose=False,
):
    v_parameterization = v2 if v_parameterization is None else v_parameterization
    load_dtype_torch = _str_to_dtype(load_precision)
    save_dtype_torch = _str_to_dtype(save_precision) if save_precision else torch.float
    
    svd_computation_device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using SVD computation device: {svd_computation_device}")
    diff_calculation_device = torch.device("cpu")
    logger.info(f"Calculating weight differences on: {diff_calculation_device}")
    final_weights_device = torch.device("cpu")

    if not sdxl:
        text_encoders_o, unet_o = _load_sd_model_components(model_org, v2, load_dtype_torch)
        text_encoders_t, unet_t = _load_sd_model_components(model_tuned, v2, load_dtype_torch)
        # This is Kohya's specific model version string
        kohya_model_version = model_util.get_model_version_str_for_sd1_sd2(v2, v_parameterization)
    else:
        text_encoders_o, unet_o = _load_sdxl_model_components(model_org, load_original_model_to, load_dtype_torch)
        text_encoders_t, unet_t = _load_sdxl_model_components(model_tuned, load_tuned_model_to, load_dtype_torch)
        # This is Kohya's specific model version string for SDXL
        kohya_model_version = sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0

    init_dim_val = 1 
    lora_conv_dim_init = conv_dim if conv_dim is not None else init_dim_val
    kwargs_lora = {"conv_dim": lora_conv_dim_init, "conv_alpha": lora_conv_dim_init}

    lora_network_o = lora.create_network(1.0, init_dim_val, init_dim_val, None, text_encoders_o, unet_o, **kwargs_lora)
    lora_network_t = lora.create_network(1.0, init_dim_val, init_dim_val, None, text_encoders_t, unet_t, **kwargs_lora)
    
    assert len(lora_network_o.text_encoder_loras) == len(lora_network_t.text_encoder_loras), \
        f"Model versions differ: {len(lora_network_o.text_encoder_loras)} vs {len(lora_network_t.text_encoder_loras)} TEs"

    all_diffs = {}
    te_diffs, text_encoder_different = _calculate_module_diffs_and_check(
        lora_network_o.text_encoder_loras, lora_network_t.text_encoder_loras,
        diff_calculation_device, min_diff, "Text Encoder"
    )

    if text_encoder_different:
        all_diffs.update(te_diffs)
    else:
        logger.warning("Text encoders are considered identical based on min_diff. Not extracting TE LoRA.")
        lora_network_o.text_encoder_loras = [] 
    
    del text_encoders_t

    unet_diffs, _ = _calculate_module_diffs_and_check(
        lora_network_o.unet_loras, lora_network_t.unet_loras,
        diff_calculation_device, min_diff, "U-Net"
    )
    all_diffs.update(unet_diffs)
    del lora_network_t, unet_t

    lora_names_to_process = set(lora.lora_name for lora in lora_network_o.text_encoder_loras + lora_network_o.unet_loras)
    
    logger.info("Extracting and resizing LoRA via SVD")
    lora_weights = {}
    with torch.no_grad():
        for lora_name in tqdm(lora_names_to_process):
            if lora_name not in all_diffs:
                logger.warning(f"Skipping {lora_name} as no diff was calculated for it.")
                continue

            original_diff_tensor = all_diffs[lora_name]
            is_conv2d_layer = len(original_diff_tensor.size()) == 4
            kernel_s = original_diff_tensor.size()[2:4] if is_conv2d_layer else None
            is_conv2d_3x3_layer = is_conv2d_layer and kernel_s != (1, 1)
            module_true_out_channels, module_true_in_channels = original_diff_tensor.size()[0:2]
            mat_for_svd = original_diff_tensor.to(svd_computation_device, dtype=torch.float)

            if is_conv2d_layer:
                if is_conv2d_3x3_layer:
                    mat_for_svd = mat_for_svd.flatten(start_dim=1)
                else: 
                    mat_for_svd = mat_for_svd.squeeze()

            if mat_for_svd.numel() == 0 or mat_for_svd.shape[0] == 0 or mat_for_svd.shape[1] == 0 :
                logger.warning(f"Skipping SVD for {lora_name} due to empty/invalid shape: {mat_for_svd.shape}")
                continue
            
            try:
                U_full, S_full, Vh_full = torch.linalg.svd(mat_for_svd)
            except Exception as e:
                logger.error(f"SVD failed for {lora_name} with shape {mat_for_svd.shape}. Error: {e}")
                continue

            eff_out_dim, eff_in_dim = mat_for_svd.shape[0], mat_for_svd.shape[1]
            current_max_rank = dim if not is_conv2d_3x3_layer or conv_dim is None else conv_dim
            
            rank = _determine_rank(S_full, dynamic_method, dynamic_param,
                                   current_max_rank, eff_in_dim, eff_out_dim, MIN_SV)

            U_clamped, Vh_clamped = _construct_lora_weights_from_svd_components(
                U_full, S_full, Vh_full, rank, clamp_quantile,
                is_conv2d_layer, is_conv2d_3x3_layer, kernel_s,
                module_true_out_channels, module_true_in_channels,
                final_weights_device, save_dtype_torch
            )
            lora_weights[lora_name] = (U_clamped, Vh_clamped)

            if verbose:
                _log_svd_stats(lora_name, S_full, rank, MIN_SV)

    lora_sd = {}
    for lora_name, (up_weight, down_weight) in lora_weights.items():
        lora_sd[lora_name + ".lora_up.weight"] = up_weight
        lora_sd[lora_name + ".lora_down.weight"] = down_weight
        lora_sd[lora_name + ".alpha"] = torch.tensor(down_weight.size()[0], dtype=save_dtype_torch, device=final_weights_device)

    del text_encoders_o, unet_o, lora_network_o, all_diffs
    if 'torch' in sys.modules and hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    os.makedirs(os.path.dirname(save_to), exist_ok=True)

    metadata_to_save = _prepare_lora_metadata(
        output_path=save_to, 
        is_v2_flag=v2,  # The script's --v2 flag
        kohya_base_model_version_str=kohya_model_version, # The specific version string from model_util / sdxl_model_util
        network_conv_dim_val=conv_dim, 
        use_dynamic_method_flag=bool(dynamic_method), 
        network_dim_config_val=dim, # 'dim' is the general network dim if not dynamic
        is_v_param_flag=v_parameterization, # The script's v_param flag
        is_sdxl_flag=sdxl, # The script's --sdxl flag
        skip_sai_meta=no_metadata
    )
    
    save_to_file(save_to, lora_sd, save_dtype_torch, metadata_to_save)
    logger.info(f"LoRA saved to: {save_to}")


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="Load Stable Diffusion v2.x model")
    parser.add_argument("--v_parameterization", action="store_true", help="Set v-parameterization metadata (defaults to v2)")
    parser.add_argument("--sdxl", action="store_true", help="Load Stable Diffusion SDXL base model")
    parser.add_argument("--load_precision", type=str, choices=["float", "fp16", "bf16"], default=None, help="Precision for loading models (applied after initial load)")
    parser.add_argument("--save_precision", type=str, choices=["float", "fp16", "bf16"], default="float", help="Precision for saving LoRA weights")
    parser.add_argument("--model_org", type=str, required=True, help="Original Stable Diffusion model (ckpt/safetensors)")
    parser.add_argument("--model_tuned", type=str, required=True, help="Tuned Stable Diffusion model (ckpt/safetensors)")
    parser.add_argument("--save_to", type=str, required=True, help="Output file name (ckpt/safetensors)")
    parser.add_argument("--dim", type=int, default=4, help="Max dimension (rank) of LoRA for linear layers")
    parser.add_argument("--conv_dim", type=int, default=None, help="Max dimension (rank) of LoRA for Conv2d-3x3. Defaults to 'dim' if not set.")
    parser.add_argument("--device", type=str, default=None, help="Device for SVD computation (e.g., cuda, cpu). Defaults to cuda if available, else cpu.")
    parser.add_argument("--clamp_quantile", type=float, default=0.99, help="Quantile for clamping weights")
    parser.add_argument("--min_diff", type=float, default=0.01, help="Minimum weight difference to extract LoRA for a module")
    parser.add_argument("--no_metadata", action="store_true", help="Omit detailed metadata from SAI and Kohya_ss")
    parser.add_argument("--load_original_model_to", type=str, default=None, help="Device for original model (e.g. 'cpu', 'cuda:0'). Defaults to CPU.")
    parser.add_argument("--load_tuned_model_to", type=str, default=None, help="Device for tuned model (e.g. 'cpu', 'cuda:0'). Defaults to CPU.")
    parser.add_argument("--dynamic_param", type=float, help="Parameter for dynamic rank reduction")
    parser.add_argument("--verbose", action="store_true", help="Show detailed rank reduction info for each module")
    parser.add_argument(
        "--dynamic_method", type=str,
        choices=[None, "sv_ratio", "sv_fro", "sv_cumulative", "sv_knee", "sv_rel_decrease", "sv_cumulative_knee"],
        default=None, help="Dynamic rank reduction method"
    )
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if args.conv_dim is None:
        args.conv_dim = args.dim
        logger.info(f"--conv_dim not set, using value of --dim: {args.conv_dim}")

    methods_requiring_param = ["sv_ratio", "sv_fro", "sv_cumulative", "sv_rel_decrease"]
    if args.dynamic_method in methods_requiring_param and args.dynamic_param is None:
        parser.error(f"Dynamic method '{args.dynamic_method}' requires --dynamic_param to be set.")
    
    if not args.dynamic_method:
        if args.dim <= 0: parser.error(f"--dim (rank) must be > 0. Got {args.dim}")
        if args.conv_dim <=0: parser.error(f"--conv_dim (rank) must be > 0. Got {args.conv_dim}")
    
    if MIN_SV <= 0: logger.warning(f"Global MIN_SV ({MIN_SV}) should be positive.")

    # Pass the correct CLI arguments to the svd function
    # Note: 'v2', 'sdxl', 'v_parameterization' are directly from args
    # 'kohya_model_version' is determined inside svd() and then passed to _prepare_lora_metadata
    svd_args = vars(args).copy()
    # No change needed here as svd() internally determines kohya_model_version and passes it correctly
    svd(**svd_args)