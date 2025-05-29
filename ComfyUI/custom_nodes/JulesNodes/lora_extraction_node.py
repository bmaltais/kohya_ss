import torch
import os
import json
import time
import logging
# from safetensors.torch import save_file # Removed, not saving file directly
# from tqdm import tqdm # Removed, not used directly for now

# Basic logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

MIN_SV = 1e-6

# --- Localized sd-scripts constants and utility functions ---
_LOCAL_MODEL_VERSION_SDXL_BASE_V1_0 = "sdxl_v10"

def _local_get_model_version_str_for_sd1_sd2(is_v2: bool, is_v_parameterization: bool) -> str:
    if is_v2:
        return "v2-v" if is_v_parameterization else "v2"
    return "v1"

# --- Localized LoRA Placeholder and Network Creation ---
class LocalLoRAModulePlaceholder:
    def __init__(self, lora_name: str, org_module: torch.nn.Module):
        self.lora_name = lora_name
        self.org_module = org_module

def _local_create_network_placeholders(text_encoders: list, unet: torch.nn.Module, lora_conv_dim_init: int):
    unet_loras = []
    text_encoder_loras = []
    for name, module in unet.named_modules():
        lora_name = "lora_unet_" + name.replace(".", "_")
        if isinstance(module, torch.nn.Linear):
            unet_loras.append(LocalLoRAModulePlaceholder(lora_name, module))
        elif isinstance(module, torch.nn.Conv2d):
            if lora_conv_dim_init > 0:
                unet_loras.append(LocalLoRAModulePlaceholder(lora_name, module))
    for i, text_encoder in enumerate(text_encoders):
        if text_encoder is None: continue
        te_prefix = f"lora_te{i+1}_" if len(text_encoders) > 1 else "lora_te_"
        for name, module in text_encoder.named_modules():
            lora_name = te_prefix + name.replace(".", "_")
            if isinstance(module, torch.nn.Linear):
                text_encoder_loras.append(LocalLoRAModulePlaceholder(lora_name, module))
            elif isinstance(module, torch.nn.Conv2d):
                if lora_conv_dim_init > 0:
                     text_encoder_loras.append(LocalLoRAModulePlaceholder(lora_name, module))
    logger.info(f"Found {len(text_encoder_loras)} LoRA-able placeholder modules in Text Encoders.")
    logger.info(f"Found {len(unet_loras)} LoRA-able placeholder modules in U-Net.")
    return text_encoder_loras, unet_loras

# --- Singular Value Indexing Functions ---
def index_sv_cumulative(S, target):
    original_sum = float(torch.sum(S))
    cumulative_sums = torch.cumsum(S, dim=0) / original_sum
    index = int(torch.searchsorted(cumulative_sums, target)) + 1
    return max(1, min(index, len(S) - 1))

def index_sv_fro(S, target):
    S_squared = S.pow(2)
    S_fro_sq = float(torch.sum(S_squared))
    sum_S_squared = torch.cumsum(S_squared, dim=0) / S_fro_sq
    index = int(torch.searchsorted(sum_S_squared, target**2)) + 1
    return max(1, min(index, len(S) - 1))

def index_sv_ratio(S, target):
    max_sv = S[0]
    min_sv_val = max_sv / target
    index = int(torch.sum(S > min_sv_val).item())
    return max(1, min(index, len(S) - 1))

def index_sv_knee(S, MIN_SV_KNEE_PARAM=1e-8):
    n = len(S)
    if n < 3: return 1
    s_max, s_min = S[0], S[-1]
    if (s_max - s_min).abs() < MIN_SV_KNEE_PARAM: return 1
    s_normalized = (S - s_min) / (s_max - s_min)
    x_normalized = torch.linspace(0, 1, n, device=S.device, dtype=S.dtype)
    distances = (x_normalized + s_normalized - 1).abs()
    knee_index_0based = torch.argmax(distances).item()
    rank = knee_index_0based + 1
    return max(1, min(rank, n - 1))

def index_sv_cumulative_knee(S, min_sv_threshold_param=1e-8):
    n = len(S)
    if n < 3: return 1
    s_sum = torch.sum(S)
    if s_sum.abs() < min_sv_threshold_param: return 1
    y_values = torch.cumsum(S, dim=0) / s_sum
    y_min, y_max = y_values[0], y_values[n-1]
    if (y_max - y_min).abs() < min_sv_threshold_param: return 1
    y_norm = (y_values - y_min) / (y_max - y_min)
    x_norm = torch.linspace(0, 1, n, device=S.device, dtype=S.dtype)
    distances = (y_norm - x_norm).abs()
    knee_index_0based = torch.argmax(distances).item()
    rank = knee_index_0based + 1
    return max(1, min(rank, n - 1))

def index_sv_rel_decrease(S, tau=0.1):
    if len(S) < 2: return 1
    ratios = S[1:] / S[:-1]
    for k_idx in range(len(ratios)):
        if ratios[k_idx] < tau:
            return k_idx + 1
    return len(S)

# --- Utility Functions ---
def _str_to_dtype(p):
    if p == "float": return torch.float
    if p == "fp16": return torch.float16
    if p == "bf16": return torch.bfloat16
    return None

def _calculate_module_diffs_and_check(module_loras_o, module_loras_t, diff_calc_device, min_diff_thresh, module_type_str):
    diffs_map = {}
    is_different_flag = False
    first_diff_logged = False
    for lora_o, lora_t in zip(module_loras_o, module_loras_t):
        lora_name = lora_o.lora_name
        if not (hasattr(lora_o, 'org_module') and lora_o.org_module and hasattr(lora_o.org_module, 'weight') and lora_o.org_module.weight is not None and
                hasattr(lora_t, 'org_module') and lora_t.org_module and hasattr(lora_t.org_module, 'weight') and lora_t.org_module.weight is not None):
            logger.warning(f"Skipping {lora_name} in {module_type_str} due to missing org_module or weight.")
            continue
        weight_o = lora_o.org_module.weight.to(diff_calc_device, non_blocking=True)
        weight_t = lora_t.org_module.weight.to(diff_calc_device, non_blocking=True)
        diff = weight_t - weight_o
        diffs_map[lora_name] = diff
        if not is_different_flag and torch.max(torch.abs(diff)) > min_diff_thresh:
            is_different_flag = True
            if not first_diff_logged:
                 logger.info(f"{module_type_str} '{lora_name}' differs: max diff {torch.max(torch.abs(diff))} > {min_diff_thresh}")
                 first_diff_logged = True
    return diffs_map, is_different_flag

def _determine_rank(S_values, dynamic_method_name, dynamic_param_value, max_rank_limit, 
                    module_eff_in_dim, module_eff_out_dim, min_sv_threshold=MIN_SV):
    if not S_values.numel() or S_values[0].abs() <= min_sv_threshold: return 1
    rank = max_rank_limit
    if dynamic_method_name == "sv_ratio": rank = index_sv_ratio(S_values, dynamic_param_value)
    elif dynamic_method_name == "sv_cumulative": rank = index_sv_cumulative(S_values, dynamic_param_value)
    elif dynamic_method_name == "sv_fro": rank = index_sv_fro(S_values, dynamic_param_value)
    elif dynamic_method_name == "sv_knee": rank = index_sv_knee(S_values, MIN_SV_KNEE_PARAM=min_sv_threshold) 
    elif dynamic_method_name == "sv_cumulative_knee": rank = index_sv_cumulative_knee(S_values, min_sv_threshold_param=min_sv_threshold)
    elif dynamic_method_name == "sv_rel_decrease": rank = index_sv_rel_decrease(S_values, dynamic_param_value)
    
    rank = min(rank, max_rank_limit, module_eff_in_dim, module_eff_out_dim, len(S_values))
    return max(1, rank)

def _construct_lora_weights_from_svd_components(U_full, S_all_values, Vh_full, rank,
                                                clamp_quantile_val, is_conv2d, is_conv2d_3x3,
                                                conv_kernel_size,
                                                module_out_channels, module_in_channels,
                                                target_device_for_final_weights, target_dtype_for_final_weights):
    S_k = S_all_values[:rank]
    U_k = U_full[:, :rank]
    Vh_k = Vh_full[:rank, :]
    S_k_non_negative = torch.clamp(S_k, min=0.0) # Ensure non-negative before sqrt
    s_sqrt = torch.sqrt(S_k_non_negative)
    U_final = U_k * s_sqrt.unsqueeze(0)
    Vh_final = Vh_k * s_sqrt.unsqueeze(1)
    
    combined_dist = torch.cat([U_final.flatten(), Vh_final.flatten()])
    if combined_dist.numel() > 0: # Ensure dist is not empty before quantile
        hi_val = torch.quantile(combined_dist.abs(), clamp_quantile_val) # Use abs for quantile on magnitude
        if hi_val == 0 and torch.max(torch.abs(combined_dist)) > 1e-9: # Check if hi_val is unexpectedly zero
            logger.debug(f"Clamping hi_val is zero for non-zero distribution. Max abs val: {torch.max(torch.abs(combined_dist))}. Quantile: {clamp_quantile_val}")
            # Potentially use a small epsilon or max_abs_val as hi_val if this happens
            hi_val = torch.max(torch.abs(combined_dist)) * 0.1 # Fallback, can be adjusted
        
        U_clamped = U_final.clamp(-hi_val, hi_val)
        Vh_clamped = Vh_final.clamp(-hi_val, hi_val)
    else: # Should not happen if SVD produced results
        U_clamped = U_final
        Vh_clamped = Vh_final

    if is_conv2d:
        U_clamped = U_clamped.reshape(module_out_channels, rank, 1, 1)
        if is_conv2d_3x3: Vh_clamped = Vh_clamped.reshape(rank, module_in_channels, *conv_kernel_size)
        else: Vh_clamped = Vh_clamped.reshape(rank, module_in_channels, 1, 1)
        
    return (U_clamped.to(target_device_for_final_weights, dtype=target_dtype_for_final_weights).contiguous(),
            Vh_clamped.to(target_device_for_final_weights, dtype=target_dtype_for_final_weights).contiguous())

def _log_svd_stats(lora_module_name, S_all_values, rank_used, min_sv_for_calc=MIN_SV):
    if not S_all_values.numel():
        logger.info(f"{lora_module_name:75} | rank: {rank_used}, SVD not performed (empty singular values).")
        return
    S_cpu = S_all_values.to('cpu', non_blocking=True)
    s_sum_total = float(torch.sum(S_cpu))
    s_sum_rank = float(torch.sum(S_cpu[:rank_used]))
    fro_orig_total = float(torch.sqrt(torch.sum(S_cpu.pow(2))))
    fro_reconstructed_rank = float(torch.sqrt(torch.sum(S_cpu[:rank_used].pow(2))))
    ratio_sv = float('inf')
    if rank_used > 0 and S_cpu[rank_used - 1].abs() > min_sv_for_calc:
        ratio_sv = S_cpu[0] / S_cpu[rank_used - 1]
    sum_s_retained_percentage = (s_sum_rank / s_sum_total) if abs(s_sum_total) > min_sv_for_calc else 1.0
    fro_retained_percentage = (fro_reconstructed_rank / fro_orig_total) if abs(fro_orig_total) > min_sv_for_calc else 1.0
    logger.info(
        f"{lora_module_name:75} | rank: {rank_used}, "
        f"sum(S) retained: {sum_s_retained_percentage:.2%}, "
        f"Frobenius norm retained: {fro_retained_percentage:.2%}, "
        f"max_retained_sv/min_retained_sv ratio: {ratio_sv:.2f}"
    )

def _build_local_sai_metadata(title, creation_time, is_v2_flag, is_v_param_flag, is_sdxl_flag):
    metadata = {"ss_sd_model_name": str(title), "ss_creation_time": str(int(creation_time))}
    if is_sdxl_flag:
        metadata.update({"ss_base_model_version": "sdxl_v10", "ss_sdxl_model_version": "1.0"})
        if is_v_param_flag: metadata["ss_v_parameterization"] = "true"
    elif is_v2_flag:
        metadata.update({"ss_base_model_version": "sd_v2"})
        if is_v_param_flag: metadata["ss_v_parameterization"] = "true"
    else:
        metadata.update({"ss_base_model_version": "sd_v1"})
        if is_v_param_flag: metadata["ss_v_parameterization"] = "true"
    return metadata

def _prepare_lora_metadata(output_path_hint, is_v2_flag, kohya_base_model_version_str, network_conv_dim_val, 
                           use_dynamic_method_flag, network_dim_config_val, 
                           is_v_param_flag, is_sdxl_flag, skip_sai_meta):
    net_kwargs = ({"conv_dim": str(network_conv_dim_val), "conv_alpha": str(float(network_conv_dim_val))} 
                  if network_conv_dim_val is not None and network_conv_dim_val > 0 else {})
    network_dim_meta = "Dynamic" if use_dynamic_method_flag else str(network_dim_config_val)
    network_alpha_meta = "Dynamic" if use_dynamic_method_flag else str(float(network_dim_config_val))
    
    final_metadata = {
        "ss_v2": str(is_v2_flag), "ss_base_model_version": kohya_base_model_version_str,
        "ss_network_module": "networks.lora", "ss_network_dim": network_dim_meta,
        "ss_network_alpha": network_alpha_meta, "ss_network_args": json.dumps(net_kwargs),
        "ss_lowram": "False", "ss_num_train_images": "N/A", # Default/Placeholder values
    }
    if not skip_sai_meta:
        title = os.path.splitext(os.path.basename(output_path_hint))[0]
        sai_metadata_content = _build_local_sai_metadata(
            title=title, creation_time=time.time(), is_v2_flag=is_v2_flag,
            is_v_param_flag=is_v_param_flag, is_sdxl_flag=is_sdxl_flag
        )
        final_metadata.update(sai_metadata_content)
    return final_metadata

class LoraExtractionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("MODEL",), "tuned_model": ("MODEL",),
                "rank_dim": ("INT", {"default": 4, "min": 1, "max": 1024, "step": 1}),
                "conv_rank_dim": ("INT", {"default": 4, "min": 0, "max": 1024, "step": 1}), # 0 means don't use conv lora
                "save_precision": (["float", "fp16", "bf16"], {"default": "float"}),
                "computation_device": (["AUTO", "CPU", "CUDA"], {"default": "AUTO"}),
                "clamp_quantile": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.001}),
                "min_module_diff": ("FLOAT", {"default": 0.001, "min": 0.0, "step": 0.0001}), # Reduced default
                "dynamic_rank_method": (["None", "sv_ratio", "sv_fro", "sv_cumulative", "sv_knee", "sv_rel_decrease", "sv_cumulative_knee"], {"default": "None"}),
                "dynamic_rank_param": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}), # Adjusted default for cumulative
            },
            "optional": {
                "is_sdxl": ("BOOLEAN", {"default": False}), "is_v2": ("BOOLEAN", {"default": False}),
                "v_parameterization": ("BOOLEAN", {"default": False}),
                "skip_node_metadata": ("BOOLEAN", {"default": False}),
                "verbose_logging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "extract_lora"
    CATEGORY = "JulesNodes/LoRA"

    def extract_lora(self, base_model_sd, tuned_model_sd, rank_dim, conv_rank_dim, save_precision,
                     computation_device_str, clamp_quantile, min_module_diff, dynamic_rank_method_str,
                     dynamic_rank_param, is_sdxl=False, is_v2=False, v_parameterization=False,
                     skip_node_metadata=False, verbose_logging=False):

        if verbose_logging: logger.setLevel(logging.DEBUG)
        else: logger.setLevel(logging.INFO)
        logger.info("Starting LoRA Extraction Node")

        eff_dynamic_method = None if dynamic_rank_method_str == "None" else dynamic_rank_method_str
        actual_v_param = (is_v2 or is_sdxl) if v_parameterization is None else v_parameterization # v_param default for SDXL too
        save_dtype_torch = _str_to_dtype(save_precision)
        
        svd_computation_device = torch.device(computation_device_str.lower() if computation_device_str != "AUTO" 
                                             else "cuda" if torch.cuda.is_available() else "cpu")
        diff_calculation_device = torch.device("cpu")
        final_weights_device = torch.device("cpu")
        logger.info(f"Devices: SVD={svd_computation_device}, DiffCalc={diff_calculation_device}, FinalWeights={final_weights_device}")

        # --- Model Component Extraction ---
        def get_model_components(comfy_model_obj, model_name_for_log="model"):
            unet_model = comfy_model_obj.model.diffusion_model
            te_models = []
            if hasattr(comfy_model_obj.model, 'conditioner') and hasattr(comfy_model_obj.model.conditioner, 'embedders'):
                logger.info(f"Extracting text encoders from {model_name_for_log}.conditioner.embedders")
                for embedder_wrapper in comfy_model_obj.model.conditioner.embedders:
                    # This needs to handle various ComfyUI wrapper types for text encoders
                    actual_te_module = None
                    if hasattr(embedder_wrapper, 'embedder') and hasattr(embedder_wrapper.embedder, 'model') and hasattr(embedder_wrapper.embedder.model, 'transformer'): # e.g. CLIPTextModelEmbedder
                        actual_te_module = embedder_wrapper.embedder.model.transformer
                    elif hasattr(embedder_wrapper, 'model') and hasattr(embedder_wrapper.model, 'transformer'): # Direct model or AIT-like
                        actual_te_module = embedder_wrapper.model.transformer
                    elif hasattr(embedder_wrapper, 'wrapped') and hasattr(embedder_wrapper.wrapped, 'transformer'): # Diffusers Hosted Style
                         actual_te_module = embedder_wrapper.wrapped.transformer
                    elif hasattr(embedder_wrapper, 'wrapped_model') and hasattr(embedder_wrapper.wrapped_model, 'transformer'):
                        actual_te_module = embedder_wrapper.wrapped_model.transformer
                    
                    if actual_te_module: te_models.append(actual_te_module)
                    else: logger.warning(f"Could not extract transformer from an embedder in {model_name_for_log}: {type(embedder_wrapper)}")
            
            if not te_models: # Fallback for non-embedder based TEs (e.g. SD1.5)
                logger.info(f"No TEs via 'conditioner.embedders' for {model_name_for_log}. Trying legacy paths.")
                if hasattr(comfy_model_obj.model, 'cond_stage_model') and hasattr(comfy_model_obj.model.cond_stage_model, 'transformer'):
                    te_models.append(comfy_model_obj.model.cond_stage_model.transformer)
                elif hasattr(comfy_model_obj.model, 'clip_l') and hasattr(comfy_model_obj.model.clip_l, 'transformer'):
                    te_models.append(comfy_model_obj.model.clip_l.transformer)
            
            if not te_models: logger.error(f"NO TEXT ENCODERS EXTRACTED FOR {model_name_for_log.upper()}!")
            return unet_model, te_models

        unet_o, text_encoders_o_list = get_model_components(base_model_sd, "base_model")
        unet_t, text_encoders_t_list = get_model_components(tuned_model_sd, "tuned_model")

        if not text_encoders_o_list or not text_encoders_t_list or \
           len(text_encoders_o_list) != len(text_encoders_t_list):
            logger.error("Text encoder mismatch or extraction failure. Aborting.")
            return ({},) # Return empty LoRA

        kohya_model_version = _LOCAL_MODEL_VERSION_SDXL_BASE_V1_0 if is_sdxl else \
                              _local_get_model_version_str_for_sd1_sd2(is_v2, actual_v_param)
        
        # Use conv_rank_dim (0 means no conv lora)
        text_encoder_loras_o, unet_loras_o = _local_create_network_placeholders(text_encoders_o_list, unet_o, conv_rank_dim if conv_rank_dim > 0 else 0)
        text_encoder_loras_t, unet_loras_t = _local_create_network_placeholders(text_encoders_t_list, unet_t, conv_rank_dim if conv_rank_dim > 0 else 0)

        all_diffs = {}
        te_diffs, te_is_diff = _calculate_module_diffs_and_check(text_encoder_loras_o, text_encoder_loras_t, diff_calculation_device, min_module_diff, "Text Encoder")
        if te_is_diff: all_diffs.update(te_diffs)
        else: logger.warning("Text Encoders identical or below diff threshold.")
        
        unet_diffs, unet_is_diff = _calculate_module_diffs_and_check(unet_loras_o, unet_loras_t, diff_calculation_device, min_module_diff, "U-Net")
        if unet_is_diff: all_diffs.update(unet_diffs)
        else: logger.warning("U-Net identical or below diff threshold.")

        if not all_diffs:
            logger.error("No differences found above threshold. Cannot extract LoRA.")
            return ({},)

        lora_weights_map = {}
        lora_module_placeholders = (text_encoder_loras_o if te_is_diff else []) + (unet_loras_o if unet_is_diff else [])

        for placeholder in lora_module_placeholders:
            lora_name = placeholder.lora_name
            if lora_name not in all_diffs: continue # Should not happen if logic is correct

            original_diff_tensor = all_diffs[lora_name]
            is_conv2d = len(original_diff_tensor.shape) == 4
            kernel_s = original_diff_tensor.shape[2:4] if is_conv2d else None
            is_conv2d_3x3 = is_conv2d and kernel_s != (1,1)

            # Skip conv if conv_rank_dim is 0
            if is_conv2d and conv_rank_dim == 0:
                logger.debug(f"Skipping conv layer {lora_name} as conv_rank_dim is 0.")
                continue

            mat_for_svd = original_diff_tensor.to(svd_computation_device, dtype=torch.float)
            if is_conv2d: mat_for_svd = mat_for_svd.flatten(start_dim=1) if is_conv2d_3x3 else mat_for_svd.squeeze()
            
            if mat_for_svd.numel() == 0 or mat_for_svd.shape[0] == 0 or mat_for_svd.shape[1] == 0:
                logger.warning(f"Skipping SVD for {lora_name} due to empty/invalid shape: {mat_for_svd.shape}")
                continue
            try:
                U_full, S_full, Vh_full = torch.linalg.svd(mat_for_svd)
            except Exception as e:
                logger.error(f"SVD failed for {lora_name} (shape {mat_for_svd.shape}): {e}")
                continue
            
            module_max_rank = conv_rank_dim if is_conv2d_3x3 else rank_dim # Use conv_rank_dim for 3x3 convs if >0
            if is_conv2d and not is_conv2d_3x3: # For 1x1 conv
                module_max_rank = rank_dim # Default to linear rank_dim for 1x1 conv, unless conv_rank_dim is specified for all convs

            eff_out_dim, eff_in_dim = mat_for_svd.shape[0], mat_for_svd.shape[1]
            current_rank = _determine_rank(S_full, eff_dynamic_method, dynamic_rank_param, module_max_rank, eff_in_dim, eff_out_dim)
            
            U_clamped, Vh_clamped = _construct_lora_weights_from_svd_components(
                U_full, S_full, Vh_full, current_rank, clamp_quantile, is_conv2d, is_conv2d_3x3, kernel_s,
                original_diff_tensor.shape[0], original_diff_tensor.shape[1], # true out/in channels
                final_weights_device, save_dtype_torch)
            
            lora_weights_map[lora_name] = (U_clamped, Vh_clamped, float(current_rank))
            if verbose_logging: _log_svd_stats(lora_name, S_full, current_rank)

        lora_sd = {}
        for name, (up, down, rank_val) in lora_weights_map.items():
            lora_sd[name + ".lora_up.weight"] = up
            lora_sd[name + ".lora_down.weight"] = down
            lora_sd[name + ".alpha"] = torch.tensor(rank_val, dtype=save_dtype_torch, device=final_weights_device)

        if not skip_node_metadata:
            # Use a hint for output path for metadata, not actually saving
            metadata_path_hint = f"lora_node_rank{rank_dim}_conv{conv_rank_dim}.safetensors"
            kohya_meta = _prepare_lora_metadata(metadata_path_hint, is_v2, kohya_model_version, 
                                               conv_rank_dim if conv_rank_dim > 0 else None, # Pass None if conv_dim is 0
                                               bool(eff_dynamic_method), rank_dim, actual_v_param, is_sdxl, True)
            for k, v in kohya_meta.items():
                # Store metadata as strings in a sub-dictionary or prefixed keys
                lora_sd[f"kohya_metadata.{k}"] = str(v) 

        if hasattr(torch, 'cuda') and torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info(f"LoRA extraction complete. Modules extracted: {len(lora_weights_map)}")
        return (lora_sd,)

NODE_CLASS_MAPPINGS = {"LoraExtractionNode": LoraExtractionNode}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraExtractionNode": "LoRA Extraction Node (by Jules)"}
