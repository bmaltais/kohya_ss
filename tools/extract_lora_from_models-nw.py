import argparse
import json
import os
import time
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from library import sai_model_spec, model_util, sdxl_model_util
import lora # Assuming this is your existing lora script/library
from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

MIN_SV = 1e-6

# ... (Keep all your existing helper functions: index_sv_cumulative, index_sv_fro, etc.)
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
            return max(1, k + 1)
    return min(len(S), len(S) - 1 if len(S) > 1 else 1)


def save_to_file(file_name, model_sd, dtype, metadata=None): # Changed model to model_sd for clarity
    if dtype is not None:
        for key in list(model_sd.keys()):
            if isinstance(model_sd[key], torch.Tensor):
                model_sd[key] = model_sd[key].to(dtype)
    
    # Filter out non-tensor metadata if it accidentally gets into model_sd
    final_sd = {k: v for k, v in model_sd.items() if isinstance(v, torch.Tensor)}

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(final_sd, file_name, metadata=metadata) # Pass metadata here
    else:
        # For .pt, metadata is typically not saved in this manner.
        # If you need to save metadata with .pt, you might save a dict like {'state_dict': final_sd, 'metadata': metadata}
        torch.save(final_sd, file_name)
        if metadata:
            logger.warning(".pt format does not standardly support metadata like safetensors. Metadata not saved in file.")

def svd_decomposition(
    model_org_path=None, # Renamed for clarity
    model_tuned_path=None, # Renamed for clarity
    save_to=None,
    algo="lora", # New: lora or loha
    network_dim=4, # For LoRA: rank. For LoHA: "factor" or "hada_dim"
    network_alpha=None, # For LoRA: alpha (often same as rank). For LoHA: "rank_initial" or "hada_alpha"
    conv_dim=None, # For LoRA: conv_rank. For LoHA: "conv_factor"
    conv_alpha=None, # For LoRA: conv_alpha. For LoHA: "conv_rank_initial"
    v2=None,
    sdxl=None,
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
        if p == "float": return torch.float
        if p == "fp16": return torch.float16
        if p == "bf16": return torch.bfloat16
        return None

    assert not (v2 and sdxl), "v2 and sdxl cannot be specified at the same time"
    v_parameterization = v2 if v_parameterization is None else v_parameterization

    load_dtype = str_to_dtype(load_precision) if load_precision else None
    save_dtype = str_to_dtype(save_precision) if save_precision else torch.float
    work_device = "cpu" # Perform SVD and weight manipulation on CPU then move
    compute_device = device if device else "cpu"

    # Handle default alpha values based on dim values if not provided
    if network_alpha is None: network_alpha = network_dim
    if conv_dim is None: conv_dim = network_dim # default conv_dim to network_dim
    if conv_alpha is None: conv_alpha = conv_dim # default conv_alpha to conv_dim

    # Load models
    if not sdxl:
        logger.info(f"Loading original SD model: {model_org_path}")
        text_encoder_o, _, unet_o = model_util.load_models_from_stable_diffusion_checkpoint(v2, model_org_path, load_dtype)
        text_encoders_o = [text_encoder_o]
        
        logger.info(f"Loading tuned SD model: {model_tuned_path}")
        text_encoder_t, _, unet_t = model_util.load_models_from_stable_diffusion_checkpoint(v2, model_tuned_path, load_dtype)
        text_encoders_t = [text_encoder_t]
        
        model_version = model_util.get_model_version_str_for_sd1_sd2(v2, v_parameterization)
    else: # SDXL
        device_org = load_original_model_to or "cpu"
        device_tuned = load_tuned_model_to or "cpu"

        logger.info(f"Loading original SDXL model: {model_org_path}")
        text_encoder_o1, text_encoder_o2, _, unet_o, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
            sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, model_org_path, device_org, load_dtype
        )
        text_encoders_o = [text_encoder_o1, text_encoder_o2]

        logger.info(f"Loading tuned SDXL model: {model_tuned_path}")
        text_encoder_t1, text_encoder_t2, _, unet_t, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
            sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, model_tuned_path, device_tuned, load_dtype
        )
        text_encoders_t = [text_encoder_t1, text_encoder_t2]
        model_version = sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0

    # Create temporary LoRA network to identify modules and get original weights
    # Use a minimal fixed dimension for this stage as we only need module structure and original weights
    temp_lora_kwargs = {"conv_dim": 1, "conv_alpha": 1.0} # Minimal conv settings for network creation
    lora_network_o = lora.create_network(1.0, 1, 1.0, None, text_encoders_o, unet_o, **temp_lora_kwargs)
    lora_network_t = lora.create_network(1.0, 1, 1.0, None, text_encoders_t, unet_t, **temp_lora_kwargs)
    assert len(lora_network_o.text_encoder_loras) == len(lora_network_t.text_encoder_loras)

    diffs = {}
    text_encoder_differs = False
    # Text Encoders
    for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.text_encoder_loras, lora_network_t.text_encoder_loras)):
        module_key_name = lora_o.lora_name # e.g. "lora_te1_text_model_encoder_layers_0_mlp_fc1"
        org_weight = lora_o.org_module.weight.to(device=work_device, dtype=torch.float)
        tuned_weight = lora_t.org_module.weight.to(device=work_device, dtype=torch.float)
        diff = tuned_weight - org_weight
        
        if torch.max(torch.abs(diff)) > min_diff:
            text_encoder_differs = True
            logger.info(f"Text encoder {i+1} module {module_key_name} differs: max diff {torch.max(torch.abs(diff))}")
            diffs[module_key_name] = diff
        else:
            logger.info(f"Text encoder {i+1} module {module_key_name} has no significant difference.")
        
        # Free memory
        lora_o.org_module.weight = None 
        lora_t.org_module.weight = None
        del org_weight, tuned_weight
    
    # UNet
    for lora_o, lora_t in zip(lora_network_o.unet_loras, lora_network_t.unet_loras):
        module_key_name = lora_o.lora_name # e.g. "lora_unet_input_blocks_1_1_proj_in"
        org_weight = lora_o.org_module.weight.to(device=work_device, dtype=torch.float)
        tuned_weight = lora_t.org_module.weight.to(device=work_device, dtype=torch.float)
        diff = tuned_weight - org_weight

        if torch.max(torch.abs(diff)) > min_diff:
             logger.info(f"UNet module {module_key_name} differs: max diff {torch.max(torch.abs(diff))}")
             diffs[module_key_name] = diff
        else:
            logger.info(f"UNet module {module_key_name} has no significant difference.")

        lora_o.org_module.weight = None
        lora_t.org_module.weight = None
        del org_weight, tuned_weight

    if not text_encoder_differs:
        logger.warning("Text encoder weights are identical or below min_diff. Text encoder LoRA modules will not be included.")
        # Remove text encoder diffs if none were significant
        diffs = {k: v for k, v in diffs.items() if "unet" in k}

    del lora_network_o, lora_network_t, text_encoders_o, text_encoders_t, unet_o, unet_t
    torch.cuda.empty_cache()

    lora_module_weights = {} # This will store the final decomposed weights for LoRA/LoHA

    logger.info(f"Extracting and resizing {algo.upper()} modules via SVD")

    with torch.no_grad():
        for module_key_name, mat_diff in tqdm(diffs.items()):
            if compute_device != "cpu": # Move to GPU for SVD if specified
                mat_diff = mat_diff.to(compute_device)
            
            # Determine if the layer is convolutional and its properties
            is_conv = len(mat_diff.shape) == 4
            kernel_size = None
            if is_conv:
                kernel_size = mat_diff.shape[2:4]
            
            # For LoRA, conv_dim/alpha are specific to 3x3 convs. For others, it uses network_dim/alpha.
            # For LoHA, we use conv_dim/alpha for any conv layer, and network_dim/alpha for linear.
            # This logic can be refined based on how LyCORIS typically handles different conv kernel sizes for LoHA.
            # Here, we'll use conv parameters if it's a conv layer, otherwise network parameters.
            
            current_dim_target = conv_dim if is_conv else network_dim
            current_alpha_target = conv_alpha if is_conv else network_alpha

            # For LoHA, 'alpha_target' is the rank of the first SVD (rank_initial)
            # and 'dim_target' is the rank of the second SVD (rank_factor).
            # For LoRA, 'dim_target' is the rank of the SVD, and 'alpha_target' is the scaling factor.

            # Reshape convolutional weights for SVD
            original_shape = mat_diff.shape
            if is_conv:
                if kernel_size == (1, 1):
                    mat_diff = mat_diff.squeeze() # Becomes (out_channels, in_channels)
                else: # kernel_size (3,3) or others
                    mat_diff = mat_diff.flatten(start_dim=1) # Becomes (out_channels, in_channels*k_w*k_h)
            
            out_features, in_features = mat_diff.shape[0], mat_diff.shape[1]

            # Perform first SVD
            try:
                U, S, Vh = torch.linalg.svd(mat_diff)
            except Exception as e:
                logger.error(f"SVD failed for {module_key_name} with shape {mat_diff.shape}: {e}")
                continue
            
            if compute_device != "cpu": # Move results back to CPU if computation was on GPU
                U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()

            # Determine rank for the first SVD (rank_initial for LoHA, rank for LoRA)
            max_rank_initial = min(out_features, in_features) # Theoretical max rank
            
            # Default rank_initial to current_alpha_target (which is network_alpha or conv_alpha)
            rank_initial = current_alpha_target 
            
            if dynamic_method:
                if S[0] <= MIN_SV:
                    determined_rank = 1
                elif dynamic_method == "sv_ratio": determined_rank = index_sv_ratio(S, dynamic_param)
                elif dynamic_method == "sv_cumulative": determined_rank = index_sv_cumulative(S, dynamic_param)
                elif dynamic_method == "sv_fro": determined_rank = index_sv_fro(S, dynamic_param)
                elif dynamic_method == "sv_knee": determined_rank = index_sv_knee_improved(S, MIN_SV)
                elif dynamic_method == "sv_cumulative_knee": determined_rank = index_sv_cumulative_knee(S, MIN_SV)
                elif dynamic_method == "sv_rel_decrease": determined_rank = index_sv_rel_decrease(S, dynamic_param)
                else: determined_rank = rank_initial # Fallback if dynamic method unknown
                rank_initial = min(determined_rank, current_alpha_target, max_rank_initial)
            else:
                rank_initial = min(current_alpha_target, max_rank_initial)
            rank_initial = max(1, rank_initial) # Ensure rank is at least 1

            # --- LoRA specific decomposition ---
            if algo == 'lora':
                lora_down = Vh[:rank_initial, :]
                lora_up = U[:, :rank_initial] @ torch.diag(S[:rank_initial])

                # Clamp values
                dist = torch.cat([lora_up.flatten(), lora_down.flatten()])
                hi_val = torch.quantile(dist, clamp_quantile) if clamp_quantile < 1.0 else dist.abs().max()
                lora_up = lora_up.clamp(-hi_val, hi_val)
                lora_down = lora_down.clamp(-hi_val, hi_val)

                # Reshape for conv layers if necessary
                if is_conv:
                    if kernel_size == (1,1):
                        # These are already (out_c, rank) and (rank, in_c)
                        # Some LoRA impls might expect them to be 4D (out_c, rank, 1, 1) and (rank, in_c, 1, 1)
                        lora_up = lora_up.reshape(out_features, rank_initial, 1, 1)
                        lora_down = lora_down.reshape(rank_initial, in_features, 1, 1)
                    else: # e.g. 3x3 conv
                        # lora_down was (rank, in_c*k_w*k_h), needs to be (rank, in_c, k_w, k_h)
                        # lora_up was (out_c, rank)
                        lora_up = lora_up.reshape(out_features, rank_initial, 1, 1) # often up is kept as 1x1 conv like
                        lora_down = lora_down.reshape(rank_initial, original_shape[1], *kernel_size)

                lora_module_weights[f"{module_key_name}.lora_down.weight"] = lora_down.to(work_device, dtype=save_dtype).contiguous()
                lora_module_weights[f"{module_key_name}.lora_up.weight"] = lora_up.to(work_device, dtype=save_dtype).contiguous()
                lora_module_weights[f"{module_key_name}.alpha"] = torch.tensor(float(current_alpha_target), dtype=save_dtype) # Use actual alpha target from params

            # --- LoHA specific decomposition ---
            elif algo == 'loha':
                lora_down_equivalent = Vh[:rank_initial, :] 
                lora_up_equivalent = U[:, :rank_initial] @ torch.diag(S[:rank_initial])
                
                # current_dim_target is the "factor" for LoHA's second SVD
                rank_factor = min(current_dim_target, rank_initial) # Factor cannot exceed rank_initial
                if is_conv and kernel_size != (1,1): # For conv3x3, factor also limited by in/out features
                     rank_factor = min(rank_factor, original_shape[1], original_shape[0]) #original_shape[1] is in_channels
                else: # Linear or Conv1x1
                     rank_factor = min(rank_factor, in_features, out_features)
                rank_factor = max(1, rank_factor)


                # Decompose Lora_Down_Equivalent (shape: rank_initial x in_features_eff)
                # Target: hada_w1_b (rank_initial x rank_factor) @ hada_w1_a (rank_factor x in_features_eff)
                if compute_device != "cpu": lora_down_equivalent = lora_down_equivalent.to(compute_device)
                Ud, Sd, Vhd = torch.linalg.svd(lora_down_equivalent)
                if compute_device != "cpu": Ud, Sd, Vhd = Ud.cpu(), Sd.cpu(), Vhd.cpu()
                
                hada_w1_a = Vhd[:rank_factor, :]
                hada_w1_b = Ud[:, :rank_factor] @ torch.diag(Sd[:rank_factor])

                # Decompose Lora_Up_Equivalent (shape: out_features_eff x rank_initial)
                # Target: hada_w2_b (out_features_eff x rank_factor) @ hada_w2_a (rank_factor x rank_initial)
                if compute_device != "cpu": lora_up_equivalent = lora_up_equivalent.to(compute_device)
                Uu, Su, Vhu = torch.linalg.svd(lora_up_equivalent)
                if compute_device != "cpu": Uu, Su, Vhu = Uu.cpu(), Su.cpu(), Vhu.cpu()

                hada_w2_a = Vhu[:rank_factor, :]
                hada_w2_b = Uu[:, :rank_factor] @ torch.diag(Su[:rank_factor])

                # Clamp LoHA components
                dist_w1a = hada_w1_a.flatten()
                dist_w1b = hada_w1_b.flatten()
                dist_w2a = hada_w2_a.flatten()
                dist_w2b = hada_w2_b.flatten()
                
                if clamp_quantile < 1.0:
                    hi_val_w1a = torch.quantile(dist_w1a.abs(), clamp_quantile)
                    hi_val_w1b = torch.quantile(dist_w1b.abs(), clamp_quantile)
                    hi_val_w2a = torch.quantile(dist_w2a.abs(), clamp_quantile)
                    hi_val_w2b = torch.quantile(dist_w2b.abs(), clamp_quantile)
                else: # Use max abs value if quantile is 1.0 or more
                    hi_val_w1a = dist_w1a.abs().max()
                    hi_val_w1b = dist_w1b.abs().max()
                    hi_val_w2a = dist_w2a.abs().max()
                    hi_val_w2b = dist_w2b.abs().max()

                hada_w1_a = hada_w1_a.clamp(-hi_val_w1a, hi_val_w1a)
                hada_w1_b = hada_w1_b.clamp(-hi_val_w1b, hi_val_w1b)
                hada_w2_a = hada_w2_a.clamp(-hi_val_w2a, hi_val_w2a)
                hada_w2_b = hada_w2_b.clamp(-hi_val_w2b, hi_val_w2b)
                
                # LoHA weights are typically stored as 2D matrices.
                # LyCORIS library handles reshaping or uses 1x1 convs internally.
                lora_module_weights[f"{module_key_name}.hada_w1_a"] = hada_w1_a.to(work_device, dtype=save_dtype).contiguous()
                lora_module_weights[f"{module_key_name}.hada_w1_b"] = hada_w1_b.to(work_device, dtype=save_dtype).contiguous()
                lora_module_weights[f"{module_key_name}.hada_w2_a"] = hada_w2_a.to(work_device, dtype=save_dtype).contiguous()
                lora_module_weights[f"{module_key_name}.hada_w2_b"] = hada_w2_b.to(work_device, dtype=save_dtype).contiguous()
                lora_module_weights[f"{module_key_name}.alpha"] = torch.tensor(float(current_alpha_target), dtype=save_dtype) # This is rank_initial
            
            if verbose:
                s_sum = float(torch.sum(S))
                s_rank_initial = float(torch.sum(S[:rank_initial]))
                fro_initial = float(torch.sqrt(torch.sum(S.pow(2))))
                fro_rank_initial = float(torch.sqrt(torch.sum(S[:rank_initial].pow(2))))
                # This verbose output is for the first SVD. Adding verbose for second SVD would be more complex.
                logger.info(
                    f"{module_key_name[:75]:75} | Algo: {algo.upper()}, Rank/Alpha (initial): {rank_initial}, Dim/Factor (final): {rank_factor if algo=='loha' else rank_initial} "
                    f"| Sum(S) Retained (1st SVD): {s_rank_initial/s_sum if s_sum > 0 else 0:.1%}, Fro Retained (1st SVD): {fro_rank_initial/fro_initial if fro_initial > 0 else 0:.1%}"
                )

            del U, S, Vh, mat_diff
            if algo == 'loha':
                del lora_down_equivalent, lora_up_equivalent, Ud, Sd, Vhd, Uu, Su, Vhu
                del hada_w1_a, hada_w1_b, hada_w2_a, hada_w2_b
            else: # lora
                del lora_down, lora_up
            if compute_device != "cpu":
                torch.cuda.empty_cache()


    if not lora_module_weights:
        logger.error("No LoRA/LoHA modules were generated. This might be due to models being too similar or min_diff being too high.")
        return

    os.makedirs(os.path.dirname(save_to), exist_ok=True)

    # Metadata
    metadata = {
        "ss_v2": str(v2) if v2 is not None else "false", # More explicit "false"
        "ss_base_model_version": model_version,
        "ss_sdxl_model_version": "1.0" if sdxl else "null", # LyCORIS convention
        "ss_network_module": "networks.lora" if algo == "lora" else "lycoris.kohya", # For LoHA
        # For LoRA, dim and alpha are usually the same as what's passed.
        # For LoHA, network_dim is the 'factor' (our network_dim/conv_dim),
        # and network_alpha is the 'rank_initial' (our network_alpha/conv_alpha).
        "ss_network_dim": str(network_dim) if not dynamic_method or algo == "loha" else "Dynamic", # For LoHA, this is 'factor'
        "ss_network_alpha": str(network_alpha) if not dynamic_method or algo == "loha" else "Dynamic", # For LoHA, this is 'rank_initial'
    }
    
    net_kwargs = {}
    if algo == "lora":
        if conv_dim is not None: # Only add if conv_dim was actually specified for LoRA
            net_kwargs["conv_dim"] = str(conv_dim)
            net_kwargs["conv_alpha"] = str(float(conv_alpha if conv_alpha is not None else conv_dim))
    elif algo == "loha":
        net_kwargs["algo"] = "loha"
        # For LoHA, conv_dim and conv_alpha are distinct concepts (factor and rank_initial for conv layers)
        net_kwargs["conv_dim"] = str(conv_dim) 
        net_kwargs["conv_alpha"] = str(float(conv_alpha))
        # LyCORIS sometimes uses dropout, but we are not implementing it here.
        # net_kwargs["dropout"] = "0" # Example if we had dropout

    metadata["ss_network_args"] = json.dumps(net_kwargs) if net_kwargs else "null" # LyCORIS uses "null" for empty

    if not no_metadata:
        title = os.path.splitext(os.path.basename(save_to))[0]
        # sai_model_spec might need adjustment if it expects specific lora types not lycoris
        try:
            sai_metadata = sai_model_spec.build_metadata(
                None, v2, v_parameterization, sdxl, True, 
                is_lycoris=(algo != "lora"), # Pass if it's LyCORIS
                is_lora=(algo == "lora"), # Pass if it's LoRA
                creation_time=time.time(), title=title
            )
            metadata.update(sai_metadata)
        except TypeError as e:
            logger.warning(f"Could not generate full SAI metadata, possibly due to outdated sai_model_spec.py or new flags: {e}")
            logger.warning("Falling back to basic metadata for SAI fields.")
            metadata.update({ # Basic fallback
                "sai_model_name": title,
                "sai_base_model": model_version,
                "sai_is_sdxl": str(sdxl).lower(),
            })


    save_to_file(save_to, lora_module_weights, save_dtype, metadata)
    logger.info(f"{algo.upper()} saved to: {save_to}")


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="Load Stable Diffusion v2.x model")
    parser.add_argument("--v_parameterization", action="store_true", help="Set v-parameterization metadata (defaults to v2 if applicable)")
    parser.add_argument("--sdxl", action="store_true", help="Load Stable Diffusion SDXL base model")
    
    parser.add_argument("--model_org_path", type=str, required=True, help="Path to the original Stable Diffusion model (ckpt/safetensors)")
    parser.add_argument("--model_tuned_path", type=str, required=True, help="Path to the tuned Stable Diffusion model (ckpt/safetensors)")
    parser.add_argument("--save_to", type=str, required=True, help="Output file name for the LoRA/LoHA (ckpt/safetensors)")

    parser.add_argument("--algo", type=str, default="lora", choices=["lora", "loha"], help="Algorithm to use: lora or loha")
    
    parser.add_argument("--network_dim", type=int, default=4, help="Network dimension. For LoRA: rank. For LoHA: 'factor' or 'hada_dim'.")
    parser.add_argument("--network_alpha", type=int, default=None, help="Network alpha. For LoRA: alpha (often same as rank). For LoHA: 'rank_initial' or 'hada_alpha'. Defaults to network_dim if not set.")
    parser.add_argument("--conv_dim", type=int, default=None, help="Conv dimension for conv layers. For LoRA: rank. For LoHA: 'factor'. Defaults to network_dim if not set.")
    parser.add_argument("--conv_alpha", type=int, default=None, help="Conv alpha for conv layers. For LoRA: alpha. For LoHA: 'rank_initial'. Defaults to conv_dim if not set.")

    parser.add_argument("--load_precision", type=str, choices=[None, "float", "fp16", "bf16"], default=None, help="Precision for loading models (None means default float32)")
    parser.add_argument("--save_precision", type=str, choices=[None, "float", "fp16", "bf16"], default=None, help="Precision for saving LoRA/LoHA (None means float32)")
    
    parser.add_argument("--device", type=str, default=None, help="Device for SVD computation (e.g., 'cuda', 'cpu'). If None, defaults to 'cuda' if available, else 'cpu'. SVD results are moved to CPU for storage.")
    parser.add_argument("--clamp_quantile", type=float, default=0.99, help="Quantile for clamping weights (0.0 to 1.0). 1.0 means clamp to max abs value.")
    parser.add_argument("--min_diff", type=float, default=1e-6, help="Minimum weight difference threshold for a module to be considered for extraction.") # Lowered default
    parser.add_argument("--no_metadata", action="store_true", help="Do not save detailed metadata (minimal ss_ metadata will still be saved).")
    
    parser.add_argument("--load_original_model_to", type=str, default=None, help="Device to load original model to (SDXL only, e.g., 'cpu', 'cuda:0')")
    parser.add_argument("--load_tuned_model_to", type=str, default=None, help="Device to load tuned model to (SDXL only, e.g., 'cpu', 'cuda:1')")
    
    parser.add_argument("--dynamic_method", type=str, choices=[None, "sv_ratio", "sv_fro", "sv_cumulative", "sv_knee", "sv_rel_decrease", "sv_cumulative_knee"], default=None, help="Dynamic method to determine rank/alpha (for the first SVD in LoHA). Overrides fixed network_alpha/conv_alpha if set.")
    parser.add_argument("--dynamic_param", type=float, default=0.9, help="Parameter for the chosen dynamic_method (e.g., target ratio/cumulative sum/Frobenius norm percentage).")
    parser.add_argument("--verbose", action="store_true", help="Show detailed SVD info for each module.")
    
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if args.dynamic_method and (args.dynamic_param is None):
        # Default dynamic_param for methods if not specified, or raise error
        if args.dynamic_method in ["sv_cumulative", "sv_fro"]:
            args.dynamic_param = 0.99 # Example: 99% variance/energy
            logger.info(f"Dynamic method {args.dynamic_method} chosen, dynamic_param defaulted to {args.dynamic_param}")
        elif args.dynamic_method in ["sv_ratio"]:
            args.dynamic_param = 1000 # Example: ratio of 1000
            logger.info(f"Dynamic method {args.dynamic_method} chosen, dynamic_param defaulted to {args.dynamic_param}")
        elif args.dynamic_method in ["sv_rel_decrease"]:
            args.dynamic_param = 0.05 # Example: 5% relative decrease
            logger.info(f"Dynamic method {args.dynamic_method} chosen, dynamic_param defaulted to {args.dynamic_param}")
        # sv_knee and sv_cumulative_knee do not require dynamic_param in this implementation.
        elif args.dynamic_method not in ["sv_knee", "sv_cumulative_knee"]:
             parser.error("--dynamic_method requires --dynamic_param for most methods.")


    # Default device selection
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {args.device} for SVD computation.")

    # Rename args for clarity before passing to function
    func_args = vars(args).copy()
    func_args["model_org_path"] = func_args.pop("model_org_path")
    func_args["model_tuned_path"] = func_args.pop("model_tuned_path")
    
    svd_decomposition(**func_args)