import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from tqdm import tqdm
import math
import json
from collections import OrderedDict
import signal
import sys

# --- Global variables (ensure they are defined as before) ---
extracted_loha_state_dict_global = OrderedDict()
layer_optimization_stats_global = []
args_global = None
processed_layers_count_global = 0
skipped_identical_count_global = 0
skipped_other_count_global = 0
keys_scanned_this_run_global = 0 # This will track scans for the outer pbar
save_attempted_on_interrupt = False
outer_pbar_global = None

# --- optimize_loha_for_layer and get_module_shape_info_from_weight (UNCHANGED from your last version) ---
def optimize_loha_for_layer(
    layer_name: str, delta_W_target: torch.Tensor, out_dim: int, in_dim_effective: int,
    k_h: int, k_w: int, rank: int, initial_alpha_val: float, lr: float = 1e-3,
    max_iterations: int = 1000, min_iterations: int = 100, target_loss: float = None,
    weight_decay: float = 1e-4, device: str = 'cuda', dtype: torch.dtype = torch.float32,
    is_conv: bool = True, verbose_layer_debug: bool = False
):
    delta_W_target = delta_W_target.to(device, dtype=dtype)

    if is_conv:
        k_ops = k_h * k_w
        hada_w1_a = nn.Parameter(torch.empty(out_dim, rank, device=device, dtype=dtype))
        hada_w1_b = nn.Parameter(torch.empty(rank, in_dim_effective * k_ops, device=device, dtype=dtype))
        hada_w2_a = nn.Parameter(torch.empty(out_dim, rank, device=device, dtype=dtype))
        hada_w2_b = nn.Parameter(torch.empty(rank, in_dim_effective * k_ops, device=device, dtype=dtype))
    else: # Linear
        hada_w1_a = nn.Parameter(torch.empty(out_dim, rank, device=device, dtype=dtype))
        hada_w1_b = nn.Parameter(torch.empty(rank, in_dim_effective, device=device, dtype=dtype))
        hada_w2_a = nn.Parameter(torch.empty(out_dim, rank, device=device, dtype=dtype))
        hada_w2_b = nn.Parameter(torch.empty(rank, in_dim_effective, device=device, dtype=dtype))

    nn.init.kaiming_uniform_(hada_w1_a, a=math.sqrt(5))
    nn.init.normal_(hada_w1_b, std=0.02)
    nn.init.kaiming_uniform_(hada_w2_a, a=math.sqrt(5))
    nn.init.normal_(hada_w2_b, std=0.02)
    alpha_param = nn.Parameter(torch.tensor(initial_alpha_val, device=device, dtype=dtype))

    optimizer = torch.optim.AdamW(
        [hada_w1_a, hada_w1_b, hada_w2_a, hada_w2_b, alpha_param], lr=lr, weight_decay=weight_decay
    )
    patience_epochs = max(10, int(max_iterations * 0.05))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_epochs, factor=0.5, min_lr=1e-7, verbose=False)

    iter_pbar = tqdm(range(max_iterations), desc=f"Opt: {layer_name}", leave=False, dynamic_ncols=True, position=1)
    
    final_loss = float('inf')
    stopped_early_by_loss = False
    iterations_actually_done = 0

    for i in iter_pbar:
        iterations_actually_done = i + 1
        if save_attempted_on_interrupt:
            print(f"\n  Interrupt during opt of {layer_name}. Stopping this layer after {i} iters.")
            break

        optimizer.zero_grad()
        eff_alpha_scale = alpha_param / rank

        if is_conv:
            term1_flat = hada_w1_a @ hada_w1_b; term1_reshaped = term1_flat.view(out_dim, in_dim_effective, k_h, k_w)
            term2_flat = hada_w2_a @ hada_w2_b; term2_reshaped = term2_flat.view(out_dim, in_dim_effective, k_h, k_w)
            delta_W_loha = eff_alpha_scale * term1_reshaped * term2_reshaped
        else:
            term1 = hada_w1_a @ hada_w1_b; term2 = hada_w2_a @ hada_w2_b
            delta_W_loha = eff_alpha_scale * term1 * term2

        loss = F.mse_loss(delta_W_loha, delta_W_target)
        final_loss = loss.item()
        loss.backward(); optimizer.step(); scheduler.step(loss)

        current_lr = optimizer.param_groups[0]['lr']
        iter_pbar.set_postfix_str(f"Loss={final_loss:.3e}, AlphaP={alpha_param.item():.2f}, LR={current_lr:.1e}", refresh=True)

        if verbose_layer_debug and (i == 0 or (i + 1) % (max_iterations // 10 if max_iterations >= 10 else 1) == 0 or i == max_iterations - 1):
            iter_pbar.write(f"  Debug {layer_name} - Iter {i+1}/{max_iterations}: Loss: {final_loss:.6e}, LR: {current_lr:.2e}, AlphaP: {alpha_param.item():.4f}")

        if target_loss is not None and i >= min_iterations -1 :
            if final_loss <= target_loss:
                if verbose_layer_debug or (args_global and args_global.verbose):
                     iter_pbar.write(f"  Target loss {target_loss:.2e} reached for {layer_name} at iter {i+1}.")
                stopped_early_by_loss = True; break
    
    if not save_attempted_on_interrupt :
         iter_pbar.set_description_str(f"Opt: {layer_name} (Done)")
         iter_pbar.set_postfix_str(f"FinalLoss={final_loss:.2e}, It={iterations_actually_done}{', EarlyStop' if stopped_early_by_loss else ''}")
    iter_pbar.close()

    if save_attempted_on_interrupt and not stopped_early_by_loss and iterations_actually_done < max_iterations:
         return {'final_loss': final_loss, 'stopped_early': False, 'iterations_done': iterations_actually_done, 'interrupted_mid_layer': True}
    return {
        'hada_w1_a': hada_w1_a.data.cpu().contiguous(), 'hada_w1_b': hada_w1_b.data.cpu().contiguous(),
        'hada_w2_a': hada_w2_a.data.cpu().contiguous(), 'hada_w2_b': hada_w2_b.data.cpu().contiguous(),
        'alpha': alpha_param.data.cpu().contiguous(), 'final_loss': final_loss,
        'stopped_early': stopped_early_by_loss, 'iterations_done': iterations_actually_done,
        'interrupted_mid_layer': False
    }

def get_module_shape_info_from_weight(weight_tensor: torch.Tensor):
    if len(weight_tensor.shape) == 4: is_conv = True; out_dim, in_dim_effective, k_h, k_w = weight_tensor.shape; groups = 1; return out_dim, in_dim_effective, k_h, k_w, groups, is_conv
    elif len(weight_tensor.shape) == 2: is_conv = False; out_dim, in_dim = weight_tensor.shape; return out_dim, in_dim, None, None, 1, is_conv
    return None

# --- perform_graceful_save (UNCHANGED from previous version) ---
def perform_graceful_save(output_path_override=None):
    global extracted_loha_state_dict_global, layer_optimization_stats_global, args_global
    global processed_layers_count_global, save_attempted_on_interrupt
    global skipped_identical_count_global, skipped_other_count_global, keys_scanned_this_run_global

    if not extracted_loha_state_dict_global: print("No layers were processed enough to save."); return
    args_to_use = args_global
    if not args_to_use: print("Error: Global args not available for saving metadata."); return
    final_save_path = output_path_override if output_path_override else args_to_use.save_to
    if args_to_use.save_weights_dtype == "fp16": final_save_dtype_torch = torch.float16
    elif args_to_use.save_weights_dtype == "bf16": final_save_dtype_torch = torch.bfloat16
    else: final_save_dtype_torch = torch.float32
    final_state_dict_to_save = OrderedDict()
    for k, v_tensor in extracted_loha_state_dict_global.items():
        if v_tensor.is_floating_point(): final_state_dict_to_save[k] = v_tensor.to(final_save_dtype_torch)
        else: final_state_dict_to_save[k] = v_tensor
    print(f"\nAttempting to save {len(final_state_dict_to_save)} LoHA param sets for {processed_layers_count_global} layers to {final_save_path}")
    eff_global_network_alpha_val = args_to_use.initial_alpha; eff_global_network_alpha_str = f"{eff_global_network_alpha_val:.8f}"
    global_rank_str = str(args_to_use.rank)
    conv_rank_str = str(args_to_use.conv_rank if args_to_use.conv_rank is not None else args_to_use.rank)
    eff_conv_alpha_val = args_to_use.initial_conv_alpha; conv_alpha_str = f"{eff_conv_alpha_val:.8f}"
    network_args_dict = {
        "algo": "loha", "dim": global_rank_str, "alpha": eff_global_network_alpha_str,
        "conv_dim": conv_rank_str, "conv_alpha": conv_alpha_str,
        "dropout": str(args_to_use.dropout), "rank_dropout": str(args_to_use.rank_dropout), "module_dropout": str(args_to_use.module_dropout),
        "use_tucker": "false", "use_scalar": "false", "block_size": "1",}
    sf_metadata = {
        "ss_network_module": "lycoris.kohya", "ss_network_rank": global_rank_str,
        "ss_network_alpha": eff_global_network_alpha_str, "ss_network_algo": "loha",
        "ss_network_args": json.dumps(network_args_dict),
        "ss_comment": f"Extracted LoHA (Interrupt: {save_attempted_on_interrupt}). OptPrec: {args_to_use.precision}. SaveDtype: {args_to_use.save_weights_dtype}. ATOL: {args_to_use.atol_fp32_check}. Layers: {processed_layers_count_global}. MaxIter: {args_to_use.max_iterations}. TargetLoss: {args_to_use.target_loss}",
        "ss_base_model_name": os.path.splitext(os.path.basename(args_to_use.base_model_path))[0],
        "ss_ft_model_name": os.path.splitext(os.path.basename(args_to_use.ft_model_path))[0],
        "ss_save_weights_dtype": args_to_use.save_weights_dtype, "ss_optimization_precision": args_to_use.precision,}
    json_metadata_for_file = {
        "comfyui_lora_type": "LyCORIS_LoHa", "model_name": os.path.splitext(os.path.basename(final_save_path))[0],
        "base_model_path": args_to_use.base_model_path, "ft_model_path": args_to_use.ft_model_path,
        "loha_extraction_settings": {k: str(v) if isinstance(v, type(os.pathsep)) else v for k,v in vars(args_to_use).items()},
        "extraction_summary":{"processed_layers_count": processed_layers_count_global, "skipped_identical_count": skipped_identical_count_global, "skipped_other_count": skipped_other_count_global, "total_candidate_keys_scanned": keys_scanned_this_run_global,},
        "layer_optimization_details": layer_optimization_stats_global, "embedded_safetensors_metadata": sf_metadata, "interrupted_save": save_attempted_on_interrupt}
    if final_save_path.endswith(".safetensors"):
        try: save_file(final_state_dict_to_save, final_save_path, metadata=sf_metadata); print(f"LoHA state_dict saved to: {final_save_path}")
        except Exception as e: print(f"Error saving .safetensors file: {e}"); return
        metadata_json_file_path = os.path.splitext(final_save_path)[0] + "_extraction_metadata.json"
        try:
            with open(metadata_json_file_path, 'w') as f: json.dump(json_metadata_for_file, f, indent=4)
            print(f"Extended metadata saved to: {metadata_json_file_path}")
        except Exception as e: print(f"Could not save extended metadata JSON: {e}")
    else: print(f"Saving to .pt not fully supported with interrupt metadata.")

# --- handle_interrupt (UNCHANGED from previous version) ---
def handle_interrupt(signum, frame):
    global save_attempted_on_interrupt, outer_pbar_global
    print("\n" + "="*30 + "\nCtrl+C (SIGINT) detected!\n" + "="*30)
    if save_attempted_on_interrupt: print("Save already attempted. Force exiting."); os._exit(1); return
    save_attempted_on_interrupt = True
    if outer_pbar_global: outer_pbar_global.close()
    print("Attempting to save progress for processed layers...")
    perform_graceful_save()
    print("Graceful save attempt finished. Exiting.")
    sys.exit(0)


def main(cli_args):
    global args_global, extracted_loha_state_dict_global, layer_optimization_stats_global
    global processed_layers_count_global, save_attempted_on_interrupt, outer_pbar_global
    global skipped_identical_count_global, skipped_other_count_global, keys_scanned_this_run_global
    
    args_global = cli_args
    signal.signal(signal.SIGINT, handle_interrupt)

    if args_global.precision == "fp16": target_opt_dtype = torch.float16
    elif args_global.precision == "bf16": target_opt_dtype = torch.bfloat16
    else: target_opt_dtype = torch.float32

    if args_global.save_weights_dtype == "fp16": final_save_dtype = torch.float16
    elif args_global.save_weights_dtype == "bf16": final_save_dtype = torch.bfloat16
    else: final_save_dtype = torch.float32
    
    print(f"Using device: {args_global.device}, Opt Dtype: {target_opt_dtype}, Save Dtype: {final_save_dtype}")
    if args_global.target_loss: print(f"Target Loss: {args_global.target_loss:.2e} (after {args_global.min_iterations} min iters)")
    print(f"Max Iters/Layer: {args_global.max_iterations}")

    print(f"Loading base: {args_global.base_model_path}")
    if args_global.base_model_path.endswith(".safetensors"): base_model_sd = load_file(args_global.base_model_path, device='cpu')
    else: base_model_sd = torch.load(args_global.base_model_path, map_location='cpu'); base_model_sd = base_model_sd.get('state_dict', base_model_sd)
    
    print(f"Loading fine-tuned: {args_global.ft_model_path}")
    if args_global.ft_model_path.endswith(".safetensors"): ft_model_sd = load_file(args_global.ft_model_path, device='cpu')
    else: ft_model_sd = torch.load(args_global.ft_model_path, map_location='cpu'); ft_model_sd = ft_model_sd.get('state_dict', ft_model_sd)

    extracted_loha_state_dict_global = OrderedDict()
    layer_optimization_stats_global = []
    processed_layers_count_global = 0; skipped_identical_count_global = 0; skipped_other_count_global = 0; keys_scanned_this_run_global = 0

    all_candidate_keys = []
    for k in base_model_sd.keys():
        if k.endswith('.weight') and k in ft_model_sd and (len(base_model_sd[k].shape) == 2 or len(base_model_sd[k].shape) == 4):
            all_candidate_keys.append(k)
        elif k.endswith('.weight') and k not in ft_model_sd and args_global.verbose:
            print(f"Note: Base key '{k}' not in FT model.")
    all_candidate_keys.sort()
    total_candidate_keys_to_scan = len(all_candidate_keys)
    
    print(f"Found {total_candidate_keys_to_scan} candidate '.weight' keys common to both models and of suitable shape.")

    # The outer progress bar will now track the scan through ALL candidate keys
    outer_pbar_global = tqdm(total=total_candidates_to_scan, desc=f"Scanning Layers (Processed 0)", dynamic_ncols=True, position=0)
    
    try:
        for key_name in all_candidate_keys:
            if save_attempted_on_interrupt: break
            keys_scanned_this_run_global += 1
            outer_pbar_global.update(1) # Update for every key scanned

            if args_global.max_layers is not None and args_global.max_layers > 0 and processed_layers_count_global >= args_global.max_layers:
                if args_global.verbose: print(f"\nReached max_layers limit ({args_global.max_layers}). All further candidate layers will be skipped by the scan.")
                # No 'break' here, let the loop finish scanning so the pbar completes to 100% of total_candidates_to_scan
                # The actual optimization will be skipped by the condition above.
                # Update description to show scanning is finishing due to max_layers
                outer_pbar_global.set_description_str(f"Scan (Max Layers Reached: {processed_layers_count_global}/{args_global.max_layers}, Scanned {keys_scanned_this_run_global}/{total_candidates_to_scan})")
                skipped_other_count_global +=1 # Count as skipped_other if we don't even check identical
                continue # Continue to scan remaining keys but don't process them


            base_W = base_model_sd[key_name].to(dtype=torch.float32)
            ft_W = ft_model_sd[key_name].to(dtype=torch.float32)

            if base_W.shape != ft_W.shape: 
                if args_global.verbose: print(f"\nSkipping {key_name} (scan): shape mismatch.")
                skipped_other_count_global +=1; continue
            shape_info = get_module_shape_info_from_weight(base_W)
            if shape_info is None: 
                if args_global.verbose: print(f"\nSkipping {key_name} (scan): not Linear or Conv2d.")
                skipped_other_count_global +=1; continue

            delta_W_fp32 = (ft_W - base_W)
            if torch.allclose(delta_W_fp32, torch.zeros_like(delta_W_fp32), atol=args_global.atol_fp32_check):
                if args_global.verbose: print(f"\nSkipping {key_name} (scan): weights effectively identical.")
                skipped_identical_count_global += 1; continue
            
            # This layer WILL be processed. Update description.
            max_layers_target_str = f"/{args_global.max_layers}" if args_global.max_layers is not None and args_global.max_layers > 0 else ""
            outer_pbar_global.set_description_str(f"Optimizing L{processed_layers_count_global + 1}{max_layers_target_str} (Scanned {keys_scanned_this_run_global}/{total_candidates_to_scan})")

            original_module_path = key_name[:-len(".weight")]
            loha_key_prefix = ""
            if original_module_path.startswith("model.diffusion_model."): loha_key_prefix = "lora_unet_" + original_module_path[len("model.diffusion_model."):].replace(".", "_")
            elif original_module_path.startswith("conditioner.embedders.0.transformer."): loha_key_prefix = "lora_te1_" + original_module_path[len("conditioner.embedders.0.transformer."):].replace(".", "_")
            elif original_module_path.startswith("conditioner.embedders.1.model.transformer."): loha_key_prefix = "lora_te2_" + original_module_path[len("conditioner.embedders.1.model.transformer."):].replace(".", "_")
            else: loha_key_prefix = "lora_" + original_module_path.replace(".", "_")

            if args_global.verbose: tqdm.write(f"\n  Orig: {key_name} -> LoHA: {loha_key_prefix}") # Use tqdm.write for multiline

            out_dim, in_dim_effective, k_h, k_w, _, is_conv = shape_info
            delta_W_target_for_opt = delta_W_fp32.to(dtype=target_opt_dtype)
            current_rank = args_global.conv_rank if is_conv and args_global.conv_rank is not None else args_global.rank
            current_initial_alpha = args_global.initial_conv_alpha if is_conv else args_global.initial_alpha
            
            tqdm.write(f"Optimizing Layer {processed_layers_count_global + 1}{max_layers_target_str}: {loha_key_prefix} (Orig: {original_module_path}, Shp: {list(base_W.shape)}, R: {current_rank}, Alpha_init: {current_initial_alpha:.1f})")


            try:
                opt_results = optimize_loha_for_layer(
                    layer_name=loha_key_prefix, delta_W_target=delta_W_target_for_opt,
                    out_dim=out_dim, in_dim_effective=in_dim_effective, k_h=k_h, k_w=k_w, rank=current_rank,
                    initial_alpha_val=current_initial_alpha, lr=args_global.lr,
                    max_iterations=args_global.max_iterations, min_iterations=args_global.min_iterations,
                    target_loss=args_global.target_loss, weight_decay=args_global.weight_decay,
                    device=args_global.device, dtype=target_opt_dtype, is_conv=is_conv, 
                    verbose_layer_debug=args_global.verbose_layer_debug
                )
                
                if not opt_results.get('interrupted_mid_layer'):
                    for p_name, p_val in opt_results.items():
                        if p_name not in ['final_loss', 'stopped_early', 'iterations_done', 'interrupted_mid_layer']:
                            extracted_loha_state_dict_global[f'{loha_key_prefix}.{p_name}'] = p_val.to(final_save_dtype)
                    
                    layer_optimization_stats_global.append({
                        "name": loha_key_prefix, "original_name": original_module_path,
                        "final_loss": opt_results['final_loss'], "iterations_done": opt_results['iterations_done'],
                        "stopped_early_by_loss_target": opt_results['stopped_early']
                    })
                    tqdm.write(f"  Layer {loha_key_prefix} Done. Loss: {opt_results['final_loss']:.4e}, Iters: {opt_results['iterations_done']}{', Stopped by Loss' if opt_results['stopped_early'] else ''}")

                    if args_global.use_bias:
                        original_bias_key = f"{original_module_path}.bias"
                        if original_bias_key in ft_model_sd and original_bias_key in base_model_sd:
                            base_B = base_model_sd[original_bias_key].to(dtype=torch.float32); ft_B = ft_model_sd[original_bias_key].to(dtype=torch.float32)
                            if not torch.allclose(base_B, ft_B, atol=args_global.atol_fp32_check):
                                extracted_loha_state_dict_global[original_bias_key] = ft_B.cpu().to(final_save_dtype)
                                if args_global.verbose: tqdm.write(f"    Saved differing bias for {original_bias_key}")
                        elif original_bias_key in ft_model_sd and original_bias_key not in base_model_sd:
                            if args_global.verbose: tqdm.write(f"    Bias {original_bias_key} in FT only. Saving.")
                            extracted_loha_state_dict_global[original_bias_key] = ft_model_sd[original_bias_key].cpu().to(final_save_dtype)
                    
                    processed_layers_count_global += 1
                    # Outer pbar update is now at the start of the loop for each scanned key.
                    # The description will reflect the processed count.
                else:
                     if args_global.verbose: tqdm.write(f"  Opt for {loha_key_prefix} interrupted; not saving params.")
            except Exception as e:
                print(f"\nError during optimization for {original_module_path} ({loha_key_prefix}): {e}")
                import traceback; traceback.print_exc()
                skipped_other_count_global +=1
        
    finally:
        if outer_pbar_global:
            if outer_pbar_global.n < outer_pbar_global.total: # Fill to 100% if loop broke early
                outer_pbar_global.update(outer_pbar_global.total - outer_pbar_global.n)
            outer_pbar_global.close()

    if not save_attempted_on_interrupt: # Normal completion
        print("\n--- Final Optimization Summary (Normal Completion) ---")
        for stat in layer_optimization_stats_global:
            print(f"Layer: {stat['name']}, Final Loss: {stat['final_loss']:.4e}, Iters: {stat['iterations_done']}{', Stopped by Loss' if stat['stopped_early_by_loss_target'] else ''}")
        print(f"\n--- Overall Summary ---")
        print(f"Total candidate weight keys: {total_candidates_to_scan}")
        print(f"Total keys scanned by loop: {keys_scanned_this_run_global}")
        print(f"Processed {processed_layers_count_global} layers for LoHA extraction.")
        print(f"Skipped {skipped_identical_count_global} layers (identical).")
        print(f"Skipped {skipped_other_count_global} layers (other reasons).")
        perform_graceful_save()
    else: # Interrupted
        print("\nProcess was interrupted. Saved data for fully completed layers.")

# --- __main__ block (UNCHANGED from previous version) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LoHA parameters by optimizing against weight differences.")
    parser.add_argument("base_model_path", type=str, help="Path to the base model state_dict file (.pt, .pth, .safetensors)")
    parser.add_argument("ft_model_path", type=str, help="Path to the fine-tuned model state_dict file (.pt, .pth, .safetensors)")
    parser.add_argument("save_to", type=str, help="Path to save the extracted LoHA file (recommended .safetensors)")

    parser.add_argument("--rank", type=int, default=4, help="Default rank for LoHA decomposition (used for linear layers and as fallback for conv).")
    parser.add_argument("--conv_rank", type=int, default=None, help="Specific rank for convolutional LoHA layers. Defaults to --rank if not set.")
    
    parser.add_argument("--initial_alpha", type=float, default=None,
                        help="Global initial alpha for optimization (used for linear and as fallback for conv). Defaults to 'rank'. This is also used for 'ss_network_alpha'.")
    parser.add_argument("--initial_conv_alpha", type=float, default=None,
                        help="Specific initial alpha for convolutional LoHA layers. Defaults to '--initial_alpha' or conv_rank if neither initial_alpha nor initial_conv_alpha is set, it defaults to the respective rank.")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for LoHA optimization per layer.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of optimization iterations per layer.")
    parser.add_argument("--min_iterations", type=int, default=100, help="Minimum number of optimization iterations per layer before checking target loss. Default 100.")
    parser.add_argument("--target_loss", type=float, default=None, help="Target MSE loss to achieve for stopping optimization for a layer. If None, runs for max_iterations.")
    
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for LoHA optimization.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for computation ('cuda' or 'cpu').")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                        help="Computation precision for LoHA optimization. This is 'ss_mixed_precision'. Default: fp32")
    parser.add_argument("--save_weights_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"],
                        help="Data type for saving the final LoHA weights in the .safetensors file. Default: bf16.")
    parser.add_argument("--atol_fp32_check", type=float, default=1e-6,
                        help="Absolute tolerance for fp32 weight difference check to consider layers identical and skip them.")
    parser.add_argument("--use_bias", action="store_true", help="If set, save fine-tuned bias terms if they differ from the base model's bias (saved with original key names).")
    
    parser.add_argument("--dropout", type=float, default=0.0, help="General dropout for LoHA modules (for metadata).")
    parser.add_argument("--rank_dropout", type=float, default=0.0, help="Rank dropout rate for LoHA modules (for metadata).")
    parser.add_argument("--module_dropout", type=float, default=0.0, help="Module dropout rate for LoHA modules (for metadata).")

    parser.add_argument("--max_layers", type=int, default=None,
                        help="Process at most N differing layers for quick testing. Layers are sorted by name before processing.")
    parser.add_argument("--verbose", action="store_true", help="Print general verbose information during processing.")
    parser.add_argument("--verbose_layer_debug", action="store_true",
                        help="Print very detailed per-iteration debug info for each layer's optimization (can be very spammy).")
    
    parsed_args = parser.parse_args()
    
    if not os.path.exists(parsed_args.base_model_path): print(f"Error: Base model path not found: {parsed_args.base_model_path}"); exit(1)
    if not os.path.exists(parsed_args.ft_model_path): print(f"Error: Fine-tuned model path not found: {parsed_args.ft_model_path}"); exit(1)
    
    save_dir = os.path.dirname(parsed_args.save_to)
    if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True); print(f"Created directory: {save_dir}")
    
    if parsed_args.initial_alpha is None: parsed_args.initial_alpha = float(parsed_args.rank)
    conv_rank_for_alpha_default = parsed_args.conv_rank if parsed_args.conv_rank is not None else parsed_args.rank
    if parsed_args.initial_conv_alpha is None: parsed_args.initial_conv_alpha = parsed_args.initial_alpha

    main(parsed_args)