import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
import safetensors # Import the main library to use safetensors.safe_open
from tqdm import tqdm
import math
import json
from collections import OrderedDict
import signal
import sys
import glob

# --- Global variables ---
extracted_loha_state_dict_global = OrderedDict()
layer_optimization_stats_global = []
args_global = None
processed_layers_this_session_count_global = 0
previously_completed_module_prefixes_global = set()
all_completed_module_prefixes_ever_global = set() # Tracks all module prefixes ever completed (resumed + current)
skipped_identical_count_global = 0
skipped_other_reason_count_global = 0
keys_scanned_this_run_global = 0
save_attempted_on_interrupt = False
outer_pbar_global = None
main_loop_completed_scan_flag_global = False # True if the main key loop finished a full scan

# --- MODIFIED: optimize_loha_for_layer ---
def optimize_loha_for_layer(
    layer_name: str, delta_W_target: torch.Tensor, out_dim: int, in_dim_effective: int,
    k_h: int, k_w: int, initial_rank_for_layer: int, initial_alpha_for_layer: float,
    lr: float = 1e-3, max_iterations: int = 1000, min_iterations: int = 100,
    target_loss: float = None, weight_decay: float = 1e-4,
    device: str = 'cuda', dtype: torch.dtype = torch.float32,
    is_conv: bool = True, verbose_layer_debug: bool = False,
    max_rank_retries: int = 0, 
    rank_increase_factor: float = 1.25 
):
    delta_W_target = delta_W_target.to(device, dtype=dtype)
        
    best_result_so_far = {
        'final_loss': float('inf'),
        'stopped_early': False,
        'iterations_done': 0,
        'final_rank_used': initial_rank_for_layer,
        'interrupted_mid_layer': False
    }

    # Variables for the current attempt being processed in the loop
    current_rank_for_this_attempt = initial_rank_for_layer
    alpha_init_for_this_attempt = initial_alpha_for_layer
    # Stores the rank that was actually used in the immediately preceding completed attempt,
    # used as the base for calculating the next rank increase.
    rank_base_for_next_increase = initial_rank_for_layer 

    for attempt in range(max_rank_retries + 1): 
        if save_attempted_on_interrupt:
            if attempt == 0: 
                 return {**best_result_so_far, 'interrupted_mid_layer': True, 'iterations_done': 0}
            else: 
                 best_result_so_far['interrupted_mid_layer'] = True
                 return best_result_so_far

        if attempt > 0: 
            # rank_base_for_next_increase holds the rank of the attempt that just finished. The loss printed is the best overall for context.
            tqdm.write(f"  Retrying {layer_name}: rank {rank_base_for_next_increase} (best loss so far: {best_result_so_far.get('final_loss', float('inf')):.2e}) didn't meet target or ran max iterations. Increasing rank.")
            
            new_rank_float = rank_base_for_next_increase * rank_increase_factor
            increased_rank = math.ceil(new_rank_float)
            current_rank_for_this_attempt = max(rank_base_for_next_increase + 1, increased_rank)

            original_alpha_to_rank_ratio = 1.0 
            if initial_rank_for_layer > 0: 
                original_alpha_to_rank_ratio = initial_alpha_for_layer / float(initial_rank_for_layer)
                alpha_init_for_this_attempt = original_alpha_to_rank_ratio * float(current_rank_for_this_attempt)
            else: 
                alpha_init_for_this_attempt = float(current_rank_for_this_attempt) 

            tqdm.write(f"    Rank for {layer_name} increased from {rank_base_for_next_increase} to {current_rank_for_this_attempt}. Initial alpha for this attempt adjusted proportionally to {alpha_init_for_this_attempt:.2f} (Original ratio to rank: {original_alpha_to_rank_ratio:.2f}).")
 
        # Parameter Initialization
        if is_conv:
            k_ops = k_h * k_w
            hada_w1_a = nn.Parameter(torch.empty(out_dim, current_rank_for_this_attempt, device=device, dtype=dtype)); nn.init.kaiming_uniform_(hada_w1_a, a=math.sqrt(5))
            hada_w1_b = nn.Parameter(torch.empty(current_rank_for_this_attempt, in_dim_effective * k_ops, device=device, dtype=dtype)); nn.init.normal_(hada_w1_b, std=0.02)
            hada_w2_a = nn.Parameter(torch.empty(out_dim, current_rank_for_this_attempt, device=device, dtype=dtype)); nn.init.kaiming_uniform_(hada_w2_a, a=math.sqrt(5))
            hada_w2_b = nn.Parameter(torch.empty(current_rank_for_this_attempt, in_dim_effective * k_ops, device=device, dtype=dtype)); nn.init.normal_(hada_w2_b, std=0.02)
        else: # Linear
            hada_w1_a = nn.Parameter(torch.empty(out_dim, current_rank_for_this_attempt, device=device, dtype=dtype)); nn.init.kaiming_uniform_(hada_w1_a, a=math.sqrt(5))
            hada_w1_b = nn.Parameter(torch.empty(current_rank_for_this_attempt, in_dim_effective, device=device, dtype=dtype)); nn.init.normal_(hada_w1_b, std=0.02)
            hada_w2_a = nn.Parameter(torch.empty(out_dim, current_rank_for_this_attempt, device=device, dtype=dtype)); nn.init.kaiming_uniform_(hada_w2_a, a=math.sqrt(5))
            hada_w2_b = nn.Parameter(torch.empty(current_rank_for_this_attempt, in_dim_effective, device=device, dtype=dtype)); nn.init.normal_(hada_w2_b, std=0.02)
        
        alpha_param = nn.Parameter(torch.tensor(alpha_init_for_this_attempt, device=device, dtype=dtype))
        params_to_optimize = [hada_w1_a, hada_w1_b, hada_w2_a, hada_w2_b, alpha_param]
        optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
        patience_epochs = max(10, int(max_iterations * 0.05))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_epochs, factor=0.5, min_lr=1e-7, verbose=False)
        
        iter_pbar_desc_prefix = f"Opt Att {attempt+1}/{max_rank_retries+1} (R:{current_rank_for_this_attempt})"
        iter_pbar = tqdm(range(max_iterations), desc=f"{iter_pbar_desc_prefix}: {layer_name}", leave=False, dynamic_ncols=True, position=1, mininterval=0.5)
        
        current_attempt_final_loss = float('inf')
        current_attempt_stopped_early_by_loss = False
        current_attempt_iterations_done = 0

        for i in iter_pbar:
            current_attempt_iterations_done = i + 1
            if save_attempted_on_interrupt:
                iter_pbar.close()
                tqdm.write(f"\n  Interrupt during opt of {layer_name} (Rank {current_rank_for_this_attempt}, Iter {i}). Stopping layer.")
                if current_attempt_final_loss < best_result_so_far['final_loss'] and i > 0 :
                     best_result_so_far.update({
                        'final_loss': current_attempt_final_loss, 
                        'stopped_early': False, 
                        'iterations_done': current_attempt_iterations_done,
                        'final_rank_used': current_rank_for_this_attempt,
                     })
                best_result_so_far['interrupted_mid_layer'] = True
                return best_result_so_far

            optimizer.zero_grad()
            eff_alpha_scale = alpha_param / current_rank_for_this_attempt 
            
            if is_conv:
                term1_flat = hada_w1_a @ hada_w1_b; term1_reshaped = term1_flat.view(out_dim, in_dim_effective, k_h, k_w)
                term2_flat = hada_w2_a @ hada_w2_b; term2_reshaped = term2_flat.view(out_dim, in_dim_effective, k_h, k_w)
                delta_W_loha = eff_alpha_scale * term1_reshaped * term2_reshaped
            else:
                term1 = hada_w1_a @ hada_w1_b; term2 = hada_w2_a @ hada_w2_b
                delta_W_loha = eff_alpha_scale * term1 * term2
            
            loss = F.mse_loss(delta_W_loha, delta_W_target)
            current_attempt_final_loss = loss.item()
            loss.backward(); optimizer.step(); scheduler.step(loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            iter_pbar.set_postfix_str(f"Loss={current_attempt_final_loss:.3e}, AlphaP={alpha_param.item():.2f}, LR={current_lr:.1e}", refresh=True)
            if verbose_layer_debug and (i == 0 or (i + 1) % (max_iterations // 10 if max_iterations >= 10 else 1) == 0 or i == max_iterations - 1):
                iter_pbar.write(f"  Debug {layer_name} (R:{current_rank_for_this_attempt}) - Iter {i+1}/{max_iterations}: Loss: {current_attempt_final_loss:.6e}, LR: {current_lr:.2e}, AlphaP: {alpha_param.item():.4f}")
            
            if target_loss is not None and i >= min_iterations -1 and current_attempt_final_loss <= target_loss:
                if verbose_layer_debug or (args_global and args_global.verbose): iter_pbar.write(f"  Target loss {target_loss:.2e} reached for {layer_name} (R:{current_rank_for_this_attempt}) at iter {i+1}.")
                current_attempt_stopped_early_by_loss = True
                break 
        
        iter_pbar_final_desc = f"{iter_pbar_desc_prefix}: {layer_name} (Done)"
        iter_pbar_final_postfix = f"FinalLoss={current_attempt_final_loss:.2e}, It={current_attempt_iterations_done}{', EarlyStop' if current_attempt_stopped_early_by_loss else ''}"
        if not save_attempted_on_interrupt: iter_pbar.set_description_str(iter_pbar_final_desc); iter_pbar.set_postfix_str(iter_pbar_final_postfix)
        iter_pbar.close()

        if save_attempted_on_interrupt: 
            best_result_so_far['interrupted_mid_layer'] = True
            if current_attempt_final_loss < best_result_so_far['final_loss'] and current_attempt_iterations_done > 0:
                 best_result_so_far.update({
                    'final_loss': current_attempt_final_loss,
                    'stopped_early': False,
                    'iterations_done': current_attempt_iterations_done,
                    'final_rank_used': current_rank_for_this_attempt,
                 })
            return best_result_so_far

        if current_attempt_final_loss < best_result_so_far['final_loss']:
            best_result_so_far = {
                'hada_w1_a': hada_w1_a.data.cpu().contiguous(), 'hada_w1_b': hada_w1_b.data.cpu().contiguous(),
                'hada_w2_a': hada_w2_a.data.cpu().contiguous(), 'hada_w2_b': hada_w2_b.data.cpu().contiguous(),
                'alpha': alpha_param.data.cpu().contiguous(), 
                'final_loss': current_attempt_final_loss,
                'stopped_early': current_attempt_stopped_early_by_loss, 
                'iterations_done': current_attempt_iterations_done,
                'final_rank_used': current_rank_for_this_attempt,
                'interrupted_mid_layer': False 
            }
        elif current_attempt_final_loss == best_result_so_far['final_loss'] and current_rank_for_this_attempt < best_result_so_far['final_rank_used']:
             best_result_so_far = {
                'hada_w1_a': hada_w1_a.data.cpu().contiguous(), 'hada_w1_b': hada_w1_b.data.cpu().contiguous(),
                'hada_w2_a': hada_w2_a.data.cpu().contiguous(), 'hada_w2_b': hada_w2_b.data.cpu().contiguous(),
                'alpha': alpha_param.data.cpu().contiguous(), 
                'final_loss': current_attempt_final_loss,
                'stopped_early': current_attempt_stopped_early_by_loss, 
                'iterations_done': current_attempt_iterations_done,
                'final_rank_used': current_rank_for_this_attempt,
                'interrupted_mid_layer': False 
            }

        # The rank used in this just-completed attempt becomes the base for the next increase, if any.
        rank_base_for_next_increase = current_rank_for_this_attempt

        if target_loss is not None and current_attempt_final_loss <= target_loss:
            if args_global and args_global.verbose: tqdm.write(f"  Layer {layer_name} (R:{current_rank_for_this_attempt}) met target loss. Using this result.")
            break 
        
        if current_attempt_iterations_done < max_iterations and not current_attempt_stopped_early_by_loss:
            if args_global and args_global.verbose: tqdm.write(f"  Layer {layer_name} (R:{current_rank_for_this_attempt}) finished with {current_attempt_iterations_done} iterations (less than max_iterations {max_iterations}) and did not meet target loss. Not retrying with increased rank.")
            break 

        if attempt >= max_rank_retries: 
            if target_loss is not None and current_attempt_final_loss > target_loss:
                 if args_global and args_global.verbose: tqdm.write(f"  Layer {layer_name} (R:{current_rank_for_this_attempt}) did not meet target loss after max attempts. Using best result found (Loss: {best_result_so_far['final_loss']:.2e}, Rank: {best_result_so_far['final_rank_used']}).")
            elif target_loss is None:
                 if args_global and args_global.verbose: tqdm.write(f"  Layer {layer_name} (R:{current_rank_for_this_attempt}) completed all attempts without target loss. Using best result (Loss: {best_result_so_far['final_loss']:.2e}, Rank: {best_result_so_far['final_rank_used']}).")
            break 
            
        if target_loss is not None and current_attempt_final_loss > target_loss and current_attempt_iterations_done >= max_iterations:
            if args_global and args_global.verbose: tqdm.write(f"  Layer {layer_name} (R:{current_rank_for_this_attempt}) loss {current_attempt_final_loss:.2e} > target {target_loss:.2e} after {current_attempt_iterations_done} iters. Will try next attempt if available.")
        elif target_loss is None and current_attempt_iterations_done >= max_iterations: 
            if args_global and args_global.verbose and attempt < max_rank_retries: 
                tqdm.write(f"  Layer {layer_name} (R:{current_rank_for_this_attempt}) completed max iterations ({current_attempt_iterations_done}) for this attempt. Proceeding to next attempt with increased rank as max_rank_retries allows.")
            elif args_global and args_global.verbose: 
                 tqdm.write(f"  Layer {layer_name} (R:{current_rank_for_this_attempt}) completed max iterations ({current_attempt_iterations_done}) for this attempt (last one). Using this result.")
                 break 


    if 'hada_w1_a' not in best_result_so_far: 
        return {
            'final_loss': best_result_so_far.get('final_loss', float('inf')),
            'stopped_early': False,
            'iterations_done': best_result_so_far.get('iterations_done', 0),
            'interrupted_mid_layer': best_result_so_far.get('interrupted_mid_layer', True), 
            'final_rank_used': best_result_so_far.get('final_rank_used', initial_rank_for_layer)
        }
    return best_result_so_far

def get_module_shape_info_from_weight(weight_tensor: torch.Tensor):
    if len(weight_tensor.shape) == 4: is_conv = True; out_dim, in_dim_effective, k_h, k_w = weight_tensor.shape; groups = 1; return out_dim, in_dim_effective, k_h, k_w, groups, is_conv
    elif len(weight_tensor.shape) == 2: is_conv = False; out_dim, in_dim = weight_tensor.shape; return out_dim, in_dim, None, None, 1, is_conv
    return None

def generate_intermediate_filename(base_save_path: str, num_total_completed_layers: int) -> str:
    base, ext = os.path.splitext(base_save_path)
    return f"{base}_resume_L{num_total_completed_layers}{ext}"

def find_best_resume_file(intended_final_path: str) -> tuple[str | None, int]:
    output_dir = os.path.dirname(intended_final_path)
    if not output_dir: output_dir = "." 
    base_save_name, save_ext = os.path.splitext(os.path.basename(intended_final_path))

    potential_files = []
    if os.path.exists(intended_final_path):
        potential_files.append(intended_final_path)
    
    intermediate_pattern = os.path.join(output_dir, f"{base_save_name}_resume_L*{save_ext}")
    potential_files.extend(glob.glob(intermediate_pattern))

    best_file_path = None
    max_completed_modules = -1

    if not potential_files:
        return None, -1

    for file_path in sorted(potential_files): 
        try:
            if not os.path.exists(file_path): continue
            with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
            
            if metadata and "ss_completed_loha_modules" in metadata:
                num_completed = len(json.loads(metadata["ss_completed_loha_modules"]))
                if num_completed > max_completed_modules:
                    max_completed_modules = num_completed
                    best_file_path = file_path
                elif num_completed == max_completed_modules:
                    if file_path == intended_final_path and best_file_path != intended_final_path:
                         best_file_path = file_path
                    elif best_file_path and base_save_name+"_resume_L" in os.path.basename(best_file_path) and file_path == intended_final_path: 
                        best_file_path = file_path
            elif max_completed_modules == -1: 
                if best_file_path is None or (file_path == intended_final_path and best_file_path != intended_final_path):
                    best_file_path = file_path 
                    max_completed_modules = 0 
                    if args_global and args_global.verbose: print(f"    File {file_path} has no 'ss_completed_loha_modules' metadata. Treating as 0 completed for now (potential candidate).")
        except Exception as e:
            print(f"    Warning: Could not read or parse metadata from {file_path}: {e}")
            if best_file_path is None and file_path == intended_final_path and max_completed_modules == -1 :
                 best_file_path = file_path 
                 max_completed_modules = 0

    if best_file_path:
        print(f"  Selected '{os.path.basename(best_file_path)}' for resume (has {max_completed_modules} modules based on metadata/filename).")
    elif not potential_files: 
        pass 
    else:
        print(f"  Could not determine a best file to resume from among candidates, or no valid metadata found.")
    return best_file_path, max_completed_modules

def cleanup_intermediate_files(final_intended_path: str):
    output_dir = os.path.dirname(final_intended_path)
    if not output_dir: output_dir = "."
    base_save_name, save_ext = os.path.splitext(os.path.basename(final_intended_path))
    intermediate_pattern = os.path.join(output_dir, f"{base_save_name}_resume_L*{save_ext}")
    
    cleaned_count = 0
    for file_path in glob.glob(intermediate_pattern):
        try:
            os.remove(file_path)
            if args_global and args_global.verbose: print(f"  Cleaned up intermediate file: {file_path}")
            cleaned_count +=1
        except OSError as e:
            print(f"  Warning: Could not clean up intermediate file {file_path}: {e}")
    if cleaned_count > 0:
        print(f"  Cleaned up {cleaned_count} intermediate file(s).")

def perform_graceful_save(output_path_to_save: str):
    global extracted_loha_state_dict_global, layer_optimization_stats_global, args_global
    global processed_layers_this_session_count_global, save_attempted_on_interrupt
    global skipped_identical_count_global, skipped_other_reason_count_global, keys_scanned_this_run_global
    global all_completed_module_prefixes_ever_global 
    
    total_processed_ever = len(all_completed_module_prefixes_ever_global)

    if not extracted_loha_state_dict_global and not previously_completed_module_prefixes_global : 
        if not all_completed_module_prefixes_ever_global:
            print(f"No layers were processed or loaded to save to {output_path_to_save}. Save aborted.")
            return

    args_to_use = args_global
    if not args_to_use: print("Error: Global args not available for saving metadata."); return

    final_save_path = output_path_to_save 

    if args_to_use.save_weights_dtype == "fp16": final_save_dtype_torch = torch.float16
    elif args_to_use.save_weights_dtype == "bf16": final_save_dtype_torch = torch.bfloat16
    else: final_save_dtype_torch = torch.float32
    
    final_state_dict_to_save = OrderedDict()
    for k, v_tensor in extracted_loha_state_dict_global.items():
        if hasattr(v_tensor, 'is_floating_point') and v_tensor.is_floating_point(): final_state_dict_to_save[k] = v_tensor.to(final_save_dtype_torch)
        else: final_state_dict_to_save[k] = v_tensor

    print(f"\nAttempting to save LoHA for {total_processed_ever} unique modules in total "
          f"({processed_layers_this_session_count_global} new this session) to {final_save_path}")
    
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
        "ss_network_module": "lycoris.kohya", "ss_network_rank": global_rank_str, # This still reflects initial config rank
        "ss_network_alpha": eff_global_network_alpha_str, # This still reflects initial config alpha
        "ss_network_algo": "loha",
        "ss_network_args": json.dumps(network_args_dict),
        "ss_comment": f"Extracted LoHA (Interrupt: {save_attempted_on_interrupt}). OptPrec: {args_to_use.precision}. SaveDtype: {args_to_use.save_weights_dtype}. ATOL: {args_to_use.atol_fp32_check}. Layers: {total_processed_ever}. MaxIter: {args_to_use.max_iterations}. TargetLoss: {args_to_use.target_loss}. MaxRankRetries: {args_to_use.max_rank_retries}. RankIncrFactor: {args_to_use.rank_increase_factor}",
        "ss_base_model_name": os.path.splitext(os.path.basename(args_to_use.base_model_path))[0],
        "ss_ft_model_name": os.path.splitext(os.path.basename(args_to_use.ft_model_path))[0],
        "ss_save_weights_dtype": args_to_use.save_weights_dtype, "ss_optimization_precision": args_to_use.precision,
        "ss_completed_loha_modules": json.dumps(list(all_completed_module_prefixes_ever_global)) 
    }
    
    json_metadata_for_file = {
        "comfyui_lora_type": "LyCORIS_LoHa", "model_name": os.path.splitext(os.path.basename(final_save_path))[0],
        "base_model_path": args_to_use.base_model_path, "ft_model_path": args_to_use.ft_model_path,
        "loha_extraction_settings": {k: str(v) if isinstance(v, type(os.pathsep)) else v for k,v in vars(args_to_use).items()},
        "extraction_summary":{
            "processed_layers_in_total_cumulative": total_processed_ever, 
            "processed_this_session": processed_layers_this_session_count_global, 
            "skipped_identical_count_this_session": skipped_identical_count_global, 
            "skipped_other_reason_count_this_session": skipped_other_reason_count_global, 
            "total_candidate_keys_scanned_in_loop_this_session": keys_scanned_this_run_global,
        },
        "layer_optimization_details_this_session": layer_optimization_stats_global, # This will contain final_rank_used for each layer
        "embedded_safetensors_metadata": sf_metadata, 
        "interrupted_save": save_attempted_on_interrupt
    }

    if final_save_path.endswith(".safetensors"):
        try: 
            save_file(final_state_dict_to_save, final_save_path, metadata=sf_metadata)
            print(f"LoHA state_dict saved to: {final_save_path}")
        except Exception as e: 
            print(f"Error saving .safetensors file: {e}"); return
        
        metadata_json_file_path = os.path.splitext(final_save_path)[0] + "_extraction_metadata.json"
        try:
            with open(metadata_json_file_path, 'w') as f: json.dump(json_metadata_for_file, f, indent=4)
            print(f"Extended metadata saved to: {metadata_json_file_path}")
        except Exception as e: print(f"Could not save extended metadata JSON: {e}")
    else: 
        print(f"Saving to .pt not fully supported with extended metadata JSON. Saving basic .pt file.")
        torch.save({'state_dict': final_state_dict_to_save, 'metadata': sf_metadata}, final_save_path)
        print(f"LoHA state_dict saved to: {final_save_path} (basic .pt save)")

def handle_interrupt(signum, frame):
    global save_attempted_on_interrupt, outer_pbar_global, args_global, all_completed_module_prefixes_ever_global
    
    print("\n" + "="*30 + "\nCtrl+C (SIGINT) detected!\n" + "="*30)
    if save_attempted_on_interrupt: print("Save already attempted or in progress. Force exiting if stuck, or wait."); return 
    save_attempted_on_interrupt = True 
    
    if outer_pbar_global: outer_pbar_global.close() 
    
    print("Attempting to save progress for processed layers...")
    if args_global and args_global.save_to:
        num_layers_for_filename = len(all_completed_module_prefixes_ever_global)
        interrupt_save_path = generate_intermediate_filename(args_global.save_to, num_layers_for_filename)
        print(f"Interrupt save will be to: {interrupt_save_path}")
        perform_graceful_save(output_path_to_save=interrupt_save_path)
    else:
        print("Cannot perform interrupt save: args_global or save_to path not defined.")
        
    print("Graceful save attempt finished. Exiting.")
    sys.exit(0) 

def main(cli_args):
    global args_global, extracted_loha_state_dict_global, layer_optimization_stats_global
    global processed_layers_this_session_count_global, save_attempted_on_interrupt, outer_pbar_global
    global skipped_identical_count_global, skipped_other_reason_count_global, keys_scanned_this_run_global
    global previously_completed_module_prefixes_global, all_completed_module_prefixes_ever_global
    global main_loop_completed_scan_flag_global
    
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
    print(f"Max Iters/Layer: {args_global.max_iterations}, Max Rank Retries on Failure: {args_global.max_rank_retries}, Rank Increase Factor: {args_global.rank_increase_factor}")

    chosen_resume_file = None
    if not args_global.overwrite:
        print(f"\nChecking for existing LoHA file or resume states for: {args_global.save_to}")
        chosen_resume_file, num_modules_in_chosen_file = find_best_resume_file(args_global.save_to)
        
        if chosen_resume_file:
            print(f"  Attempting to resume from: {chosen_resume_file} ({num_modules_in_chosen_file} modules reported in metadata/filename).")
            try:
                file_metadata = None
                completed_modules_in_file = set()

                with safetensors.safe_open(chosen_resume_file, framework="pt", device="cpu") as f: 
                    file_metadata = f.metadata()
                    if file_metadata and "ss_completed_loha_modules" in file_metadata:
                        try:
                            completed_modules_in_file = set(json.loads(file_metadata.get("ss_completed_loha_modules")))
                            if len(completed_modules_in_file) != num_modules_in_chosen_file and num_modules_in_chosen_file >= 0 : 
                                 if args_global.verbose: print(f"  Info: Metadata module count ({len(completed_modules_in_file)}) differs from find_best_resume_file's heuristic count ({num_modules_in_chosen_file}). Using parsed set from metadata.")
                        except json.JSONDecodeError:
                            print(f"  Warning: Could not parse 'ss_completed_loha_modules' metadata from {os.path.basename(chosen_resume_file)}. Will load all tensors and infer completed from names if possible, or just load all.")
                            completed_modules_in_file = set() 
                    elif not file_metadata or "ss_completed_loha_modules" not in file_metadata:
                         if args_global.verbose: print(f"  'ss_completed_loha_modules' not found or no metadata in {os.path.basename(chosen_resume_file)}'s metadata. Will load all tensors from it.")
                         
                loaded_sd_for_resume = load_file(chosen_resume_file, device='cpu') 
                
                if not completed_modules_in_file and loaded_sd_for_resume: 
                    inferred_prefixes = set()
                    for key in loaded_sd_for_resume.keys():
                        if "hada_w1_a" in key: 
                             inferred_prefixes.add(".".join(key.split('.')[:-1]))
                    if inferred_prefixes:
                        if args_global.verbose: print(f"  Inferred {len(inferred_prefixes)} module prefixes from tensor keys in {os.path.basename(chosen_resume_file)} as 'ss_completed_loha_modules' was missing/empty.")
                        completed_modules_in_file = inferred_prefixes
                    elif not inferred_prefixes and num_modules_in_chosen_file > 0: 
                        if args_global.verbose: print(f"  Warning: Filename suggested {num_modules_in_chosen_file} modules, but could not infer LoHA prefixes from keys in {os.path.basename(chosen_resume_file)}. Treating as empty for specific resume.")

                resumed_tensor_count = 0
                if completed_modules_in_file: 
                    for key, tensor_val in loaded_sd_for_resume.items():
                        module_prefix_for_check = ".".join(key.split('.')[:-1]) 
                        is_bias_key = key.endswith(".bias") 
                        
                        if module_prefix_for_check in completed_modules_in_file or is_bias_key: 
                            extracted_loha_state_dict_global[key] = tensor_val
                            resumed_tensor_count +=1
                    
                    previously_completed_module_prefixes_global = completed_modules_in_file
                    all_completed_module_prefixes_ever_global.update(previously_completed_module_prefixes_global) 
                    print(f"  Successfully loaded {len(previously_completed_module_prefixes_global)} module prefixes "
                          f"with {resumed_tensor_count} tensors for resume from {os.path.basename(chosen_resume_file)}.")
                elif loaded_sd_for_resume: 
                    if args_global.verbose: print(f"  No specific completed module list from metadata, but file has content. Loading all {len(loaded_sd_for_resume)} tensors from {os.path.basename(chosen_resume_file)}.")
                    extracted_loha_state_dict_global.update(loaded_sd_for_resume)
                    inferred_prefixes = set()
                    for key in extracted_loha_state_dict_global.keys():
                        if "hada_w1_a" in key: inferred_prefixes.add(".".join(key.split('.')[:-1]))
                    if inferred_prefixes: previously_completed_module_prefixes_global = inferred_prefixes
                    all_completed_module_prefixes_ever_global.update(previously_completed_module_prefixes_global)
                    if args_global.verbose: print(f"  Inferred {len(previously_completed_module_prefixes_global)} module prefixes from loaded tensors.")
                del loaded_sd_for_resume
                
                resume_metadata_json_path = os.path.splitext(chosen_resume_file)[0] + "_extraction_metadata.json"
                if os.path.exists(resume_metadata_json_path):
                    try:
                        with open(resume_metadata_json_path, 'r') as f_meta: json.load(f_meta) 
                        if args_global.verbose: print(f"  Found accompanying metadata JSON: {os.path.basename(resume_metadata_json_path)}")
                    except Exception as e_json: print(f"  Could not load or parse JSON metadata from {resume_metadata_json_path}: {e_json}")

            except Exception as e:
                print(f"  Error loading or parsing chosen LoHA file '{chosen_resume_file}': {e}. Starting fresh for new layers.")
                extracted_loha_state_dict_global.clear()
                previously_completed_module_prefixes_global.clear()
                all_completed_module_prefixes_ever_global.clear()
        else:
            print("  No suitable existing LoHA file found to resume from. Starting fresh.")

    elif args_global.overwrite and os.path.exists(args_global.save_to):
        print(f"Overwriting specified output file as per --overwrite: {args_global.save_to}")
        extracted_loha_state_dict_global.clear()
        previously_completed_module_prefixes_global.clear()
        all_completed_module_prefixes_ever_global.clear()

    print(f"\nLoading base model: {args_global.base_model_path}")
    if args_global.base_model_path.endswith(".safetensors"): base_model_sd = load_file(args_global.base_model_path, device='cpu')
    else: base_model_sd = torch.load(args_global.base_model_path, map_location='cpu'); base_model_sd = base_model_sd.get('state_dict', base_model_sd)
    
    print(f"Loading fine-tuned model: {args_global.ft_model_path}")
    if args_global.ft_model_path.endswith(".safetensors"): ft_model_sd = load_file(args_global.ft_model_path, device='cpu')
    else: ft_model_sd = torch.load(args_global.ft_model_path, map_location='cpu'); ft_model_sd = ft_model_sd.get('state_dict', ft_model_sd)

    processed_layers_this_session_count_global = 0
    skipped_identical_count_global = 0 
    skipped_other_reason_count_global = 0 
    keys_scanned_this_run_global = 0
    layer_optimization_stats_global.clear() 
    main_loop_completed_scan_flag_global = False

    all_candidate_keys = []
    for k in base_model_sd.keys():
        if k.endswith('.weight') and k in ft_model_sd and (len(base_model_sd[k].shape) == 2 or len(base_model_sd[k].shape) == 4):
            all_candidate_keys.append(k)
    all_candidate_keys.sort()
    total_candidates_to_scan = len(all_candidate_keys)
    
    print(f"Found {total_candidates_to_scan} candidate '.weight' keys common to both models and of suitable shape.")
    
    outer_pbar_global = tqdm(total=total_candidates_to_scan, desc="Scanning Layers", dynamic_ncols=True, position=0)
    
    try:
        for key_name in all_candidate_keys:
            if save_attempted_on_interrupt: break
            keys_scanned_this_run_global += 1
            outer_pbar_global.update(1)

            original_module_path = key_name[:-len(".weight")]
            loha_key_prefix = ""
            if original_module_path.startswith("model.diffusion_model."): loha_key_prefix = "lora_unet_" + original_module_path[len("model.diffusion_model."):].replace(".", "_")
            elif original_module_path.startswith("first_stage_model."): loha_key_prefix = "lora_vae_" + original_module_path[len("first_stage_model."):].replace(".", "_") 
            elif "conditioner.embedders.0.transformer." in original_module_path : loha_key_prefix = "lora_te1_" + original_module_path[original_module_path.find("conditioner.embedders.0.transformer.")+len("conditioner.embedders.0.transformer."):].replace(".", "_")
            elif "conditioner.embedders.1.model.transformer." in original_module_path : loha_key_prefix = "lora_te2_" + original_module_path[original_module_path.find("conditioner.embedders.1.model.transformer.")+len("conditioner.embedders.1.model.transformer."):].replace(".", "_")
            elif original_module_path.startswith("text_encoder.") : loha_key_prefix = "lora_te_" + original_module_path[len("text_encoder."):].replace(".","_")
            elif original_module_path.startswith("text_encoder_2.") : loha_key_prefix = "lora_te2_" + original_module_path[len("text_encoder_2."):].replace(".","_")
            elif original_module_path.startswith("unet.") : loha_key_prefix = "lora_unet_" + original_module_path[len("unet."):].replace(".","_") 
            else: loha_key_prefix = "lora_" + original_module_path.replace(".", "_")


            if loha_key_prefix in all_completed_module_prefixes_ever_global:
                outer_pbar_global.set_description_str(f"Scan {keys_scanned_this_run_global}/{total_candidates_to_scan} (Resumed: {len(previously_completed_module_prefixes_global)}, New Opt: {processed_layers_this_session_count_global})")
                continue
            
            if args_global.max_layers is not None and args_global.max_layers > 0 and processed_layers_this_session_count_global >= args_global.max_layers:
                if args_global.verbose and processed_layers_this_session_count_global == args_global.max_layers and (keys_scanned_this_run_global - (len(all_completed_module_prefixes_ever_global) - processed_layers_this_session_count_global) - skipped_identical_count_global - skipped_other_reason_count_global) == (args_global.max_layers +1) :
                     tqdm.write(f"\nReached max_layers limit ({args_global.max_layers}) for new layers this session. Continuing scan only.")
                outer_pbar_global.set_description_str(f"Scan {keys_scanned_this_run_global}/{total_candidates_to_scan} (Max New Layers Reached)")
                continue 

            base_W = base_model_sd[key_name].to(dtype=torch.float32)
            ft_W = ft_model_sd[key_name].to(dtype=torch.float32)
            if base_W.shape != ft_W.shape: 
                skipped_other_reason_count_global +=1
                if args_global.verbose: tqdm.write(f"Skipping {key_name} (shape mismatch).")
                continue
            shape_info = get_module_shape_info_from_weight(base_W)
            if shape_info is None: 
                skipped_other_reason_count_global +=1
                if args_global.verbose: tqdm.write(f"Skipping {key_name} (unsupported shape).")
                continue
            
            delta_W_fp32 = (ft_W - base_W)
            if torch.allclose(delta_W_fp32, torch.zeros_like(delta_W_fp32), atol=args_global.atol_fp32_check):
                if args_global.verbose: tqdm.write(f"Skipping {key_name} (identical weights).")
                skipped_identical_count_global += 1
                continue
            
            max_layers_target_str = f"/{args_global.max_layers}" if args_global.max_layers is not None and args_global.max_layers > 0 else ""
            outer_pbar_global.set_description_str(f"Optimizing L{processed_layers_this_session_count_global + 1}{max_layers_target_str} (Scan {keys_scanned_this_run_global}/{total_candidates_to_scan})")
            if args_global.verbose: tqdm.write(f"\n  Orig: {key_name} -> LoHA: {loha_key_prefix}")
            
            out_dim, in_dim_effective, k_h, k_w, _, is_conv = shape_info
            delta_W_target_for_opt = delta_W_fp32 # Passed as is, .to(device, dtype) happens inside optimize_loha_for_layer
            
            initial_rank_for_layer_opt = args_global.conv_rank if is_conv and args_global.conv_rank is not None else args_global.rank
            initial_alpha_for_layer_opt = args_global.initial_conv_alpha if is_conv else args_global.initial_alpha
            
            tqdm.write(f"Optimizing Layer {processed_layers_this_session_count_global + 1}{max_layers_target_str}: {loha_key_prefix} (Orig: {original_module_path}, Shp: {list(base_W.shape)}, Initial R: {initial_rank_for_layer_opt}, Initial Alpha: {initial_alpha_for_layer_opt:.1f})")
            try:
                opt_results = optimize_loha_for_layer(
                    layer_name=loha_key_prefix, delta_W_target=delta_W_target_for_opt,
                    out_dim=out_dim, in_dim_effective=in_dim_effective, k_h=k_h, k_w=k_w, 
                    initial_rank_for_layer=initial_rank_for_layer_opt, 
                    initial_alpha_for_layer=initial_alpha_for_layer_opt, 
                    lr=args_global.lr,
                    max_iterations=args_global.max_iterations, min_iterations=args_global.min_iterations,
                    target_loss=args_global.target_loss, weight_decay=args_global.weight_decay,
                    device=args_global.device, dtype=target_opt_dtype, is_conv=is_conv, 
                    verbose_layer_debug=args_global.verbose_layer_debug,
                    max_rank_retries=args_global.max_rank_retries,
                    rank_increase_factor=args_global.rank_increase_factor
                )

                if not opt_results.get('interrupted_mid_layer') and 'hada_w1_a' in opt_results : 
                    for p_name, p_val in opt_results.items():
                        if p_name not in ['final_loss', 'stopped_early', 'iterations_done', 'interrupted_mid_layer', 'final_rank_used']:
                            if torch.is_tensor(p_val): 
                                extracted_loha_state_dict_global[f'{loha_key_prefix}.{p_name}'] = p_val.to(final_save_dtype)
                    
                    final_rank_used_for_layer = opt_results['final_rank_used']
                    rank_was_increased = final_rank_used_for_layer > initial_rank_for_layer_opt

                    layer_optimization_stats_global.append({
                        "name": loha_key_prefix, "original_name": original_module_path,
                        "initial_rank_attempted": initial_rank_for_layer_opt,
                        "final_rank_used": final_rank_used_for_layer,
                        "rank_was_increased": rank_was_increased, # Renamed from rank_was_doubled
                        "final_loss": opt_results['final_loss'], "iterations_done": opt_results['iterations_done'],
                        "stopped_early_by_loss_target": opt_results['stopped_early']
                    })
                    
                    all_completed_module_prefixes_ever_global.add(loha_key_prefix) 
                    
                    rank_info_str = f"R_used: {final_rank_used_for_layer}"
                    if rank_was_increased: rank_info_str += f" (Initial: {initial_rank_for_layer_opt})"

                    tqdm.write(f"  Layer {loha_key_prefix} Done. {rank_info_str}, Loss: {opt_results['final_loss']:.4e}, Iters: {opt_results['iterations_done']}{', Stopped by Loss' if opt_results['stopped_early'] else ''}")
                    
                    if args_global.use_bias:
                        original_bias_key = f"{original_module_path}.bias"
                        bias_differs = False
                        if original_bias_key in ft_model_sd:
                            ft_B = ft_model_sd[original_bias_key].to(dtype=torch.float32)
                            if original_bias_key in base_model_sd:
                                base_B = base_model_sd[original_bias_key].to(dtype=torch.float32)
                                if not torch.allclose(base_B, ft_B, atol=args_global.atol_fp32_check): bias_differs = True
                            else: bias_differs = True
                            
                            if bias_differs:
                                extracted_loha_state_dict_global[original_bias_key] = ft_B.cpu().to(final_save_dtype)
                                if args_global.verbose: tqdm.write(f"    Saved differing/new bias for {original_bias_key}")
                    processed_layers_this_session_count_global += 1
                else: 
                     tqdm.write(f"  Opt for {loha_key_prefix} interrupted or yielded no valid tensors; not saving params for this layer. Interrupt flag: {opt_results.get('interrupted_mid_layer', 'N/A')}, Loss: {opt_results.get('final_loss', 'N/A')}")
                     if not opt_results.get('interrupted_mid_layer', False) and 'hada_w1_a' not in opt_results : 
                        skipped_other_reason_count_global +=1
            except Exception as e:
                print(f"\nError during optimization for {original_module_path} ({loha_key_prefix}): {e}")
                import traceback; traceback.print_exc()
                skipped_other_reason_count_global +=1 
        
        if not save_attempted_on_interrupt and keys_scanned_this_run_global == total_candidates_to_scan:
            main_loop_completed_scan_flag_global = True

    finally: 
        if outer_pbar_global:
            if not outer_pbar_global.disable and hasattr(outer_pbar_global, 'n') and hasattr(outer_pbar_global, 'total') and outer_pbar_global.n < outer_pbar_global.total:
                 outer_pbar_global.update(outer_pbar_global.total - outer_pbar_global.n) 
            outer_pbar_global.close()

    if not save_attempted_on_interrupt: 
        print("\n--- Final Optimization Summary (This Session) ---")
        for stat in layer_optimization_stats_global: 
            rank_info = f"InitialR: {stat['initial_rank_attempted']}, FinalR: {stat['final_rank_used']}"
            if stat['rank_was_increased']: rank_info += " (Increased)" # Changed from Doubled
            print(f"Layer: {stat['name']}, {rank_info}, Loss: {stat['final_loss']:.4e}, Iters: {stat['iterations_done']}{', Stopped by Loss' if stat['stopped_early_by_loss_target'] else ''}")
        
        print(f"\n--- Overall Summary ---")
        print(f"Total unique LoHA modules accumulated (resumed + new): {len(all_completed_module_prefixes_ever_global)}")
        print(f"  Processed new this session: {processed_layers_this_session_count_global}")
        print(f"  Skipped as identical (this session's scan): {skipped_identical_count_global}")
        print(f"  Skipped for other reasons (this session's scan, e.g., shape error, opt error/no tensors): {skipped_other_reason_count_global}")
        print(f"  Total candidate keys scanned in loop (this session): {keys_scanned_this_run_global}/{total_candidates_to_scan}")
        
        actual_save_path: str
        save_to_final_name = False

        if main_loop_completed_scan_flag_global: 
            num_potential_optimizable_layers_in_models = total_candidates_to_scan - skipped_identical_count_global - skipped_other_reason_count_global 
            
            if len(all_completed_module_prefixes_ever_global) >= num_potential_optimizable_layers_in_models:
                if args_global.max_layers is None or args_global.max_layers == 0: 
                    save_to_final_name = True
                else: 
                    # If max_layers is set, it's final only if all potential layers are covered
                    # AND (we processed up to max_layers OR fewer than max_layers were needed to cover all)
                    if (len(previously_completed_module_prefixes_global) + processed_layers_this_session_count_global) >= num_potential_optimizable_layers_in_models:
                        save_to_final_name = True
                    # else: max_layers might have been hit, but not all optimizable layers are in the cumulative set yet.
                    # This case is handled by the outer `len(all_completed...) >= num_potential...` being false.
            else: 
                print(f"  Scan completed, but not all {num_potential_optimizable_layers_in_models} differing/optimizable layers are processed yet "
                      f"(current total processed: {len(all_completed_module_prefixes_ever_global)}).")
        else: 
            print("  Scan did not complete fully. Saving intermediate state.")


        if save_to_final_name:
            actual_save_path = args_global.save_to
            print(f"\nAll differing/optimizable layers from this scan appear to be processed into the LoHA. Saving to final path: {actual_save_path}")
        else:
            num_layers_for_filename = len(all_completed_module_prefixes_ever_global)
            actual_save_path = generate_intermediate_filename(args_global.save_to, num_layers_for_filename)
            if main_loop_completed_scan_flag_global and not save_to_final_name: 
                 print(f"\nFull scan completed, but not all layers processed yet. Saving intermediate state to: {actual_save_path}")
            else: 
                 print(f"\nRun incomplete or not all differing layers processed. Saving intermediate state to: {actual_save_path}")

        perform_graceful_save(output_path_to_save=actual_save_path)

        if save_to_final_name and actual_save_path == args_global.save_to : 
            print("\nCleaning up intermediate resume files...")
            cleanup_intermediate_files(args_global.save_to)
            
    else: 
        print("\nProcess was interrupted. Graceful save to an intermediate file was attempted by signal handler.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LoHA parameters by optimizing against weight differences. Saves intermediate files like 'name_resume_L{count}.safetensors'.")
    parser.add_argument("base_model_path", type=str, help="Path to the base model state_dict file (.pt, .pth, .safetensors)")
    parser.add_argument("ft_model_path", type=str, help="Path to the fine-tuned model state_dict file (.pt, .pth, .safetensors)")
    parser.add_argument("save_to", type=str, help="Path to save the FINAL extracted LoHA file (recommended .safetensors). Intermediate files will be based on this name.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore and overwrite any existing FINAL LoHA output file. Does NOT clean intermediates until a successful final save of the current run.")
    parser.add_argument("--rank", type=int, default=4, help="Default rank for LoHA decomposition (used for linear layers and as fallback for conv).")
    parser.add_argument("--conv_rank", type=int, default=None, help="Specific rank for convolutional LoHA layers. Defaults to --rank if not set.")
    parser.add_argument("--initial_alpha", type=float, default=None, help="Global initial alpha for optimization. Defaults to 'rank'.")
    parser.add_argument("--initial_conv_alpha", type=float, default=None, help="Specific initial alpha for Conv LoHA. Defaults to '--initial_alpha' or conv_rank.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for LoHA optimization per layer.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of optimization iterations per layer PER ATTEMPT (rank increase restarts iterations).")
    parser.add_argument("--min_iterations", type=int, default=100, help="Minimum iterations before checking target loss PER ATTEMPT.")
    parser.add_argument("--target_loss", type=float, default=None, help="Target MSE loss for early stopping per layer. If met, rank increase is not attempted.")
    parser.add_argument("--max_rank_retries", type=int, default=0, help="Number of times to increase rank and retry if target_loss is not met (0 for no retries, 1 for one retry, etc.). Default: 0.")
    parser.add_argument("--rank_increase_factor", type=float, default=1.25, help="Factor by which to increase rank on each retry (e.g., 1.25 for 25%% increase, 2.0 for doubling). Rank is ceiling rounded. Default: 1.25.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for LoHA optimization.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device ('cuda' or 'cpu').")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Optimization precision. Default: fp32.")
    parser.add_argument("--save_weights_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Dtype for saved LoHA weights. Default: bf16.")
    parser.add_argument("--atol_fp32_check", type=float, default=1e-6, help="Tolerance for identical weight check.")
    parser.add_argument("--use_bias", action="store_true", help="Save differing bias terms.")
    parser.add_argument("--dropout", type=float, default=0.0, help="General dropout (metadata only).")
    parser.add_argument("--rank_dropout", type=float, default=0.0, help="Rank dropout (metadata only).")
    parser.add_argument("--module_dropout", type=float, default=0.0, help="Module dropout (metadata only).")
    parser.add_argument("--max_layers", type=int, default=None, help="Max NEW differing layers to process this session. Scan will continue to assess all layers.")
    parser.add_argument("--verbose", action="store_true", help="General verbose output.")
    parser.add_argument("--verbose_layer_debug", action="store_true", help="Detailed per-iteration optimization debug output.")
    
    parsed_args = parser.parse_args()
    if not os.path.exists(parsed_args.base_model_path): print(f"Error: Base model path not found: {parsed_args.base_model_path}"); sys.exit(1)
    if not os.path.exists(parsed_args.ft_model_path): print(f"Error: Fine-tuned model path not found: {parsed_args.ft_model_path}"); sys.exit(1)
    
    save_dir = os.path.dirname(parsed_args.save_to)
    if save_dir and not os.path.exists(save_dir): 
        try:
            os.makedirs(save_dir, exist_ok=True)
            if parsed_args.verbose: print(f"Created directory: {save_dir}")
        except OSError as e:
            print(f"Error: Could not create directory {save_dir}: {e}"); sys.exit(1)

    if parsed_args.initial_alpha is None: parsed_args.initial_alpha = float(parsed_args.rank)
    if parsed_args.initial_conv_alpha is None: 
        conv_rank_for_alpha_default = parsed_args.conv_rank if parsed_args.conv_rank is not None else parsed_args.rank
        parsed_args.initial_conv_alpha = float(conv_rank_for_alpha_default) if parsed_args.conv_rank is not None else parsed_args.initial_alpha

    main(parsed_args)