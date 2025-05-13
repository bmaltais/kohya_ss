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
import traceback # For detailed error printing

# --- Global variables --- (Keep these as they are defined in your original script)
extracted_loha_state_dict_global = OrderedDict()
layer_optimization_stats_global = []
args_global = None # This will be set by main()
processed_layers_this_session_count_global = 0
previously_completed_module_prefixes_global = set()
all_completed_module_prefixes_ever_global = set()
skipped_identical_count_global = 0
skipped_other_reason_count_global = 0
keys_scanned_this_run_global = 0
save_attempted_on_interrupt = False
outer_pbar_global = None
main_loop_completed_scan_flag_global = False


def _get_closest_ema_value_before_iter(target_iter: int, ema_history: list[tuple[int, float]]) -> tuple[int | None, float | None]:
    """Helper to find the iteration and EMA loss value at or just before target_iter."""
    if not ema_history:
        return None, None
    best_match_iter = None
    best_match_loss = None
    for hist_iter, hist_loss in reversed(ema_history):
        if hist_iter <= target_iter:
            if best_match_iter is None or hist_iter > best_match_iter :
                best_match_iter = hist_iter
                best_match_loss = hist_loss
            # Optimization: break early if we've found a match and gone far enough back
            if best_match_iter is not None and target_iter - hist_iter > getattr(args_global, 'projection_sample_interval', 20) * 2 : # Heuristic break
                 break
    # If no value found before or at target_iter (e.g., target_iter is before first sample), return the earliest sample
    if best_match_iter is not None:
        return best_match_iter, best_match_loss
    return ema_history[0] if ema_history else (None, None)


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
    effective_verbose_layer_debug = verbose_layer_debug or (args_global and args_global.verbose_layer_debug)

    best_result_so_far = {
        'final_loss': float('inf'),
        'stopped_early_by_loss': False,
        'stopped_by_insufficient_progress': False,
        'stopped_by_projection': False,
        'projection_type_used': 'none',
        'iterations_done': 0,
        'final_rank_used': initial_rank_for_layer,
        'interrupted_mid_layer': False,
        'final_projected_loss_on_stop': None
    }

    current_rank_for_this_attempt = initial_rank_for_layer
    alpha_init_for_this_attempt = initial_alpha_for_layer
    rank_base_for_next_increase = initial_rank_for_layer

    prog_check_interval_val = args_global.progress_check_interval
    min_prog_ratio_val = args_global.min_progress_loss_ratio
    iter_to_begin_first_progress_window = args_global.progress_check_start_iter
    adv_proj_decay_cap_min = getattr(args_global, 'advanced_projection_decay_cap_min', 0.5)
    adv_proj_decay_cap_max = getattr(args_global, 'advanced_projection_decay_cap_max', 1.05)

    proj_sample_interval = getattr(args_global, 'projection_sample_interval', 20)
    proj_ema_alpha = getattr(args_global, 'projection_ema_alpha', 0.1)
    proj_min_ema_hist = getattr(args_global, 'projection_min_ema_history', 5)

    for attempt in range(max_rank_retries + 1):
        is_last_rank_attempt = (attempt == max_rank_retries)

        if save_attempted_on_interrupt:
            # ... (interrupt handling)
             if attempt == 0: return {**best_result_so_far, 'interrupted_mid_layer': True, 'iterations_done': 0, 'projection_type_used': 'interrupted', 'final_projected_loss_on_stop': None}
             else: best_result_so_far.update({'interrupted_mid_layer': True, 'projection_type_used': 'interrupted', 'final_projected_loss_on_stop': None}); return best_result_so_far

        can_warm_start = False
        if attempt > 0:
            # ... (rank increase logic as before) ...
            tqdm.write(f"  Retrying {layer_name}: rank {rank_base_for_next_increase} (best loss so far: {best_result_so_far.get('final_loss', float('inf')):.2e}) ... Increasing rank.")
            new_rank_float = rank_base_for_next_increase * rank_increase_factor
            increased_rank = math.ceil(new_rank_float)
            current_rank_for_this_attempt = max(rank_base_for_next_increase + 1, increased_rank)
            original_alpha_to_rank_ratio = 1.0
            if initial_rank_for_layer > 0: original_alpha_to_rank_ratio = initial_alpha_for_layer / float(initial_rank_for_layer)
            alpha_init_for_this_attempt = original_alpha_to_rank_ratio * float(current_rank_for_this_attempt)
            tqdm.write(f"    Rank for {layer_name} increased to {current_rank_for_this_attempt}. Alpha adjusted to {alpha_init_for_this_attempt:.2f}.")

            if 'hada_w1_a' in best_result_so_far and not args_global.no_warm_start:
                previous_best_rank_for_warm_start = best_result_so_far['final_rank_used']
                if previous_best_rank_for_warm_start < current_rank_for_this_attempt:
                    can_warm_start = True; tqdm.write(f"    Warm-starting rank {current_rank_for_this_attempt} from {previous_best_rank_for_warm_start}.")
                else: tqdm.write(f"    Cannot warm-start: prev rank {previous_best_rank_for_warm_start} >= current {current_rank_for_this_attempt}.")
            elif args_global.no_warm_start and 'hada_w1_a' in best_result_so_far: tqdm.write(f"    Warm-start skipped (--no_warm_start).")


        # ... (parameter initialization, optimizer setup as before) ...
        k_ops = k_h * k_w if is_conv else 1
        hada_w1_a_p = nn.Parameter(torch.empty(out_dim, current_rank_for_this_attempt, device=device, dtype=dtype))
        hada_w1_b_p = nn.Parameter(torch.empty(current_rank_for_this_attempt, in_dim_effective * k_ops, device=device, dtype=dtype))
        hada_w2_a_p = nn.Parameter(torch.empty(out_dim, current_rank_for_this_attempt, device=device, dtype=dtype))
        hada_w2_b_p = nn.Parameter(torch.empty(current_rank_for_this_attempt, in_dim_effective * k_ops, device=device, dtype=dtype))

        with torch.no_grad(): # Initialization logic (warm or standard)
            if can_warm_start:
                prev_w1_a = best_result_so_far['hada_w1_a'].to(device, dtype); prev_w1_b = best_result_so_far['hada_w1_b'].to(device, dtype)
                prev_w2_a = best_result_so_far['hada_w2_a'].to(device, dtype); prev_w2_b = best_result_so_far['hada_w2_b'].to(device, dtype)
                hada_w1_a_p.data[:, :previous_best_rank_for_warm_start] = prev_w1_a; hada_w1_b_p.data[:previous_best_rank_for_warm_start, :] = prev_w1_b
                hada_w2_a_p.data[:, :previous_best_rank_for_warm_start] = prev_w2_a; hada_w2_b_p.data[:previous_best_rank_for_warm_start, :] = prev_w2_b
                nn.init.kaiming_uniform_(hada_w1_a_p.data[:, previous_best_rank_for_warm_start:], a=math.sqrt(5)); nn.init.normal_(hada_w1_b_p.data[previous_best_rank_for_warm_start:, :], std=0.02)
                nn.init.kaiming_uniform_(hada_w2_a_p.data[:, previous_best_rank_for_warm_start:], a=math.sqrt(5)); nn.init.normal_(hada_w2_b_p.data[previous_best_rank_for_warm_start:, :], std=0.02)
            else:
                nn.init.kaiming_uniform_(hada_w1_a_p.data, a=math.sqrt(5)); nn.init.normal_(hada_w1_b_p.data, std=0.02)
                nn.init.kaiming_uniform_(hada_w2_a_p.data, a=math.sqrt(5)); nn.init.normal_(hada_w2_b_p.data, std=0.02)

        alpha_param = nn.Parameter(torch.tensor(alpha_init_for_this_attempt, device=device, dtype=dtype))
        params_to_optimize = [hada_w1_a_p, hada_w1_b_p, hada_w2_a_p, hada_w2_b_p, alpha_param]
        optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
        patience_epochs = max(10, int(max_iterations * 0.05)); scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_epochs, factor=0.5, min_lr=1e-7, verbose=False)


        iter_pbar_desc_prefix = f"Opt Att {attempt+1}/{max_rank_retries+1} (R:{current_rank_for_this_attempt})"
        if is_last_rank_attempt: iter_pbar_desc_prefix += " [LastRank]"
        iter_pbar = tqdm(range(max_iterations), desc=f"{iter_pbar_desc_prefix}: {layer_name}", leave=False, dynamic_ncols=True, position=1, mininterval=0.5)

        current_attempt_final_loss = float('inf'); current_attempt_stopped_early_by_loss = False; current_attempt_insufficient_progress = False
        current_attempt_stopped_by_projection = False; current_attempt_projection_type = "none"; current_attempt_iterations_done = 0
        loss_at_start_of_current_window = float('inf'); progress_window_started_for_attempt = False
        relative_improvement_history = []
        final_projected_loss_if_failed = None

        loss_samples_for_ema = []
        ema_loss_history = []
        current_ema_loss_value = None
        latest_reliable_R_ema_for_pbar = None # Store the Rate from the last full check
        latest_reliable_decay_factor_ema = 1.0 # Store the Decay Factor from the last full check (default 1.0 = no decay)

        for i in iter_pbar:
            current_attempt_iterations_done = i + 1
            if save_attempted_on_interrupt:
                # ... (interrupt handling)
                iter_pbar.close(); tqdm.write(f"\n  Interrupt during opt of {layer_name}...")
                best_result_so_far.update({'interrupted_mid_layer': True, 'projection_type_used': 'interrupted', 'final_projected_loss_on_stop': None}); return best_result_so_far

            if prog_check_interval_val > 0 and not progress_window_started_for_attempt and current_attempt_iterations_done >= iter_to_begin_first_progress_window:
                 loss_at_start_of_current_window = current_attempt_final_loss
                 progress_window_started_for_attempt = True

            optimizer.zero_grad()
            # ... (loss calculation as before) ...
            eff_alpha_scale = alpha_param / current_rank_for_this_attempt
            if is_conv:
                term1_flat = hada_w1_a_p @ hada_w1_b_p; term1_reshaped = term1_flat.view(out_dim, in_dim_effective, k_h, k_w)
                term2_flat = hada_w2_a_p @ hada_w2_b_p; term2_reshaped = term2_flat.view(out_dim, in_dim_effective, k_h, k_w)
                delta_W_loha = eff_alpha_scale * term1_reshaped * term2_reshaped
            else:
                term1 = hada_w1_a_p @ hada_w1_b_p; term2 = hada_w2_a_p @ hada_w2_b_p
                delta_W_loha = eff_alpha_scale * term1 * term2
            loss = F.mse_loss(delta_W_loha, delta_W_target)

            raw_current_loss_item = loss.item()
            if i == 0 and progress_window_started_for_attempt: loss_at_start_of_current_window = raw_current_loss_item
            current_attempt_final_loss = raw_current_loss_item
            loss.backward(); optimizer.step(); scheduler.step(loss)

            pbar_projected_loss_str = ""
            if target_loss is not None and prog_check_interval_val > 0:
                if (i + 1) % proj_sample_interval == 0:
                    loss_samples_for_ema.append((i + 1, raw_current_loss_item))
                    if current_ema_loss_value is None:
                        current_ema_loss_value = raw_current_loss_item
                    else:
                        current_ema_loss_value = proj_ema_alpha * raw_current_loss_item + \
                                               (1 - proj_ema_alpha) * current_ema_loss_value
                    ema_loss_history.append((i + 1, current_ema_loss_value))

                    # --- Calculate Improved Live Pbar Projection (with decay factor) ---
                    if latest_reliable_R_ema_for_pbar is not None and current_ema_loss_value is not None and prog_check_interval_val > 0:
                        if latest_reliable_R_ema_for_pbar > 1e-9: # If there's a meaningful improvement rate
                            remaining_iterations_pbar = max_iterations - current_attempt_iterations_done
                            num_future_intervals_pbar = remaining_iterations_pbar // prog_check_interval_val

                            sim_loss_for_pbar = current_ema_loss_value
                            current_R_for_pbar_sim = latest_reliable_R_ema_for_pbar # Start with the last known rate

                            for _ in range(num_future_intervals_pbar):
                                sim_loss_for_pbar *= (1.0 - max(0, current_R_for_pbar_sim))
                                # Apply the decay factor to the rate for the *next* simulated interval
                                current_R_for_pbar_sim *= latest_reliable_decay_factor_ema # Use the stored decay factor
                                current_R_for_pbar_sim = max(1e-7, current_R_for_pbar_sim) # Prevent rate from going too low/negative

                                if target_loss is not None and sim_loss_for_pbar < target_loss:
                                    sim_loss_for_pbar = target_loss # Cap at target
                                    break

                            if target_loss is not None and current_ema_loss_value <= target_loss :
                                pbar_projected_loss_str = f", PBarProjL={current_ema_loss_value:.2e}(Met)"
                            else:
                                pbar_projected_loss_str = f", PBarProjL={sim_loss_for_pbar:.2e}"

                        elif current_ema_loss_value is not None : # Stalled rate, projection is current EMA
                             pbar_projected_loss_str = f", PBarProjL={current_ema_loss_value:.2e}(Stall)"
                    elif current_ema_loss_value is not None: # Before first R is known
                        pbar_projected_loss_str = f", PBarProjL={current_ema_loss_value:.2e}(Init)"


            current_lr = optimizer.param_groups[0]['lr']
            pbar_postfix = f"Loss={current_attempt_final_loss:.3e}{pbar_projected_loss_str}"
            pbar_postfix += f", AlphaP={alpha_param.item():.2f}, LR={current_lr:.1e}"
            iter_pbar.set_postfix_str(pbar_postfix, refresh=True)

            if target_loss is not None and current_attempt_iterations_done >= min_iterations and current_attempt_final_loss <= target_loss:
                # ... (target loss reached handling)
                if effective_verbose_layer_debug or (args_global and args_global.verbose): tqdm.write(f"  Target loss {target_loss:.2e} reached...")
                current_attempt_stopped_early_by_loss = True; break

            # --- Progress/Projection Check Block ---
            if prog_check_interval_val > 0 and progress_window_started_for_attempt and \
               (current_attempt_iterations_done >= iter_to_begin_first_progress_window + prog_check_interval_val) and \
               ((current_attempt_iterations_done - iter_to_begin_first_progress_window) % prog_check_interval_val == 0):

                if is_last_rank_attempt and (effective_verbose_layer_debug or args_global.verbose_layer_debug):
                    tqdm.write(f"    On last rank attempt (R:{current_rank_for_this_attempt}). Progress/Projection checks will be logged but not cause early stop.")
                perform_stop_checks = not is_last_rank_attempt

                projection_check_failed_this_iter = False; insufficient_progress_check_failed_this_iter = False
                current_attempt_projection_type = "none"; temp_final_projected_loss = None

                raw_relative_improvement_this_window = 0.0
                # ... (raw progress check as before, sets insufficient_progress_check_failed_this_iter if perform_stop_checks is true) ...
                if loss_at_start_of_current_window > 1e-12:
                    raw_relative_improvement_this_window = (loss_at_start_of_current_window - current_attempt_final_loss) / loss_at_start_of_current_window
                if (target_loss is None or current_attempt_final_loss > target_loss) and \
                   raw_relative_improvement_this_window < min_prog_ratio_val:
                    if effective_verbose_layer_debug or (args_global and args_global.verbose): tqdm.write(f"  Insufficient progress (Raw Loss) on {layer_name}. Raw Rel.Imprv: {raw_relative_improvement_this_window:.2e} < {min_prog_ratio_val:.1e}")
                    if perform_stop_checks:
                        insufficient_progress_check_failed_this_iter = True


                if target_loss is not None and current_attempt_final_loss > target_loss and not insufficient_progress_check_failed_this_iter:
                    use_ema_projection = len(ema_loss_history) >= proj_min_ema_hist
                    # ... (logging if EMA skipped)
                    if not use_ema_projection and (effective_verbose_layer_debug or (args_global and args_global.verbose)):
                        tqdm.write(f"    EMA projection skipped: Not enough EMA history ({len(ema_loss_history)} < {proj_min_ema_hist}).")

                    required_future_iterations = float('inf'); temp_final_projected_loss = None
                    iter_current_window_start_target = current_attempt_iterations_done - prog_check_interval_val

                    if use_ema_projection:
                        ema_iter_current, ema_loss_current = ema_loss_history[-1]
                        ema_iter_window_start, ema_loss_window_start = _get_closest_ema_value_before_iter(iter_current_window_start_target, ema_loss_history)

                        smoothed_current_relative_improvement = 0.0
                        if ema_loss_window_start is not None and ema_loss_window_start > 1e-12 and ema_iter_window_start is not None and ema_iter_current > ema_iter_window_start:
                            smoothed_current_relative_improvement = (ema_loss_window_start - ema_loss_current) / ema_loss_window_start

                        if smoothed_current_relative_improvement > 1e-9 :
                            relative_improvement_history.append(smoothed_current_relative_improvement)
                            relative_improvement_history = relative_improvement_history[-2:]
                            latest_reliable_R_ema_for_pbar = relative_improvement_history[-1] # STORE RATE FOR PBAR

                            if len(relative_improvement_history) >= 2:
                                R_curr_for_adv_ema = relative_improvement_history[-1]
                                R_prev_for_adv_ema = relative_improvement_history[-2]
                                if R_prev_for_adv_ema > 1e-9:
                                    current_attempt_projection_type = "advanced_ema"
                                    decay_factor_R_ema = R_curr_for_adv_ema / R_prev_for_adv_ema
                                    decay_factor_R_ema = max(adv_proj_decay_cap_min, min(adv_proj_decay_cap_max, decay_factor_R_ema))
                                    latest_reliable_decay_factor_ema = decay_factor_R_ema # STORE DECAY FACTOR FOR PBAR

                                    # ... (Full advanced simulation using decay_factor_R_ema) ...
                                    sim_loss = ema_loss_current
                                    sim_R = R_curr_for_adv_ema; sim_iters = 0; sim_req_iters = 0; avail_iters_sim = max_iterations - current_attempt_iterations_done
                                    max_win_sim = avail_iters_sim // prog_check_interval_val if prog_check_interval_val > 0 else 0; sim_time_out = False
                                    if sim_loss <= target_loss: sim_req_iters = 0
                                    else:
                                        for _w in range(max_win_sim + 1):
                                            sim_loss *= (1.0 - max(0, sim_R)); sim_iters += prog_check_interval_val
                                            if sim_loss <= target_loss: sim_req_iters = sim_iters; temp_final_projected_loss = None; break
                                            if sim_iters >= avail_iters_sim:
                                                if sim_loss > target_loss: sim_time_out = True; temp_final_projected_loss = sim_loss
                                                else: sim_req_iters = sim_iters; temp_final_projected_loss = None
                                                break
                                            sim_R = max(1e-7, sim_R * decay_factor_R_ema) # Apply decay here
                                        if sim_req_iters == 0 and sim_loss > target_loss: required_future_iterations = float('inf'); temp_final_projected_loss = sim_loss if temp_final_projected_loss is None else temp_final_projected_loss
                                        elif sim_time_out and sim_loss > target_loss: required_future_iterations = float('inf')
                                        else: required_future_iterations = sim_req_iters

                                else: # Fallback from advanced (prev R too small)
                                     current_attempt_projection_type = "simple_ema (fallback from adv_ema - prev EMA R too small)"
                                     latest_reliable_decay_factor_ema = 1.0 # Reset decay factor for PBar if advanced failed

                            if current_attempt_projection_type not in ["advanced_ema"]:
                                # ... (Simple EMA projection - doesn't calculate decay factor) ...
                                latest_reliable_decay_factor_ema = 1.0 # Reset decay factor for PBar if only simple used
                                if "fallback" not in current_attempt_projection_type : current_attempt_projection_type = "simple_ema"
                                if smoothed_current_relative_improvement > 1e-9:
                                    if ema_loss_current > target_loss:
                                        # ... (simple simulation)
                                        temp_sim_loss = ema_loss_current; temp_sim_iters = 0; avail_iters_sim_simple = max_iterations - current_attempt_iterations_done; projected_simple = False
                                        while temp_sim_loss > target_loss and temp_sim_iters <= avail_iters_sim_simple:
                                            temp_sim_loss *= (1.0 - max(0, smoothed_current_relative_improvement)); temp_sim_iters += prog_check_interval_val
                                            if temp_sim_loss <= target_loss: required_future_iterations = temp_sim_iters; projected_simple = True; temp_final_projected_loss = None; break
                                        if not projected_simple and temp_sim_loss > target_loss: required_future_iterations = float('inf'); temp_final_projected_loss = temp_sim_loss
                                else: current_attempt_projection_type = "stalled_ema"; required_future_iterations = float('inf')
                        elif use_ema_projection : # EMA available but shows no improvement
                            current_attempt_projection_type = "stalled_ema"; required_future_iterations = float('inf')
                            latest_reliable_R_ema_for_pbar = 0.0 # Store zero improvement for pbar
                            latest_reliable_decay_factor_ema = 1.0 # Reset decay factor

                    # --- Fallback Projection ---
                    if required_future_iterations == float('inf') and not current_attempt_projection_type.endswith("_ema"):
                        # ... (Raw fallback projection logic) ...
                        latest_reliable_decay_factor_ema = 1.0 # Reset decay factor if falling back
                        if effective_verbose_layer_debug and use_ema_projection: tqdm.write("    EMA projection inconclusive/stalled, trying original raw-loss projection.")
                        if raw_relative_improvement_this_window > 1e-9:
                            current_attempt_projection_type = "simple_raw_fallback"
                            # ... (simple raw simulation) ...
                            temp_sim_loss_raw = current_attempt_final_loss; temp_sim_iters_raw = 0; avail_iters_sim_simple_raw = max_iterations - current_attempt_iterations_done; projected_simple_raw = False
                            while temp_sim_loss_raw > target_loss and temp_sim_iters_raw <= avail_iters_sim_simple_raw:
                                temp_sim_loss_raw *= (1.0 - max(0, raw_relative_improvement_this_window)); temp_sim_iters_raw += prog_check_interval_val
                                if temp_sim_loss_raw <= target_loss: required_future_iterations = temp_sim_iters_raw; projected_simple_raw = True; temp_final_projected_loss = None; break
                            if not projected_simple_raw and temp_sim_loss_raw > target_loss: required_future_iterations = float('inf'); temp_final_projected_loss = temp_sim_loss_raw
                        else:
                            current_attempt_projection_type = "stalled_raw_fallback"; required_future_iterations = float('inf')

                    # --- Make Stop Decision (based on projection and is_last_rank_attempt) ---
                    available_iterations = max_iterations - current_attempt_iterations_done
                    if required_future_iterations > available_iterations:
                        if effective_verbose_layer_debug or (args_global and args_global.verbose):
                             # ... (logging projection failure as before) ...
                            tqdm.write(f"  Projection Stop ({current_attempt_projection_type}) on {layer_name}...")
                            log_msg_proj = f"    Est. {required_future_iterations:.0f} more iters needed (based on {current_attempt_projection_type})."
                            if temp_final_projected_loss is not None and required_future_iterations == float('inf'): log_msg_proj += f" Proj.FinalLoss: ~{temp_final_projected_loss:.2e}."
                            elif required_future_iterations == float('inf'): log_msg_proj += f" Target unreachable..."
                            tqdm.write(log_msg_proj); tqdm.write(f"    Available: {available_iterations}...")
                        if perform_stop_checks: # Only flag to stop if not last rank
                            projection_check_failed_this_iter = True
                        elif is_last_rank_attempt: # Log that we would have stopped
                             tqdm.write(f"    (LastRank): Would have stopped due to projection, but continuing.")
                        # Store the calculated projected loss regardless of stopping for logging purposes
                        final_projected_loss_if_failed = temp_final_projected_loss

                loss_at_start_of_current_window = current_attempt_final_loss # Update baseline for next raw check

                # Combine stop conditions
                if projection_check_failed_this_iter and perform_stop_checks: current_attempt_stopped_by_projection = True; break
                elif insufficient_progress_check_failed_this_iter and perform_stop_checks: current_attempt_insufficient_progress = True; break
        # --- End Inner Optimization Loop ---

        # ... (rest of the function: pbar final update, best_result_so_far update using CORRECTED variable name, retry logic, return statement as before) ...
        # Final pbar update
        iter_pbar_final_desc = f"{iter_pbar_desc_prefix}: {layer_name} (Done)"
        postfix_extras = []; projected_final_loss_str = ""
        if current_attempt_stopped_by_projection:
            postfix_extras.append(f'EarlyStop(Proj:{current_attempt_projection_type})')
            if final_projected_loss_if_failed is not None: projected_final_loss_str = f", Proj.FinalLoss: ~{final_projected_loss_if_failed:.2e}"
        elif current_attempt_stopped_early_by_loss: postfix_extras.append('EarlyStop(Loss)')
        elif current_attempt_insufficient_progress:
            postfix_extras.append('EarlyStop(Prog)')

        iter_pbar_final_postfix = f"FinalLoss={current_attempt_final_loss:.2e}{projected_final_loss_str}, It={current_attempt_iterations_done}{', ' + ', '.join(postfix_extras) if postfix_extras else ''}"
        if not save_attempted_on_interrupt: iter_pbar.set_description_str(iter_pbar_final_desc); iter_pbar.set_postfix_str(iter_pbar_final_postfix)
        iter_pbar.close()

        # Update best result
        if current_attempt_final_loss < best_result_so_far['final_loss'] or \
           (current_attempt_final_loss == best_result_so_far['final_loss'] and current_rank_for_this_attempt < best_result_so_far['final_rank_used']):
            update_dict = {
               'hada_w1_a': hada_w1_a_p.data.cpu().contiguous(), 'hada_w1_b': hada_w1_b_p.data.cpu().contiguous(),
               'hada_w2_a': hada_w2_a_p.data.cpu().contiguous(), 'hada_w2_b': hada_w2_b_p.data.cpu().contiguous(),
               'alpha': alpha_param.data.cpu().contiguous(), 'final_loss': current_attempt_final_loss,
               'stopped_early_by_loss': current_attempt_stopped_early_by_loss,
               'stopped_by_insufficient_progress': current_attempt_insufficient_progress, # CORRECTED NAME USED HERE
               'stopped_by_projection': current_attempt_stopped_by_projection,
               'projection_type_used': current_attempt_projection_type,
               'iterations_done': current_attempt_iterations_done, 'final_rank_used': current_rank_for_this_attempt,
               'interrupted_mid_layer': False
            }
            update_dict['final_projected_loss_on_stop'] = final_projected_loss_if_failed if (current_attempt_stopped_by_projection or final_projected_loss_if_failed is not None) else None
            best_result_so_far = update_dict


        rank_base_for_next_increase = current_rank_for_this_attempt
        # Decide whether to retry
        if current_attempt_stopped_early_by_loss: break # Always break if target loss met

        if not is_last_rank_attempt:
             # Check if finished early without explicit stop flags (only for non-last ranks)
            if current_attempt_iterations_done < max_iterations and not any([current_attempt_insufficient_progress, current_attempt_stopped_by_projection, current_attempt_stopped_early_by_loss]):
                if args_global and args_global.verbose: tqdm.write(f"  Layer {layer_name} attempt finished early ({current_attempt_iterations_done}/{max_iterations} iters) without explicit stop. Using this result.")
                break


        # Check if max retries reached
        if attempt >= max_rank_retries:
             if args_global and args_global.verbose:
                 # ... (max retries logging as before) ...
                  reason = ""
                  if best_result_so_far.get('stopped_by_projection', False) and not is_last_rank_attempt :
                       proj_loss_str = f" (Proj.FinalLoss: ~{best_result_so_far['final_projected_loss_on_stop']:.2e})" if best_result_so_far.get('final_projected_loss_on_stop') is not None else ""
                       reason = f"was stopped by projection ({best_result_so_far.get('projection_type_used', 'N/A')}){proj_loss_str}"
                  elif target_loss is not None and best_result_so_far['final_loss'] > target_loss : reason = f"did not meet target loss ({target_loss:.2e})"
                  elif best_result_so_far.get('stopped_by_insufficient_progress', False) and not is_last_rank_attempt: reason = "was stopped by insufficient progress"
                  elif is_last_rank_attempt and target_loss is not None and best_result_so_far['final_loss'] > target_loss: reason = f"finished last rank attempt without meeting target loss ({target_loss:.2e})"
                  else: reason = "completed all attempts"
                  tqdm.write(f"  Layer {layer_name} {reason} after max attempts. Using best result (Loss: {best_result_so_far['final_loss']:.2e}, Rank: {best_result_so_far['final_rank_used']}).")
             break # Break outer loop

        # Log reason for retry
        if args_global and args_global.verbose and attempt < max_rank_retries and not current_attempt_stopped_early_by_loss:
             # ... (retry logging as before) ...
            reason_for_retry = ""
            if current_attempt_stopped_by_projection : # This means perform_stop_checks was true
                proj_loss_info = ""
                if final_projected_loss_if_failed is not None: proj_loss_info = f" (Proj.FinalLoss: ~{final_projected_loss_if_failed:.2e})"
                reason_for_retry = f"because target loss {target_loss:.2e} was projected as unreachable{proj_loss_info} ({current_attempt_projection_type})"
            elif current_attempt_insufficient_progress: # This means perform_stop_checks was true
                reason_for_retry = "due to insufficient progress"
            elif current_attempt_iterations_done >= max_iterations and (target_loss is None or current_attempt_final_loss > target_loss):
                 reason_for_retry = f"after max iterations (Loss {current_attempt_final_loss:.2e}, Target not met)"
            if reason_for_retry: tqdm.write(f"  Layer {layer_name} (R:{current_rank_for_this_attempt}) attempt finished {reason_for_retry}. Will try next attempt...")
            elif not current_attempt_stopped_early_by_loss and (target_loss is None or current_attempt_final_loss > target_loss) :
                 tqdm.write(f"  Layer {layer_name} (R:{current_rank_for_this_attempt}) attempt finished (Loss {current_attempt_final_loss:.2e}). Target not met. Will try next attempt...")


    if 'hada_w1_a' not in best_result_so_far:
        # ... (return empty result)
        return {'final_loss': float('inf'), 'stopped_early_by_loss': False, 'stopped_by_insufficient_progress': False, 'stopped_by_projection': False, 'projection_type_used': 'none', 'iterations_done': 0, 'interrupted_mid_layer': True, 'final_rank_used': initial_rank_for_layer, 'final_projected_loss_on_stop': None}

    # ... (setdefault and return best_result_so_far as before) ...
    best_result_so_far.setdefault('stopped_early_by_loss', False); best_result_so_far.setdefault('stopped_by_insufficient_progress', False)
    best_result_so_far.setdefault('stopped_by_projection', False); best_result_so_far.setdefault('projection_type_used', 'none')
    best_result_so_far.setdefault('interrupted_mid_layer', False); best_result_so_far.setdefault('final_projected_loss_on_stop', None)

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
    if os.path.exists(intended_final_path): potential_files.append(intended_final_path)
    intermediate_pattern = os.path.join(output_dir, f"{base_save_name}_resume_L*{save_ext}")
    potential_files.extend(glob.glob(intermediate_pattern))
    best_file_path = None
    max_completed_modules = -1
    if not potential_files: return None, -1
    for file_path in sorted(potential_files):
        try:
            if not os.path.exists(file_path): continue
            with safetensors.safe_open(file_path, framework="pt", device="cpu") as f: metadata = f.metadata()
            if metadata and "ss_completed_loha_modules" in metadata:
                num_completed = len(json.loads(metadata["ss_completed_loha_modules"]))
                if num_completed > max_completed_modules:
                    max_completed_modules = num_completed; best_file_path = file_path
                elif num_completed == max_completed_modules:
                    # Prefer final file name if counts are equal
                    if file_path == intended_final_path and best_file_path != intended_final_path: best_file_path = file_path
                    # Keep final file if already selected and current is intermediate
                    elif best_file_path == intended_final_path and base_save_name+"_resume_L" in os.path.basename(file_path): pass
                    # Otherwise, prefer later intermediate files if counts equal (higher L number usually better)
                    elif base_save_name+"_resume_L" in os.path.basename(file_path): best_file_path = file_path

            elif max_completed_modules == -1: # If no files with metadata found yet
                # Treat files without metadata as having 0 completed, prefer final name if possible
                if best_file_path is None or (file_path == intended_final_path and best_file_path != intended_final_path):
                    best_file_path = file_path; max_completed_modules = 0 # Tentative count
                    if args_global and args_global.verbose: print(f"    File {os.path.basename(file_path)} has no 'ss_completed_loha_modules' metadata. Treating as 0 completed for now (potential candidate).")
        except Exception as e:
            print(f"    Warning: Could not read or parse metadata from {file_path}: {e}")
            # If error reading, still consider it as 0 if it's the final path and nothing better found
            if best_file_path is None and file_path == intended_final_path and max_completed_modules == -1 : best_file_path = file_path; max_completed_modules = 0
    
    if best_file_path: print(f"  Selected '{os.path.basename(best_file_path)}' for resume (estimated {max_completed_modules} modules based on metadata/filename).")
    elif not potential_files: pass # No files found is normal
    else: print(f"  Could not determine a best file to resume from among candidates, or no valid metadata found.")
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
        except OSError as e: print(f"  Warning: Could not clean up intermediate file {file_path}: {e}")
    if cleaned_count > 0: print(f"  Cleaned up {cleaned_count} intermediate file(s).")

def perform_graceful_save(output_path_to_save: str):
    global extracted_loha_state_dict_global, layer_optimization_stats_global, args_global, processed_layers_this_session_count_global, save_attempted_on_interrupt, skipped_identical_count_global, skipped_other_reason_count_global, keys_scanned_this_run_global, all_completed_module_prefixes_ever_global
    total_processed_ever = len(all_completed_module_prefixes_ever_global)
    
    # Check if there's anything to save
    if not extracted_loha_state_dict_global and not all_completed_module_prefixes_ever_global: 
        print(f"No layers were processed or loaded to save to {output_path_to_save}. Save aborted.")
        return

    args_to_use = args_global
    if not args_to_use: 
        print("Error: Global args not available for saving metadata.")
        return
        
    final_save_path = output_path_to_save
    
    # Determine save dtype
    if args_to_use.save_weights_dtype == "fp16": final_save_dtype_torch = torch.float16
    elif args_to_use.save_weights_dtype == "bf16": final_save_dtype_torch = torch.bfloat16
    else: final_save_dtype_torch = torch.float32 # Default to fp32
    
    # Prepare state dict with correct dtype
    final_state_dict_to_save = OrderedDict()
    for k, v_tensor in extracted_loha_state_dict_global.items():
        # Check if it's a floating point tensor before converting
        if hasattr(v_tensor, 'is_floating_point') and v_tensor.is_floating_point(): 
            final_state_dict_to_save[k] = v_tensor.to(final_save_dtype_torch)
        else: 
            final_state_dict_to_save[k] = v_tensor # Keep non-float tensors as they are
            
    print(f"\nAttempting to save LoHA for {total_processed_ever} unique modules in total ({processed_layers_this_session_count_global} new this session) to {final_save_path}")
    
    # Prepare metadata for safetensors
    eff_global_network_alpha_val = args_to_use.initial_alpha
    eff_global_network_alpha_str = f"{eff_global_network_alpha_val:.8f}" if eff_global_network_alpha_val is not None else str(args_to_use.rank)
    global_rank_str = str(args_to_use.rank)
    conv_rank_str = str(args_to_use.conv_rank if args_to_use.conv_rank is not None else args_to_use.rank)
    eff_conv_alpha_val = args_to_use.initial_conv_alpha
    conv_alpha_str = f"{eff_conv_alpha_val:.8f}" if eff_conv_alpha_val is not None else str(conv_rank_str) # Use conv_rank if alpha not set

    network_args_dict = {
        "algo": "loha", "dim": global_rank_str, "alpha": eff_global_network_alpha_str, 
        "conv_dim": conv_rank_str, "conv_alpha": conv_alpha_str, 
        "dropout": str(args_to_use.dropout), "rank_dropout": str(args_to_use.rank_dropout), 
        "module_dropout": str(args_to_use.module_dropout), 
        "use_tucker": "false", "use_scalar": "false", "block_size": "1",
    }

    prog_check_info = f"ProgCheck: Int={args_to_use.progress_check_interval}/StartWinAt={args_to_use.progress_check_start_iter}/Ratio={args_to_use.min_progress_loss_ratio:.1e}" if args_to_use.progress_check_interval > 0 else "ProgCheck: Off"
    adv_proj_info = f"AdvProjCaps: min={getattr(args_to_use, 'advanced_projection_decay_cap_min', 'N/A')}/max={getattr(args_to_use, 'advanced_projection_decay_cap_max', 'N/A')}" if args_to_use.progress_check_interval > 0 and args_to_use.target_loss is not None else ""
    
    sf_metadata = {
        "ss_network_module": "lycoris.kohya", 
        "ss_network_rank": global_rank_str, 
        "ss_network_alpha": eff_global_network_alpha_str, 
        "ss_network_algo": "loha",
        "ss_network_args": json.dumps(network_args_dict),
        "ss_comment": f"Extracted LoHA (Interrupt: {save_attempted_on_interrupt}). OptPrec: {args_to_use.precision}. SaveDtype: {args_to_use.save_weights_dtype}. ATOL: {args_to_use.atol_fp32_check}. Layers: {total_processed_ever}. MaxIter: {args_to_use.max_iterations}. TargetLoss: {args_to_use.target_loss}. MaxRankRetries: {args_to_use.max_rank_retries}. RankIncrFactor: {args_to_use.rank_increase_factor}. {prog_check_info}. {adv_proj_info}",
        "ss_base_model_name": os.path.splitext(os.path.basename(args_to_use.base_model_path))[0], 
        "ss_ft_model_name": os.path.splitext(os.path.basename(args_to_use.ft_model_path))[0],
        "ss_save_weights_dtype": args_to_use.save_weights_dtype, 
        "ss_optimization_precision": args_to_use.precision, 
        "ss_completed_loha_modules": json.dumps(list(all_completed_module_prefixes_ever_global)) # Store the set of completed modules
    }
    
    # Prepare extended metadata for JSON file
    # Convert Path objects or other non-serializable types to string for JSON
    serializable_args = {}
    for k, v in vars(args_to_use).items():
        if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            serializable_args[k] = v
        else:
            serializable_args[k] = str(v)

    json_metadata_for_file = {
        "comfyui_lora_type": "LyCORIS_LoHa", 
        "model_name": os.path.splitext(os.path.basename(final_save_path))[0],
        "base_model_path": args_to_use.base_model_path, 
        "ft_model_path": args_to_use.ft_model_path,
        "loha_extraction_settings": serializable_args,
        "extraction_summary": {
            "processed_layers_in_total_cumulative": total_processed_ever, 
            "processed_this_session": processed_layers_this_session_count_global, 
            "skipped_identical_count_this_session": skipped_identical_count_global, 
            "skipped_other_reason_count_this_session": skipped_other_reason_count_global, 
            "total_candidate_keys_scanned_in_loop_this_session": keys_scanned_this_run_global,
        },
        # Ensure stats are serializable (e.g., convert numpy floats if any)
        "layer_optimization_details_this_session": [{k: float(v) if isinstance(v, (torch.Tensor, float)) and k == 'final_loss' else v for k, v in stat.items()} for stat in layer_optimization_stats_global], 
        "embedded_safetensors_metadata": sf_metadata, 
        "interrupted_save": save_attempted_on_interrupt
    }
    
    # Save the files
    if final_save_path.endswith(".safetensors"):
        try: 
            save_file(final_state_dict_to_save, final_save_path, metadata=sf_metadata)
            print(f"LoHA state_dict saved to: {final_save_path}")
        except Exception as e: 
            print(f"Error saving .safetensors file: {e}")
            traceback.print_exc() # Print stack trace for debugging
            return # Abort if safetensors save fails
            
        # Save the extended metadata JSON companion file
        metadata_json_file_path = os.path.splitext(final_save_path)[0] + "_extraction_metadata.json"
        try:
            with open(metadata_json_file_path, 'w') as f: 
                json.dump(json_metadata_for_file, f, indent=4)
            print(f"Extended metadata saved to: {metadata_json_file_path}")
        except Exception as e: 
            print(f"Could not save extended metadata JSON: {e}")
            traceback.print_exc()
            
    else: # Basic .pt saving if not using .safetensors
        print(f"Saving to .pt format. Extended metadata will not be in a separate JSON.")
        # Note: .pt doesn't have a standard metadata field like safetensors.
        # We can save it within the dictionary, but other tools might not read it.
        save_dict = {
            'state_dict': final_state_dict_to_save, 
            '__metadata__': sf_metadata, # Use a non-standard key for metadata
            '__extended_metadata__': json_metadata_for_file # Embed JSON data too if needed
        }
        try:
            torch.save(save_dict, final_save_path)
            print(f"LoHA state_dict saved to: {final_save_path} (basic .pt save with embedded metadata)")
        except Exception as e:
             print(f"Error saving .pt file: {e}")
             traceback.print_exc()


def handle_interrupt(signum, frame):
    global save_attempted_on_interrupt, outer_pbar_global, args_global, all_completed_module_prefixes_ever_global
    print("\n" + "="*30 + "\nCtrl+C (SIGINT) detected!\n" + "="*30)
    if save_attempted_on_interrupt: 
        print("Save already attempted or in progress. Force exiting if stuck, or wait.")
        # Consider calling sys.exit(1) here if it should force exit on second interrupt
        return 
        
    save_attempted_on_interrupt = True # Set flag immediately
    
    # Attempt to close the progress bar cleanly
    if outer_pbar_global: 
        outer_pbar_global.close()
        
    print("Attempting to save progress for processed layers...")
    if args_global and args_global.save_to:
        # Use the total number of unique modules processed *ever* (resumed + current) for filename
        num_layers_for_filename = len(all_completed_module_prefixes_ever_global)
        interrupt_save_path = generate_intermediate_filename(args_global.save_to, num_layers_for_filename)
        print(f"Interrupt save will be to: {interrupt_save_path}")
        perform_graceful_save(output_path_to_save=interrupt_save_path) # Use the save function
    else: 
        print("Cannot perform interrupt save: args_global or save_to path not defined.")
        
    print("Graceful save attempt finished. Exiting.")
    sys.exit(0) # Exit after attempting save

def main(cli_args):
    global args_global, extracted_loha_state_dict_global, layer_optimization_stats_global, processed_layers_this_session_count_global, save_attempted_on_interrupt, outer_pbar_global, skipped_identical_count_global, skipped_other_reason_count_global, keys_scanned_this_run_global, previously_completed_module_prefixes_global, all_completed_module_prefixes_ever_global, main_loop_completed_scan_flag_global
    args_global = cli_args
    signal.signal(signal.SIGINT, handle_interrupt) # Register signal handler

    # --- Argument Defaults and Processing ---
    # Default for progress_check_start_iter
    if args_global.progress_check_start_iter is None: # If user did NOT provide a specific start iter
        if args_global.progress_check_interval > 0:
            # Default: Start the first progress window after one interval.
            # The first evaluation will occur after two intervals. Ensure it's at least 1.
            args_global.progress_check_start_iter = max(1, args_global.progress_check_interval)
        else:
            # Progress check is off, start_iter is moot but set it high to avoid logic errors.
            args_global.progress_check_start_iter = args_global.max_iterations + 1
    elif args_global.progress_check_interval <= 0 : # User set start_iter, but interval is off
         # Effectively disable start iter too if interval is off
         args_global.progress_check_start_iter = args_global.max_iterations + 1 

    # Determine Dtypes
    if args_global.precision == "fp16": target_opt_dtype = torch.float16
    elif args_global.precision == "bf16": target_opt_dtype = torch.bfloat16
    else: target_opt_dtype = torch.float32
    
    if args_global.save_weights_dtype == "fp16": final_save_dtype = torch.float16
    elif args_global.save_weights_dtype == "bf16": final_save_dtype = torch.bfloat16
    else: final_save_dtype = torch.float32
    
    # Print initial settings
    print(f"Using device: {args_global.device}, Opt Dtype: {target_opt_dtype}, Save Dtype: {final_save_dtype}")
    if args_global.target_loss: print(f"Target Loss: {args_global.target_loss:.2e} (after {args_global.min_iterations} min iters for target loss check)")
    else: print(f"No Target Loss specified. Min iterations for early stop check: {args_global.min_iterations}.")
    print(f"Max Iters/Layer: {args_global.max_iterations}, Max Rank Retries on Failure: {args_global.max_rank_retries}, Rank Increase Factor: {args_global.rank_increase_factor}")

    if args_global.progress_check_interval > 0:
        first_eval_iter = args_global.progress_check_start_iter + args_global.progress_check_interval
        print(f"Progress Check: Enabled. Interval: {args_global.progress_check_interval} iters, Min Rel. Loss Decrease: {args_global.min_progress_loss_ratio:.1e}.")
        print(f"  Progress window starts at iter: {args_global.progress_check_start_iter}, first evaluation at iter: {first_eval_iter}.")
        if args_global.target_loss is not None:
             print(f"  Projection Check: Enabled (if target loss specified). Decay Caps: min={getattr(args_global, 'advanced_projection_decay_cap_min', 'N/A')}, max={getattr(args_global, 'advanced_projection_decay_cap_max', 'N/A')}")
    else:
        print("Progress Check: Disabled (and Projection Check disabled).")
    # --- End Argument Processing ---

    # --- Resume Logic ---
    chosen_resume_file = None
    if not args_global.overwrite:
        print(f"\nChecking for existing LoHA file or resume states for: {args_global.save_to}")
        chosen_resume_file, num_modules_in_chosen_file = find_best_resume_file(args_global.save_to)
        if chosen_resume_file:
            print(f"  Attempting to resume from: {chosen_resume_file} (estimated {num_modules_in_chosen_file} modules).")
            try:
                file_metadata = None; completed_modules_in_file = set()
                # Use safe_open to read metadata first without loading all tensors
                with safetensors.safe_open(chosen_resume_file, framework="pt", device="cpu") as f:
                    file_metadata = f.metadata()
                    if file_metadata and "ss_completed_loha_modules" in file_metadata:
                        try:
                            # Parse the list of completed modules from metadata
                            completed_modules_in_file = set(json.loads(file_metadata.get("ss_completed_loha_modules")))
                            if len(completed_modules_in_file) != num_modules_in_chosen_file and num_modules_in_chosen_file >= 0 :
                                 if args_global.verbose: print(f"  Info: Metadata module count ({len(completed_modules_in_file)}) differs from find_best_resume_file's heuristic count ({num_modules_in_chosen_file}). Using parsed set from metadata.")
                        except json.JSONDecodeError: 
                            print(f"  Warning: Could not parse 'ss_completed_loha_modules' metadata from {os.path.basename(chosen_resume_file)}. Will load all tensors and infer completed from names if possible, or just load all.")
                            completed_modules_in_file = set() # Reset if parsing fails
                    elif not file_metadata or "ss_completed_loha_modules" not in file_metadata:
                         if args_global.verbose: print(f"  'ss_completed_loha_modules' not found or no metadata in {os.path.basename(chosen_resume_file)}'s metadata. Will load all tensors from it.")
                
                # Now load the tensors
                loaded_sd_for_resume = load_file(chosen_resume_file, device='cpu')
                
                # If metadata didn't provide completed list, try to infer from keys
                if not completed_modules_in_file and loaded_sd_for_resume:
                    inferred_prefixes = set()
                    for key in loaded_sd_for_resume.keys():
                        # Assume keys are like 'lora_unet_....hada_w1_a'
                        if key.endswith(".hada_w1_a"): # A reasonable anchor point for a LoHA module
                            inferred_prefixes.add(".".join(key.split('.')[:-1])) 
                    if inferred_prefixes:
                        if args_global.verbose: print(f"  Inferred {len(inferred_prefixes)} module prefixes from tensor keys in {os.path.basename(chosen_resume_file)} as 'ss_completed_loha_modules' was missing/empty.")
                        completed_modules_in_file = inferred_prefixes
                    elif not inferred_prefixes and num_modules_in_chosen_file > 0: # Filename suggested modules, but keys didn't match expected pattern
                        if args_global.verbose: print(f"  Warning: Filename suggested {num_modules_in_chosen_file} modules, but could not infer LoHA prefixes from keys in {os.path.basename(chosen_resume_file)}. Treating as empty for specific resume.")
                        
                # Load tensors based on the determined completed modules list
                resumed_tensor_count = 0
                if completed_modules_in_file:
                    for key, tensor_val in loaded_sd_for_resume.items():
                        module_prefix_for_check = ".".join(key.split('.')[:-1])
                        is_bias_key = key.endswith(".bias") # Also resume biases if they exist
                        
                        # Load if the module prefix is in the completed set OR if it's a bias
                        if module_prefix_for_check in completed_modules_in_file or is_bias_key: 
                            extracted_loha_state_dict_global[key] = tensor_val
                            resumed_tensor_count += 1
                            
                    previously_completed_module_prefixes_global = completed_modules_in_file
                    all_completed_module_prefixes_ever_global.update(previously_completed_module_prefixes_global)
                    print(f"  Successfully loaded {len(previously_completed_module_prefixes_global)} module prefixes with {resumed_tensor_count} tensors for resume from {os.path.basename(chosen_resume_file)}.")
                
                elif loaded_sd_for_resume: # No completed list, but file has content - load everything
                    if args_global.verbose: print(f"  No specific completed module list from metadata or inference. Loading all {len(loaded_sd_for_resume)} tensors from {os.path.basename(chosen_resume_file)}.")
                    extracted_loha_state_dict_global.update(loaded_sd_for_resume)
                    # Try to infer prefixes from the loaded keys again for tracking
                    inferred_prefixes = set()
                    for key in extracted_loha_state_dict_global.keys():
                         if key.endswith(".hada_w1_a"): inferred_prefixes.add(".".join(key.split('.')[:-1]))
                    if inferred_prefixes: 
                        previously_completed_module_prefixes_global = inferred_prefixes
                        all_completed_module_prefixes_ever_global.update(previously_completed_module_prefixes_global)
                        if args_global.verbose: print(f"  Inferred {len(previously_completed_module_prefixes_global)} module prefixes from all loaded tensors.")

                del loaded_sd_for_resume # Free memory

                # Check for companion JSON metadata
                resume_metadata_json_path = os.path.splitext(chosen_resume_file)[0] + "_extraction_metadata.json"
                if os.path.exists(resume_metadata_json_path):
                    try:
                        with open(resume_metadata_json_path, 'r') as f_meta: json.load(f_meta) # Try parsing
                        if args_global.verbose: print(f"  Found accompanying metadata JSON: {os.path.basename(resume_metadata_json_path)}")
                    except Exception as e_json: print(f"  Could not load or parse JSON metadata from {resume_metadata_json_path}: {e_json}")
            except Exception as e:
                print(f"  Error loading or processing chosen LoHA file '{chosen_resume_file}': {e}.")
                traceback.print_exc()
                print("  Starting fresh for new layers.")
                extracted_loha_state_dict_global.clear()
                previously_completed_module_prefixes_global.clear()
                all_completed_module_prefixes_ever_global.clear()
        else: print("  No suitable existing LoHA file found to resume from. Starting fresh.")
    elif args_global.overwrite and os.path.exists(args_global.save_to):
        print(f"Overwriting specified output file as per --overwrite: {args_global.save_to}")
        extracted_loha_state_dict_global.clear()
        previously_completed_module_prefixes_global.clear()
        all_completed_module_prefixes_ever_global.clear()
    # --- End Resume Logic ---

    # --- Load Base and Fine-tuned Models ---
    print(f"\nLoading base model: {args_global.base_model_path}")
    try:
        if args_global.base_model_path.endswith(".safetensors"): 
            base_model_sd = load_file(args_global.base_model_path, device='cpu')
        else: 
            base_model_pt = torch.load(args_global.base_model_path, map_location='cpu')
            # Handle common state_dict nesting
            base_model_sd = base_model_pt.get('state_dict', base_model_pt) 
            del base_model_pt
    except Exception as e:
        print(f"Error loading base model: {e}"); traceback.print_exc(); sys.exit(1)

    print(f"Loading fine-tuned model: {args_global.ft_model_path}")
    try:
        if args_global.ft_model_path.endswith(".safetensors"): 
            ft_model_sd = load_file(args_global.ft_model_path, device='cpu')
        else: 
            ft_model_pt = torch.load(args_global.ft_model_path, map_location='cpu')
            ft_model_sd = ft_model_pt.get('state_dict', ft_model_pt)
            del ft_model_pt
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}"); traceback.print_exc(); sys.exit(1)
    # --- End Model Loading ---

    # --- Main Optimization Loop ---
    processed_layers_this_session_count_global = 0
    skipped_identical_count_global = 0
    skipped_other_reason_count_global = 0
    keys_scanned_this_run_global = 0
    layer_optimization_stats_global.clear() # Clear stats for this run
    main_loop_completed_scan_flag_global = False # Reset flag

    # Find candidate keys (Linear and Conv weights common to both models)
    all_candidate_keys = []
    for k in base_model_sd.keys():
        if k.endswith('.weight') and k in ft_model_sd:
             # Check shape compatibility (2D for Linear, 4D for Conv)
            base_shape = base_model_sd[k].shape
            ft_shape = ft_model_sd[k].shape
            if base_shape == ft_shape and (len(base_shape) == 2 or len(base_shape) == 4): 
                all_candidate_keys.append(k)
            elif base_shape != ft_shape and args_global.verbose:
                 print(f"  Skipping {k} from candidate list due to shape mismatch: Base {base_shape}, FT {ft_shape}")

    all_candidate_keys.sort()
    total_candidates_to_scan = len(all_candidate_keys)
    print(f"Found {total_candidates_to_scan} candidate '.weight' keys common to both models and of suitable shape (Linear/Conv).")
    
    # Initialize progress bar
    outer_pbar_global = tqdm(total=total_candidates_to_scan, desc="Scanning Layers", dynamic_ncols=True, position=0)
    
    skipped_vae_layers_count = 0 # Counter for VAE skips
    
    try: # Wrap the main loop in try/finally to ensure progress bar closes
        for key_name in all_candidate_keys:
            if save_attempted_on_interrupt: break # Exit loop if Ctrl+C handled

            keys_scanned_this_run_global += 1
            outer_pbar_global.update(1)
            
            # --- Determine LoHA Key Prefix (Kohya format) ---
            # Needs adjustment based on the specific model architecture (SD1.5, SDXL, etc.)
            original_module_path = key_name[:-len(".weight")]
            
            # --- VAE SKIP LOGIC ---
            # Common VAE prefixes for Stable Diffusion
            is_vae_layer = False
            vae_prefixes_to_skip = [
                "first_stage_model.encoder.",
                "first_stage_model.decoder.",
                "first_stage_model.quant_conv.", # Often a single conv layer
                "first_stage_model.post_quant_conv.", # Often a single conv layer
                "var.decoder.", # SDXL VAE
                "var.encoder.", # SDXL VAE
                # Add other specific VAE prefixes if you identify them in your model
                # e.g., some SDXL VAEs might have paths like "vae.encoder.", "vae.decoder."
                # or "autoencoder.encoder.", "autoencoder.decoder."
            ]
            for vae_prefix in vae_prefixes_to_skip:
                if original_module_path.startswith(vae_prefix):
                    is_vae_layer = True
                    break
            
            if is_vae_layer:
                if args_global.verbose_layer_debug: # Use verbose_layer_debug for less clutter
                    tqdm.write(f"  Skipping VAE layer: {original_module_path}")
                skipped_vae_layers_count += 1
                skipped_other_reason_count_global += 1 # Count it in general skips
                # Add to completed set so we don't try again if --max_layers used and it's somehow "completed"
                # This assumes you *never* want to process these, even if identical.
                # Create a dummy loha_key_prefix just for adding to the skip set
                dummy_loha_prefix_for_vae_skip = "skipped_vae_" + original_module_path.replace(".", "_")
                all_completed_module_prefixes_ever_global.add(dummy_loha_prefix_for_vae_skip)
                continue
            # --- END VAE SKIP LOGIC ---
            
            loha_key_prefix = ""
            # Example prefixes (adjust as needed for your model structure)
            if original_module_path.startswith("model.diffusion_model."): # SD 1.x U-Net
                loha_key_prefix = "lora_unet_" + original_module_path[len("model.diffusion_model."):].replace(".", "_")
            elif original_module_path.startswith("first_stage_model."): # SD 1.x VAE
                 loha_key_prefix = "lora_vae_" + original_module_path[len("first_stage_model."):].replace(".", "_")
            elif "conditioner.embedders.0.transformer." in original_module_path : # SDXL Clip L
                idx = original_module_path.find("conditioner.embedders.0.transformer.") + len("conditioner.embedders.0.transformer.")
                loha_key_prefix = "lora_te1_" + original_module_path[idx:].replace(".", "_")
            elif "conditioner.embedders.1.model." in original_module_path : # SDXL OpenClip G / Other encoders
                 idx = original_module_path.find("conditioner.embedders.1.model.") + len("conditioner.embedders.1.model.")
                 sub_path = original_module_path[idx:]
                 # Handle potential nested 'transformer' etc.
                 if sub_path.startswith("transformer."): sub_path = sub_path[len("transformer."):]
                 loha_key_prefix = "lora_te2_" + sub_path.replace(".", "_")
            elif original_module_path.startswith("text_encoder.") : # SD 1.x Clip
                 loha_key_prefix = "lora_te_" + original_module_path[len("text_encoder."):].replace(".","_")
            # Add more elif conditions for other model types (e.g., SD 2.x, other VAE paths) if needed
            elif original_module_path.startswith("unet.") : # Common alternate Unet path
                 loha_key_prefix = "lora_unet_" + original_module_path[len("unet."):].replace(".","_")
            else: # Fallback generic prefix
                loha_key_prefix = "lora_" + original_module_path.replace(".", "_")
            # --- End LoHA Key Prefix ---

            # Check if already processed (from resume or earlier in this run)
            if loha_key_prefix in all_completed_module_prefixes_ever_global:
                # Update description but don't print skip message unless verbose
                outer_pbar_global.set_description_str(f"Scan {keys_scanned_this_run_global}/{total_candidates_to_scan} (Resumed: {len(previously_completed_module_prefixes_global)}, New Opt: {processed_layers_this_session_count_global})")
                if args_global.verbose_layer_debug: tqdm.write(f"  Skipping {loha_key_prefix} (already processed/resumed).")
                continue 

            # Check max_layers limit for *new* layers this session
            if args_global.max_layers is not None and args_global.max_layers > 0 and processed_layers_this_session_count_global >= args_global.max_layers:
                # Log only once when limit is first hit during the scan
                if args_global.verbose and processed_layers_this_session_count_global == args_global.max_layers:
                     layers_scanned_when_limit_hit = keys_scanned_this_run_global - (len(all_completed_module_prefixes_ever_global) - processed_layers_this_session_count_global) - skipped_identical_count_global - skipped_other_reason_count_global
                     if layers_scanned_when_limit_hit == (args_global.max_layers +1): # Only print the first time we hit the limit + 1 scan
                        tqdm.write(f"\nReached max_layers limit ({args_global.max_layers}) for new layers this session. Continuing scan only.")
                outer_pbar_global.set_description_str(f"Scan {keys_scanned_this_run_global}/{total_candidates_to_scan} (Max New Layers Reached)")
                continue # Continue scanning even if max layers hit

            # Get weights and calculate delta
            base_W = base_model_sd[key_name].to(dtype=torch.float32)
            ft_W = ft_model_sd[key_name].to(dtype=torch.float32)
            
            shape_info = get_module_shape_info_from_weight(base_W) # Already checked for None above
            out_dim, in_dim_effective, k_h, k_w, _, is_conv = shape_info

            # Calculate difference and check if negligible
            delta_W_fp32 = (ft_W - base_W)
            if torch.allclose(delta_W_fp32, torch.zeros_like(delta_W_fp32), atol=args_global.atol_fp32_check):
                if args_global.verbose_layer_debug: tqdm.write(f"  Skipping {loha_key_prefix} (weights identical within atol={args_global.atol_fp32_check:.1e}).")
                skipped_identical_count_global += 1
                # Add to completed set so we don't try again if --max_layers used
                all_completed_module_prefixes_ever_global.add(loha_key_prefix) 
                continue

            # Prepare for optimization
            max_layers_target_str = f"/{args_global.max_layers}" if args_global.max_layers is not None and args_global.max_layers > 0 else ""
            outer_pbar_global.set_description_str(f"Optimizing L{processed_layers_this_session_count_global + 1}{max_layers_target_str} (Scan {keys_scanned_this_run_global}/{total_candidates_to_scan})")
            if args_global.verbose: tqdm.write(f"\nProcessing Layer {processed_layers_this_session_count_global + 1}: {loha_key_prefix} (Orig: {original_module_path})")
            
            delta_W_target_for_opt = delta_W_fp32
            initial_rank_for_layer_opt = args_global.conv_rank if is_conv and args_global.conv_rank is not None else args_global.rank
            initial_alpha_for_layer_opt = args_global.initial_conv_alpha if is_conv else args_global.initial_alpha
            
            tqdm.write(f"  Optimizing: Shp: {list(base_W.shape)}, Initial R: {initial_rank_for_layer_opt}, Initial Alpha: {initial_alpha_for_layer_opt:.1f}")
            
            try: # Wrap the optimization call itself
                opt_results = optimize_loha_for_layer(
                    layer_name=loha_key_prefix, delta_W_target=delta_W_target_for_opt, 
                    out_dim=out_dim, in_dim_effective=in_dim_effective, 
                    k_h=k_h, k_w=k_w, 
                    initial_rank_for_layer=initial_rank_for_layer_opt, 
                    initial_alpha_for_layer=initial_alpha_for_layer_opt, 
                    lr=args_global.lr, max_iterations=args_global.max_iterations, 
                    min_iterations=args_global.min_iterations, 
                    target_loss=args_global.target_loss, 
                    weight_decay=args_global.weight_decay, 
                    device=args_global.device, dtype=target_opt_dtype, 
                    is_conv=is_conv, verbose_layer_debug=args_global.verbose_layer_debug, 
                    max_rank_retries=args_global.max_rank_retries, 
                    rank_increase_factor=args_global.rank_increase_factor
                )
                
                # Check results and store if successful
                if not opt_results.get('interrupted_mid_layer') and 'hada_w1_a' in opt_results :
                    # Store optimized parameters
                    for p_name, p_val in opt_results.items():
                        # Store only tensor parameters, exclude metadata keys
                        if p_name not in ['final_loss', 'stopped_early_by_loss', 'stopped_by_insufficient_progress', 'stopped_by_projection', 'projection_type_used', 'iterations_done', 'final_rank_used', 'interrupted_mid_layer']:
                            if torch.is_tensor(p_val): 
                                extracted_loha_state_dict_global[f'{loha_key_prefix}.{p_name}'] = p_val.to(final_save_dtype)
                                
                    final_rank_used_for_layer = opt_results['final_rank_used']
                    rank_was_increased = final_rank_used_for_layer > initial_rank_for_layer_opt
                    
                    # Store optimization statistics
                    layer_stats = {
                        "name": loha_key_prefix, 
                        "original_name": original_module_path, 
                        "initial_rank_attempted": initial_rank_for_layer_opt, 
                        "final_rank_used": final_rank_used_for_layer, 
                        "rank_was_increased": rank_was_increased, 
                        "final_loss": opt_results['final_loss'], 
                        "iterations_done": opt_results['iterations_done'], 
                        "stopped_early_by_loss_target": opt_results['stopped_early_by_loss'], 
                        "stopped_by_insufficient_progress": opt_results['stopped_by_insufficient_progress'],
                        "stopped_by_projection": opt_results.get('stopped_by_projection', False), # Use .get for backward compat if loading old results
                        "projection_type_used": opt_results.get('projection_type_used', 'none')
                    }
                    layer_optimization_stats_global.append(layer_stats)
                    
                    # Mark this module prefix as completed
                    all_completed_module_prefixes_ever_global.add(loha_key_prefix) 
                    
                    # Log layer completion summary
                    rank_info_str = f"R_used: {final_rank_used_for_layer}"
                    if rank_was_increased: rank_info_str += f" (Initial: {initial_rank_for_layer_opt})"
                    stop_reason = ""
                    if opt_results['stopped_early_by_loss']: stop_reason = ", Stop:Loss"
                    elif opt_results.get('stopped_by_projection', False): stop_reason = f", Stop:Proj({opt_results.get('projection_type_used','?')})"
                    elif opt_results['stopped_by_insufficient_progress']: stop_reason = ", Stop:Prog"
                    
                    tqdm.write(f"  Layer {loha_key_prefix} Done. {rank_info_str}, Loss: {opt_results['final_loss']:.4e}, Iters: {opt_results['iterations_done']}{stop_reason}")
                    
                    # Handle bias if requested
                    if args_global.use_bias:
                        original_bias_key = f"{original_module_path}.bias"
                        bias_differs = False
                        if original_bias_key in ft_model_sd: # Check if bias exists in FT model
                            ft_B = ft_model_sd[original_bias_key].to(dtype=torch.float32)
                            if original_bias_key in base_model_sd: # Check if bias exists in base model
                                base_B = base_model_sd[original_bias_key].to(dtype=torch.float32)
                                # Check if biases differ significantly
                                if not torch.allclose(base_B, ft_B, atol=args_global.atol_fp32_check): 
                                    bias_differs = True
                            else: # Bias exists in FT but not in base
                                bias_differs = True 
                                
                            if bias_differs: 
                                extracted_loha_state_dict_global[original_bias_key] = ft_B.cpu().to(final_save_dtype)
                                if args_global.verbose: tqdm.write(f"    Saved differing/new bias for {original_bias_key}")
                                
                    processed_layers_this_session_count_global += 1 # Increment count of successfully processed layers
                    
                else: # Optimization interrupted or failed to produce tensors
                     tqdm.write(f"  Opt for {loha_key_prefix} interrupted or yielded no valid tensors; not saving params. Interrupt: {opt_results.get('interrupted_mid_layer', 'N/A')}, Loss: {opt_results.get('final_loss', 'N/A')}")
                     # Count as skipped if not interrupted but failed for other reason
                     if not opt_results.get('interrupted_mid_layer', False) and 'hada_w1_a' not in opt_results : 
                         skipped_other_reason_count_global += 1
                         # Add to completed set so we don't retry if --max_layers used
                         all_completed_module_prefixes_ever_global.add(loha_key_prefix) 

            except Exception as e: 
                print(f"\nError during optimization call for {original_module_path} ({loha_key_prefix}): {e}")
                traceback.print_exc()
                skipped_other_reason_count_global +=1
                # Add to completed set so we don't retry if --max_layers used
                all_completed_module_prefixes_ever_global.add(loha_key_prefix) 

        # Check if the loop completed fully without interrupt
        if not save_attempted_on_interrupt and keys_scanned_this_run_global == total_candidates_to_scan: 
            main_loop_completed_scan_flag_global = True
            
    finally: # Ensure progress bar is closed
        if outer_pbar_global:
            # Ensure the bar visually completes if loop finished early
            if not outer_pbar_global.disable and hasattr(outer_pbar_global, 'n') and hasattr(outer_pbar_global, 'total') and outer_pbar_global.n < outer_pbar_global.total: 
                outer_pbar_global.update(outer_pbar_global.total - outer_pbar_global.n)
            outer_pbar_global.close()
    # --- End Main Optimization Loop ---

    # --- Final Summary and Saving ---
    if not save_attempted_on_interrupt:
        print("\n--- Final Optimization Summary (This Session) ---")
        for stat in layer_optimization_stats_global:
            rank_info = f"InitialR: {stat['initial_rank_attempted']}, FinalR: {stat['final_rank_used']}"
            if stat['rank_was_increased']: rank_info += " (Increased)"
            stop_info = ""
            if stat['stopped_early_by_loss_target']: stop_info = ", Stop:Loss"
            elif stat.get('stopped_by_projection', False): stop_info = f", Stop:Proj({stat.get('projection_type_used','?')})"
            elif stat.get('stopped_by_insufficient_progress', False): stop_info = ", Stop:Prog"
            
            print(f"Layer: {stat['name']}, {rank_info}, Loss: {stat['final_loss']:.4e}, Iters: {stat['iterations_done']}{stop_info}")
            
        print(f"\n--- Overall Summary ---")
        print(f"Total unique LoHA modules accumulated (resumed + new): {len(all_completed_module_prefixes_ever_global)}")
        print(f"  Processed new this session: {processed_layers_this_session_count_global}")
        print(f"  Skipped as identical (this session's scan): {skipped_identical_count_global}")
        print(f"  Skipped for other reasons (this session's scan, e.g., shape, opt error): {skipped_other_reason_count_global}")
        print(f"  Total candidate keys scanned in loop (this session): {keys_scanned_this_run_global}/{total_candidates_to_scan}")
        
        actual_save_path: str
        save_to_final_name = False
        
        # Determine if ready for final save
        if main_loop_completed_scan_flag_global:
            # Calculate how many layers *could* have been optimized (excluding identical/error)
            # Note: This includes layers skipped due to error/other reasons in *this* run.
            # We also add layers skipped as identical, as they are "accounted for".
            # We add previously completed layers from resume.
            num_accounted_for_layers = len(previously_completed_module_prefixes_global) + \
                                        processed_layers_this_session_count_global + \
                                        skipped_identical_count_global + \
                                        skipped_other_reason_count_global

            # If total accumulated modules >= total candidates, assume all done.
            # This logic might be imperfect if errors caused skips.
            # A safer check might be if `num_accounted_for_layers >= total_candidates_to_scan`
            if len(all_completed_module_prefixes_ever_global) >= total_candidates_to_scan:
                 save_to_final_name = True
            elif num_accounted_for_layers >= total_candidates_to_scan :
                 save_to_final_name = True # All candidates were either resumed, processed, skipped ident, or skipped error.
                 if args_global.verbose: print(f"  All {total_candidates_to_scan} candidates accounted for (Resumed/Processed/Skipped).")
            else: 
                 print(f"  Scan completed, but not all {total_candidates_to_scan} candidates are processed ({len(all_completed_module_prefixes_ever_global)} unique modules processed total).")
        else: 
            print("  Scan did not complete fully (likely interrupted or --max_layers hit early). Saving intermediate state.")
            
        # Decide final vs intermediate save path
        if save_to_final_name:
            actual_save_path = args_global.save_to
            print(f"\nAll differing/optimizable layers from this scan appear to be processed or accounted for. Saving to final path: {actual_save_path}")
        else:
            num_layers_for_filename = len(all_completed_module_prefixes_ever_global)
            actual_save_path = generate_intermediate_filename(args_global.save_to, num_layers_for_filename)
            if main_loop_completed_scan_flag_global and not save_to_final_name: 
                print(f"\nFull scan completed, but not all layers processed yet. Saving intermediate state to: {actual_save_path}")
            else: # Incomplete scan or max_layers hit
                print(f"\nRun incomplete or --max_layers limit reached before full processing. Saving intermediate state to: {actual_save_path}")
                
        # Perform the save
        perform_graceful_save(output_path_to_save=actual_save_path)
        
        # Clean up intermediate files ONLY if the final save was successful
        if save_to_final_name and actual_save_path == args_global.save_to :
            print("\nCleaning up intermediate resume files...")
            cleanup_intermediate_files(args_global.save_to)
            
    else: # Process was interrupted
        print("\nProcess was interrupted. Graceful save to an intermediate file was attempted by signal handler.")
    # --- End Final Summary and Saving ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LoHA parameters by optimizing against weight differences. Saves intermediate files like 'name_resume_L{count}.safetensors'.")
    
    # --- Positional Arguments ---
    parser.add_argument("base_model_path", type=str, help="Path to the base model state_dict file (.pt, .pth, .safetensors)")
    parser.add_argument("ft_model_path", type=str, help="Path to the fine-tuned model state_dict file (.pt, .pth, .safetensors)")
    parser.add_argument("save_to", type=str, help="Path to save the FINAL extracted LoHA file (recommended .safetensors). Intermediate files will be based on this name.")
    
    # --- File Handling & Resume ---
    parser.add_argument("--overwrite", action="store_true", help="Ignore and overwrite any existing FINAL LoHA output file. Does NOT clean intermediates until a successful final save of the current run.")
    
    # --- LoHA Parameters ---
    parser.add_argument("--rank", type=int, default=4, help="Default rank for LoHA decomposition (used for linear layers and as fallback for conv).")
    parser.add_argument("--conv_rank", type=int, default=None, help="Specific rank for convolutional LoHA layers. Defaults to --rank if not set.")
    parser.add_argument("--initial_alpha", type=float, default=None, help="Global initial alpha for optimization. Defaults to 'rank'.")
    parser.add_argument("--initial_conv_alpha", type=float, default=None, help="Specific initial alpha for Conv LoHA. Defaults to '--initial_alpha' or conv_rank.")
    
    # --- Optimization Parameters ---
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for LoHA optimization per layer.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of optimization iterations per layer PER ATTEMPT (rank increase restarts iterations).")
    parser.add_argument("--min_iterations", type=int, default=100, help="Minimum iterations before checking target_loss PER ATTEMPT. This does NOT prevent earlier stop by progress check.")
    parser.add_argument("--target_loss", type=float, default=None, help="Target MSE loss for early stopping per layer. If met (after min_iterations), rank increase is not attempted.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for LoHA optimization.")
    
    # --- Rank Retry Parameters ---
    parser.add_argument("--max_rank_retries", type=int, default=0, help="Number of times to increase rank and retry if target_loss is not met or progress is too slow (0 for no retries). Default: 0.")
    parser.add_argument("--rank_increase_factor", type=float, default=1.25, help="Factor by which to increase rank on each retry. Rank is ceiling rounded. Default: 1.25.")
    
    # --- Progress Check Parameters ---
    parser.add_argument("--progress_check_interval", type=int, default=100, help="Check loss improvement every N iterations. Set to 0 to disable. Default: 100.")
    parser.add_argument("--min_progress_loss_ratio", type=float, default=0.001, help="Minimum relative loss decrease required over interval (e.g., 0.001 for 0.1%%). Stops attempt if not met. Default: 0.001.")
    parser.add_argument("--progress_check_start_iter", type=int, default=None, help="Iteration to record loss for the START of the first progress window. First evaluation occurs 'interval' iterations later. Default: value of 'progress_check_interval'.")
    parser.add_argument("--advanced_projection_decay_cap_min", type=float, default=0.5, help="Min cap for the estimated decay factor of relative improvement in advanced projection. Default: 0.5")
    parser.add_argument("--advanced_projection_decay_cap_max", type=float, default=1.05, help="Max cap for the estimated decay factor of relative improvement in advanced projection. Default: 1.05")

    # --- Technical Parameters ---
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device ('cuda' or 'cpu').")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Optimization precision. Default: fp32.")
    parser.add_argument("--save_weights_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Dtype for saved LoHA weights. Default: bf16.")
    parser.add_argument("--atol_fp32_check", type=float, default=1e-6, help="Tolerance for identical weight check (weights considered same if |W_ft - W_base| <= atol). Default: 1e-6.")
    parser.add_argument("--no_warm_start", action="store_true", help="Disable warm-starting higher rank attempts from previous best results.") # <-- ADDED

    # --- Metadata & Output Options ---
    parser.add_argument("--use_bias", action="store_true", help="Save differing bias terms directly into the LoHA file.")
    parser.add_argument("--dropout", type=float, default=0.0, help="General dropout (metadata only).")
    parser.add_argument("--rank_dropout", type=float, default=0.0, help="Rank dropout (metadata only).")
    parser.add_argument("--module_dropout", type=float, default=0.0, help="Module dropout (metadata only).")
    parser.add_argument("--max_layers", type=int, default=None, help="Max NEW differing layers to process this session. Scan will continue to assess all layers. Useful for partial runs.")
    parser.add_argument("--verbose", action="store_true", help="General verbose output (progress, major steps).")
    parser.add_argument("--verbose_layer_debug", action="store_true", help="Detailed per-iteration optimization debug output (implies --verbose).")
    
    # --- EMA Projection Parameters ---
    parser.add_argument("--projection_sample_interval", type=int, default=20, help="How often to sample loss for EMA smoothing (iterations). Default: 20.")
    parser.add_argument("--projection_ema_alpha", type=float, default=0.1, help="Smoothing factor for EMA (0 to 1, lower is more smoothing). Default: 0.1.")
    parser.add_argument("--projection_min_ema_history", type=int, default=5, help="Minimum number of EMA samples required before using EMA-based projection. Default: 5.")
    
    # Parse arguments
    parsed_args = parser.parse_args()

    # Post-processing for args
    if parsed_args.verbose_layer_debug: # Enable base verbose if debug is on
        parsed_args.verbose = True
        
    # Validate paths
    if not os.path.exists(parsed_args.base_model_path): 
        print(f"Error: Base model path not found: {parsed_args.base_model_path}"); sys.exit(1)
    if not os.path.exists(parsed_args.ft_model_path): 
        print(f"Error: Fine-tuned model path not found: {parsed_args.ft_model_path}"); sys.exit(1)
        
    # Ensure save directory exists
    save_dir = os.path.dirname(parsed_args.save_to)
    if save_dir and not os.path.exists(save_dir):
        try: 
            os.makedirs(save_dir, exist_ok=True)
            if parsed_args.verbose: print(f"Created directory: {save_dir}")
        except OSError as e: 
            print(f"Error: Could not create directory {save_dir}: {e}"); sys.exit(1)
            
    # Default alpha values if not provided
    if parsed_args.initial_alpha is None: 
        parsed_args.initial_alpha = float(parsed_args.rank)
    if parsed_args.initial_conv_alpha is None:
        # Default conv alpha to conv rank if set, otherwise fallback to global alpha
        conv_rank_for_alpha_default = parsed_args.conv_rank if parsed_args.conv_rank is not None else parsed_args.rank
        parsed_args.initial_conv_alpha = float(conv_rank_for_alpha_default) if parsed_args.conv_rank is not None else parsed_args.initial_alpha

    # Run main function
    main(parsed_args)