import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
import safetensors
from tqdm import tqdm
import math
import json
from collections import OrderedDict
import signal
import sys
import glob
import traceback
import re
from enum import Enum, auto

# --- Global variables ---
extracted_loha_state_dict_global = OrderedDict()
layer_optimization_stats_global = []
args_global = None 
processed_layers_this_session_count_global = 0
previously_completed_module_prefixes_global = set()
all_completed_module_prefixes_ever_global = set()
skipped_identical_count_global = 0
skipped_other_reason_count_global = 0
skipped_good_initial_loss_count_global = 0
keys_scanned_this_run_global = 0
save_attempted_on_interrupt = False
outer_pbar_global = None
main_loop_completed_scan_flag_global = False
params_to_seed_optimizer_global = {}
skipped_vae_layers_count = 0 # Ensure this is a global if accessed in main and other places

# --- Logging Helper ---
class LogType(Enum):
    RANK_RETRY_STARTING = auto()
    RANK_INCREASED_INFO = auto()
    INITIAL_PARAMS_LOADED = auto()
    INITIAL_PARAMS_KAIMING_NORMAL = auto()
    INSUFFICIENT_PROGRESS_STOP = auto()
    PROJECTION_STOP = auto()
    INSUFFICIENT_PROGRESS_LOG_ONLY = auto()
    PROJECTION_LOG_ONLY = auto()
    EMA_PROJECTION_SKIPPED_HISTORY = auto()
    EMA_PROJECTION_INCONCLUSIVE_FALLBACK_RAW = auto()
    NEW_BEST_RESULT_FOR_LAYER = auto()
    TARGET_LOSS_REACHED_IN_ATTEMPT = auto()
    TARGET_LOSS_MET_STOP_ALL_RETRIES = auto()
    ATTEMPT_EARLY_FINISH_NO_STOP_FLAG = auto()
    LAST_RANK_ATTEMPT_SUMMARY = auto()
    ATTEMPT_ENDED_WILL_RETRY = auto()
    NO_VALID_OPTIMIZATION_RESULT = auto()

def log_layer_optimization_event(log_type: LogType, layer_name: str, **kwargs):
    if not (args_global and args_global.verbose):
        return
    if not args_global.verbose_layer_debug:
        if log_type in [
            LogType.INITIAL_PARAMS_LOADED, LogType.INITIAL_PARAMS_KAIMING_NORMAL,
            LogType.EMA_PROJECTION_SKIPPED_HISTORY, LogType.EMA_PROJECTION_INCONCLUSIVE_FALLBACK_RAW
        ]:
            return
    prefix = f"  {layer_name}: "
    msg = ""
    if log_type == LogType.RANK_RETRY_STARTING:
        msg = f"Retrying (PrevR: {kwargs.get('prev_rank', 'N/A')}, BestLoss: {kwargs.get('prev_best_loss', float('inf')):.2e}) -> Increasing rank..."
    elif log_type == LogType.RANK_INCREASED_INFO:
        warm_start_msg = ""
        status = kwargs.get('warm_start_status')
        if status == 'applied': warm_start_msg = f" Warm-start from R: {kwargs['prev_rank_for_warm_start']}."
        elif status == 'skipped_no_warm_start_arg': warm_start_msg = " Warm-start skipped (--no_warm_start)."
        elif status == 'skipped_cannot_warm_start': warm_start_msg = f" Cannot warm-start (PrevR {kwargs['prev_rank_for_warm_start']} !< NewR {kwargs['new_rank']})."
        elif status == 'no_prior_params_for_warm_start': warm_start_msg = " No prior best params to warm-start."
        msg = f"  Increased Rank: {kwargs['new_rank']}, Alpha: {kwargs['new_alpha']:.2f}.{warm_start_msg}"
    elif log_type == LogType.INITIAL_PARAMS_LOADED:
        msg = f"    R:{kwargs['rank']} Initialized from existing LoHA."
    elif log_type == LogType.INITIAL_PARAMS_KAIMING_NORMAL:
        msg = f"    R:{kwargs['rank']} Initialized Kaiming/Normal (Attempt {kwargs.get('attempt', 1)})."
    elif log_type == LogType.INSUFFICIENT_PROGRESS_STOP:
        msg = f"Att {kwargs['attempt']}(R:{kwargs['rank']}): Stop - RawProg Low (Imprv: {kwargs['rel_imprv']:.1e} < {kwargs['min_ratio']:.1e}; Loss: {kwargs['current_loss']:.2e})."
    elif log_type == LogType.PROJECTION_STOP:
        details = f"Est. {kwargs.get('iters_needed', 'inf'):.0f} iters"
        if kwargs.get('proj_final_loss') is not None: details += f", ProjLoss: ~{kwargs['proj_final_loss']:.2e}"
        else: details += ", Target Unreachable"
        msg = f"Att {kwargs['attempt']}(R:{kwargs['rank']}): Stop - Proj ({kwargs['proj_type']}). {details} vs Target: {kwargs['target_loss']:.2e} (Avail: {kwargs['avail_iters']})."
    elif log_type == LogType.INSUFFICIENT_PROGRESS_LOG_ONLY:
        msg = f"Att {kwargs['attempt']}(R:{kwargs['rank']}) [LastRankLog]: RawProg Low (Imprv: {kwargs['rel_imprv']:.1e} < {kwargs['min_ratio']:.1e}; Loss: {kwargs['current_loss']:.2e})."
    elif log_type == LogType.PROJECTION_LOG_ONLY:
        details = f"Est. {kwargs.get('iters_needed', 'inf'):.0f} iters"
        if kwargs.get('proj_final_loss') is not None: details += f", ProjLoss: ~{kwargs['proj_final_loss']:.2e}"
        else: details += ", Target Unreachable"
        msg = f"Att {kwargs['attempt']}(R:{kwargs['rank']}) [LastRankLog]: Proj ({kwargs['proj_type']}). {details} vs Target: {kwargs['target_loss']:.2e} (Avail: {kwargs['avail_iters']})."
    elif log_type == LogType.EMA_PROJECTION_SKIPPED_HISTORY:
        msg = f"    Att {kwargs['attempt']}(R:{kwargs['rank']}): EMA proj. skipped (Hist {kwargs['hist_len']}/{kwargs['min_hist']})."
    elif log_type == LogType.EMA_PROJECTION_INCONCLUSIVE_FALLBACK_RAW:
        msg = f"    Att {kwargs['attempt']}(R:{kwargs['rank']}): EMA proj. inconclusive, using raw."
    elif log_type == LogType.NEW_BEST_RESULT_FOR_LAYER:
        msg = f"Att {kwargs['attempt']}(R:{kwargs['rank']}): New Best -> Loss {kwargs['loss']:.2e}."
    elif log_type == LogType.TARGET_LOSS_REACHED_IN_ATTEMPT:
        msg = f"Att {kwargs['attempt']}(R:{kwargs['rank']}): Target loss {kwargs['target_loss']:.2e} met at iter {kwargs['iter']}."
    elif log_type == LogType.TARGET_LOSS_MET_STOP_ALL_RETRIES:
        msg = "Target loss met. Halting rank retries for this layer."
    elif log_type == LogType.ATTEMPT_EARLY_FINISH_NO_STOP_FLAG:
        msg = f"Att {kwargs['attempt']}(R:{kwargs['rank']}): Finished early ({kwargs['iters_done']}/{kwargs['max_iters']}), no stop/target. Using result."
    elif log_type == LogType.LAST_RANK_ATTEMPT_SUMMARY:
        reason = "All rank attempts completed."
        if kwargs.get('target_loss') is not None and kwargs.get('final_loss_for_layer',0) > kwargs['target_loss']:
            reason = f"Last rank (R:{kwargs['final_rank_for_layer']}) finished, target {kwargs['target_loss']:.2e} not met."
        msg = f"{reason} Best: Loss {kwargs.get('final_loss_for_layer',0):.2e}, Rank {kwargs['final_rank_for_layer']}."
    elif log_type == LogType.ATTEMPT_ENDED_WILL_RETRY:
        reason_detail = ""
        if kwargs.get('reason_type') == 'projection_unreachable':
            proj_loss_info = f" (ProjL: ~{kwargs['proj_final_loss']:.2e})" if kwargs.get('proj_final_loss') is not None else ""
            reason_detail = f"target {kwargs['target_loss']:.2e} proj. unreachable{proj_loss_info} ({kwargs['proj_type']})"
        elif kwargs.get('reason_type') == 'insufficient_progress': reason_detail = "insufficient raw progress"
        elif kwargs.get('reason_type') == 'max_iterations_no_target': reason_detail = f"max iters (Loss {kwargs['current_loss']:.2e}, Target not met)"
        elif kwargs.get('reason_type') == 'max_iterations_no_target_set': reason_detail = f"max iters (Loss {kwargs['current_loss']:.2e})"
        if reason_detail:
             msg = f"Att {kwargs['attempt']}(R:{kwargs['rank']}) ended: {reason_detail}. Will try next rank..."
    elif log_type == LogType.NO_VALID_OPTIMIZATION_RESULT:
        msg = "No valid optimization result (likely interrupted)."
    if msg:
        tqdm.write(prefix + msg)

def _get_closest_ema_value_before_iter(target_iter: int, ema_history: list[tuple[int, float]]) -> tuple[int | None, float | None]:
    if not ema_history: return None, None
    best_match_iter, best_match_loss = None, None
    for hist_iter, hist_loss in reversed(ema_history):
        if hist_iter <= target_iter:
            if best_match_iter is None or hist_iter > best_match_iter :
                best_match_iter, best_match_loss = hist_iter, hist_loss
            if best_match_iter is not None and target_iter - hist_iter > getattr(args_global, 'projection_sample_interval', 20) * 2: break
    return (best_match_iter, best_match_loss) if best_match_iter is not None else (ema_history[0] if ema_history else (None, None))

def initialize_loha_parameters(
    out_dim: int, current_rank: int, in_dim_effective_k_ops: int,
    device: str, dtype: torch.dtype, layer_name: str, attempt_idx: int,
    is_continuation_attempt: bool,
    existing_params_to_load: dict | None = None,
    warm_start_status: str | None = None,
    prev_rank_for_warm_start: int | None = None
):
    hada_w1_a_p = nn.Parameter(torch.empty(out_dim, current_rank, device=device, dtype=dtype))
    hada_w1_b_p = nn.Parameter(torch.empty(current_rank, in_dim_effective_k_ops, device=device, dtype=dtype))
    hada_w2_a_p = nn.Parameter(torch.empty(out_dim, current_rank, device=device, dtype=dtype))
    hada_w2_b_p = nn.Parameter(torch.empty(current_rank, in_dim_effective_k_ops, device=device, dtype=dtype))
    initialized_from_external_or_warm_start = False
    with torch.no_grad():
        if is_continuation_attempt and attempt_idx == 0 and existing_params_to_load:
            try:
                if existing_params_to_load['hada_w1_a'].shape[1] == current_rank:
                    for p_name_suffix in ['hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b']:
                        target_param = locals()[f"{p_name_suffix}_p"]
                        target_param.data.copy_(existing_params_to_load[p_name_suffix].to(device, dtype))
                    log_layer_optimization_event(LogType.INITIAL_PARAMS_LOADED, layer_name, rank=current_rank)
                    initialized_from_external_or_warm_start = True
            except Exception as e:
                if args_global.verbose_layer_debug:
                    tqdm.write(f"    {layer_name}: Error loading initial existing params: {e}. Falling back.")
                pass
        if not initialized_from_external_or_warm_start and warm_start_status == 'applied' and existing_params_to_load and prev_rank_for_warm_start is not None:
            prev_params_cpu = {k: existing_params_to_load[k] for k in ['hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b']}
            hada_w1_a_p.data[:, :prev_rank_for_warm_start] = prev_params_cpu['hada_w1_a'].to(device, dtype)
            hada_w1_b_p.data[:prev_rank_for_warm_start, :] = prev_params_cpu['hada_w1_b'].to(device, dtype)
            hada_w2_a_p.data[:, :prev_rank_for_warm_start] = prev_params_cpu['hada_w2_a'].to(device, dtype)
            hada_w2_b_p.data[:prev_rank_for_warm_start, :] = prev_params_cpu['hada_w2_b'].to(device, dtype)
            if current_rank > prev_rank_for_warm_start:
                for p_slice in [hada_w1_a_p.data[:, prev_rank_for_warm_start:], hada_w2_a_p.data[:, prev_rank_for_warm_start:]]:
                    nn.init.kaiming_uniform_(p_slice, a=math.sqrt(5))
                for p_slice in [hada_w1_b_p.data[prev_rank_for_warm_start:, :], hada_w2_b_p.data[prev_rank_for_warm_start:, :]]:
                    nn.init.normal_(p_slice, std=0.02)
            initialized_from_external_or_warm_start = True
        if not initialized_from_external_or_warm_start:
            log_layer_optimization_event(LogType.INITIAL_PARAMS_KAIMING_NORMAL, layer_name, rank=current_rank, attempt=attempt_idx + 1)
            for p in [hada_w1_a_p, hada_w2_a_p]: nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
            for p in [hada_w1_b_p, hada_w2_b_p]: nn.init.normal_(p.data, std=0.02)
    return hada_w1_a_p, hada_w1_b_p, hada_w2_a_p, hada_w2_b_p

def check_insufficient_progress(
    current_loss: float,
    loss_at_window_start: float,
    min_progress_ratio: float,
    target_loss: float | None,
    perform_stop_check: bool, 
    layer_name: str,
    attempt_idx: int, 
    rank: int
) -> tuple[bool, dict | None]:
    raw_rel_imprv = 0.0
    if loss_at_window_start > 1e-12 and loss_at_window_start > current_loss:
        raw_rel_imprv = (loss_at_window_start - current_loss) / loss_at_window_start
    log_details = {
        'attempt': attempt_idx + 1, 'rank': rank,
        'rel_imprv': raw_rel_imprv, 'min_ratio': min_progress_ratio,
        'current_loss': current_loss
    }
    if (target_loss is None or current_loss > target_loss * 1.01) and raw_rel_imprv < min_progress_ratio:
        log_event_type = LogType.INSUFFICIENT_PROGRESS_STOP if perform_stop_check else LogType.INSUFFICIENT_PROGRESS_LOG_ONLY
        log_layer_optimization_event(log_event_type, layer_name, **log_details)
        if perform_stop_check:
            return True, log_details
    return False, None

def check_loss_projection(
    ema_loss_history: list[tuple[int, float]], 
    current_raw_loss: float,
    raw_rel_imprv_for_fallback: float,
    target_loss: float, 
    max_total_iters_for_attempt: int,
    current_iter_num: int, 
    prog_check_interval: int,
    proj_min_ema_hist: int,
    adv_proj_decay_cap_min: float,
    adv_proj_decay_cap_max: float,
    perform_stop_check: bool, 
    layer_name: str,
    attempt_idx: int, 
    rank: int,
    relative_improvement_history_ref: list[float]
) -> tuple[bool, dict | None]:
    use_ema = len(ema_loss_history) >= proj_min_ema_hist
    if not use_ema:
        log_layer_optimization_event(LogType.EMA_PROJECTION_SKIPPED_HISTORY, layer_name,
                                     attempt=attempt_idx + 1, rank=rank,
                                     hist_len=len(ema_loss_history), min_hist=proj_min_ema_hist)
    req_iters_to_target = float('inf')
    proj_type_at_check = "none"
    projected_final_loss_val = None
    iters_remaining_in_attempt = max_total_iters_for_attempt - current_iter_num
    num_intervals_remaining = iters_remaining_in_attempt // prog_check_interval if prog_check_interval > 0 else 0

    if use_ema:
        ema_curr_iter, ema_curr_loss = ema_loss_history[-1]
        ema_start_iter, ema_start_loss = _get_closest_ema_value_before_iter(current_iter_num - prog_check_interval, ema_loss_history)
        smooth_ema_imprv = 0.0
        if ema_start_loss and ema_start_loss > ema_curr_loss and ema_start_iter is not None and ema_curr_iter > ema_start_iter:
            smooth_ema_imprv = (ema_start_loss - ema_curr_loss) / ema_start_loss
        if smooth_ema_imprv > 1e-9:
            relative_improvement_history_ref.append(smooth_ema_imprv)
            relative_improvement_history_ref[:] = relative_improvement_history_ref[-2:]
            if len(relative_improvement_history_ref) >= 2 and relative_improvement_history_ref[-2] > 1e-9:
                proj_type_at_check = "advanced_ema"
                decay_R_factor = max(adv_proj_decay_cap_min, min(adv_proj_decay_cap_max, smooth_ema_imprv / relative_improvement_history_ref[-2]))
                sim_loss, sim_R_per_interval, sim_iters_elapsed = ema_curr_loss, smooth_ema_imprv, 0
                if sim_loss > target_loss:
                    for _window_idx in range(num_intervals_remaining + 1):
                        sim_loss *= (1.0 - max(0, sim_R_per_interval))
                        sim_iters_elapsed += prog_check_interval
                        if sim_loss <= target_loss: break
                        if sim_iters_elapsed >= iters_remaining_in_attempt + prog_check_interval: break
                        sim_R_per_interval = max(1e-7, sim_R_per_interval * decay_R_factor)
                    req_iters_to_target = sim_iters_elapsed if sim_loss <= target_loss else float('inf')
                    if req_iters_to_target == float('inf'): projected_final_loss_val = sim_loss
            else:
                proj_type_at_check = "simple_ema" + (" (fallback_adv)" if len(relative_improvement_history_ref) >=2 else "")
                if ema_curr_loss > target_loss:
                    try:
                        intervals_needed = math.log(target_loss / ema_curr_loss) / math.log(1.0 - smooth_ema_imprv)
                        req_iters_to_target = math.ceil(intervals_needed) * prog_check_interval
                    except (ValueError, OverflowError, ZeroDivisionError):
                        req_iters_to_target = float('inf')
                        projected_final_loss_val = ema_curr_loss * ((1.0 - max(0, smooth_ema_imprv))**(num_intervals_remaining + 1))
                else: req_iters_to_target = 0
        elif use_ema:
            proj_type_at_check = "stalled_ema"; projected_final_loss_val = ema_curr_loss
    if req_iters_to_target == float('inf') and not proj_type_at_check.startswith("stalled_"):
        if use_ema and not proj_type_at_check.endswith("_ema"):
             log_layer_optimization_event(LogType.EMA_PROJECTION_INCONCLUSIVE_FALLBACK_RAW, layer_name, attempt=attempt_idx+1, rank=rank)
        if raw_rel_imprv_for_fallback > 1e-9 and current_raw_loss > target_loss:
            proj_type_at_check = "simple_raw_fallback"
            try:
                intervals_needed = math.log(target_loss / current_raw_loss) / math.log(1.0 - raw_rel_imprv_for_fallback)
                req_iters_to_target = math.ceil(intervals_needed) * prog_check_interval
                projected_final_loss_val = None
            except (ValueError, OverflowError, ZeroDivisionError):
                req_iters_to_target = float('inf')
                projected_final_loss_val = current_raw_loss * ((1.0 - max(0, raw_rel_imprv_for_fallback))**(num_intervals_remaining + 1))
        elif current_raw_loss <= target_loss:
            proj_type_at_check = "raw_target_met_fallback"; req_iters_to_target = 0; projected_final_loss_val = None
        elif raw_rel_imprv_for_fallback <= 1e-9 :
            proj_type_at_check = "stalled_raw_fallback"; projected_final_loss_val = current_raw_loss
    log_details = {
        'attempt': attempt_idx + 1, 'rank': rank, 'proj_type': proj_type_at_check,
        'iters_needed': req_iters_to_target, 'proj_final_loss': projected_final_loss_val,
        'target_loss': target_loss, 'avail_iters': iters_remaining_in_attempt
    }
    if req_iters_to_target > iters_remaining_in_attempt :
        log_event_type = LogType.PROJECTION_STOP if perform_stop_check else LogType.PROJECTION_LOG_ONLY
        log_layer_optimization_event(log_event_type, layer_name, **log_details)
        if perform_stop_check:
            return True, log_details
    return False, log_details if proj_type_at_check != "none" else None

def get_module_shape_info_from_weight(weight_tensor: torch.Tensor):
    if len(weight_tensor.shape) == 4: is_conv = True; out_dim, in_dim_effective, k_h, k_w = weight_tensor.shape; return out_dim, in_dim_effective, k_h, k_w, True
    elif len(weight_tensor.shape) == 2: is_conv = False; out_dim, in_dim = weight_tensor.shape; return out_dim, in_dim, None, None, False
    return None

def generate_intermediate_filename(base_save_path: str, num_total_completed_layers: int) -> str:
    base, ext = os.path.splitext(base_save_path)
    return f"{base}_resume_L{num_total_completed_layers}{ext}"

def prepare_save_metadata(
    script_args: argparse.Namespace,
    output_filename: str, 
    total_completed_modules: int,
    processed_this_run: int,
    skipped_identical_this_run: int,
    skipped_other_this_run: int,
    skipped_good_initial_this_run: int,
    scanned_keys_this_run: int,
    all_completed_module_prefixes_list: list[str],
    layer_opt_stats_this_run: list[dict],
    is_interrupted_save: bool
) -> tuple[dict, dict]:
    net_alpha_str = f"{script_args.initial_alpha:.8f}" if script_args.initial_alpha is not None else str(script_args.rank)
    conv_alpha_val = script_args.initial_conv_alpha if script_args.initial_conv_alpha is not None else (script_args.conv_rank or script_args.rank)
    conv_alpha_str = f"{conv_alpha_val:.8f}" if isinstance(conv_alpha_val, float) else str(conv_alpha_val)
    network_args_data = {
        "algo": "loha", "dim": str(script_args.rank), "alpha": net_alpha_str,
        "conv_dim": str(script_args.conv_rank or script_args.rank), "conv_alpha": conv_alpha_str,
        **{k: str(getattr(script_args, k)) for k in ["dropout", "rank_dropout", "module_dropout"]}
    }
    sf_meta = {
        "ss_network_module": "lycoris.kohya", "ss_network_rank": str(script_args.rank),
        "ss_network_alpha": net_alpha_str, "ss_network_algo": "loha",
        "ss_network_args": json.dumps(network_args_data),
        "ss_comment": f"Extracted LoHA (Int: {is_interrupted_save}). OptPrec: {script_args.precision}. SaveDtype: {script_args.save_weights_dtype}. Layers: {total_completed_modules}.",
        "ss_base_model_name": os.path.splitext(os.path.basename(script_args.base_model_path))[0],
        "ss_ft_model_name": os.path.splitext(os.path.basename(script_args.ft_model_path))[0],
        "ss_save_weights_dtype": script_args.save_weights_dtype,
        "ss_optimization_precision": script_args.precision,
        "ss_completed_loha_modules": json.dumps(all_completed_module_prefixes_list)
    }
    serializable_script_args = {
        k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
        for k, v in vars(script_args).items()
    }
    extraction_summary_data = {
        "total_cumulative": total_completed_modules, "this_session": processed_this_run,
        "skipped_identical": skipped_identical_this_run, "skipped_other": skipped_other_this_run,
        "skipped_good_initial": skipped_good_initial_this_run, "scanned_keys": scanned_keys_this_run
    }
    json_meta = {
        "comfyui_lora_type": "LyCORIS_LoHa", "model_name": os.path.splitext(output_filename)[0],
        "base_model_path": script_args.base_model_path, "ft_model_path": script_args.ft_model_path,
        "loha_extraction_settings": serializable_script_args,
        "extraction_summary": extraction_summary_data,
        "layer_optimization_details_this_session": layer_opt_stats_this_run,
        "embedded_safetensors_metadata": sf_meta, "interrupted_save": is_interrupted_save
    }
    return sf_meta, json_meta

def perform_graceful_save(output_path_to_save: str):
    global extracted_loha_state_dict_global, layer_optimization_stats_global, args_global, \
           processed_layers_this_session_count_global, save_attempted_on_interrupt, \
           skipped_identical_count_global, skipped_other_reason_count_global, keys_scanned_this_run_global, \
           all_completed_module_prefixes_ever_global, skipped_good_initial_loss_count_global

    total_processed_ever = len(all_completed_module_prefixes_ever_global)
    if not extracted_loha_state_dict_global and not total_processed_ever:
        print(f"No layers to save to {output_path_to_save}. Aborted.")
        return False
    if not args_global: 
        print("Error: Global args not set for saving metadata.")
        return False

    save_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args_global.save_weights_dtype, torch.bfloat16)
    final_sd = OrderedDict(
        (k, v.to(save_dtype) if hasattr(v, 'is_floating_point') and v.is_floating_point() else v)
        for k, v in extracted_loha_state_dict_global.items()
    )
    print(f"\nSaving LoHA for {total_processed_ever} modules ({processed_layers_this_session_count_global} this session) to {output_path_to_save}")
    sf_meta, json_meta = prepare_save_metadata(
        script_args=args_global, output_filename=os.path.basename(output_path_to_save),
        total_completed_modules=total_processed_ever,
        processed_this_run=processed_layers_this_session_count_global,
        skipped_identical_this_run=skipped_identical_count_global,
        skipped_other_this_run=skipped_other_reason_count_global,
        skipped_good_initial_this_run=skipped_good_initial_loss_count_global,
        scanned_keys_this_run=keys_scanned_this_run_global,
        all_completed_module_prefixes_list=list(all_completed_module_prefixes_ever_global),
        layer_opt_stats_this_run=layer_optimization_stats_global,
        is_interrupted_save=save_attempted_on_interrupt
    )
    temp_sf_path, temp_json_path = None, None
    try:
        if output_path_to_save.endswith(".safetensors"):
            temp_sf_path = output_path_to_save + ".part"
            final_json_path = os.path.splitext(output_path_to_save)[0] + "_extraction_metadata.json"
            temp_json_path = final_json_path + ".part"
            save_file(final_sd, temp_sf_path, metadata=sf_meta)
            with open(temp_json_path, 'w') as f: json.dump(json_meta, f, indent=4)
            os.replace(temp_sf_path, output_path_to_save)
            os.replace(temp_json_path, final_json_path)
            print(f"Saved: {output_path_to_save} and {final_json_path}")
        else:
            torch.save({'state_dict': final_sd, '__metadata__': sf_meta, '__extended_metadata__': json_meta}, output_path_to_save)
            print(f"Saved (basic .pt): {output_path_to_save}")
        return True
    except Exception as e:
        print(f"Error saving to {output_path_to_save}: {e}"); traceback.print_exc()
        if temp_sf_path and os.path.exists(temp_sf_path):
            try: os.remove(temp_sf_path)
            except OSError: pass
        if temp_json_path and os.path.exists(temp_json_path):
            try: os.remove(temp_json_path)
            except OSError: pass
        return False

def cleanup_intermediate_files(final_intended_path: str, for_resume_management: bool = False, keep_n: int = 0):
    output_dir = os.path.dirname(final_intended_path); base_name, save_ext = os.path.splitext(os.path.basename(final_intended_path))
    if not output_dir: output_dir = "."
    intermediate_pattern = os.path.join(output_dir, f"{base_name}_resume_L*{save_ext}")
    files_to_consider = [{'path': fp, 'l_count': int(m.group(1))} for fp in glob.glob(intermediate_pattern) if (m := re.search(r'_resume_L(\d+)', os.path.basename(fp)))]
    if not files_to_consider: return
    cleaned_count = 0
    if for_resume_management:
        if keep_n <= 0 or len(files_to_consider) <= keep_n: return
        files_to_consider.sort(key=lambda x: x['l_count'])
        files_to_delete = files_to_consider[:-keep_n]
        if args_global and args_global.verbose: print(f"  Resume Manager: Found {len(files_to_consider)} files. Deleting {len(files_to_delete)} oldest to keep {keep_n}.")
    else:
        files_to_delete = files_to_consider
        if args_global and args_global.verbose: print(f"  Cleaning ALL {len(files_to_delete)} intermediate files...")
    for file_info in files_to_delete:
        try:
            os.remove(file_info['path'])
            if args_global and args_global.verbose: print(f"    Cleaned: {file_info['path']}")
            cleaned_count += 1
            json_path = os.path.splitext(file_info['path'])[0] + "_extraction_metadata.json"
            if os.path.exists(json_path): os.remove(json_path)
        except OSError as e: print(f"    Warning: Could not clean {file_info['path']}: {e}")
    if cleaned_count > 0: print(f"  Cleaned {cleaned_count} file(s).")

def find_best_resume_file(intended_final_path: str) -> tuple[str | None, int]:
    output_dir = os.path.dirname(intended_final_path)
    if not output_dir: output_dir = "."
    base_save_name, save_ext = os.path.splitext(os.path.basename(intended_final_path))
    potential_files = []
    if os.path.exists(intended_final_path): potential_files.append(intended_final_path)
    intermediate_pattern = os.path.join(output_dir, f"{base_save_name}_resume_L*{save_ext}")
    potential_files.extend(glob.glob(intermediate_pattern))
    best_file_path = None; max_completed_modules = -1
    if not potential_files: return None, -1
    for file_path in sorted(potential_files):
        try:
            if not os.path.exists(file_path): continue
            with safetensors.safe_open(file_path, framework="pt", device="cpu") as f: metadata = f.metadata()
            if metadata and "ss_completed_loha_modules" in metadata:
                num_completed = len(json.loads(metadata["ss_completed_loha_modules"]))
                if num_completed > max_completed_modules: max_completed_modules, best_file_path = num_completed, file_path
                elif num_completed == max_completed_modules:
                    if file_path == intended_final_path or (base_save_name+"_resume_L" in os.path.basename(file_path) and best_file_path != intended_final_path):
                        best_file_path = file_path
            elif max_completed_modules == -1 and (best_file_path is None or (file_path == intended_final_path and best_file_path != intended_final_path)):
                best_file_path, max_completed_modules = file_path, 0
                if args_global and args_global.verbose: print(f"    File {os.path.basename(file_path)} no metadata. Treating as 0 completed.")
        except Exception as e:
            print(f"    Warning: Could not read metadata from {file_path}: {e}")
            if best_file_path is None and file_path == intended_final_path and max_completed_modules == -1 : best_file_path, max_completed_modules = file_path, 0
    if best_file_path: print(f"  Selected '{os.path.basename(best_file_path)}' for resume (est. {max_completed_modules} modules).")
    return best_file_path, max_completed_modules

def handle_resume_or_continue_loha(
    current_args: argparse.Namespace,
    extracted_sd_ref: OrderedDict,
    params_to_seed_opt_ref: dict,
    prev_completed_prefixes_ref: set,
    all_completed_prefixes_ref: set
):
    if current_args.continue_training_from_loha:
        print(f"\nMode: Continue/Refine from LoHA: {current_args.continue_training_from_loha}")
        if not os.path.exists(current_args.continue_training_from_loha):
            print(f"  Error: LoHA not found: {current_args.continue_training_from_loha}"); sys.exit(1)
        try:
            loaded_sd = load_file(current_args.continue_training_from_loha, device='cpu')
            extracted_sd_ref.update(loaded_sd) 
            module_prefixes = {".".join(k.split('.')[:-1]) for k in loaded_sd if ".hada_w1_a" in k}
            loaded_count = 0
            for prefix in module_prefixes:
                params = {p: loaded_sd.get(f"{prefix}.{p}") for p in ['hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b']}
                alpha_t = loaded_sd.get(f"{prefix}.alpha")
                if all(params.values()) and alpha_t is not None:
                    params_to_seed_opt_ref[prefix] = {'params': params, 'rank': params['hada_w1_a'].shape[1], 'alpha': alpha_t.item()}
                    loaded_count += 1
                elif current_args.verbose:
                    tqdm.write(f"  Info: Module {prefix} from LoHA missing components. Will treat as new if encountered.")
            print(f"  Loaded {len(extracted_sd_ref)} tensors. Identified {loaded_count} full LoHA modules for re-optimization.")
            del loaded_sd
            if os.path.exists(current_args.save_to) and not current_args.overwrite:
                print(f"  Warning: Output {current_args.save_to} exists and may be overwritten.")
            elif os.path.exists(current_args.save_to) and current_args.overwrite:
                print(f"  Info: Output {current_args.save_to} will be overwritten due to --overwrite.")
        except Exception as e:
            print(f"  Error loading LoHA for continuation: {e}."); traceback.print_exc(); sys.exit(1)
    elif not current_args.overwrite:
        print(f"\nMode: Standard extraction. Checking resume states for: {current_args.save_to}")
        resume_file, num_modules_resume = find_best_resume_file(current_args.save_to) 
        if resume_file:
            print(f"  Attempting resume from: {resume_file} (est. {num_modules_resume} modules).")
            try:
                completed_in_file = set()
                with safetensors.safe_open(resume_file, framework="pt", device="cpu") as f:
                    meta = f.metadata()
                    if meta and "ss_completed_loha_modules" in meta:
                        completed_in_file = set(json.loads(meta["ss_completed_loha_modules"]))
                loaded_sd_resume = load_file(resume_file, device='cpu')
                if not completed_in_file and loaded_sd_resume: 
                    completed_in_file = {".".join(k.split('.')[:-1]) for k in loaded_sd_resume if k.endswith(".hada_w1_a")}
                res_tensor_count = 0
                if completed_in_file:
                    for k, v in loaded_sd_resume.items():
                        module_prefix_of_key = ".".join(k.split('.')[:-1])
                        if module_prefix_of_key in completed_in_file or k.endswith(".bias"): 
                            extracted_sd_ref[k] = v
                            res_tensor_count += 1
                    prev_completed_prefixes_ref.update(completed_in_file)
                    all_completed_prefixes_ref.update(completed_in_file) 
                    print(f"  Loaded {len(prev_completed_prefixes_ref)} module prefixes, {res_tensor_count} tensors for resume.")
                elif loaded_sd_resume: 
                    extracted_sd_ref.update(loaded_sd_resume)
                    inferred_completed = {".".join(k.split('.')[:-1]) for k in loaded_sd_resume if k.endswith(".hada_w1_a")}
                    prev_completed_prefixes_ref.update(inferred_completed)
                    all_completed_prefixes_ref.update(inferred_completed)
                    print(f"  Loaded all {len(loaded_sd_resume)} tensors from resume file (metadata for completed modules missing/empty, inferred {len(inferred_completed)}).")
                del loaded_sd_resume
            except Exception as e:
                print(f"  Error loading resume file '{resume_file}': {e}. Starting fresh.")
                extracted_sd_ref.clear()
                prev_completed_prefixes_ref.clear()
                all_completed_prefixes_ref.clear()
        else:
            print("  No suitable existing LoHA to resume from. Starting fresh.")
    elif current_args.overwrite:
        print(f"\nMode: Standard extraction with --overwrite. Final output {current_args.save_to} will be overwritten.")

def print_script_summary(
    layer_stats: list[dict],
    all_completed_prefixes: set,
    processed_session: int,
    skipped_identical_session: int,
    skipped_good_initial_session: int,
    skipped_other_session: int,
    skipped_vae_session: int,
    keys_scanned_session: int,
    total_candidates: int
):
    print("\n--- Final Optimization Summary (This Session) ---")
    if not layer_stats:
        print("  No layers were optimized in this session.")
    for stat in layer_stats:
        rank_info = f"InitialR: {stat['initial_rank_attempted']}, FinalR: {stat['final_rank_used']}"
        if stat.get('rank_was_increased', False): rank_info += " (Increased)"
        proj_loss_info = ""
        if stat.get('final_projected_loss_on_stop') is not None:
            proj_loss_info = f" (Proj.FinalLoss ~{stat['final_projected_loss_on_stop']:.2e})"
        stop_info = ""
        if stat.get('skipped_reopt_due_to_initial_good_loss'): stop_info = ", SkipReOpt:GoodInitialLoss"
        elif stat.get('stopped_early_by_loss_target'): stop_info = ", Stop:LossTarget"
        elif stat.get('stopped_by_projection', False):
            stop_info = f", Stop:Proj({stat.get('projection_type_used','?')})" + proj_loss_info
        elif stat.get('stopped_by_insufficient_progress', False):
            stop_info = ", Stop:RawProg"
        alpha_str = f", Alpha: {stat.get('alpha_final', 'N/A'):.2f}" if 'alpha_final' in stat else ""
        print(f"Layer: {stat['name']}, {rank_info}{alpha_str}, Loss: {stat['final_loss']:.4e}, Iters: {stat['iterations_done']}{stop_info}")

    print(f"\n--- Overall Summary ---")
    print(f"Total unique LoHA modules in final state (processed, skipped, errored): {len(all_completed_prefixes)}")
    print(f"  Processed (new/re-opt/skipped-good) this session: {processed_session}")
    print(f"  Skipped identical (this session): {skipped_identical_session}")
    print(f"  Skipped re-opt due to good initial loss (this session): {skipped_good_initial_session}")
    print(f"  Skipped other reasons (this session, VAE, opt error): {skipped_other_session} (incl. {skipped_vae_session} VAE)")
    print(f"  Total candidate keys scanned (this session): {keys_scanned_session}/{total_candidates}")

def setup_and_print_configuration(current_args: argparse.Namespace):
    if current_args.progress_check_start_iter is None:
        current_args.progress_check_start_iter = max(1, current_args.progress_check_interval) if current_args.progress_check_interval > 0 else current_args.max_iterations + 1
    elif current_args.progress_check_interval <= 0 :
         current_args.progress_check_start_iter = current_args.max_iterations + 1
    opt_dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    target_opt_dtype = opt_dtype_map.get(current_args.precision, torch.float32)
    final_save_dtype_torch = opt_dtype_map.get(current_args.save_weights_dtype, torch.bfloat16)
    print(f"Device: {current_args.device}, Opt Dtype: {target_opt_dtype}, Save Dtype: {final_save_dtype_torch}")
    if current_args.target_loss: print(f"Target Loss: {current_args.target_loss:.2e} (min iters: {current_args.min_iterations} for target check)")
    else: print(f"No Target Loss. Min iters for any early stop: {current_args.min_iterations}.")
    print(f"Max Iters/Layer: {current_args.max_iterations}, Max Rank Retries: {current_args.max_rank_retries}, Rank Incr Factor: {current_args.rank_increase_factor}")
    if current_args.save_every_n_layers > 0: print(f"Save every {current_args.save_every_n_layers} processed layers enabled.")
    if current_args.keep_n_resume_files > 0: print(f"Keeping the {current_args.keep_n_resume_files} most recent resume files.")
    if current_args.progress_check_interval > 0:
        first_eval_iter = current_args.progress_check_start_iter + current_args.progress_check_interval
        print(f"Progress Check: Enabled. Interval: {current_args.progress_check_interval} iters, Min Rel. Loss Decrease: {current_args.min_progress_loss_ratio:.1e}.")
        print(f"  Progress window starts at iter: {current_args.progress_check_start_iter}, first evaluation at iter: {first_eval_iter}.")
        if current_args.target_loss is not None:
             print(f"  Projection Check: Enabled (if target loss specified). Decay Caps: min={getattr(current_args, 'advanced_projection_decay_cap_min', 'N/A')}, max={getattr(current_args, 'advanced_projection_decay_cap_max', 'N/A')}")
    else: print("Progress Check: Disabled (and Projection Check disabled).")
    return current_args

def load_models(base_model_path: str, ft_model_path: str) -> tuple[OrderedDict, OrderedDict]:
    print(f"\nLoading base model: {base_model_path}")
    try:
        base_sd_raw = load_file(base_model_path, device='cpu') if base_model_path.endswith(".safetensors") else torch.load(base_model_path, map_location='cpu')
        base_model_sd = base_sd_raw.get('state_dict', base_sd_raw) if not isinstance(base_sd_raw, OrderedDict) and hasattr(base_sd_raw, 'get') else base_sd_raw
    except Exception as e:
        print(f"Error loading base model: {e}"); traceback.print_exc(); sys.exit(1)
    print(f"Loading fine-tuned model: {ft_model_path}")
    try:
        ft_sd_raw = load_file(ft_model_path, device='cpu') if ft_model_path.endswith(".safetensors") else torch.load(ft_model_path, map_location='cpu')
        ft_model_sd = ft_sd_raw.get('state_dict', ft_sd_raw) if not isinstance(ft_sd_raw, OrderedDict) and hasattr(ft_sd_raw, 'get') else ft_sd_raw
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}"); traceback.print_exc(); sys.exit(1)
    return base_model_sd, ft_model_sd

def optimize_loha_for_layer(
    layer_name: str, delta_W_target: torch.Tensor, out_dim: int, in_dim_effective: int,
    k_h: int, k_w: int, initial_rank_for_layer: int, initial_alpha_for_layer: float,
    lr: float = 1e-3, max_iterations: int = 1000, min_iterations: int = 100,
    target_loss: float = None, weight_decay: float = 1e-4,
    device: str = 'cuda', dtype: torch.dtype = torch.float32,
    is_conv: bool = True, verbose_layer_debug: bool = False,
    max_rank_retries: int = 0,
    rank_increase_factor: float = 1.25,
    existing_loha_layer_parameters: dict | None = None
):
    delta_W_target = delta_W_target.to(device, dtype=dtype)
    is_initial_call_with_existing_params = existing_loha_layer_parameters is not None
    best_result_so_far = {
        'final_loss': float('inf'), 'stopped_early_by_loss': False, 'stopped_by_insufficient_progress': False,
        'stopped_by_projection': False, 'projection_type_used': 'none', 'iterations_done': 0,
        'final_rank_used': initial_rank_for_layer, 'interrupted_mid_layer': False, 'final_projected_loss_on_stop': None
    }
    current_rank_for_this_attempt = initial_rank_for_layer
    alpha_init_for_this_attempt = initial_alpha_for_layer
    rank_base_for_next_increase = initial_rank_for_layer

    prog_check_interval_val = args_global.progress_check_interval
    min_prog_ratio_val = args_global.min_progress_loss_ratio
    iter_to_begin_first_progress_window = args_global.progress_check_start_iter
    adv_proj_decay_cap_min_val = getattr(args_global, 'advanced_projection_decay_cap_min', 0.5)
    adv_proj_decay_cap_max_val = getattr(args_global, 'advanced_projection_decay_cap_max', 1.05)
    proj_sample_interval_val = getattr(args_global, 'projection_sample_interval', 20)
    proj_ema_alpha_val = getattr(args_global, 'projection_ema_alpha', 0.1)
    proj_min_ema_hist_val = getattr(args_global, 'projection_min_ema_history', 5)

    for attempt_idx in range(max_rank_retries + 1):
        is_last_rank_attempt = (attempt_idx == max_rank_retries)
        if save_attempted_on_interrupt:
            return {**best_result_so_far, 'interrupted_mid_layer': True, 'iterations_done': 0 if attempt_idx == 0 else best_result_so_far['iterations_done'], 'projection_type_used': 'interrupted'}

        current_warm_start_status = None
        params_for_initialization = None
        prev_rank_for_warm_start_log = None
        if attempt_idx > 0:
            log_layer_optimization_event(LogType.RANK_RETRY_STARTING, layer_name, prev_rank=rank_base_for_next_increase, prev_best_loss=best_result_so_far.get('final_loss', float('inf')))
            new_rank_float = rank_base_for_next_increase * rank_increase_factor
            increased_rank = math.ceil(new_rank_float)
            current_rank_for_this_attempt = max(rank_base_for_next_increase + 1, increased_rank)
            original_alpha_to_rank_ratio = initial_alpha_for_layer / float(initial_rank_for_layer) if initial_rank_for_layer > 0 else 1.0
            alpha_init_for_this_attempt = original_alpha_to_rank_ratio * float(current_rank_for_this_attempt)
            prev_rank_for_warm_start_log = best_result_so_far.get('final_rank_used')
            if 'hada_w1_a' in best_result_so_far and not args_global.no_warm_start:
                if prev_rank_for_warm_start_log < current_rank_for_this_attempt: current_warm_start_status = 'applied'; params_for_initialization = best_result_so_far
                else: current_warm_start_status = 'skipped_cannot_warm_start'
            elif args_global.no_warm_start and 'hada_w1_a' in best_result_so_far: current_warm_start_status = 'skipped_no_warm_start_arg'
            elif 'hada_w1_a' not in best_result_so_far: current_warm_start_status = 'no_prior_params_for_warm_start'
            log_layer_optimization_event(LogType.RANK_INCREASED_INFO, layer_name, new_rank=current_rank_for_this_attempt, new_alpha=alpha_init_for_this_attempt, warm_start_status=current_warm_start_status, prev_rank_for_warm_start=prev_rank_for_warm_start_log)
        elif is_initial_call_with_existing_params:
            params_for_initialization = existing_loha_layer_parameters

        k_ops = k_h * k_w if is_conv else 1
        hada_w1_a_p, hada_w1_b_p, hada_w2_a_p, hada_w2_b_p = initialize_loha_parameters(
            out_dim, current_rank_for_this_attempt, (in_dim_effective * k_ops), device, dtype, layer_name, attempt_idx,
            (attempt_idx == 0 and is_initial_call_with_existing_params),
            params_for_initialization, current_warm_start_status,
            prev_rank_for_warm_start_log if current_warm_start_status == 'applied' else None
        )
        alpha_param = nn.Parameter(torch.tensor(alpha_init_for_this_attempt, device=device, dtype=dtype))
        params_to_optimize = [hada_w1_a_p, hada_w1_b_p, hada_w2_a_p, hada_w2_b_p, alpha_param]
        optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=max(10, int(max_iterations * 0.05)), factor=0.5, min_lr=max(1e-7, lr * 0.001))
        iter_pbar_desc = f"Opt Att {attempt_idx+1}/{max_rank_retries+1} (R:{current_rank_for_this_attempt}){' [LastRank]' if is_last_rank_attempt else ''}: {layer_name}"
        iter_pbar = tqdm(range(max_iterations), desc=iter_pbar_desc, leave=False, dynamic_ncols=True, position=1, mininterval=0.5)
        
        current_attempt_final_loss = float('inf'); current_attempt_stopped_early_by_loss = False
        current_attempt_insufficient_progress = False; current_attempt_stopped_by_projection = False
        current_attempt_projection_type = "none"; current_attempt_iterations_done = 0
        loss_at_start_of_current_window = float('inf'); progress_window_started_for_attempt = False
        relative_improvement_history_for_proj = []
        final_projected_loss_if_failed = None 
        ema_loss_history = [] 
        current_ema_loss_value = None 

        for i in iter_pbar:
            current_attempt_iterations_done = i + 1
            if save_attempted_on_interrupt:
                iter_pbar.close()
                best_result_so_far.update({'interrupted_mid_layer': True, 'projection_type_used': 'interrupted', 'iterations_done': i})
                return best_result_so_far

            if prog_check_interval_val > 0 and not progress_window_started_for_attempt and current_attempt_iterations_done >= iter_to_begin_first_progress_window:
                 loss_at_start_of_current_window = current_attempt_final_loss 
                 progress_window_started_for_attempt = True

            optimizer.zero_grad()
            eff_alpha_scale = alpha_param / current_rank_for_this_attempt
            term1_flat = hada_w1_a_p @ hada_w1_b_p; term2_flat = hada_w2_a_p @ hada_w2_b_p
            delta_W_loha = eff_alpha_scale * (term1_flat.view(out_dim, in_dim_effective, k_h, k_w) * term2_flat.view(out_dim, in_dim_effective, k_h, k_w)) if is_conv else (eff_alpha_scale * term1_flat * term2_flat)
            
            loss = F.mse_loss(delta_W_loha, delta_W_target)
            raw_current_loss_item = loss.item()
            if i == 0 and progress_window_started_for_attempt and loss_at_start_of_current_window == float('inf'): 
                loss_at_start_of_current_window = raw_current_loss_item
            current_attempt_final_loss = raw_current_loss_item
            loss.backward(); optimizer.step(); scheduler.step(raw_current_loss_item) 

            if target_loss is not None and prog_check_interval_val > 0 and (i + 1) % proj_sample_interval_val == 0:
                current_ema_loss_value = proj_ema_alpha_val * raw_current_loss_item + (1 - proj_ema_alpha_val) * current_ema_loss_value if current_ema_loss_value is not None else raw_current_loss_item
                ema_loss_history.append((current_attempt_iterations_done, current_ema_loss_value))

            iter_pbar.set_postfix_str(f"Loss={current_attempt_final_loss:.3e}, AlphaP={alpha_param.item():.2f}, LR={optimizer.param_groups[0]['lr']:.1e}", refresh=True)

            if target_loss is not None and current_attempt_iterations_done >= min_iterations and current_attempt_final_loss <= target_loss:
                log_layer_optimization_event(LogType.TARGET_LOSS_REACHED_IN_ATTEMPT, layer_name, attempt=attempt_idx+1, rank=current_rank_for_this_attempt, target_loss=target_loss, iter=current_attempt_iterations_done)
                current_attempt_stopped_early_by_loss = True; break

            if prog_check_interval_val > 0 and progress_window_started_for_attempt and \
               (current_attempt_iterations_done >= iter_to_begin_first_progress_window + prog_check_interval_val) and \
               (((current_attempt_iterations_done - iter_to_begin_first_progress_window) % prog_check_interval_val) == 0):
                
                perform_early_stop_checks = not is_last_rank_attempt
                
                stop_insufficient_prog, _ = check_insufficient_progress(
                    current_attempt_final_loss, loss_at_start_of_current_window, min_prog_ratio_val, target_loss,
                    perform_early_stop_checks, layer_name, attempt_idx, current_rank_for_this_attempt
                )
                if stop_insufficient_prog:
                    current_attempt_insufficient_progress = True; break
                
                if target_loss is not None and current_attempt_final_loss > target_loss: 
                    raw_rel_imprv_for_fallback_proj = (loss_at_start_of_current_window - current_attempt_final_loss) / loss_at_start_of_current_window if loss_at_start_of_current_window > 1e-12 and loss_at_start_of_current_window > current_attempt_final_loss else 0.0
                    stop_projection, proj_details = check_loss_projection(
                        ema_loss_history, current_attempt_final_loss, raw_rel_imprv_for_fallback_proj, target_loss,
                        max_iterations, current_attempt_iterations_done, prog_check_interval_val,
                        proj_min_ema_hist_val, adv_proj_decay_cap_min_val, adv_proj_decay_cap_max_val,
                        perform_early_stop_checks, layer_name, attempt_idx, current_rank_for_this_attempt,
                        relative_improvement_history_for_proj 
                    )
                    if proj_details: 
                        current_attempt_projection_type = proj_details.get('proj_type', 'none')
                        if stop_projection : final_projected_loss_if_failed = proj_details.get('proj_final_loss')
                        
                    if stop_projection:
                        current_attempt_stopped_by_projection = True; break
                
                loss_at_start_of_current_window = current_attempt_final_loss 
        iter_pbar.close()
        
        if current_attempt_final_loss < best_result_so_far['final_loss'] or \
           (current_attempt_final_loss == best_result_so_far['final_loss'] and current_rank_for_this_attempt < best_result_so_far['final_rank_used']):
            log_layer_optimization_event(LogType.NEW_BEST_RESULT_FOR_LAYER, layer_name, attempt=attempt_idx+1, rank=current_rank_for_this_attempt, loss=current_attempt_final_loss)
            best_result_so_far.update({
               'hada_w1_a': hada_w1_a_p.data.cpu().contiguous(), 'hada_w1_b': hada_w1_b_p.data.cpu().contiguous(),
               'hada_w2_a': hada_w2_a_p.data.cpu().contiguous(), 'hada_w2_b': hada_w2_b_p.data.cpu().contiguous(),
               'alpha': alpha_param.data.cpu().contiguous(), 'final_loss': current_attempt_final_loss,
               'stopped_early_by_loss': current_attempt_stopped_early_by_loss,
               'stopped_by_insufficient_progress': current_attempt_insufficient_progress,
               'stopped_by_projection': current_attempt_stopped_by_projection,
               'projection_type_used': current_attempt_projection_type, 
               'iterations_done': current_attempt_iterations_done, 'final_rank_used': current_rank_for_this_attempt,
               'interrupted_mid_layer': False, 
               'final_projected_loss_on_stop': final_projected_loss_if_failed if current_attempt_stopped_by_projection else None
            })
        rank_base_for_next_increase = current_rank_for_this_attempt

        if current_attempt_stopped_early_by_loss:
            log_layer_optimization_event(LogType.TARGET_LOSS_MET_STOP_ALL_RETRIES, layer_name)
            break 

        if current_attempt_iterations_done < max_iterations and not any([current_attempt_insufficient_progress, current_attempt_stopped_by_projection, current_attempt_stopped_early_by_loss]):
            if not is_last_rank_attempt:
                log_layer_optimization_event(LogType.ATTEMPT_EARLY_FINISH_NO_STOP_FLAG, layer_name, attempt=attempt_idx+1, rank=current_rank_for_this_attempt, iters_done=current_attempt_iterations_done, max_iters=max_iterations)
        
        if is_last_rank_attempt:
             if not (current_attempt_stopped_early_by_loss or current_attempt_insufficient_progress or current_attempt_stopped_by_projection):
                _, prog_log_details = check_insufficient_progress(current_attempt_final_loss, loss_at_start_of_current_window, min_prog_ratio_val, target_loss, False, layer_name, attempt_idx, current_rank_for_this_attempt)
                if target_loss and current_attempt_final_loss > target_loss and prog_check_interval_val > 0: 
                    raw_rel_imprv_for_fallback_proj_last = (loss_at_start_of_current_window - current_attempt_final_loss) / loss_at_start_of_current_window if loss_at_start_of_current_window > 1e-12 and loss_at_start_of_current_window > current_attempt_final_loss else 0.0
                    _, proj_log_details_last = check_loss_projection(
                        ema_loss_history, current_attempt_final_loss, raw_rel_imprv_for_fallback_proj_last, target_loss,
                        max_iterations, current_attempt_iterations_done, prog_check_interval_val,
                        proj_min_ema_hist_val, adv_proj_decay_cap_min_val, adv_proj_decay_cap_max_val,
                        False, layer_name, attempt_idx, current_rank_for_this_attempt,
                        relative_improvement_history_for_proj 
                    )
                    if proj_log_details_last: 
                        best_result_so_far['projection_type_used'] = proj_log_details_last.get('proj_type', best_result_so_far['projection_type_used'])
                        best_result_so_far['final_projected_loss_on_stop'] = proj_log_details_last.get('proj_final_loss', best_result_so_far['final_projected_loss_on_stop'])
             log_layer_optimization_event(LogType.LAST_RANK_ATTEMPT_SUMMARY, layer_name, target_loss=target_loss,
                                          final_loss_for_layer=best_result_so_far['final_loss'], final_rank_for_layer=best_result_so_far['final_rank_used'])
             break 

        if not current_attempt_stopped_early_by_loss : 
            reason_kwargs = {'attempt': attempt_idx + 1, 'rank': current_rank_for_this_attempt, 'is_last_rank_attempt': is_last_rank_attempt}
            if current_attempt_stopped_by_projection:
                reason_kwargs.update({'reason_type': 'projection_unreachable', 'target_loss': target_loss, 'proj_final_loss': final_projected_loss_if_failed, 'proj_type': current_attempt_projection_type})
            elif current_attempt_insufficient_progress:
                reason_kwargs.update({'reason_type': 'insufficient_progress'})
            elif current_attempt_iterations_done >= max_iterations:
                reason_kwargs.update({'reason_type': 'max_iterations_no_target' if target_loss else 'max_iterations_no_target_set', 'current_loss': current_attempt_final_loss})
            
            if 'reason_type' in reason_kwargs : 
                log_layer_optimization_event(LogType.ATTEMPT_ENDED_WILL_RETRY, layer_name, **reason_kwargs)
    
    if 'hada_w1_a' not in best_result_so_far:
        log_layer_optimization_event(LogType.NO_VALID_OPTIMIZATION_RESULT, layer_name)
        return {'final_loss': float('inf'), 'interrupted_mid_layer': True, 'final_rank_used': initial_rank_for_layer, 'iterations_done':0}

    for key, default_val in [('stopped_early_by_loss', False), ('stopped_by_insufficient_progress', False), 
                             ('stopped_by_projection', False), ('projection_type_used', 'none'), 
                             ('interrupted_mid_layer', False), ('final_projected_loss_on_stop', None),
                             ('final_rank_used', initial_rank_for_layer)]:
        best_result_so_far.setdefault(key, default_val)
    return best_result_so_far


def handle_interrupt(signum, frame):
    # ... (remains the same) ...
    global save_attempted_on_interrupt, outer_pbar_global, args_global, all_completed_module_prefixes_ever_global
    print("\n" + "="*30 + "\nCtrl+C Detected!\n" + "="*30)
    if save_attempted_on_interrupt: print("Save already attempted. Exiting."); return
    save_attempted_on_interrupt = True
    if outer_pbar_global: outer_pbar_global.close()
    if args_global and args_global.save_to:
        save_path = generate_intermediate_filename(args_global.save_to, len(all_completed_module_prefixes_ever_global))
        print(f"Attempting interrupt save to: {save_path}")
        if perform_graceful_save(save_path) and args_global.keep_n_resume_files > 0:
            cleanup_intermediate_files(args_global.save_to, True, args_global.keep_n_resume_files)
    else: print("Cannot perform interrupt save: args not defined.")
    print("Exiting.")
    sys.exit(0)


def main(cli_args):
    global args_global, extracted_loha_state_dict_global, layer_optimization_stats_global, \
           processed_layers_this_session_count_global, save_attempted_on_interrupt, outer_pbar_global, \
           skipped_identical_count_global, skipped_other_reason_count_global, keys_scanned_this_run_global, \
           previously_completed_module_prefixes_global, all_completed_module_prefixes_ever_global, \
           main_loop_completed_scan_flag_global, params_to_seed_optimizer_global, skipped_good_initial_loss_count_global, \
           skipped_vae_layers_count 

    args_global = cli_args
    signal.signal(signal.SIGINT, handle_interrupt) 
    extracted_loha_state_dict_global.clear(); layer_optimization_stats_global.clear()
    params_to_seed_optimizer_global.clear(); previously_completed_module_prefixes_global.clear()
    all_completed_module_prefixes_ever_global.clear()
    processed_layers_this_session_count_global = skipped_identical_count_global = skipped_other_reason_count_global = 0
    skipped_good_initial_loss_count_global = keys_scanned_this_run_global = skipped_vae_layers_count = 0 
    main_loop_completed_scan_flag_global = False; save_attempted_on_interrupt = False

    args_global = setup_and_print_configuration(args_global)
    target_opt_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(args_global.precision, torch.float32)
    final_save_dtype_torch = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(args_global.save_weights_dtype, torch.bfloat16)
    handle_resume_or_continue_loha(
        args_global, extracted_loha_state_dict_global, params_to_seed_optimizer_global,
        previously_completed_module_prefixes_global, all_completed_module_prefixes_ever_global
    )
    base_model_sd, ft_model_sd = load_models(args_global.base_model_path, args_global.ft_model_path)
    all_candidate_keys = sorted([k for k in base_model_sd if k.endswith('.weight') and k in ft_model_sd and base_model_sd[k].shape == ft_model_sd[k].shape and (len(base_model_sd[k].shape) in [2,4])])
    total_candidates_to_scan = len(all_candidate_keys) 
    print(f"Found {total_candidates_to_scan} candidate '.weight' keys for LoHA extraction.")
    outer_pbar_global = tqdm(total=total_candidates_to_scan, desc="Scanning Layers", dynamic_ncols=True, position=0)
    
    try:
        for key_name in all_candidate_keys:
            if save_attempted_on_interrupt: break
            keys_scanned_this_run_global += 1; outer_pbar_global.update(1)
            original_module_path = key_name[:-len(".weight")]
            loha_key_prefix = "lora_" + original_module_path.replace(".", "_")
            if "model.diffusion_model." in original_module_path: loha_key_prefix = "lora_unet_" + original_module_path.split("model.diffusion_model.")[-1].replace(".", "_")
            elif "first_stage_model." in original_module_path: loha_key_prefix = "lora_vae_" + original_module_path.split("first_stage_model.")[-1].replace(".", "_")
            if any(vp in original_module_path for vp in [".encoder.", ".decoder.", ".quant_conv."]) and any(tp in original_module_path for tp in ["first_stage_model.", "autoencoder."]):
                if args_global.verbose_layer_debug: tqdm.write(f"  Skipping VAE layer: {original_module_path}")
                skipped_vae_layers_count += 1; skipped_other_reason_count_global += 1
                all_completed_module_prefixes_ever_global.add(loha_key_prefix)
                continue
            is_reopt_target = args_global.continue_training_from_loha and loha_key_prefix in params_to_seed_optimizer_global
            if loha_key_prefix in all_completed_module_prefixes_ever_global and not is_reopt_target:
                if args_global.verbose_layer_debug: tqdm.write(f"  Skipping {loha_key_prefix} (already processed/resumed, not re-opt target).")
                continue
            if args_global.max_layers is not None and args_global.max_layers > 0 and processed_layers_this_session_count_global >= args_global.max_layers:
                if args_global.verbose and processed_layers_this_session_count_global == args_global.max_layers:
                    tqdm.write(f"\nMax_layers ({args_global.max_layers}) for new/re-optimized hit. Scan continues to find all identical/skipped layers.")
                outer_pbar_global.set_description_str(f"Scan {keys_scanned_this_run_global}/{total_candidates_to_scan} (Max Layers Reached)")
                continue
            base_W = base_model_sd[key_name].to(dtype=torch.float32)
            ft_W = ft_model_sd[key_name].to(dtype=torch.float32)
            out_dim, in_dim_effective, k_h, k_w, is_conv = get_module_shape_info_from_weight(base_W)
            delta_W_fp32 = (ft_W - base_W)
            if torch.allclose(delta_W_fp32, torch.zeros_like(delta_W_fp32), atol=args_global.atol_fp32_check):
                if args_global.verbose_layer_debug: tqdm.write(f"  Skipping {loha_key_prefix} (weights identical atol={args_global.atol_fp32_check:.1e}).")
                skipped_identical_count_global += 1
                all_completed_module_prefixes_ever_global.add(loha_key_prefix)
                continue
            current_key_processed_or_skipped_good = False 
            should_skip_due_to_pre_existing_good_loss = False
            if is_reopt_target and args_global.target_loss is not None:
                seed_data = params_to_seed_optimizer_global[loha_key_prefix]
                loaded_params_cpu = seed_data['params']; loaded_rank_check = seed_data['rank']; loaded_alpha_check = seed_data['alpha']
                if all(k_ in loaded_params_cpu for k_ in ['hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b']):
                    try:
                        with torch.no_grad():
                            w1a,w1b,w2a,w2b = (loaded_params_cpu[p].to(args_global.device, target_opt_dtype) for p in ['hada_w1_a','hada_w1_b','hada_w2_a','hada_w2_b'])
                            alpha_v = torch.tensor(loaded_alpha_check, device=args_global.device, dtype=target_opt_dtype)
                            eff_a_s = alpha_v / loaded_rank_check; delta_W_target_c = delta_W_fp32.to(args_global.device, target_opt_dtype)
                            init_loha_d = eff_a_s * (w1a@w1b).view(out_dim,in_dim_effective,k_h,k_w) * (w2a@w2b).view(out_dim,in_dim_effective,k_h,k_w) if is_conv else eff_a_s * (w1a@w1b) * (w2a@w2b)
                            init_loss_c = F.mse_loss(init_loha_d, delta_W_target_c).item()
                            if init_loss_c <= args_global.target_loss:
                                tqdm.write(f"  Skip Re-Opt {loha_key_prefix}: Loaded (R:{loaded_rank_check}, A:{loaded_alpha_check:.2f}) meets target. Loss: {init_loss_c:.4e} <= {args_global.target_loss:.4e}")
                                stat_entry_skip = {"name": str(loha_key_prefix), "original_name": str(original_module_path),"initial_rank_attempted": int(loaded_rank_check), "final_rank_used": int(loaded_rank_check),"rank_was_increased": False, "final_loss": float(init_loss_c),"alpha_final": float(loaded_alpha_check), "iterations_done": 0,"stopped_early_by_loss_target": True, "stopped_by_insufficient_progress": False,"stopped_by_projection": False, "projection_type_used": "none","final_projected_loss_on_stop": None,"skipped_reopt_due_to_initial_good_loss": True, "interrupted_mid_layer": False}
                                layer_optimization_stats_global.append(stat_entry_skip)
                                all_completed_module_prefixes_ever_global.add(loha_key_prefix)
                                processed_layers_this_session_count_global += 1; skipped_good_initial_loss_count_global += 1
                                should_skip_due_to_pre_existing_good_loss = True; current_key_processed_or_skipped_good = True
                                outer_pbar_global.set_description_str(f"Scan {keys_scanned_this_run_global}/{total_candidates_to_scan} (New/ReOpt: {processed_layers_this_session_count_global - skipped_good_initial_loss_count_global}, SkipGood:{skipped_good_initial_loss_count_global})")
                            elif args_global.verbose_layer_debug: tqdm.write(f"    Initial loss for loaded {loha_key_prefix}: {init_loss_c:.4e}. Re-optimizing.")
                    except Exception as e_c: tqdm.write(f"    Warn: Pre-opt loss check failed for {loha_key_prefix}: {e_c}. Optimizing.");
            if not should_skip_due_to_pre_existing_good_loss:
                current_op_mode_str = "ReOpt" if is_reopt_target else "NewOpt"
                if args_global.verbose: tqdm.write(f"\n--- {current_op_mode_str} Layer {processed_layers_this_session_count_global + 1 - skipped_good_initial_loss_count_global}: {loha_key_prefix} (Orig: {original_module_path}) ---")
                initial_rank_opt = args_global.conv_rank if is_conv and args_global.conv_rank is not None else args_global.rank
                initial_alpha_opt = args_global.initial_conv_alpha if is_conv else args_global.initial_alpha
                existing_params_init = None; max_retries_layer = args_global.max_rank_retries
                if is_reopt_target:
                    seed_data = params_to_seed_optimizer_global[loha_key_prefix]
                    initial_rank_opt, initial_alpha_opt = seed_data['rank'], seed_data['alpha']
                    existing_params_init = seed_data['params']
                    base_rank_est = args_global.conv_rank if is_conv and args_global.conv_rank is not None else args_global.rank
                    if initial_rank_opt > base_rank_est and args_global.max_rank_retries > 0 :
                        est_retries_used = 0; cur_sim_rank = float(base_rank_est)
                        for _ in range(args_global.max_rank_retries + 10):
                            if cur_sim_rank >= initial_rank_opt: break
                            cur_sim_rank = max(math.ceil(cur_sim_rank * args_global.rank_increase_factor), cur_sim_rank + 1); est_retries_used += 1
                        max_retries_layer = max(0, args_global.max_rank_retries - est_retries_used)
                        if args_global.verbose: tqdm.write(f"    Using loaded R:{initial_rank_opt}, A:{initial_alpha_opt:.1f}. Max further retries for layer: {max_retries_layer}.")
                outer_pbar_global.set_description_str(f"{current_op_mode_str} L{processed_layers_this_session_count_global + 1 - skipped_good_initial_loss_count_global} (Scan {keys_scanned_this_run_global}/{total_candidates_to_scan}, SkipGood:{skipped_good_initial_loss_count_global})")
                opt_results = optimize_loha_for_layer(loha_key_prefix, delta_W_fp32, out_dim, in_dim_effective, k_h, k_w, initial_rank_opt, initial_alpha_opt, args_global.lr, args_global.max_iterations, args_global.min_iterations, args_global.target_loss, args_global.weight_decay, args_global.device, target_opt_dtype, is_conv, args_global.verbose_layer_debug, max_retries_layer, args_global.rank_increase_factor, existing_params_init)
                if not opt_results.get('interrupted_mid_layer') and 'hada_w1_a' in opt_results :
                    for p_name, p_val in opt_results.items():
                        if p_name not in ['final_loss', 'stopped_early_by_loss', 'stopped_by_insufficient_progress', 'stopped_by_projection', 'projection_type_used', 'iterations_done', 'final_rank_used', 'interrupted_mid_layer', 'final_projected_loss_on_stop']:
                            if torch.is_tensor(p_val): extracted_loha_state_dict_global[f'{loha_key_prefix}.{p_name}'] = p_val.to(final_save_dtype_torch)
                    final_rank_used = opt_results['final_rank_used']
                    stat_entry = {"name": str(loha_key_prefix),"original_name": str(original_module_path),"initial_rank_attempted": int(initial_rank_opt),"final_rank_used": int(final_rank_used),"rank_was_increased": bool(final_rank_used > initial_rank_opt),"final_loss": float(opt_results['final_loss']),"alpha_final": float(opt_results['alpha'].item()) if isinstance(opt_results.get('alpha'), torch.Tensor) else float(opt_results.get('alpha', 0.0)),"iterations_done": int(opt_results['iterations_done']),"stopped_early_by_loss_target": bool(opt_results['stopped_early_by_loss']),"stopped_by_insufficient_progress": bool(opt_results.get('stopped_by_insufficient_progress', False)),"stopped_by_projection": bool(opt_results.get('stopped_by_projection', False)),"projection_type_used": str(opt_results.get('projection_type_used', 'none')),"final_projected_loss_on_stop": float(l_val) if (l_val := opt_results.get('final_projected_loss_on_stop')) is not None else None,"skipped_reopt_due_to_initial_good_loss": False,"interrupted_mid_layer": bool(opt_results.get('interrupted_mid_layer', False))}
                    layer_optimization_stats_global.append(stat_entry)
                    all_completed_module_prefixes_ever_global.add(loha_key_prefix)
                    stop_reason_short = ""
                    if opt_results['stopped_early_by_loss']: stop_reason_short = ", Stop:LossTarget"
                    elif opt_results.get('stopped_by_projection', False): stop_reason_short = f", Stop:Proj({opt_results.get('projection_type_used','?')})"
                    elif opt_results['stopped_by_insufficient_progress']: stop_reason_short = ", Stop:RawProg"
                    tqdm.write(f"  Layer {loha_key_prefix} Opt. Done. R_used: {final_rank_used}, FinalLoss: {opt_results['final_loss']:.4e}, Iters: {opt_results['iterations_done']}{stop_reason_short}")
                    if args_global.use_bias:
                        bias_key = f"{original_module_path}.bias"
                        if bias_key in ft_model_sd and (bias_key not in base_model_sd or not torch.allclose(base_model_sd[bias_key], ft_model_sd[bias_key], atol=args_global.atol_fp32_check)):
                            extracted_loha_state_dict_global[bias_key] = ft_model_sd[bias_key].cpu().to(final_save_dtype_torch)
                            if args_global.verbose: tqdm.write(f"    Saved differing/new bias for {bias_key}")
                    processed_layers_this_session_count_global += 1
                    current_key_processed_or_skipped_good = True 
                else:
                     tqdm.write(f"  Optimization for {loha_key_prefix} did not yield saveable results (Interrupt: {opt_results.get('interrupted_mid_layer', 'N/A')}, Loss: {opt_results.get('final_loss', 'N/A')})")
                     if not opt_results.get('interrupted_mid_layer', False) and 'hada_w1_a' not in opt_results :
                         skipped_other_reason_count_global += 1
                         all_completed_module_prefixes_ever_global.add(loha_key_prefix)
            if current_key_processed_or_skipped_good: 
                if args_global.save_every_n_layers > 0 and processed_layers_this_session_count_global > 0 and processed_layers_this_session_count_global % args_global.save_every_n_layers == 0 and keys_scanned_this_run_global < total_candidates_to_scan: 
                    periodic_save_path = generate_intermediate_filename(args_global.save_to, len(all_completed_module_prefixes_ever_global))
                    tqdm.write(f"\n--- Periodic Save: Processed {processed_layers_this_session_count_global} layers this session. Saving to {periodic_save_path} ---")
                    if perform_graceful_save(periodic_save_path) and args_global.keep_n_resume_files > 0:
                        cleanup_intermediate_files(args_global.save_to, True, args_global.keep_n_resume_files)
        if not save_attempted_on_interrupt and keys_scanned_this_run_global == total_candidates_to_scan:
            main_loop_completed_scan_flag_global = True
    finally:
        if outer_pbar_global: outer_pbar_global.close()

    if not save_attempted_on_interrupt:
        print_script_summary(
            layer_stats=layer_optimization_stats_global,
            all_completed_prefixes=all_completed_module_prefixes_ever_global,
            processed_session=processed_layers_this_session_count_global,
            skipped_identical_session=skipped_identical_count_global,
            skipped_good_initial_session=skipped_good_initial_loss_count_global,
            skipped_other_session=skipped_other_reason_count_global,
            skipped_vae_session=skipped_vae_layers_count, 
            keys_scanned_session=keys_scanned_this_run_global,
            total_candidates=total_candidates_to_scan 
        )
        
        is_fully_complete = main_loop_completed_scan_flag_global and len(all_completed_module_prefixes_ever_global) >= total_candidates_to_scan
        actual_save_path = args_global.save_to if is_fully_complete else generate_intermediate_filename(args_global.save_to, len(all_completed_module_prefixes_ever_global))
        reason_for_save_path = "Saving to final path (all candidates processed/skipped)" if is_fully_complete else \
                               ("Run incomplete (scan not finished or --max_layers hit)." if not main_loop_completed_scan_flag_global else "Full scan done, but not all layers processed/accounted for (e.g. new errors).")
        print(f"\n{reason_for_save_path}: {actual_save_path}")
        if perform_graceful_save(output_path_to_save=actual_save_path):
            if args_global.keep_n_resume_files > 0 and not is_fully_complete : 
                 cleanup_intermediate_files(args_global.save_to, True, args_global.keep_n_resume_files)
            if is_fully_complete and actual_save_path == args_global.save_to :
                print("\nCleaning up ALL intermediate resume files (from this script's previous runs)...")
                cleanup_intermediate_files(args_global.save_to, False)
    else: print("\nProcess interrupted. Graceful save to intermediate file attempted.")

def post_process_cli_args(parsed_args: argparse.Namespace) -> argparse.Namespace:
    if parsed_args.verbose_layer_debug:
        parsed_args.verbose = True
    if not os.path.exists(parsed_args.base_model_path):
        print(f"Error: Base model not found: {parsed_args.base_model_path}"); sys.exit(1)
    if not os.path.exists(parsed_args.ft_model_path):
        print(f"Error: FT model not found: {parsed_args.ft_model_path}"); sys.exit(1)
    save_dir = os.path.dirname(parsed_args.save_to)
    if save_dir and not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {save_dir}: {e}"); sys.exit(1)
    if parsed_args.initial_alpha is None:
        parsed_args.initial_alpha = float(parsed_args.rank)
    if parsed_args.initial_conv_alpha is None:
        if parsed_args.conv_rank is not None:
            parsed_args.initial_conv_alpha = float(parsed_args.conv_rank)
        else:
            parsed_args.initial_conv_alpha = parsed_args.initial_alpha
    return parsed_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LoHA parameters. Saves intermediate files like 'name_resume_L{count}.safetensors'.")
    parser.add_argument("base_model_path", type=str, help="Path to base model (.pt, .pth, .safetensors)")
    parser.add_argument("ft_model_path", type=str, help="Path to fine-tuned model (.pt, .pth, .safetensors)")
    parser.add_argument("save_to", type=str, help="Path for FINAL LoHA output (recommended .safetensors).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing FINAL LoHA. Does NOT clean intermediates until successful final save.")
    parser.add_argument("--continue_training_from_loha", type=str, default=None, help="Path to existing LoHA to load and continue optimizing.")
    parser.add_argument("--rank", type=int, default=4, help="Default rank for LoHA.")
    parser.add_argument("--conv_rank", type=int, default=None, help="Specific rank for Conv LoHA. Defaults to --rank.")
    parser.add_argument("--initial_alpha", type=float, default=None, help="Global initial alpha. Defaults to 'rank'.")
    parser.add_argument("--initial_conv_alpha", type=float, default=None, help="Specific initial alpha for Conv LoHA. Defaults to conv_rank or initial_alpha.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate per layer.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Max optimization iterations per layer/attempt.")
    parser.add_argument("--min_iterations", type=int, default=100, help="Min iterations before target_loss check per attempt.")
    parser.add_argument("--target_loss", type=float, default=None, help="Target MSE loss for early stopping. Also for pre-re-opt check.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimization.")
    parser.add_argument("--max_rank_retries", type=int, default=0, help="Rank increase retries if target_loss not met (0 for no retries).")
    parser.add_argument("--rank_increase_factor", type=float, default=1.25, help="Factor to increase rank on retry.")
    parser.add_argument("--progress_check_interval", type=int, default=100, help="Check loss improvement every N iterations (0 to disable).")
    parser.add_argument("--min_progress_loss_ratio", type=float, default=0.001, help="Min relative loss decrease over interval.")
    parser.add_argument("--progress_check_start_iter", type=int, default=None, help="Iteration for start of first progress window. Default: 'progress_check_interval'.")
    parser.add_argument("--advanced_projection_decay_cap_min", type=float, default=0.5, help="Min cap for decay factor in advanced projection.")
    parser.add_argument("--advanced_projection_decay_cap_max", type=float, default=1.05, help="Max cap for decay factor in advanced projection.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device ('cuda' or 'cpu').")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Optimization precision.")
    parser.add_argument("--save_weights_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Dtype for saved LoHA weights.")
    parser.add_argument("--atol_fp32_check", type=float, default=1e-6, help="Tolerance for identical weight check.")
    parser.add_argument("--no_warm_start", action="store_true", help="Disable warm-starting higher rank attempts from previous best.")
    parser.add_argument("--use_bias", action="store_true", help="Save differing bias terms into LoHA.")
    parser.add_argument("--dropout", type=float, default=0.0, help="General dropout (metadata only).")
    parser.add_argument("--rank_dropout", type=float, default=0.0, help="Rank dropout (metadata only).")
    parser.add_argument("--module_dropout", type=float, default=0.0, help="Module dropout (metadata only).")
    parser.add_argument("--max_layers", type=int, default=None, help="Max NEW differing layers to process this session.")
    parser.add_argument("--verbose", action="store_true", help="General verbose output.")
    parser.add_argument("--verbose_layer_debug", action="store_true", help="Detailed per-iteration debug output (implies --verbose).")
    parser.add_argument("--projection_sample_interval", type=int, default=20, help="Loss sample interval for EMA (iterations).")
    parser.add_argument("--projection_ema_alpha", type=float, default=0.1, help="Smoothing factor for EMA.")
    parser.add_argument("--projection_min_ema_history", type=int, default=5, help="Min EMA samples for EMA-based projection.")
    parser.add_argument("--save_every_n_layers", type=int, default=0, help="Save intermediate LoHA every N processed layers (0 to disable).")
    parser.add_argument("--keep_n_resume_files", type=int, default=0, help="Keep only N most recent intermediate resume files (0 to keep all).")
    
    raw_parsed_args = parser.parse_args()
    processed_args = post_process_cli_args(raw_parsed_args)
    main(processed_args)