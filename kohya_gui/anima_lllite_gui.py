"""Anima ControlNet-LLLite training tab (GH issue #3525).

Wraps sd-scripts/anima_train_control_net_lllite.py. The backend is experimental
and asserts if blocks_to_swap / cpu_offload_checkpointing /
unsloth_offload_checkpointing / deepspeed / fused_backward_pass are enabled —
those options are intentionally not exposed here.
"""

import json
import os
import time
import toml

from datetime import datetime

import gradio as gr

from .class_accelerate_launch import AccelerateLaunch
from .class_anima import animaTraining
from .class_command_executor import CommandExecutor
from .class_configuration_file import ConfigurationFile
from .class_gui_config import KohyaSSGUIConfig
from .class_source_model import default_models
from .common_gui import (
    SaveConfigFile,
    check_if_model_exist,
    create_refresh_button,
    get_any_file_path,
    get_executable_path,
    get_file_path,
    get_folder_path,
    get_saveasfile_path,
    join_config_path,
    list_dirs,
    list_files,
    output_message,
    print_command_and_toml,
    require_writable_directory,
    run_cmd_advanced_training,
    scriptdir,
    setup_environment,
    update_my_data,
    validate_args_setting,
    write_toml_config,
    validate_file_path,
    validate_folder_path,
    validate_model_path,
    validate_toml_file,
)
from .custom_logging import setup_logging

log = setup_logging()

executor = None

use_shell = False
train_state_value = time.time()

# Populated by anima_lllite_tab() with the (param_name, component) pairs backing
# settings_list, in the same order as train_model's/save_configuration's/
# open_configuration's shared keyword-argument order.
last_built_field_registry = None

# Populated by anima_lllite_tab() with the dict-keyed adapter callables wired
# to the train/save/load buttons (GH #3543 M3).
last_built_gui_entries = None

document_symbol = "\U0001f4c4"  # 📄
folder_symbol = "\U0001f4c2"  # 📂

# Presets and atomic target-layer choices from
# sd-scripts/networks/control_net_lllite_anima.py (documented surface).
LLLITE_TARGET_LAYER_CHOICES = [
    "self_attn_q",
    "self_attn_qkv",
    "self_attn_qkv_cross_q",
    "self_attn_q_pre",
    "self_attn_kv_pre",
    "cross_attn_q_pre",
    "mlp_fc1_pre",
    "self_attn_q_pre,mlp_fc1_pre",
    "self_attn_q_pre,self_attn_kv_pre,mlp_fc1_pre",
    "self_attn_q_pre,self_attn_kv_pre,cross_attn_q_pre,mlp_fc1_pre",
]


def save_configuration(
    save_as_bool,
    file_path,
    # model section
    pretrained_model_name_or_path,
    output_dir,
    output_name,
    save_model_as,
    save_precision,
    training_comment,
    no_metadata,
    # dataset section
    train_data_dir,
    conditioning_data_dir,
    dataset_config,
    resolution,
    enable_bucket,
    min_bucket_reso,
    max_bucket_reso,
    train_batch_size,
    max_train_epochs,
    max_train_steps,
    caption_extension,
    cache_latents,
    cache_latents_to_disk,
    cache_text_encoder_outputs,
    cache_text_encoder_outputs_to_disk,
    seed,
    gradient_accumulation_steps,
    # anima section
    anima_qwen3,
    anima_vae,
    anima_llm_adapter_path,
    anima_t5_tokenizer_path,
    anima_discrete_flow_shift,
    anima_timestep_sampling,
    anima_sigmoid_scale,
    anima_qwen3_max_token_length,
    anima_t5_max_token_length,
    anima_attn_mode,
    anima_split_attn,
    anima_vae_chunk_size,
    anima_vae_disable_cache,
    anima_qwen_image_vae_2d,
    anima_compile,
    anima_torch_compile,
    anima_compile_backend,
    anima_compile_mode,
    anima_compile_dynamic,
    anima_compile_fullgraph,
    anima_compile_cache_size_limit,
    # lllite section
    cond_emb_dim,
    lllite_mlp_dim,
    lllite_target_layers,
    lllite_cond_dim,
    lllite_cond_resblocks,
    lllite_use_aspp,
    lllite_dropout,
    lllite_multiplier,
    network_weights,
    lllite_cond_in_channels,
    lllite_inpaint_masked_input,
    # training section
    learning_rate,
    optimizer,
    optimizer_args,
    lr_scheduler,
    lr_scheduler_args,
    lr_warmup_steps,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    max_grad_norm,
    gradient_checkpointing,
    full_fp16,
    full_bf16,
    # accelerate launch section
    mixed_precision,
    num_cpu_threads_per_process,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    # save / logging section
    save_every_n_epochs,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    save_state,
    save_state_on_train_end,
    resume,
    logging_dir,
    log_with,
    log_tracker_name,
    log_tracker_config,
    log_config,
    wandb_api_key,
    wandb_run_name,
    show_timesteps,
    show_timesteps_resolution,
):
    parameters = list(locals().items())

    original_file_path = file_path

    if save_as_bool:
        log.info("Save as...")
        file_path = get_saveasfile_path(file_path)
    else:
        log.info("Save...")
        if file_path is None or file_path == "":
            file_path = get_saveasfile_path(file_path)

    if file_path is None or file_path == "":
        return original_file_path

    SaveConfigFile(
        parameters=parameters,
        file_path=file_path,
        exclusion=["file_path", "save_as", "save_as_bool"],
    )

    log.info(f"Config saved to {file_path}")

    return file_path


def open_configuration(
    ask_for_file,
    file_path,
    # model section
    pretrained_model_name_or_path,
    output_dir,
    output_name,
    save_model_as,
    save_precision,
    training_comment,
    no_metadata,
    # dataset section
    train_data_dir,
    conditioning_data_dir,
    dataset_config,
    resolution,
    enable_bucket,
    min_bucket_reso,
    max_bucket_reso,
    train_batch_size,
    max_train_epochs,
    max_train_steps,
    caption_extension,
    cache_latents,
    cache_latents_to_disk,
    cache_text_encoder_outputs,
    cache_text_encoder_outputs_to_disk,
    seed,
    gradient_accumulation_steps,
    # anima section
    anima_qwen3,
    anima_vae,
    anima_llm_adapter_path,
    anima_t5_tokenizer_path,
    anima_discrete_flow_shift,
    anima_timestep_sampling,
    anima_sigmoid_scale,
    anima_qwen3_max_token_length,
    anima_t5_max_token_length,
    anima_attn_mode,
    anima_split_attn,
    anima_vae_chunk_size,
    anima_vae_disable_cache,
    anima_qwen_image_vae_2d,
    anima_compile,
    anima_torch_compile,
    anima_compile_backend,
    anima_compile_mode,
    anima_compile_dynamic,
    anima_compile_fullgraph,
    anima_compile_cache_size_limit,
    # lllite section
    cond_emb_dim,
    lllite_mlp_dim,
    lllite_target_layers,
    lllite_cond_dim,
    lllite_cond_resblocks,
    lllite_use_aspp,
    lllite_dropout,
    lllite_multiplier,
    network_weights,
    lllite_cond_in_channels,
    lllite_inpaint_masked_input,
    # training section
    learning_rate,
    optimizer,
    optimizer_args,
    lr_scheduler,
    lr_scheduler_args,
    lr_warmup_steps,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    max_grad_norm,
    gradient_checkpointing,
    full_fp16,
    full_bf16,
    # accelerate launch section
    mixed_precision,
    num_cpu_threads_per_process,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    # save / logging section
    save_every_n_epochs,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    save_state,
    save_state_on_train_end,
    resume,
    logging_dir,
    log_with,
    log_tracker_name,
    log_tracker_config,
    log_config,
    wandb_api_key,
    wandb_run_name,
    show_timesteps,
    show_timesteps_resolution,
):
    parameters = list(locals().items())

    original_file_path = file_path

    if ask_for_file:
        file_path = get_file_path(file_path)

    if not file_path == "" and not file_path == None:
        if not os.path.isfile(file_path):
            log.error(f"Config file {file_path} does not exist.")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            my_data = json.load(f)
            log.info("Loading config...")

            my_data = update_my_data(my_data)
    else:
        file_path = original_file_path
        my_data = {}

    values = [file_path]
    for key, value in parameters:
        if not key in ["ask_for_file", "file_path"]:
            json_value = my_data.get(key)
            values.append(json_value if json_value is not None else value)

    return tuple(values)


def train_model(
    headless,
    print_only,
    # model section
    pretrained_model_name_or_path,
    output_dir,
    output_name,
    save_model_as,
    save_precision,
    training_comment,
    no_metadata,
    # dataset section
    train_data_dir,
    conditioning_data_dir,
    dataset_config,
    resolution,
    enable_bucket,
    min_bucket_reso,
    max_bucket_reso,
    train_batch_size,
    max_train_epochs,
    max_train_steps,
    caption_extension,
    cache_latents,
    cache_latents_to_disk,
    cache_text_encoder_outputs,
    cache_text_encoder_outputs_to_disk,
    seed,
    gradient_accumulation_steps,
    # anima section
    anima_qwen3,
    anima_vae,
    anima_llm_adapter_path,
    anima_t5_tokenizer_path,
    anima_discrete_flow_shift,
    anima_timestep_sampling,
    anima_sigmoid_scale,
    anima_qwen3_max_token_length,
    anima_t5_max_token_length,
    anima_attn_mode,
    anima_split_attn,
    anima_vae_chunk_size,
    anima_vae_disable_cache,
    anima_qwen_image_vae_2d,
    anima_compile,
    anima_torch_compile,
    anima_compile_backend,
    anima_compile_mode,
    anima_compile_dynamic,
    anima_compile_fullgraph,
    anima_compile_cache_size_limit,
    # lllite section
    cond_emb_dim,
    lllite_mlp_dim,
    lllite_target_layers,
    lllite_cond_dim,
    lllite_cond_resblocks,
    lllite_use_aspp,
    lllite_dropout,
    lllite_multiplier,
    network_weights,
    lllite_cond_in_channels,
    lllite_inpaint_masked_input,
    # training section
    learning_rate,
    optimizer,
    optimizer_args,
    lr_scheduler,
    lr_scheduler_args,
    lr_warmup_steps,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    max_grad_norm,
    gradient_checkpointing,
    full_fp16,
    full_bf16,
    # accelerate launch section
    mixed_precision,
    num_cpu_threads_per_process,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    # save / logging section
    save_every_n_epochs,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    save_state,
    save_state_on_train_end,
    resume,
    logging_dir,
    log_with,
    log_tracker_name,
    log_tracker_config,
    log_config,
    wandb_api_key,
    wandb_run_name,
    show_timesteps,
    show_timesteps_resolution,
):
    parameters = list(locals().items())
    global train_state_value

    TRAIN_BUTTON_VISIBLE = [
        gr.Button(visible=True),
        gr.Button(visible=False or headless),
        gr.Textbox(value=train_state_value),
    ]

    if executor.is_running():
        log.error("Training is already running. Can't start another training session.")
        return TRAIN_BUTTON_VISIBLE

    log.info("Start training Anima ControlNet-LLLite ...")

    log.info("Validating lr scheduler arguments...")
    if not validate_args_setting(lr_scheduler_args):
        return TRAIN_BUTTON_VISIBLE

    log.info("Validating optimizer arguments...")
    if not validate_args_setting(optimizer_args):
        return TRAIN_BUTTON_VISIBLE

    # Backend asserts if both compile paths are set (anima_train_control_net_lllite.py).
    if anima_compile and anima_torch_compile:
        log.error(
            "Per-block Torch Compile and Legacy Torch Compile cannot both be "
            "enabled at the same time."
        )
        return TRAIN_BUTTON_VISIBLE

    #
    # Validate paths / required fields
    #

    if dataset_config:
        if not validate_toml_file(dataset_config):
            return TRAIN_BUTTON_VISIBLE
    else:
        if train_data_dir == "":
            log.error(
                "Train data dir is required when Dataset config is not set "
                "(ControlNet-format dataset: image + conditioning image)."
            )
            return TRAIN_BUTTON_VISIBLE
        if conditioning_data_dir == "":
            log.error(
                "Conditioning data dir is required when Dataset config is not set. "
                "See docs/train_lllite_README.md#preparing-the-dataset."
            )
            return TRAIN_BUTTON_VISIBLE
        if not validate_folder_path(train_data_dir):
            return TRAIN_BUTTON_VISIBLE
        if not validate_folder_path(conditioning_data_dir):
            return TRAIN_BUTTON_VISIBLE

    if anima_qwen3 == "" or anima_vae == "":
        log.error(
            "Anima Qwen3 path and VAE path are required for ControlNet-LLLite training."
        )
        return TRAIN_BUTTON_VISIBLE

    if not validate_model_path(pretrained_model_name_or_path):
        return TRAIN_BUTTON_VISIBLE

    if not validate_model_path(anima_qwen3):
        return TRAIN_BUTTON_VISIBLE

    if not validate_model_path(anima_vae):
        return TRAIN_BUTTON_VISIBLE

    if anima_llm_adapter_path and not validate_model_path(anima_llm_adapter_path):
        return TRAIN_BUTTON_VISIBLE

    if anima_t5_tokenizer_path and not validate_folder_path(anima_t5_tokenizer_path):
        return TRAIN_BUTTON_VISIBLE

    if not validate_file_path(log_tracker_config):
        return TRAIN_BUTTON_VISIBLE

    if not validate_folder_path(
        logging_dir, can_be_written_to=True, create_if_not_exists=True
    ):
        return TRAIN_BUTTON_VISIBLE

    if not validate_file_path(network_weights):
        return TRAIN_BUTTON_VISIBLE

    if not require_writable_directory(output_dir, headless=headless):
        return TRAIN_BUTTON_VISIBLE

    if not validate_folder_path(resume):
        return TRAIN_BUTTON_VISIBLE

    #
    # End of path validation
    #

    if not print_only and check_if_model_exist(
        output_name, output_dir, save_model_as, headless=headless
    ):
        return TRAIN_BUTTON_VISIBLE

    accelerate_path = get_executable_path("accelerate")
    if accelerate_path == "":
        log.error("accelerate not found")
        return TRAIN_BUTTON_VISIBLE

    run_cmd = [rf"{accelerate_path}", "launch"]

    run_cmd = AccelerateLaunch.run_cmd(
        run_cmd=run_cmd,
        dynamo_backend=dynamo_backend,
        dynamo_mode=dynamo_mode,
        dynamo_use_fullgraph=dynamo_use_fullgraph,
        dynamo_use_dynamic=dynamo_use_dynamic,
        num_processes=num_processes,
        num_machines=num_machines,
        multi_gpu=multi_gpu,
        gpu_ids=gpu_ids,
        main_process_port=main_process_port,
        num_cpu_threads_per_process=num_cpu_threads_per_process,
        mixed_precision=mixed_precision,
        extra_accelerate_launch_args=extra_accelerate_launch_args,
    )

    run_cmd.append(rf"{scriptdir}/sd-scripts/anima_train_control_net_lllite.py")

    # Resolution may be "1024" or "1024,1024"
    resolution_value = resolution
    if isinstance(resolution, (int, float)):
        resolution_value = str(int(resolution))

    config_toml_data = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "output_dir": output_dir,
        "output_name": output_name,
        "save_model_as": save_model_as,
        "save_precision": save_precision,
        "training_comment": training_comment,
        "no_metadata": no_metadata,
        "dataset_config": dataset_config if dataset_config else None,
        "train_data_dir": train_data_dir if not dataset_config else None,
        "conditioning_data_dir": (
            conditioning_data_dir if not dataset_config else None
        ),
        "resolution": resolution_value,
        "enable_bucket": enable_bucket,
        "min_bucket_reso": int(min_bucket_reso) if enable_bucket else None,
        "max_bucket_reso": int(max_bucket_reso) if enable_bucket else None,
        "train_batch_size": int(train_batch_size),
        "max_train_epochs": (
            int(max_train_epochs) if int(max_train_epochs) != 0 else None
        ),
        "max_train_steps": (
            int(max_train_steps) if int(max_train_steps) != 0 else None
        ),
        "caption_extension": caption_extension,
        "cache_latents": cache_latents,
        "cache_latents_to_disk": cache_latents_to_disk,
        "cache_text_encoder_outputs": cache_text_encoder_outputs,
        "cache_text_encoder_outputs_to_disk": cache_text_encoder_outputs_to_disk,
        "seed": int(seed) if int(seed) != 0 else None,
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        # Anima
        "qwen3": anima_qwen3,
        "vae": anima_vae,
        "llm_adapter_path": anima_llm_adapter_path if anima_llm_adapter_path else None,
        "t5_tokenizer_path": (
            anima_t5_tokenizer_path if anima_t5_tokenizer_path else None
        ),
        "discrete_flow_shift": float(anima_discrete_flow_shift),
        "timestep_sampling": anima_timestep_sampling,
        "sigmoid_scale": float(anima_sigmoid_scale),
        "qwen3_max_token_length": int(anima_qwen3_max_token_length),
        "t5_max_token_length": int(anima_t5_max_token_length),
        "attn_mode": anima_attn_mode if anima_attn_mode != "torch" else None,
        "split_attn": anima_split_attn,
        "vae_chunk_size": (int(anima_vae_chunk_size) if anima_vae_chunk_size else None),
        "vae_disable_cache": anima_vae_disable_cache,
        "qwen_image_vae_2d": anima_qwen_image_vae_2d,
        "compile": anima_compile,
        "torch_compile": anima_torch_compile,
        "compile_backend": anima_compile_backend if anima_compile else None,
        "compile_mode": anima_compile_mode if anima_compile else None,
        "compile_dynamic": (
            anima_compile_dynamic
            if anima_compile and anima_compile_dynamic != "auto"
            else None
        ),
        "compile_fullgraph": anima_compile_fullgraph if anima_compile else None,
        "compile_cache_size_limit": (
            int(anima_compile_cache_size_limit)
            if anima_compile and anima_compile_cache_size_limit
            else None
        ),
        # LLLite-specific (never emit unsupported MVP offload flags)
        "cond_emb_dim": int(cond_emb_dim),
        "lllite_mlp_dim": int(lllite_mlp_dim),
        "lllite_target_layers": lllite_target_layers,
        "lllite_cond_dim": int(lllite_cond_dim),
        "lllite_cond_resblocks": int(lllite_cond_resblocks),
        "lllite_use_aspp": lllite_use_aspp,
        "lllite_dropout": (
            float(lllite_dropout) if lllite_dropout not in ("", None, 0) else None
        ),
        "lllite_multiplier": float(lllite_multiplier),
        "network_weights": network_weights if network_weights else None,
        "lllite_cond_in_channels": int(lllite_cond_in_channels),
        "lllite_inpaint_masked_input": lllite_inpaint_masked_input,
        # Training
        "learning_rate": learning_rate,
        "optimizer_type": optimizer,
        "optimizer_args": (
            str(optimizer_args).replace('"', "").split()
            if optimizer_args not in ("", [], None)
            else None
        ),
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_args": (
            str(lr_scheduler_args).replace('"', "").split()
            if lr_scheduler_args not in ("", [], None)
            else None
        ),
        "lr_warmup_steps": lr_warmup_steps,
        "lr_scheduler_num_cycles": (
            int(lr_scheduler_num_cycles) if lr_scheduler_num_cycles != "" else None
        ),
        "lr_scheduler_power": lr_scheduler_power,
        "max_grad_norm": max_grad_norm,
        "gradient_checkpointing": gradient_checkpointing,
        "full_fp16": full_fp16,
        "full_bf16": full_bf16,
        "mixed_precision": mixed_precision,
        "save_every_n_epochs": (
            int(save_every_n_epochs) if int(save_every_n_epochs) != 0 else None
        ),
        "save_every_n_steps": (
            int(save_every_n_steps) if int(save_every_n_steps) != 0 else None
        ),
        "save_last_n_steps": (
            int(save_last_n_steps) if int(save_last_n_steps) != 0 else None
        ),
        "save_last_n_steps_state": (
            int(save_last_n_steps_state) if int(save_last_n_steps_state) != 0 else None
        ),
        "save_state": save_state,
        "save_state_on_train_end": save_state_on_train_end,
        "resume": resume,
        "logging_dir": logging_dir,
        "log_with": log_with,
        "log_tracker_name": log_tracker_name,
        "log_tracker_config": log_tracker_config,
        "log_config": log_config,
        "wandb_api_key": wandb_api_key,
        "wandb_run_name": wandb_run_name if wandb_run_name != "" else output_name,
        "show_timesteps": show_timesteps if show_timesteps else None,
        "show_timesteps_resolution": (
            show_timesteps_resolution if show_timesteps else None
        ),
    }

    # Remove empty / False / None values so flags like lllite_use_aspp only
    # appear when enabled.
    config_toml_data = {
        key: value
        for key, value in config_toml_data.items()
        if value not in ["", False, None]
    }

    config_toml_data = dict(sorted(config_toml_data.items()))

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
    tmpfilename = join_config_path(
        output_dir, f"config_anima_lllite-{formatted_datetime}.toml"
    )

    if not write_toml_config(tmpfilename, config_toml_data, headless=headless):
        return TRAIN_BUTTON_VISIBLE

    run_cmd.append("--config_file")
    run_cmd.append(rf"{tmpfilename}")

    run_cmd = run_cmd_advanced_training(run_cmd=run_cmd)

    if print_only:
        print_command_and_toml(run_cmd, tmpfilename)
    else:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
        file_path = join_config_path(
            output_dir, f"{output_name}_{formatted_datetime}.json"
        )

        log.info(f"Saving training config to {file_path}...")

        try:
            SaveConfigFile(
                parameters=parameters,
                file_path=file_path,
                exclusion=["file_path", "save_as", "headless", "print_only"],
            )
        except OSError as exc:
            msg = f"Failed to write training config {file_path}: {exc}"
            log.error(msg)
            output_message(msg=msg, headless=headless)
            return TRAIN_BUTTON_VISIBLE

        env = setup_environment()

        executor.execute_command(run_cmd=run_cmd, env=env)

        train_state_value = time.time()

        return (
            gr.Button(visible=False or headless),
            gr.Button(visible=True),
            gr.Textbox(value=train_state_value),
        )


def anima_lllite_tab(
    headless=False,
    config: KohyaSSGUIConfig = {},
    use_shell_flag: bool = False,
):
    global use_shell, executor
    use_shell = use_shell_flag

    dummy_db_true = gr.Checkbox(value=True, visible=False)
    dummy_db_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)
    # animaTraining accordion is always shown on this dedicated tab.
    anima_always_on = gr.Checkbox(value=True, visible=False)

    current_models_dir = config.get(
        "model.models_dir", os.path.join(scriptdir, "models")
    )
    current_train_data_dir = config.get(
        "model.train_data_dir", os.path.join(scriptdir, "data")
    )
    current_cond_data_dir = config.get(
        "anima_lllite.conditioning_data_dir",
        os.path.join(scriptdir, "data", "conditioning"),
    )
    current_dataset_config_dir = config.get(
        "model.dataset_config", os.path.join(scriptdir, "dataset_config")
    )

    def list_models(path):
        nonlocal current_models_dir
        current_models_dir = path if os.path.isdir(path) else os.path.dirname(path)
        return default_models + list(
            list_files(path, exts=[".ckpt", ".safetensors"], all=True)
        )

    def list_train_data_dirs(path):
        nonlocal current_train_data_dir
        current_train_data_dir = path if not path == "" else "."
        return list(list_dirs(current_train_data_dir))

    def list_cond_data_dirs(path):
        nonlocal current_cond_data_dir
        current_cond_data_dir = path if not path == "" else "."
        return list(list_dirs(current_cond_data_dir))

    def list_dataset_config_dirs(path):
        nonlocal current_dataset_config_dir
        current_dataset_config_dir = path if not path == "" else "."
        return list(list_files(current_dataset_config_dir, exts=[".toml"], all=True))

    model_checkpoints = list(
        list_files(current_models_dir, exts=[".ckpt", ".safetensors"], all=True)
    )

    with gr.Tab("Training"), gr.Column(variant="compact"):
        gr.Markdown(
            "Train an Anima **ControlNet-LLLite** adapter "
            "(`anima_train_control_net_lllite.py`). Requires a ControlNet-format "
            "dataset (training image + conditioning image with matching basenames). "
            "See `sd-scripts/docs/anima_train_control_net_lllite.md` and "
            "`docs/train_lllite_README.md#preparing-the-dataset`. "
            "**Experimental:** blocks-to-swap, CPU/Unsloth offload checkpointing, "
            "DeepSpeed, and fused backward pass are not supported and are not "
            "exposed in this tab."
        )

        with gr.Accordion("Configuration", open=False):
            configuration = ConfigurationFile(headless=headless, config=config)

        with gr.Accordion("Accelerate launch", open=False), gr.Column():
            accelerate_launch = AccelerateLaunch(config=config)

        with gr.Accordion("Model", open=True):
            with gr.Column(), gr.Group():
                model_ext = gr.Textbox(value="*.safetensors *.ckpt", visible=False)
                model_ext_name = gr.Textbox(value="Model types", visible=False)

                with gr.Row():
                    pretrained_model_name_or_path = gr.Dropdown(
                        label="Pretrained model name or path (Anima DiT)",
                        choices=default_models + model_checkpoints,
                        value=config.get(
                            "model.pretrained_model_name_or_path",
                            "",
                        ),
                        allow_custom_value=True,
                        min_width=100,
                    )
                    create_refresh_button(
                        pretrained_model_name_or_path,
                        lambda: None,
                        lambda: {"choices": list_models(current_models_dir)},
                        "open_folder_small",
                    )
                    pretrained_model_name_or_path_file = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        elem_classes=["tool"],
                        visible=(not headless),
                    )
                    pretrained_model_name_or_path_file.click(
                        get_file_path,
                        inputs=[
                            pretrained_model_name_or_path,
                            model_ext,
                            model_ext_name,
                        ],
                        outputs=pretrained_model_name_or_path,
                        show_progress=False,
                    )
                    pretrained_model_name_or_path_folder = gr.Button(
                        folder_symbol,
                        elem_id="open_folder_small",
                        elem_classes=["tool"],
                        visible=(not headless),
                    )
                    pretrained_model_name_or_path_folder.click(
                        get_folder_path,
                        inputs=pretrained_model_name_or_path,
                        outputs=pretrained_model_name_or_path,
                        show_progress=False,
                    )

                with gr.Row():
                    output_dir = gr.Textbox(
                        label="Output directory",
                        placeholder="Directory to output the trained LLLite model",
                        value=config.get(
                            "model.output_dir", os.path.join(scriptdir, "outputs")
                        ),
                        interactive=True,
                    )
                    output_dir_button = gr.Button(
                        folder_symbol,
                        elem_id="open_folder_small",
                        elem_classes=["tool"],
                        visible=(not headless),
                    )
                    output_dir_button.click(
                        get_folder_path,
                        outputs=output_dir,
                        show_progress=False,
                    )
                    output_name = gr.Textbox(
                        label="Trained model output name",
                        placeholder="(Name of the model to output)",
                        value=config.get("model.output_name", "last"),
                        interactive=True,
                    )
                    training_comment = gr.Textbox(
                        label="Training comment",
                        placeholder="(Optional) Add training comment to be included in metadata",
                        value=config.get("model.training_comment", ""),
                        interactive=True,
                    )

                with gr.Row():
                    save_model_as = gr.Radio(
                        ["ckpt", "safetensors"],
                        label="Save trained model as",
                        value=config.get("model.save_model_as", "safetensors"),
                    )
                    save_precision = gr.Radio(
                        ["float", "fp16", "bf16"],
                        label="Save precision",
                        value=config.get("model.save_precision", "bf16"),
                    )
                    no_metadata = gr.Checkbox(
                        label="No metadata",
                        value=config.get("model.no_metadata", False),
                        info="Do not save metadata in output model",
                    )

        # Anima model paths / flow-matching options (always visible on this tab).
        # Unsloth offload is hidden: unsupported by anima_train_control_net_lllite.
        anima_training = animaTraining(
            headless=headless,
            config=config,
            anima_checkbox=anima_always_on,
            always_visible=True,
            show_unsloth_offload_checkpointing=False,
        )

        with gr.Accordion("ControlNet dataset", open=True), gr.Group():
            gr.Markdown(
                "Provide either a dataset TOML (`conditioning_data_dir` per subset) "
                "or folder pair: training images + conditioning images with matching "
                "basenames. Same layout as SDXL ControlNet-LLLite."
            )
            with gr.Row():
                train_data_dir = gr.Dropdown(
                    label="Image folder (training images)",
                    choices=[""] + list(list_dirs(current_train_data_dir)),
                    value=config.get("model.train_data_dir", ""),
                    interactive=True,
                    allow_custom_value=True,
                )
                create_refresh_button(
                    train_data_dir,
                    lambda: None,
                    lambda: {
                        "choices": [""] + list_train_data_dirs(current_train_data_dir)
                    },
                    "open_folder_small",
                )
                train_data_dir_folder = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not headless),
                )
                train_data_dir_folder.click(
                    get_folder_path,
                    outputs=train_data_dir,
                    show_progress=False,
                )

                conditioning_data_dir = gr.Dropdown(
                    label="Conditioning image folder",
                    choices=[""] + list(list_dirs(current_cond_data_dir)),
                    value=config.get("anima_lllite.conditioning_data_dir", ""),
                    interactive=True,
                    allow_custom_value=True,
                    info="Conditioning images (canny, lineart, depth, …) with the same basenames as training images",
                )
                create_refresh_button(
                    conditioning_data_dir,
                    lambda: None,
                    lambda: {
                        "choices": [""] + list_cond_data_dirs(current_cond_data_dir)
                    },
                    "open_folder_small",
                )
                conditioning_data_dir_folder = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not headless),
                )
                conditioning_data_dir_folder.click(
                    get_folder_path,
                    outputs=conditioning_data_dir,
                    show_progress=False,
                )

            with gr.Row():
                dataset_config = gr.Dropdown(
                    label="Dataset config (TOML)",
                    choices=[""]
                    + list(
                        list_files(current_dataset_config_dir, exts=[".toml"], all=True)
                    ),
                    value=config.get("model.dataset_config", ""),
                    interactive=True,
                    allow_custom_value=True,
                    info="When set, train/conditioning folder fields above are ignored",
                )
                create_refresh_button(
                    dataset_config,
                    lambda: None,
                    lambda: {
                        "choices": [""]
                        + list_dataset_config_dirs(current_dataset_config_dir)
                    },
                    "open_folder_small",
                )
                dataset_config_file = gr.Button(
                    document_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not headless),
                )
                dataset_config_file.click(
                    get_file_path,
                    inputs=[
                        dataset_config,
                        gr.Textbox(value="*.toml", visible=False),
                        gr.Textbox(value="Dataset config TOML", visible=False),
                    ],
                    outputs=dataset_config,
                    show_progress=False,
                )

            with gr.Row():
                resolution = gr.Textbox(
                    label="Resolution",
                    value=config.get("anima_lllite.resolution", "1024"),
                    info='e.g. "1024" or "1024,1024"',
                    interactive=True,
                )
                enable_bucket = gr.Checkbox(
                    label="Enable buckets",
                    value=config.get("anima_lllite.enable_bucket", True),
                )
                min_bucket_reso = gr.Number(
                    label="Min bucket resolution",
                    value=config.get("anima_lllite.min_bucket_reso", 256),
                    precision=0,
                )
                max_bucket_reso = gr.Number(
                    label="Max bucket resolution",
                    value=config.get("anima_lllite.max_bucket_reso", 2048),
                    precision=0,
                )

            with gr.Row():
                train_batch_size = gr.Slider(
                    label="Train batch size",
                    value=config.get("anima_lllite.train_batch_size", 1),
                    minimum=1,
                    maximum=64,
                    step=1,
                )
                max_train_epochs = gr.Number(
                    label="Max train epochs",
                    value=config.get("anima_lllite.max_train_epochs", 10),
                    precision=0,
                    info="0 leaves steps-based training if Max train steps is set",
                )
                max_train_steps = gr.Number(
                    label="Max train steps",
                    value=config.get("anima_lllite.max_train_steps", 0),
                    precision=0,
                    info="0 uses epochs only",
                )
                caption_extension = gr.Textbox(
                    label="Caption extension",
                    value=config.get("anima_lllite.caption_extension", ".txt"),
                    interactive=True,
                )

            with gr.Row():
                cache_latents = gr.Checkbox(
                    label="Cache latents",
                    value=config.get("anima_lllite.cache_latents", True),
                )
                cache_latents_to_disk = gr.Checkbox(
                    label="Cache latents to disk",
                    value=config.get("anima_lllite.cache_latents_to_disk", False),
                )
                # Text-encoder cache checkboxes live on the Anima accordion
                # (animaTraining); wired via FIELD_REGISTRY below.

            with gr.Row():
                seed = gr.Number(
                    label="Seed",
                    value=config.get("anima_lllite.seed", 0),
                    precision=0,
                    info="0 lets sd-scripts pick a random seed",
                )
                gradient_accumulation_steps = gr.Slider(
                    label="Gradient accumulate steps",
                    value=config.get("anima_lllite.gradient_accumulation_steps", 1),
                    minimum=1,
                    maximum=120,
                    step=1,
                )

        with gr.Accordion("LLLite parameters", open=True), gr.Group():
            gr.Markdown(
                "Parameters unique to Anima ControlNet-LLLite. "
                "`target_layers` accepts a preset name or comma-separated atomic "
                "specifiers (including `mlp_fc1_pre`). Optional ASPP tail via "
                "`lllite_use_aspp`."
            )
            with gr.Row():
                cond_emb_dim = gr.Number(
                    label="Conditioning embedding dim",
                    value=config.get("anima_lllite.cond_emb_dim", 32),
                    precision=0,
                    info="--cond_emb_dim (shared cond_emb width)",
                )
                lllite_mlp_dim = gr.Number(
                    label="LLLite MLP dim",
                    value=config.get("anima_lllite.lllite_mlp_dim", 64),
                    precision=0,
                    info="--lllite_mlp_dim (LoRA-rank-like hidden dim)",
                )
                lllite_cond_dim = gr.Number(
                    label="LLLite cond trunk dim",
                    value=config.get("anima_lllite.lllite_cond_dim", 64),
                    precision=0,
                    info="--lllite_cond_dim (conditioning1 internal width)",
                )
                lllite_cond_resblocks = gr.Number(
                    label="LLLite cond ResBlocks",
                    value=config.get("anima_lllite.lllite_cond_resblocks", 1),
                    precision=0,
                    info="--lllite_cond_resblocks",
                )
            with gr.Row():
                lllite_target_layers = gr.Dropdown(
                    label="Target layers",
                    choices=LLLITE_TARGET_LAYER_CHOICES,
                    value=config.get(
                        "anima_lllite.lllite_target_layers", "self_attn_q"
                    ),
                    allow_custom_value=True,
                    info="Preset or comma-separated atomic specifiers (e.g. self_attn_q_pre,mlp_fc1_pre)",
                    interactive=True,
                )
                lllite_use_aspp = gr.Checkbox(
                    label="Use ASPP",
                    value=config.get("anima_lllite.lllite_use_aspp", False),
                    info="--lllite_use_aspp: multi-scale ASPP tail on conditioning1",
                )
                lllite_dropout = gr.Number(
                    label="LLLite dropout",
                    value=config.get("anima_lllite.lllite_dropout", 0),
                    info="0 disables; otherwise mid-output dropout rate",
                )
                lllite_multiplier = gr.Number(
                    label="LLLite multiplier",
                    value=config.get("anima_lllite.lllite_multiplier", 1.0),
                    info="Do not set to 0 (disables LLLite during training)",
                )
            with gr.Row():
                network_weights = gr.Textbox(
                    label="Network weights (resume LLLite)",
                    placeholder="(Optional) Path to pretrained LLLite .safetensors",
                    value=config.get("anima_lllite.network_weights", ""),
                    interactive=True,
                )
                network_weights_file = gr.Button(
                    document_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not headless),
                )
                network_weights_file.click(
                    get_any_file_path,
                    inputs=[network_weights],
                    outputs=network_weights,
                    show_progress=False,
                )
                lllite_cond_in_channels = gr.Dropdown(
                    label="Cond in channels",
                    choices=[3, 4],
                    value=config.get("anima_lllite.lllite_cond_in_channels", 3),
                    info="3=RGB control; 4=inpainting (RGB+mask)",
                    interactive=True,
                )
                lllite_inpaint_masked_input = gr.Checkbox(
                    label="Inpaint masked input",
                    value=config.get("anima_lllite.lllite_inpaint_masked_input", False),
                    info="Zero RGB inside mask before concat (only with cond in channels=4)",
                )

        with gr.Accordion("Training parameters", open=True), gr.Group():
            with gr.Row():
                learning_rate = gr.Number(
                    label="Learning rate",
                    value=config.get("training.learning_rate", 5e-5),
                    minimum=0,
                    maximum=1,
                )
                optimizer = gr.Dropdown(
                    label="Optimizer",
                    choices=[
                        "AdamW",
                        "AdamW8bit",
                        "PagedAdamW",
                        "PagedAdamW8bit",
                        "PagedAdamW32bit",
                        "Lion8bit",
                        "PagedLion8bit",
                        "Lion",
                        "SGDNesterov",
                        "SGDNesterov8bit",
                        "AdaFactor",
                    ],
                    value=config.get("training.optimizer", "AdamW8bit"),
                    interactive=True,
                )
                optimizer_args = gr.Textbox(
                    label="Optimizer extra arguments",
                    placeholder="(Optional) eg: weight_decay=0.01 betas=0.9,0.999",
                    value=config.get("training.optimizer_args", ""),
                    interactive=True,
                )
            with gr.Row():
                lr_scheduler = gr.Dropdown(
                    label="LR Scheduler",
                    choices=[
                        "constant",
                        "constant_with_warmup",
                        "cosine",
                        "cosine_with_restarts",
                        "linear",
                        "polynomial",
                    ],
                    value=config.get("training.lr_scheduler", "constant"),
                    interactive=True,
                )
                lr_warmup_steps = gr.Number(
                    label="LR warmup steps",
                    value=config.get("training.lr_warmup_steps", 0),
                    precision=0,
                )
                lr_scheduler_num_cycles = gr.Number(
                    label="LR number of cycles",
                    value=config.get("training.lr_scheduler_num_cycles", ""),
                    info="Only used with cosine_with_restarts scheduler",
                )
                lr_scheduler_power = gr.Number(
                    label="LR power",
                    value=config.get("training.lr_scheduler_power", 1),
                    info="Only used with polynomial scheduler",
                )
            with gr.Row():
                lr_scheduler_args = gr.Textbox(
                    label="LR Scheduler extra arguments",
                    placeholder="(Optional) eg: T_max=100",
                    value=config.get("training.lr_scheduler_args", ""),
                    interactive=True,
                )
                max_grad_norm = gr.Number(
                    label="Max grad norm",
                    value=config.get("training.max_grad_norm", 1.0),
                    info="0 disables gradient clipping",
                )
                gradient_checkpointing = gr.Checkbox(
                    label="Gradient checkpointing",
                    value=config.get("advanced.gradient_checkpointing", True),
                )
                full_fp16 = gr.Checkbox(
                    label="Full fp16 training",
                    value=config.get("advanced.full_fp16", False),
                )
                full_bf16 = gr.Checkbox(
                    label="Full bf16 training",
                    value=config.get("advanced.full_bf16", False),
                )

            with gr.Row():
                show_timesteps = gr.Dropdown(
                    label="Show timesteps",
                    choices=["", "console", "image"],
                    value=config.get("anima_lllite.show_timesteps", ""),
                    info="Visualize timestep sampling then exit (does not train)",
                )
                show_timesteps_resolution = gr.Textbox(
                    label="Show timesteps resolution",
                    value=config.get("anima_lllite.show_timesteps_resolution", "1024"),
                    interactive=True,
                )

        with gr.Accordion("Save, resume and logging", open=False), gr.Group():
            with gr.Row():
                save_every_n_epochs = gr.Number(
                    label="Save every N epochs",
                    value=config.get("save.save_every_n_epochs", 1),
                    precision=0,
                )
                save_every_n_steps = gr.Number(
                    label="Save every N steps",
                    value=config.get("save.save_every_n_steps", 0),
                    precision=0,
                )
                save_last_n_steps = gr.Number(
                    label="Save last N steps",
                    value=config.get("save.save_last_n_steps", 0),
                    precision=0,
                )
                save_last_n_steps_state = gr.Number(
                    label="Save last N steps state",
                    value=config.get("save.save_last_n_steps_state", 0),
                    precision=0,
                )
            with gr.Row():
                save_state = gr.Checkbox(
                    label="Save state",
                    value=config.get("save.save_state", False),
                )
                save_state_on_train_end = gr.Checkbox(
                    label="Save state on train end",
                    value=config.get("save.save_state_on_train_end", False),
                )
                resume = gr.Textbox(
                    label="Resume from saved state",
                    placeholder="(Optional) path to a saved training state to resume from",
                    value=config.get("save.resume", ""),
                    interactive=True,
                )
                resume_button = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not headless),
                )
                resume_button.click(
                    get_folder_path,
                    outputs=resume,
                    show_progress=False,
                )
            with gr.Row():
                logging_dir = gr.Textbox(
                    label="Logging directory",
                    placeholder="(Optional) directory to output TensorBoard logs",
                    value=config.get("save.logging_dir", ""),
                    interactive=True,
                )
                logging_dir_button = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not headless),
                )
                logging_dir_button.click(
                    get_folder_path,
                    outputs=logging_dir,
                    show_progress=False,
                )
                log_with = gr.Dropdown(
                    label="Logging tool",
                    choices=["", "tensorboard", "wandb", "all"],
                    value=config.get("save.log_with", ""),
                )
            with gr.Row():
                log_tracker_name = gr.Textbox(
                    label="Log tracker name",
                    value=config.get("save.log_tracker_name", ""),
                    interactive=True,
                )
                log_tracker_config = gr.Textbox(
                    label="Log tracker config",
                    value=config.get("save.log_tracker_config", ""),
                    interactive=True,
                )
                log_config = gr.Checkbox(
                    label="Log training configuration",
                    value=config.get("save.log_config", False),
                )
            with gr.Row():
                wandb_api_key = gr.Textbox(
                    label="WANDB API Key",
                    value=config.get("save.wandb_api_key", ""),
                    interactive=True,
                )
                wandb_run_name = gr.Textbox(
                    label="WANDB run name",
                    value=config.get("save.wandb_run_name", ""),
                    interactive=True,
                )

        executor = CommandExecutor(headless=headless)

        button_print = gr.Button("Print training command")

        FIELD_REGISTRY = [
            ("pretrained_model_name_or_path", pretrained_model_name_or_path),
            ("output_dir", output_dir),
            ("output_name", output_name),
            ("save_model_as", save_model_as),
            ("save_precision", save_precision),
            ("training_comment", training_comment),
            ("no_metadata", no_metadata),
            ("train_data_dir", train_data_dir),
            ("conditioning_data_dir", conditioning_data_dir),
            ("dataset_config", dataset_config),
            ("resolution", resolution),
            ("enable_bucket", enable_bucket),
            ("min_bucket_reso", min_bucket_reso),
            ("max_bucket_reso", max_bucket_reso),
            ("train_batch_size", train_batch_size),
            ("max_train_epochs", max_train_epochs),
            ("max_train_steps", max_train_steps),
            ("caption_extension", caption_extension),
            ("cache_latents", cache_latents),
            ("cache_latents_to_disk", cache_latents_to_disk),
            (
                "cache_text_encoder_outputs",
                anima_training.cache_text_encoder_outputs,
            ),
            (
                "cache_text_encoder_outputs_to_disk",
                anima_training.cache_text_encoder_outputs_to_disk,
            ),
            ("seed", seed),
            ("gradient_accumulation_steps", gradient_accumulation_steps),
            ("anima_qwen3", anima_training.qwen3),
            ("anima_vae", anima_training.vae),
            ("anima_llm_adapter_path", anima_training.llm_adapter_path),
            ("anima_t5_tokenizer_path", anima_training.t5_tokenizer_path),
            ("anima_discrete_flow_shift", anima_training.discrete_flow_shift),
            ("anima_timestep_sampling", anima_training.timestep_sampling),
            ("anima_sigmoid_scale", anima_training.sigmoid_scale),
            ("anima_qwen3_max_token_length", anima_training.qwen3_max_token_length),
            ("anima_t5_max_token_length", anima_training.t5_max_token_length),
            ("anima_attn_mode", anima_training.attn_mode),
            ("anima_split_attn", anima_training.split_attn),
            ("anima_vae_chunk_size", anima_training.vae_chunk_size),
            ("anima_vae_disable_cache", anima_training.vae_disable_cache),
            ("anima_qwen_image_vae_2d", anima_training.qwen_image_vae_2d),
            ("anima_compile", anima_training.compile),
            ("anima_torch_compile", anima_training.torch_compile),
            ("anima_compile_backend", anima_training.compile_backend),
            ("anima_compile_mode", anima_training.compile_mode),
            ("anima_compile_dynamic", anima_training.compile_dynamic),
            ("anima_compile_fullgraph", anima_training.compile_fullgraph),
            (
                "anima_compile_cache_size_limit",
                anima_training.compile_cache_size_limit,
            ),
            ("cond_emb_dim", cond_emb_dim),
            ("lllite_mlp_dim", lllite_mlp_dim),
            ("lllite_target_layers", lllite_target_layers),
            ("lllite_cond_dim", lllite_cond_dim),
            ("lllite_cond_resblocks", lllite_cond_resblocks),
            ("lllite_use_aspp", lllite_use_aspp),
            ("lllite_dropout", lllite_dropout),
            ("lllite_multiplier", lllite_multiplier),
            ("network_weights", network_weights),
            ("lllite_cond_in_channels", lllite_cond_in_channels),
            ("lllite_inpaint_masked_input", lllite_inpaint_masked_input),
            ("learning_rate", learning_rate),
            ("optimizer", optimizer),
            ("optimizer_args", optimizer_args),
            ("lr_scheduler", lr_scheduler),
            ("lr_scheduler_args", lr_scheduler_args),
            ("lr_warmup_steps", lr_warmup_steps),
            ("lr_scheduler_num_cycles", lr_scheduler_num_cycles),
            ("lr_scheduler_power", lr_scheduler_power),
            ("max_grad_norm", max_grad_norm),
            ("gradient_checkpointing", gradient_checkpointing),
            ("full_fp16", full_fp16),
            ("full_bf16", full_bf16),
            ("mixed_precision", accelerate_launch.mixed_precision),
            (
                "num_cpu_threads_per_process",
                accelerate_launch.num_cpu_threads_per_process,
            ),
            ("num_processes", accelerate_launch.num_processes),
            ("num_machines", accelerate_launch.num_machines),
            ("multi_gpu", accelerate_launch.multi_gpu),
            ("gpu_ids", accelerate_launch.gpu_ids),
            ("main_process_port", accelerate_launch.main_process_port),
            ("dynamo_backend", accelerate_launch.dynamo_backend),
            ("dynamo_mode", accelerate_launch.dynamo_mode),
            ("dynamo_use_fullgraph", accelerate_launch.dynamo_use_fullgraph),
            ("dynamo_use_dynamic", accelerate_launch.dynamo_use_dynamic),
            (
                "extra_accelerate_launch_args",
                accelerate_launch.extra_accelerate_launch_args,
            ),
            ("save_every_n_epochs", save_every_n_epochs),
            ("save_every_n_steps", save_every_n_steps),
            ("save_last_n_steps", save_last_n_steps),
            ("save_last_n_steps_state", save_last_n_steps_state),
            ("save_state", save_state),
            ("save_state_on_train_end", save_state_on_train_end),
            ("resume", resume),
            ("logging_dir", logging_dir),
            ("log_with", log_with),
            ("log_tracker_name", log_tracker_name),
            ("log_tracker_config", log_tracker_config),
            ("log_config", log_config),
            ("wandb_api_key", wandb_api_key),
            ("wandb_run_name", wandb_run_name),
            ("show_timesteps", show_timesteps),
            ("show_timesteps_resolution", show_timesteps_resolution),
        ]
        settings_list = [comp for _, comp in FIELD_REGISTRY]

        global last_built_field_registry
        last_built_field_registry = FIELD_REGISTRY

        def _kwargs_from_registry(data: dict) -> dict:
            return {name: data[comp] for name, comp in FIELD_REGISTRY}

        def _make_open_configuration_entry(ask_for_file_comp):
            def _entry(data: dict):
                output_components = [configuration.config_file_name] + settings_list
                result = open_configuration(
                    ask_for_file=data[ask_for_file_comp],
                    file_path=data[configuration.config_file_name],
                    **_kwargs_from_registry(data),
                )
                if result is None:
                    return {comp: gr.update() for comp in output_components}
                if len(result) != len(output_components):
                    raise ValueError(
                        f"open_configuration returned {len(result)} values, "
                        f"expected {len(output_components)} "
                        f"(FIELD_REGISTRY/signature drift?)"
                    )
                return dict(zip(output_components, result))

            return _entry

        def _save_configuration_entry(data: dict):
            return save_configuration(
                save_as_bool=data[dummy_db_false],
                file_path=data[configuration.config_file_name],
                **_kwargs_from_registry(data),
            )

        def _make_train_model_entry(print_only_comp):
            def _entry(data: dict):
                return train_model(
                    headless=data[dummy_headless],
                    print_only=data[print_only_comp],
                    **_kwargs_from_registry(data),
                )

            return _entry

        open_config_entry = _make_open_configuration_entry(dummy_db_true)
        load_config_entry = _make_open_configuration_entry(dummy_db_false)
        train_model_entry = _make_train_model_entry(dummy_db_false)
        print_command_entry = _make_train_model_entry(dummy_db_true)

        global last_built_gui_entries
        last_built_gui_entries = {
            "open_configuration": open_config_entry,
            "load_configuration": load_config_entry,
            "save_configuration": _save_configuration_entry,
            "train_model": train_model_entry,
            "print_command": print_command_entry,
            "components": {
                "dummy_headless": dummy_headless,
                "dummy_db_true": dummy_db_true,
                "dummy_db_false": dummy_db_false,
                "config_file_name": configuration.config_file_name,
            },
        }

        configuration.button_open_config.click(
            open_config_entry,
            inputs={dummy_db_true, configuration.config_file_name, *settings_list},
            outputs={configuration.config_file_name, *settings_list},
            show_progress=False,
        )

        configuration.button_load_config.click(
            load_config_entry,
            inputs={dummy_db_false, configuration.config_file_name, *settings_list},
            outputs={configuration.config_file_name, *settings_list},
            show_progress=False,
        )

        configuration.button_save_config.click(
            _save_configuration_entry,
            inputs={dummy_db_false, configuration.config_file_name, *settings_list},
            outputs=[configuration.config_file_name],
            show_progress=False,
        )

        run_state = gr.Textbox(value=train_state_value, visible=False)

        run_state.change(
            fn=executor.wait_for_training_to_end,
            outputs=[executor.button_run, executor.button_stop_training],
        )

        executor.button_run.click(
            train_model_entry,
            inputs={dummy_headless, dummy_db_false, *settings_list},
            outputs=[executor.button_run, executor.button_stop_training, run_state],
            show_progress=False,
        )

        executor.button_stop_training.click(
            executor.kill_command,
            outputs=[executor.button_run, executor.button_stop_training],
        )

        button_print.click(
            print_command_entry,
            inputs={dummy_headless, dummy_db_true, *settings_list},
            show_progress=False,
        )
