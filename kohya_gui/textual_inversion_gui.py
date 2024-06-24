import gradio as gr
import json
import math
import os
import toml
import time
from datetime import datetime
from .common_gui import (
    check_if_model_exist,
    color_aug_changed,
    create_refresh_button,
    get_executable_path,
    get_file_path,
    get_saveasfile_path,
    list_files,
    output_message,
    print_command_and_toml,
    run_cmd_advanced_training,
    SaveConfigFile,
    scriptdir,
    update_my_data,
    validate_file_path, validate_folder_path, validate_model_path,
    validate_args_setting, setup_environment,
)
from .class_accelerate_launch import AccelerateLaunch
from .class_configuration_file import ConfigurationFile
from .class_source_model import SourceModel
from .class_basic_training import BasicTraining
from .class_advanced_training import AdvancedTraining
from .class_folders import Folders
from .class_sdxl_parameters import SDXLParameters
from .class_command_executor import CommandExecutor
from .class_huggingface import HuggingFace
from .class_metadata import MetaData
from .class_tensorboard import TensorboardManager
from .dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from .dataset_balancing_gui import gradio_dataset_balancing_tab
from .class_sample_images import SampleImages, create_prompt_file
from .class_gui_config import KohyaSSGUIConfig

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = None

# Setup huggingface
huggingface = None
use_shell = False
train_state_value = time.time()


def save_configuration(
    save_as_bool,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    dataset_config,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training,
    min_bucket_reso,
    max_bucket_reso,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    save_state_on_train_end,
    resume,
    prior_loss_weight,
    color_aug,
    flip_aug,
    clip_skip,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    vae,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    output_name,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    mem_eff_attn,
    gradient_accumulation_steps,
    model_list,
    token_string,
    init_word,
    num_vectors_per_token,
    max_train_steps,
    weights,
    template,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    v_pred_like_loss,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    lr_scheduler_args,
    lr_scheduler_type,
    noise_offset_type,
    noise_offset,
    noise_offset_random_strength,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    ip_noise_gamma,
    ip_noise_gamma_random_strength,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    loss_type,
    huber_schedule,
    huber_c,
    vae_batch_size,
    min_snr_gamma,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    log_with,
    wandb_api_key,
    wandb_run_name,
    log_tracker_name,
    log_tracker_config,
    log_config,
    scale_v_pred_loss_like_noise_pred,
    disable_mmap_load_safetensors,
    min_timestep,
    max_timestep,
    sdxl_no_half_vae,
    huggingface_repo_id,
    huggingface_token,
    huggingface_repo_type,
    huggingface_repo_visibility,
    huggingface_path_in_repo,
    save_state_to_huggingface,
    resume_from_huggingface,
    async_upload,
    metadata_author,
    metadata_description,
    metadata_license,
    metadata_tags,
    metadata_title,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    original_file_path = file_path

    if save_as_bool:
        log.info("Save as...")
        file_path = get_saveasfile_path(file_path)
    else:
        log.info("Save...")
        if file_path == None or file_path == "":
            file_path = get_saveasfile_path(file_path)

    # log.info(file_path)

    if file_path == None or file_path == "":
        return original_file_path  # In case a file_path was provided and the user decide to cancel the open action

    # Extract the destination directory from the file path
    destination_directory = os.path.dirname(file_path)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    SaveConfigFile(
        parameters=parameters,
        file_path=file_path,
        exclusion=["file_path", "save_as"],
    )

    return file_path


def open_configuration(
    ask_for_file,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    dataset_config,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training,
    min_bucket_reso,
    max_bucket_reso,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    save_state_on_train_end,
    resume,
    prior_loss_weight,
    color_aug,
    flip_aug,
    clip_skip,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    vae,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    output_name,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    mem_eff_attn,
    gradient_accumulation_steps,
    model_list,
    token_string,
    init_word,
    num_vectors_per_token,
    max_train_steps,
    weights,
    template,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    v_pred_like_loss,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    lr_scheduler_args,
    lr_scheduler_type,
    noise_offset_type,
    noise_offset,
    noise_offset_random_strength,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    ip_noise_gamma,
    ip_noise_gamma_random_strength,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    loss_type,
    huber_schedule,
    huber_c,
    vae_batch_size,
    min_snr_gamma,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    log_with,
    wandb_api_key,
    wandb_run_name,
    log_tracker_name,
    log_tracker_config,
    log_config,
    scale_v_pred_loss_like_noise_pred,
    disable_mmap_load_safetensors,
    min_timestep,
    max_timestep,
    sdxl_no_half_vae,
    huggingface_repo_id,
    huggingface_token,
    huggingface_repo_type,
    huggingface_repo_visibility,
    huggingface_path_in_repo,
    save_state_to_huggingface,
    resume_from_huggingface,
    async_upload,
    metadata_author,
    metadata_description,
    metadata_license,
    metadata_tags,
    metadata_title,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    original_file_path = file_path

    if ask_for_file:
        file_path = get_file_path(file_path)

    if not file_path == "" and not file_path == None:
        # load variables from JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            my_data = json.load(f)
            log.info("Loading config...")
            # Update values to fix deprecated use_8bit_adam checkbox and set appropriate optimizer if it is set to True
            my_data = update_my_data(my_data)
    else:
        file_path = original_file_path  # In case a file_path was provided and the user decide to cancel the open action
        my_data = {}

    values = [file_path]
    for key, value in parameters:
        # Set the value in the dictionary to the corresponding value in `my_data`, or the default value if not found
        if not key in ["ask_for_file", "file_path"]:
            values.append(my_data.get(key, value))
    return tuple(values)


def train_model(
    headless,
    print_only,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    dataset_config,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training_pct,
    min_bucket_reso,
    max_bucket_reso,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    save_state_on_train_end,
    resume,
    prior_loss_weight,
    color_aug,
    flip_aug,
    clip_skip,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    vae,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    output_name,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    mem_eff_attn,
    gradient_accumulation_steps,
    model_list,  # Keep this. Yes, it is unused here but required given the common list used
    token_string,
    init_word,
    num_vectors_per_token,
    max_train_steps,
    weights,
    template,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    v_pred_like_loss,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    lr_scheduler_args,
    lr_scheduler_type,
    noise_offset_type,
    noise_offset,
    noise_offset_random_strength,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    ip_noise_gamma,
    ip_noise_gamma_random_strength,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    loss_type,
    huber_schedule,
    huber_c,
    vae_batch_size,
    min_snr_gamma,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    log_with,
    wandb_api_key,
    wandb_run_name,
    log_tracker_name,
    log_tracker_config,
    log_config,
    scale_v_pred_loss_like_noise_pred,
    disable_mmap_load_safetensors,
    min_timestep,
    max_timestep,
    sdxl_no_half_vae,
    huggingface_repo_id,
    huggingface_token,
    huggingface_repo_type,
    huggingface_repo_visibility,
    huggingface_path_in_repo,
    save_state_to_huggingface,
    resume_from_huggingface,
    async_upload,
    metadata_author,
    metadata_description,
    metadata_license,
    metadata_tags,
    metadata_title,
):
    # Get list of function parameters and values
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

    log.info(f"Start training TI...")

    log.info(f"Validating lr scheduler arguments...")
    if not validate_args_setting(lr_scheduler_args):
        return
    
    log.info(f"Validating optimizer arguments...")
    if not validate_args_setting(optimizer_args):
        return

    #
    # Validate paths
    # 
    
    if not validate_file_path(dataset_config):
        return TRAIN_BUTTON_VISIBLE
    
    if not validate_file_path(log_tracker_config):
        return TRAIN_BUTTON_VISIBLE
    
    if not validate_folder_path(logging_dir, can_be_written_to=True, create_if_not_exists=True):
        return TRAIN_BUTTON_VISIBLE
    
    if not validate_folder_path(output_dir, can_be_written_to=True, create_if_not_exists=True):
        return TRAIN_BUTTON_VISIBLE
    
    if not validate_model_path(pretrained_model_name_or_path):
        return TRAIN_BUTTON_VISIBLE
    
    if not validate_folder_path(reg_data_dir):
        return TRAIN_BUTTON_VISIBLE
    
    if not validate_folder_path(resume):
        return TRAIN_BUTTON_VISIBLE
    
    if not validate_folder_path(train_data_dir):
        return TRAIN_BUTTON_VISIBLE
    
    if not validate_model_path(vae):
        return TRAIN_BUTTON_VISIBLE
    
    #
    # End of path validation
    #

    # if not validate_paths(
    #     dataset_config=dataset_config,
    #     headless=headless,
    #     log_tracker_config=log_tracker_config,
    #     logging_dir=logging_dir,
    #     output_dir=output_dir,
    #     pretrained_model_name_or_path=pretrained_model_name_or_path,
    #     reg_data_dir=reg_data_dir,
    #     resume=resume,
    #     train_data_dir=train_data_dir,
    #     vae=vae,
    # ):
    #     return TRAIN_BUTTON_VISIBLE

    if token_string == "":
        output_message(msg="Token string is missing", headless=headless)
        return TRAIN_BUTTON_VISIBLE

    if init_word == "":
        output_message(msg="Init word is missing", headless=headless)
        return TRAIN_BUTTON_VISIBLE

    if not print_only and check_if_model_exist(
        output_name, output_dir, save_model_as, headless
    ):
        return TRAIN_BUTTON_VISIBLE

    if dataset_config:
        log.info(
            "Dataset config toml file used, skipping total_steps, train_batch_size, gradient_accumulation_steps, epoch, reg_factor, max_train_steps calculations..."
        )
        if max_train_steps > 0:
            # calculate stop encoder training
            if stop_text_encoder_training_pct == 0:
                stop_text_encoder_training = 0
            else:
                stop_text_encoder_training = math.ceil(
                    float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
                )

            if lr_warmup != 0:
                lr_warmup_steps = round(
                    float(int(lr_warmup) * int(max_train_steps) / 100)
                )
            else:
                lr_warmup_steps = 0
        else:
            stop_text_encoder_training = 0
            lr_warmup_steps = 0

        if max_train_steps == 0:
            max_train_steps_info = f"Max train steps: 0. sd-scripts will therefore default to 1600. Please specify a different value if required."
        else:
            max_train_steps_info = f"Max train steps: {max_train_steps}"

    else:
        if train_data_dir == "":
            log.error("Train data dir is empty")
            return TRAIN_BUTTON_VISIBLE

        # Get a list of all subfolders in train_data_dir
        subfolders = [
            f
            for f in os.listdir(train_data_dir)
            if os.path.isdir(os.path.join(train_data_dir, f))
        ]

        total_steps = 0

        # Loop through each subfolder and extract the number of repeats
        for folder in subfolders:
            try:
                # Extract the number of repeats from the folder name
                repeats = int(folder.split("_")[0])
                log.info(f"Folder {folder}: {repeats} repeats found")

                # Count the number of images in the folder
                num_images = len(
                    [
                        f
                        for f, lower_f in (
                            (file, file.lower())
                            for file in os.listdir(os.path.join(train_data_dir, folder))
                        )
                        if lower_f.endswith((".jpg", ".jpeg", ".png", ".webp"))
                    ]
                )

                log.info(f"Folder {folder}: {num_images} images found")

                # Calculate the total number of steps for this folder
                steps = repeats * num_images

                # log.info the result
                log.info(f"Folder {folder}: {num_images} * {repeats} = {steps} steps")

                total_steps += steps

            except ValueError:
                # Handle the case where the folder name does not contain an underscore
                log.info(
                    f"Error: '{folder}' does not contain an underscore, skipping..."
                )

        if reg_data_dir == "":
            reg_factor = 1
        else:
            log.warning(
                "Regularisation images are used... Will double the number of steps required..."
            )
            reg_factor = 2

        log.info(f"Regulatization factor: {reg_factor}")

        if max_train_steps == 0:
            # calculate max_train_steps
            max_train_steps = int(
                math.ceil(
                    float(total_steps)
                    / int(train_batch_size)
                    / int(gradient_accumulation_steps)
                    * int(epoch)
                    * int(reg_factor)
                )
            )
            max_train_steps_info = f"max_train_steps ({total_steps} / {train_batch_size} / {gradient_accumulation_steps} * {epoch} * {reg_factor}) = {max_train_steps}"
        else:
            if max_train_steps == 0:
                max_train_steps_info = f"Max train steps: 0. sd-scripts will therefore default to 1600. Please specify a different value if required."
            else:
                max_train_steps_info = f"Max train steps: {max_train_steps}"

        # calculate stop encoder training
        if stop_text_encoder_training_pct == 0:
            stop_text_encoder_training = 0
        else:
            stop_text_encoder_training = math.ceil(
                float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
            )

        if lr_warmup != 0:
            lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
        else:
            lr_warmup_steps = 0

        log.info(f"Total steps: {total_steps}")

    log.info(f"Train batch size: {train_batch_size}")
    log.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    log.info(f"Epoch: {epoch}")
    log.info(max_train_steps_info)
    log.info(f"stop_text_encoder_training = {stop_text_encoder_training}")
    log.info(f"lr_warmup_steps = {lr_warmup_steps}")

    accelerate_path = get_executable_path("accelerate")
    if accelerate_path == "":
        log.error("accelerate not found")
        return TRAIN_BUTTON_VISIBLE

    run_cmd = [rf'{accelerate_path}', "launch"]

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

    if sdxl:
        run_cmd.append(rf"{scriptdir}/sd-scripts/sdxl_train_textual_inversion.py")
    else:
        run_cmd.append(rf"{scriptdir}/sd-scripts/train_textual_inversion.py")

    if max_data_loader_n_workers == "" or None:
        max_data_loader_n_workers = 0
    else:
        max_data_loader_n_workers = int(max_data_loader_n_workers)

    if max_train_steps == "" or None:
        max_train_steps = 0
    else:
        max_train_steps = int(max_train_steps)

    # def save_huggingface_to_toml(self, toml_file_path: str):
    config_toml_data = {
        # Update the values in the TOML data
        "adaptive_noise_scale": (
            adaptive_noise_scale if adaptive_noise_scale != 0 else None
        ),
        "async_upload": async_upload,
        "bucket_no_upscale": bucket_no_upscale,
        "bucket_reso_steps": bucket_reso_steps,
        "cache_latents": cache_latents,
        "cache_latents_to_disk": cache_latents_to_disk,
        "caption_dropout_every_n_epochs": int(caption_dropout_every_n_epochs),
        "caption_extension": caption_extension,
        "clip_skip": clip_skip if clip_skip != 0 else None,
        "color_aug": color_aug,
        "dataset_config": dataset_config,
        "disable_mmap_load_safetensors": disable_mmap_load_safetensors,
        "dynamo_backend": dynamo_backend,
        "enable_bucket": enable_bucket,
        "epoch": int(epoch),
        "flip_aug": flip_aug,
        "full_fp16": full_fp16,
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "gradient_checkpointing": gradient_checkpointing,
        "huber_c": huber_c,
        "huber_schedule": huber_schedule,
        "huggingface_repo_id": huggingface_repo_id,
        "huggingface_token": huggingface_token,
        "huggingface_repo_type": huggingface_repo_type,
        "huggingface_repo_visibility": huggingface_repo_visibility,
        "huggingface_path_in_repo": huggingface_path_in_repo,
        "init_word": init_word,
        "ip_noise_gamma": ip_noise_gamma if ip_noise_gamma != 0 else None,
        "ip_noise_gamma_random_strength": ip_noise_gamma_random_strength,
        "keep_tokens": int(keep_tokens),
        "learning_rate": learning_rate,
        "logging_dir": logging_dir,
        "log_config": log_config,
        "log_tracker_name": log_tracker_name,
        "log_tracker_config": log_tracker_config,
        "loss_type": loss_type,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_args": str(lr_scheduler_args).replace('"', "").split(),
        "lr_scheduler_num_cycles": (
            int(lr_scheduler_num_cycles) if lr_scheduler_num_cycles != "" else int(epoch)
        ),
        "lr_scheduler_power": lr_scheduler_power,
        "lr_scheduler_type": lr_scheduler_type if lr_scheduler_type != "" else None,
        "lr_warmup_steps": lr_warmup_steps,
        "max_bucket_reso": max_bucket_reso,
        "max_timestep": max_timestep if max_timestep != 0 else None,
        "max_token_length": int(max_token_length),
        "max_train_epochs": int(max_train_epochs) if int(max_train_epochs) != 0 else None,
        "max_train_steps": int(max_train_steps) if int(max_train_steps) != 0 else None,
        "mem_eff_attn": mem_eff_attn,
        "metadata_author": metadata_author,
        "metadata_description": metadata_description,
        "metadata_license": metadata_license,
        "metadata_tags": metadata_tags,
        "metadata_title": metadata_title,
        "min_bucket_reso": int(min_bucket_reso),
        "min_snr_gamma": min_snr_gamma if min_snr_gamma != 0 else None,
        "min_timestep": min_timestep if min_timestep != 0 else None,
        "mixed_precision": mixed_precision,
        "multires_noise_discount": multires_noise_discount,
        "multires_noise_iterations": (
            multires_noise_iterations if multires_noise_iterations != 0 else None
        ),
        "no_half_vae": sdxl_no_half_vae,
        "no_token_padding": no_token_padding,
        "noise_offset": noise_offset if noise_offset != 0 else None,
        "noise_offset_random_strength": noise_offset_random_strength,
        "noise_offset_type": noise_offset_type,
        "num_vectors_per_token": int(num_vectors_per_token),
        "optimizer_type": optimizer,
        "optimizer_args": str(optimizer_args).replace('"', "").split(),
        "output_dir": output_dir,
        "output_name": output_name,
        "persistent_data_loader_workers": int(persistent_data_loader_workers),
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "prior_loss_weight": prior_loss_weight,
        "random_crop": random_crop,
        "reg_data_dir": reg_data_dir,
        "resolution": max_resolution,
        "resume": resume,
        "resume_from_huggingface": resume_from_huggingface,
        "sample_every_n_epochs": (
            sample_every_n_epochs if sample_every_n_epochs != 0 else None
        ),
        "sample_every_n_steps": (
            sample_every_n_steps if sample_every_n_steps != 0 else None
        ),
        "sample_prompts": create_prompt_file(sample_prompts, output_dir),
        "sample_sampler": sample_sampler,
        "save_every_n_epochs": (
            save_every_n_epochs if save_every_n_epochs != 0 else None
        ),
        "save_every_n_steps": save_every_n_steps if save_every_n_steps != 0 else None,
        "save_last_n_steps": save_last_n_steps if save_last_n_steps != 0 else None,
        "save_last_n_steps_state": (
            save_last_n_steps_state if save_last_n_steps_state != 0 else None
        ),
        "save_model_as": save_model_as,
        "save_precision": save_precision,
        "save_state": save_state,
        "save_state_on_train_end": save_state_on_train_end,
        "save_state_to_huggingface": save_state_to_huggingface,
        "scale_v_pred_loss_like_noise_pred": scale_v_pred_loss_like_noise_pred,
        "sdpa": True if xformers == "sdpa" else None,
        "seed": int(seed) if int(seed) != 0 else None,
        "shuffle_caption": shuffle_caption,
        "stop_text_encoder_training": (
            stop_text_encoder_training if stop_text_encoder_training != 0 else None
        ),
        "token_string": token_string,
        "train_batch_size": train_batch_size,
        "train_data_dir": train_data_dir,
        "log_with": log_with,
        "v2": v2,
        "v_parameterization": v_parameterization,
        "v_pred_like_loss": v_pred_like_loss if v_pred_like_loss != 0 else None,
        "vae": vae,
        "vae_batch_size": vae_batch_size if vae_batch_size != 0 else None,
        "wandb_api_key": wandb_api_key,
        "wandb_run_name": wandb_run_name if wandb_run_name != "" else output_name,
        "weigts": weights,
        "use_object_template": True if template == "object template" else None,
        "use_style_template": True if template == "style template" else None,
        "xformers": True if xformers == "xformers" else None,
    }

    # Given dictionary `config_toml_data`
    # Remove all values = ""
    config_toml_data = {
        key: value
        for key, value in config_toml_data.items()
        if value not in ["", False, None]
    }
    
    config_toml_data["max_data_loader_n_workers"] = int(max_data_loader_n_workers)
    
    # Sort the dictionary by keys
    config_toml_data = dict(sorted(config_toml_data.items()))

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
    tmpfilename = fr"{output_dir}/config_textual_inversion-{formatted_datetime}.toml"
    
    # Save the updated TOML data back to the file
    with open(tmpfilename, "w", encoding="utf-8") as toml_file:
        toml.dump(config_toml_data, toml_file)

        if not os.path.exists(toml_file.name):
            log.error(f"Failed to write TOML file: {toml_file.name}")

    run_cmd.append("--config_file")
    run_cmd.append(rf"{tmpfilename}")

    # Initialize a dictionary with always-included keyword arguments
    kwargs_for_training = {
        "additional_parameters": additional_parameters,
    }

    # Pass the dynamically constructed keyword arguments to the function
    run_cmd = run_cmd_advanced_training(run_cmd=run_cmd, **kwargs_for_training)

    if print_only:
        print_command_and_toml(run_cmd, tmpfilename)
    else:
        # Saving config file for model
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
        # config_dir = os.path.dirname(os.path.dirname(train_data_dir))
        file_path = os.path.join(output_dir, f"{output_name}_{formatted_datetime}.json")

        log.info(f"Saving training config to {file_path}...")

        SaveConfigFile(
            parameters=parameters,
            file_path=file_path,
            exclusion=["file_path", "save_as", "headless", "print_only"],
        )

        env = setup_environment()

        # Run the command

        executor.execute_command(run_cmd=run_cmd, env=env)
        
        train_state_value = time.time()

        return (
            gr.Button(visible=False or headless),
            gr.Button(visible=True),
            gr.Textbox(value=train_state_value),
        )


def ti_tab(
    headless=False,
    default_output_dir=None,
    config: KohyaSSGUIConfig = {},
    use_shell_flag: bool = False,
):
    dummy_db_true = gr.Checkbox(value=True, visible=False)
    dummy_db_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    global use_shell
    use_shell = use_shell_flag

    current_embedding_dir = (
        default_output_dir
        if default_output_dir is not None and default_output_dir != ""
        else os.path.join(scriptdir, "outputs")
    )

    with gr.Tab("Training"), gr.Column(variant="compact"):
        gr.Markdown("Train a TI using kohya textual inversion python code...")

        # Setup Configuration Files Gradio
        with gr.Accordion("Configuration", open=False):
            configuration = ConfigurationFile(headless=headless, config=config)

        with gr.Accordion("Accelerate launch", open=False), gr.Column():
            accelerate_launch = AccelerateLaunch(config=config)

        with gr.Column():
            source_model = SourceModel(
                save_model_as_choices=[
                    "ckpt",
                    "safetensors",
                ],
                headless=headless,
                config=config,
            )

        with gr.Accordion("Folders", open=False), gr.Group():
            folders = Folders(headless=headless, config=config)

        with gr.Accordion("Metadata", open=False), gr.Group():
            metadata = MetaData(config=config)

        with gr.Accordion("Dataset Preparation", open=False):
            gr.Markdown(
                "This section provide Dreambooth tools to help setup your dataset..."
            )
            gradio_dreambooth_folder_creation_tab(
                train_data_dir_input=source_model.train_data_dir,
                reg_data_dir_input=folders.reg_data_dir,
                output_dir_input=folders.output_dir,
                logging_dir_input=folders.logging_dir,
                headless=headless,
                config=config,
            )

            gradio_dataset_balancing_tab(headless=headless)

        with gr.Accordion("Parameters", open=False), gr.Column():
            with gr.Accordion("Basic", open="True"):
                with gr.Group(elem_id="basic_tab"):
                    with gr.Row():

                        def list_embedding_files(path):
                            nonlocal current_embedding_dir
                            current_embedding_dir = path
                            return list(
                                list_files(
                                    path,
                                    exts=[".pt", ".ckpt", ".safetensors"],
                                    all=True,
                                )
                            )

                        weights = gr.Dropdown(
                            label="Resume TI training (Optional. Path to existing TI embedding file to keep training)",
                            choices=[""] + list_embedding_files(current_embedding_dir),
                            value="",
                            interactive=True,
                            allow_custom_value=True,
                        )
                        create_refresh_button(
                            weights,
                            lambda: None,
                            lambda: {
                                "choices": list_embedding_files(current_embedding_dir)
                            },
                            "open_folder_small",
                        )
                        weights_file_input = gr.Button(
                            "📂",
                            elem_id="open_folder_small",
                            elem_classes=["tool"],
                            visible=(not headless),
                        )
                        weights_file_input.click(
                            get_file_path,
                            outputs=weights,
                            show_progress=False,
                        )
                        weights.change(
                            fn=lambda path: gr.Dropdown(
                                choices=[""] + list_embedding_files(path)
                            ),
                            inputs=weights,
                            outputs=weights,
                            show_progress=False,
                        )

                    with gr.Row():
                        token_string = gr.Textbox(
                            label="Token string",
                            placeholder="eg: cat",
                        )
                        init_word = gr.Textbox(
                            label="Init word",
                            value="*",
                        )
                        num_vectors_per_token = gr.Slider(
                            minimum=1,
                            maximum=75,
                            value=1,
                            step=1,
                            label="Vectors",
                        )
                        # max_train_steps = gr.Textbox(
                        #     label='Max train steps',
                        #     placeholder='(Optional) Maximum number of steps',
                        # )
                        template = gr.Dropdown(
                            label="Template",
                            choices=[
                                "caption",
                                "object template",
                                "style template",
                            ],
                            value="caption",
                        )
                    basic_training = BasicTraining(
                        learning_rate_value=1e-5,
                        lr_scheduler_value="cosine",
                        lr_warmup_value=10,
                        sdxl_checkbox=source_model.sdxl_checkbox,
                        config=config,
                    )

                    # Add SDXL Parameters
                    sdxl_params = SDXLParameters(
                        source_model.sdxl_checkbox,
                        show_sdxl_cache_text_encoder_outputs=False,
                        config=config,
                    )

            with gr.Accordion("Advanced", open=False, elem_id="advanced_tab"):
                advanced_training = AdvancedTraining(headless=headless, config=config)
                advanced_training.color_aug.change(
                    color_aug_changed,
                    inputs=[advanced_training.color_aug],
                    outputs=[basic_training.cache_latents],
                )

            with gr.Accordion("Samples", open=False, elem_id="samples_tab"):
                sample = SampleImages(config=config)

            global huggingface
            with gr.Accordion("HuggingFace", open=False):
                huggingface = HuggingFace(config=config)

        global executor
        executor = CommandExecutor(headless=headless)
        
        with gr.Column(), gr.Group():
            with gr.Row():
                button_print = gr.Button("Print training command")

        # Setup gradio tensorboard buttons
        TensorboardManager(headless=headless, logging_dir=folders.logging_dir)

        settings_list = [
            source_model.pretrained_model_name_or_path,
            source_model.v2,
            source_model.v_parameterization,
            source_model.sdxl_checkbox,
            folders.logging_dir,
            source_model.train_data_dir,
            folders.reg_data_dir,
            folders.output_dir,
            source_model.dataset_config,
            basic_training.max_resolution,
            basic_training.learning_rate,
            basic_training.lr_scheduler,
            basic_training.lr_warmup,
            basic_training.train_batch_size,
            basic_training.epoch,
            basic_training.save_every_n_epochs,
            accelerate_launch.mixed_precision,
            source_model.save_precision,
            basic_training.seed,
            accelerate_launch.num_cpu_threads_per_process,
            basic_training.cache_latents,
            basic_training.cache_latents_to_disk,
            basic_training.caption_extension,
            basic_training.enable_bucket,
            advanced_training.gradient_checkpointing,
            advanced_training.full_fp16,
            advanced_training.no_token_padding,
            basic_training.stop_text_encoder_training,
            basic_training.min_bucket_reso,
            basic_training.max_bucket_reso,
            advanced_training.xformers,
            source_model.save_model_as,
            advanced_training.shuffle_caption,
            advanced_training.save_state,
            advanced_training.save_state_on_train_end,
            advanced_training.resume,
            advanced_training.prior_loss_weight,
            advanced_training.color_aug,
            advanced_training.flip_aug,
            advanced_training.clip_skip,
            accelerate_launch.num_processes,
            accelerate_launch.num_machines,
            accelerate_launch.multi_gpu,
            accelerate_launch.gpu_ids,
            accelerate_launch.main_process_port,
            advanced_training.vae,
            accelerate_launch.dynamo_backend,
            accelerate_launch.dynamo_mode,
            accelerate_launch.dynamo_use_fullgraph,
            accelerate_launch.dynamo_use_dynamic,
            accelerate_launch.extra_accelerate_launch_args,
            source_model.output_name,
            advanced_training.max_token_length,
            basic_training.max_train_epochs,
            advanced_training.max_data_loader_n_workers,
            advanced_training.mem_eff_attn,
            advanced_training.gradient_accumulation_steps,
            source_model.model_list,
            token_string,
            init_word,
            num_vectors_per_token,
            basic_training.max_train_steps,
            weights,
            template,
            advanced_training.keep_tokens,
            basic_training.lr_scheduler_num_cycles,
            basic_training.lr_scheduler_power,
            advanced_training.persistent_data_loader_workers,
            advanced_training.bucket_no_upscale,
            advanced_training.random_crop,
            advanced_training.bucket_reso_steps,
            advanced_training.v_pred_like_loss,
            advanced_training.caption_dropout_every_n_epochs,
            advanced_training.caption_dropout_rate,
            basic_training.optimizer,
            basic_training.optimizer_args,
            basic_training.lr_scheduler_args,
            basic_training.lr_scheduler_type,
            advanced_training.noise_offset_type,
            advanced_training.noise_offset,
            advanced_training.noise_offset_random_strength,
            advanced_training.adaptive_noise_scale,
            advanced_training.multires_noise_iterations,
            advanced_training.multires_noise_discount,
            advanced_training.ip_noise_gamma,
            advanced_training.ip_noise_gamma_random_strength,
            sample.sample_every_n_steps,
            sample.sample_every_n_epochs,
            sample.sample_sampler,
            sample.sample_prompts,
            advanced_training.additional_parameters,
            advanced_training.loss_type,
            advanced_training.huber_schedule,
            advanced_training.huber_c,
            advanced_training.vae_batch_size,
            advanced_training.min_snr_gamma,
            advanced_training.save_every_n_steps,
            advanced_training.save_last_n_steps,
            advanced_training.save_last_n_steps_state,
            advanced_training.log_with,
            advanced_training.wandb_api_key,
            advanced_training.wandb_run_name,
            advanced_training.log_tracker_name,
            advanced_training.log_tracker_config,
            advanced_training.log_config,
            advanced_training.scale_v_pred_loss_like_noise_pred,
            sdxl_params.disable_mmap_load_safetensors,
            advanced_training.min_timestep,
            advanced_training.max_timestep,
            sdxl_params.sdxl_no_half_vae,
            huggingface.huggingface_repo_id,
            huggingface.huggingface_token,
            huggingface.huggingface_repo_type,
            huggingface.huggingface_repo_visibility,
            huggingface.huggingface_path_in_repo,
            huggingface.save_state_to_huggingface,
            huggingface.resume_from_huggingface,
            huggingface.async_upload,
            metadata.metadata_author,
            metadata.metadata_description,
            metadata.metadata_license,
            metadata.metadata_tags,
            metadata.metadata_title,
        ]

        configuration.button_open_config.click(
            open_configuration,
            inputs=[dummy_db_true, configuration.config_file_name] + settings_list,
            outputs=[configuration.config_file_name] + settings_list,
            show_progress=False,
        )

        configuration.button_load_config.click(
            open_configuration,
            inputs=[dummy_db_false, configuration.config_file_name] + settings_list,
            outputs=[configuration.config_file_name] + settings_list,
            show_progress=False,
        )

        configuration.button_save_config.click(
            save_configuration,
            inputs=[dummy_db_false, configuration.config_file_name] + settings_list,
            outputs=[configuration.config_file_name],
            show_progress=False,
        )
        
        run_state = gr.Textbox(value=train_state_value, visible=False)
            
        run_state.change(
            fn=executor.wait_for_training_to_end,
            outputs=[executor.button_run, executor.button_stop_training],
        )

        executor.button_run.click(
            train_model,
            inputs=[dummy_headless] + [dummy_db_false] + settings_list,
            outputs=[executor.button_run, executor.button_stop_training, run_state],
            show_progress=False,
        )

        executor.button_stop_training.click(
            executor.kill_command, outputs=[executor.button_run, executor.button_stop_training]
        )

        button_print.click(
            train_model,
            inputs=[dummy_headless] + [dummy_db_true] + settings_list,
            show_progress=False,
        )

        return (
            source_model.train_data_dir,
            folders.reg_data_dir,
            folders.output_dir,
            folders.logging_dir,
        )
