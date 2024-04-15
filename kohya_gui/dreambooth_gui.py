import gradio as gr
import json
import math
import os
import time
import sys
import toml
from datetime import datetime
from .common_gui import (
    get_file_path,
    get_saveasfile_path,
    color_aug_changed,
    run_cmd_advanced_training,
    update_my_data,
    check_if_model_exist,
    SaveConfigFile,
    save_to_file,
    scriptdir,
    validate_paths,
)
from .class_accelerate_launch import AccelerateLaunch
from .class_configuration_file import ConfigurationFile
from .class_gui_config import KohyaSSGUIConfig
from .class_source_model import SourceModel
from .class_basic_training import BasicTraining
from .class_advanced_training import AdvancedTraining
from .class_folders import Folders
from .class_command_executor import CommandExecutor
from .class_huggingface import HuggingFace

from .dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from .dataset_balancing_gui import gradio_dataset_balancing_tab
from .class_sample_images import SampleImages, run_cmd_sample, create_prompt_file
from .class_tensorboard import TensorboardManager

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = CommandExecutor()

# Setup huggingface
huggingface = None

PYTHON = sys.executable

TRAIN_BUTTON_VISIBLE = [gr.Button(visible=True), gr.Button(visible=False)]


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
    learning_rate_te,
    learning_rate_te1,
    learning_rate_te2,
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
    full_bf16,
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
    masked_loss,
    clip_skip,
    vae,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    output_name,
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    mem_eff_attn,
    gradient_accumulation_steps,
    model_list,
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
    weighted_captions,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    wandb_run_name,
    log_tracker_name,
    log_tracker_config,
    scale_v_pred_loss_like_noise_pred,
    min_timestep,
    max_timestep,
    debiased_estimation_loss,
    extra_accelerate_launch_args,
    huggingface_repo_id,
    huggingface_token,
    huggingface_repo_type,
    huggingface_repo_visibility,
    huggingface_path_in_repo,
    save_state_to_huggingface,
    resume_from_huggingface,
    async_upload,
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
    learning_rate_te,
    learning_rate_te1,
    learning_rate_te2,
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
    full_bf16,
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
    masked_loss,
    clip_skip,
    vae,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    output_name,
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    mem_eff_attn,
    gradient_accumulation_steps,
    model_list,
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
    weighted_captions,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    wandb_run_name,
    log_tracker_name,
    log_tracker_config,
    scale_v_pred_loss_like_noise_pred,
    min_timestep,
    max_timestep,
    debiased_estimation_loss,
    extra_accelerate_launch_args,
    huggingface_repo_id,
    huggingface_token,
    huggingface_repo_type,
    huggingface_repo_visibility,
    huggingface_path_in_repo,
    save_state_to_huggingface,
    resume_from_huggingface,
    async_upload,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    original_file_path = file_path

    if ask_for_file:
        file_path = get_file_path(file_path)

    if not file_path == "" and not file_path == None:
        # load variables from JSON file
        with open(file_path, "r") as f:
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
    learning_rate_te,
    learning_rate_te1,
    learning_rate_te2,
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
    full_bf16,
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
    masked_loss,
    clip_skip,
    vae,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    output_name,
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    mem_eff_attn,
    gradient_accumulation_steps,
    model_list,  # Keep this. Yes, it is unused here but required given the common list used
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
    weighted_captions,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    wandb_run_name,
    log_tracker_name,
    log_tracker_config,
    scale_v_pred_loss_like_noise_pred,
    min_timestep,
    max_timestep,
    debiased_estimation_loss,
    extra_accelerate_launch_args,
    huggingface_repo_id,
    huggingface_token,
    huggingface_repo_type,
    huggingface_repo_visibility,
    huggingface_path_in_repo,
    save_state_to_huggingface,
    resume_from_huggingface,
    async_upload,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    log.info(f"Start training Dreambooth...")

    # This function validates files or folder paths. Simply add new variables containing file of folder path
    # to validate below
    if not validate_paths(
        output_dir=output_dir,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        train_data_dir=train_data_dir,
        reg_data_dir=reg_data_dir,
        headless=headless,
        logging_dir=logging_dir,
        log_tracker_config=log_tracker_config,
        resume=resume,
        vae=vae,
        dataset_config=dataset_config,
    ):
        return TRAIN_BUTTON_VISIBLE

    if not print_only and check_if_model_exist(
        output_name, output_dir, save_model_as, headless=headless
    ):
        return TRAIN_BUTTON_VISIBLE

    try:
        max_train_steps = int(max_train_steps)
    except ValueError:
        max_train_steps = 0

    if dataset_config:
        log.info(
            "Dataset config toml file used, skipping total_steps, train_batch_size, gradient_accumulation_steps, epoch, reg_factor, max_train_steps calculations..."
        )
    else:
        if train_data_dir == "":
            log.error("Train data dir is empty")
            return TRAIN_BUTTON_VISIBLE

        # Get a list of all subfolders in train_data_dir, excluding hidden folders
        subfolders = [
            f
            for f in os.listdir(train_data_dir)
            if os.path.isdir(os.path.join(train_data_dir, f)) and not f.startswith(".")
        ]

        # Check if subfolders are present. If not let the user know and return
        if not subfolders:
            log.info(f"No {subfolders} were found in train_data_dir can't train...")
            return TRAIN_BUTTON_VISIBLE

        total_steps = 0

        # Loop through each subfolder and extract the number of repeats
        for folder in subfolders:
            # Extract the number of repeats from the folder name
            try:
                repeats = int(folder.split("_")[0])
            except ValueError:
                log.info(
                    f"Subfolder {folder} does not have a proper repeat value, please correct the name or remove it... can't train..."
                )
                continue

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

            if num_images == 0:
                log.info(f"{folder} folder contain no images, skipping...")
            else:
                # Calculate the total number of steps for this folder
                steps = repeats * num_images
                total_steps += steps

                # Print the result
                log.info(f"Folder {folder} : steps {steps}")

        if total_steps == 0:
            log.info(
                f"No images were found in folder {train_data_dir}... please rectify!"
            )
            return TRAIN_BUTTON_VISIBLE

        # Print the result
        # log.info(f"{total_steps} total steps")

        if reg_data_dir == "":
            reg_factor = 1
        else:
            log.info(
                f"Regularisation images are used... Will double the number of steps required..."
            )
            reg_factor = 2

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
            log.info(
                f"max_train_steps ({total_steps} / {train_batch_size} / {gradient_accumulation_steps} * {epoch} * {reg_factor}) = {max_train_steps}"
            )

    # calculate stop encoder training
    if stop_text_encoder_training > 0:
        if max_train_steps != 0:
            stop_text_encoder_training = int(
                math.ceil(float(max_train_steps) / 100 * int(stop_text_encoder_training))
            )
        else:
            stop_text_encoder_training = 0
            log.warning("Can't use stop text encoder training without max_train_steps... setting to 0...")
            
    log.info(f"stop_text_encoder_training = {stop_text_encoder_training}")

    if not max_train_steps == "":
        lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    else:
        lr_warmup_steps = 0
    log.info(f"lr_warmup_steps = {lr_warmup_steps}")

    run_cmd = ["accelerate", "launch"]

    run_cmd = AccelerateLaunch.run_cmd(
        run_cmd=run_cmd,
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
        run_cmd.append(f"{scriptdir}/sd-scripts/sdxl_train.py")
    else:
        run_cmd.append(f"{scriptdir}/sd-scripts/train_db.py")

    # Initialize a dictionary with always-included keyword arguments
    kwargs_for_training = {
        # "adaptive_noise_scale": adaptive_noise_scale,
        # "bucket_no_upscale": bucket_no_upscale,
        # "bucket_reso_steps": bucket_reso_steps,
        # "cache_latents": cache_latents,
        # "cache_latents_to_disk": cache_latents_to_disk,
        # "caption_dropout_every_n_epochs": caption_dropout_every_n_epochs,
        # "caption_dropout_rate": caption_dropout_rate,
        # "caption_extension": caption_extension,
        # "clip_skip": clip_skip,
        # "color_aug": color_aug,
        # "dataset_config": dataset_config,
        # "debiased_estimation_loss": debiased_estimation_loss,
        # "enable_bucket": enable_bucket,
        # "epoch": epoch,
        # "flip_aug": flip_aug,
        # "masked_loss": masked_loss,
        # "full_bf16": full_bf16,
        # "full_fp16": full_fp16,
        # "gradient_accumulation_steps": gradient_accumulation_steps,
        # "gradient_checkpointing": gradient_checkpointing,
        # "ip_noise_gamma": ip_noise_gamma,
        # "ip_noise_gamma_random_strength": ip_noise_gamma_random_strength,
        # "keep_tokens": keep_tokens,
        # "learning_rate": learning_rate,
        # "logging_dir": logging_dir,
        # "log_tracker_name": log_tracker_name,
        # "log_tracker_config": log_tracker_config,
        # "lr_scheduler": lr_scheduler,
        # "lr_scheduler_args": lr_scheduler_args,
        # "lr_scheduler_num_cycles": lr_scheduler_num_cycles,
        # "lr_scheduler_power": lr_scheduler_power,
        # "lr_warmup_steps": lr_warmup_steps,
        # "max_bucket_reso": max_bucket_reso,
        "max_data_loader_n_workers": max_data_loader_n_workers,
        # "max_resolution": max_resolution,
        # "max_timestep": max_timestep,
        # "max_token_length": max_token_length,
        # "max_train_epochs": max_train_epochs,
        # "max_train_steps": max_train_steps,
        # "mem_eff_attn": mem_eff_attn,
        # "min_bucket_reso": min_bucket_reso,
        # "min_snr_gamma": min_snr_gamma,
        # "min_timestep": min_timestep,
        # "mixed_precision": mixed_precision,
        # "multires_noise_discount": multires_noise_discount,
        # "multires_noise_iterations": multires_noise_iterations,
        # "no_token_padding": no_token_padding,
        # "noise_offset": noise_offset,
        # "noise_offset_random_strength": noise_offset_random_strength,
        # "noise_offset_type": noise_offset_type,
        # "optimizer": optimizer,
        # "optimizer_args": optimizer_args,
        # "output_dir": output_dir,
        # "output_name": output_name,
        # "persistent_data_loader_workers": persistent_data_loader_workers,
        # "pretrained_model_name_or_path": pretrained_model_name_or_path,
        # "prior_loss_weight": prior_loss_weight,
        # "random_crop": random_crop,
        # "reg_data_dir": reg_data_dir,
        # "resume": resume,
        # "save_every_n_epochs": save_every_n_epochs,
        # "save_every_n_steps": save_every_n_steps,
        # "save_last_n_steps": save_last_n_steps,
        # "save_last_n_steps_state": save_last_n_steps_state,
        # "save_model_as": save_model_as,
        # "save_precision": save_precision,
        # "save_state": save_state,
        # "save_state_on_train_end": save_state_on_train_end,
        # "scale_v_pred_loss_like_noise_pred": scale_v_pred_loss_like_noise_pred,
        # "seed": seed,
        # "shuffle_caption": shuffle_caption,
        # "stop_text_encoder_training": stop_text_encoder_training,
        # "train_batch_size": train_batch_size,
        # "train_data_dir": train_data_dir,
        # "use_wandb": use_wandb,
        # "v2": v2,
        # "v_parameterization": v_parameterization,
        # "v_pred_like_loss": v_pred_like_loss,
        # "vae": vae,
        # "vae_batch_size": vae_batch_size,
        # "wandb_api_key": wandb_api_key,
        # "wandb_run_name": wandb_run_name,
        # "weighted_captions": weighted_captions,
        # "xformers": xformers,
        "additional_parameters": additional_parameters,
        # "loss_type": loss_type,
        # "huber_schedule": huber_schedule,
        # "huber_c": huber_c,
    }

    # Conditionally include specific keyword arguments based on sdxl
    # if sdxl:
    #     kwargs_for_training["learning_rate_te1"] = learning_rate_te1
    #     kwargs_for_training["learning_rate_te2"] = learning_rate_te2
    # else:
    #     kwargs_for_training["learning_rate_te"] = learning_rate_te

    # Pass the dynamically constructed keyword arguments to the function
    run_cmd = run_cmd_advanced_training(run_cmd=run_cmd, **kwargs_for_training)

    # run_cmd = run_cmd_sample(
    #     run_cmd,
    #     sample_every_n_steps,
    #     sample_every_n_epochs,
    #     sample_sampler,
    #     sample_prompts,
    #     output_dir,
    # )

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
        "huggingface_repo_id": huggingface_repo_id,
        "huggingface_token": huggingface_token,
        "huggingface_repo_type": huggingface_repo_type,
        "huggingface_repo_visibility": huggingface_repo_visibility,
        "huggingface_path_in_repo": huggingface_path_in_repo,
        "save_state_to_huggingface": save_state_to_huggingface,
        "resume_from_huggingface": resume_from_huggingface,
        "async_upload": async_upload,
        "adaptive_noise_scale": adaptive_noise_scale,
        "bucket_no_upscale": bucket_no_upscale,
        "bucket_reso_steps": bucket_reso_steps,
        "cache_latents": cache_latents,
        "cache_latents_to_disk": cache_latents_to_disk,
        "caption_dropout_every_n_epochs": caption_dropout_every_n_epochs,
        "caption_dropout_rate": caption_dropout_rate,
        "caption_extension": caption_extension,
        "clip_skip": int(clip_skip),
        "color_aug": color_aug,
        "dataset_config": dataset_config,
        "debiased_estimation_loss": debiased_estimation_loss,
        "enable_bucket": enable_bucket,
        "epoch": int(epoch),
        "flip_aug": flip_aug,
        "masked_loss": masked_loss,
        "full_bf16": full_bf16,
        "full_fp16": full_fp16,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "huber_c": huber_c,
        "huber_schedule": huber_schedule,
        "ip_noise_gamma": ip_noise_gamma,
        "ip_noise_gamma_random_strength": ip_noise_gamma_random_strength,
        "keep_tokens": int(keep_tokens),
        "learning_rate": learning_rate,
        "learning_rate_te": learning_rate_te,
        "learning_rate_te1": learning_rate_te1,
        "learning_rate_te2": learning_rate_te2,
        "logging_dir": logging_dir,
        "log_tracker_name": log_tracker_name,
        "log_tracker_config": log_tracker_config,
        "loss_type": loss_type,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_args": str(lr_scheduler_args).replace('"', "").split(),
        "lr_scheduler_num_cycles": (
            lr_scheduler_num_cycles if lr_scheduler_num_cycles != "" else int(epoch)
        ),
        "lr_scheduler_power": lr_scheduler_power,
        "lr_warmup_steps": lr_warmup_steps,
        "max_bucket_reso": max_bucket_reso,
        "max_data_loader_n_workers": max_data_loader_n_workers,
        "max_timestep": max_timestep,
        "max_token_length": int(max_token_length),
        "max_train_epochs": max_train_epochs,
        "max_train_steps": int(max_train_steps),
        "mem_eff_attn": mem_eff_attn,
        "min_bucket_reso": int(min_bucket_reso),
        "min_snr_gamma": min_snr_gamma,
        "min_timestep": int(min_timestep),
        "mixed_precision": mixed_precision,
        "multires_noise_discount": multires_noise_discount,
        "multires_noise_iterations": multires_noise_iterations,
        "no_token_padding": no_token_padding,
        "noise_offset": noise_offset,
        "noise_offset_random_strength": noise_offset_random_strength,
        "noise_offset_type": noise_offset_type,
        "optimizer_type": optimizer,
        "optimizer_args": (
            str(optimizer_args).replace('"', "").split()
            if optimizer_args != ""
            else None
        ),
        "output_dir": output_dir,
        "output_name": output_name,
        "persistent_data_loader_workers": persistent_data_loader_workers,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "prior_loss_weight": prior_loss_weight,
        "random_crop": random_crop,
        "reg_data_dir": reg_data_dir,
        "resolution": max_resolution,
        "resume": resume,
        "sample_every_n_epochs": sample_every_n_epochs,
        "sample_every_n_steps": sample_every_n_steps,
        "sample_prompts": create_prompt_file(output_dir, output_dir),
        "sample_sampler": sample_sampler,
        "save_every_n_epochs": save_every_n_epochs,
        "save_every_n_steps": save_every_n_steps,
        "save_last_n_steps": save_last_n_steps,
        "save_last_n_steps_state": save_last_n_steps_state,
        "save_model_as": save_model_as,
        "save_precision": save_precision,
        "save_state": save_state,
        "save_state_on_train_end": save_state_on_train_end,
        "scale_v_pred_loss_like_noise_pred": scale_v_pred_loss_like_noise_pred,
        "sdpa": True if xformers == "sdpa" else None,
        "seed": int(seed),
        "shuffle_caption": shuffle_caption,
        "stop_text_encoder_training": stop_text_encoder_training,
        "train_batch_size": train_batch_size,
        "train_data_dir": train_data_dir,
        "use_wandb": use_wandb,
        "v2": v2,
        "v_parameterization": v_parameterization,
        "v_pred_like_loss": v_pred_like_loss,
        "vae": vae,
        "vae_batch_size": vae_batch_size,
        "wandb_api_key": wandb_api_key,
        "wandb_run_name": wandb_run_name,
        "weighted_captions": weighted_captions,
        "xformers": True if xformers == "xformers" else None,
    }

    # Given dictionary `config_toml_data`
    # Remove all values = ""
    config_toml_data = {
        key: value
        for key, value in config_toml_data.items()
        if value != "" and value != False
    }

    tmpfilename = "./outputs/tmpfiledbooth.toml"
    # Save the updated TOML data back to the file
    with open(tmpfilename, "w") as toml_file:
        toml.dump(config_toml_data, toml_file)

        if not os.path.exists(toml_file.name):
            log.error(f"Failed to write TOML file: {toml_file.name}")

        toml_file_path = (
            os.path.abspath(os.path.normpath(toml_file.name)).replace("\\", "/")
            if os.name == "nt"
            else toml_file.name
        )

    run_cmd.append(f"--config_file")
    run_cmd.append(rf"{toml_file_path}")

    if print_only:
        log.warning(
            "Here is the trainer command as a reference. It will not be executed:\n"
        )
        # Reconstruct the safe command string for display
        command_to_run = " ".join(run_cmd)

        print(command_to_run)

        save_to_file(command_to_run)
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

        # log.info(run_cmd)

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            rf"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"
        )
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"

        # Run the command

        executor.execute_command(run_cmd=run_cmd, env=env)

        return (
            gr.Button(visible=False),
            gr.Button(visible=True),
            gr.Textbox(value=time.time()),
        )


def dreambooth_tab(
    # train_data_dir=gr.Textbox(),
    # reg_data_dir=gr.Textbox(),
    # output_dir=gr.Textbox(),
    # logging_dir=gr.Textbox(),
    headless=False,
    config: KohyaSSGUIConfig = {},
):
    dummy_db_true = gr.Checkbox(value=True, visible=False)
    dummy_db_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    with gr.Tab("Training"), gr.Column(variant="compact"):
        gr.Markdown("Train a custom model using kohya dreambooth python code...")

        # Setup Configuration Files Gradio
        with gr.Accordion("Configuration", open=False):
            configuration = ConfigurationFile(headless=headless, config=config)

        with gr.Accordion("Accelerate launch", open=False), gr.Column():
            accelerate_launch = AccelerateLaunch(config=config)

        with gr.Column():
            source_model = SourceModel(headless=headless, config=config)

        with gr.Accordion("Folders", open=False), gr.Group():
            folders = Folders(headless=headless, config=config)

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
                    basic_training = BasicTraining(
                        learning_rate_value=1e-5,
                        lr_scheduler_value="cosine",
                        lr_warmup_value=10,
                        dreambooth=True,
                        sdxl_checkbox=source_model.sdxl_checkbox,
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

        with gr.Column(), gr.Group():
            with gr.Row():
                button_run = gr.Button("Start training", variant="primary")

                button_stop_training = gr.Button(
                    "Stop training", visible=False, variant="stop"
                )

        with gr.Column(), gr.Group():
            with gr.Row():
                button_print = gr.Button("Print training command")

        # Setup gradio tensorboard buttons
        with gr.Column(), gr.Group():
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
            basic_training.learning_rate_te,
            basic_training.learning_rate_te1,
            basic_training.learning_rate_te2,
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
            advanced_training.full_bf16,
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
            advanced_training.masked_loss,
            advanced_training.clip_skip,
            advanced_training.vae,
            accelerate_launch.num_processes,
            accelerate_launch.num_machines,
            accelerate_launch.multi_gpu,
            accelerate_launch.gpu_ids,
            accelerate_launch.main_process_port,
            source_model.output_name,
            advanced_training.max_token_length,
            basic_training.max_train_epochs,
            basic_training.max_train_steps,
            advanced_training.max_data_loader_n_workers,
            advanced_training.mem_eff_attn,
            advanced_training.gradient_accumulation_steps,
            source_model.model_list,
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
            advanced_training.weighted_captions,
            advanced_training.save_every_n_steps,
            advanced_training.save_last_n_steps,
            advanced_training.save_last_n_steps_state,
            advanced_training.use_wandb,
            advanced_training.wandb_api_key,
            advanced_training.wandb_run_name,
            advanced_training.log_tracker_name,
            advanced_training.log_tracker_config,
            advanced_training.scale_v_pred_loss_like_noise_pred,
            advanced_training.min_timestep,
            advanced_training.max_timestep,
            advanced_training.debiased_estimation_loss,
            accelerate_launch.extra_accelerate_launch_args,
            huggingface.huggingface_repo_id,
            huggingface.huggingface_token,
            huggingface.huggingface_repo_type,
            huggingface.huggingface_repo_visibility,
            huggingface.huggingface_path_in_repo,
            huggingface.save_state_to_huggingface,
            huggingface.resume_from_huggingface,
            huggingface.async_upload,
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

        # config.button_save_as_config.click(
        #    save_configuration,
        #    inputs=[dummy_db_true, config.config_file_name] + settings_list,
        #    outputs=[config.config_file_name],
        #    show_progress=False,
        # )

        # def wait_for_training_to_end():
        #     while executor.is_running():
        #         time.sleep(1)
        #         log.debug("Waiting for training to end...")
        #     log.info("Training has ended.")
        #     return gr.Button(visible=True), gr.Button(visible=False)

        # Hidden textbox used to run the wait_for_training_to_end function to hide stop and show start at the end of the training
        run_state = gr.Textbox(value="", visible=False)
        run_state.change(
            fn=executor.wait_for_training_to_end,
            outputs=[button_run, button_stop_training],
        )

        button_run.click(
            train_model,
            inputs=[dummy_headless] + [dummy_db_false] + settings_list,
            outputs=[button_run, button_stop_training, run_state],
            show_progress=False,
        )

        button_stop_training.click(
            executor.kill_command, outputs=[button_run, button_stop_training]
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
