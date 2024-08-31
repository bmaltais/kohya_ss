import gradio as gr
import json
import math
import os
import time
import toml

from datetime import datetime
from .common_gui import (
    check_if_model_exist,
    color_aug_changed,
    get_any_file_path,
    get_executable_path,
    get_file_path,
    get_saveasfile_path,
    output_message,
    print_command_and_toml,
    run_cmd_advanced_training,
    SaveConfigFile,
    scriptdir,
    update_my_data,
    validate_file_path,
    validate_folder_path,
    validate_model_path,
    validate_toml_file,
    validate_args_setting,
    setup_environment,
)
from .class_accelerate_launch import AccelerateLaunch
from .class_configuration_file import ConfigurationFile
from .class_source_model import SourceModel
from .class_basic_training import BasicTraining
from .class_advanced_training import AdvancedTraining
from .class_sdxl_parameters import SDXLParameters
from .class_folders import Folders
from .class_command_executor import CommandExecutor
from .class_tensorboard import TensorboardManager
from .class_sample_images import SampleImages, create_prompt_file
from .class_lora_tab import LoRATools
from .class_huggingface import HuggingFace
from .class_metadata import MetaData
from .class_gui_config import KohyaSSGUIConfig
from .class_flux1 import flux1Training

from .dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from .dataset_balancing_gui import gradio_dataset_balancing_tab

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = None

# Setup huggingface
huggingface = None
use_shell = False
train_state_value = time.time()

document_symbol = "\U0001F4C4"  # 📄


presets_dir = rf"{scriptdir}/presets"

LYCORIS_PRESETS_CHOICES = [
    "attn-mlp",
    "attn-only",
    "full",
    "full-lin",
    "unet-transformer-only",
    "unet-convblock-only",
]


def save_configuration(
    save_as_bool,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    flux1_checkbox,
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
    fp8_base,
    fp8_base_unet,
    full_fp16,
    highvram,
    lowvram,
    stop_text_encoder_training,
    min_bucket_reso,
    max_bucket_reso,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    save_state_on_train_end,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    network_weights,
    dim_from_weights,
    color_aug,
    flip_aug,
    masked_loss,
    clip_skip,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    network_alpha,
    training_comment,
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
    max_grad_norm,
    noise_offset_type,
    noise_offset,
    noise_offset_random_strength,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    ip_noise_gamma,
    ip_noise_gamma_random_strength,
    LoRA_type,
    factor,
    bypass_mode,
    dora_wd,
    use_cp,
    use_tucker,
    use_scalar,
    rank_dropout_scale,
    constrain,
    rescaled,
    train_norm,
    decompose_both,
    train_on_input,
    conv_dim,
    conv_alpha,
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
    down_lr_weight,
    mid_lr_weight,
    up_lr_weight,
    block_lr_zero_threshold,
    block_dims,
    block_alphas,
    conv_block_dims,
    conv_block_alphas,
    weighted_captions,
    unit,
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
    scale_weight_norms,
    network_dropout,
    rank_dropout,
    module_dropout,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    full_bf16,
    min_timestep,
    max_timestep,
    vae,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    LyCORIS_preset,
    debiased_estimation_loss,
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
    loraplus_lr_ratio,
    loraplus_text_encoder_lr_ratio,
    loraplus_unet_lr_ratio,
    # Flux1
    flux1_cache_text_encoder_outputs,
    flux1_cache_text_encoder_outputs_to_disk,
    ae,
    clip_l,
    t5xxl,
    discrete_flow_shift,
    model_prediction_type,
    timestep_sampling,
    split_mode,
    train_blocks,
    t5xxl_max_token_length,
    guidance_scale,
    mem_eff_save,
    apply_t5_attn_mask,
    split_qkv,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    original_file_path = file_path

    # If saving as a new file, get the file path for saving
    if save_as_bool:
        log.info("Save as...")
        file_path = get_saveasfile_path(file_path)
    # If not saving as a new file, check if a file path was provided
    else:
        log.info("Save...")
        # If no file path was provided, get the file path for saving
        if file_path == None or file_path == "":
            file_path = get_saveasfile_path(file_path)

    # Log the file path for debugging purposes
    log.debug(file_path)

    # If no file path was provided, return the original file path
    if file_path == None or file_path == "":
        return original_file_path  # In case a file_path was provided and the user decide to cancel the open action

    # Extract the destination directory from the file path
    destination_directory = os.path.dirname(file_path)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Save the configuration file
    SaveConfigFile(
        parameters=parameters,
        file_path=file_path,
        exclusion=["file_path", "save_as"],
    )

    # Return the file path of the saved configuration
    return file_path


def open_configuration(
    ask_for_file,
    apply_preset,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    flux1_checkbox,
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
    fp8_base,
    fp8_base_unet,
    full_fp16,
    highvram,
    lowvram,
    stop_text_encoder_training,
    min_bucket_reso,
    max_bucket_reso,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    save_state_on_train_end,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    network_weights,
    dim_from_weights,
    color_aug,
    flip_aug,
    masked_loss,
    clip_skip,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    network_alpha,
    training_comment,
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
    max_grad_norm,
    noise_offset_type,
    noise_offset,
    noise_offset_random_strength,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    ip_noise_gamma,
    ip_noise_gamma_random_strength,
    LoRA_type,
    factor,
    bypass_mode,
    dora_wd,
    use_cp,
    use_tucker,
    use_scalar,
    rank_dropout_scale,
    constrain,
    rescaled,
    train_norm,
    decompose_both,
    train_on_input,
    conv_dim,
    conv_alpha,
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
    down_lr_weight,
    mid_lr_weight,
    up_lr_weight,
    block_lr_zero_threshold,
    block_dims,
    block_alphas,
    conv_block_dims,
    conv_block_alphas,
    weighted_captions,
    unit,
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
    scale_weight_norms,
    network_dropout,
    rank_dropout,
    module_dropout,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    full_bf16,
    min_timestep,
    max_timestep,
    vae,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    LyCORIS_preset,
    debiased_estimation_loss,
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
    loraplus_lr_ratio,
    loraplus_text_encoder_lr_ratio,
    loraplus_unet_lr_ratio,
    flux1_cache_text_encoder_outputs,
    flux1_cache_text_encoder_outputs_to_disk,
    ae,
    clip_l,
    t5xxl,
    discrete_flow_shift,
    model_prediction_type,
    timestep_sampling,
    split_mode,
    train_blocks,
    t5xxl_max_token_length,
    guidance_scale,
    mem_eff_save,
    apply_t5_attn_mask,
    split_qkv,
    training_preset,
):
    # Get list of function parameters and their values
    parameters = list(locals().items())

    # Determine if a preset configuration is being applied
    if apply_preset:
        if training_preset != "none":
            log.info(f"Applying preset {training_preset}...")
            file_path = rf"{presets_dir}/lora/{training_preset}.json"
    else:
        # If not applying a preset, set the `training_preset` field to an empty string
        # Find the index of the `training_preset` parameter using the `index()` method
        training_preset_index = parameters.index(("training_preset", training_preset))

        # Update the value of `training_preset` by directly assigning an empty string value
        parameters[training_preset_index] = ("training_preset", "none")

    # Store the original file path for potential reuse
    original_file_path = file_path

    # Request a file path from the user if required
    if ask_for_file:
        file_path = get_file_path(file_path)

    # Proceed if the file path is valid (not empty or None)
    if not file_path == "" and not file_path == None:
        # Check if the file exists before opening it
        if not os.path.isfile(file_path):
            log.error(f"Config file {file_path} does not exist.")
            return

        # Load variables from JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            my_data = json.load(f)
            log.info("Loading config...")

            # Update values to fix deprecated options, set appropriate optimizer if it is set to True, etc.
            my_data = update_my_data(my_data)
    else:
        # Reset the file path to the original if the operation was cancelled or invalid
        file_path = original_file_path  # In case a file_path was provided and the user decides to cancel the open action
        my_data = {}  # Initialize an empty dict if no data was loaded

    values = [file_path]
    # Iterate over parameters to set their values from `my_data` or use default if not found
    for key, value in parameters:
        if not key in ["ask_for_file", "apply_preset", "file_path"]:
            json_value = my_data.get(key)
            # Append the value from JSON if present; otherwise, use the parameter's default value
            values.append(json_value if json_value is not None else value)

    # Display LoCon parameters based on the 'LoRA_type' from the loaded data
    # This section dynamically adjusts visibility of certain parameters in the UI
    if my_data.get("LoRA_type", "Standard") in {
        "Flux1",
        "LoCon",
        "Kohya DyLoRA",
        "Kohya LoCon",
        "LoRA-FA",
        "LyCORIS/Diag-OFT",
        "LyCORIS/DyLoRA",
        "LyCORIS/LoHa",
        "LyCORIS/LoKr",
        "LyCORIS/LoCon",
        "LyCORIS/GLoRA",
    }:
        values.append(gr.Row(visible=True))
    else:
        values.append(gr.Row(visible=False))

    return tuple(values)


def train_model(
    headless,
    print_only,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    flux1_checkbox,
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
    fp8_base,
    fp8_base_unet,
    full_fp16,
    highvram,
    lowvram,
    stop_text_encoder_training_pct,
    min_bucket_reso,
    max_bucket_reso,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    save_state_on_train_end,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    network_weights,
    dim_from_weights,
    color_aug,
    flip_aug,
    masked_loss,
    clip_skip,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,  # Keep this. Yes, it is unused here but required given the common list used
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    network_alpha,
    training_comment,
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
    max_grad_norm,
    noise_offset_type,
    noise_offset,
    noise_offset_random_strength,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    ip_noise_gamma,
    ip_noise_gamma_random_strength,
    LoRA_type,
    factor,
    bypass_mode,
    dora_wd,
    use_cp,
    use_tucker,
    use_scalar,
    rank_dropout_scale,
    constrain,
    rescaled,
    train_norm,
    decompose_both,
    train_on_input,
    conv_dim,
    conv_alpha,
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
    down_lr_weight,
    mid_lr_weight,
    up_lr_weight,
    block_lr_zero_threshold,
    block_dims,
    block_alphas,
    conv_block_dims,
    conv_block_alphas,
    weighted_captions,
    unit,
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
    scale_weight_norms,
    network_dropout,
    rank_dropout,
    module_dropout,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    full_bf16,
    min_timestep,
    max_timestep,
    vae,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    LyCORIS_preset,
    debiased_estimation_loss,
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
    loraplus_lr_ratio,
    loraplus_text_encoder_lr_ratio,
    loraplus_unet_lr_ratio,
    flux1_cache_text_encoder_outputs,
    flux1_cache_text_encoder_outputs_to_disk,
    ae,
    clip_l,
    t5xxl,
    discrete_flow_shift,
    model_prediction_type,
    timestep_sampling,
    split_mode,
    train_blocks,
    t5xxl_max_token_length,
    guidance_scale,
    mem_eff_save,
    apply_t5_attn_mask,
    split_qkv,
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

    log.info(f"Start training LoRA {LoRA_type} ...")

    log.info(f"Validating lr scheduler arguments...")
    if not validate_args_setting(lr_scheduler_args):
        return TRAIN_BUTTON_VISIBLE

    log.info(f"Validating optimizer arguments...")
    if not validate_args_setting(optimizer_args):
        return TRAIN_BUTTON_VISIBLE

    #
    # Validate paths
    #

    if not validate_file_path(dataset_config):
        return TRAIN_BUTTON_VISIBLE

    if not validate_file_path(log_tracker_config):
        return TRAIN_BUTTON_VISIBLE

    if not validate_folder_path(
        logging_dir, can_be_written_to=True, create_if_not_exists=True
    ):
        return TRAIN_BUTTON_VISIBLE

    if LyCORIS_preset not in LYCORIS_PRESETS_CHOICES:
        if not validate_toml_file(LyCORIS_preset):
            return TRAIN_BUTTON_VISIBLE

    if not validate_file_path(network_weights):
        return TRAIN_BUTTON_VISIBLE

    if not validate_folder_path(
        output_dir, can_be_written_to=True, create_if_not_exists=True
    ):
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
    #     network_weights=network_weights,
    #     output_dir=output_dir,
    #     pretrained_model_name_or_path=pretrained_model_name_or_path,
    #     reg_data_dir=reg_data_dir,
    #     resume=resume,
    #     train_data_dir=train_data_dir,
    #     vae=vae,
    # ):
    #     return TRAIN_BUTTON_VISIBLE

    if int(bucket_reso_steps) < 1:
        output_message(
            msg="Bucket resolution steps need to be greater than 0",
            headless=headless,
        )
        return TRAIN_BUTTON_VISIBLE

    # if noise_offset == "":
    #     noise_offset = 0

    if float(noise_offset) > 1 or float(noise_offset) < 0:
        output_message(
            msg="Noise offset need to be a value between 0 and 1",
            headless=headless,
        )
        return TRAIN_BUTTON_VISIBLE

    if output_dir != "":
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if stop_text_encoder_training_pct > 0:
        output_message(
            msg='Output "stop text encoder training" is not yet supported. Ignoring',
            headless=headless,
        )
        stop_text_encoder_training_pct = 0

    if not print_only and check_if_model_exist(
        output_name, output_dir, save_model_as, headless=headless
    ):
        return TRAIN_BUTTON_VISIBLE

    # If string is empty set string to 0.
    # if text_encoder_lr == "":
    #     text_encoder_lr = 0
    # if unet_lr == "":
    #     unet_lr = 0

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

    if sdxl:
        run_cmd.append(rf"{scriptdir}/sd-scripts/sdxl_train_network.py")
    elif flux1_checkbox:
        run_cmd.append(rf"{scriptdir}/sd-scripts/flux_train_network.py")
    else:
        run_cmd.append(rf"{scriptdir}/sd-scripts/train_network.py")

    network_args = ""

    if LoRA_type == "LyCORIS/BOFT":
        network_module = "lycoris.kohya"
        network_args = f" preset={LyCORIS_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} module_dropout={module_dropout} use_tucker={use_tucker} use_scalar={use_scalar} rank_dropout={rank_dropout} rank_dropout_scale={rank_dropout_scale} constrain={constrain} rescaled={rescaled} algo=boft train_norm={train_norm}"

    if LoRA_type == "LyCORIS/Diag-OFT":
        network_module = "lycoris.kohya"
        network_args = f" preset={LyCORIS_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} module_dropout={module_dropout} use_tucker={use_tucker} use_scalar={use_scalar} rank_dropout={rank_dropout} rank_dropout_scale={rank_dropout_scale} constrain={constrain} rescaled={rescaled} algo=diag-oft train_norm={train_norm}"

    if LoRA_type == "LyCORIS/DyLoRA":
        network_module = "lycoris.kohya"
        network_args = f' preset={LyCORIS_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} use_tucker={use_tucker} block_size={unit} rank_dropout={rank_dropout} module_dropout={module_dropout} algo="dylora" train_norm={train_norm}'

    if LoRA_type == "LyCORIS/GLoRA":
        network_module = "lycoris.kohya"
        network_args = f' preset={LyCORIS_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} rank_dropout={rank_dropout} module_dropout={module_dropout} rank_dropout_scale={rank_dropout_scale} algo="glora" train_norm={train_norm}'

    if LoRA_type == "LyCORIS/iA3":
        network_module = "lycoris.kohya"
        network_args = f" preset={LyCORIS_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} train_on_input={train_on_input} algo=ia3"

    if LoRA_type == "LoCon" or LoRA_type == "LyCORIS/LoCon":
        network_module = "lycoris.kohya"
        network_args = f" preset={LyCORIS_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} rank_dropout={rank_dropout} bypass_mode={bypass_mode} dora_wd={dora_wd} module_dropout={module_dropout} use_tucker={use_tucker} use_scalar={use_scalar} rank_dropout_scale={rank_dropout_scale} algo=locon train_norm={train_norm}"

    if LoRA_type == "LyCORIS/LoHa":
        network_module = "lycoris.kohya"
        network_args = f' preset={LyCORIS_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} rank_dropout={rank_dropout} bypass_mode={bypass_mode} dora_wd={dora_wd} module_dropout={module_dropout} use_tucker={use_tucker} use_scalar={use_scalar} rank_dropout_scale={rank_dropout_scale} algo="loha" train_norm={train_norm}'

    if LoRA_type == "LyCORIS/LoKr":
        network_module = "lycoris.kohya"
        network_args = f" preset={LyCORIS_preset} conv_dim={conv_dim} conv_alpha={conv_alpha} rank_dropout={rank_dropout} bypass_mode={bypass_mode} dora_wd={dora_wd} module_dropout={module_dropout} factor={factor} use_cp={use_cp} use_scalar={use_scalar} decompose_both={decompose_both} rank_dropout_scale={rank_dropout_scale} algo=lokr train_norm={train_norm}"

    if LoRA_type == "LyCORIS/Native Fine-Tuning":
        network_module = "lycoris.kohya"
        network_args = f" preset={LyCORIS_preset} rank_dropout={rank_dropout} module_dropout={module_dropout} use_tucker={use_tucker} use_scalar={use_scalar} rank_dropout_scale={rank_dropout_scale} algo=full train_norm={train_norm}"

    if LoRA_type == "Flux1":
        # Add a list of supported network arguments for Flux1 below when supported
        kohya_lora_var_list = []
        network_module = "networks.lora_flux"
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }
        if split_mode:
            if train_blocks != "single":
                log.warning(
                    f"train_blocks is currently set to '{train_blocks}'. split_mode is enabled, forcing train_blocks to 'single'."
                )
            kohya_lora_vars["train_blocks"] = "single"
        if split_qkv:
            kohya_lora_vars["split_qkv"] = True
            
        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f" {key}={value}"

    if LoRA_type in ["Kohya LoCon", "Standard"]:
        kohya_lora_var_list = [
            "down_lr_weight",
            "mid_lr_weight",
            "up_lr_weight",
            "block_lr_zero_threshold",
            "block_dims",
            "block_alphas",
            "conv_block_dims",
            "conv_block_alphas",
            "rank_dropout",
            "module_dropout",
        ]
        network_module = "networks.lora"
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        # Not sure if Flux1 is Standard... or LoCon style... flip a coin... going for LoCon style...
        if LoRA_type in ["Kohya LoCon"]:
            network_args += f' conv_dim="{conv_dim}" conv_alpha="{conv_alpha}"'

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f" {key}={value}"

    if LoRA_type in ["LoRA-FA"]:
        kohya_lora_var_list = [
            "down_lr_weight",
            "mid_lr_weight",
            "up_lr_weight",
            "block_lr_zero_threshold",
            "block_dims",
            "block_alphas",
            "conv_block_dims",
            "conv_block_alphas",
            "rank_dropout",
            "module_dropout",
        ]

        network_module = "networks.lora_fa"
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ""
        if LoRA_type == "Kohya LoCon":
            network_args += f' conv_dim="{conv_dim}" conv_alpha="{conv_alpha}"'

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f" {key}={value}"

    if LoRA_type in ["Kohya DyLoRA"]:
        kohya_lora_var_list = [
            "conv_dim",
            "conv_alpha",
            "down_lr_weight",
            "mid_lr_weight",
            "up_lr_weight",
            "block_lr_zero_threshold",
            "block_dims",
            "block_alphas",
            "conv_block_dims",
            "conv_block_alphas",
            "rank_dropout",
            "module_dropout",
            "unit",
        ]

        network_module = "networks.dylora"
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ""

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f" {key}={value}"

    # Convert learning rates to float once and store the result for re-use
    learning_rate = float(learning_rate) if learning_rate is not None else 0.0
    text_encoder_lr_float = (
        float(text_encoder_lr) if text_encoder_lr is not None else 0.0
    )
    unet_lr_float = float(unet_lr) if unet_lr is not None else 0.0

    # Determine the training configuration based on learning rate values
    # Sets flags for training specific components based on the provided learning rates.
    if float(learning_rate) == unet_lr_float == text_encoder_lr_float == 0:
        output_message(msg="Please input learning rate values.", headless=headless)
        return TRAIN_BUTTON_VISIBLE
    # Flag to train text encoder only if its learning rate is non-zero and unet's is zero.
    network_train_text_encoder_only = text_encoder_lr_float != 0 and unet_lr_float == 0
    # Flag to train unet only if its learning rate is non-zero and text encoder's is zero.
    network_train_unet_only = text_encoder_lr_float == 0 and unet_lr_float != 0

    if text_encoder_lr_float != 0 or unet_lr_float != 0:
        do_not_set_learning_rate = True

    config_toml_data = {
        "adaptive_noise_scale": (
            adaptive_noise_scale
            if (adaptive_noise_scale != 0 and noise_offset_type == "Original")
            else None
        ),
        "async_upload": async_upload,
        "bucket_no_upscale": bucket_no_upscale,
        "bucket_reso_steps": bucket_reso_steps,
        "cache_latents": cache_latents,
        "cache_latents_to_disk": cache_latents_to_disk,
        "cache_text_encoder_outputs": (
            True
            if (sdxl and sdxl_cache_text_encoder_outputs)
            or (flux1_checkbox and flux1_cache_text_encoder_outputs)
            else None
        ),
        "caption_dropout_every_n_epochs": int(caption_dropout_every_n_epochs),
        "caption_dropout_rate": caption_dropout_rate,
        "caption_extension": caption_extension,
        "clip_skip": clip_skip if clip_skip != 0 else None,
        "color_aug": color_aug,
        "dataset_config": dataset_config,
        "debiased_estimation_loss": debiased_estimation_loss,
        "dynamo_backend": dynamo_backend,
        "dim_from_weights": dim_from_weights,
        "enable_bucket": enable_bucket,
        "epoch": int(epoch),
        "flip_aug": flip_aug,
        "fp8_base": fp8_base,
        "fp8_base_unet": fp8_base_unet if flux1_checkbox else None,
        "full_bf16": full_bf16,
        "full_fp16": full_fp16,
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "gradient_checkpointing": gradient_checkpointing,
        "highvram": highvram,
        "huber_c": huber_c,
        "huber_schedule": huber_schedule,
        "huggingface_repo_id": huggingface_repo_id,
        "huggingface_token": huggingface_token,
        "huggingface_repo_type": huggingface_repo_type,
        "huggingface_repo_visibility": huggingface_repo_visibility,
        "huggingface_path_in_repo": huggingface_path_in_repo,
        "ip_noise_gamma": ip_noise_gamma if ip_noise_gamma != 0 else None,
        "ip_noise_gamma_random_strength": ip_noise_gamma_random_strength,
        "keep_tokens": int(keep_tokens),
        "learning_rate": None if do_not_set_learning_rate else learning_rate,
        "logging_dir": logging_dir,
        "log_config": log_config,
        "log_tracker_name": log_tracker_name,
        "log_tracker_config": log_tracker_config,
        "loraplus_lr_ratio": loraplus_lr_ratio if not 0 else None,
        "loraplus_text_encoder_lr_ratio": (
            loraplus_text_encoder_lr_ratio if not 0 else None
        ),
        "loraplus_unet_lr_ratio": loraplus_unet_lr_ratio if not 0 else None,
        "loss_type": loss_type,
        "lowvram": lowvram,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_args": str(lr_scheduler_args).replace('"', "").split(),
        "lr_scheduler_num_cycles": (
            int(lr_scheduler_num_cycles)
            if lr_scheduler_num_cycles != ""
            else int(epoch)
        ),
        "lr_scheduler_power": lr_scheduler_power,
        "lr_scheduler_type": lr_scheduler_type if lr_scheduler_type != "" else None,
        "lr_warmup_steps": lr_warmup_steps,
        "masked_loss": masked_loss,
        "max_bucket_reso": max_bucket_reso,
        "max_grad_norm": max_grad_norm,
        "max_timestep": max_timestep if max_timestep != 0 else None,
        "max_token_length": int(max_token_length) if not flux1_checkbox else None,
        "max_train_epochs": (
            int(max_train_epochs) if int(max_train_epochs) != 0 else None
        ),
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
        "multires_noise_discount": (
            multires_noise_discount if noise_offset_type == "Multires" else None
        ),
        "multires_noise_iterations": (
            multires_noise_iterations
            if (multires_noise_iterations != 0 and noise_offset_type == "Multires")
            else None
        ),
        "network_alpha": network_alpha,
        "network_args": str(network_args).replace('"', "").split(),
        "network_dim": network_dim,
        "network_dropout": network_dropout,
        "network_module": network_module,
        "network_train_unet_only": network_train_unet_only,
        "network_train_text_encoder_only": network_train_text_encoder_only,
        "network_weights": network_weights,
        "no_half_vae": True if sdxl and sdxl_no_half_vae else None,
        "noise_offset": (
            noise_offset
            if (noise_offset != 0 and noise_offset_type == "Original")
            else None
        ),
        "noise_offset_random_strength": (
            noise_offset_random_strength if noise_offset_type == "Original" else None
        ),
        "noise_offset_type": noise_offset_type,
        "optimizer_type": optimizer,
        "optimizer_args": (
            str(optimizer_args).replace('"', "").split()
            if optimizer_args != []
            else None
        ),
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
        "scale_weight_norms": scale_weight_norms,
        "sdpa": True if xformers == "sdpa" else None,
        "seed": int(seed) if int(seed) != 0 else None,
        "shuffle_caption": shuffle_caption,
        "stop_text_encoder_training": (
            stop_text_encoder_training if stop_text_encoder_training != 0 else None
        ),
        "text_encoder_lr": text_encoder_lr if not 0 else None,
        "train_batch_size": train_batch_size,
        "train_data_dir": train_data_dir,
        "training_comment": training_comment,
        "unet_lr": unet_lr if not 0 else None,
        "log_with": log_with,
        "v2": v2,
        "v_parameterization": v_parameterization,
        "v_pred_like_loss": v_pred_like_loss if v_pred_like_loss != 0 else None,
        "vae": vae,
        "vae_batch_size": vae_batch_size if vae_batch_size != 0 else None,
        "wandb_api_key": wandb_api_key,
        "wandb_run_name": wandb_run_name if wandb_run_name != "" else output_name,
        "weighted_captions": weighted_captions,
        "xformers": True if xformers == "xformers" else None,
        # Flux.1 specific parameters
        # "cache_text_encoder_outputs": see previous assignment above for code
        "cache_text_encoder_outputs_to_disk": (
            flux1_cache_text_encoder_outputs_to_disk if flux1_checkbox else None
        ),
        "ae": ae if flux1_checkbox else None,
        "clip_l": clip_l if flux1_checkbox else None,
        "t5xxl": t5xxl if flux1_checkbox else None,
        "discrete_flow_shift": float(discrete_flow_shift) if flux1_checkbox else None,
        "model_prediction_type": model_prediction_type if flux1_checkbox else None,
        "timestep_sampling": timestep_sampling if flux1_checkbox else None,
        "split_mode": split_mode if flux1_checkbox else None,
        "t5xxl_max_token_length": t5xxl_max_token_length if flux1_checkbox else None,
        "guidance_scale": float(guidance_scale) if flux1_checkbox else None,
        "mem_eff_save": mem_eff_save if flux1_checkbox else None,
        "apply_t5_attn_mask": apply_t5_attn_mask if flux1_checkbox else None,
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
    tmpfilename = rf"{output_dir}/config_lora-{formatted_datetime}.toml"

    # Save the updated TOML data back to the file
    with open(tmpfilename, "w", encoding="utf-8") as toml_file:
        toml.dump(config_toml_data, toml_file)

        if not os.path.exists(toml_file.name):
            log.error(f"Failed to write TOML file: {toml_file.name}")

    run_cmd.append("--config_file")
    run_cmd.append(rf"{tmpfilename}")

    # Define a dictionary of parameters
    run_cmd_params = {
        "additional_parameters": additional_parameters,
    }

    # Use the ** syntax to unpack the dictionary when calling the function
    run_cmd = run_cmd_advanced_training(run_cmd=run_cmd, **run_cmd_params)

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

        # log.info(run_cmd)
        env = setup_environment()

        # Run the command

        executor.execute_command(run_cmd=run_cmd, env=env)

        train_state_value = time.time()

        return (
            gr.Button(visible=False or headless),
            gr.Button(visible=True),
            gr.Textbox(value=train_state_value),
        )


def lora_tab(
    train_data_dir_input=gr.Dropdown(),
    reg_data_dir_input=gr.Dropdown(),
    output_dir_input=gr.Dropdown(),
    logging_dir_input=gr.Dropdown(),
    headless=False,
    config: KohyaSSGUIConfig = {},
    use_shell_flag: bool = False,
):
    dummy_db_true = gr.Checkbox(value=True, visible=False)
    dummy_db_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    global use_shell
    use_shell = use_shell_flag

    with gr.Tab("Training"), gr.Column(variant="compact") as tab:
        gr.Markdown(
            "Train a custom model using kohya train network LoRA python code..."
        )

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

        with gr.Accordion("Metadata", open=False), gr.Group():
            metadata = MetaData(config=config)

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

            def list_presets(path):
                json_files = []

                # Insert an empty string at the beginning
                # json_files.insert(0, "none")

                for file in os.listdir(path):
                    if file.endswith(".json"):
                        json_files.append(os.path.splitext(file)[0])

                user_presets_path = os.path.join(path, "user_presets")
                if os.path.isdir(user_presets_path):
                    for file in os.listdir(user_presets_path):
                        if file.endswith(".json"):
                            preset_name = os.path.splitext(file)[0]
                            json_files.append(os.path.join("user_presets", preset_name))

                return json_files

            training_preset = gr.Dropdown(
                label="Presets",
                choices=["none"] + list_presets(rf"{presets_dir}/lora"),
                # elem_id="myDropdown",
                value="none",
            )

            with gr.Accordion("Basic", open="True"):
                with gr.Group(elem_id="basic_tab"):
                    with gr.Row():
                        LoRA_type = gr.Dropdown(
                            label="LoRA type",
                            choices=[
                                "Flux1",
                                "Kohya DyLoRA",
                                "Kohya LoCon",
                                "LoRA-FA",
                                "LyCORIS/iA3",
                                "LyCORIS/BOFT",
                                "LyCORIS/Diag-OFT",
                                "LyCORIS/DyLoRA",
                                "LyCORIS/GLoRA",
                                "LyCORIS/LoCon",
                                "LyCORIS/LoHa",
                                "LyCORIS/LoKr",
                                "LyCORIS/Native Fine-Tuning",
                                "Standard",
                            ],
                            value="Standard",
                        )
                        LyCORIS_preset = gr.Dropdown(
                            label="LyCORIS Preset",
                            choices=LYCORIS_PRESETS_CHOICES,
                            value="full",
                            visible=False,
                            interactive=True,
                            allow_custom_value=True,
                            # info="https://github.com/KohakuBlueleaf/LyCORIS/blob/0006e2ffa05a48d8818112d9f70da74c0cd30b99/docs/Preset.md"
                        )
                        with gr.Group():
                            with gr.Row():
                                network_weights = gr.Textbox(
                                    label="Network weights",
                                    placeholder="(Optional)",
                                    info="Path to an existing LoRA network weights to resume training from",
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
                                dim_from_weights = gr.Checkbox(
                                    label="DIM from weights",
                                    value=False,
                                    info="Automatically determine the dim(rank) from the weight file.",
                                )
                    basic_training = BasicTraining(
                        learning_rate_value=0.0001,
                        lr_scheduler_value="cosine",
                        lr_warmup_value=10,
                        sdxl_checkbox=source_model.sdxl_checkbox,
                        config=config,
                    )

                    with gr.Row():
                        text_encoder_lr = gr.Number(
                            label="Text Encoder learning rate",
                            value=0.0001,
                            info="(Optional)",
                            minimum=0,
                            maximum=1,
                        )

                        unet_lr = gr.Number(
                            label="Unet learning rate",
                            value=0.0001,
                            info="(Optional)",
                            minimum=0,
                            maximum=1,
                        )

                    with gr.Row() as loraplus:
                        loraplus_lr_ratio = gr.Number(
                            label="LoRA+ learning rate ratio",
                            value=0,
                            info="(Optional) starting with 16 is suggested",
                            minimum=0,
                            maximum=128,
                        )

                        loraplus_unet_lr_ratio = gr.Number(
                            label="LoRA+ Unet learning rate ratio",
                            value=0,
                            info="(Optional) starting with 16 is suggested",
                            minimum=0,
                            maximum=128,
                        )

                        loraplus_text_encoder_lr_ratio = gr.Number(
                            label="LoRA+ Text Encoder learning rate ratio",
                            value=0,
                            info="(Optional) starting with 16 is suggested",
                            minimum=0,
                            maximum=128,
                        )
                    # Add SDXL Parameters
                    sdxl_params = SDXLParameters(
                        source_model.sdxl_checkbox, config=config
                    )

                    # LyCORIS Specific parameters
                    with gr.Accordion("LyCORIS", visible=False) as lycoris_accordion:
                        with gr.Row():
                            factor = gr.Slider(
                                label="LoKr factor",
                                value=-1,
                                minimum=-1,
                                maximum=64,
                                step=1,
                                visible=False,
                            )
                            bypass_mode = gr.Checkbox(
                                value=False,
                                label="Bypass mode",
                                info="Designed for bnb 8bit/4bit linear layer. (QLyCORIS)",
                                visible=False,
                            )
                            dora_wd = gr.Checkbox(
                                value=False,
                                label="DoRA Weight Decompose",
                                info="Enable the DoRA method for these algorithms",
                                visible=False,
                            )
                            use_cp = gr.Checkbox(
                                value=False,
                                label="Use CP decomposition",
                                info="A two-step approach utilizing tensor decomposition and fine-tuning to accelerate convolution layers in large neural networks, resulting in significant CPU speedups with minor accuracy drops.",
                                visible=False,
                            )
                            use_tucker = gr.Checkbox(
                                value=False,
                                label="Use Tucker decomposition",
                                info="Efficiently decompose tensor shapes, resulting in a sequence of convolution layers with varying dimensions and Hadamard product implementation through multiplication of two distinct tensors.",
                                visible=False,
                            )
                            use_scalar = gr.Checkbox(
                                value=False,
                                label="Use Scalar",
                                info="Train an additional scalar in front of the weight difference, use a different weight initialization strategy.",
                                visible=False,
                            )
                        with gr.Row():
                            rank_dropout_scale = gr.Checkbox(
                                value=False,
                                label="Rank Dropout Scale",
                                info="Adjusts the scale of the rank dropout to maintain the average dropout rate, ensuring more consistent regularization across different layers.",
                                visible=False,
                            )
                            constrain = gr.Number(
                                value=0.0,
                                label="Constrain OFT",
                                info="Limits the norm of the oft_blocks, ensuring that their magnitude does not exceed a specified threshold, thus controlling the extent of the transformation applied.",
                                visible=False,
                            )
                            rescaled = gr.Checkbox(
                                value=False,
                                label="Rescaled OFT",
                                info="applies an additional scaling factor to the oft_blocks, allowing for further adjustment of their impact on the model's transformations.",
                                visible=False,
                            )
                            train_norm = gr.Checkbox(
                                value=False,
                                label="Train Norm",
                                info="Selects trainable layers in a network, but trains normalization layers identically across methods as they lack matrix decomposition.",
                                visible=False,
                            )
                            decompose_both = gr.Checkbox(
                                value=False,
                                label="LoKr decompose both",
                                info="Controls whether both input and output dimensions of the layer's weights are decomposed into smaller matrices for reparameterization.",
                                visible=False,
                            )
                            train_on_input = gr.Checkbox(
                                value=True,
                                label="iA3 train on input",
                                info="Set if we change the information going into the system (True) or the information coming out of it (False).",
                                visible=False,
                            )
                    with gr.Row() as network_row:
                        network_dim = gr.Slider(
                            minimum=1,
                            maximum=512,
                            label="Network Rank (Dimension)",
                            value=8,
                            step=1,
                            interactive=True,
                        )
                        network_alpha = gr.Slider(
                            minimum=0.00001,
                            maximum=1024,
                            label="Network Alpha",
                            value=1,
                            step=0.00001,
                            interactive=True,
                            info="alpha for LoRA weight scaling",
                        )
                    with gr.Row(visible=False) as convolution_row:
                        # locon= gr.Checkbox(label='Train a LoCon instead of a general LoRA (does not support v2 base models) (may not be able to some utilities now)', value=False)
                        conv_dim = gr.Slider(
                            minimum=0,
                            maximum=512,
                            value=1,
                            step=1,
                            label="Convolution Rank (Dimension)",
                        )
                        conv_alpha = gr.Slider(
                            minimum=0,
                            maximum=512,
                            value=1,
                            step=1,
                            label="Convolution Alpha",
                        )
                    with gr.Row():
                        scale_weight_norms = gr.Slider(
                            label="Scale weight norms",
                            value=0,
                            minimum=0,
                            maximum=10,
                            step=0.01,
                            info="Max Norm Regularization is a technique to stabilize network training by limiting the norm of network weights. It may be effective in suppressing overfitting of LoRA and improving stability when used with other LoRAs. See PR #545 on kohya_ss/sd_scripts repo for details. Recommended setting: 1. Higher is weaker, lower is stronger.",
                            interactive=True,
                        )
                        network_dropout = gr.Slider(
                            label="Network dropout",
                            value=0,
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            info="Is a normal probability dropout at the neuron level. In the case of LoRA, it is applied to the output of down. Recommended range 0.1 to 0.5",
                        )
                        rank_dropout = gr.Slider(
                            label="Rank dropout",
                            value=0,
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            info="can specify `rank_dropout` to dropout each rank with specified probability. Recommended range 0.1 to 0.3",
                        )
                        module_dropout = gr.Slider(
                            label="Module dropout",
                            value=0.0,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            info="can specify `module_dropout` to dropout each rank with specified probability. Recommended range 0.1 to 0.3",
                        )
                    with gr.Row(visible=False):
                        unit = gr.Slider(
                            minimum=1,
                            maximum=64,
                            label="DyLoRA Unit / Block size",
                            value=1,
                            step=1,
                            interactive=True,
                        )

                        # Show or hide LoCon conv settings depending on LoRA type selection
                        def update_LoRA_settings(
                            LoRA_type,
                            conv_dim,
                            network_dim,
                        ):
                            log.debug("LoRA type changed...")

                            lora_settings_config = {
                                "network_row": {
                                    "gr_type": gr.Row,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "Flux1",
                                            "Kohya DyLoRA",
                                            "Kohya LoCon",
                                            "LoRA-FA",
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/DyLoRA",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKr",
                                            "Standard",
                                        },
                                    },
                                },
                                "convolution_row": {
                                    "gr_type": gr.Row,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LoCon",
                                            "Kohya DyLoRA",
                                            "Kohya LoCon",
                                            "LoRA-FA",
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/DyLoRA",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKr",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/GLoRA",
                                        },
                                    },
                                },
                                "kohya_advanced_lora": {
                                    "gr_type": gr.Row,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "Flux1",
                                            "Standard",
                                            "Kohya DyLoRA",
                                            "Kohya LoCon",
                                            "LoRA-FA",
                                        },
                                    },
                                },
                                "network_weights": {
                                    "gr_type": gr.Textbox,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "Flux1",
                                            "Standard",
                                            "LoCon",
                                            "Kohya DyLoRA",
                                            "Kohya LoCon",
                                            "LoRA-FA",
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/DyLoRA",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoKr",
                                        },
                                    },
                                },
                                "network_weights_file": {
                                    "gr_type": gr.Button,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "Flux1",
                                            "Standard",
                                            "LoCon",
                                            "Kohya DyLoRA",
                                            "Kohya LoCon",
                                            "LoRA-FA",
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/DyLoRA",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoKr",
                                        },
                                    },
                                },
                                "dim_from_weights": {
                                    "gr_type": gr.Checkbox,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "Flux1",
                                            "Standard",
                                            "LoCon",
                                            "Kohya DyLoRA",
                                            "Kohya LoCon",
                                            "LoRA-FA",
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/DyLoRA",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoKr",
                                        }
                                    },
                                },
                                "factor": {
                                    "gr_type": gr.Slider,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LyCORIS/LoKr",
                                        },
                                    },
                                },
                                "conv_dim": {
                                    "gr_type": gr.Slider,
                                    "update_params": {
                                        "maximum": (
                                            100000
                                            if LoRA_type
                                            in {
                                                "LyCORIS/LoHa",
                                                "LyCORIS/LoKr",
                                                "LyCORIS/BOFT",
                                                "LyCORIS/Diag-OFT",
                                            }
                                            else 512
                                        ),
                                        "value": conv_dim,  # if conv_dim > 512 else conv_dim,
                                    },
                                },
                                "network_dim": {
                                    "gr_type": gr.Slider,
                                    "update_params": {
                                        "maximum": (
                                            100000
                                            if LoRA_type
                                            in {
                                                "LyCORIS/LoHa",
                                                "LyCORIS/LoKr",
                                                "LyCORIS/BOFT",
                                                "LyCORIS/Diag-OFT",
                                            }
                                            else 512
                                        ),
                                        "value": network_dim,  # if network_dim > 512 else network_dim,
                                    },
                                },
                                "bypass_mode": {
                                    "gr_type": gr.Checkbox,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKr",
                                        },
                                    },
                                },
                                "dora_wd": {
                                    "gr_type": gr.Checkbox,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKr",
                                        },
                                    },
                                },
                                "use_cp": {
                                    "gr_type": gr.Checkbox,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LyCORIS/LoKr",
                                        },
                                    },
                                },
                                "use_tucker": {
                                    "gr_type": gr.Checkbox,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/DyLoRA",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/Native Fine-Tuning",
                                        },
                                    },
                                },
                                "use_scalar": {
                                    "gr_type": gr.Checkbox,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKr",
                                            "LyCORIS/Native Fine-Tuning",
                                        },
                                    },
                                },
                                "rank_dropout_scale": {
                                    "gr_type": gr.Checkbox,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKr",
                                            "LyCORIS/Native Fine-Tuning",
                                        },
                                    },
                                },
                                "constrain": {
                                    "gr_type": gr.Number,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                        },
                                    },
                                },
                                "rescaled": {
                                    "gr_type": gr.Checkbox,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                        },
                                    },
                                },
                                "train_norm": {
                                    "gr_type": gr.Checkbox,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LyCORIS/DyLoRA",
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKr",
                                            "LyCORIS/Native Fine-Tuning",
                                        },
                                    },
                                },
                                "decompose_both": {
                                    "gr_type": gr.Checkbox,
                                    "update_params": {
                                        "visible": LoRA_type in {"LyCORIS/LoKr"},
                                    },
                                },
                                "train_on_input": {
                                    "gr_type": gr.Checkbox,
                                    "update_params": {
                                        "visible": LoRA_type in {"LyCORIS/iA3"},
                                    },
                                },
                                "scale_weight_norms": {
                                    "gr_type": gr.Slider,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LoCon",
                                            "Kohya DyLoRA",
                                            "Kohya LoCon",
                                            "LoRA-FA",
                                            "LyCORIS/DyLoRA",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoKr",
                                            "Standard",
                                        },
                                    },
                                },
                                "network_dropout": {
                                    "gr_type": gr.Slider,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LoCon",
                                            "Kohya DyLoRA",
                                            "Kohya LoCon",
                                            "LoRA-FA",
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/DyLoRA",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKr",
                                            "LyCORIS/Native Fine-Tuning",
                                            "Standard",
                                        },
                                    },
                                },
                                "rank_dropout": {
                                    "gr_type": gr.Slider,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LoCon",
                                            "Kohya DyLoRA",
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKR",
                                            "Kohya LoCon",
                                            "LoRA-FA",
                                            "LyCORIS/Native Fine-Tuning",
                                            "Standard",
                                        },
                                    },
                                },
                                "module_dropout": {
                                    "gr_type": gr.Slider,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LoCon",
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "Kohya DyLoRA",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKR",
                                            "Kohya LoCon",
                                            "LyCORIS/Native Fine-Tuning",
                                            "LoRA-FA",
                                            "Standard",
                                        },
                                    },
                                },
                                "LyCORIS_preset": {
                                    "gr_type": gr.Dropdown,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LyCORIS/DyLoRA",
                                            "LyCORIS/iA3",
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKr",
                                            "LyCORIS/Native Fine-Tuning",
                                        },
                                    },
                                },
                                "unit": {
                                    "gr_type": gr.Slider,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "Kohya DyLoRA",
                                            "LyCORIS/DyLoRA",
                                        },
                                    },
                                },
                                "lycoris_accordion": {
                                    "gr_type": gr.Accordion,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LyCORIS/DyLoRA",
                                            "LyCORIS/iA3",
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKr",
                                            "LyCORIS/Native Fine-Tuning",
                                        },
                                    },
                                },
                                "loraplus": {
                                    "gr_type": gr.Row,
                                    "update_params": {
                                        "visible": LoRA_type
                                        in {
                                            "LoCon",
                                            "Kohya DyLoRA",
                                            "LyCORIS/BOFT",
                                            "LyCORIS/Diag-OFT",
                                            "LyCORIS/GLoRA",
                                            "LyCORIS/LoCon",
                                            "LyCORIS/LoHa",
                                            "LyCORIS/LoKR",
                                            "Kohya LoCon",
                                            "LoRA-FA",
                                            "LyCORIS/Native Fine-Tuning",
                                            "Standard",
                                        },
                                    },
                                },
                            }

                            results = []
                            for attr, settings in lora_settings_config.items():
                                update_params = settings["update_params"]

                                results.append(settings["gr_type"](**update_params))

                            return tuple(results)

            with gr.Group():
                # Add FLUX1 Parameters
                flux1_training = flux1Training(
                    headless=headless,
                    config=config,
                    flux1_checkbox=source_model.flux1_checkbox,
                )

            with gr.Accordion("Advanced", open=False, elem_id="advanced_tab"):
                # with gr.Accordion('Advanced Configuration', open=False):
                with gr.Row(visible=True) as kohya_advanced_lora:
                    with gr.Tab(label="Weights"):
                        with gr.Row(visible=True):
                            down_lr_weight = gr.Textbox(
                                label="Down LR weights",
                                placeholder="(Optional) eg: 0,0,0,0,0,0,1,1,1,1,1,1",
                                info="Specify the learning rate weight of the down blocks of U-Net.",
                            )
                            mid_lr_weight = gr.Textbox(
                                label="Mid LR weights",
                                placeholder="(Optional) eg: 0.5",
                                info="Specify the learning rate weight of the mid block of U-Net.",
                            )
                            up_lr_weight = gr.Textbox(
                                label="Up LR weights",
                                placeholder="(Optional) eg: 0,0,0,0,0,0,1,1,1,1,1,1",
                                info="Specify the learning rate weight of the up blocks of U-Net. The same as down_lr_weight.",
                            )
                            block_lr_zero_threshold = gr.Textbox(
                                label="Blocks LR zero threshold",
                                placeholder="(Optional) eg: 0.1",
                                info="If the weight is not more than this value, the LoRA module is not created. The default is 0.",
                            )
                    with gr.Tab(label="Blocks"):
                        with gr.Row(visible=True):
                            block_dims = gr.Textbox(
                                label="Block dims",
                                placeholder="(Optional) eg: 2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",
                                info="Specify the dim (rank) of each block. Specify 25 numbers.",
                            )
                            block_alphas = gr.Textbox(
                                label="Block alphas",
                                placeholder="(Optional) eg: 2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",
                                info="Specify the alpha of each block. Specify 25 numbers as with block_dims. If omitted, the value of network_alpha is used.",
                            )
                    with gr.Tab(label="Conv"):
                        with gr.Row(visible=True):
                            conv_block_dims = gr.Textbox(
                                label="Conv dims",
                                placeholder="(Optional) eg: 2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",
                                info="Extend LoRA to Conv2d 3x3 and specify the dim (rank) of each block. Specify 25 numbers.",
                            )
                            conv_block_alphas = gr.Textbox(
                                label="Conv alphas",
                                placeholder="(Optional) eg: 2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",
                                info="Specify the alpha of each block when expanding LoRA to Conv2d 3x3. Specify 25 numbers. If omitted, the value of conv_alpha is used.",
                            )
                advanced_training = AdvancedTraining(
                    headless=headless, training_type="lora", config=config
                )
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

            LoRA_type.change(
                update_LoRA_settings,
                inputs=[
                    LoRA_type,
                    conv_dim,
                    network_dim,
                ],
                outputs=[
                    network_row,
                    convolution_row,
                    kohya_advanced_lora,
                    network_weights,
                    network_weights_file,
                    dim_from_weights,
                    factor,
                    conv_dim,
                    network_dim,
                    bypass_mode,
                    dora_wd,
                    use_cp,
                    use_tucker,
                    use_scalar,
                    rank_dropout_scale,
                    constrain,
                    rescaled,
                    train_norm,
                    decompose_both,
                    train_on_input,
                    scale_weight_norms,
                    network_dropout,
                    rank_dropout,
                    module_dropout,
                    LyCORIS_preset,
                    unit,
                    lycoris_accordion,
                    loraplus,
                ],
            )

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
            source_model.flux1_checkbox,
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
            advanced_training.fp8_base,
            advanced_training.fp8_base_unet,
            advanced_training.full_fp16,
            advanced_training.highvram,
            advanced_training.lowvram,
            # advanced_training.no_token_padding,
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
            text_encoder_lr,
            unet_lr,
            network_dim,
            network_weights,
            dim_from_weights,
            advanced_training.color_aug,
            advanced_training.flip_aug,
            advanced_training.masked_loss,
            advanced_training.clip_skip,
            accelerate_launch.num_processes,
            accelerate_launch.num_machines,
            accelerate_launch.multi_gpu,
            accelerate_launch.gpu_ids,
            accelerate_launch.main_process_port,
            advanced_training.gradient_accumulation_steps,
            advanced_training.mem_eff_attn,
            source_model.output_name,
            source_model.model_list,
            advanced_training.max_token_length,
            basic_training.max_train_epochs,
            basic_training.max_train_steps,
            advanced_training.max_data_loader_n_workers,
            network_alpha,
            source_model.training_comment,
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
            basic_training.max_grad_norm,
            advanced_training.noise_offset_type,
            advanced_training.noise_offset,
            advanced_training.noise_offset_random_strength,
            advanced_training.adaptive_noise_scale,
            advanced_training.multires_noise_iterations,
            advanced_training.multires_noise_discount,
            advanced_training.ip_noise_gamma,
            advanced_training.ip_noise_gamma_random_strength,
            LoRA_type,
            factor,
            bypass_mode,
            dora_wd,
            use_cp,
            use_tucker,
            use_scalar,
            rank_dropout_scale,
            constrain,
            rescaled,
            train_norm,
            decompose_both,
            train_on_input,
            conv_dim,
            conv_alpha,
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
            down_lr_weight,
            mid_lr_weight,
            up_lr_weight,
            block_lr_zero_threshold,
            block_dims,
            block_alphas,
            conv_block_dims,
            conv_block_alphas,
            advanced_training.weighted_captions,
            unit,
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
            scale_weight_norms,
            network_dropout,
            rank_dropout,
            module_dropout,
            sdxl_params.sdxl_cache_text_encoder_outputs,
            sdxl_params.sdxl_no_half_vae,
            advanced_training.full_bf16,
            advanced_training.min_timestep,
            advanced_training.max_timestep,
            advanced_training.vae,
            accelerate_launch.dynamo_backend,
            accelerate_launch.dynamo_mode,
            accelerate_launch.dynamo_use_fullgraph,
            accelerate_launch.dynamo_use_dynamic,
            accelerate_launch.extra_accelerate_launch_args,
            LyCORIS_preset,
            advanced_training.debiased_estimation_loss,
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
            loraplus_lr_ratio,
            loraplus_text_encoder_lr_ratio,
            loraplus_unet_lr_ratio,
            # Flux1 parameters
            flux1_training.flux1_cache_text_encoder_outputs,
            flux1_training.flux1_cache_text_encoder_outputs_to_disk,
            flux1_training.ae,
            flux1_training.clip_l,
            flux1_training.t5xxl,
            flux1_training.discrete_flow_shift,
            flux1_training.model_prediction_type,
            flux1_training.timestep_sampling,
            flux1_training.split_mode,
            flux1_training.train_blocks,
            flux1_training.t5xxl_max_token_length,
            flux1_training.guidance_scale,
            flux1_training.mem_eff_save,
            flux1_training.apply_t5_attn_mask,
            flux1_training.split_qkv,
        ]

        configuration.button_open_config.click(
            open_configuration,
            inputs=[dummy_db_true, dummy_db_false, configuration.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[configuration.config_file_name]
            + settings_list
            + [training_preset, convolution_row],
            show_progress=False,
        )

        configuration.button_load_config.click(
            open_configuration,
            inputs=[dummy_db_false, dummy_db_false, configuration.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[configuration.config_file_name]
            + settings_list
            + [training_preset, convolution_row],
            show_progress=False,
        )

        training_preset.input(
            open_configuration,
            inputs=[dummy_db_false, dummy_db_true, configuration.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[gr.Textbox(visible=False)]
            + settings_list
            + [training_preset, convolution_row],
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
            executor.kill_command,
            outputs=[executor.button_run, executor.button_stop_training],
        )

        button_print.click(
            train_model,
            inputs=[dummy_headless] + [dummy_db_true] + settings_list,
            show_progress=False,
        )

    with gr.Tab("Tools"):
        lora_tools = LoRATools(headless=headless)

    with gr.Tab("Guides"):
        gr.Markdown("This section provide Various LoRA guides and information...")
        if os.path.exists(rf"{scriptdir}/docs/LoRA/top_level.md"):
            with open(
                os.path.join(rf"{scriptdir}/docs/LoRA/top_level.md"),
                "r",
                encoding="utf-8",
            ) as file:
                guides_top_level = file.read() + "\n"
            gr.Markdown(guides_top_level)

    return (
        source_model.train_data_dir,
        folders.reg_data_dir,
        folders.output_dir,
        folders.logging_dir,
    )
