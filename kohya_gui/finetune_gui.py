import gradio as gr
import json
import math
import os
import subprocess
import time
import sys
import toml
from datetime import datetime
from .common_gui import (
    check_if_model_exist,
    color_aug_changed,
    get_executable_path,
    get_file_path,
    get_saveasfile_path,
    print_command_and_toml,
    run_cmd_advanced_training,
    SaveConfigFile,
    scriptdir,
    update_my_data,
    validate_file_path,
    validate_folder_path,
    validate_model_path,
    validate_args_setting,
    setup_environment,
)
from .class_accelerate_launch import AccelerateLaunch
from .class_configuration_file import ConfigurationFile
from .class_source_model import SourceModel
from .class_basic_training import BasicTraining
from .class_advanced_training import AdvancedTraining
from .class_sd3 import sd3Training
from .class_folders import Folders
from .class_sdxl_parameters import SDXLParameters
from .class_command_executor import CommandExecutor
from .class_tensorboard import TensorboardManager
from .class_sample_images import SampleImages, create_prompt_file
from .class_huggingface import HuggingFace
from .class_metadata import MetaData
from .class_gui_config import KohyaSSGUIConfig
from .class_flux1 import flux1Training

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = None

# Setup huggingface
huggingface = None
use_shell = False
train_state_value = time.time()

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

PYTHON = sys.executable

presets_dir = rf"{scriptdir}/presets"


def save_configuration(
    save_as_bool,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl_checkbox,
    flux1_checkbox,
    train_dir,
    image_folder,
    output_dir,
    dataset_config,
    logging_dir,
    max_resolution,
    min_bucket_reso,
    max_bucket_reso,
    batch_size,
    flip_aug,
    masked_loss,
    caption_metadata_filename,
    latent_metadata_filename,
    full_path,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    dataset_repeats,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    learning_rate_te,
    learning_rate_te1,
    learning_rate_te2,
    train_text_encoder,
    full_bf16,
    create_caption,
    create_buckets,
    save_model_as,
    caption_extension,
    # use_8bit_adam,
    xformers,
    clip_skip,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    save_state,
    save_state_on_train_end,
    resume,
    gradient_checkpointing,
    fp8_base,
    gradient_accumulation_steps,
    block_lr,
    mem_eff_attn,
    shuffle_caption,
    output_name,
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    full_fp16,
    color_aug,
    model_list,
    cache_latents,
    cache_latents_to_disk,
    use_latent_files,
    keep_tokens,
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
    weighted_captions,
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
    fused_backward_pass,
    fused_optimizer_groups,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    min_timestep,
    max_timestep,
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
    # SD3 parameters
    sd3_cache_text_encoder_outputs,
    sd3_cache_text_encoder_outputs_to_disk,
    clip_g,
    clip_l,
    logit_mean,
    logit_std,
    mode_scale,
    save_clip,
    save_t5xxl,
    t5xxl,
    t5xxl_device,
    t5xxl_dtype,
    sd3_text_encoder_batch_size,
    weighting_scheme,
    sd3_checkbox,
    # Flux.1
    flux1_cache_text_encoder_outputs,
    flux1_cache_text_encoder_outputs_to_disk,
    ae,
    flux1_clip_l,
    flux1_t5xxl,
    discrete_flow_shift,
    model_prediction_type,
    timestep_sampling,
    split_mode,
    train_blocks,
    t5xxl_max_token_length,
    guidance_scale,
    blockwise_fused_optimizers,
    flux_fused_backward_pass,
    cpu_offload_checkpointing,
    single_blocks_to_swap,
    double_blocks_to_swap,
    mem_eff_save,
    apply_t5_attn_mask,
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
    apply_preset,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl_checkbox,
    flux1_checkbox,
    train_dir,
    image_folder,
    output_dir,
    dataset_config,
    logging_dir,
    max_resolution,
    min_bucket_reso,
    max_bucket_reso,
    batch_size,
    flip_aug,
    masked_loss,
    caption_metadata_filename,
    latent_metadata_filename,
    full_path,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    dataset_repeats,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    learning_rate_te,
    learning_rate_te1,
    learning_rate_te2,
    train_text_encoder,
    full_bf16,
    create_caption,
    create_buckets,
    save_model_as,
    caption_extension,
    # use_8bit_adam,
    xformers,
    clip_skip,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    save_state,
    save_state_on_train_end,
    resume,
    gradient_checkpointing,
    fp8_base,
    gradient_accumulation_steps,
    block_lr,
    mem_eff_attn,
    shuffle_caption,
    output_name,
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    full_fp16,
    color_aug,
    model_list,
    cache_latents,
    cache_latents_to_disk,
    use_latent_files,
    keep_tokens,
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
    weighted_captions,
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
    fused_backward_pass,
    fused_optimizer_groups,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    min_timestep,
    max_timestep,
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
    # SD3 parameters
    sd3_cache_text_encoder_outputs,
    sd3_cache_text_encoder_outputs_to_disk,
    clip_g,
    clip_l,
    logit_mean,
    logit_std,
    mode_scale,
    save_clip,
    save_t5xxl,
    t5xxl,
    t5xxl_device,
    t5xxl_dtype,
    sd3_text_encoder_batch_size,
    weighting_scheme,
    sd3_checkbox,
    # Flux.1
    flux1_cache_text_encoder_outputs,
    flux1_cache_text_encoder_outputs_to_disk,
    ae,
    flux1_clip_l,
    flux1_t5xxl,
    discrete_flow_shift,
    model_prediction_type,
    timestep_sampling,
    split_mode,
    train_blocks,
    t5xxl_max_token_length,
    guidance_scale,
    blockwise_fused_optimizers,
    flux_fused_backward_pass,
    cpu_offload_checkpointing,
    single_blocks_to_swap,
    double_blocks_to_swap,
    mem_eff_save,
    training_preset,
    apply_t5_attn_mask,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    # Check if we are "applying" a preset or a config
    if apply_preset:
        log.info(f"Applying preset {training_preset}...")
        file_path = rf"{presets_dir}/finetune/{training_preset}.json"
    else:
        # If not applying a preset, set the `training_preset` field to an empty string
        # Find the index of the `training_preset` parameter using the `index()` method
        training_preset_index = parameters.index(("training_preset", training_preset))

        # Update the value of `training_preset` by directly assigning an empty string value
        parameters[training_preset_index] = ("training_preset", "")

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
        json_value = my_data.get(key)
        # Set the value in the dictionary to the corresponding value in `my_data`, or the default value if not found
        if not key in ["ask_for_file", "apply_preset", "file_path"]:
            values.append(json_value if json_value is not None else value)
    return tuple(values)


def train_model(
    headless,
    print_only,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl_checkbox,
    flux1_checkbox,
    train_dir,
    image_folder,
    output_dir,
    dataset_config,
    logging_dir,
    max_resolution,
    min_bucket_reso,
    max_bucket_reso,
    batch_size,
    flip_aug,
    masked_loss,
    caption_metadata_filename,
    latent_metadata_filename,
    full_path,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    dataset_repeats,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    learning_rate_te,
    learning_rate_te1,
    learning_rate_te2,
    train_text_encoder,
    full_bf16,
    generate_caption_database,
    generate_image_buckets,
    save_model_as,
    caption_extension,
    # use_8bit_adam,
    xformers,
    clip_skip,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    save_state,
    save_state_on_train_end,
    resume,
    gradient_checkpointing,
    fp8_base,
    gradient_accumulation_steps,
    block_lr,
    mem_eff_attn,
    shuffle_caption,
    output_name,
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    full_fp16,
    color_aug,
    model_list,  # Keep this. Yes, it is unused here but required given the common list used
    cache_latents,
    cache_latents_to_disk,
    use_latent_files,
    keep_tokens,
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
    weighted_captions,
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
    fused_backward_pass,
    fused_optimizer_groups,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    min_timestep,
    max_timestep,
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
    # SD3 parameters
    sd3_cache_text_encoder_outputs,
    sd3_cache_text_encoder_outputs_to_disk,
    clip_g,
    clip_l,
    logit_mean,
    logit_std,
    mode_scale,
    save_clip,
    save_t5xxl,
    t5xxl,
    t5xxl_device,
    t5xxl_dtype,
    sd3_text_encoder_batch_size,
    weighting_scheme,
    sd3_checkbox,
    # Flux.1
    flux1_cache_text_encoder_outputs,
    flux1_cache_text_encoder_outputs_to_disk,
    ae,
    flux1_clip_l,
    flux1_t5xxl,
    discrete_flow_shift,
    model_prediction_type,
    timestep_sampling,
    split_mode,
    train_blocks,
    t5xxl_max_token_length,
    guidance_scale,
    blockwise_fused_optimizers,
    flux_fused_backward_pass,
    cpu_offload_checkpointing,
    single_blocks_to_swap,
    double_blocks_to_swap,
    mem_eff_save,
    apply_t5_attn_mask,
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

    log.debug(f"headless = {headless} ; print_only = {print_only}")

    log.info(f"Start Finetuning...")

    log.info(f"Validating lr scheduler arguments...")
    if not validate_args_setting(lr_scheduler_args):
        return

    log.info(f"Validating optimizer arguments...")
    if not validate_args_setting(optimizer_args):
        return

    if train_dir != "" and not os.path.exists(train_dir):
        os.mkdir(train_dir)

    #
    # Validate paths
    #

    if not validate_file_path(dataset_config):
        return TRAIN_BUTTON_VISIBLE

    if not validate_folder_path(image_folder):
        return TRAIN_BUTTON_VISIBLE

    if not validate_file_path(log_tracker_config):
        return TRAIN_BUTTON_VISIBLE

    if not validate_folder_path(
        logging_dir, can_be_written_to=True, create_if_not_exists=True
    ):
        return TRAIN_BUTTON_VISIBLE

    if not validate_folder_path(
        output_dir, can_be_written_to=True, create_if_not_exists=True
    ):
        return TRAIN_BUTTON_VISIBLE

    if not validate_model_path(pretrained_model_name_or_path):
        return TRAIN_BUTTON_VISIBLE

    if not validate_folder_path(resume):
        return TRAIN_BUTTON_VISIBLE

    #
    # End of path validation
    #

    # if not validate_paths(
    #     dataset_config=dataset_config,
    #     finetune_image_folder=image_folder,
    #     headless=headless,
    #     log_tracker_config=log_tracker_config,
    #     logging_dir=logging_dir,
    #     output_dir=output_dir,
    #     pretrained_model_name_or_path=pretrained_model_name_or_path,
    #     resume=resume,
    # ):
    #     return TRAIN_BUTTON_VISIBLE

    if not print_only and check_if_model_exist(
        output_name, output_dir, save_model_as, headless
    ):
        return TRAIN_BUTTON_VISIBLE

    if dataset_config:
        log.info(
            "Dataset config toml file used, skipping caption json file, image buckets, total_steps, train_batch_size, gradient_accumulation_steps, epoch, reg_factor, max_train_steps creation..."
        )

        if max_train_steps == 0:
            max_train_steps_info = f"Max train steps: 0. sd-scripts will therefore default to 1600. Please specify a different value if required."
        else:
            max_train_steps_info = f"Max train steps: {max_train_steps}"
    else:
        # create caption json file
        if generate_caption_database:
            # Define the command components
            run_cmd = [
                PYTHON,
                rf"{scriptdir}/sd-scripts/finetune/merge_captions_to_metadata.py",
            ]

            # Add the caption extension
            run_cmd.append("--caption_extension")
            if caption_extension == "":
                run_cmd.append(".caption")  # Default extension
            else:
                run_cmd.append(caption_extension)

            # Add paths for the image folder and the caption metadata file
            run_cmd.append(rf"{image_folder}")
            run_cmd.append(rf"{os.path.join(train_dir, caption_metadata_filename)}")

            # Include the full path flag if specified
            if full_path:
                run_cmd.append("--full_path")

            # Log the built command
            log.info(" ".join(run_cmd))

            # Prepare environment variables
            env = setup_environment()

            # Execute the command if not just for printing
            if not print_only:
                subprocess.run(run_cmd, env=env)

        # create images buckets
        if generate_image_buckets:
            # Build the command to run the preparation script
            run_cmd = [
                PYTHON,
                rf"{scriptdir}/sd-scripts/finetune/prepare_buckets_latents.py",
                rf"{image_folder}",
                rf"{os.path.join(train_dir, caption_metadata_filename)}",
                rf"{os.path.join(train_dir, latent_metadata_filename)}",
                rf"{pretrained_model_name_or_path}",
                "--batch_size",
                str(batch_size),
                "--max_resolution",
                str(max_resolution),
                "--min_bucket_reso",
                str(min_bucket_reso),
                "--max_bucket_reso",
                str(max_bucket_reso),
                "--mixed_precision",
                str(mixed_precision),
            ]

            # Conditional flags
            if full_path:
                run_cmd.append("--full_path")
            if sdxl_checkbox and sdxl_no_half_vae:
                log.info(
                    "Using mixed_precision = no because no half vae is selected..."
                )
                # Ensure 'no' is correctly handled without extra quotes that might be interpreted literally in command line
                run_cmd.append("--mixed_precision=no")

            # Log the complete command as a string for clarity
            log.info(" ".join(run_cmd))

            # Copy and modify environment variables
            env = setup_environment()

            # Execute the command if not just for printing
            if not print_only:
                subprocess.run(run_cmd, env=env)

        if image_folder == "":
            log.error("Image folder dir is empty")
            return TRAIN_BUTTON_VISIBLE

        image_num = len(
            [
                f
                for f, lower_f in (
                    (file, file.lower()) for file in os.listdir(image_folder)
                )
                if lower_f.endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
        )
        log.info(f"image_num = {image_num}")

        repeats = int(image_num) * int(dataset_repeats)
        log.info(f"repeats = {str(repeats)}")

        if max_train_steps == 0:
            # calculate max_train_steps
            max_train_steps = int(
                math.ceil(
                    float(repeats)
                    / int(train_batch_size)
                    / int(gradient_accumulation_steps)
                    * int(epoch)
                )
            )

        # Divide by two because flip augmentation create two copied of the source images
        if flip_aug and max_train_steps:
            max_train_steps = int(math.ceil(float(max_train_steps) / 2))

        if max_train_steps == 0:
            max_train_steps_info = f"Max train steps: 0. sd-scripts will therefore default to 1600. Please specify a different value if required."
        else:
            max_train_steps_info = f"Max train steps: {max_train_steps}"

    log.info(max_train_steps_info)

    if max_train_steps != 0:
        lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    else:
        lr_warmup_steps = 0
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

    if sdxl_checkbox:
        run_cmd.append(rf"{scriptdir}/sd-scripts/sdxl_train.py")
    elif sd3_checkbox:
        run_cmd.append(rf"{scriptdir}/sd-scripts/sd3_train.py")
    elif flux1_checkbox:
        run_cmd.append(rf"{scriptdir}/sd-scripts/flux_train.py")
    else:
        run_cmd.append(rf"{scriptdir}/sd-scripts/fine_tune.py")

    in_json = (
        f"{train_dir}/{latent_metadata_filename}"
        if use_latent_files == "Yes"
        else f"{train_dir}/{caption_metadata_filename}"
    )
    cache_text_encoder_outputs = (
        (sdxl_checkbox and sdxl_cache_text_encoder_outputs)
        or (sd3_checkbox and sd3_cache_text_encoder_outputs)
        or (flux1_checkbox and flux1_cache_text_encoder_outputs)
    )
    cache_text_encoder_outputs_to_disk = (
        sd3_checkbox and sd3_cache_text_encoder_outputs_to_disk
    ) or (flux1_checkbox and flux1_cache_text_encoder_outputs_to_disk)
    no_half_vae = sdxl_checkbox and sdxl_no_half_vae

    if max_data_loader_n_workers == "" or None:
        max_data_loader_n_workers = 0
    else:
        max_data_loader_n_workers = int(max_data_loader_n_workers)

    if max_train_steps == "" or None:
        max_train_steps = 0
    else:
        max_train_steps = int(max_train_steps)

    config_toml_data = {
        # Update the values in the TOML data
        "adaptive_noise_scale": (
            adaptive_noise_scale if adaptive_noise_scale != 0 else None
        ),
        "async_upload": async_upload,
        "block_lr": block_lr,
        "bucket_no_upscale": bucket_no_upscale,
        "bucket_reso_steps": bucket_reso_steps,
        "cache_latents": cache_latents,
        "cache_latents_to_disk": cache_latents_to_disk,
        "cache_text_encoder_outputs": cache_text_encoder_outputs,
        "cache_text_encoder_outputs_to_disk": cache_text_encoder_outputs_to_disk,
        "caption_dropout_every_n_epochs": int(caption_dropout_every_n_epochs),
        "caption_dropout_rate": caption_dropout_rate,
        "caption_extension": caption_extension,
        "clip_l": flux1_clip_l if flux1_checkbox else clip_l if sd3_checkbox else None,
        "clip_skip": clip_skip if clip_skip != 0 else None,
        "color_aug": color_aug,
        "dataset_config": dataset_config,
        "dataset_repeats": int(dataset_repeats),
        "debiased_estimation_loss": debiased_estimation_loss,
        "disable_mmap_load_safetensors": disable_mmap_load_safetensors,
        "dynamo_backend": dynamo_backend,
        "enable_bucket": True,
        "flip_aug": flip_aug,
        "fp8_base": fp8_base,
        "full_bf16": full_bf16,
        "full_fp16": full_fp16,
        "fused_backward_pass": fused_backward_pass if not flux1_checkbox else flux_fused_backward_pass,
        "fused_optimizer_groups": (
            int(fused_optimizer_groups) if fused_optimizer_groups > 0 else None
        ),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "gradient_checkpointing": gradient_checkpointing,
        "huber_c": huber_c,
        "huber_schedule": huber_schedule,
        "huggingface_repo_id": huggingface_repo_id,
        "huggingface_token": huggingface_token,
        "huggingface_repo_type": huggingface_repo_type,
        "huggingface_repo_visibility": huggingface_repo_visibility,
        "huggingface_path_in_repo": huggingface_path_in_repo,
        "in_json": in_json,
        "ip_noise_gamma": ip_noise_gamma if ip_noise_gamma != 0 else None,
        "ip_noise_gamma_random_strength": ip_noise_gamma_random_strength,
        "keep_tokens": int(keep_tokens),
        "learning_rate": learning_rate,  # both for sd1.5 and sdxl
        "learning_rate_te": (
            learning_rate_te if not sdxl_checkbox else None
        ),  # only for sd1.5
        "learning_rate_te1": (
            learning_rate_te1 if sdxl_checkbox else None
        ),  # only for sdxl
        "learning_rate_te2": (
            learning_rate_te2 if sdxl_checkbox else None
        ),  # only for sdxl
        "logging_dir": logging_dir,
        "log_config": log_config,
        "log_tracker_name": log_tracker_name,
        "log_tracker_config": log_tracker_config,
        "loss_type": loss_type,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_args": str(lr_scheduler_args).replace('"', "").split(),
        "lr_scheduler_type": lr_scheduler_type if lr_scheduler_type != "" else None,
        "lr_warmup_steps": lr_warmup_steps,
        "masked_loss": masked_loss,
        "max_bucket_reso": int(max_bucket_reso),
        "max_timestep": max_timestep if max_timestep != 0 else None,
        "max_token_length": int(max_token_length),
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
        "multires_noise_discount": multires_noise_discount,
        "multires_noise_iterations": (
            multires_noise_iterations if multires_noise_iterations != 0 else None
        ),
        "no_half_vae": no_half_vae,
        "noise_offset": noise_offset if noise_offset != 0 else None,
        "noise_offset_random_strength": noise_offset_random_strength,
        "noise_offset_type": noise_offset_type,
        "optimizer_type": optimizer,
        "optimizer_args": str(optimizer_args).replace('"', "").split(),
        "output_dir": output_dir,
        "output_name": output_name,
        "persistent_data_loader_workers": int(persistent_data_loader_workers),
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "random_crop": random_crop,
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
        "t5xxl": t5xxl if sd3_checkbox else flux1_t5xxl if flux1_checkbox else None,
        "train_batch_size": train_batch_size,
        "train_data_dir": image_folder,
        "train_text_encoder": train_text_encoder,
        "log_with": log_with,
        "v2": v2,
        "v_parameterization": v_parameterization,
        "v_pred_like_loss": v_pred_like_loss if v_pred_like_loss != 0 else None,
        "vae_batch_size": vae_batch_size if vae_batch_size != 0 else None,
        "wandb_api_key": wandb_api_key,
        "wandb_run_name": wandb_run_name if wandb_run_name != "" else output_name,
        "weighted_captions": weighted_captions,
        "xformers": True if xformers == "xformers" else None,
        # SD3 only Parameters
        # "cache_text_encoder_outputs": see previous assignment above for code
        # "cache_text_encoder_outputs_to_disk": see previous assignment above for code
        "clip_g": clip_g if sd3_checkbox else None,
        # "clip_l": see previous assignment above for code
        "logit_mean": logit_mean if sd3_checkbox else None,
        "logit_std": logit_std if sd3_checkbox else None,
        "mode_scale": mode_scale if sd3_checkbox else None,
        "save_clip": save_clip if sd3_checkbox else None,
        "save_t5xxl": save_t5xxl if sd3_checkbox else None,
        # "t5xxl": see previous assignment above for code
        "t5xxl_device": t5xxl_device if sd3_checkbox else None,
        "t5xxl_dtype": t5xxl_dtype if sd3_checkbox else None,
        "text_encoder_batch_size": (
            sd3_text_encoder_batch_size if sd3_checkbox else None
        ),
        "weighting_scheme": weighting_scheme if sd3_checkbox else None,
        # Flux.1 specific parameters
        # "cache_text_encoder_outputs": see previous assignment above for code
        # "cache_text_encoder_outputs_to_disk": see previous assignment above for code
        "ae": ae if flux1_checkbox else None,
        # "clip_l": see previous assignment above for code
        # "t5xxl": see previous assignment above for code
        "discrete_flow_shift": discrete_flow_shift if flux1_checkbox else None,
        "model_prediction_type": model_prediction_type if flux1_checkbox else None,
        "timestep_sampling": timestep_sampling if flux1_checkbox else None,
        "split_mode": split_mode if flux1_checkbox else None,
        "train_blocks": train_blocks if flux1_checkbox else None,
        "t5xxl_max_token_length": t5xxl_max_token_length if flux1_checkbox else None,
        "guidance_scale": guidance_scale if flux1_checkbox else None,
        "blockwise_fused_optimizers": (
            blockwise_fused_optimizers if flux1_checkbox else None
        ),
        # "flux_fused_backward_pass": see previous assignment of fused_backward_pass in above code
        "cpu_offload_checkpointing": (
            cpu_offload_checkpointing if flux1_checkbox else None
        ),
        "single_blocks_to_swap": single_blocks_to_swap if flux1_checkbox else None,
        "double_blocks_to_swap": double_blocks_to_swap if flux1_checkbox else None,
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
    tmpfilename = rf"{output_dir}/config_finetune-{formatted_datetime}.toml"
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


def finetune_tab(
    headless=False,
    config: KohyaSSGUIConfig = {},
    use_shell_flag: bool = False,
):
    dummy_db_true = gr.Checkbox(value=True, visible=False)
    dummy_db_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    global use_shell
    use_shell = use_shell_flag

    with gr.Tab("Training"), gr.Column(variant="compact"):
        gr.Markdown("Train a custom model using kohya finetune python code...")

        # Setup Configuration Files Gradio
        with gr.Accordion("Configuration", open=False):
            configuration = ConfigurationFile(headless=headless, config=config)

        with gr.Accordion("Accelerate launch", open=False), gr.Column():
            accelerate_launch = AccelerateLaunch(config=config)

        with gr.Column():
            source_model = SourceModel(
                headless=headless, finetuning=True, config=config
            )
            image_folder = source_model.train_data_dir
            output_name = source_model.output_name

        with gr.Accordion("Folders", open=False), gr.Group():
            folders = Folders(headless=headless, finetune=True, config=config)
            output_dir = folders.output_dir
            logging_dir = folders.logging_dir
            train_dir = folders.reg_data_dir

        with gr.Accordion("Metadata", open=False), gr.Group():
            metadata = MetaData(config=config)

        with gr.Accordion("Dataset Preparation", open=False):
            with gr.Row():
                max_resolution = gr.Textbox(
                    label="Resolution (width,height)", value="512,512"
                )
                min_bucket_reso = gr.Textbox(label="Min bucket resolution", value="256")
                max_bucket_reso = gr.Textbox(
                    label="Max bucket resolution", value="1024"
                )
                batch_size = gr.Textbox(label="Batch size", value="1")
            with gr.Row():
                create_caption = gr.Checkbox(
                    label="Generate caption metadata", value=True
                )
                create_buckets = gr.Checkbox(
                    label="Generate image buckets metadata", value=True
                )
                use_latent_files = gr.Dropdown(
                    label="Use latent files",
                    choices=[
                        "No",
                        "Yes",
                    ],
                    value="Yes",
                )
            with gr.Accordion("Advanced parameters", open=False):
                with gr.Row():
                    caption_metadata_filename = gr.Textbox(
                        label="Caption metadata filename",
                        value="meta_cap.json",
                    )
                    latent_metadata_filename = gr.Textbox(
                        label="Latent metadata filename", value="meta_lat.json"
                    )
                with gr.Row():
                    full_path = gr.Checkbox(label="Use full path", value=True)
                    weighted_captions = gr.Checkbox(
                        label="Weighted captions", value=False
                    )

        with gr.Accordion("Parameters", open=False), gr.Column():

            def list_presets(path):
                json_files = []

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
                choices=["none"] + list_presets(f"{presets_dir}/finetune"),
                # elem_id="myDropdown",
                value="none",
            )

            with gr.Accordion("Basic", open="True"):
                with gr.Group(elem_id="basic_tab"):
                    basic_training = BasicTraining(
                        learning_rate_value=1e-5,
                        finetuning=True,
                        sdxl_checkbox=source_model.sdxl_checkbox,
                        config=config,
                    )

                    # Add SDXL Parameters
                    sdxl_params = SDXLParameters(
                        source_model.sdxl_checkbox,
                        config=config,
                        trainer="finetune",
                    )

                    with gr.Row():
                        dataset_repeats = gr.Textbox(label="Dataset repeats", value=40)
                        train_text_encoder = gr.Checkbox(
                            label="Train text encoder", value=True
                        )

            # Add FLUX1 Parameters
            flux1_training = flux1Training(
                headless=headless,
                config=config,
                flux1_checkbox=source_model.flux1_checkbox,
                finetuning=True,
            )

            # Add SD3 Parameters
            sd3_training = sd3Training(
                headless=headless, config=config, sd3_checkbox=source_model.sd3_checkbox
            )

            with gr.Accordion("Advanced", open=False, elem_id="advanced_tab"):
                with gr.Row():
                    gradient_accumulation_steps = gr.Slider(
                        label="Gradient accumulate steps",
                        info="Number of updates steps to accumulate before performing a backward/update pass",
                        value=config.get("advanced.gradient_accumulation_steps", 1),
                        minimum=1,
                        maximum=120,
                        step=1,
                    )
                    block_lr = gr.Textbox(
                        label="Block LR (SDXL)",
                        placeholder="(Optional)",
                        info="Specify the different learning rates for each U-Net block. Specify 23 values separated by commas like 1e-3,1e-3 ... 1e-3",
                    )
                advanced_training = AdvancedTraining(
                    headless=headless, finetuning=True, config=config
                )
                advanced_training.color_aug.change(
                    color_aug_changed,
                    inputs=[advanced_training.color_aug],
                    outputs=[
                        basic_training.cache_latents
                    ],  # Not applicable to fine_tune.py
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

        TensorboardManager(headless=headless, logging_dir=folders.logging_dir)

        settings_list = [
            source_model.pretrained_model_name_or_path,
            source_model.v2,
            source_model.v_parameterization,
            source_model.sdxl_checkbox,
            source_model.flux1_checkbox,
            train_dir,
            image_folder,
            output_dir,
            source_model.dataset_config,
            logging_dir,
            max_resolution,
            min_bucket_reso,
            max_bucket_reso,
            batch_size,
            advanced_training.flip_aug,
            advanced_training.masked_loss,
            caption_metadata_filename,
            latent_metadata_filename,
            full_path,
            basic_training.learning_rate,
            basic_training.lr_scheduler,
            basic_training.lr_warmup,
            dataset_repeats,
            basic_training.train_batch_size,
            basic_training.epoch,
            basic_training.save_every_n_epochs,
            accelerate_launch.mixed_precision,
            source_model.save_precision,
            basic_training.seed,
            accelerate_launch.num_cpu_threads_per_process,
            basic_training.learning_rate_te,
            basic_training.learning_rate_te1,
            basic_training.learning_rate_te2,
            train_text_encoder,
            advanced_training.full_bf16,
            create_caption,
            create_buckets,
            source_model.save_model_as,
            basic_training.caption_extension,
            advanced_training.xformers,
            advanced_training.clip_skip,
            accelerate_launch.dynamo_backend,
            accelerate_launch.dynamo_mode,
            accelerate_launch.dynamo_use_fullgraph,
            accelerate_launch.dynamo_use_dynamic,
            accelerate_launch.extra_accelerate_launch_args,
            accelerate_launch.num_processes,
            accelerate_launch.num_machines,
            accelerate_launch.multi_gpu,
            accelerate_launch.gpu_ids,
            accelerate_launch.main_process_port,
            advanced_training.save_state,
            advanced_training.save_state_on_train_end,
            advanced_training.resume,
            advanced_training.gradient_checkpointing,
            advanced_training.fp8_base,
            gradient_accumulation_steps,
            block_lr,
            advanced_training.mem_eff_attn,
            advanced_training.shuffle_caption,
            output_name,
            advanced_training.max_token_length,
            basic_training.max_train_epochs,
            basic_training.max_train_steps,
            advanced_training.max_data_loader_n_workers,
            advanced_training.full_fp16,
            advanced_training.color_aug,
            source_model.model_list,
            basic_training.cache_latents,
            basic_training.cache_latents_to_disk,
            use_latent_files,
            advanced_training.keep_tokens,
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
            weighted_captions,
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
            sdxl_params.fused_backward_pass,
            sdxl_params.fused_optimizer_groups,
            sdxl_params.sdxl_cache_text_encoder_outputs,
            sdxl_params.sdxl_no_half_vae,
            advanced_training.min_timestep,
            advanced_training.max_timestep,
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
            # SD3 Parameters
            sd3_training.sd3_cache_text_encoder_outputs,
            sd3_training.sd3_cache_text_encoder_outputs_to_disk,
            sd3_training.clip_g,
            sd3_training.clip_l,
            sd3_training.logit_mean,
            sd3_training.logit_std,
            sd3_training.mode_scale,
            sd3_training.save_clip,
            sd3_training.save_t5xxl,
            sd3_training.t5xxl,
            sd3_training.t5xxl_device,
            sd3_training.t5xxl_dtype,
            sd3_training.sd3_text_encoder_batch_size,
            sd3_training.weighting_scheme,
            source_model.sd3_checkbox,
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
            flux1_training.blockwise_fused_optimizers,
            flux1_training.flux_fused_backward_pass,
            flux1_training.cpu_offload_checkpointing,
            flux1_training.single_blocks_to_swap,
            flux1_training.double_blocks_to_swap,
            flux1_training.mem_eff_save,
            flux1_training.apply_t5_attn_mask,
        ]

        configuration.button_open_config.click(
            open_configuration,
            inputs=[dummy_db_true, dummy_db_false, configuration.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[configuration.config_file_name]
            + settings_list
            + [training_preset],
            show_progress=False,
        )

        # config.button_open_config.click(
        #     open_configuration,
        #     inputs=[dummy_db_true, dummy_db_false, config.config_file_name] + settings_list,
        #     outputs=[config.config_file_name] + settings_list,
        #     show_progress=False,
        # )

        configuration.button_load_config.click(
            open_configuration,
            inputs=[dummy_db_false, dummy_db_false, configuration.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[configuration.config_file_name]
            + settings_list
            + [training_preset],
            show_progress=False,
        )

        training_preset.input(
            open_configuration,
            inputs=[dummy_db_false, dummy_db_true, configuration.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[gr.Textbox(visible=False)] + settings_list + [training_preset],
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

    with gr.Tab("Guides"):
        gr.Markdown("This section provide Various Finetuning guides and information...")
        top_level_path = rf'"{scriptdir}/docs/Finetuning/top_level.md"'
        if os.path.exists(top_level_path):
            with open(os.path.join(top_level_path), "r", encoding="utf-8") as file:
                guides_top_level = file.read() + "\n"
            gr.Markdown(guides_top_level)
