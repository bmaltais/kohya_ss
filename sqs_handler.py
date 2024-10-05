from datetime import datetime
import logging
import math
import os
import shutil
import sys
from typing import List
import boto3
from dotenv import load_dotenv
import toml
import time
import json

from kohya_gui import dreambooth_folder_creation_gui
from kohya_gui.common_gui import (
    check_if_model_exist,
    get_executable_path,
    output_message,
    print_command_and_toml,
    run_cmd_advanced_training,
    SaveConfigFile,
    scriptdir,
    validate_file_path,
    validate_folder_path,
    validate_model_path,
    validate_toml_file,
    validate_args_setting,
    setup_environment,
)
from kohya_gui.class_accelerate_launch import AccelerateLaunch
from kohya_gui.class_command_executor import CommandExecutor
from kohya_gui.class_sample_images import create_prompt_file

from app.models import Image, LoraModelStatus, LoraModel
from app.database import SessionLocal

load_dotenv()

PROJECT_DIR = os.environ.get("PROJECT_DIR")

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION")

SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL")
sqs = boto3.client(
    "sqs",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

BUCKET_NAME = "gazai"
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name="ap-northeast-1",
)


# Set up logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Setup command executor
executor = None

LYCORIS_PRESETS_CHOICES = [
    "attn-mlp",
    "attn-only",
    "full",
    "full-lin",
    "unet-transformer-only",
    "unet-convblock-only",
]


def upload_model_to_s3(user_id, model_id, model_file):
    object_key = f"loras/{user_id}/{model_file.split('/')[-1]}"
    s3.upload_file(model_file, BUCKET_NAME, object_key)
    logging.info(f"Uploaded {object_key} ...")


def _folder_preparation(
    user_id: str,
    model_id: str,
    instance_prompt: str,
    class_prompt: str,
    training_images: List[Image],
    training_dir_output: str,
):
    training_images_dir_input = rf"{PROJECT_DIR}/{user_id}/{model_id}/raw/img"
    os.makedirs(training_images_dir_input, exist_ok=True)

    for index, training_image in enumerate(training_images):
        object_key = training_image.image.objectKey
        if not object_key.startswith("assets"):
            object_key = rf"assets/{user_id}/{object_key}"

        file_extension = os.path.splitext(object_key)[1]
        local_img_file_path = os.path.join(
            training_images_dir_input, f"{index}{file_extension}"
        )

        s3.download_file(BUCKET_NAME, object_key, local_img_file_path)
        logging.info(f"Downloaded {object_key} to {local_img_file_path}...")

        if training_image.caption is not None:
            local_caption_file_path = os.path.join(
                training_images_dir_input, f"{index}.txt"
            )
            with open(local_caption_file_path, "a") as file:
                file.write(f"{training_image.caption}")
            logging.info(
                f"Wrote {training_image.caption} to {local_caption_file_path}..."
            )

    dreambooth_folder_creation_gui.dreambooth_folder_preparation(
        util_training_images_dir_input=training_images_dir_input,
        util_training_images_repeat_input=40,
        util_instance_prompt_input=instance_prompt,
        util_regularization_images_dir_input="",
        util_regularization_images_repeat_input=1,
        util_class_prompt_input=class_prompt,
        util_training_dir_output=training_dir_output,
    )


def _train_model(
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
    fp8_base,
    full_fp16,
    # no_token_padding,
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
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    global executor
    if executor is None:
        executor = CommandExecutor(headless=headless)

    if executor.is_running():
        logging.error(
            "Training is already running. Can't start another training session."
        )
        return

    logging.info(f"Start training LoRA {LoRA_type} ...")

    logging.info(f"Validating lr scheduler arguments...")
    if not validate_args_setting(lr_scheduler_args):
        return

    logging.info(f"Validating optimizer arguments...")
    if not validate_args_setting(optimizer_args):
        return

    #
    # Validate paths
    #

    if not validate_file_path(dataset_config):
        return

    if not validate_file_path(log_tracker_config):
        return

    if not validate_folder_path(
        logging_dir, can_be_written_to=True, create_if_not_exists=True
    ):
        return

    if LyCORIS_preset not in LYCORIS_PRESETS_CHOICES:
        if not validate_toml_file(LyCORIS_preset):
            return

    if not validate_file_path(network_weights):
        return

    if not validate_folder_path(
        output_dir, can_be_written_to=True, create_if_not_exists=True
    ):
        return

    if not validate_model_path(pretrained_model_name_or_path):
        return

    if not validate_folder_path(reg_data_dir):
        return

    if not validate_folder_path(resume):
        return

    if not validate_folder_path(train_data_dir):
        return

    if not validate_model_path(vae):
        return

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
        return

    # if noise_offset == "":
    #     noise_offset = 0

    if float(noise_offset) > 1 or float(noise_offset) < 0:
        output_message(
            msg="Noise offset need to be a value between 0 and 1",
            headless=headless,
        )
        return

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
        return

    # If string is empty set string to 0.
    # if text_encoder_lr == "":
    #     text_encoder_lr = 0
    # if unet_lr == "":
    #     unet_lr = 0

    if dataset_config:
        logging.info(
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
            logging.error("Train data dir is empty")
            return

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
                logging.info(f"Folder {folder}: {repeats} repeats found")

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

                logging.info(f"Folder {folder}: {num_images} images found")

                # Calculate the total number of steps for this folder
                steps = repeats * num_images

                # logging.info the result
                logging.info(
                    f"Folder {folder}: {num_images} * {repeats} = {steps} steps"
                )

                total_steps += steps

            except ValueError:
                # Handle the case where the folder name does not contain an underscore
                logging.info(
                    f"Error: '{folder}' does not contain an underscore, skipping..."
                )

        if reg_data_dir == "":
            reg_factor = 1
        else:
            logging.warning(
                "Regularization images are used... Will double the number of steps required..."
            )
            reg_factor = 2

        logging.info(f"Regularization factor: {reg_factor}")

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

        logging.info(f"Total steps: {total_steps}")

    logging.info(f"Train batch size: {train_batch_size}")
    logging.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logging.info(f"Epoch: {epoch}")
    logging.info(max_train_steps_info)
    logging.info(f"stop_text_encoder_training = {stop_text_encoder_training}")
    logging.info(f"lr_warmup_steps = {lr_warmup_steps}")

    accelerate_path = get_executable_path("accelerate")
    if accelerate_path == "":
        logging.error("accelerate not found")
        return

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
        if LoRA_type == "Kohya LoCon":
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
        return
    # Flag to train text encoder only if its learning rate is non-zero and unet's is zero.
    network_train_text_encoder_only = text_encoder_lr_float != 0 and unet_lr_float == 0
    # Flag to train unet only if its learning rate is non-zero and text encoder's is zero.
    network_train_unet_only = text_encoder_lr_float == 0 and unet_lr_float != 0

    config_toml_data = {
        "adaptive_noise_scale": (
            adaptive_noise_scale if adaptive_noise_scale != 0 else None
        ),
        "async_upload": async_upload,
        "bucket_no_upscale": bucket_no_upscale,
        "bucket_reso_steps": bucket_reso_steps,
        "cache_latents": cache_latents,
        "cache_latents_to_disk": cache_latents_to_disk,
        "cache_text_encoder_outputs": (
            True if sdxl and sdxl_cache_text_encoder_outputs else None
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
        "full_bf16": full_bf16,
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
        "ip_noise_gamma": ip_noise_gamma if ip_noise_gamma != 0 else None,
        "ip_noise_gamma_random_strength": ip_noise_gamma_random_strength,
        "keep_tokens": int(keep_tokens),
        "learning_rate": learning_rate,
        "logging_dir": logging_dir,
        "log_tracker_name": log_tracker_name,
        "log_tracker_config": log_tracker_config,
        "loss_type": loss_type,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_args": str(lr_scheduler_args).replace('"', "").split(),
        "lr_scheduler_num_cycles": (
            int(lr_scheduler_num_cycles)
            if lr_scheduler_num_cycles != ""
            else int(epoch)
        ),
        "lr_scheduler_power": lr_scheduler_power,
        "lr_warmup_steps": lr_warmup_steps,
        "masked_loss": masked_loss,
        "max_bucket_reso": max_bucket_reso,
        "max_grad_norm": max_grad_norm,
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
        "network_alpha": network_alpha,
        "network_args": str(network_args).replace('"', "").split(),
        "network_dim": network_dim,
        "network_dropout": network_dropout,
        "network_module": network_module,
        "network_train_unet_only": network_train_unet_only,
        "network_train_text_encoder_only": network_train_text_encoder_only,
        "network_weights": network_weights,
        "no_half_vae": True if sdxl and sdxl_no_half_vae else None,
        "noise_offset": noise_offset if noise_offset != 0 else None,
        "noise_offset_random_strength": noise_offset_random_strength,
        "noise_offset_type": noise_offset_type,
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
        "wandb_run_name": wandb_run_name,
        "weighted_captions": weighted_captions,
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
    tmpfilename = rf"{output_dir}/config_lora-{formatted_datetime}.toml"

    # Save the updated TOML data back to the file
    with open(tmpfilename, "w", encoding="utf-8") as toml_file:
        toml.dump(config_toml_data, toml_file)

        if not os.path.exists(toml_file.name):
            logging.error(f"Failed to write TOML file: {toml_file.name}")

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

        logging.info(f"Saving training config to {file_path}...")

        SaveConfigFile(
            parameters=parameters,
            file_path=file_path,
            exclusion=["file_path", "save_as", "headless", "print_only"],
        )

        # logging.info(run_cmd)
        env = setup_environment()

        # Run the command
        executor.execute_command(run_cmd=run_cmd, env=env)
        executor.wait_for_training_to_end()

        logging.info(f"Training completed ...")


def train_lora_model(model_data: dict):
    logging.info("train_lora_model task started...")

    try:
        # 1. Load the model from the database
        database = SessionLocal()
        db_model: LoraModel = (
            database.query(LoraModel).filter_by(id=model_data["id"]).first()
        )

        model_root_dir = rf"{PROJECT_DIR}/{db_model.userId}/{db_model.id}"
        os.makedirs(model_root_dir, exist_ok=True)
        if db_model:
            # 2. Update the status to "training"
            db_model.status = LoraModelStatus.TRAINING
            database.commit()

            # 3. Perform the actual model training
            training_dir_output = os.path.join(model_root_dir, "output")
            os.makedirs(training_dir_output, exist_ok=True)
            _folder_preparation(
                user_id=db_model.userId,
                model_id=db_model.id,
                instance_prompt=db_model.instancePrompt,
                class_prompt=db_model.classPrompt,
                training_images=db_model.trainingImages,
                training_dir_output=training_dir_output,
            )

            logging_dir = os.path.join(training_dir_output, "log")
            train_data_dir = os.path.join(training_dir_output, "img")
            output_dir = os.path.join(training_dir_output, "model")
            save_model_as = "safetensors"
            _train_model(
                headless=False,
                print_only=False,
                pretrained_model_name_or_path=f"/workspace/storage/stable_diffusion/models/ckpt/{db_model.trainingBaseModel}",
                v2=False,
                v_parameterization=False,
                sdxl=True,
                logging_dir=logging_dir,
                train_data_dir=train_data_dir,
                reg_data_dir="",
                output_dir=output_dir,
                dataset_config="",
                max_resolution=db_model.resolution,
                learning_rate=0.0001,
                lr_scheduler="cosine_with_restarts",
                lr_warmup=10,
                train_batch_size=2,
                epoch=1,
                save_every_n_epochs=2,
                mixed_precision="fp16",
                save_precision="fp16",
                seed=420,
                num_cpu_threads_per_process=2,
                cache_latents=True,
                cache_latents_to_disk=False,
                caption_extension=".txt",
                enable_bucket=True,
                gradient_checkpointing=True,
                fp8_base=False,
                full_fp16=False,
                stop_text_encoder_training_pct=0,
                min_bucket_reso=768,
                max_bucket_reso=1024,
                xformers="xformers",
                save_model_as=save_model_as,
                shuffle_caption=True,
                save_state=False,
                save_state_on_train_end=False,
                resume="",
                prior_loss_weight=1,
                text_encoder_lr=0.00005,
                unet_lr=0.0001,
                network_dim=160,
                network_weights="",
                dim_from_weights=False,
                color_aug=False,
                flip_aug=False,
                masked_loss=False,
                clip_skip=2,
                num_processes=1,
                num_machines=1,
                multi_gpu=False,
                gpu_ids="",
                main_process_port=0,
                gradient_accumulation_steps=1,
                mem_eff_attn=False,
                output_name=db_model.fileName,
                model_list="custom",
                max_token_length="150",
                max_train_epochs=6,
                max_train_steps=1600,
                max_data_loader_n_workers=0,
                network_alpha=160,
                training_comment="LORA:",
                keep_tokens=0,
                lr_scheduler_num_cycles=1,
                lr_scheduler_power=1,
                persistent_data_loader_workers=False,
                bucket_no_upscale=True,
                random_crop=False,
                bucket_reso_steps=64,
                v_pred_like_loss=0,
                caption_dropout_every_n_epochs=0,
                caption_dropout_rate=0,
                optimizer="AdamW8bit",
                optimizer_args="",
                lr_scheduler_args="",
                max_grad_norm=1,
                noise_offset_type="Original",
                noise_offset=0,
                noise_offset_random_strength=False,
                adaptive_noise_scale=0,
                multires_noise_iterations=0,
                multires_noise_discount=0.3,
                ip_noise_gamma=0,
                ip_noise_gamma_random_strength=False,
                LoRA_type="Standard",
                factor=-1,
                bypass_mode=False,
                dora_wd=False,
                use_cp=False,
                use_tucker=False,
                use_scalar=False,
                rank_dropout_scale=False,
                constrain=0,
                rescaled=False,
                train_norm=False,
                decompose_both=False,
                train_on_input=True,
                conv_dim=1,
                conv_alpha=1,
                sample_every_n_steps=0,
                sample_every_n_epochs=0,
                sample_sampler="euler_a",
                sample_prompts="",
                additional_parameters="",
                loss_type="l2",
                huber_schedule="snr",
                huber_c=0.1,
                vae_batch_size=0,
                min_snr_gamma=0,
                down_lr_weight="",
                mid_lr_weight="",
                up_lr_weight="",
                block_lr_zero_threshold="",
                block_dims="",
                block_alphas="",
                conv_block_dims="",
                conv_block_alphas="",
                weighted_captions=False,
                unit=1,
                save_every_n_steps=0,
                save_last_n_steps=0,
                save_last_n_steps_state=0,
                log_with="",
                wandb_api_key="",
                wandb_run_name="",
                log_tracker_name="",
                log_tracker_config="",
                scale_v_pred_loss_like_noise_pred=False,
                scale_weight_norms=0,
                network_dropout=0,
                rank_dropout=0,
                module_dropout=0,
                sdxl_cache_text_encoder_outputs=False,
                sdxl_no_half_vae=True,
                full_bf16=False,
                min_timestep=0,
                max_timestep=1000,
                vae="",
                dynamo_backend="no",
                dynamo_mode="default",
                dynamo_use_fullgraph=False,
                dynamo_use_dynamic=False,
                extra_accelerate_launch_args="",
                LyCORIS_preset="full",
                debiased_estimation_loss=False,
                huggingface_repo_id="",
                huggingface_token="",
                huggingface_repo_type="",
                huggingface_repo_visibility="",
                huggingface_path_in_repo="",
                save_state_to_huggingface=False,
                resume_from_huggingface="",
                async_upload=False,
                metadata_author="",
                metadata_description="",
                metadata_license="",
                metadata_tags="",
                metadata_title="",
            )

            # 4. Update the status to "ready"
            model_file = os.path.join(
                output_dir, f"{db_model.fileName}.{save_model_as}"
            )
            object_key = (
                f"loras/{db_model.userId}/{db_model.id}/{model_file.split('/')[-1]}"
            )
            s3.upload_file(model_file, BUCKET_NAME, object_key)

            database = SessionLocal()
            db_model = database.query(LoraModel).filter_by(id=model_data["id"]).first()
            db_model.objectKey = object_key
            db_model.status = LoraModelStatus.READY
            database.commit()

            # 5. Copy file to a1111 Lora dir
            shutil.copy(
                model_file,
                f"/workspace/stable-diffusion-webui/models/Lora/{model_file.split('/')[-1]}",
            )
        else:
            raise Exception("Model not found")

    except Exception as e:
        database.rollback()
        db_model = database.query(LoraModel).filter_by(id=model_data["id"]).first()
        if db_model:
            db_model.status = LoraModelStatus.ERROR
            database.commit()
        logging.error(f"Error during training: {e}")

    finally:
        database.close()
        shutil.rmtree(model_root_dir)


def process_message(message):
    # Parse the message payload
    payload = json.loads(message["Body"])

    # Execute the LoRA training script
    train_lora_model(payload)


def main():
    logging.info("Listening for messages...")
    while True:
        # Receive message from SQS queue
        response = sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20
        )

        # Check if a message was received
        if "Messages" in response:
            for message in response["Messages"]:
                try:
                    process_message(message)

                    # Delete the message from the queue
                    sqs.delete_message(
                        QueueUrl=SQS_QUEUE_URL, ReceiptHandle=message["ReceiptHandle"]
                    )
                except Exception as e:
                    logging.error(f"Error processing message: {e}")

        # Optional: Add a small delay to avoid excessive polling
        time.sleep(1)


if __name__ == "__main__":
    main()
