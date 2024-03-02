import gradio as gr
import json
import math
import os
import lycoris
from datetime import datetime
from .common_gui import (
    get_file_path,
    get_any_file_path,
    get_saveasfile_path,
    color_aug_changed,
    run_cmd_advanced_training,
    update_my_data,
    check_if_model_exist,
    output_message,
    verify_image_folder_pattern,
    SaveConfigFile,
    save_to_file,
    check_duplicate_filenames,
    scriptdir,
)
from .class_configuration_file import ConfigurationFile
from .class_source_model import SourceModel
from .class_basic_training import BasicTraining
from .class_advanced_training import AdvancedTraining
from .class_sdxl_parameters import SDXLParameters
from .class_folders import Folders
from .class_command_executor import CommandExecutor
from .tensorboard_gui import (
    gradio_tensorboard,
    start_tensorboard,
    stop_tensorboard,
)
from .class_sample_images import SampleImages, run_cmd_sample
from .class_lora_tab import LoRATools

from .dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from .dataset_balancing_gui import gradio_dataset_balancing_tab

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = CommandExecutor()

button_run = gr.Button("Start training", variant="primary")

button_stop_training = gr.Button("Stop training")

document_symbol = "\U0001F4C4"  # 📄

presets_dir = fr'{scriptdir}/presets'

def save_configuration(
    save_as,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
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
    stop_text_encoder_training,
    min_bucket_reso,
    max_bucket_reso,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    lora_network_weights,
    dim_from_weights,
    color_aug,
    flip_aug,
    clip_skip,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
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
    max_grad_norm,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    LoRA_type,
    factor,
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
    use_wandb,
    wandb_api_key,
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
    LyCORIS_preset,
    debiased_estimation_loss,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    original_file_path = file_path

    save_as_bool = True if save_as.get("label") == "True" else False

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
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
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
    stop_text_encoder_training,
    min_bucket_reso,
    max_bucket_reso,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    lora_network_weights,
    dim_from_weights,
    color_aug,
    flip_aug,
    clip_skip,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
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
    max_grad_norm,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    LoRA_type,
    factor,
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
    use_wandb,
    wandb_api_key,
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
    LyCORIS_preset,
    debiased_estimation_loss,
    training_preset,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    ask_for_file = True if ask_for_file.get("label") == "True" else False
    apply_preset = True if apply_preset.get("label") == "True" else False

    # Check if we are "applying" a preset or a config
    if apply_preset:
        if training_preset != "none":
            log.info(f"Applying preset {training_preset}...")
            file_path = fr'{presets_dir}/lora/{training_preset}.json'
    else:
        # If not applying a preset, set the `training_preset` field to an empty string
        # Find the index of the `training_preset` parameter using the `index()` method
        training_preset_index = parameters.index(("training_preset", training_preset))

        # Update the value of `training_preset` by directly assigning an empty string value
        parameters[training_preset_index] = ("training_preset", "none")

    original_file_path = file_path

    if ask_for_file:
        file_path = get_file_path(file_path)

    if not file_path == "" and not file_path == None:
        # Load variables from JSON file
        with open(file_path, "r") as f:
            my_data = json.load(f)
            log.info("Loading config...")

            # Update values to fix deprecated options, set appropriate optimizer if it is set to True, etc.
            my_data = update_my_data(my_data)
    else:
        file_path = original_file_path  # In case a file_path was provided and the user decides to cancel the open action
        my_data = {}

    values = [file_path]
    for key, value in parameters:
        # Set the value in the dictionary to the corresponding value in `my_data`, or the default value if not found
        if not key in ["ask_for_file", "apply_preset", "file_path"]:
            json_value = my_data.get(key)
            # if isinstance(json_value, str) and json_value == '':
            #     # If the JSON value is an empty string, use the default value
            #     values.append(value)
            # else:
            # Otherwise, use the JSON value if not None, otherwise use the default value
            values.append(json_value if json_value is not None else value)

    # This next section is about making the LoCon parameters visible if LoRA_type = 'Standard'
    if my_data.get("LoRA_type", "Standard") in {
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
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
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
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    lora_network_weights,
    dim_from_weights,
    color_aug,
    flip_aug,
    clip_skip,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
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
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    LoRA_type,
    factor,
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
    use_wandb,
    wandb_api_key,
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
    LyCORIS_preset,
    debiased_estimation_loss,
):
    # Get list of function parameters and values
    parameters = list(locals().items())
    global command_running

    print_only_bool = True if print_only.get("label") == "True" else False
    log.info(f"Start training LoRA {LoRA_type} ...")
    headless_bool = True if headless.get("label") == "True" else False

    if pretrained_model_name_or_path == "":
        output_message(
            msg="Source model information is missing", headless=headless_bool
        )
        return

    if train_data_dir == "":
        output_message(msg="Image folder path is missing", headless=headless_bool)
        return

    # Check if there are files with the same filename but different image extension... warn the user if it is the case.
    check_duplicate_filenames(train_data_dir)

    if not os.path.exists(train_data_dir):
        output_message(msg="Image folder does not exist", headless=headless_bool)
        return

    if not verify_image_folder_pattern(train_data_dir):
        return

    if reg_data_dir != "":
        if not os.path.exists(reg_data_dir):
            output_message(
                msg="Regularisation folder does not exist",
                headless=headless_bool,
            )
            return

        if not verify_image_folder_pattern(reg_data_dir):
            return

    if output_dir == "":
        output_message(msg="Output folder path is missing", headless=headless_bool)
        return

    if int(bucket_reso_steps) < 1:
        output_message(
            msg="Bucket resolution steps need to be greater than 0",
            headless=headless_bool,
        )
        return

    if noise_offset == "":
        noise_offset = 0

    if float(noise_offset) > 1 or float(noise_offset) < 0:
        output_message(
            msg="Noise offset need to be a value between 0 and 1",
            headless=headless_bool,
        )
        return

    # if float(noise_offset) > 0 and (
    #     multires_noise_iterations > 0 or multires_noise_discount > 0
    # ):
    #     output_message(
    #         msg="noise offset and multires_noise can't be set at the same time. Only use one or the other.",
    #         title='Error',
    #         headless=headless_bool,
    #     )
    #     return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if stop_text_encoder_training_pct > 0:
        output_message(
            msg='Output "stop text encoder training" is not yet supported. Ignoring',
            headless=headless_bool,
        )
        stop_text_encoder_training_pct = 0

    if not print_only_bool and check_if_model_exist(
        output_name, output_dir, save_model_as, headless=headless_bool
    ):
        return

    # if optimizer == 'Adafactor' and lr_warmup != '0':
    #     output_message(
    #         msg="Warning: lr_scheduler is set to 'Adafactor', so 'LR warmup (% of steps)' will be considered 0.",
    #         title='Warning',
    #         headless=headless_bool,
    #     )
    #     lr_warmup = '0'

    # If string is empty set string to 0.
    if text_encoder_lr == "":
        text_encoder_lr = 0
    if unet_lr == "":
        unet_lr = 0

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
            log.info(f"Folder {folder}: {steps} steps")

            total_steps += steps

        except ValueError:
            # Handle the case where the folder name does not contain an underscore
            log.info(f"Error: '{folder}' does not contain an underscore, skipping...")

    if reg_data_dir == "":
        reg_factor = 1
    else:
        log.warning(
            "Regularisation images are used... Will double the number of steps required..."
        )
        reg_factor = 2

    log.info(f"Total steps: {total_steps}")
    log.info(f"Train batch size: {train_batch_size}")
    log.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    log.info(f"Epoch: {epoch}")
    log.info(f"Regulatization factor: {reg_factor}")

    if max_train_steps == "" or max_train_steps == "0":
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
    if stop_text_encoder_training_pct == None:
        stop_text_encoder_training = 0
    else:
        stop_text_encoder_training = math.ceil(
            float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
        )
    log.info(f"stop_text_encoder_training = {stop_text_encoder_training}")

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    log.info(f"lr_warmup_steps = {lr_warmup_steps}")

    run_cmd = "accelerate launch"

    run_cmd += run_cmd_advanced_training(
        num_processes=num_processes,
        num_machines=num_machines,
        multi_gpu=multi_gpu,
        gpu_ids=gpu_ids,
        num_cpu_threads_per_process=num_cpu_threads_per_process,
    )

    if sdxl:
        run_cmd += fr' "{scriptdir}/sd-scripts/sdxl_train_network.py"'
    else:
        run_cmd += fr' "{scriptdir}/sd-scripts/train_network.py"'

    if LoRA_type == "LyCORIS/Diag-OFT":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "module_dropout={module_dropout}" "use_tucker={use_tucker}" "use_scalar={use_scalar}" "rank_dropout_scale={rank_dropout_scale}" "constrain={constrain}" "rescaled={rescaled}" "algo=diag-oft" '

    if LoRA_type == "LyCORIS/DyLoRA":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "use_tucker={use_tucker}" "block_size={unit}" "rank_dropout={rank_dropout}" "module_dropout={module_dropout}" "algo=dylora" "train_norm={train_norm}"'

    if LoRA_type == "LyCORIS/GLoRA":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "rank_dropout={rank_dropout}" "module_dropout={module_dropout}" "rank_dropout_scale={rank_dropout_scale}" "algo=glora" "train_norm={train_norm}"'

    if LoRA_type == "LyCORIS/iA3":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "train_on_input={train_on_input}" "algo=ia3"'

    if LoRA_type == "LoCon" or LoRA_type == "LyCORIS/LoCon":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "rank_dropout={rank_dropout}" "module_dropout={module_dropout}" "use_tucker={use_tucker}" "use_scalar={use_scalar}" "rank_dropout_scale={rank_dropout_scale}" "algo=locon" "train_norm={train_norm}"'

    if LoRA_type == "LyCORIS/LoHa":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "rank_dropout={rank_dropout}" "module_dropout={module_dropout}" "use_tucker={use_tucker}" "use_scalar={use_scalar}" "rank_dropout_scale={rank_dropout_scale}" "algo=loha" "train_norm={train_norm}"'

    if LoRA_type == "LyCORIS/LoKr":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "rank_dropout={rank_dropout}" "module_dropout={module_dropout}" "factor={factor}" "use_cp={use_cp}" "use_scalar={use_scalar}" "decompose_both={decompose_both}" "rank_dropout_scale={rank_dropout_scale}" "algo=lokr" "train_norm={train_norm}"'

    if LoRA_type == "LyCORIS/Native Fine-Tuning":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "rank_dropout={rank_dropout}" "module_dropout={module_dropout}" "use_tucker={use_tucker}" "use_scalar={use_scalar}" "rank_dropout_scale={rank_dropout_scale}" "algo=full" "train_norm={train_norm}"'

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

        network_args = ""
        if LoRA_type == "Kohya LoCon":
            network_args += f' conv_dim="{conv_dim}" conv_alpha="{conv_alpha}"'

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

    if LoRA_type in [
        "LoRA-FA",
    ]:
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
                network_args += f' {key}="{value}"'

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
                network_args += f' {key}="{value}"'

    network_train_text_encoder_only = False
    network_train_unet_only = False

    # Convert learning rates to float once and store the result for re-use
    if text_encoder_lr is None:
        output_message(
            msg="Please input valid Text Encoder learning rate (between 0 and 1)", headless=headless_bool
        )
        return
    if unet_lr is None:
        output_message(
            msg="Please input valid Unet learning rate (between 0 and 1)", headless=headless_bool
        )
        return
    text_encoder_lr_float = float(text_encoder_lr)
    unet_lr_float = float(unet_lr)
    
    

    # Determine the training configuration based on learning rate values
    if text_encoder_lr_float == 0 and unet_lr_float == 0:
        if float(learning_rate) == 0:
            output_message(
                msg="Please input learning rate values.", headless=headless_bool
            )
            return
    elif text_encoder_lr_float != 0 and unet_lr_float == 0:
        network_train_text_encoder_only = True
    elif text_encoder_lr_float == 0 and unet_lr_float != 0:
        network_train_unet_only = True
    # If both learning rates are non-zero, no specific flags need to be set

    run_cmd += run_cmd_advanced_training(
        adaptive_noise_scale=adaptive_noise_scale,
        additional_parameters=additional_parameters,
        bucket_no_upscale=bucket_no_upscale,
        bucket_reso_steps=bucket_reso_steps,
        cache_latents=cache_latents,
        cache_latents_to_disk=cache_latents_to_disk,
        cache_text_encoder_outputs=True if sdxl and sdxl_cache_text_encoder_outputs else None,
        caption_dropout_every_n_epochs=caption_dropout_every_n_epochs,
        caption_dropout_rate=caption_dropout_rate,
        caption_extension=caption_extension,
        clip_skip=clip_skip,
        color_aug=color_aug,
        debiased_estimation_loss=debiased_estimation_loss,
        dim_from_weights=dim_from_weights,
        enable_bucket=enable_bucket,
        epoch=epoch,
        flip_aug=flip_aug,
        fp8_base=fp8_base,
        full_bf16=full_bf16,
        full_fp16=full_fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        keep_tokens=keep_tokens,
        learning_rate=learning_rate,
        logging_dir=logging_dir,
        lora_network_weights=lora_network_weights,
        lr_scheduler=lr_scheduler,
        lr_scheduler_args=lr_scheduler_args,
        lr_scheduler_num_cycles=lr_scheduler_num_cycles,
        lr_scheduler_power=lr_scheduler_power,
        lr_warmup_steps=lr_warmup_steps,
        max_bucket_reso=max_bucket_reso,
        max_data_loader_n_workers=max_data_loader_n_workers,
        max_grad_norm=max_grad_norm,
        max_resolution=max_resolution,
        max_timestep=max_timestep,
        max_token_length=max_token_length,
        max_train_epochs=max_train_epochs,
        max_train_steps=max_train_steps,
        mem_eff_attn=mem_eff_attn,
        min_bucket_reso=min_bucket_reso,
        min_snr_gamma=min_snr_gamma,
        min_timestep=min_timestep,
        mixed_precision=mixed_precision,
        multires_noise_discount=multires_noise_discount,
        multires_noise_iterations=multires_noise_iterations,
        network_alpha=network_alpha,
        network_args=network_args,
        network_dim=network_dim,
        network_dropout=network_dropout,
        network_module=network_module,
        network_train_unet_only=network_train_unet_only,
        network_train_text_encoder_only=network_train_text_encoder_only,
        no_half_vae=True if sdxl and sdxl_no_half_vae else None,
        # no_token_padding=no_token_padding,
        noise_offset=noise_offset,
        noise_offset_type=noise_offset_type,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        output_dir=output_dir,
        output_name=output_name,
        persistent_data_loader_workers=persistent_data_loader_workers,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        prior_loss_weight=prior_loss_weight,
        random_crop=random_crop,
        reg_data_dir=reg_data_dir,
        resume=resume,
        save_every_n_epochs=save_every_n_epochs,
        save_every_n_steps=save_every_n_steps,
        save_last_n_steps=save_last_n_steps,
        save_last_n_steps_state=save_last_n_steps_state,
        save_model_as=save_model_as,
        save_precision=save_precision,
        save_state=save_state,
        scale_v_pred_loss_like_noise_pred=scale_v_pred_loss_like_noise_pred,
        scale_weight_norms=scale_weight_norms,
        seed=seed,
        shuffle_caption=shuffle_caption,
        stop_text_encoder_training=stop_text_encoder_training,
        text_encoder_lr=text_encoder_lr,
        train_batch_size=train_batch_size,
        train_data_dir=train_data_dir,
        training_comment=training_comment,
        unet_lr=unet_lr,
        use_wandb=use_wandb,
        v2=v2,
        v_parameterization=v_parameterization,
        v_pred_like_loss=v_pred_like_loss,
        vae=vae,
        vae_batch_size=vae_batch_size,
        wandb_api_key=wandb_api_key,
        weighted_captions=weighted_captions,
        xformers=xformers,
    )

    run_cmd += run_cmd_sample(
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_sampler,
        sample_prompts,
        output_dir,
    )

    if print_only_bool:
        log.warning(
            "Here is the trainer command as a reference. It will not be executed:\n"
        )
        print(run_cmd)

        save_to_file(run_cmd)
    else:
        # Saving config file for model
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(output_dir, f"{output_name}_{formatted_datetime}.json")

        log.info(f"Saving training config to {file_path}...")

        SaveConfigFile(
            parameters=parameters,
            file_path=file_path,
            exclusion=["file_path", "save_as", "headless", "print_only"],
        )

        log.info(run_cmd)
        # Run the command
        executor.execute_command(run_cmd=run_cmd)

        # # check if output_dir/last is a folder... therefore it is a diffuser model
        # last_dir = pathlib.Path(f'{output_dir}/{output_name}')

        # if not last_dir.is_dir():
        #     # Copy inference model for v2 if required
        #     save_inference_file(
        #         output_dir, v2, v_parameterization, output_name
        #     )


def lora_tab(
    train_data_dir_input=gr.Textbox(),
    reg_data_dir_input=gr.Textbox(),
    output_dir_input=gr.Textbox(),
    logging_dir_input=gr.Textbox(),
    headless=False,
):
    dummy_db_true = gr.Label(value=True, visible=False)
    dummy_db_false = gr.Label(value=False, visible=False)
    dummy_headless = gr.Label(value=headless, visible=False)

    with gr.Tab("Training"):
        gr.Markdown(
            "Train a custom model using kohya train network LoRA python code..."
        )

        # Setup Configuration Files Gradio
        config = ConfigurationFile(headless)

        source_model = SourceModel(
            save_model_as_choices=[
                "ckpt",
                "safetensors",
            ],
            headless=headless,
        )

        with gr.Tab("Folders"):
            folders = Folders(headless=headless)

        with gr.Tab("Parameters"):

            def list_presets(path):
                json_files = []
                
                # Insert an empty string at the beginning
                json_files.insert(0, "none")

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
                choices=list_presets(f"{presets_dir}/lora"),
                elem_id="myDropdown",
                value="none"
            )

            with gr.Tab("Basic", elem_id="basic_tab"):
                with gr.Row():
                    LoRA_type = gr.Dropdown(
                        label="LoRA type",
                        choices=[
                            "Kohya DyLoRA",
                            "Kohya LoCon",
                            "LoRA-FA",
                            "LyCORIS/DyLoRA",
                            "LyCORIS/iA3",
                            "LyCORIS/Diag-OFT",
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
                        choices=[
                            "attn-mlp",
                            "attn-only",
                            "full",
                            "full-lin",
                            "unet-transformer-only",
                            "unet-convblock-only",
                        ],
                        value="full",
                        visible=False,
                        interactive=True
                        # info="https://github.com/KohakuBlueleaf/LyCORIS/blob/0006e2ffa05a48d8818112d9f70da74c0cd30b99/docs/Preset.md"
                    )
                    with gr.Group():
                        with gr.Row():
                            lora_network_weights = gr.Textbox(
                                label="LoRA network weights",
                                placeholder="(Optional)",
                                info="Path to an existing LoRA network weights to resume training from",
                            )
                            lora_network_weights_file = gr.Button(
                                document_symbol,
                                elem_id="open_folder_small",
                                visible=(not headless),
                            )
                            lora_network_weights_file.click(
                                get_any_file_path,
                                inputs=[lora_network_weights],
                                outputs=lora_network_weights,
                                show_progress=False,
                            )
                            dim_from_weights = gr.Checkbox(
                                label="DIM from weights",
                                value=False,
                                info="Automatically determine the dim(rank) from the weight file.",
                            )
                basic_training = BasicTraining(
                    learning_rate_value="0.0001",
                    lr_scheduler_value="cosine",
                    lr_warmup_value="10",
                    sdxl_checkbox=source_model.sdxl_checkbox,
                )

                with gr.Row():
                    text_encoder_lr = gr.Number(
                        label="Text Encoder learning rate",
                        value="0.0001",
                        info="Optional",
                        minimum=0,
                        maximum=1,
                    )
                    
                    unet_lr = gr.Number(
                        label="Unet learning rate",
                        value="0.0001",
                        info="Optional",
                        minimum=0,
                        maximum=1,
                    )

                # Add SDXL Parameters
                sdxl_params = SDXLParameters(source_model.sdxl_checkbox)

                with gr.Row():
                    factor = gr.Slider(
                        label="LoKr factor",
                        value=-1,
                        minimum=-1,
                        maximum=64,
                        step=1,
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
                    rank_dropout_scale = gr.Checkbox(
                        value=False,
                        label="Rank Dropout Scale",
                        info="Adjusts the scale of the rank dropout to maintain the average dropout rate, ensuring more consistent regularization across different layers.",
                        visible=False,
                    )
                    constrain = gr.Number(
                        value="0.0",
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
                        info=" Controls whether both input and output dimensions of the layer's weights are decomposed into smaller matrices for reparameterization.",
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
                        minimum=0.1,
                        maximum=1024,
                        label="Network Alpha",
                        value=1,
                        step=0.1,
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
                with gr.Row(visible=False) as kohya_dylora:
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
                        log.info("LoRA type changed...")

                        lora_settings_config = {
                            "network_row": {
                                "gr_type": gr.Row,
                                "update_params": {
                                    "visible": LoRA_type
                                    in {
                                        "Kohya DyLoRA",
                                        "Kohya LoCon",
                                        "LoRA-FA",
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
                                        "Standard",
                                        "Kohya DyLoRA",
                                        "Kohya LoCon",
                                        "LoRA-FA",
                                    },
                                },
                            },
                            "kohya_dylora": {
                                "gr_type": gr.Row,
                                "update_params": {
                                    "visible": LoRA_type
                                    in {
                                        "Kohya DyLoRA",
                                        "LyCORIS/DyLoRA",
                                    },
                                },
                            },
                            "lora_network_weights": {
                                "gr_type": gr.Textbox,
                                "update_params": {
                                    "visible": LoRA_type
                                    in {
                                        "Standard",
                                        "LoCon",
                                        "Kohya DyLoRA",
                                        "Kohya LoCon",
                                        "LoRA-FA",
                                        "LyCORIS/Diag-OFT",
                                        "LyCORIS/DyLoRA",
                                        "LyCORIS/GLoRA",
                                        "LyCORIS/LoHa",
                                        "LyCORIS/LoCon",
                                        "LyCORIS/LoKr",
                                    },
                                },
                            },
                            "lora_network_weights_file": {
                                "gr_type": gr.Button,
                                "update_params": {
                                    "visible": LoRA_type
                                    in {
                                        "Standard",
                                        "LoCon",
                                        "Kohya DyLoRA",
                                        "Kohya LoCon",
                                        "LoRA-FA",
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
                                        "Standard",
                                        "LoCon",
                                        "Kohya DyLoRA",
                                        "Kohya LoCon",
                                        "LoRA-FA",
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
                                    "maximum": 100000
                                    if LoRA_type
                                    in {
                                        "LyCORIS/LoHa",
                                        "LyCORIS/LoKr",
                                        "LyCORIS/Diag-OFT",
                                    }
                                    else 512,
                                    "value": conv_dim,  # if conv_dim > 512 else conv_dim,
                                },
                            },
                            "network_dim": {
                                "gr_type": gr.Slider,
                                "update_params": {
                                    "maximum": 100000
                                    if LoRA_type
                                    in {
                                        "LyCORIS/LoHa",
                                        "LyCORIS/LoKr",
                                        "LyCORIS/Diag-OFT",
                                    }
                                    else 512,
                                    "value": network_dim,  # if network_dim > 512 else network_dim,
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
                                        "LyCORIS/Diag-OFT",
                                    },
                                },
                            },
                            "rescaled": {
                                "gr_type": gr.Checkbox,
                                "update_params": {
                                    "visible": LoRA_type
                                    in {
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
                                        "LyCORIS/Diag-OFT",
                                        "LyCORIS/GLoRA",
                                        "LyCORIS/LoCon",
                                        "LyCORIS/LoHa",
                                        "LyCORIS/LoKr",
                                        "LyCORIS/Native Fine-Tuning",
                                    },
                                },
                            },
                        }

                        results = []
                        for attr, settings in lora_settings_config.items():
                            update_params = settings["update_params"]

                            results.append(settings["gr_type"](**update_params))

                        return tuple(results)

            with gr.Tab("Advanced", elem_id="advanced_tab"):
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
                    headless=headless, training_type="lora"
                )
                advanced_training.color_aug.change(
                    color_aug_changed,
                    inputs=[advanced_training.color_aug],
                    outputs=[basic_training.cache_latents],
                )

            with gr.Tab("Samples", elem_id="samples_tab"):
                sample = SampleImages()

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
                    kohya_dylora,
                    lora_network_weights,
                    lora_network_weights_file,
                    dim_from_weights,
                    factor,
                    conv_dim,
                    network_dim,
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
                ],
            )

        with gr.Tab("Dataset Preparation"):
            gr.Markdown(
                "This section provide Dreambooth tools to help setup your dataset..."
            )
            gradio_dreambooth_folder_creation_tab(
                train_data_dir_input=folders.train_data_dir,
                reg_data_dir_input=folders.reg_data_dir,
                output_dir_input=folders.output_dir,
                logging_dir_input=folders.logging_dir,
                headless=headless,
            )
            gradio_dataset_balancing_tab(headless=headless)

        with gr.Row():
            button_run = gr.Button("Start training", variant="primary")

            button_stop_training = gr.Button("Stop training")

        button_print = gr.Button("Print training command")

        # Setup gradio tensorboard buttons
        (
            button_start_tensorboard,
            button_stop_tensorboard,
        ) = gradio_tensorboard()

        button_start_tensorboard.click(
            start_tensorboard,
            inputs=[dummy_headless, folders.logging_dir],
            show_progress=False,
        )

        button_stop_tensorboard.click(
            stop_tensorboard,
            show_progress=False,
        )

        settings_list = [
            source_model.pretrained_model_name_or_path,
            source_model.v2,
            source_model.v_parameterization,
            source_model.sdxl_checkbox,
            folders.logging_dir,
            folders.train_data_dir,
            folders.reg_data_dir,
            folders.output_dir,
            basic_training.max_resolution,
            basic_training.learning_rate,
            basic_training.lr_scheduler,
            basic_training.lr_warmup,
            basic_training.train_batch_size,
            basic_training.epoch,
            basic_training.save_every_n_epochs,
            basic_training.mixed_precision,
            basic_training.save_precision,
            basic_training.seed,
            basic_training.num_cpu_threads_per_process,
            basic_training.cache_latents,
            basic_training.cache_latents_to_disk,
            basic_training.caption_extension,
            basic_training.enable_bucket,
            advanced_training.gradient_checkpointing,
            advanced_training.fp8_base,
            advanced_training.full_fp16,
            # advanced_training.no_token_padding,
            basic_training.stop_text_encoder_training,
            basic_training.min_bucket_reso,
            basic_training.max_bucket_reso,
            advanced_training.xformers,
            source_model.save_model_as,
            advanced_training.shuffle_caption,
            advanced_training.save_state,
            advanced_training.resume,
            advanced_training.prior_loss_weight,
            text_encoder_lr,
            unet_lr,
            network_dim,
            lora_network_weights,
            dim_from_weights,
            advanced_training.color_aug,
            advanced_training.flip_aug,
            advanced_training.clip_skip,
            advanced_training.num_processes,
            advanced_training.num_machines,
            advanced_training.multi_gpu,
            advanced_training.gpu_ids,
            advanced_training.gradient_accumulation_steps,
            advanced_training.mem_eff_attn,
            folders.output_name,
            source_model.model_list,
            advanced_training.max_token_length,
            basic_training.max_train_epochs,
            basic_training.max_train_steps,
            advanced_training.max_data_loader_n_workers,
            network_alpha,
            folders.training_comment,
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
            basic_training.max_grad_norm,
            advanced_training.noise_offset_type,
            advanced_training.noise_offset,
            advanced_training.adaptive_noise_scale,
            advanced_training.multires_noise_iterations,
            advanced_training.multires_noise_discount,
            LoRA_type,
            factor,
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
            advanced_training.use_wandb,
            advanced_training.wandb_api_key,
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
            LyCORIS_preset,
            advanced_training.debiased_estimation_loss,
        ]

        config.button_open_config.click(
            open_configuration,
            inputs=[dummy_db_true, dummy_db_false, config.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[config.config_file_name]
            + settings_list
            + [training_preset, convolution_row],
            show_progress=False,
        )

        config.button_load_config.click(
            open_configuration,
            inputs=[dummy_db_false, dummy_db_false, config.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[config.config_file_name]
            + settings_list
            + [training_preset, convolution_row],
            show_progress=False,
        )

        training_preset.input(
            open_configuration,
            inputs=[dummy_db_false, dummy_db_true, config.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[gr.Textbox(visible=False)] + settings_list + [training_preset, convolution_row],
            show_progress=False,
        )

        config.button_save_config.click(
            save_configuration,
            inputs=[dummy_db_false, config.config_file_name] + settings_list,
            outputs=[config.config_file_name],
            show_progress=False,
        )

        config.button_save_as_config.click(
            save_configuration,
            inputs=[dummy_db_true, config.config_file_name] + settings_list,
            outputs=[config.config_file_name],
            show_progress=False,
        )

        button_run.click(
            train_model,
            inputs=[dummy_headless] + [dummy_db_false] + settings_list,
            show_progress=False,
        )

        button_stop_training.click(executor.kill_command)

        button_print.click(
            train_model,
            inputs=[dummy_headless] + [dummy_db_true] + settings_list,
            show_progress=False,
        )

    with gr.Tab("Tools"):
        lora_tools = LoRATools(folders=folders, headless=headless)

    with gr.Tab("Guides"):
        gr.Markdown("This section provide Various LoRA guides and information...")
        if os.path.exists(fr"{scriptdir}/docs/LoRA/top_level.md"):
            with open(
                os.path.join(fr"{scriptdir}/docs/LoRA/top_level.md"), "r", encoding="utf8"
            ) as file:
                guides_top_level = file.read() + "\n"
            gr.Markdown(guides_top_level)

    return (
        folders.train_data_dir,
        folders.reg_data_dir,
        folders.output_dir,
        folders.logging_dir,
    )
