import gradio as gr
import json
import math
import os
import sys
import pathlib
from datetime import datetime
from .common_gui import (
    get_file_path,
    get_saveasfile_path,
    color_aug_changed,
    save_inference_file,
    run_cmd_advanced_training,
    update_my_data,
    check_if_model_exist,
    SaveConfigFile,
    save_to_file,
    scriptdir,
    validate_paths,
)
from .class_configuration_file import ConfigurationFile
from .class_source_model import SourceModel
from .class_basic_training import BasicTraining
from .class_advanced_training import AdvancedTraining
from .class_folders import Folders
from .class_command_executor import CommandExecutor
from .class_sdxl_parameters import SDXLParameters
from .tensorboard_gui import (
    gradio_tensorboard,
    start_tensorboard,
    stop_tensorboard,
)
from .dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from .dataset_balancing_gui import gradio_dataset_balancing_tab
from .class_sample_images import SampleImages, run_cmd_sample

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = CommandExecutor()

PYTHON = sys.executable


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
    resume,
    prior_loss_weight,
    color_aug,
    flip_aug,
    clip_skip,
    vae,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
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
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
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
    resume,
    prior_loss_weight,
    color_aug,
    flip_aug,
    clip_skip,
    vae,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
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
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
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
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    ask_for_file = True if ask_for_file.get("label") == "True" else False

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
    color_aug,
    flip_aug,
    clip_skip,
    vae,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
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
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
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
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    print_only_bool = True if print_only.get("label") == "True" else False
    log.info(f"Start training Dreambooth...")

    headless_bool = True if headless.get("label") == "True" else False

    # This function validates files or folder paths. Simply add new variables containing file of folder path
    # to validate below
    if not validate_paths(
        output_dir=output_dir,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        train_data_dir=train_data_dir,
        reg_data_dir=reg_data_dir,
        headless=headless_bool,
        logging_dir=logging_dir,
        log_tracker_config=log_tracker_config,
        resume=resume,
        vae=vae,
        dataset_config=dataset_config,
    ):
        return

    if not print_only_bool and check_if_model_exist(
        output_name, output_dir, save_model_as, headless=headless_bool
    ):
        return

    if dataset_config:
        log.info("Dataset config toml file used, skipping total_steps, train_batch_size, gradient_accumulation_steps, epoch, reg_factor, max_train_steps calculations...")
    else:
        # Get a list of all subfolders in train_data_dir, excluding hidden folders
        subfolders = [
            f
            for f in os.listdir(train_data_dir)
            if os.path.isdir(os.path.join(train_data_dir, f)) and not f.startswith(".")
        ]

        # Check if subfolders are present. If not let the user know and return
        if not subfolders:
            log.info(f"No {subfolders} were found in train_data_dir can't train...")
            return

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
            log.info(f"No images were found in folder {train_data_dir}... please rectify!")
            return

        # Print the result
        # log.info(f"{total_steps} total steps")

        if reg_data_dir == "":
            reg_factor = 1
        else:
            log.info(
                f"Regularisation images are used... Will double the number of steps required..."
            )
            reg_factor = 2

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
    if int(stop_text_encoder_training_pct) == -1:
        stop_text_encoder_training = -1
    elif stop_text_encoder_training_pct == None or (not max_train_steps == "" or not max_train_steps == "0"):
        stop_text_encoder_training = 0
    else:
        stop_text_encoder_training = math.ceil(
            float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
        )
    log.info(f"stop_text_encoder_training = {stop_text_encoder_training}")

    if not max_train_steps == "":
        lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    else:
        lr_warmup_steps = 0
    log.info(f"lr_warmup_steps = {lr_warmup_steps}")

    # run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process} "train_db.py"'
    run_cmd = "accelerate launch"

    run_cmd += run_cmd_advanced_training(
        num_processes=num_processes,
        num_machines=num_machines,
        multi_gpu=multi_gpu,
        gpu_ids=gpu_ids,
        num_cpu_threads_per_process=num_cpu_threads_per_process,
    )

    if sdxl:
        run_cmd += rf' "{scriptdir}/sd-scripts/sdxl_train.py"'
    else:
        run_cmd += rf' "{scriptdir}/sd-scripts/train_db.py"'

    # Initialize a dictionary with always-included keyword arguments
    kwargs_for_training = {
        "adaptive_noise_scale": adaptive_noise_scale,
        "additional_parameters": additional_parameters,
        "bucket_no_upscale": bucket_no_upscale,
        "bucket_reso_steps": bucket_reso_steps,
        "cache_latents": cache_latents,
        "cache_latents_to_disk": cache_latents_to_disk,
        "caption_dropout_every_n_epochs": caption_dropout_every_n_epochs,
        "caption_dropout_rate": caption_dropout_rate,
        "caption_extension": caption_extension,
        "clip_skip": clip_skip,
        "color_aug": color_aug,
        "dataset_config": dataset_config,
        "enable_bucket": enable_bucket,
        "epoch": epoch,
        "flip_aug": flip_aug,
        "full_bf16": full_bf16,
        "full_fp16": full_fp16,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "keep_tokens": keep_tokens,
        "learning_rate": learning_rate,
        "logging_dir": logging_dir,
        "log_tracker_name": log_tracker_name,
        "log_tracker_config": log_tracker_config,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_args": lr_scheduler_args,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles,
        "lr_scheduler_power": lr_scheduler_power,
        "lr_warmup_steps": lr_warmup_steps,
        "max_bucket_reso": max_bucket_reso,
        "max_data_loader_n_workers": max_data_loader_n_workers,
        "max_resolution": max_resolution,
        "max_timestep": max_timestep,
        "max_token_length": max_token_length,
        "max_train_epochs": max_train_epochs,
        "max_train_steps": max_train_steps,
        "mem_eff_attn": mem_eff_attn,
        "min_bucket_reso": min_bucket_reso,
        "min_snr_gamma": min_snr_gamma,
        "min_timestep": min_timestep,
        "mixed_precision": mixed_precision,
        "multires_noise_discount": multires_noise_discount,
        "multires_noise_iterations": multires_noise_iterations,
        "no_token_padding": no_token_padding,
        "noise_offset": noise_offset,
        "noise_offset_type": noise_offset_type,
        "optimizer": optimizer,
        "optimizer_args": optimizer_args,
        "output_dir": output_dir,
        "output_name": output_name,
        "persistent_data_loader_workers": persistent_data_loader_workers,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "prior_loss_weight": prior_loss_weight,
        "random_crop": random_crop,
        "reg_data_dir": reg_data_dir,
        "resume": resume,
        "save_every_n_epochs": save_every_n_epochs,
        "save_every_n_steps": save_every_n_steps,
        "save_last_n_steps": save_last_n_steps,
        "save_last_n_steps_state": save_last_n_steps_state,
        "save_model_as": save_model_as,
        "save_precision": save_precision,
        "save_state": save_state,
        "scale_v_pred_loss_like_noise_pred": scale_v_pred_loss_like_noise_pred,
        "seed": seed,
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
        "xformers": xformers,
    }

    # Conditionally include specific keyword arguments based on sdxl
    if sdxl:
        kwargs_for_training["learning_rate_te1"] = learning_rate_te1
        kwargs_for_training["learning_rate_te2"] = learning_rate_te2
    else:
        kwargs_for_training["learning_rate_te"] = learning_rate_te

    # Pass the dynamically constructed keyword arguments to the function
    run_cmd += run_cmd_advanced_training(**kwargs_for_training)

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
        # config_dir = os.path.dirname(os.path.dirname(train_data_dir))
        file_path = os.path.join(output_dir, f"{output_name}_{formatted_datetime}.json")

        log.info(f"Saving training config to {file_path}...")

        SaveConfigFile(
            parameters=parameters,
            file_path=file_path,
            exclusion=["file_path", "save_as", "headless", "print_only"],
        )

        log.info(run_cmd)

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            rf"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"
        )

        # Run the command

        executor.execute_command(run_cmd=run_cmd, env=env)

        # # check if output_dir/last is a folder... therefore it is a diffuser model
        # last_dir = pathlib.Path(f"{output_dir}/{output_name}")

        # if not last_dir.is_dir():
        #     # Copy inference model for v2 if required
        #     save_inference_file(output_dir, v2, v_parameterization, output_name)


def dreambooth_tab(
    # train_data_dir=gr.Textbox(),
    # reg_data_dir=gr.Textbox(),
    # output_dir=gr.Textbox(),
    # logging_dir=gr.Textbox(),
    headless=False,
    config: dict = {},
):
    dummy_db_true = gr.Label(value=True, visible=False)
    dummy_db_false = gr.Label(value=False, visible=False)
    dummy_headless = gr.Label(value=headless, visible=False)

    with gr.Tab("Training"), gr.Column(variant="compact"):
        gr.Markdown("Train a custom model using kohya dreambooth python code...")

        with gr.Column():
            source_model = SourceModel(headless=headless, config=config)

        with gr.Accordion("Folders", open=False), gr.Group():
            folders = Folders(headless=headless, config=config)
        with gr.Accordion("Parameters", open=False), gr.Column():
            with gr.Group(elem_id="basic_tab"):
                basic_training = BasicTraining(
                    learning_rate_value="1e-5",
                    lr_scheduler_value="cosine",
                    lr_warmup_value="10",
                    dreambooth=True,
                    sdxl_checkbox=source_model.sdxl_checkbox,
                )

                # # Add SDXL Parameters
                # sdxl_params = SDXLParameters(source_model.sdxl_checkbox, show_sdxl_cache_text_encoder_outputs=False)

            with gr.Accordion("Advanced", open=False, elem_id="advanced_tab"):
                advanced_training = AdvancedTraining(headless=headless, config=config)
                advanced_training.color_aug.change(
                    color_aug_changed,
                    inputs=[advanced_training.color_aug],
                    outputs=[basic_training.cache_latents],
                )

            with gr.Accordion("Samples", open=False, elem_id="samples_tab"):
                sample = SampleImages()

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
            )
            gradio_dataset_balancing_tab(headless=headless)

        # Setup Configuration Files Gradio
        with gr.Accordion("Configuration", open=False):
            configuration = ConfigurationFile(headless=headless, config=config)

        with gr.Column(), gr.Group():
            with gr.Row():
                button_run = gr.Button("Start training", variant="primary")

                button_stop_training = gr.Button("Stop training")

            button_print = gr.Button("Print training command")

        # Setup gradio tensorboard buttons
        with gr.Column(), gr.Group():
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
            basic_training.mixed_precision,
            source_model.save_precision,
            basic_training.seed,
            basic_training.num_cpu_threads_per_process,
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
            advanced_training.resume,
            advanced_training.prior_loss_weight,
            advanced_training.color_aug,
            advanced_training.flip_aug,
            advanced_training.clip_skip,
            advanced_training.vae,
            advanced_training.num_processes,
            advanced_training.num_machines,
            advanced_training.multi_gpu,
            advanced_training.gpu_ids,
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
            advanced_training.adaptive_noise_scale,
            advanced_training.multires_noise_iterations,
            advanced_training.multires_noise_discount,
            sample.sample_every_n_steps,
            sample.sample_every_n_epochs,
            sample.sample_sampler,
            sample.sample_prompts,
            advanced_training.additional_parameters,
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

        return (
            source_model.train_data_dir,
            folders.reg_data_dir,
            folders.output_dir,
            folders.logging_dir,
        )
