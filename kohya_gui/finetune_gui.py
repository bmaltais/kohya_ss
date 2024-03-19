import gradio as gr
import json
import math
import os
import subprocess
import sys
import pathlib
from datetime import datetime
from .common_gui import (
    get_file_path,
    get_saveasfile_path,
    save_inference_file,
    run_cmd_advanced_training,
    color_aug_changed,
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
from .class_sdxl_parameters import SDXLParameters
from .class_command_executor import CommandExecutor
from .tensorboard_gui import (
    gradio_tensorboard,
    start_tensorboard,
    stop_tensorboard,
)
from .class_sample_images import SampleImages, run_cmd_sample

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = CommandExecutor()

# from easygui import msgbox

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

PYTHON = sys.executable

presets_dir = fr'{scriptdir}/presets'

def save_configuration(
    save_as,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl_checkbox,
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
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    save_state,
    resume,
    gradient_checkpointing,
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
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
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
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    save_state,
    resume,
    gradient_checkpointing,
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
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    min_timestep,
    max_timestep,
    training_preset,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    ask_for_file = True if ask_for_file.get("label") == "True" else False
    apply_preset = True if apply_preset.get("label") == "True" else False

    # Check if we are "applying" a preset or a config
    if apply_preset:
        log.info(f"Applying preset {training_preset}...")
        file_path = fr'{presets_dir}/finetune/{training_preset}.json'
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
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    save_state,
    resume,
    gradient_checkpointing,
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
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    min_timestep,
    max_timestep,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    print_only_bool = True if print_only.get("label") == "True" else False
    log.info(f"Start Finetuning...")

    headless_bool = True if headless.get("label") == "True" else False

    if train_dir != "" and not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if not validate_paths(
        output_dir=output_dir,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        finetune_image_folder=image_folder,
        headless=headless_bool,
        logging_dir=logging_dir,
        log_tracker_config=log_tracker_config,
        resume=resume,
        dataset_config=dataset_config
    ):
        return

    if not print_only_bool and check_if_model_exist(output_name, output_dir, save_model_as, headless_bool):
        return

    if dataset_config:
        log.info("Dataset config toml file used, skipping caption json file, image buckets, total_steps, train_batch_size, gradient_accumulation_steps, epoch, reg_factor, max_train_steps creation...")
    else:   
        # create caption json file
        if generate_caption_database:
            run_cmd = fr'"{PYTHON}" "{scriptdir}/sd-scripts/finetune/merge_captions_to_metadata.py"'
            if caption_extension == "":
                run_cmd += f' --caption_extension=".caption"'
            else:
                run_cmd += f" --caption_extension={caption_extension}"
            run_cmd += fr' "{image_folder}"'
            run_cmd += fr' "{train_dir}/{caption_metadata_filename}"'
            if full_path:
                run_cmd += f" --full_path"

            log.info(run_cmd)

            env = os.environ.copy()
            env['PYTHONPATH'] = fr"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"

            if not print_only_bool:
                # Run the command
                subprocess.run(run_cmd, shell=True, env=env)

        # create images buckets
        if generate_image_buckets:
            run_cmd = fr'"{PYTHON}" "{scriptdir}/sd-scripts/finetune/prepare_buckets_latents.py"'
            run_cmd += fr' "{image_folder}"'
            run_cmd += fr' "{train_dir}/{caption_metadata_filename}"'
            run_cmd += fr' "{train_dir}/{latent_metadata_filename}"'
            run_cmd += fr' "{pretrained_model_name_or_path}"'
            run_cmd += f" --batch_size={batch_size}"
            run_cmd += f" --max_resolution={max_resolution}"
            run_cmd += f" --min_bucket_reso={min_bucket_reso}"
            run_cmd += f" --max_bucket_reso={max_bucket_reso}"
            run_cmd += f" --mixed_precision={mixed_precision}"
            # if flip_aug:
            #     run_cmd += f' --flip_aug'
            if full_path:
                run_cmd += f" --full_path"
            if sdxl_checkbox and sdxl_no_half_vae:
                log.info("Using mixed_precision = no because no half vae is selected...")
                run_cmd += f' --mixed_precision="no"'

            log.info(run_cmd)

            env = os.environ.copy()
            env['PYTHONPATH'] = fr"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"

            if not print_only_bool:
                # Run the command
                subprocess.run(run_cmd, shell=True, env=env)

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

    if max_train_steps != "":
        log.info(f"max_train_steps = {max_train_steps}")
        lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    else:
        lr_warmup_steps = 0
    log.info(f"lr_warmup_steps = {lr_warmup_steps}")

    run_cmd = "accelerate launch"

    run_cmd += run_cmd_advanced_training(
        num_processes=num_processes,
        num_machines=num_machines,
        multi_gpu=multi_gpu,
        gpu_ids=gpu_ids,
        num_cpu_threads_per_process=num_cpu_threads_per_process,
    )

    if sdxl_checkbox:
        run_cmd += fr' "{scriptdir}/sd-scripts/sdxl_train.py"'
    else:
        run_cmd += fr' "{scriptdir}/sd-scripts/fine_tune.py"'

    in_json = (
        fr"{train_dir}/{latent_metadata_filename}"
        if use_latent_files == "Yes"
        else fr"{train_dir}/{caption_metadata_filename}"
    )
    cache_text_encoder_outputs = sdxl_checkbox and sdxl_cache_text_encoder_outputs
    no_half_vae = sdxl_checkbox and sdxl_no_half_vae

    # Initialize a dictionary with always-included keyword arguments
    kwargs_for_training = {
        "adaptive_noise_scale": adaptive_noise_scale,
        "additional_parameters": additional_parameters,
        "block_lr": block_lr,
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
        "dataset_repeats": dataset_repeats,
        "enable_bucket": True,
        "flip_aug": flip_aug,
        "full_bf16": full_bf16,
        "full_fp16": full_fp16,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "in_json": in_json,
        "keep_tokens": keep_tokens,
        "learning_rate": learning_rate,
        "logging_dir": logging_dir,
        "log_tracker_name": log_tracker_name,
        "log_tracker_config": log_tracker_config,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_args": lr_scheduler_args,
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
        "noise_offset": noise_offset,
        "noise_offset_type": noise_offset_type,
        "optimizer": optimizer,
        "optimizer_args": optimizer_args,
        "output_dir": output_dir,
        "output_name": output_name,
        "persistent_data_loader_workers": persistent_data_loader_workers,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "random_crop": random_crop,
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
        "train_batch_size": train_batch_size,
        "train_data_dir": image_folder,
        "train_text_encoder": train_text_encoder,
        "use_wandb": use_wandb,
        "v2": v2,
        "v_parameterization": v_parameterization,
        "v_pred_like_loss": v_pred_like_loss,
        "vae_batch_size": vae_batch_size,
        "wandb_api_key": wandb_api_key,
        "wandb_run_name": wandb_run_name,
        "weighted_captions": weighted_captions,
        "xformers": xformers,
    }

    # Conditionally include specific keyword arguments based on sdxl_checkbox
    if sdxl_checkbox:
        kwargs_for_training["cache_text_encoder_outputs"] = cache_text_encoder_outputs
        kwargs_for_training["learning_rate_te1"] = learning_rate_te1
        kwargs_for_training["learning_rate_te2"] = learning_rate_te2
        kwargs_for_training["no_half_vae"] = no_half_vae
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
        env['PYTHONPATH'] = fr"{scriptdir}{os.pathsep}{scriptdir}/sd-scripts{os.pathsep}{env.get('PYTHONPATH', '')}"

        # Run the command
        executor.execute_command(run_cmd=run_cmd, env=env)

        # # check if output_dir/last is a folder... therefore it is a diffuser model
        # last_dir = pathlib.Path(f"{output_dir}/{output_name}")

        # if not last_dir.is_dir():
        #     # Copy inference model for v2 if required
        #     save_inference_file(output_dir, v2, v_parameterization, output_name)


def finetune_tab(headless=False, config: dict = {}):
    dummy_db_true = gr.Label(value=True, visible=False)
    dummy_db_false = gr.Label(value=False, visible=False)
    dummy_headless = gr.Label(value=headless, visible=False)
    with gr.Tab("Training"), gr.Column(variant="compact"):
        gr.Markdown("Train a custom model using kohya finetune python code...")

        with gr.Column():
            source_model = SourceModel(headless=headless, finetuning=True, config=config)
            image_folder = source_model.train_data_dir
            output_name = source_model.output_name

        with gr.Accordion("Folders", open=False), gr.Group():
            folders = Folders(headless=headless, finetune=True, config=config)
            output_dir = folders.output_dir
            logging_dir = folders.logging_dir
            train_dir = folders.reg_data_dir

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
                choices=[""] + list_presets(f"{presets_dir}/finetune"),
                elem_id="myDropdown",
            )

            with gr.Group(elem_id="basic_tab"):
                basic_training = BasicTraining(
                    learning_rate_value="1e-5",
                    finetuning=True,
                    sdxl_checkbox=source_model.sdxl_checkbox,
                )

                # Add SDXL Parameters
                sdxl_params = SDXLParameters(source_model.sdxl_checkbox)

                with gr.Row():
                    dataset_repeats = gr.Textbox(label="Dataset repeats", value=40)
                    train_text_encoder = gr.Checkbox(
                        label="Train text encoder", value=True
                    )

            with gr.Accordion("Advanced", open=False, elem_id="advanced_tab"):
                with gr.Row():
                    gradient_accumulation_steps = gr.Number(
                        label="Gradient accumulate steps", value="1",
                    )
                    block_lr = gr.Textbox(
                        label="Block LR (SDXL)",
                        placeholder="(Optional)",
                        info="Specify the different learning rates for each U-Net block. Specify 23 values separated by commas like 1e-3,1e-3 ... 1e-3",
                    )
                advanced_training = AdvancedTraining(headless=headless, finetuning=True, config=config)
                advanced_training.color_aug.change(
                    color_aug_changed,
                    inputs=[advanced_training.color_aug],
                    outputs=[
                        basic_training.cache_latents
                    ],  # Not applicable to fine_tune.py
                )

            with gr.Accordion("Samples", open=False, elem_id="samples_tab"):
                sample = SampleImages()

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
            inputs=[dummy_headless, logging_dir],
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
            basic_training.mixed_precision,
            source_model.save_precision,
            basic_training.seed,
            basic_training.num_cpu_threads_per_process,
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
            advanced_training.num_processes,
            advanced_training.num_machines,
            advanced_training.multi_gpu,
            advanced_training.gpu_ids,
            advanced_training.save_state,
            advanced_training.resume,
            advanced_training.gradient_checkpointing,
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
            weighted_captions,
            advanced_training.save_every_n_steps,
            advanced_training.save_last_n_steps,
            advanced_training.save_last_n_steps_state,
            advanced_training.use_wandb,
            advanced_training.wandb_api_key,
            advanced_training.wandb_run_name,
            advanced_training.log_tracker_name,
            advanced_training.log_tracker_config,
            advanced_training.scale_v_pred_loss_like_noise_pred,
            sdxl_params.sdxl_cache_text_encoder_outputs,
            sdxl_params.sdxl_no_half_vae,
            advanced_training.min_timestep,
            advanced_training.max_timestep,
        ]

        configuration.button_open_config.click(
            open_configuration,
            inputs=[dummy_db_true, dummy_db_false, configuration.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[configuration.config_file_name] + settings_list + [training_preset],
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
            outputs=[configuration.config_file_name] + settings_list + [training_preset],
            show_progress=False,
        )

        # config.button_load_config.click(
        #     open_configuration,
        #     inputs=[dummy_db_false, config.config_file_name] + settings_list,
        #     outputs=[config.config_file_name] + settings_list,
        #     show_progress=False,
        # )

        training_preset.input(
            open_configuration,
            inputs=[dummy_db_false, dummy_db_true, configuration.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[gr.Textbox(visible=False)] + settings_list + [training_preset],
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

        configuration.button_save_config.click(
            save_configuration,
            inputs=[dummy_db_false, configuration.config_file_name] + settings_list,
            outputs=[configuration.config_file_name],
            show_progress=False,
        )

        #config.button_save_as_config.click(
        #    save_configuration,
        #    inputs=[dummy_db_true, config.config_file_name] + settings_list,
        #    outputs=[config.config_file_name],
        #    show_progress=False,
        #)

    with gr.Tab("Guides"):
        gr.Markdown("This section provide Various Finetuning guides and information...")
        top_level_path = fr"{scriptdir}/docs/Finetuning/top_level.md"
        if os.path.exists(top_level_path):
            with open(os.path.join(top_level_path), "r", encoding="utf8") as file:
                guides_top_level = file.read() + "\n"
            gr.Markdown(guides_top_level)
