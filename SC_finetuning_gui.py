import gradio as gr
import json
import math
import os
import subprocess
import pathlib
import argparse
from datetime import datetime
from library.common_gui import (
    get_folder_path,
    get_file_path,
    get_any_file_path,
    get_saveasfile_path,
    save_inference_file,
    run_cmd_advanced_training,
    color_aug_changed,
    update_my_data,
    check_if_model_exist,
    SaveConfigFile,
    save_to_file,
)
from library.class_configuration_file import ConfigurationFile
from library.class_source_model import SourceModel
from library.class_basic_training import BasicTraining
from library.class_advanced_training import AdvancedTraining
from library.class_command_executor import CommandExecutor
from library.tensorboard_gui import (
    gradio_tensorboard,
    start_tensorboard,
    stop_tensorboard,
)
from library.utilities import utilities_tab
from library.class_sample_images import SampleImages, run_cmd_sample

from library.custom_logging import setup_logging
from library.localization_ext import add_javascript

# Set up logging
log = setup_logging()

# Setup command executor
executor = CommandExecutor()

# from easygui import msgbox

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
document_symbol = "\U0001F4C4"  # 📄

PYTHON = "python3" if os.name == "posix" else "./venv/Scripts/python.exe"


def save_configuration(
    save_as,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    effnet_checkpoint_path,
    previewer_checkpoint_path,
    dataset_config_path,
    sample_prompts_path,
    output_dir,
    logging_dir,
    flip_aug,
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
    max_data_loader_n_workers,
    full_fp16,
    color_aug,
    model_list,
    cache_latents,
    cache_latents_to_disk,
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
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
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
    effnet_checkpoint_path,
    previewer_checkpoint_path,
    dataset_config_path,
    sample_prompts_path,
    output_dir,
    logging_dir,
    flip_aug,
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
    max_data_loader_n_workers,
    full_fp16,
    color_aug,
    model_list,
    cache_latents,
    cache_latents_to_disk,
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
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
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
        file_path = f"./presets/finetune/{training_preset}.json"
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
    effnet_checkpoint_path,
    previewer_checkpoint_path,
    dataset_config_path,
    sample_prompts_path,
    output_dir,
    logging_dir,
    flip_aug,
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
    max_data_loader_n_workers,
    full_fp16,
    color_aug,
    model_list,  # Keep this. Yes, it is unused here but required given the common list used
    cache_latents,
    cache_latents_to_disk,
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
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    min_timestep,
    max_timestep,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    print_only_bool = True if print_only.get("label") == "True" else False
    log.info(f"Start Finetuning...")

    headless_bool = True if headless.get("label") == "True" else False

    if check_if_model_exist(output_name, output_dir, save_model_as, headless_bool):
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

    # if optimizer == 'Adafactor' and lr_warmup != '0':
    #     output_message(
    #         msg="Warning: lr_scheduler is set to 'Adafactor', so 'LR warmup (% of steps)' will be considered 0.",
    #         title='Warning',
    #         headless=headless_bool,
    #     )
    #     lr_warmup = '0'

    # image_num = len(
    #     [
    #         f
    #         for f, lower_f in (
    #             (file, file.lower()) for file in os.listdir(image_folder)
    #         )
    #         if lower_f.endswith((".jpg", ".jpeg", ".png", ".webp"))
    #     ]
    # )
    # log.info(f"image_num = {image_num}")

    # repeats = int(image_num) * int(dataset_repeats)
    # log.info(f"repeats = {str(repeats)}")

    # # calculate max_train_steps
    # max_train_steps = int(
    #     math.ceil(
    #         float(repeats)
    #         / int(train_batch_size)
    #         / int(gradient_accumulation_steps)
    #         * int(epoch)
    #     )
    # )

    # # Divide by two because flip augmentation create two copied of the source images
    # if flip_aug:
    #     max_train_steps = int(math.ceil(float(max_train_steps) / 2))

    # log.info(f"max_train_steps = {max_train_steps}")

    # lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    # log.info(f"lr_warmup_steps = {lr_warmup_steps}")

    run_cmd = "accelerate launch"

    run_cmd += run_cmd_advanced_training(
        num_processes=num_processes,
        num_machines=num_machines,
        multi_gpu=multi_gpu,
        gpu_ids=gpu_ids,
        num_cpu_threads_per_process=num_cpu_threads_per_process,
    )

    run_cmd += f' --mixed_precision bf16  "./stable_cascade_train_stage_c.py"'

    cache_text_encoder_outputs = False
    
    run_cmd += f' --stage_c_checkpoint_path "{pretrained_model_name_or_path}"'
    run_cmd += f' --effnet_checkpoint_path "{effnet_checkpoint_path}"'
    run_cmd += f' --previewer_checkpoint_path "{previewer_checkpoint_path}"'
    run_cmd += f' --dataset_config "{dataset_config_path}"'
    # run_cmd += f' --sample_prompts "{sample_prompts_path}"'

    run_cmd += run_cmd_advanced_training(
        adaptive_noise_scale=adaptive_noise_scale,
        additional_parameters=additional_parameters,
        block_lr=block_lr,
        bucket_no_upscale=bucket_no_upscale,
        bucket_reso_steps=bucket_reso_steps,
        cache_latents=cache_latents,
        cache_latents_to_disk=cache_latents_to_disk,
        cache_text_encoder_outputs=cache_text_encoder_outputs,
        caption_dropout_every_n_epochs=caption_dropout_every_n_epochs,
        caption_dropout_rate=caption_dropout_rate,
        caption_extension=caption_extension,
        clip_skip=clip_skip,
        color_aug=color_aug,
        dataset_repeats=dataset_repeats,
        enable_bucket=True,
        flip_aug=flip_aug,
        full_bf16=full_bf16,
        full_fp16=full_fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        keep_tokens=keep_tokens,
        learning_rate=learning_rate,
        # learning_rate_te1=learning_rate_te1,
        # learning_rate_te2=learning_rate_te2,
        learning_rate_te=learning_rate_te,
        logging_dir=logging_dir,
        lr_scheduler=lr_scheduler,
        lr_scheduler_args=lr_scheduler_args,
        # lr_warmup_steps=lr_warmup_steps,
        max_data_loader_n_workers=max_data_loader_n_workers,
        max_timestep=max_timestep,
        max_token_length=max_token_length,
        max_train_epochs=max_train_epochs,
        # max_train_steps=max_train_steps,
        mem_eff_attn=mem_eff_attn,
        min_snr_gamma=min_snr_gamma,
        min_timestep=min_timestep,
        mixed_precision=mixed_precision,
        multires_noise_discount=multires_noise_discount,
        multires_noise_iterations=multires_noise_iterations,
        noise_offset=noise_offset,
        noise_offset_type=noise_offset_type,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        output_dir=output_dir,
        output_name=output_name,
        persistent_data_loader_workers=persistent_data_loader_workers,
        # pretrained_model_name_or_path=pretrained_model_name_or_path,
        random_crop=random_crop,
        resume=resume,
        save_every_n_epochs=save_every_n_epochs,
        save_every_n_steps=save_every_n_steps,
        save_last_n_steps=save_last_n_steps,
        save_last_n_steps_state=save_last_n_steps_state,
        # save_model_as=save_model_as,
        save_precision=save_precision,
        save_state=save_state,
        scale_v_pred_loss_like_noise_pred=scale_v_pred_loss_like_noise_pred,
        seed=seed,
        shuffle_caption=shuffle_caption,
        train_batch_size=train_batch_size,
        # train_data_dir=image_folder,
        train_text_encoder=train_text_encoder,
        use_wandb=use_wandb,
        v2=v2,
        v_parameterization=v_parameterization,
        v_pred_like_loss=v_pred_like_loss,
        vae_batch_size=vae_batch_size,
        wandb_api_key=wandb_api_key,
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

        # check if output_dir/last is a folder... therefore it is a diffuser model
        last_dir = pathlib.Path(f"{output_dir}/{output_name}")

        if not last_dir.is_dir():
            # Copy inference model for v2 if required
            save_inference_file(output_dir, v2, v_parameterization, output_name)


def remove_doublequote(file_path):
    if file_path != None:
        file_path = file_path.replace('"', "")

    return file_path


def sc_finetune_tab(headless=False):
    dummy_db_true = gr.Label(value=True, visible=False)
    dummy_db_false = gr.Label(value=False, visible=False)
    dummy_headless = gr.Label(value=headless, visible=False)
    with gr.Tab("Training"):
        gr.Markdown("Train a custom model using kohya finetune python code...")

        # Setup Configuration Files Gradio
        config = ConfigurationFile(headless)

        source_model = SourceModel(headless=headless, model_type_visibility=False, model_list_value="custom")

        with gr.Tab(label="Folders"):
            with gr.Row():
                effnet_checkpoint_path = gr.Textbox(
                    label='effnet checkpoint path',
                    placeholder='enter the path to the SC effnet checkpoint file',
                )
                effnet_checkpoint_path_file = gr.Button(
                    document_symbol,
                    elem_id='open_folder_small',
                )
                effnet_checkpoint_path_file.click(
                    get_file_path,
                    inputs=effnet_checkpoint_path,
                    outputs=effnet_checkpoint_path,
                    show_progress=False,
                )
                previewer_checkpoint_path = gr.Textbox(
                    label='SC Previewer checkpoint path',
                    placeholder='enter the path to the SC Previewer checkpoint file',
                )
                previewer_checkpoint_path_file = gr.Button(
                    document_symbol,
                    elem_id='open_folder_small',
                )
                previewer_checkpoint_path_file.click(
                    get_file_path,
                    inputs=previewer_checkpoint_path,
                    outputs=previewer_checkpoint_path,
                    show_progress=False,
                )
            with gr.Row():
                dataset_config_path = gr.Textbox(
                    label='effnet checkpoint path',
                    placeholder='enter the path to the SC effnet checkpoint file',
                )
                dataset_config_path_file = gr.Button(
                    document_symbol,
                    elem_id='open_folder_small',
                )
                dataset_config_path_file.click(
                    get_file_path,
                    inputs=dataset_config_path,
                    outputs=dataset_config_path,
                    show_progress=False,
                )
                sample_prompts_path = gr.Textbox(
                    label='Sample image prompt path',
                    placeholder='enter the path to the sample image prompt file',
                )
                sample_prompts_path_file = gr.Button(
                    document_symbol,
                    elem_id='open_folder_small',
                )
                sample_prompts_path_file.click(
                    get_file_path,
                    inputs=sample_prompts_path,
                    outputs=sample_prompts_path,
                    show_progress=False,
                )
            with gr.Row():
                logging_dir = gr.Textbox(
                    label="Logging folder",
                    placeholder="Optional: enable logging and output TensorBoard log to this folder",
                )
                logging_dir_input_folder = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    visible=(not headless),
                )
                logging_dir_input_folder.click(
                    get_folder_path,
                    outputs=logging_dir,
                    show_progress=False,
                )
            with gr.Row():
                output_dir = gr.Textbox(
                    label="Model output folder",
                    placeholder="folder where the model will be saved",
                )
                output_dir_input_folder = gr.Button(
                    folder_symbol,
                    elem_id="open_folder_small",
                    visible=(not headless),
                )
                output_dir_input_folder.click(
                    get_folder_path,
                    outputs=output_dir,
                    show_progress=False,
                )
                output_name = gr.Textbox(
                    label="Model output name",
                    placeholder="Name of the model to output",
                    value="last",
                    interactive=True,
                )
            # effnet_checkpoint_path.change(
            #     remove_doublequote,
            #     inputs=[effnet_checkpoint_path],
            #     outputs=[effnet_checkpoint_path],
            # )
            # previewer_checkpoint_path.change(
            #     remove_doublequote,
            #     inputs=[previewer_checkpoint_path],
            #     outputs=[previewer_checkpoint_path],
            # )
            # dataset_config_path.change(
            #     remove_doublequote,
            #     inputs=[dataset_config_path],
            #     outputs=[dataset_config_path],
            # )
            # sample_prompts_path.change(
            #     remove_doublequote,
            #     inputs=[sample_prompts_path],
            #     outputs=[sample_prompts_path],
            # )
        
        with gr.Tab("Parameters"):

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
                choices=list_presets("./presets/finetune"),
                elem_id="myDropdown",
            )

            with gr.Tab("Basic", elem_id="basic_tab"):
                basic_training = BasicTraining(
                    learning_rate_value="1e-5",
                    finetuning=True,
                )

                with gr.Row():
                    dataset_repeats = gr.Textbox(label="Dataset repeats", value=40)
                    train_text_encoder = gr.Checkbox(
                        label="Train text encoder", value=True
                    )

            with gr.Tab("Advanced", elem_id="advanced_tab"):
                with gr.Row():
                    gradient_accumulation_steps = gr.Number(
                        label="Gradient accumulate steps", value="1"
                    )
                    block_lr = gr.Textbox(
                        label="Block LR",
                        placeholder="(Optional)",
                        info="Specify the different learning rates for each U-Net block. Specify 23 values separated by commas like 1e-3,1e-3 ... 1e-3",
                    )
                advanced_training = AdvancedTraining(headless=headless, finetuning=True)
                advanced_training.color_aug.change(
                    color_aug_changed,
                    inputs=[advanced_training.color_aug],
                    outputs=[
                        basic_training.cache_latents
                    ],  # Not applicable to fine_tune.py
                )

            with gr.Tab("Samples", elem_id="samples_tab"):
                sample = SampleImages()

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
            effnet_checkpoint_path,
            previewer_checkpoint_path,
            dataset_config_path,
            sample_prompts_path,
            output_dir,
            logging_dir,
            advanced_training.flip_aug,
            basic_training.learning_rate,
            basic_training.lr_scheduler,
            basic_training.lr_warmup,
            dataset_repeats,
            basic_training.train_batch_size,
            basic_training.epoch,
            basic_training.save_every_n_epochs,
            basic_training.mixed_precision,
            basic_training.save_precision,
            basic_training.seed,
            basic_training.num_cpu_threads_per_process,
            basic_training.learning_rate_te,
            basic_training.learning_rate_te1,
            basic_training.learning_rate_te2,
            train_text_encoder,
            advanced_training.full_bf16,
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
            advanced_training.max_data_loader_n_workers,
            advanced_training.full_fp16,
            advanced_training.color_aug,
            source_model.model_list,
            basic_training.cache_latents,
            basic_training.cache_latents_to_disk,
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
            advanced_training.save_every_n_steps,
            advanced_training.save_last_n_steps,
            advanced_training.save_last_n_steps_state,
            advanced_training.use_wandb,
            advanced_training.wandb_api_key,
            advanced_training.scale_v_pred_loss_like_noise_pred,
            advanced_training.min_timestep,
            advanced_training.max_timestep,
        ]

        config.button_open_config.click(
            open_configuration,
            inputs=[dummy_db_true, dummy_db_false, config.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[config.config_file_name] + settings_list + [training_preset],
            show_progress=False,
        )

        # config.button_open_config.click(
        #     open_configuration,
        #     inputs=[dummy_db_true, dummy_db_false, config.config_file_name] + settings_list,
        #     outputs=[config.config_file_name] + settings_list,
        #     show_progress=False,
        # )

        config.button_load_config.click(
            open_configuration,
            inputs=[dummy_db_false, dummy_db_false, config.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[config.config_file_name] + settings_list + [training_preset],
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
            inputs=[dummy_db_false, dummy_db_true, config.config_file_name]
            + settings_list
            + [training_preset],
            outputs=[gr.Textbox()] + settings_list + [training_preset],
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

    with gr.Tab("Guides"):
        gr.Markdown("This section provide Various Finetuning guides and information...")
        top_level_path = "./docs/Finetuning/top_level.md"
        if os.path.exists(top_level_path):
            with open(os.path.join(top_level_path), "r", encoding="utf8") as file:
                guides_top_level = file.read() + "\n"
        gr.Markdown(guides_top_level)


def UI(**kwargs):
    add_javascript(kwargs.get("language"))
    css = ""

    headless = kwargs.get("headless", False)
    log.info(f"headless: {headless}")

    if os.path.exists("./style.css"):
        with open(os.path.join("./style.css"), "r", encoding="utf8") as file:
            log.info("Load CSS...")
            css += file.read() + "\n"

    interface = gr.Blocks(css=css, title="Kohya_ss GUI", theme=gr.themes.Default())

    with interface:
        with gr.Tab("SC Finetuning"):
            sc_finetune_tab(headless=headless)
        with gr.Tab("Utilities"):
            utilities_tab(enable_dreambooth_tab=False, headless=headless)

    # Show the interface
    launch_kwargs = {}
    username = kwargs.get("username")
    password = kwargs.get("password")
    server_port = kwargs.get("server_port", 0)
    inbrowser = kwargs.get("inbrowser", False)
    share = kwargs.get("share", False)
    server_name = kwargs.get("listen")

    launch_kwargs["server_name"] = server_name
    if username and password:
        launch_kwargs["auth"] = (username, password)
    if server_port > 0:
        launch_kwargs["server_port"] = server_port
    if inbrowser:
        launch_kwargs["inbrowser"] = inbrowser
    if share:
        launch_kwargs["share"] = share
    interface.launch(**launch_kwargs)


if __name__ == "__main__":
    # torch.cuda.set_per_process_memory_fraction(0.48)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1",
        help="IP to listen on for connections to Gradio",
    )
    parser.add_argument(
        "--username", type=str, default="", help="Username for authentication"
    )
    parser.add_argument(
        "--password", type=str, default="", help="Password for authentication"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=0,
        help="Port to run the server listener on",
    )
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--share", action="store_true", help="Share the gradio UI")
    parser.add_argument(
        "--headless", action="store_true", help="Is the server headless"
    )
    parser.add_argument(
        "--language", type=str, default=None, help="Set custom language"
    )

    args = parser.parse_args()

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
        headless=args.headless,
        language=args.language,
    )
