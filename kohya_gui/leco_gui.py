import gradio as gr
import json
import os
import time
import toml

from datetime import datetime
from .common_gui import (
    get_any_file_path,
    get_executable_path,
    get_file_path,
    get_folder_path,
    get_saveasfile_path,
    check_if_model_exist,
    join_config_path,
    list_files,
    create_refresh_button,
    output_message,
    print_command_and_toml,
    require_writable_directory,
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
    write_toml_config,
)
from .class_source_model import default_models
from .class_accelerate_launch import AccelerateLaunch
from .class_configuration_file import ConfigurationFile
from .class_command_executor import CommandExecutor
from .class_gui_config import KohyaSSGUIConfig
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Setup command executor
executor = None

use_shell = False
train_state_value = time.time()

# Populated by leco_tab() with the (param_name, component) pairs backing
# settings_list, in the same order as train_model's/save_configuration's/
# open_configuration's shared keyword-argument order. Exposed at module level
# so tests can assert it stays in sync without rebuilding the whole GUI.
last_built_field_registry = None

# Populated by leco_tab() with the dict-keyed adapter callables wired to the
# train/save/load buttons (GH #3543 M3). Exposed at module level so tests can
# invoke the real .click()-bound callables directly instead of calling
# train_model/save_configuration/open_configuration themselves, exercising the
# same component-identity lookup the Gradio wiring uses.
last_built_gui_entries = None

document_symbol = "\U0001f4c4"  # 📄
folder_symbol = "\U0001f4c2"  # 📂


def save_configuration(
    save_as_bool,
    file_path,
    # model section
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    output_dir,
    output_name,
    save_model_as,
    save_precision,
    training_comment,
    no_metadata,
    # prompts section
    prompts_file,
    # network section
    network_module,
    network_dim,
    network_alpha,
    network_dropout,
    network_args,
    network_weights,
    dim_from_weights,
    # training section
    learning_rate,
    unet_lr,
    optimizer,
    optimizer_args,
    lr_scheduler,
    lr_scheduler_args,
    lr_warmup_steps,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    max_train_steps,
    max_grad_norm,
    max_denoising_steps,
    leco_denoise_guidance_scale,
    seed,
    gradient_accumulation_steps,
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
    # advanced section
    gradient_checkpointing,
    full_fp16,
    full_bf16,
    xformers,
    mem_eff_attn,
    clip_skip,
    noise_offset,
    zero_terminal_snr,
    min_snr_gamma,
    # save section
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    save_state,
    save_state_on_train_end,
    resume,
    # logging section
    logging_dir,
    log_with,
    log_tracker_name,
    log_tracker_config,
    log_config,
    wandb_api_key,
    wandb_run_name,
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

    log.debug(file_path)

    if file_path == None or file_path == "":
        return original_file_path

    destination_directory = os.path.dirname(file_path)

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
    # model section
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    output_dir,
    output_name,
    save_model_as,
    save_precision,
    training_comment,
    no_metadata,
    # prompts section
    prompts_file,
    # network section
    network_module,
    network_dim,
    network_alpha,
    network_dropout,
    network_args,
    network_weights,
    dim_from_weights,
    # training section
    learning_rate,
    unet_lr,
    optimizer,
    optimizer_args,
    lr_scheduler,
    lr_scheduler_args,
    lr_warmup_steps,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    max_train_steps,
    max_grad_norm,
    max_denoising_steps,
    leco_denoise_guidance_scale,
    seed,
    gradient_accumulation_steps,
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
    # advanced section
    gradient_checkpointing,
    full_fp16,
    full_bf16,
    xformers,
    mem_eff_attn,
    clip_skip,
    noise_offset,
    zero_terminal_snr,
    min_snr_gamma,
    # save section
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    save_state,
    save_state_on_train_end,
    resume,
    # logging section
    logging_dir,
    log_with,
    log_tracker_name,
    log_tracker_config,
    log_config,
    wandb_api_key,
    wandb_run_name,
):
    # Get list of function parameters and their values
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
    v2,
    v_parameterization,
    sdxl,
    output_dir,
    output_name,
    save_model_as,
    save_precision,
    training_comment,
    no_metadata,
    # prompts section
    prompts_file,
    # network section
    network_module,
    network_dim,
    network_alpha,
    network_dropout,
    network_args,
    network_weights,
    dim_from_weights,
    # training section
    learning_rate,
    unet_lr,
    optimizer,
    optimizer_args,
    lr_scheduler,
    lr_scheduler_args,
    lr_warmup_steps,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    max_train_steps,
    max_grad_norm,
    max_denoising_steps,
    leco_denoise_guidance_scale,
    seed,
    gradient_accumulation_steps,
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
    # advanced section
    gradient_checkpointing,
    full_fp16,
    full_bf16,
    xformers,
    mem_eff_attn,
    clip_skip,
    noise_offset,
    zero_terminal_snr,
    min_snr_gamma,
    # save section
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    save_state,
    save_state_on_train_end,
    resume,
    # logging section
    logging_dir,
    log_with,
    log_tracker_name,
    log_tracker_config,
    log_config,
    wandb_api_key,
    wandb_run_name,
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

    log.info("Start training LECO ...")

    log.info("Validating lr scheduler arguments...")
    if not validate_args_setting(lr_scheduler_args):
        return TRAIN_BUTTON_VISIBLE

    log.info("Validating optimizer arguments...")
    if not validate_args_setting(optimizer_args):
        return TRAIN_BUTTON_VISIBLE

    #
    # Validate paths
    #

    if not validate_toml_file(prompts_file):
        return TRAIN_BUTTON_VISIBLE

    if prompts_file == "":
        log.error("Prompts TOML file is required for LECO training")
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

    if not validate_model_path(pretrained_model_name_or_path):
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

    if sdxl:
        run_cmd.append(rf"{scriptdir}/sd-scripts/sdxl_train_leco.py")
    else:
        run_cmd.append(rf"{scriptdir}/sd-scripts/train_leco.py")

    config_toml_data = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "v2": v2 if not sdxl else None,
        "v_parameterization": v_parameterization if not sdxl else None,
        "output_dir": output_dir,
        "output_name": output_name,
        "save_model_as": save_model_as,
        "save_precision": save_precision,
        "training_comment": training_comment,
        "no_metadata": no_metadata,
        "prompts_file": prompts_file,
        "network_module": network_module,
        "network_dim": int(network_dim),
        "network_alpha": float(network_alpha),
        "network_dropout": network_dropout if network_dropout != 0 else None,
        "network_args": str(network_args).replace('"', "").split(),
        "network_weights": network_weights,
        "dim_from_weights": dim_from_weights,
        "learning_rate": learning_rate,
        "unet_lr": unet_lr if unet_lr != 0 else None,
        "optimizer_type": optimizer,
        "optimizer_args": (
            str(optimizer_args).replace('"', "").split()
            if optimizer_args != []
            else None
        ),
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_args": str(lr_scheduler_args).replace('"', "").split(),
        "lr_warmup_steps": lr_warmup_steps,
        "lr_scheduler_num_cycles": (
            int(lr_scheduler_num_cycles) if lr_scheduler_num_cycles != "" else None
        ),
        "lr_scheduler_power": lr_scheduler_power,
        "max_train_steps": int(max_train_steps) if int(max_train_steps) != 0 else None,
        "max_grad_norm": max_grad_norm,
        "max_denoising_steps": int(max_denoising_steps),
        "leco_denoise_guidance_scale": float(leco_denoise_guidance_scale),
        "seed": int(seed) if int(seed) != 0 else None,
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": gradient_checkpointing,
        "full_fp16": full_fp16,
        "full_bf16": full_bf16,
        "sdpa": True if xformers == "sdpa" else None,
        "xformers": True if xformers == "xformers" else None,
        "mem_eff_attn": mem_eff_attn,
        "clip_skip": clip_skip if clip_skip != 0 and not sdxl else None,
        "noise_offset": noise_offset if noise_offset != 0 else None,
        "zero_terminal_snr": zero_terminal_snr,
        "min_snr_gamma": min_snr_gamma if min_snr_gamma != 0 else None,
        "save_every_n_steps": save_every_n_steps if save_every_n_steps != 0 else None,
        "save_last_n_steps": save_last_n_steps if save_last_n_steps != 0 else None,
        "save_last_n_steps_state": (
            save_last_n_steps_state if save_last_n_steps_state != 0 else None
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
    }

    # Remove all values that are empty, False or None
    config_toml_data = {
        key: value
        for key, value in config_toml_data.items()
        if value not in ["", False, None]
    }

    # Sort the dictionary by keys
    config_toml_data = dict(sorted(config_toml_data.items()))

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
    tmpfilename = join_config_path(output_dir, f"config_leco-{formatted_datetime}.toml")

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


def leco_tab(
    headless=False,
    config: KohyaSSGUIConfig = {},
    use_shell_flag: bool = False,
):
    global use_shell, executor
    use_shell = use_shell_flag

    dummy_db_true = gr.Checkbox(value=True, visible=False)
    dummy_db_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    current_models_dir = config.get(
        "model.models_dir", os.path.join(scriptdir, "models")
    )

    def list_models(path):
        nonlocal current_models_dir
        current_models_dir = path if os.path.isdir(path) else os.path.dirname(path)
        return default_models + list(
            list_files(path, exts=[".ckpt", ".safetensors"], all=True)
        )

    model_checkpoints = list(
        list_files(current_models_dir, exts=[".ckpt", ".safetensors"], all=True)
    )

    with gr.Tab("Training"), gr.Column(variant="compact"):
        gr.Markdown(
            "Train a LECO (Low-rank adaptation for Erasing COncepts) LoRA using only text"
            " prompts — no image dataset is required. See sd-scripts/docs/train_leco.md"
            " for details."
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
                        label="Pretrained model name or path",
                        choices=default_models + model_checkpoints,
                        value=config.get(
                            "model.pretrained_model_name_or_path",
                            "runwayml/stable-diffusion-v1-5",
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
                    sdxl = gr.Checkbox(
                        label="SDXL",
                        value=config.get("model.sdxl", False),
                        info="Enable to train against sdxl_train_leco.py instead of train_leco.py",
                        min_width=60,
                    )
                    v2 = gr.Checkbox(
                        label="v2",
                        value=config.get("model.v2", False),
                        min_width=60,
                        interactive=True,
                    )
                    v_parameterization = gr.Checkbox(
                        label="v_parameterization",
                        value=config.get("model.v_parameterization", False),
                        min_width=130,
                        interactive=True,
                    )

                    def toggle_v_family(sdxl_value):
                        if sdxl_value:
                            return (
                                gr.Checkbox(interactive=False, value=False),
                                gr.Checkbox(interactive=False, value=False),
                            )
                        return (
                            gr.Checkbox(interactive=True),
                            gr.Checkbox(interactive=True),
                        )

                    sdxl.change(
                        fn=toggle_v_family,
                        inputs=sdxl,
                        outputs=[v2, v_parameterization],
                        show_progress=False,
                    )

                with gr.Row():
                    output_dir = gr.Textbox(
                        label="Output directory",
                        placeholder="Directory to output the trained model and logs",
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
                        label="Trained Model output name",
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
                        value=config.get("model.save_precision", "fp16"),
                    )
                    no_metadata = gr.Checkbox(
                        label="No metadata",
                        value=config.get("model.no_metadata", False),
                        info="Do not save metadata in output model",
                    )

        with gr.Accordion("LECO prompts", open=True), gr.Group():
            gr.Markdown(
                "LECO does not use an image dataset. Training prompts (target /"
                " positive / negative / neutral) are defined entirely in this TOML file."
                " See sd-scripts/docs/train_leco.md for the file format."
            )
            with gr.Row():
                prompts_file = gr.Textbox(
                    label="Prompts TOML file",
                    placeholder="Path to the LECO prompts TOML file (required)",
                    value=config.get("leco.prompts_file", ""),
                    interactive=True,
                )
                prompts_file_button = gr.Button(
                    document_symbol,
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not headless),
                )
                prompts_file_button.click(
                    get_file_path,
                    inputs=[
                        prompts_file,
                        gr.Textbox(value="*.toml", visible=False),
                        gr.Textbox(value="LECO prompts TOML", visible=False),
                    ],
                    outputs=prompts_file,
                    show_progress=False,
                )

        with gr.Accordion("LoRA network parameters", open=True), gr.Group():
            with gr.Row():
                network_module = gr.Textbox(
                    label="Network module",
                    value=config.get("network.network_module", "networks.lora"),
                    interactive=True,
                )
                network_dim = gr.Slider(
                    minimum=1,
                    maximum=512,
                    label="Network Rank (Dimension)",
                    value=config.get("network.network_dim", 8),
                    step=1,
                    interactive=True,
                )
                network_alpha = gr.Slider(
                    minimum=0.00001,
                    maximum=1024,
                    label="Network Alpha",
                    value=config.get("network.network_alpha", 4),
                    step=0.00001,
                    interactive=True,
                )
            with gr.Row():
                network_dropout = gr.Slider(
                    label="Network Dropout",
                    value=config.get("network.network_dropout", 0),
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    info="Percentage of neurons that dropout randomly when training",
                    interactive=True,
                )
                network_args = gr.Textbox(
                    label="Network args",
                    placeholder="(Optional) eg: conv_dim=4 conv_alpha=4",
                    value=config.get("network.network_args", ""),
                    interactive=True,
                )
            with gr.Row():
                network_weights = gr.Textbox(
                    label="Network weights",
                    placeholder="(Optional) Path to existing LoRA network weights to resume from",
                    value=config.get("network.network_weights", ""),
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
                dim_from_weights = gr.Checkbox(
                    label="DIM from weights",
                    value=config.get("network.dim_from_weights", False),
                    info="Automatically determine the dim(rank) from the weight file.",
                )

        with gr.Accordion("Training parameters", open=True), gr.Group():
            with gr.Row():
                learning_rate = gr.Number(
                    label="Learning rate",
                    value=config.get("training.learning_rate", 0.0001),
                    minimum=0,
                    maximum=1,
                    info="Learning rate used by the optimizer, and fallback for unet_lr if unset",
                )
                unet_lr = gr.Number(
                    label="U-Net learning rate",
                    value=config.get("training.unet_lr", 0.0001),
                    minimum=0,
                    maximum=1,
                    info="(Optional) Overrides Learning rate for the U-Net LoRA",
                )
                max_train_steps = gr.Number(
                    label="Max train steps",
                    value=config.get("training.max_train_steps", 500),
                    precision=0,
                    info="LECO is step-based only; epochs are not supported",
                )
            with gr.Row():
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
                seed = gr.Number(
                    label="Seed",
                    value=config.get("training.seed", 0),
                    precision=0,
                    info="0 lets sd-scripts pick a random seed",
                )
                gradient_accumulation_steps = gr.Slider(
                    label="Gradient accumulate steps",
                    value=config.get("training.gradient_accumulation_steps", 1),
                    minimum=1,
                    maximum=120,
                    step=1,
                )

        with gr.Accordion("LECO parameters", open=True), gr.Group():
            gr.Markdown(
                "Parameters unique to LECO training (not present in standard LoRA"
                " training scripts)."
            )
            with gr.Row():
                max_denoising_steps = gr.Slider(
                    label="Max denoising steps",
                    value=config.get("leco.max_denoising_steps", 40),
                    minimum=1,
                    maximum=200,
                    step=1,
                    info="Number of partial denoising steps per training iteration",
                )
                leco_denoise_guidance_scale = gr.Number(
                    label="LECO denoise guidance scale",
                    value=config.get("leco.leco_denoise_guidance_scale", 3.0),
                    info="Guidance scale for the partial denoising pass",
                )

        with gr.Accordion("Advanced", open=False), gr.Group():
            with gr.Row():
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
                mem_eff_attn = gr.Checkbox(
                    label="Memory efficient attention",
                    value=config.get("advanced.mem_eff_attn", False),
                )
                xformers = gr.Dropdown(
                    label="CrossAttention",
                    choices=["none", "sdpa", "xformers"],
                    value=config.get("advanced.xformers", "sdpa"),
                )
            with gr.Row():
                clip_skip = gr.Slider(
                    label="Clip skip",
                    value=config.get("advanced.clip_skip", 1),
                    minimum=1,
                    maximum=12,
                    step=1,
                    info="SD 1.x/2.x only",
                )
                noise_offset = gr.Slider(
                    label="Noise offset",
                    value=config.get("advanced.noise_offset", 0),
                    minimum=0,
                    maximum=1,
                    step=0.01,
                )
                zero_terminal_snr = gr.Checkbox(
                    label="Zero terminal SNR",
                    value=config.get("advanced.zero_terminal_snr", False),
                )
                min_snr_gamma = gr.Slider(
                    label="Min SNR gamma",
                    value=config.get("advanced.min_snr_gamma", 0),
                    minimum=0,
                    maximum=20,
                    step=1,
                )

        with gr.Accordion("Save, resume and logging", open=False), gr.Group():
            with gr.Row():
                save_every_n_steps = gr.Number(
                    label="Save every N steps",
                    value=config.get("save.save_every_n_steps", 100),
                    precision=0,
                    info="If unset, only the final model is saved",
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
            ("v2", v2),
            ("v_parameterization", v_parameterization),
            ("sdxl", sdxl),
            ("output_dir", output_dir),
            ("output_name", output_name),
            ("save_model_as", save_model_as),
            ("save_precision", save_precision),
            ("training_comment", training_comment),
            ("no_metadata", no_metadata),
            ("prompts_file", prompts_file),
            ("network_module", network_module),
            ("network_dim", network_dim),
            ("network_alpha", network_alpha),
            ("network_dropout", network_dropout),
            ("network_args", network_args),
            ("network_weights", network_weights),
            ("dim_from_weights", dim_from_weights),
            ("learning_rate", learning_rate),
            ("unet_lr", unet_lr),
            ("optimizer", optimizer),
            ("optimizer_args", optimizer_args),
            ("lr_scheduler", lr_scheduler),
            ("lr_scheduler_args", lr_scheduler_args),
            ("lr_warmup_steps", lr_warmup_steps),
            ("lr_scheduler_num_cycles", lr_scheduler_num_cycles),
            ("lr_scheduler_power", lr_scheduler_power),
            ("max_train_steps", max_train_steps),
            ("max_grad_norm", max_grad_norm),
            ("max_denoising_steps", max_denoising_steps),
            ("leco_denoise_guidance_scale", leco_denoise_guidance_scale),
            ("seed", seed),
            ("gradient_accumulation_steps", gradient_accumulation_steps),
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
            ("gradient_checkpointing", gradient_checkpointing),
            ("full_fp16", full_fp16),
            ("full_bf16", full_bf16),
            ("xformers", xformers),
            ("mem_eff_attn", mem_eff_attn),
            ("clip_skip", clip_skip),
            ("noise_offset", noise_offset),
            ("zero_terminal_snr", zero_terminal_snr),
            ("min_snr_gamma", min_snr_gamma),
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
        ]
        settings_list = [comp for _, comp in FIELD_REGISTRY]

        global last_built_field_registry
        last_built_field_registry = FIELD_REGISTRY

        # GH #3543 M3: adapters at the Gradio boundary look up each argument by
        # component identity (via FIELD_REGISTRY) rather than by position, so a
        # field added out of order can no longer silently shift every
        # subsequent value into the wrong parameter. train_model's/
        # save_configuration's/open_configuration's own signatures and bodies
        # are untouched; only the .click() wiring below changes.
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
                # Missing config file returns None — emit no-op updates so every
                # wired output is accounted for (partial {} is not guaranteed).
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

        # Open asks for a file (True); Load re-reads the path already in the
        # config_file_name field (False). Pre-adapter wiring incorrectly used
        # False for both, so the Open button never opened a file dialog —
        # align with lora/dreambooth/finetune/TI while converting to adapters.
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
