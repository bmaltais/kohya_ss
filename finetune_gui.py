import gradio as gr
import json
import math
import os
import subprocess
import pathlib
import shutil
from glob import glob
from os.path import join


def save_variables(
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_model,
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
    convert_to_safetensors,
    convert_to_ckpt,
    cache_latent,
    caption_extention,
    use_safetensors,
    enable_bucket
):
    # Return the values of the variables as a dictionary
    variables = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "v2": v2,
        "v_model": v_model,
        "logging_dir": logging_dir,
        "train_data_dir": train_data_dir,
        "reg_data_dir": reg_data_dir,
        "output_dir": output_dir,
        "max_resolution": max_resolution,
        "learning_rate": learning_rate,
        "lr_scheduler": lr_scheduler,
        "lr_warmup": lr_warmup,
        "train_batch_size": train_batch_size,
        "epoch": epoch,
        "save_every_n_epochs": save_every_n_epochs,
        "mixed_precision": mixed_precision,
        "save_precision": save_precision,
        "seed": seed,
        "num_cpu_threads_per_process": num_cpu_threads_per_process,
        "convert_to_safetensors": convert_to_safetensors,
        "convert_to_ckpt": convert_to_ckpt,
        "cache_latent": cache_latent,
        "caption_extention": caption_extention,
        "use_safetensors": use_safetensors,
        "enable_bucket": enable_bucket
    }

    # Save the data to the selected file
    with open(file_path, "w") as file:
        json.dump(variables, file)


def load_variables(file_path):
    # load variables from JSON file
    with open(file_path, "r") as f:
        my_data = json.load(f)

    # Return the values of the variables as a dictionary
    return (
        my_data.get("pretrained_model_name_or_path", None),
        my_data.get("v2", None),
        my_data.get("v_model", None),
        my_data.get("logging_dir", None),
        my_data.get("train_data_dir", None),
        my_data.get("reg_data_dir", None),
        my_data.get("output_dir", None),
        my_data.get("max_resolution", None),
        my_data.get("learning_rate", None),
        my_data.get("lr_scheduler", None),
        my_data.get("lr_warmup", None),
        my_data.get("train_batch_size", None),
        my_data.get("epoch", None),
        my_data.get("save_every_n_epochs", None),
        my_data.get("mixed_precision", None),
        my_data.get("save_precision", None),
        my_data.get("seed", None),
        my_data.get("num_cpu_threads_per_process", None),
        my_data.get("convert_to_safetensors", None),
        my_data.get("convert_to_ckpt", None),
        my_data.get("cache_latent", None),
        my_data.get("caption_extention", None),
        my_data.get("use_safetensors", None),
        my_data.get("enable_bucket", None),
    )


def train_model(
    pretrained_model_name_or_path,
    v2,
    v_model,
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
    convert_to_safetensors,
    convert_to_ckpt,
    cache_latent,
    caption_extention,
    use_safetensors,
    enable_bucket
):
    def save_inference_file(output_dir, v2, v_model):
        # Copy inference model for v2 if required
        if v2 and v_model:
            print(f"Saving v2-inference-v.yaml as {output_dir}/last.yaml")
            shutil.copy(
                f"./v2_inference/v2-inference-v.yaml",
                f"{output_dir}/last.yaml",
            )
        elif v2:
            print(f"Saving v2-inference.yaml as {output_dir}/last.yaml")
            shutil.copy(
                f"./v2_inference/v2-inference.yaml",
                f"{output_dir}/last.yaml",
            )

    # Get a list of all subfolders in train_data_dir
    subfolders = [f for f in os.listdir(train_data_dir) if os.path.isdir(
        os.path.join(train_data_dir, f))]

    total_steps = 0

    # Loop through each subfolder and extract the number of repeats
    for folder in subfolders:
        # Extract the number of repeats from the folder name
        repeats = int(folder.split("_")[0])

        # Count the number of images in the folder
        num_images = len([f for f in os.listdir(os.path.join(train_data_dir, folder)) if f.endswith(
            ".jpg") or f.endswith(".jpeg") or f.endswith(".png") or f.endswith(".webp")])

        # Calculate the total number of steps for this folder
        steps = repeats * num_images
        total_steps += steps

        # Print the result
        print(f"Folder {folder}: {steps} steps")

    # Print the result
    # print(f"{total_steps} total steps")

    # calculate max_train_steps
    max_train_steps = int(
        math.ceil(float(total_steps) / int(train_batch_size) * int(epoch))
    )
    print(f"max_train_steps = {max_train_steps}")

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    print(f"lr_warmup_steps = {lr_warmup_steps}")

    run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process} "train_db_fixed.py"'
    if v2:
        run_cmd += " --v2"
    if v_model:
        run_cmd += " --v_parameterization"
    if cache_latent:
        run_cmd += " --cache_latents"
    if use_safetensors:
        run_cmd += " --use_safetensors"
    if enable_bucket:
        run_cmd += " --enable_bucket"
    run_cmd += f" --pretrained_model_name_or_path={pretrained_model_name_or_path}"
    run_cmd += f" --train_data_dir={train_data_dir}"
    run_cmd += f" --reg_data_dir={reg_data_dir}"
    run_cmd += f" --resolution={max_resolution}"
    run_cmd += f" --output_dir={output_dir}"
    run_cmd += f" --train_batch_size={train_batch_size}"
    run_cmd += f" --learning_rate={learning_rate}"
    run_cmd += f" --lr_scheduler={lr_scheduler}"
    run_cmd += f" --lr_warmup_steps={lr_warmup_steps}"
    run_cmd += f" --max_train_steps={max_train_steps}"
    run_cmd += f" --use_8bit_adam"
    run_cmd += f" --xformers"
    run_cmd += f" --mixed_precision={mixed_precision}"
    run_cmd += f" --save_every_n_epochs={save_every_n_epochs}"
    run_cmd += f" --seed={seed}"
    run_cmd += f" --save_precision={save_precision}"
    run_cmd += f" --logging_dir={logging_dir}"
    run_cmd += f" --caption_extention={caption_extention}"

    print(run_cmd)
    # Run the command
    subprocess.run(run_cmd)

    # check if output_dir/last is a directory... therefore it is a diffuser model
    last_dir = pathlib.Path(f"{output_dir}/last")
    print(last_dir)
    if last_dir.is_dir():
        if convert_to_ckpt:
            print(f"Converting diffuser model {last_dir} to {last_dir}.ckpt")
            os.system(
                f"python ./tools/convert_diffusers20_original_sd.py {last_dir} {last_dir}.ckpt --{save_precision}"
            )

            save_inference_file(output_dir, v2, v_model)

        if convert_to_safetensors:
            print(
                f"Converting diffuser model {last_dir} to {last_dir}.safetensors")
            os.system(
                f"python ./tools/convert_diffusers20_original_sd.py {last_dir} {last_dir}.safetensors --{save_precision}"
            )

            save_inference_file(output_dir, v2, v_model)
    else:
        # Copy inference model for v2 if required
        save_inference_file(output_dir, v2, v_model)

    # Return the values of the variables as a dictionary
    # return


def set_pretrained_model_name_or_path_input(value, v2, v_model):
    # define a list of substrings to search for
    substrings_v2 = ["stabilityai/stable-diffusion-2-1-base", "stabilityai/stable-diffusion-2-base"]

    # check if $v2 and $v_model are empty and if $pretrained_model_name_or_path contains any of the substrings in the v2 list
    if str(value) in substrings_v2:
        print("SD v2 model detected. Setting --v2 parameter")
        v2 = True
        v_model = False

        return value, v2, v_model

    # define a list of substrings to search for v-objective
    substrings_v_model = ["stabilityai/stable-diffusion-2-1", "stabilityai/stable-diffusion-2"]

    # check if $v2 and $v_model are empty and if $pretrained_model_name_or_path contains any of the substrings in the v_model list
    if str(value) in substrings_v_model:
        print("SD v2 v_model detected. Setting --v2 parameter and --v_parameterization")
        v2 = True
        v_model = True

        return value, v2, v_model

    # define a list of substrings to v1.x
    substrings_v1_model = ["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5"]

    if str(value) in substrings_v1_model:
        v2 = False
        v_model = False

        return value, v2, v_model

    if value == "custom":
        value = ""
        v2 = False
        v_model = False

        return value, v2, v_model

interface = gr.Blocks()

with interface:
    gr.Markdown("Enter kohya finetuner parameter using this interface.")
    with gr.Accordion("Configuration File Load/Save", open=False):
        with gr.Row():
            config_file_name = gr.Textbox(
                label="Config file name")
            b1 = gr.Button("Load config")
            b2 = gr.Button("Save config")
    with gr.Tab("Source model"):
        # Define the input elements
        with gr.Row():
            pretrained_model_name_or_path_input = gr.Textbox(
                label="Pretrained model name or path",
                placeholder="enter the path to custom model or name of pretrained model",
            )
            model_list = gr.Dropdown(
                label="(Optional) Model Quick Pick",
                choices=[
                    "custom",
                    "stabilityai/stable-diffusion-2-1-base",
                    "stabilityai/stable-diffusion-2-base",
                    "stabilityai/stable-diffusion-2-1",
                    "stabilityai/stable-diffusion-2",
                    "runwayml/stable-diffusion-v1-5",
                    "CompVis/stable-diffusion-v1-4"
                ],
            )
        with gr.Row():
            v2_input = gr.Checkbox(label="v2", value=True)
            v_model_input = gr.Checkbox(label="v_model", value=False)
        model_list.change(
            set_pretrained_model_name_or_path_input,
            inputs=[model_list, v2_input, v_model_input],
            outputs=[pretrained_model_name_or_path_input,
                     v2_input, v_model_input],
        )
    with gr.Tab("Directories"):
        with gr.Row():
            train_data_dir_input = gr.Textbox(
                label="Image folder", placeholder="directory where the training folders containing the images are located"
            )
            reg_data_dir_input = gr.Textbox(
                label="Regularisation folder", placeholder="directory where where the regularization folders containing the images are located"
            )

        with gr.Row():
            output_dir_input = gr.Textbox(
                label="Output directory",
                placeholder="directory to output trained model",
            )
            logging_dir_input = gr.Textbox(
                label="Logging directory", placeholder="Optional: enable logging and output TensorBoard log to this directory"
            )
    with gr.Tab("Training parameters"):
        with gr.Row():
            learning_rate_input = gr.Textbox(
                label="Learning rate", value=1e-6)
            lr_scheduler_input = gr.Dropdown(
                label="LR Scheduler",
                choices=[
                    "constant",
                    "constant_with_warmup",
                    "cosine",
                    "cosine_with_restarts",
                    "linear",
                    "polynomial",
                ],
                value="constant",
            )
            lr_warmup_input = gr.Textbox(label="LR warmup", value=0)
        with gr.Row():
            train_batch_size_input = gr.Textbox(
                label="Train batch size", value=1
            )
            epoch_input = gr.Textbox(label="Epoch", value=1)
            save_every_n_epochs_input = gr.Textbox(
                label="Save every N epochs", value=1
            )
        with gr.Row():
            mixed_precision_input = gr.Dropdown(
                label="Mixed precision",
                choices=[
                    "no",
                    "fp16",
                    "bf16",
                ],
                value="fp16",
            )
            save_precision_input = gr.Dropdown(
                label="Save precision",
                choices=[
                    "float",
                    "fp16",
                    "bf16",
                ],
                value="fp16",
            )
            num_cpu_threads_per_process_input = gr.Textbox(
                label="Number of CPU threads per process", value=4
            )
        with gr.Row():
            seed_input = gr.Textbox(label="Seed", value=1234)
            max_resolution_input = gr.Textbox(
                label="Max resolution", value="512,512"
            )
            caption_extention_input = gr.Textbox(
                label="Caption Extension", placeholder="(Optional) Extension for caption files. default: .caption")

        with gr.Row():
            use_safetensors_input = gr.Checkbox(
                label="Use safetensor when saving checkpoint", value=False
            )
            enable_bucket_input = gr.Checkbox(
                label="Enable buckets", value=False
            )
            cache_latent_input = gr.Checkbox(
                label="Cache latent", value=True
            )

    with gr.Tab("Model conversion"):
        convert_to_safetensors_input = gr.Checkbox(
            label="Convert to SafeTensors", value=False
        )
        convert_to_ckpt_input = gr.Checkbox(
            label="Convert to CKPT", value=False
        )

    b3 = gr.Button("Run")

    b1.click(
        load_variables,
        inputs=[config_file_name],
        outputs=[
            pretrained_model_name_or_path_input,
            v2_input,
            v_model_input,
            logging_dir_input,
            train_data_dir_input,
            reg_data_dir_input,
            output_dir_input,
            max_resolution_input,
            learning_rate_input,
            lr_scheduler_input,
            lr_warmup_input,
            train_batch_size_input,
            epoch_input,
            save_every_n_epochs_input,
            mixed_precision_input,
            save_precision_input,
            seed_input,
            num_cpu_threads_per_process_input,
            convert_to_safetensors_input,
            convert_to_ckpt_input,
            cache_latent_input,
            caption_extention_input,
            use_safetensors_input,
            enable_bucket_input
        ]
    )

    b2.click(
        save_variables,
        inputs=[
            config_file_name,
            pretrained_model_name_or_path_input,
            v2_input,
            v_model_input,
            logging_dir_input,
            train_data_dir_input,
            reg_data_dir_input,
            output_dir_input,
            max_resolution_input,
            learning_rate_input,
            lr_scheduler_input,
            lr_warmup_input,
            train_batch_size_input,
            epoch_input,
            save_every_n_epochs_input,
            mixed_precision_input,
            save_precision_input,
            seed_input,
            num_cpu_threads_per_process_input,
            convert_to_safetensors_input,
            convert_to_ckpt_input,
            cache_latent_input,
            caption_extention_input,
            use_safetensors_input,
            enable_bucket_input
        ]
    )
    b3.click(
        train_model,
        inputs=[
            pretrained_model_name_or_path_input,
            v2_input,
            v_model_input,
            logging_dir_input,
            train_data_dir_input,
            reg_data_dir_input,
            output_dir_input,
            max_resolution_input,
            learning_rate_input,
            lr_scheduler_input,
            lr_warmup_input,
            train_batch_size_input,
            epoch_input,
            save_every_n_epochs_input,
            mixed_precision_input,
            save_precision_input,
            seed_input,
            num_cpu_threads_per_process_input,
            convert_to_safetensors_input,
            convert_to_ckpt_input,
            cache_latent_input,
            caption_extention_input,
            use_safetensors_input,
            enable_bucket_input
        ]
    )

# Show the interface
interface.launch()
