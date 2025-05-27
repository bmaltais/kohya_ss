import gradio as gr
import subprocess
import os
import sys
from kohya_gui import common_gui

# Define the script directory
scriptdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PYTHON = sys.executable

# Placeholder for the extract_lora_new function
def extract_lora_new(
    model_tuned,
    model_org,
    save_to,
    save_precision,
    load_precision,
    dim,
    conv_dim,
    device,
    sdxl,
    v2,
    v_parameterization,
    clamp_quantile,
    min_diff,
    load_original_model_to,
    load_tuned_model_to,
    dynamic_method,
    dynamic_param,
    verbose,
    no_metadata,
):
    # Construct the command
    command = [
        PYTHON,
        os.path.join(scriptdir, "tools", "extract_lora_from_models-new.py"),
    ]

    # Add arguments based on user input
    if model_tuned:
        command.extend(["--model_tuned", model_tuned])
    if model_org:
        command.extend(["--model_org", model_org])
    if save_to:
        command.extend(["--save_to", save_to])
    if save_precision:
        command.extend(["--save_precision", save_precision])
    if load_precision and load_precision != "None":
        command.extend(["--load_precision", load_precision])
    command.extend(["--dim", str(dim)])
    if conv_dim > 0:
        command.extend(["--conv_dim", str(conv_dim)])
    if device:
        command.extend(["--device", device])
    if sdxl:
        command.append("--sdxl")
    if v2:
        command.append("--v2")
    if v_parameterization:
        command.append("--v_parameterization")
    command.extend(["--clamp_quantile", str(clamp_quantile)])
    command.extend(["--min_diff", str(min_diff)])

    if sdxl:
        if load_original_model_to:
            command.extend(["--load_original_model_to", load_original_model_to])
        if load_tuned_model_to:
            command.extend(["--load_tuned_model_to", load_tuned_model_to])
    
    if dynamic_method and dynamic_method != "None":
        command.extend(["--dynamic_method", dynamic_method])
        command.extend(["--dynamic_param", str(dynamic_param)])

    if verbose:
        command.append("--verbose")
    if no_metadata:
        command.append("--no_metadata")

    # Run the script
    print(f"Running command: {' '.join(command)}") # Log to console
    
    all_logs = ""
    try:
        # Use Popen to capture stdout and stderr
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Capture stderr separately
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        
        # Stream stdout
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    print(line) # Log to console
                    all_logs += line + '\n'
                    yield all_logs # Yield accumulated logs
        
        # Stream stderr
        # After stdout is exhausted, check stderr
        stderr_output = ""
        if process.stderr:
            for line in iter(process.stderr.readline, ''):
                line = line.strip()
                if line:
                    print(f"Error: {line}") # Log to console
                    stderr_output += f"ERROR: {line}\n"
        
        process.wait() # Wait for the process to complete

        # Append any stderr output to all_logs after stdout and process completion
        if stderr_output:
            all_logs += "\n--- Errors/Warnings ---\n" + stderr_output
            yield all_logs

        if process.returncode == 0:
            all_logs += "\nLoRA extraction completed successfully."
        else:
            all_logs += f"\nError during LoRA extraction. Return code: {process.returncode}"
        
        yield all_logs

    except Exception as e:
        all_logs += f"\nFailed to run script: {e}"
        yield all_logs
    finally:
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

# Gradio UI function
def gradio_extract_lora_new_tab(headless=False):
    with gr.Tab("Extract LoRA (New)"):
        # Add hidden textboxes for file extensions
        model_ext = gr.Textbox(value="*.ckpt *.safetensors", visible=False)
        model_ext_name = gr.Textbox(value="Model types", visible=False)
        lora_ext = gr.Textbox(value="*.safetensors", visible=False)
        lora_ext_name = gr.Textbox(value="LoRA model types", visible=False)
        
        gr.Markdown("## Extract LoRA from Models (New Script)")
        gr.Markdown(
            "This utility extracts LoRA weights from the difference between a fine-tuned model and its original base model."
            " It uses the `tools/extract_lora_from_models-new.py` script."
        )

        with gr.Row():
            model_tuned = gr.Textbox(
                label="Finetuned Model Path",
                placeholder="Path to your finetuned .safetensors model",
                interactive=True,
            )
            button_model_tuned_file = gr.Button(
                "📂", elem_id="open_folder_small", visible=not headless
            )
            button_model_tuned_file.click(
                common_gui.get_file_path,
                inputs=[model_tuned, model_ext, model_ext_name],
                outputs=model_tuned,
                show_progress=False,
            )

            model_org = gr.Textbox(
                label="Original Base Model Path",
                placeholder="Path to the original .safetensors model",
                interactive=True,
            )
            button_model_org_file = gr.Button(
                "📂", elem_id="open_folder_small", visible=not headless
            )
            button_model_org_file.click(
                common_gui.get_file_path,
                inputs=[model_org, model_ext, model_ext_name],
                outputs=model_org,
                show_progress=False,
            )

            save_to = gr.Textbox(
                label="Save LoRA Model to",
                placeholder="Path to save the extracted LoRA .safetensors model",
                interactive=True,
            )
            button_save_to_file = gr.Button(
                "📂", elem_id="open_folder_small", visible=not headless
            )
            button_save_to_file.click(
                common_gui.get_saveasfilename_path,
                inputs=[save_to, lora_ext, lora_ext_name],
                outputs=save_to,
                show_progress=False,
            )
        
        with gr.Row():
            save_precision = gr.Radio(
                label="Save Precision",
                choices=["float", "fp16", "bf16"],
                value="float",
                interactive=True,
            )
            load_precision = gr.Radio(
                label="Load Precision (for calculation)",
                choices=["None", "float", "fp16", "bf16"], # Added None
                value="None", # Default to None
                interactive=True,
            )
            device_choices = ["cuda", "cpu"]
            try:
                import torch
                if torch.cuda.is_available():
                    device_choices = ["cuda"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"]
            except ImportError:
                pass # Keep default if torch is not available
            
            device = gr.Radio(
                label="Device",
                choices=device_choices,
                value="cuda" if "cuda" in device_choices else "cpu",
                interactive=True,
            )

        with gr.Row():
            dim = gr.Slider(
                label="Dimension (Rank)",
                minimum=1,
                maximum=512,
                step=1,
                value=4,
                interactive=True,
            )
            conv_dim = gr.Slider(
                label="Convolution Dimension (Rank)",
                minimum=0, # 0 might mean use 'dim' or skip
                maximum=512,
                step=1,
                value=4, # Default to dim's value initially
                interactive=True,
            )
            # Update conv_dim when dim changes, if conv_dim is meant to default to dim
            # This requires a JavaScript link or a Python callback if we want strict mirroring
            # For now, we'll set its default and let user adjust. The script handles conv_dim=0 or unprovided.

        with gr.Row():
            sdxl = gr.Checkbox(label="SDXL Model", value=False, interactive=True)
            v2 = gr.Checkbox(label="v2 Model", value=False, interactive=True)
            v_parameterization = gr.Checkbox(
                label="v_parameterization (for v2)", value=False, interactive=True
            )
        
        with gr.Row():
            clamp_quantile = gr.Number(
                label="Clamp Quantile", value=0.99, minimum=0, maximum=1, step=0.001, interactive=True
            )
            min_diff = gr.Number(
                label="Minimum Difference", value=0.01, minimum=0, maximum=1, step=0.001, interactive=True
            )

        with gr.Blocks() as sdxl_options: # Conditional visibility for SDXL options
            gr.Markdown("### SDXL Specific Load Options")
            with gr.Row():
                load_original_model_to_choices = ["cpu", "cuda"]
                load_tuned_model_to_choices = ["cpu", "cuda"]
                try:
                    import torch
                    if torch.cuda.is_available():
                        cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                        load_original_model_to_choices.extend(cuda_devices)
                        load_tuned_model_to_choices.extend(cuda_devices)
                except ImportError:
                    pass

                load_original_model_to = gr.Dropdown(
                    label="Load Original SDXL Model to Device",
                    choices=load_original_model_to_choices,
                    value="cpu",
                    interactive=True,
                )
                load_tuned_model_to = gr.Dropdown(
                    label="Load Tuned SDXL Model to Device",
                    choices=load_tuned_model_to_choices,
                    value="cpu",
                    interactive=True,
                )
        # Logic for sdxl_options visibility:
        # sdxl.change(lambda x: gr.update(visible=x), inputs=sdxl, outputs=sdxl_options)
        # Updated to use the correct way to update visibility for a gr.Blocks() context manager
        sdxl.change(
            fn=lambda x: {"visible": x, "__type__": "update"}, inputs=[sdxl], outputs=[sdxl_options],
        )


        with gr.Row():
            dynamic_method_choices = ["None", "sv_ratio", "sv_fro", "sv_cumulative", "sv_knee", "sv_rel_decrease", "sv_cumulative_knee"]
            dynamic_method = gr.Dropdown(
                label="Dynamic Rank Method",
                choices=dynamic_method_choices,
                value="None",
                interactive=True,
            )
            dynamic_param = gr.Number(
                label="Dynamic Rank Parameter",
                value=0.9, # A generic default, might need adjustment based on method
                interactive=True,
                visible=False, # Initially hidden
            )
        
        # Show dynamic_param only if a method that uses it is selected
        # This requires knowing which methods need a parameter. Assuming all except "None".
        # dynamic_method.change(
        #     lambda x: gr.update(visible=x != "None"), 
        #     inputs=dynamic_method, 
        #     outputs=dynamic_param
        # )
        dynamic_method.change(
            fn=lambda x: {"visible": x != "None", "__type__": "update"}, inputs=[dynamic_method], outputs=[dynamic_param],
        )


        with gr.Row():
            verbose = gr.Checkbox(label="Verbose Logging", value=False, interactive=True)
            no_metadata = gr.Checkbox(label="No Metadata", value=False, interactive=True)

        extract_button = gr.Button("Extract LoRA (New)", variant="primary")
        
        # Output/logging area
        # output_text = gr.Textbox(label="Output", lines=10, interactive=False, show_copy_button=True)
        # Updated to use the custom logging text area
        output_logs = gr.Textbox(
            label="Output / Logs",
            lines=10,
            interactive=False,
            show_copy_button=True,
            max_lines=200  # Or some reasonable limit
        )


        extract_button.click(
            extract_lora_new,
            inputs=[
                model_tuned,
                model_org,
                save_to,
                save_precision,
                load_precision,
                dim,
                conv_dim,
                device,
                sdxl,
                v2,
                v_parameterization,
                clamp_quantile,
                min_diff,
                load_original_model_to,
                load_tuned_model_to,
                dynamic_method,
                dynamic_param,
                verbose,
                no_metadata,
            ],
            outputs=[output_logs], # Output to the log textbox
            show_progress="full"
        )
        
        # Add refresh buttons for file/folder pickers
        common_gui.create_refresh_button(model_tuned, common_gui.get_file_path, [model_tuned, model_ext, model_ext_name], "open_folder_small_refresh")
        common_gui.create_refresh_button(model_org, common_gui.get_file_path, [model_org, model_ext, model_ext_name], "open_folder_small_refresh")
        common_gui.create_refresh_button(save_to, common_gui.get_saveasfilename_path, [save_to, lora_ext, lora_ext_name], "open_folder_small_refresh")


if __name__ == "__main__":
    # This is for testing the UI components locally
    # You would typically import and use gradio_extract_lora_new_tab in your main Gradio app
    
    # Setup environment for testing if needed (e.g., for common_gui functions)
    # common_gui.setup_common_gui_state() # Example, if setup_common_gui_state exists and is needed

    demo = gr.Blocks()
    with demo:
        gradio_extract_lora_new_tab(headless=False)
    
    # Print Gradio version for debugging
    print(f"Gradio version: {gr.__version__}")

    # Set KㄠOHYA_GUI_HEADLESS if you want to run in headless mode for testing
    # os.environ['KOHYA_GUI_HEADLESS'] = 'true'
    # is_headless = os.environ.get("KOHYA_GUI_HEADLESS", "false").lower() == "true"
    # print(f"Running in headless mode: {is_headless}")

    # gradio_extract_lora_new_tab(headless=is_headless) # Call directly if not embedding

    print("Launching Gradio demo for Extract LoRA (New)...")
    # demo.launch(share=True) # Use share=True for public link if needed for testing
    demo.launch()
