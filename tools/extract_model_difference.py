import torch
from safetensors.torch import load_file, save_file
from collections import OrderedDict
import os
import argparse # Import argparse

def extract_model_differences(base_model_path, finetuned_model_path, output_delta_path=None, save_dtype_str="float32"):
    """
    Calculates the difference between the state dictionaries of a fine-tuned model
    and a base model.

    Args:
        base_model_path (str): Path to the base model .safetensors file.
        finetuned_model_path (str): Path to the fine-tuned model .safetensors file.
        output_delta_path (str, optional): Path to save the resulting delta weights
                                           .safetensors file. If None, not saved.
        save_dtype_str (str, optional): Data type to save the delta weights ('float32', 'float16', 'bfloat16').
                                        Defaults to 'float32'.
    Returns:
        OrderedDict: A state dictionary containing the delta weights.
                     Returns None if loading fails or other critical errors.
    """
    print(f"Loading base model from: {base_model_path}")
    try:
        # Ensure model is loaded to CPU to avoid CUDA issues if not needed for diffing
        base_state_dict = load_file(base_model_path, device="cpu")
        print(f"Base model loaded. Found {len(base_state_dict)} tensors.")
    except Exception as e:
        print(f"Error loading base model: {e}")
        return None

    print(f"\nLoading fine-tuned model from: {finetuned_model_path}")
    try:
        finetuned_state_dict = load_file(finetuned_model_path, device="cpu")
        print(f"Fine-tuned model loaded. Found {len(finetuned_state_dict)} tensors.")
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return None

    delta_state_dict = OrderedDict()
    diff_count = 0
    skipped_count = 0
    error_count = 0
    unique_to_finetuned_count = 0
    unique_to_base_count = 0

    print("\nCalculating differences...")

    # Keys in finetuned model
    finetuned_keys = set(finetuned_state_dict.keys())
    base_keys = set(base_state_dict.keys())

    common_keys = finetuned_keys.intersection(base_keys)
    keys_only_in_finetuned = finetuned_keys - base_keys
    keys_only_in_base = base_keys - finetuned_keys

    for key in common_keys:
        ft_tensor = finetuned_state_dict[key]
        base_tensor = base_state_dict[key]

        if not (ft_tensor.is_floating_point() and base_tensor.is_floating_point()):
            # print(f"Skipping key '{key}': Non-floating point tensors (FT: {ft_tensor.dtype}, Base: {base_tensor.dtype}).")
            skipped_count += 1
            continue

        if ft_tensor.shape != base_tensor.shape:
            print(f"Skipping key '{key}': Shape mismatch (FT: {ft_tensor.shape}, Base: {base_tensor.shape}).")
            skipped_count += 1
            continue

        try:
            # Calculate difference in float32 for precision, then cast to save_dtype
            delta_tensor = ft_tensor.to(dtype=torch.float32) - base_tensor.to(dtype=torch.float32)
            delta_state_dict[key] = delta_tensor
            diff_count += 1
        except Exception as e:
            print(f"Error calculating difference for key '{key}': {e}")
            error_count += 1

    for key in keys_only_in_finetuned:
        print(f"Warning: Key '{key}' (Shape: {finetuned_state_dict[key].shape}, Dtype: {finetuned_state_dict[key].dtype}) is present in fine-tuned model but not in base model. Storing as is.")
        delta_state_dict[key] = finetuned_state_dict[key] # Store the tensor from the finetuned model
        unique_to_finetuned_count += 1
        
    if keys_only_in_base:
        print(f"\nWarning: {len(keys_only_in_base)} key(s) are present only in the base model and will not be in the delta file.")
        for key in list(keys_only_in_base)[:5]: # Print first 5 as examples
             print(f"  - Example key only in base: {key}")
        if len(keys_only_in_base) > 5:
            print(f"  ... and {len(keys_only_in_base) - 5} more.")


    print(f"\nDifference calculation complete.")
    print(f"  {diff_count} layers successfully diffed.")
    print(f"  {unique_to_finetuned_count} layers unique to fine-tuned model (added as is).")
    print(f"  {skipped_count} common layers skipped (shape/type mismatch).")
    print(f"  {error_count} common layers had errors during diffing.")

    if output_delta_path and delta_state_dict:
        save_dtype = torch.float32 # Default
        if save_dtype_str == "float16":
            save_dtype = torch.float16
        elif save_dtype_str == "bfloat16":
            save_dtype = torch.bfloat16
        elif save_dtype_str != "float32":
            print(f"Warning: Invalid save_dtype '{save_dtype_str}'. Defaulting to float32.")
            save_dtype_str = "float32" # for print message

        print(f"\nPreparing to save delta weights with dtype: {save_dtype_str}")
        
        final_save_dict = OrderedDict()
        for k, v_tensor in delta_state_dict.items():
            if v_tensor.is_floating_point():
                final_save_dict[k] = v_tensor.to(dtype=save_dtype)
            else:
                final_save_dict[k] = v_tensor # Keep non-float as is (e.g. int tensors if any)
        
        try:
            save_file(final_save_dict, output_delta_path)
            print(f"Delta weights saved to: {output_delta_path}")
        except Exception as e:
            print(f"Error saving delta weights: {e}")
            import traceback
            traceback.print_exc()


    return delta_state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract weight differences between a fine-tuned and a base SDXL model.")
    parser.add_argument("base_model_path", type=str, help="File path for the BASE SDXL model (.safetensors).")
    parser.add_argument("finetuned_model_path", type=str, help="File path for the FINE-TUNED SDXL model (.safetensors).")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional: File path to save the delta weights (.safetensors). "
                             "If not provided, defaults to 'model_deltas/delta_[finetuned_model_name].safetensors'.")
    parser.add_argument("--save_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                        help="Data type for saving the delta weights. Choose from 'float32', 'float16', 'bfloat16'. "
                             "Defaults to 'float32'.")

    args = parser.parse_args()

    print("--- Model Difference Extraction Script ---")

    if not os.path.exists(args.base_model_path):
        print(f"Error: Base model file not found at {args.base_model_path}")
        exit(1)
    if not os.path.exists(args.finetuned_model_path):
        print(f"Error: Fine-tuned model file not found at {args.finetuned_model_path}")
        exit(1)

    output_delta_file = args.output_path
    if output_delta_file is None:
        output_dir = "model_deltas"
        os.makedirs(output_dir, exist_ok=True)
        finetuned_basename = os.path.splitext(os.path.basename(args.finetuned_model_path))[0]
        output_delta_file = os.path.join(output_dir, f"delta_{finetuned_basename}.safetensors")

    # Ensure the output directory exists if a full path is given
    if output_delta_file:
        output_dir_for_file = os.path.dirname(output_delta_file)
        if output_dir_for_file and not os.path.exists(output_dir_for_file):
            os.makedirs(output_dir_for_file, exist_ok=True)


    differences = extract_model_differences(
        args.base_model_path,
        args.finetuned_model_path,
        output_delta_path=output_delta_file,
        save_dtype_str=args.save_dtype
    )

    if differences:
        print(f"\nExtraction process finished. {len(differences)} total keys in the delta state_dict.")
    else:
        print("\nCould not extract differences due to errors during model loading.")