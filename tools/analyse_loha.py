import safetensors.torch
import json
from collections import OrderedDict
import sys # To redirect stdout
import traceback

class Logger(object):
    def __init__(self, filename="loha_analysis_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command, which shutil.copytree or os.system uses.
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def analyze_safetensors_file(filepath, output_filename="loha_analysis_output.txt"):
    """
    Analyzes a .safetensors file to extract and print its metadata
    and tensor information (keys, shapes, dtypes) to a file.
    """
    original_stdout = sys.stdout
    logger = Logger(filename=output_filename)
    sys.stdout = logger

    try:
        print(f"--- Analyzing: {filepath} ---\n")
        print(f"--- Output will be saved to: {output_filename} ---\n")

        # Load the tensors to get their structure
        state_dict = safetensors.torch.load_file(filepath, device="cpu") # Load to CPU to avoid potential CUDA issues

        print("--- Tensor Information ---")
        if not state_dict:
            print("No tensors found in the state dictionary.")
        else:
            # Sort keys for consistent output
            sorted_keys = sorted(state_dict.keys())
            current_module_prefix = ""
            
            # First, identify all unique module prefixes for better grouping
            module_prefixes = sorted(list(set([".".join(key.split(".")[:-1]) for key in sorted_keys if "." in key])))

            for prefix in module_prefixes:
                if not prefix: # Skip keys that don't seem to be part of a module (e.g. global metadata tensors if any)
                    continue
                print(f"\nModule: {prefix}")
                for key in sorted_keys:
                    if key.startswith(prefix + "."):
                        tensor = state_dict[key]
                        print(f"  - Key: {key}")
                        print(f"    Shape: {list(tensor.shape)}, Dtype: {tensor.dtype}") # Output shape as list for clarity
                        if key.endswith((".alpha", ".dim")):
                            try:
                                value = tensor.item()
                                # Check if value is float and format if it is
                                if isinstance(value, float):
                                     print(f"    Value: {value:.8f}") # Format float to a certain precision
                                else:
                                    print(f"    Value: {value}")
                            except Exception as e:
                                print(f"    Value: Could not extract scalar value ({tensor}, error: {e})")
                        elif tensor.numel() < 10: # Print small tensors' values
                            print(f"    Values (first few): {tensor.flatten()[:10].tolist()}")


            # Print keys that might not fit the module pattern (e.g., older formats or single tensors)
            print("\n--- Other Tensor Keys (if any, not fitting typical module.parameter pattern) ---")
            other_keys_found = False
            for key in sorted_keys:
                if not any(key.startswith(p + ".") for p in module_prefixes if p):
                    other_keys_found = True
                    tensor = state_dict[key]
                    print(f"  - Key: {key}")
                    print(f"    Shape: {list(tensor.shape)}, Dtype: {tensor.dtype}")
                    if key.endswith((".alpha", ".dim")) or tensor.numel() == 1:
                         try:
                            value = tensor.item()
                            if isinstance(value, float):
                                print(f"    Value: {value:.8f}")
                            else:
                                print(f"    Value: {value}")
                         except Exception as e:
                            print(f"    Value: Could not extract scalar value ({tensor}, error: {e})")

            if not other_keys_found:
                print("No other keys found.")

            print(f"\nTotal tensor keys found: {len(state_dict)}")

        print("\n--- Metadata (from safetensors header) ---")
        metadata_content = OrderedDict()
        malformed_metadata_keys = []
        try:
            # Use safe_open to access the metadata separately
            with safetensors.safe_open(filepath, framework="pt", device="cpu") as f:
                metadata_keys = f.metadata()
                if metadata_keys is None:
                    print("No metadata dictionary found in the file header (f.metadata() returned None).")
                else:
                    for k in metadata_keys.keys():
                        try:
                            metadata_content[k] = metadata_keys.get(k)
                        except Exception as e:
                            malformed_metadata_keys.append((k, str(e)))
                            metadata_content[k] = f"[Error reading value: {e}]"
        except Exception as e:
            print(f"Could not open or read metadata using safe_open: {e}")
            traceback.print_exc(file=sys.stdout)

        if not metadata_content and not malformed_metadata_keys:
            print("No metadata content extracted.")
        else:
            for key, value in metadata_content.items():
                print(f"- {key}: {value}")
                if key == "ss_network_args" and value and not value.startswith("[Error"):
                    try:
                        parsed_args = json.loads(value)
                        print("  Parsed ss_network_args:")
                        for arg_key, arg_value in parsed_args.items():
                            print(f"    - {arg_key}: {arg_value}")
                    except json.JSONDecodeError:
                        print("    (ss_network_args is not a valid JSON string)")
            if malformed_metadata_keys:
                print("\n--- Malformed Metadata Keys (could not be read) ---")
                for key, error_msg in malformed_metadata_keys:
                    print(f"- {key}: Error: {error_msg}")

        print("\n--- End of Analysis ---")

    except Exception as e:
        print(f"\n!!! An error occurred during analysis !!!")
        print(str(e))
        traceback.print_exc(file=sys.stdout) # Print full traceback to the log file
    finally:
        sys.stdout = original_stdout # Restore standard output
        logger.close()
        print(f"\nAnalysis complete. Output saved to: {output_filename}")


if __name__ == "__main__":
    input_file_path = input("Enter the path to your working LoHA .safetensors file: ")
    output_file_name = "loha_analysis_results.txt" # You can change this default
    
    # Suggest a default output name based on input file if desired
    # import os
    # base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    # output_file_name = f"{base_name}_analysis.txt"
    
    print(f"The analysis will be saved to: {output_file_name}")
    analyze_safetensors_file(input_file_path, output_filename=output_file_name)