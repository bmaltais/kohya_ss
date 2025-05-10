import torch
from safetensors.torch import save_file
from collections import OrderedDict
import json

# --- Script Configuration ---
# This script generates a minimal, non-functional LoHA (LyCORIS Hadamard Product Adaptation)
# .safetensors file, designed to be structurally compatible with ComfyUI and
# based on the analysis of a working SDXL LoHA file.

# --- Global LoHA Parameters (mimicking metadata from your working file) ---
# These can be overridden per layer if needed for more complex dummies.
# From your metadata: ss_network_dim: 32, ss_network_alpha: 32.0
DEFAULT_RANK = 32
DEFAULT_ALPHA = 32.0
CONV_RANK = 8 # From your ss_network_args: "conv_dim": "8"
CONV_ALPHA = 4.0 # From your ss_network_args: "conv_alpha": "4"

# Define example target layers.
# We'll use names and dimensions that are representative of SDXL and your analysis.
# Format: (layer_name, in_dim, out_dim, rank, alpha)
# Note: For Conv2d, in_dim = in_channels, out_dim = out_channels.
#       The hada_wX_b for conv will have shape (rank, in_channels * kernel_h * kernel_w)
#       For simplicity in this dummy, we'll primarily focus on linear/attention
#       layers first, and then add one representative conv-like layer.

# Layer that previously caused error:
# "ERROR loha diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v.weight shape '[640, 640]' is invalid..."
# This corresponds to lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_v
# In your working LoHA, similar attention layers (e.g., *_attn1_to_k) have out_dim=640, in_dim=640, rank=32, alpha=32.0

EXAMPLE_LAYERS_CONFIG = [
    # UNet Attention Layers (mimicking typical SDXL structure)
    {
        "name": "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q", # Query
        "in_dim": 640, "out_dim": 640, "rank": DEFAULT_RANK, "alpha": DEFAULT_ALPHA, "is_conv": False
    },
    {
        "name": "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k", # Key
        "in_dim": 640, "out_dim": 640, "rank": DEFAULT_RANK, "alpha": DEFAULT_ALPHA, "is_conv": False
    },
    {
        "name": "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_v", # Value - this one errored previously
        "in_dim": 640, "out_dim": 640, "rank": DEFAULT_RANK, "alpha": DEFAULT_ALPHA, "is_conv": False
    },
    {
        "name": "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_out_0", # Output Projection
        "in_dim": 640, "out_dim": 640, "rank": DEFAULT_RANK, "alpha": DEFAULT_ALPHA, "is_conv": False
    },
    # A deeper UNet attention block
    {
        "name": "lora_unet_middle_block_1_transformer_blocks_0_attn1_to_q",
        "in_dim": 1280, "out_dim": 1280, "rank": DEFAULT_RANK, "alpha": DEFAULT_ALPHA, "is_conv": False
    },
    {
        "name": "lora_unet_middle_block_1_transformer_blocks_0_attn1_to_out_0",
        "in_dim": 1280, "out_dim": 1280, "rank": DEFAULT_RANK, "alpha": DEFAULT_ALPHA, "is_conv": False
    },
    # Example UNet "Convolutional" LoHA (e.g., for a ResBlock's conv layer)
    # Based on your lora_unet_input_blocks_1_0_in_layers_2 which had rank 8, alpha 4
    # Assuming original conv was Conv2d(320, 320, kernel_size=3, padding=1)
    {
        "name": "lora_unet_input_blocks_1_0_in_layers_2",
        "in_dim": 320, # in_channels
        "out_dim": 320, # out_channels
        "rank": CONV_RANK,
        "alpha": CONV_ALPHA,
        "is_conv": True,
        "kernel_size": 3 # Assume 3x3 kernel for this example
    },
    # Example Text Encoder Layer (CLIP-L, first one from your list)
    # lora_te1_text_model_encoder_layers_0_mlp_fc1 (original Linear(768, 3072))
    {
        "name": "lora_te1_text_model_encoder_layers_0_mlp_fc1",
        "in_dim": 768, "out_dim": 3072, "rank": DEFAULT_RANK, "alpha": DEFAULT_ALPHA, "is_conv": False
    },
]

# Use bfloat16 as seen in the analysis
DTYPE = torch.bfloat16

# --- Main Script ---
def create_dummy_loha_file(filepath="dummy_loha_corrected.safetensors"):
    """
    Creates and saves a dummy LoHA .safetensors file with corrected structure
    and metadata based on analysis of a working file.
    """
    state_dict = OrderedDict()
    metadata = OrderedDict()

    print(f"Generating dummy LoHA with default rank={DEFAULT_RANK}, default alpha={DEFAULT_ALPHA}")
    print(f"Targeting DTYPE: {DTYPE}")

    for layer_config in EXAMPLE_LAYERS_CONFIG:
        layer_name = layer_config["name"]
        in_dim = layer_config["in_dim"]
        out_dim = layer_config["out_dim"]
        rank = layer_config["rank"]
        alpha = layer_config["alpha"]
        is_conv = layer_config["is_conv"]

        print(f"Processing layer: {layer_name} (in: {in_dim}, out: {out_dim}, rank: {rank}, alpha: {alpha}, conv: {is_conv})")

        # --- LoHA Tensor Shapes Correction based on analysis ---
        # hada_wX_a (maps to original layer's out_features): (out_dim, rank)
        # hada_wX_b (maps from original layer's in_features): (rank, in_dim)
        # For Convolutions, in_dim refers to in_channels, out_dim to out_channels.
        # For hada_wX_b in conv, the effective input dimension includes kernel size.

        if is_conv:
            kernel_size = layer_config.get("kernel_size", 3) # Default to 3x3 if not specified
            # This is for LoHA types that decompose the full kernel (e.g. LyCORIS full conv):
            # (rank, in_channels * kernel_h * kernel_w)
            # For simpler conv LoHA (like applying to 1x1 equivalent), it might just be (rank, in_channels)
            # The analysis for `lora_unet_input_blocks_1_0_in_layers_2` showed hada_w1_b as [8, 2880]
            # where in_dim=320, rank=8. 2880 = 320 * 9 (i.e., in_channels * kernel_h * kernel_w for 3x3)
            # This indicates a full kernel decomposition.
            eff_in_dim_conv_b = in_dim * kernel_size * kernel_size
            
            hada_w1_a = torch.randn(out_dim, rank, dtype=DTYPE) * 0.01
            hada_w1_b = torch.randn(rank, eff_in_dim_conv_b, dtype=DTYPE) * 0.01
            hada_w2_a = torch.randn(out_dim, rank, dtype=DTYPE) * 0.01
            hada_w2_b = torch.randn(rank, eff_in_dim_conv_b, dtype=DTYPE) * 0.01
        else: # Linear layers
            hada_w1_a = torch.randn(out_dim, rank, dtype=DTYPE) * 0.01
            hada_w1_b = torch.randn(rank, in_dim, dtype=DTYPE) * 0.01
            hada_w2_a = torch.randn(out_dim, rank, dtype=DTYPE) * 0.01
            hada_w2_b = torch.randn(rank, in_dim, dtype=DTYPE) * 0.01

        state_dict[f"{layer_name}.hada_w1_a"] = hada_w1_a
        state_dict[f"{layer_name}.hada_w1_b"] = hada_w1_b
        state_dict[f"{layer_name}.hada_w2_a"] = hada_w2_a
        state_dict[f"{layer_name}.hada_w2_b"] = hada_w2_b

        # Alpha tensor (scalar)
        state_dict[f"{layer_name}.alpha"] = torch.tensor(float(alpha), dtype=DTYPE)

        # IMPORTANT: No per-module ".dim" tensor, as per analysis of working file.
        # Rank is implicit in weight shapes and global metadata.

    # --- Metadata (mimicking the working LoHA file) ---
    metadata["ss_network_module"] = "lycoris.kohya"
    metadata["ss_network_dim"] = str(DEFAULT_RANK)  # Global/default rank
    metadata["ss_network_alpha"] = str(DEFAULT_ALPHA) # Global/default alpha
    metadata["ss_network_algo"] = "loha" # Also specified inside ss_network_args by convention

    # Mimic ss_network_args from your file
    network_args = {
        "conv_dim": str(CONV_RANK),
        "conv_alpha": str(CONV_ALPHA),
        "algo": "loha",
        # Add other args from your file if they seem relevant for loading structure,
        # but these are the most critical for type/rank.
        "dropout": "0.0", # From your file, though value might not matter for dummy
        "rank_dropout": "0", # from your file
        "module_dropout": "0", # from your file
        "use_tucker": "False", # from your file
        "use_scalar": "False", # from your file
        "rank_dropout_scale": "False", # from your file
        "train_norm": "False" # from your file
    }
    metadata["ss_network_args"] = json.dumps(network_args)

    # Other potentially useful metadata from your working file (optional for basic loading)
    metadata["ss_sd_model_name"] = "sd_xl_base_1.0.safetensors" # Example base model
    metadata["ss_resolution"] = "(1024,1024)" # Example, format might vary
    metadata["modelspec.sai_model_spec"] = "1.0.0"
    metadata["modelspec.implementation"] = "https_//github.com/Stability-AI/generative-models" # fixed typo
    metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base/lora" # Even for LoHA, this is often used
    metadata["ss_mixed_precision"] = "bf16"
    metadata["ss_note"] = "Dummy LoHA (corrected) for ComfyUI validation. Not trained."


    # --- Save the State Dictionary with Metadata ---
    try:
        save_file(state_dict, filepath, metadata=metadata)
        print(f"\nSuccessfully saved dummy LoHA file to: {filepath}")
        print("\nFile structure (tensor keys):")
        for key in state_dict.keys():
            print(f"- {key}: shape {state_dict[key].shape}, dtype {state_dict[key].dtype}")
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"- {key}: {value}")

    except Exception as e:
        print(f"\nError saving file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_dummy_loha_file()

    # --- Verification Note for ComfyUI ---
    # 1. Place `dummy_loha_corrected.safetensors` into `ComfyUI/models/loras/`.
    # 2. Load an SDXL base model in ComfyUI.
    # 3. Add a "Load LoRA" node and select `dummy_loha_corrected.safetensors`.
    # 4. Connect the LoRA node between the checkpoint loader and the KSampler.
    #
    # Expected outcome:
    # - ComfyUI should load the file without "key not loaded" or "dimension mismatch" errors
    #   for the layers defined in EXAMPLE_LAYERS_CONFIG.
    # - The LoRA node should correctly identify it as a LoHA/LyCORIS model.
    # - If you have layers in your SDXL model that match the names in EXAMPLE_LAYERS_CONFIG,
    #   ComfyUI will attempt to apply these (random) weights.