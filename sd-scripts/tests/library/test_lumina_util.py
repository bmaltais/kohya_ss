import torch
from torch.nn.modules import conv

from library import lumina_util


def test_unpack_latents():
    # Create a test tensor
    # Shape: [batch, height*width, channels*patch_height*patch_width]
    x = torch.randn(2, 4, 16)  # 2 batches, 4 tokens, 16 channels
    packed_latent_height = 2
    packed_latent_width = 2

    # Unpack the latents
    unpacked = lumina_util.unpack_latents(x, packed_latent_height, packed_latent_width)

    # Check output shape
    # Expected shape: [batch, channels, height*patch_height, width*patch_width]
    assert unpacked.shape == (2, 4, 4, 4)


def test_pack_latents():
    # Create a test tensor
    # Shape: [batch, channels, height*patch_height, width*patch_width]
    x = torch.randn(2, 4, 4, 4)

    # Pack the latents
    packed = lumina_util.pack_latents(x)

    # Check output shape
    # Expected shape: [batch, height*width, channels*patch_height*patch_width]
    assert packed.shape == (2, 4, 16)


def test_convert_diffusers_sd_to_alpha_vllm():
    num_double_blocks = 2
    # Predefined test cases based on the actual conversion map
    test_cases = [
        # Static key conversions with possible list mappings
        {
            "original_keys": ["time_caption_embed.caption_embedder.0.weight"],
            "original_pattern": ["time_caption_embed.caption_embedder.0.weight"],
            "expected_converted_keys": ["cap_embedder.0.weight"],
        },
        {
            "original_keys": ["patch_embedder.proj.weight"],
            "original_pattern": ["patch_embedder.proj.weight"],
            "expected_converted_keys": ["x_embedder.weight"],
        },
        {
            "original_keys": ["transformer_blocks.0.norm1.weight"],
            "original_pattern": ["transformer_blocks.().norm1.weight"],
            "expected_converted_keys": ["layers.0.attention_norm1.weight"],
        },
    ]


    for test_case in test_cases:
        for original_key, original_pattern, expected_converted_key in zip(
            test_case["original_keys"], test_case["original_pattern"], test_case["expected_converted_keys"]
        ):
            # Create test state dict
            test_sd = {original_key: torch.randn(10, 10)}

            # Convert the state dict
            converted_sd = lumina_util.convert_diffusers_sd_to_alpha_vllm(test_sd, num_double_blocks)

            # Verify conversion (handle both string and list keys)
            # Find the correct converted key
            match_found = False
            if expected_converted_key in converted_sd:
                # Verify tensor preservation
                assert torch.allclose(converted_sd[expected_converted_key], test_sd[original_key], atol=1e-6), (
                    f"Tensor mismatch for {original_key}"
                )
                match_found = True
                break

            assert match_found, f"Failed to convert {original_key}"

            # Ensure original key is also present
            assert original_key in converted_sd

    # Test with block-specific keys
    block_specific_cases = [
        {
            "original_pattern": "transformer_blocks.().norm1.weight",
            "converted_pattern": "layers.().attention_norm1.weight",
        }
    ]

    for case in block_specific_cases:
        for block_idx in range(2):  # Test multiple block indices
            # Prepare block-specific keys
            block_original_key = case["original_pattern"].replace("()", str(block_idx))
            block_converted_key = case["converted_pattern"].replace("()", str(block_idx))
            print(block_original_key, block_converted_key)

            # Create test state dict
            test_sd = {block_original_key: torch.randn(10, 10)}

            # Convert the state dict
            converted_sd = lumina_util.convert_diffusers_sd_to_alpha_vllm(test_sd, num_double_blocks)

            # Verify conversion
            # assert block_converted_key in converted_sd, f"Failed to convert block key {block_original_key}"
            assert torch.allclose(converted_sd[block_converted_key], test_sd[block_original_key], atol=1e-6), (
                f"Tensor mismatch for block key {block_original_key}"
            )

            # Ensure original key is also present
            assert block_original_key in converted_sd
