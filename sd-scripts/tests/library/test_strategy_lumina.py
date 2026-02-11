import os
import tempfile
import torch
import numpy as np
from unittest.mock import patch
from transformers import Gemma2Model

from library.strategy_lumina import (
    LuminaTokenizeStrategy,
    LuminaTextEncodingStrategy,
    LuminaTextEncoderOutputsCachingStrategy,
    LuminaLatentsCachingStrategy,
)


class SimpleMockGemma2Model:
    """Lightweight mock that avoids initializing the actual Gemma2Model"""

    def __init__(self, hidden_size=2304):
        self.device = torch.device("cpu")
        self._hidden_size = hidden_size
        self._orig_mod = self  # For dynamic compilation compatibility

    def __call__(self, input_ids, attention_mask, output_hidden_states=False, return_dict=False):
        # Create a mock output object with hidden states
        batch_size, seq_len = input_ids.shape
        hidden_size = self._hidden_size

        class MockOutput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        mock_hidden_states = [
            torch.randn(batch_size, seq_len, hidden_size, device=input_ids.device)
            for _ in range(3)  # Mimic multiple layers of hidden states
        ]

        return MockOutput(mock_hidden_states)


def test_lumina_tokenize_strategy():
    # Test default initialization
    try:
        tokenize_strategy = LuminaTokenizeStrategy("dummy system prompt", max_length=None)
    except OSError as e:
        # If the tokenizer is not found (due to gated repo), we can skip the test
        print(f"Skipping LuminaTokenizeStrategy test due to OSError: {e}")
        return
    assert tokenize_strategy.max_length == 256
    assert tokenize_strategy.tokenizer.padding_side == "right"

    # Test tokenization of a single string
    text = "Hello"
    tokens, attention_mask = tokenize_strategy.tokenize(text)

    assert tokens.ndim == 2
    assert attention_mask.ndim == 2
    assert tokens.shape == attention_mask.shape
    assert tokens.shape[1] == 256  # max_length

    # Test tokenize_with_weights
    tokens, attention_mask, weights = tokenize_strategy.tokenize_with_weights(text)
    assert len(weights) == 1
    assert torch.all(weights[0] == 1)


def test_lumina_text_encoding_strategy():
    # Create strategies
    try:
        tokenize_strategy = LuminaTokenizeStrategy("dummy system prompt", max_length=None)
    except OSError as e:
        # If the tokenizer is not found (due to gated repo), we can skip the test
        print(f"Skipping LuminaTokenizeStrategy test due to OSError: {e}")
        return
    encoding_strategy = LuminaTextEncodingStrategy()

    # Create a mock model
    mock_model = SimpleMockGemma2Model()

    # Patch the isinstance check to accept our simple mock
    original_isinstance = isinstance
    with patch("library.strategy_lumina.isinstance") as mock_isinstance:

        def custom_isinstance(obj, class_or_tuple):
            if obj is mock_model and class_or_tuple is Gemma2Model:
                return True
            if hasattr(obj, "_orig_mod") and obj._orig_mod is mock_model and class_or_tuple is Gemma2Model:
                return True
            return original_isinstance(obj, class_or_tuple)

        mock_isinstance.side_effect = custom_isinstance

        # Prepare sample text
        text = "Test encoding strategy"
        tokens, attention_mask = tokenize_strategy.tokenize(text)

        # Perform encoding
        hidden_states, input_ids, attention_masks = encoding_strategy.encode_tokens(
            tokenize_strategy, [mock_model], (tokens, attention_mask)
        )

        # Validate outputs
        assert original_isinstance(hidden_states, torch.Tensor)
        assert original_isinstance(input_ids, torch.Tensor)
        assert original_isinstance(attention_masks, torch.Tensor)

        # Check the shape of the second-to-last hidden state
        assert hidden_states.ndim == 3

        # Test weighted encoding (which falls back to standard encoding for Lumina)
        weights = [torch.ones_like(tokens)]
        hidden_states_w, input_ids_w, attention_masks_w = encoding_strategy.encode_tokens_with_weights(
            tokenize_strategy, [mock_model], (tokens, attention_mask), weights
        )

        # For the mock, we can't guarantee identical outputs since each call returns random tensors
        # Instead, check that the outputs have the same shape and are tensors
        assert hidden_states_w.shape == hidden_states.shape
        assert original_isinstance(hidden_states_w, torch.Tensor)
        assert torch.allclose(input_ids, input_ids_w)  # Input IDs should be the same
        assert torch.allclose(attention_masks, attention_masks_w)  # Attention masks should be the same


def test_lumina_text_encoder_outputs_caching_strategy():
    # Create a temporary directory for caching
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a cache file path
        cache_file = os.path.join(tmpdir, "test_outputs.npz")

        # Create the caching strategy
        caching_strategy = LuminaTextEncoderOutputsCachingStrategy(
            cache_to_disk=True,
            batch_size=1,
            skip_disk_cache_validity_check=False,
        )

        # Create a mock class for ImageInfo
        class MockImageInfo:
            def __init__(self, caption, cache_path):
                self.caption = caption
                self.text_encoder_outputs_npz = cache_path

        # Create a sample input info
        image_info = MockImageInfo("Test caption", cache_file)

        # Simulate a batch
        batch = [image_info]

        # Create mock strategies and model
        try:
            tokenize_strategy = LuminaTokenizeStrategy("dummy system prompt", max_length=None)
        except OSError as e:
            # If the tokenizer is not found (due to gated repo), we can skip the test
            print(f"Skipping LuminaTokenizeStrategy test due to OSError: {e}")
            return
        encoding_strategy = LuminaTextEncodingStrategy()
        mock_model = SimpleMockGemma2Model()

        # Patch the isinstance check to accept our simple mock
        original_isinstance = isinstance
        with patch("library.strategy_lumina.isinstance") as mock_isinstance:

            def custom_isinstance(obj, class_or_tuple):
                if obj is mock_model and class_or_tuple is Gemma2Model:
                    return True
                if hasattr(obj, "_orig_mod") and obj._orig_mod is mock_model and class_or_tuple is Gemma2Model:
                    return True
                return original_isinstance(obj, class_or_tuple)

            mock_isinstance.side_effect = custom_isinstance

            # Call cache_batch_outputs
            caching_strategy.cache_batch_outputs(tokenize_strategy, [mock_model], encoding_strategy, batch)

        # Verify the npz file was created
        assert os.path.exists(cache_file), f"Cache file not created at {cache_file}"

        # Verify the is_disk_cached_outputs_expected method
        assert caching_strategy.is_disk_cached_outputs_expected(cache_file)

        # Test loading from npz
        loaded_data = caching_strategy.load_outputs_npz(cache_file)
        assert len(loaded_data) == 3  # hidden_state, input_ids, attention_mask


def test_lumina_latents_caching_strategy():
    # Create a temporary directory for caching
    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare a mock absolute path
        abs_path = os.path.join(tmpdir, "test_image.png")

        # Use smaller image size for faster testing
        image_size = (64, 64)

        # Create a smaller dummy image for testing
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        # Create the caching strategy
        caching_strategy = LuminaLatentsCachingStrategy(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)

        # Create a simple mock VAE
        class MockVAE:
            def __init__(self):
                self.device = torch.device("cpu")
                self.dtype = torch.float32

            def encode(self, x):
                # Return smaller encoded tensor for faster processing
                encoded = torch.randn(1, 4, 8, 8, device=x.device)
                return type("EncodedLatents", (), {"to": lambda *args, **kwargs: encoded})

        # Prepare a mock batch
        class MockImageInfo:
            def __init__(self, path, image):
                self.absolute_path = path
                self.image = image
                self.image_path = path
                self.bucket_reso = image_size
                self.resized_size = image_size
                self.resize_interpolation = "lanczos"
                # Specify full path to the latents npz file
                self.latents_npz = os.path.join(tmpdir, f"{os.path.splitext(os.path.basename(path))[0]}_0064x0064_lumina.npz")

        batch = [MockImageInfo(abs_path, test_image)]

        # Call cache_batch_latents
        mock_vae = MockVAE()
        caching_strategy.cache_batch_latents(mock_vae, batch, flip_aug=False, alpha_mask=False, random_crop=False)

        # Generate the expected npz path
        npz_path = caching_strategy.get_latents_npz_path(abs_path, image_size)

        # Verify the file was created
        assert os.path.exists(npz_path), f"NPZ file not created at {npz_path}"

        # Verify is_disk_cached_latents_expected
        assert caching_strategy.is_disk_cached_latents_expected(image_size, npz_path, False, False)

        # Test loading from disk
        loaded_data = caching_strategy.load_latents_from_disk(npz_path, image_size)
        assert len(loaded_data) == 5  # Check for 5 expected elements
