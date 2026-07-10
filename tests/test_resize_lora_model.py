"""Unit tests for resize_lora_model — the bug-fixing cycle for GH #3475.

These tests exercise the dimension/alpha extraction logic in
`resize_lora_model()` to verify that it handles both Linear (2D) and Conv2d
(4D) LoRA weights without crashing with NoneType/NoneType.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

import torch

# Add the repo root to sys.path so we can import from tools directly
sys.path.insert(0, "/tmp/kohya_ss_repo")

from tools.resize_lora import resize_lora_model


class TestResizeLoraModelConv2d(unittest.TestCase):
    """Test that resize_lora_model handles Conv2d (4D) Flux LoRA weights."""

    def test_conv2d_weights_do_not_crash(self):
        """Flux LoRA with Conv2d lora_down/up weights should not crash.

        The original bug: `len(value.size()) == 2` filtered out all Conv2d
        weights, leaving `network_dim = None`, then the fallback set
        `network_alpha = None`, and `None / None` raised TypeError.
        """
        lora_sd = {
            "block.0.lora_down.weight": torch.randn(16, 2048, 1, 1),
            "block.0.lora_up.weight": torch.randn(2048, 16, 1, 1),
            "block.0.lora_down.alpha": torch.tensor(32.0),
        }

        # Mock merge_conv and rank_resize to avoid the full SVD pipeline
        # merge_conv returns a 4D tensor (out_size, in_size, kernel, kernel)
        with patch(
            "tools.resize_lora.merge_conv",
            return_value=torch.randn(2048, 2048, 1, 1),
        ), patch(
            "tools.resize_lora.rank_resize",
            return_value={
                "new_rank": 8,
                "new_alpha": 16.0,
                "sum_retained": 0.95,
                "fro_retained": 0.97,
                "max_ratio": 3.2,
            },
        ):
            # Should not raise TypeError
            result = resize_lora_model(
                lora_sd,
                new_rank=8,
                save_dtype=torch.float32,
                device="cpu",
                dynamic_method="sv_fro",
                dynamic_param=0.99,
                verbose=False,
            )

        self.assertIsNotNone(result)
        state_dict, network_dim, new_alpha = result
        # network_dim should be extracted from Conv2d weight (rank = 16)
        self.assertEqual(network_dim, 16)
        # new_alpha should be a float
        self.assertIsInstance(new_alpha, float)

    def test_no_alpha_key_does_not_crash(self):
        """LoRA state_dict without an explicit alpha key should not crash.

        Original bug: when no 'alpha' key exists in the state_dict,
        `network_alpha` stays None and `None / None` raises TypeError.
        """
        lora_sd = {
            "block.0.lora_down.weight": torch.randn(16, 2048, 1, 1),
            "block.0.lora_up.weight": torch.randn(2048, 16, 1, 1),
            # No alpha key
        }

        with patch(
            "tools.resize_lora.merge_conv",
            return_value=torch.randn(2048, 2048, 1, 1),
        ), patch(
            "tools.resize_lora.rank_resize",
            return_value={
                "new_rank": 8,
                "new_alpha": 8.0,
                "sum_retained": 0.95,
                "fro_retained": 0.97,
                "max_ratio": 3.2,
            },
        ):
            result = resize_lora_model(
                lora_sd,
                new_rank=8,
                save_dtype=torch.float32,
                device="cpu",
                dynamic_method=None,
                dynamic_param=None,
                verbose=False,
            )

        self.assertIsNotNone(result)
        state_dict, network_dim, new_alpha = result
        self.assertEqual(network_dim, 16)
        self.assertIsInstance(new_alpha, float)

    def test_linear_weights_still_work(self):
        """SDXL-style Linear (2D) LoRA weights should still work after the fix."""
        lora_sd = {
            "block.0.lora_down.weight": torch.randn(16, 2048),
            "block.0.lora_up.weight": torch.randn(2048, 16),
            "block.0.lora_down.alpha": torch.tensor(32.0),
        }

        with patch(
            "tools.resize_lora.merge_linear",
            return_value=torch.randn(2048, 2048),
        ), patch(
            "tools.resize_lora.rank_resize",
            return_value={
                "new_rank": 8,
                "new_alpha": 16.0,
                "sum_retained": 0.95,
                "fro_retained": 0.97,
                "max_ratio": 3.2,
            },
        ):
            result = resize_lora_model(
                lora_sd,
                new_rank=8,
                save_dtype=torch.float32,
                device="cpu",
                dynamic_method=None,
                dynamic_param=None,
                verbose=False,
            )

        self.assertIsNotNone(result)
        state_dict, network_dim, new_alpha = result
        self.assertEqual(network_dim, 16)
        self.assertIsInstance(new_alpha, float)


if __name__ == "__main__":
    unittest.main()