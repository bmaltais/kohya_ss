import pytest
import torch
from unittest.mock import MagicMock, patch
import argparse

from library import lumina_models, lumina_util
from lumina_train_network import LuminaNetworkTrainer


@pytest.fixture
def lumina_trainer():
    return LuminaNetworkTrainer()


@pytest.fixture
def mock_args():
    args = MagicMock()
    args.pretrained_model_name_or_path = "test_path"
    args.disable_mmap_load_safetensors = False
    args.use_flash_attn = False
    args.use_sage_attn = False
    args.fp8_base = False
    args.blocks_to_swap = None
    args.gemma2 = "test_gemma2_path"
    args.ae = "test_ae_path"
    args.cache_text_encoder_outputs = True
    args.cache_text_encoder_outputs_to_disk = False
    args.network_train_unet_only = False
    return args


@pytest.fixture
def mock_accelerator():
    accelerator = MagicMock()
    accelerator.device = torch.device("cpu")
    accelerator.prepare.side_effect = lambda x, **kwargs: x
    accelerator.unwrap_model.side_effect = lambda x: x
    return accelerator


def test_assert_extra_args(lumina_trainer, mock_args):
    train_dataset_group = MagicMock()
    train_dataset_group.verify_bucket_reso_steps = MagicMock()
    val_dataset_group = MagicMock()
    val_dataset_group.verify_bucket_reso_steps = MagicMock()

    # Test with default settings
    lumina_trainer.assert_extra_args(mock_args, train_dataset_group, val_dataset_group)

    # Verify verify_bucket_reso_steps was called for both groups
    assert train_dataset_group.verify_bucket_reso_steps.call_count > 0
    assert val_dataset_group.verify_bucket_reso_steps.call_count > 0

    # Check text encoder output caching
    assert lumina_trainer.train_gemma2 is (not mock_args.network_train_unet_only)
    assert mock_args.cache_text_encoder_outputs is True


def test_load_target_model(lumina_trainer, mock_args, mock_accelerator):
    # Patch lumina_util methods
    with (
        patch("library.lumina_util.load_lumina_model") as mock_load_lumina_model,
        patch("library.lumina_util.load_gemma2") as mock_load_gemma2,
        patch("library.lumina_util.load_ae") as mock_load_ae,
    ):
        # Create mock models
        mock_model = MagicMock(spec=lumina_models.NextDiT)
        mock_model.dtype = torch.float32
        mock_gemma2 = MagicMock()
        mock_ae = MagicMock()

        mock_load_lumina_model.return_value = mock_model
        mock_load_gemma2.return_value = mock_gemma2
        mock_load_ae.return_value = mock_ae

        # Test load_target_model
        version, gemma2_list, ae, model = lumina_trainer.load_target_model(mock_args, torch.float32, mock_accelerator)

        # Verify calls and return values
        assert version == lumina_util.MODEL_VERSION_LUMINA_V2
        assert gemma2_list == [mock_gemma2]
        assert ae == mock_ae
        assert model == mock_model

        # Verify load calls
        mock_load_lumina_model.assert_called_once()
        mock_load_gemma2.assert_called_once()
        mock_load_ae.assert_called_once()


def test_get_strategies(lumina_trainer, mock_args):
    # Test tokenize strategy
    try:
        tokenize_strategy = lumina_trainer.get_tokenize_strategy(mock_args)
        assert tokenize_strategy.__class__.__name__ == "LuminaTokenizeStrategy"
    except OSError as e:
        # If the tokenizer is not found (due to gated repo), we can skip the test
        print(f"Skipping LuminaTokenizeStrategy test due to OSError: {e}")

    # Test latents caching strategy
    latents_strategy = lumina_trainer.get_latents_caching_strategy(mock_args)
    assert latents_strategy.__class__.__name__ == "LuminaLatentsCachingStrategy"

    # Test text encoding strategy
    text_encoding_strategy = lumina_trainer.get_text_encoding_strategy(mock_args)
    assert text_encoding_strategy.__class__.__name__ == "LuminaTextEncodingStrategy"


def test_text_encoder_output_caching_strategy(lumina_trainer, mock_args):
    # Call assert_extra_args to set train_gemma2
    train_dataset_group = MagicMock()
    train_dataset_group.verify_bucket_reso_steps = MagicMock()
    val_dataset_group = MagicMock()
    val_dataset_group.verify_bucket_reso_steps = MagicMock()
    lumina_trainer.assert_extra_args(mock_args, train_dataset_group, val_dataset_group)

    # With text encoder caching enabled
    mock_args.skip_cache_check = False
    mock_args.text_encoder_batch_size = 16
    strategy = lumina_trainer.get_text_encoder_outputs_caching_strategy(mock_args)

    assert strategy.__class__.__name__ == "LuminaTextEncoderOutputsCachingStrategy"
    assert strategy.cache_to_disk is False  # based on mock_args

    # With text encoder caching disabled
    mock_args.cache_text_encoder_outputs = False
    strategy = lumina_trainer.get_text_encoder_outputs_caching_strategy(mock_args)
    assert strategy is None


def test_noise_scheduler(lumina_trainer, mock_args):
    device = torch.device("cpu")
    noise_scheduler = lumina_trainer.get_noise_scheduler(mock_args, device)

    assert noise_scheduler.__class__.__name__ == "FlowMatchEulerDiscreteScheduler"
    assert noise_scheduler.num_train_timesteps == 1000
    assert hasattr(lumina_trainer, "noise_scheduler_copy")


def test_sai_model_spec(lumina_trainer, mock_args):
    with patch("library.train_util.get_sai_model_spec") as mock_get_spec:
        mock_get_spec.return_value = "test_spec"
        spec = lumina_trainer.get_sai_model_spec(mock_args)
        assert spec == "test_spec"
        mock_get_spec.assert_called_once_with(None, mock_args, False, True, False, lumina="lumina2")


def test_update_metadata(lumina_trainer, mock_args):
    metadata = {}
    lumina_trainer.update_metadata(metadata, mock_args)

    assert "ss_weighting_scheme" in metadata
    assert "ss_logit_mean" in metadata
    assert "ss_logit_std" in metadata
    assert "ss_mode_scale" in metadata
    assert "ss_timestep_sampling" in metadata
    assert "ss_sigmoid_scale" in metadata
    assert "ss_model_prediction_type" in metadata
    assert "ss_discrete_flow_shift" in metadata


def test_is_text_encoder_not_needed_for_training(lumina_trainer, mock_args):
    # Test with text encoder output caching, but not training text encoder
    mock_args.cache_text_encoder_outputs = True
    with patch.object(lumina_trainer, "is_train_text_encoder", return_value=False):
        result = lumina_trainer.is_text_encoder_not_needed_for_training(mock_args)
        assert result is True

    # Test with text encoder output caching and training text encoder
    with patch.object(lumina_trainer, "is_train_text_encoder", return_value=True):
        result = lumina_trainer.is_text_encoder_not_needed_for_training(mock_args)
        assert result is False

    # Test with no text encoder output caching
    mock_args.cache_text_encoder_outputs = False
    result = lumina_trainer.is_text_encoder_not_needed_for_training(mock_args)
    assert result is False
