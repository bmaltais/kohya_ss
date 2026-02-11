import pytest
import torch
from unittest.mock import MagicMock, patch
from library.flux_train_utils import (
    get_noisy_model_input_and_timesteps,
)

# Mock classes and functions
class MockNoiseScheduler:
    def __init__(self, num_train_timesteps=1000):
        self.config = MagicMock()
        self.config.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.arange(num_train_timesteps, dtype=torch.long)


# Create fixtures for commonly used objects
@pytest.fixture
def args():
    args = MagicMock()
    args.timestep_sampling = "uniform"
    args.weighting_scheme = "uniform"
    args.logit_mean = 0.0
    args.logit_std = 1.0
    args.mode_scale = 1.0
    args.sigmoid_scale = 1.0
    args.discrete_flow_shift = 3.1582
    args.ip_noise_gamma = None
    args.ip_noise_gamma_random_strength = False
    return args


@pytest.fixture
def noise_scheduler():
    return MockNoiseScheduler(num_train_timesteps=1000)


@pytest.fixture
def latents():
    return torch.randn(2, 4, 8, 8)


@pytest.fixture
def noise():
    return torch.randn(2, 4, 8, 8)


@pytest.fixture
def device():
    # return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


# Mock the required functions
@pytest.fixture(autouse=True)
def mock_functions():
    with (
        patch("torch.sigmoid", side_effect=torch.sigmoid),
        patch("torch.rand", side_effect=torch.rand),
        patch("torch.randn", side_effect=torch.randn),
    ):
        yield


# Test different timestep sampling methods
def test_uniform_sampling(args, noise_scheduler, latents, noise, device):
    args.timestep_sampling = "uniform"
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)
    assert noisy_input.dtype == dtype
    assert timesteps.dtype == dtype


def test_sigmoid_sampling(args, noise_scheduler, latents, noise, device):
    args.timestep_sampling = "sigmoid"
    args.sigmoid_scale = 1.0
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)


def test_shift_sampling(args, noise_scheduler, latents, noise, device):
    args.timestep_sampling = "shift"
    args.sigmoid_scale = 1.0
    args.discrete_flow_shift = 3.1582
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)


def test_flux_shift_sampling(args, noise_scheduler, latents, noise, device):
    args.timestep_sampling = "flux_shift"
    args.sigmoid_scale = 1.0
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)


def test_weighting_scheme(args, noise_scheduler, latents, noise, device):
    # Mock the necessary functions for this specific test
    with patch("library.flux_train_utils.compute_density_for_timestep_sampling", 
               return_value=torch.tensor([0.3, 0.7], device=device)), \
         patch("library.flux_train_utils.get_sigmas", 
               return_value=torch.tensor([[0.3], [0.7]], device=device).view(-1, 1, 1, 1)):
               
        args.timestep_sampling = "other"  # Will trigger the weighting scheme path
        args.weighting_scheme = "uniform"
        args.logit_mean = 0.0
        args.logit_std = 1.0
        args.mode_scale = 1.0
        dtype = torch.float32
        
        noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype
        )
        
        assert noisy_input.shape == latents.shape
        assert timesteps.shape == (latents.shape[0],)
        assert sigmas.shape == (latents.shape[0], 1, 1, 1)


# Test IP noise options
def test_with_ip_noise(args, noise_scheduler, latents, noise, device):
    args.ip_noise_gamma = 0.5
    args.ip_noise_gamma_random_strength = False
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)


def test_with_random_ip_noise(args, noise_scheduler, latents, noise, device):
    args.ip_noise_gamma = 0.1
    args.ip_noise_gamma_random_strength = True
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)


# Test different data types
def test_float16_dtype(args, noise_scheduler, latents, noise, device):
    dtype = torch.float16

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.dtype == dtype
    assert timesteps.dtype == dtype


# Test different batch sizes
def test_different_batch_size(args, noise_scheduler, device):
    latents = torch.randn(5, 4, 8, 8)  # batch size of 5
    noise = torch.randn(5, 4, 8, 8)
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (5,)
    assert sigmas.shape == (5, 1, 1, 1)


# Test different image sizes
def test_different_image_size(args, noise_scheduler, device):
    latents = torch.randn(2, 4, 16, 16)  # larger image size
    noise = torch.randn(2, 4, 16, 16)
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (2,)
    assert sigmas.shape == (2, 1, 1, 1)


# Test edge cases
def test_zero_batch_size(args, noise_scheduler, device):
    with pytest.raises(AssertionError):  # expecting an error with zero batch size
        latents = torch.randn(0, 4, 8, 8)
        noise = torch.randn(0, 4, 8, 8)
        dtype = torch.float32

        get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)


def test_different_timestep_count(args, device):
    noise_scheduler = MockNoiseScheduler(num_train_timesteps=500)  # different timestep count
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn(2, 4, 8, 8)
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (2,)
    # Check that timesteps are within the proper range
    assert torch.all(timesteps < 500)
