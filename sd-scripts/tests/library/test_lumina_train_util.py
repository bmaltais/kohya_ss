import pytest
import torch
import math

from library.lumina_train_util import (
    batchify,
    time_shift,
    get_lin_function,
    get_schedule,
    compute_density_for_timestep_sampling,
    get_sigmas,
    compute_loss_weighting_for_sd3,
    get_noisy_model_input_and_timesteps,
    apply_model_prediction_type,
    retrieve_timesteps,
)
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler


def test_batchify():
    # Test case with no batch size specified
    prompts = [{"prompt": "test1"}, {"prompt": "test2"}, {"prompt": "test3"}]
    batchified = list(batchify(prompts))
    assert len(batchified) == 1
    assert len(batchified[0]) == 3

    # Test case with batch size specified
    batchified_sized = list(batchify(prompts, batch_size=2))
    assert len(batchified_sized) == 2
    assert len(batchified_sized[0]) == 2
    assert len(batchified_sized[1]) == 1

    # Test batching with prompts having same parameters
    prompts_with_params = [
        {"prompt": "test1", "width": 512, "height": 512},
        {"prompt": "test2", "width": 512, "height": 512},
        {"prompt": "test3", "width": 1024, "height": 1024},
    ]
    batchified_params = list(batchify(prompts_with_params))
    assert len(batchified_params) == 2

    # Test invalid batch size
    with pytest.raises(ValueError):
        list(batchify(prompts, batch_size=0))
    with pytest.raises(ValueError):
        list(batchify(prompts, batch_size=-1))


def test_time_shift():
    # Test standard parameters
    t = torch.tensor([0.5])
    mu = 1.0
    sigma = 1.0
    result = time_shift(mu, sigma, t)
    assert 0 <= result <= 1

    # Test with edge cases
    t_edges = torch.tensor([0.0, 1.0])
    result_edges = time_shift(1.0, 1.0, t_edges)

    # Check that results are bounded within [0, 1]
    assert torch.all(result_edges >= 0)
    assert torch.all(result_edges <= 1)


def test_get_lin_function():
    # Default parameters
    func = get_lin_function()
    assert func(256) == 0.5
    assert func(4096) == 1.15

    # Custom parameters
    custom_func = get_lin_function(x1=100, x2=1000, y1=0.1, y2=0.9)
    assert custom_func(100) == 0.1
    assert custom_func(1000) == 0.9


def test_get_schedule():
    # Basic schedule
    schedule = get_schedule(num_steps=10, image_seq_len=256)
    assert len(schedule) == 10
    assert all(0 <= x <= 1 for x in schedule)

    # Test different sequence lengths
    short_schedule = get_schedule(num_steps=5, image_seq_len=128)
    long_schedule = get_schedule(num_steps=15, image_seq_len=1024)
    assert len(short_schedule) == 5
    assert len(long_schedule) == 15

    # Test with shift disabled
    unshifted_schedule = get_schedule(num_steps=10, image_seq_len=256, shift=False)
    assert torch.allclose(torch.tensor(unshifted_schedule), torch.linspace(1, 1 / 10, 10))


def test_compute_density_for_timestep_sampling():
    # Test uniform sampling
    uniform_samples = compute_density_for_timestep_sampling("uniform", batch_size=100)
    assert len(uniform_samples) == 100
    assert torch.all((uniform_samples >= 0) & (uniform_samples <= 1))

    # Test logit normal sampling
    logit_normal_samples = compute_density_for_timestep_sampling("logit_normal", batch_size=100, logit_mean=0.0, logit_std=1.0)
    assert len(logit_normal_samples) == 100
    assert torch.all((logit_normal_samples >= 0) & (logit_normal_samples <= 1))

    # Test mode sampling
    mode_samples = compute_density_for_timestep_sampling("mode", batch_size=100, mode_scale=0.5)
    assert len(mode_samples) == 100
    assert torch.all((mode_samples >= 0) & (mode_samples <= 1))


def test_get_sigmas():
    # Create a mock noise scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    device = torch.device("cpu")

    # Test with default parameters
    timesteps = torch.tensor([100, 500, 900])
    sigmas = get_sigmas(scheduler, timesteps, device)

    # Check shape and basic properties
    assert sigmas.shape[0] == 3
    assert torch.all(sigmas >= 0)

    # Test with different n_dim
    sigmas_4d = get_sigmas(scheduler, timesteps, device, n_dim=4)
    assert sigmas_4d.ndim == 4

    # Test with different dtype
    sigmas_float16 = get_sigmas(scheduler, timesteps, device, dtype=torch.float16)
    assert sigmas_float16.dtype == torch.float16


def test_compute_loss_weighting_for_sd3():
    # Prepare some mock sigmas
    sigmas = torch.tensor([0.1, 0.5, 1.0])

    # Test sigma_sqrt weighting
    sqrt_weighting = compute_loss_weighting_for_sd3("sigma_sqrt", sigmas)
    assert torch.allclose(sqrt_weighting, 1 / (sigmas**2), rtol=1e-5)

    # Test cosmap weighting
    cosmap_weighting = compute_loss_weighting_for_sd3("cosmap", sigmas)
    bot = 1 - 2 * sigmas + 2 * sigmas**2
    expected_cosmap = 2 / (math.pi * bot)
    assert torch.allclose(cosmap_weighting, expected_cosmap, rtol=1e-5)

    # Test default weighting
    default_weighting = compute_loss_weighting_for_sd3("unknown", sigmas)
    assert torch.all(default_weighting == 1)


def test_apply_model_prediction_type():
    # Create mock args and tensors
    class MockArgs:
        model_prediction_type = "raw"
        weighting_scheme = "sigma_sqrt"

    args = MockArgs()
    model_pred = torch.tensor([1.0, 2.0, 3.0])
    noisy_model_input = torch.tensor([0.5, 1.0, 1.5])
    sigmas = torch.tensor([0.1, 0.5, 1.0])

    # Test raw prediction type
    raw_pred, raw_weighting = apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)
    assert torch.all(raw_pred == model_pred)
    assert raw_weighting is None

    # Test additive prediction type
    args.model_prediction_type = "additive"
    additive_pred, _ = apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)
    assert torch.all(additive_pred == model_pred + noisy_model_input)

    # Test sigma scaled prediction type
    args.model_prediction_type = "sigma_scaled"
    sigma_scaled_pred, sigma_weighting = apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)
    assert torch.all(sigma_scaled_pred == model_pred * (-sigmas) + noisy_model_input)
    assert sigma_weighting is not None


def test_retrieve_timesteps():
    # Create a mock scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    # Test with num_inference_steps
    timesteps, n_steps = retrieve_timesteps(scheduler, num_inference_steps=50)
    assert len(timesteps) == 50
    assert n_steps == 50

    # Test error handling with simultaneous timesteps and sigmas
    with pytest.raises(ValueError):
        retrieve_timesteps(scheduler, timesteps=[1, 2, 3], sigmas=[0.1, 0.2, 0.3])


def test_get_noisy_model_input_and_timesteps():
    # Create a mock args and setup
    class MockArgs:
        timestep_sampling = "uniform"
        weighting_scheme = "sigma_sqrt"
        sigmoid_scale = 1.0
        discrete_flow_shift = 6.0
        ip_noise_gamma = True
        ip_noise_gamma_random_strength = 0.01 

    args = MockArgs()
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    device = torch.device("cpu")

    # Prepare mock latents and noise
    latents = torch.randn(4, 16, 64, 64)
    noise = torch.randn_like(latents)

    # Test uniform sampling
    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, scheduler, latents, noise, device, torch.float32)

    # Validate output shapes and types
    assert noisy_input.shape == latents.shape
    assert timesteps.shape[0] == latents.shape[0]
    assert noisy_input.dtype == torch.float32
    assert timesteps.dtype == torch.float32

    # Test different sampling methods
    sampling_methods = ["sigmoid", "shift", "nextdit_shift"]
    for method in sampling_methods:
        args.timestep_sampling = method
        noisy_input, timesteps, _ = get_noisy_model_input_and_timesteps(args, scheduler, latents, noise, device, torch.float32)
        assert noisy_input.shape == latents.shape
        assert timesteps.shape[0] == latents.shape[0]
