import pytest
import torch

from library.lumina_models import (
    LuminaParams,
    to_cuda,
    to_cpu,
    RopeEmbedder,
    TimestepEmbedder,
    modulate,
    NextDiT,
)

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_lumina_params():
    # Test default configuration
    default_params = LuminaParams()
    assert default_params.patch_size == 2
    assert default_params.in_channels == 4
    assert default_params.axes_dims == [36, 36, 36]
    assert default_params.axes_lens == [300, 512, 512]

    # Test 2B config
    config_2b = LuminaParams.get_2b_config()
    assert config_2b.dim == 2304
    assert config_2b.in_channels == 16
    assert config_2b.n_layers == 26
    assert config_2b.n_heads == 24
    assert config_2b.cap_feat_dim == 2304

    # Test 7B config
    config_7b = LuminaParams.get_7b_config()
    assert config_7b.dim == 4096
    assert config_7b.n_layers == 32
    assert config_7b.n_heads == 32
    assert config_7b.axes_dims == [64, 64, 64]


@cuda_required
def test_to_cuda_to_cpu():
    # Test tensor conversion
    x = torch.tensor([1, 2, 3])
    x_cuda = to_cuda(x)
    x_cpu = to_cpu(x_cuda)
    assert x.cpu().tolist() == x_cpu.tolist()

    # Test list conversion
    list_data = [torch.tensor([1]), torch.tensor([2])]
    list_cuda = to_cuda(list_data)
    assert all(tensor.device.type == "cuda" for tensor in list_cuda)

    list_cpu = to_cpu(list_cuda)
    assert all(not tensor.device.type == "cuda" for tensor in list_cpu)

    # Test dict conversion
    dict_data = {"a": torch.tensor([1]), "b": torch.tensor([2])}
    dict_cuda = to_cuda(dict_data)
    assert all(tensor.device.type == "cuda" for tensor in dict_cuda.values())

    dict_cpu = to_cpu(dict_cuda)
    assert all(not tensor.device.type == "cuda" for tensor in dict_cpu.values())


def test_timestep_embedder():
    # Test initialization
    hidden_size = 256
    freq_emb_size = 128
    embedder = TimestepEmbedder(hidden_size, freq_emb_size)
    assert embedder.frequency_embedding_size == freq_emb_size

    # Test timestep embedding
    t = torch.tensor([0.5, 1.0, 2.0])
    emb_dim = freq_emb_size
    embeddings = TimestepEmbedder.timestep_embedding(t, emb_dim)

    assert embeddings.shape == (3, emb_dim)
    assert embeddings.dtype == torch.float32

    # Ensure embeddings are unique for different input times
    assert not torch.allclose(embeddings[0], embeddings[1])

    # Test forward pass
    t_emb = embedder(t)
    assert t_emb.shape == (3, hidden_size)


def test_rope_embedder_simple():
    rope_embedder = RopeEmbedder()
    batch_size, seq_len = 2, 10

    # Create position_ids with valid ranges for each axis
    position_ids = torch.stack(
        [
            torch.zeros(batch_size, seq_len, dtype=torch.int64),  # First axis: only 0 is valid
            torch.randint(0, 512, (batch_size, seq_len), dtype=torch.int64),  # Second axis: 0-511
            torch.randint(0, 512, (batch_size, seq_len), dtype=torch.int64),  # Third axis: 0-511
        ],
        dim=-1,
    )

    freqs_cis = rope_embedder(position_ids)
    # RoPE embeddings work in pairs, so output dimension is half of total axes_dims
    expected_dim = sum(rope_embedder.axes_dims) // 2  # 128 // 2 = 64
    assert freqs_cis.shape == (batch_size, seq_len, expected_dim)


def test_modulate():
    # Test modulation with different scales
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scale = torch.tensor([1.5, 2.0])

    modulated_x = modulate(x, scale)

    # Check that modulation scales correctly
    # The function does x * (1 + scale), so:
    # For scale [1.5, 2.0], (1 + scale) = [2.5, 3.0]
    expected_x = torch.tensor([[2.5 * 1.0, 2.5 * 2.0], [3.0 * 3.0, 3.0 * 4.0]])
    # Which equals: [[2.5, 5.0], [9.0, 12.0]]

    assert torch.allclose(modulated_x, expected_x)


def test_nextdit_parameter_count_optimized():
    # The constraint is: (dim // n_heads) == sum(axes_dims)
    # So for dim=120, n_heads=4: 120//4 = 30, so sum(axes_dims) must = 30
    model_small = NextDiT(
        patch_size=2,
        in_channels=4,  # Smaller
        dim=120,  # 120 // 4 = 30
        n_layers=2,  # Much fewer layers
        n_heads=4,  # Fewer heads
        n_kv_heads=2,
        axes_dims=[10, 10, 10],  # sum = 30
        axes_lens=[10, 32, 32],  # Smaller
    )
    param_count_small = model_small.parameter_count()
    assert param_count_small > 0

    # For dim=192, n_heads=6: 192//6 = 32, so sum(axes_dims) must = 32
    model_medium = NextDiT(
        patch_size=2,
        in_channels=4,
        dim=192,  # 192 // 6 = 32
        n_layers=4,  # More layers
        n_heads=6,
        n_kv_heads=3,
        axes_dims=[10, 11, 11],  # sum = 32
        axes_lens=[10, 32, 32],
    )
    param_count_medium = model_medium.parameter_count()
    assert param_count_medium > param_count_small
    print(f"Small model: {param_count_small:,} parameters")
    print(f"Medium model: {param_count_medium:,} parameters")


@torch.no_grad()
def test_precompute_freqs_cis():
    # Test precompute_freqs_cis
    dim = [16, 56, 56]
    end = [1, 512, 512]
    theta = 10000.0

    freqs_cis = NextDiT.precompute_freqs_cis(dim, end, theta)

    # Check number of frequency tensors
    assert len(freqs_cis) == len(dim)

    # Check each frequency tensor
    for i, (d, e) in enumerate(zip(dim, end)):
        assert freqs_cis[i].shape == (e, d // 2)
        assert freqs_cis[i].dtype == torch.complex128


@torch.no_grad()
def test_nextdit_patchify_and_embed():
    """Test the patchify_and_embed method which is crucial for training"""
    # Create a small NextDiT model for testing
    # The constraint is: (dim // n_heads) == sum(axes_dims)
    # For dim=120, n_heads=4: 120//4 = 30, so sum(axes_dims) must = 30
    model = NextDiT(
        patch_size=2,
        in_channels=4,
        dim=120,  # 120 // 4 = 30
        n_layers=1,  # Minimal layers for faster testing
        n_refiner_layers=1,  # Minimal refiner layers
        n_heads=4,
        n_kv_heads=2,
        axes_dims=[10, 10, 10],  # sum = 30
        axes_lens=[10, 32, 32],
        cap_feat_dim=120,  # Match dim for consistency
    )

    # Prepare test inputs
    batch_size = 2
    height, width = 64, 64  # Must be divisible by patch_size (2)
    caption_seq_len = 8

    # Create mock inputs
    x = torch.randn(batch_size, 4, height, width)  # Image latents
    cap_feats = torch.randn(batch_size, caption_seq_len, 120)  # Caption features
    cap_mask = torch.ones(batch_size, caption_seq_len, dtype=torch.bool)  # All valid tokens
    # Make second batch have shorter caption
    cap_mask[1, 6:] = False  # Only first 6 tokens are valid for second batch
    t = torch.randn(batch_size, 120)  # Timestep embeddings

    # Call patchify_and_embed
    joint_hidden_states, attention_mask, freqs_cis, l_effective_cap_len, seq_lengths = model.patchify_and_embed(
        x, cap_feats, cap_mask, t
    )

    # Validate outputs
    image_seq_len = (height // 2) * (width // 2)  # patch_size = 2
    expected_seq_lengths = [caption_seq_len + image_seq_len, 6 + image_seq_len]  # Second batch has shorter caption
    max_seq_len = max(expected_seq_lengths)

    # Check joint hidden states shape
    assert joint_hidden_states.shape == (batch_size, max_seq_len, 120)
    assert joint_hidden_states.dtype == torch.float32

    # Check attention mask shape and values
    assert attention_mask.shape == (batch_size, max_seq_len)
    assert attention_mask.dtype == torch.bool
    # First batch should have all positions valid up to its sequence length
    assert torch.all(attention_mask[0, : expected_seq_lengths[0]])
    assert torch.all(~attention_mask[0, expected_seq_lengths[0] :])
    # Second batch should have all positions valid up to its sequence length
    assert torch.all(attention_mask[1, : expected_seq_lengths[1]])
    assert torch.all(~attention_mask[1, expected_seq_lengths[1] :])

    # Check freqs_cis shape
    assert freqs_cis.shape == (batch_size, max_seq_len, sum(model.axes_dims) // 2)

    # Check effective caption lengths
    assert l_effective_cap_len == [caption_seq_len, 6]

    # Check sequence lengths
    assert seq_lengths == expected_seq_lengths

    # Validate that the joint hidden states contain non-zero values where attention mask is True
    for i in range(batch_size):
        valid_positions = attention_mask[i]
        # Check that valid positions have meaningful data (not all zeros)
        valid_data = joint_hidden_states[i][valid_positions]
        assert not torch.allclose(valid_data, torch.zeros_like(valid_data))

        # Check that invalid positions are zeros
        if valid_positions.sum() < max_seq_len:
            invalid_data = joint_hidden_states[i][~valid_positions]
            assert torch.allclose(invalid_data, torch.zeros_like(invalid_data))


@torch.no_grad()
def test_nextdit_patchify_and_embed_edge_cases():
    """Test edge cases for patchify_and_embed"""
    # Create minimal model
    model = NextDiT(
        patch_size=2,
        in_channels=4,
        dim=60,  # 60 // 3 = 20
        n_layers=1,
        n_refiner_layers=1,
        n_heads=3,
        n_kv_heads=1,
        axes_dims=[8, 6, 6],  # sum = 20
        axes_lens=[10, 16, 16],
        cap_feat_dim=60,
    )

    # Test with empty captions (all masked)
    batch_size = 1
    height, width = 32, 32
    caption_seq_len = 4

    x = torch.randn(batch_size, 4, height, width)
    cap_feats = torch.randn(batch_size, caption_seq_len, 60)
    cap_mask = torch.zeros(batch_size, caption_seq_len, dtype=torch.bool)  # All tokens masked
    t = torch.randn(batch_size, 60)

    joint_hidden_states, attention_mask, freqs_cis, l_effective_cap_len, seq_lengths = model.patchify_and_embed(
        x, cap_feats, cap_mask, t
    )

    # With all captions masked, effective length should be 0
    assert l_effective_cap_len == [0]

    # Sequence length should just be the image sequence length
    image_seq_len = (height // 2) * (width // 2)
    assert seq_lengths == [image_seq_len]

    # Joint hidden states should only contain image data
    assert joint_hidden_states.shape == (batch_size, image_seq_len, 60)
    assert attention_mask.shape == (batch_size, image_seq_len)
    assert torch.all(attention_mask[0])  # All image positions should be valid
