import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from library.custom_offloading_utils import (
    _synchronize_device, 
    swap_weight_devices_cuda,
    swap_weight_devices_no_cuda,
    weighs_to_device,
    Offloader,
    ModelOffloader
)

class TransformerBlock(nn.Module):
    def __init__(self, block_idx: int):
        super().__init__()
        self.block_idx = block_idx
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 10)
        self.seq = nn.Sequential(nn.SiLU(), nn.Linear(10, 10))
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.seq(x)
        return x


class SimpleModel(nn.Module):
    def __init__(self, num_blocks=16):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(i)
        for i in range(num_blocks)])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device


# Device Synchronization Tests
@patch('torch.cuda.synchronize')
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_synchronize(mock_cuda_sync):
    device = torch.device('cuda')
    _synchronize_device(device)
    mock_cuda_sync.assert_called_once()

@patch('torch.xpu.synchronize')
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
def test_xpu_synchronize(mock_xpu_sync):
    device = torch.device('xpu')
    _synchronize_device(device)
    mock_xpu_sync.assert_called_once()

@patch('torch.mps.synchronize')
@pytest.mark.skipif(not torch.xpu.is_available(), reason="MPS not available")
def test_mps_synchronize(mock_mps_sync):
    device = torch.device('mps')
    _synchronize_device(device)
    mock_mps_sync.assert_called_once()


# Weights to Device Tests
def test_weights_to_device():
    # Create a simple model with weights
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # Start with CPU tensors
    device = torch.device('cpu')
    for module in model.modules():
        if hasattr(module, "weight") and module.weight is not None:
            assert module.weight.device == device
    
    # Move to mock CUDA device
    mock_device = torch.device('cuda')
    with patch('torch.Tensor.to', return_value=torch.zeros(1).to(device)):
        weighs_to_device(model, mock_device)
        
        # Since we mocked the to() function, we can only verify modules were processed
        # but can't check actual device movement


# Swap Weight Devices Tests
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_swap_weight_devices_cuda():
    device = torch.device('cuda')
    layer_to_cpu = SimpleModel()
    layer_to_cuda = SimpleModel()

    # Move layer to CUDA to move to CPU
    layer_to_cpu.to(device)
    
    with patch('torch.Tensor.to', return_value=torch.zeros(1)):
        with patch('torch.Tensor.copy_'):
            swap_weight_devices_cuda(device, layer_to_cpu, layer_to_cuda)
            
            assert layer_to_cpu.device.type == 'cpu'
            assert layer_to_cuda.device.type == 'cuda'



@patch('library.custom_offloading_utils._synchronize_device')
def test_swap_weight_devices_no_cuda(mock_sync_device):
    device = torch.device('cpu')
    layer_to_cpu = SimpleModel()
    layer_to_cuda = SimpleModel()
    
    with patch('torch.Tensor.to', return_value=torch.zeros(1)):
        with patch('torch.Tensor.copy_'):
            swap_weight_devices_no_cuda(device, layer_to_cpu, layer_to_cuda)
            
            # Verify _synchronize_device was called twice
            assert mock_sync_device.call_count == 2


# Offloader Tests
@pytest.fixture
def offloader():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return Offloader(
        num_blocks=4,
        blocks_to_swap=2,
        device=device,
        debug=False
    )


def test_offloader_init(offloader):
    assert offloader.num_blocks == 4
    assert offloader.blocks_to_swap == 2
    assert hasattr(offloader, 'thread_pool')
    assert offloader.futures == {}
    assert offloader.cuda_available == (offloader.device.type == 'cuda')


@patch('library.custom_offloading_utils.swap_weight_devices_cuda')
@patch('library.custom_offloading_utils.swap_weight_devices_no_cuda')
def test_swap_weight_devices(mock_no_cuda, mock_cuda, offloader: Offloader):
    block_to_cpu = SimpleModel()
    block_to_cuda = SimpleModel()
    
    # Force test for CUDA device
    offloader.cuda_available = True
    offloader.swap_weight_devices(block_to_cpu, block_to_cuda)
    mock_cuda.assert_called_once_with(offloader.device, block_to_cpu, block_to_cuda)
    mock_no_cuda.assert_not_called()
    
    # Reset mocks
    mock_cuda.reset_mock()
    mock_no_cuda.reset_mock()
    
    # Force test for non-CUDA device
    offloader.cuda_available = False
    offloader.swap_weight_devices(block_to_cpu, block_to_cuda)
    mock_no_cuda.assert_called_once_with(offloader.device, block_to_cpu, block_to_cuda)
    mock_cuda.assert_not_called()


@patch('library.custom_offloading_utils.Offloader.swap_weight_devices')
def test_submit_move_blocks(mock_swap, offloader):
    blocks = [SimpleModel() for _ in range(4)]
    block_idx_to_cpu = 0
    block_idx_to_cuda = 2
    
    # Mock the thread pool to execute synchronously
    future = MagicMock()
    future.result.return_value = (block_idx_to_cpu, block_idx_to_cuda)
    offloader.thread_pool.submit = MagicMock(return_value=future)
    
    offloader._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
    
    # Check that the future is stored with the correct key
    assert block_idx_to_cuda in offloader.futures


def test_wait_blocks_move(offloader):
    block_idx = 2
    
    # Test with no future for the block
    offloader._wait_blocks_move(block_idx)  # Should not raise
    
    # Create a fake future and test waiting
    future = MagicMock()
    future.result.return_value = (0, block_idx)
    offloader.futures[block_idx] = future
    
    offloader._wait_blocks_move(block_idx)
    
    # Check that the future was removed
    assert block_idx not in offloader.futures
    future.result.assert_called_once()


# ModelOffloader Tests
@pytest.fixture
def model_offloader():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    blocks_to_swap = 2
    blocks = SimpleModel(4).blocks
    return ModelOffloader(
        blocks=blocks,
        blocks_to_swap=blocks_to_swap,
        device=device,
        debug=False
    )


def test_model_offloader_init(model_offloader):
    assert model_offloader.num_blocks == 4
    assert model_offloader.blocks_to_swap == 2
    assert hasattr(model_offloader, 'thread_pool')
    assert model_offloader.futures == {}
    assert len(model_offloader.remove_handles) > 0  # Should have registered hooks


def test_create_backward_hook():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    blocks_to_swap = 2
    blocks = SimpleModel(4).blocks
    model_offloader = ModelOffloader(
        blocks=blocks,
        blocks_to_swap=blocks_to_swap,
        device=device,
        debug=False
    )

    # Test hook creation for swapping case (block 0)
    hook_swap = model_offloader.create_backward_hook(blocks, 0)
    assert hook_swap is None
    
    # Test hook creation for waiting case (block 1)
    hook_wait = model_offloader.create_backward_hook(blocks, 1)
    assert hook_wait is not None
    
    # Test hook creation for no action case (block 3)
    hook_none = model_offloader.create_backward_hook(blocks, 3)
    assert hook_none is None


@patch('library.custom_offloading_utils.ModelOffloader._submit_move_blocks')
@patch('library.custom_offloading_utils.ModelOffloader._wait_blocks_move')
def test_backward_hook_execution(mock_wait, mock_submit):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    blocks_to_swap = 2
    model = SimpleModel(4)
    blocks = model.blocks
    model_offloader = ModelOffloader(
        blocks=blocks,
        blocks_to_swap=blocks_to_swap,
        device=device,
        debug=False
    )
    
    # Test swapping hook (block 1)
    hook_swap = model_offloader.create_backward_hook(blocks, 1)
    assert hook_swap is not None
    hook_swap(model, torch.zeros(1), torch.zeros(1))
    mock_submit.assert_called_once()
    
    mock_submit.reset_mock()
    
    # Test waiting hook (block 2)
    hook_wait = model_offloader.create_backward_hook(blocks, 2)
    assert hook_wait is not None
    hook_wait(model, torch.zeros(1), torch.zeros(1))
    assert mock_wait.call_count == 2


@patch('library.custom_offloading_utils.weighs_to_device')
@patch('library.custom_offloading_utils._synchronize_device')
@patch('library.custom_offloading_utils._clean_memory_on_device')
def test_prepare_block_devices_before_forward(mock_clean, mock_sync, mock_weights_to_device, model_offloader):
    model = SimpleModel(4)
    blocks = model.blocks
    
    with patch.object(nn.Module, 'to'):
        model_offloader.prepare_block_devices_before_forward(blocks)
        
        # Check that weighs_to_device was called for each block
        assert mock_weights_to_device.call_count == 4
        
        # Check that _synchronize_device and _clean_memory_on_device were called
        mock_sync.assert_called_once_with(model_offloader.device)
        mock_clean.assert_called_once_with(model_offloader.device)


@patch('library.custom_offloading_utils.ModelOffloader._wait_blocks_move')
def test_wait_for_block(mock_wait, model_offloader):
    # Test with blocks_to_swap=0
    model_offloader.blocks_to_swap = 0
    model_offloader.wait_for_block(1)
    mock_wait.assert_not_called()
    
    # Test with blocks_to_swap=2
    model_offloader.blocks_to_swap = 2
    block_idx = 1
    model_offloader.wait_for_block(block_idx)
    mock_wait.assert_called_once_with(block_idx)


@patch('library.custom_offloading_utils.ModelOffloader._submit_move_blocks')
def test_submit_move_blocks(mock_submit, model_offloader):
    model = SimpleModel()
    blocks = model.blocks
    
    # Test with blocks_to_swap=0
    model_offloader.blocks_to_swap = 0
    model_offloader.submit_move_blocks(blocks, 1)
    mock_submit.assert_not_called()
    
    mock_submit.reset_mock()
    model_offloader.blocks_to_swap = 2
    
    # Test within swap range
    block_idx = 1
    model_offloader.submit_move_blocks(blocks, block_idx)
    mock_submit.assert_called_once()
    
    mock_submit.reset_mock()
    
    # Test outside swap range
    block_idx = 3
    model_offloader.submit_move_blocks(blocks, block_idx)
    mock_submit.assert_not_called()


# Integration test for offloading in a realistic scenario
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_offloading_integration():
    device = torch.device('cuda')
    # Create a mini model with 4 blocks
    model = SimpleModel(5)
    model.to(device)
    blocks = model.blocks
    
    # Initialize model offloader
    offloader = ModelOffloader(
        blocks=blocks,
        blocks_to_swap=2,
        device=device,
        debug=True
    )
    
    # Prepare blocks for forward pass
    offloader.prepare_block_devices_before_forward(blocks)
    
    # Simulate forward pass with offloading
    input_tensor = torch.randn(1, 10, device=device)
    x = input_tensor
    
    for i, block in enumerate(blocks):
        # Wait for the current block to be ready
        offloader.wait_for_block(i)
        
        # Process through the block
        x = block(x)
        
        # Schedule moving weights for future blocks
        offloader.submit_move_blocks(blocks, i)
    
    # Verify we get a valid output
    assert x.shape == (1, 10)
    assert not torch.isnan(x).any()


# Error handling tests
def test_offloader_assertion_error():
    with pytest.raises(AssertionError):
        device = torch.device('cpu')
        layer_to_cpu = SimpleModel()
        layer_to_cuda = nn.Linear(10, 5)  # Different class
        swap_weight_devices_cuda(device, layer_to_cpu, layer_to_cuda)

if __name__ == "__main__":
    # Run all tests when file is executed directly
    import sys
    
    # Configure pytest command line arguments
    pytest_args = [
        "-v",                   # Verbose output
        "--color=yes",          # Colored output
        __file__,               # Run tests in this file
    ]
    
    # Add optional arguments from command line
    if len(sys.argv) > 1:
        pytest_args.extend(sys.argv[1:])
    
    # Print info about test execution
    print(f"Running tests with PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run the tests
    sys.exit(pytest.main(pytest_args))
