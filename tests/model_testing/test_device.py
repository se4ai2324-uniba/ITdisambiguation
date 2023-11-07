import pytest
from src.models.train import train_model
import torch

def test_device():
    # Modify EPOCHS for testing to be just 1 for faster tests
    num_epochs = 1

    # Assert training on CPU
    loss_history_cpu,_ = train_model(num_epochs=num_epochs,dev='cpu')
    assert loss_history_cpu

    # If CUDA is available, assert training on CUDA
    if torch.cuda.is_available():
        loss_history_cuda,_ = train_model(num_epochs=num_epochs,dev='cuda')
        assert loss_history_cuda

if __name__ == "__main__":
    pytest.main()
