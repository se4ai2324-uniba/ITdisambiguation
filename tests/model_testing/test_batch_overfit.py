import pytest
import torch
from src.models.train import test_overfit_single_batch


def test_overfit_batch():
    # Set the number of epochs high enough to allow overfitting
    num_epochs = 3
    
    # Use the test function to train on a single batch and get the loss history
    loss_history = test_overfit_single_batch(num_epochs=num_epochs)
    
    final_loss = loss_history[-1]
    assert final_loss < 1e-4, "The model did not overfit as expected."


if __name__ == "__main__":
    pytest.main()

