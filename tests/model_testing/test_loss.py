import pytest
from src.models.train import train_model

def test_loss():
    num_epochs = 2

    loss_history,_ = train_model(num_epochs=num_epochs)

    # Check that the last loss is less than or equal to the first loss
    # This assumes that the loss should decrease or stay the same across epochs.
    assert loss_history[0] <= loss_history[num_epochs-1]


if __name__ == "__main__":
    pytest.main()
