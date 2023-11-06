import os
import pytest
from src.models.train import train_model
from src.models.conf import config

def test_training_completion():
    num_epochs = 1  
    model_file = config['MODEL_FILE']  
    loss_history, final_lr = train_model(num_epochs=num_epochs)

    # Assert that the final learning rate is greater than a specified minimum
    min_learning_rate = 0.00001
    assert final_lr >= min_learning_rate, f"Final learning rate {final_lr} is less than minimum threshold {min_learning_rate}"

    # Assert that the loss history is not empty, implying training iterations occurred
    assert loss_history, "Train loss history is empty."

    assert os.path.exists(model_file), f"File not found {model_file}"


if __name__ == "__main__":
    pytest.main()

