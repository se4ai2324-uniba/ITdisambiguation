# Tests

This folder contains all tests needed for code, model and data. The folder is structured in the following way:

- `behavioral_testing` Check the correct behavior of the model, in particular:
    1. Directional testing
    2. Invariance testing
    3. Minimum functionality testing
- `dataset_testing` Check the correctness of the dataset.
- `model_testing` Check the training process, we use different tests to check:
    1. Overfit over one batch
    2. Training on different devices
    3. Loss decrease
    4. Change in the learning rate
- `preprocessing_testing` Check the correct behavior of the preprocessing code.

## Usage
Tests should only be ran using the command
```
python -m pytest tests/
```
while being in the project's root folder.
