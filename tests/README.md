# Tests

This folder contains all tests needed for code, model and data. The folder is structured in the following way:

- `api_testing` Check the correct behavior of the APIs:
    1. Predict context api test
    2. Predict image api test

- `behavioral_testing` Check the correct behavior of the model, in particular:
    1. Directional testing
    2. Invariance testing
    3. Minimum functionality testing
- `dataset_testing` Check the correctness of the dataset and data integrity:
    1. Testing the presence of 10 images in each sample
    2. Testing the correctness of the sample correct image
    3. Testing with great expectations the correctness of type and format of the elements in the tabular dataset
    4. Testing that all the images listed in the dataset are phisically present in the correct folder
    5. Testing for each sample if it contains the right number of contexts, and also if the target is present in the context
    6. Testing that the format of the correct image, in the target file, is the right one
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
