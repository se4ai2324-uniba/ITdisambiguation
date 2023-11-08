import pytest
import torch
import os
import great_expectations as ge
import pandas as pd 
from src.models.conf import config


train_images_names = config["TRAIN_DATA"] 
column_names = ["target", "contexts", "img1", "img2", "img3", "img4", "img5", "img6", "img7", "img8", "img9", "img10"]
df = pd.read_csv(train_images_names, sep='\t', header=None, names=column_names)
dataset = ge.dataset.PandasDataset(df)


dataset.expect_column_values_to_be_of_type(column="target", type_="str")
dataset.expect_column_values_to_match_regex(column="img1", regex=r"\.jpg$")
dataset.expect_column_values_to_match_regex(column="img2", regex=r"\.jpg$")
dataset.expect_column_values_to_match_regex(column="img3", regex=r"\.jpg$")
dataset.expect_column_values_to_match_regex(column="img3", regex=r"\.jpg$")
dataset.expect_column_values_to_match_regex(column="img4", regex=r"\.jpg$")
dataset.expect_column_values_to_match_regex(column="img5", regex=r"\.jpg$")
dataset.expect_column_values_to_match_regex(column="img6", regex=r"\.jpg$")
dataset.expect_column_values_to_match_regex(column="img7", regex=r"\.jpg$")
dataset.expect_column_values_to_match_regex(column="img8", regex=r"\.jpg$")
dataset.expect_column_values_to_match_regex(column="img9", regex=r"\.jpg$")
dataset.expect_column_values_to_match_regex(column="img10", regex=r"\.jpg$")

exs = dataset.get_expectation_suite(discard_failed_expectations=False)
print(dataset.validate(
    exs,
    only_return_failures = True
))