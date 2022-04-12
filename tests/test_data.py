from pandatorch import __version__, data
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch

df = pd.read_csv("tests/IRIS.csv")

torch_df = data.DataFrame(df.drop(columns=["petal_length", "species"]), df["species"])


def update_rst_markdown():
    os.system("pandoc -s README.md -o README.rst")


def test_cwd_has_markdown():
    cwd = os.getcwd()
    assert os.path.exists(os.path.join(cwd, "README.rst")) == True
    assert os.path.exists(os.path.join(cwd, "README.md")) == True


def test_df_has_all_columns():
    feature_cols = torch_df.get_number_of_columns(torch_df.features)
    target_cols = torch_df.get_number_of_columns(torch_df.target)
    df_cols = torch_df.get_number_of_columns(torch_df.df)
    assert target_cols == 1
    assert feature_cols + target_cols == df_cols


def test_dataloader_compatibility():
    dataloader = DataLoader(dataset=torch_df)
    batch, label = next(iter(dataloader))
    assert batch is not None
    assert label is not None


def test_targets_torch_integers():
    targets = torch.tensor(torch_df.target)
    assert targets.dtype is torch.int64
