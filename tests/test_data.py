from __future__ import annotations

import pandas as pd
import torch

from mlproject.config import Config
from mlproject.data import TorchTabularDataset, make_dataloaders, train_val_test_split


def make_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [0, 1, 2, 3], "b": [1, 0, 1, 0], "label": [0, 1, 0, 1]})


def test_split_ratios() -> None:
    df = make_df()
    splits = train_val_test_split(df, ratios=(0.5, 0.25, 0.25), stratify=False, seed=0)
    total = sum(len(v) for v in splits.values())
    assert total == len(df)


def test_dataset_shapes() -> None:
    x = torch.randn(5, 3)
    y = torch.randint(0, 2, (5,))
    ds = TorchTabularDataset(x, y)
    assert len(ds) == 5
    xb, yb = ds[0]
    assert xb.shape[0] == 3 and yb.shape == ()
