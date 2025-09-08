from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .config import Config
from .utils import ensure_dir


def load_dataframe(path: str | Path) -> DataFrame:
    """Load a CSV into a pandas DataFrame."""
    return pd.read_csv(path)


def train_val_test_split(
    df: DataFrame, ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15), stratify: bool = True, seed: int = 42
) -> Dict[str, DataFrame]:
    """Split DataFrame into train/val/test."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    train_ratio, val_ratio, test_ratio = ratios
    if stratify:
        train_df, temp_df = train_test_split(df, test_size=1 - train_ratio, stratify=df[df.columns[-1]], random_state=seed)
        val_size = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(temp_df, test_size=val_size, stratify=temp_df[temp_df.columns[-1]], random_state=seed)
    else:
        train_df, temp_df = train_test_split(df, test_size=1 - train_ratio, random_state=seed)
        val_size = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(temp_df, test_size=val_size, random_state=seed)
    return {"train": train_df, "val": val_df, "test": test_df}


class TorchTabularDataset(Dataset):
    """Simple dataset for tabular data."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x.float()
        self.y = y.float() if y.ndim == 1 else y

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.x[idx], self.y[idx]


def make_dataloaders(config: Config, df_splits: Dict[str, DataFrame], feature_cols: Sequence[str]) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split, df in df_splits.items():
        x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        y = torch.tensor(df[config.target_column].values)
        dataset = TorchTabularDataset(x, y)
        shuffle = split == "train"
        loaders[split] = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)
    return loaders


try:
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover
    raise
