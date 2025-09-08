from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlproject.config import Config
from mlproject.data import make_dataloaders, train_val_test_split
from mlproject.features import fit_transformers, transform_splits
from mlproject.train import train_model
from mlproject.evaluate import evaluate_model


def make_df() -> pd.DataFrame:
    return pd.DataFrame({"cat": ["a", "b", "a", "b", "a", "b"], "num": [1, 2, 3, 4, 5, 6], "label": [0, 1, 0, 1, 0, 1]})


def test_full_pipeline(tmp_path: Path) -> None:
    df = make_df()
    config = Config(target_column="label", categorical_cols=["cat"], numeric_cols=["num"], epochs=2, output_dir=tmp_path)
    splits = train_val_test_split(df, stratify=False)
    transformers = fit_transformers(splits["train"], config)
    processed = transform_splits(splits, config, transformers)
    feature_cols = [c for c in processed["train"].columns if c != config.target_column]
    loaders = make_dataloaders(config, processed, feature_cols)
    train_model(loaders, config)
    assert (config.checkpoints_dir / "model.pt").exists()
    metrics = evaluate_model({"train": loaders["train"], "test": loaders["test"]}, config)
    assert "accuracy" in metrics
