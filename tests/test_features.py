from __future__ import annotations

import pandas as pd
from pathlib import Path

from mlproject.config import Config
from mlproject.features import fit_transformers, save_transformers, load_transformers, transform_splits
from mlproject.data import train_val_test_split


def make_df() -> pd.DataFrame:
    return pd.DataFrame({"cat": ["a", "b", "a", "b"], "num": [1, 2, 3, 4], "label": [0, 1, 0, 1]})


def test_onehot_roundtrip(tmp_path: Path) -> None:
    df = make_df()
    config = Config(target_column="label", categorical_cols=["cat"], numeric_cols=["num"])
    splits = train_val_test_split(df, stratify=False)
    transformers = fit_transformers(splits["train"], config)
    path = tmp_path / "t.joblib"
    save_transformers(transformers, path)
    loaded = load_transformers(path)
    processed = transform_splits(splits, config, loaded)
    train_df = processed["train"]
    assert train_df.shape[1] == 3  # 2 features + label
