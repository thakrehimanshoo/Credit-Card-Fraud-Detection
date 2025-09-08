from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder

from .config import Config


TRANSFORMERS_FILE = "transformers.joblib"


def fit_transformers(train_df: DataFrame, config: Config) -> Dict[str, Any]:
    """Fit preprocessing transformers on training data."""
    transformers: Dict[str, Any] = {}
    if config.categorical_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(train_df[config.categorical_cols])
        transformers["ohe"] = ohe
    # TODO: add StandardScaler for numeric
    return transformers


def transform_df(df: DataFrame, config: Config, transformers: Dict[str, Any]) -> DataFrame:
    parts = []
    if config.categorical_cols:
        ohe: OneHotEncoder = transformers["ohe"]
        cat_arr = ohe.transform(df[config.categorical_cols])
        parts.append(cat_arr)
    if config.numeric_cols:
        parts.append(df[config.numeric_cols].values)
    if parts:
        data = np.hstack(parts)
    else:
        data = np.empty((len(df), 0))
    feature_cols = [f"f{i}" for i in range(data.shape[1])]
    feat_df = pd.DataFrame(data, columns=feature_cols, index=df.index)
    feat_df[config.target_column] = df[config.target_column].values
    return feat_df


def transform_splits(df_splits: Dict[str, DataFrame], config: Config, transformers: Dict[str, Any]) -> Dict[str, DataFrame]:
    return {k: transform_df(v, config, transformers) for k, v in df_splits.items()}


def save_transformers(transformers: Dict[str, Any], path: Path) -> None:
    joblib.dump(transformers, path)


def load_transformers(path: Path) -> Dict[str, Any]:
    return joblib.load(path)
