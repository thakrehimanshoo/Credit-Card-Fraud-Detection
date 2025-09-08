from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .data import load_dataframe, make_dataloaders, train_val_test_split
from .evaluate import evaluate_model
from .features import (
    TRANSFORMERS_FILE,
    fit_transformers,
    save_transformers,
    transform_splits,
)
from .train import train_model
from .visualize import plot_confusion_matrix, plot_training_curves
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--data_path", type=Path, required=True)
    prep.add_argument("--target_column", type=str, required=True)
    prep.add_argument("--categorical_cols", nargs="*", default=[])
    prep.add_argument("--numeric_cols", nargs="*", default=[])

    train_p = subparsers.add_parser("train")
    train_p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    train_p.add_argument("--lr", type=float, default=DEFAULT_CONFIG.lr)

    eval_p = subparsers.add_parser("evaluate")

    return parser.parse_args()


def cmd_prepare(args: argparse.Namespace) -> None:
    config = Config(
        data_path=args.data_path,
        target_column=args.target_column,
        categorical_cols=args.categorical_cols,
        numeric_cols=args.numeric_cols,
    )
    ensure_dir(config.output_dir)
    df = load_dataframe(config.data_path)
    df_splits = train_val_test_split(df, seed=config.seed)
    transformers = fit_transformers(df_splits["train"], config)
    save_transformers(transformers, config.output_dir / TRANSFORMERS_FILE)
    processed = transform_splits(df_splits, config, transformers)
    for split, d in processed.items():
        d.to_csv(config.output_dir / f"{split}.csv", index=False)


def cmd_train(args: argparse.Namespace) -> None:
    config = Config(epochs=args.epochs, lr=args.lr)
    df_train = pd.read_csv(config.output_dir / "train.csv")
    df_val = pd.read_csv(config.output_dir / "val.csv")
    df_splits = {"train": df_train, "val": df_val}
    feature_cols = [c for c in df_train.columns if c != config.target_column]
    loaders = make_dataloaders(config, df_splits, feature_cols)
    history = train_model(loaders, config)
    plot_training_curves(history, config.plots_dir / "training.png")


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = Config()
    df_train = pd.read_csv(config.output_dir / "train.csv")
    df_test = pd.read_csv(config.output_dir / "test.csv")
    df_splits = {"train": df_train, "test": df_test}
    feature_cols = [c for c in df_train.columns if c != config.target_column]
    loaders = make_dataloaders(config, df_splits, feature_cols)
    metrics = evaluate_model(loaders, config)
    print(metrics)
    plot_confusion_matrix(metrics["confusion_matrix"], config.plots_dir / "confusion_matrix.png")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
