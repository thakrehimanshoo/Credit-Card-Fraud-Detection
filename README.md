# Credit Card Fraud Detection ML Project

This repository converts a notebook-based experiment into a reusable Python package for credit card fraud detection using PyTorch.

## Architecture
```
project-root/
  README.md
  pyproject.toml
  requirements.txt
  src/mlproject/...  # package modules
  scripts/            # helper shell scripts
  tests/              # unit tests with synthetic data
```

## Features
- Data loading and splitting
- One-hot encoding for categorical features
- Feed-forward neural network with PyTorch
- Training with checkpointing and metric logging
- Evaluation with precision and recall metrics
- Plotting utilities for training curves and confusion matrix
- Command line interface for end-to-end runs

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage
```bash
python -m mlproject.cli prepare --data_path data/train.csv --target_column label
python -m mlproject.cli train --epochs 20 --lr 3e-4
python -m mlproject.cli evaluate
```

## Configuration
The system is driven by a `Config` dataclass defined in `mlproject/config.py`. Important fields include:
- `data_path`: Path to the CSV dataset
- `target_column`: Name of the target label column
- `categorical_cols`, `numeric_cols`
- `batch_size`, `epochs`, `lr`, `weight_decay`, `seed`
- `hidden_sizes`, `dropout`
- `output_dir`, `checkpoints_dir`, `plots_dir`

# TODO
- Advanced preprocessing (scaling, missing value imputation)
- Hyperparameter tuning
- Additional evaluation metrics

## Testing
Run the unit tests with:
```bash
pytest
```

## Troubleshooting
- Ensure PyTorch is installed for your system (CPU vs CUDA)
- Set `--device cpu` on the CLI if you lack a GPU
```
