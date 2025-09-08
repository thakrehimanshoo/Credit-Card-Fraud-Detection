from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass
class Config:
    """Configuration for the ML pipeline."""

    # Data
    data_path: Path = Path("data.csv")
    target_column: str = "label"
    categorical_cols: Sequence[str] = field(default_factory=list)
    numeric_cols: Sequence[str] = field(default_factory=list)

    # Training params
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cpu"

    # Model params
    hidden_sizes: Sequence[int] = field(default_factory=lambda: [32, 16])
    dropout: float = 0.1
    num_classes: int = 1

    # Paths
    output_dir: Path = Path("outputs")

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"


DEFAULT_CONFIG = Config()
