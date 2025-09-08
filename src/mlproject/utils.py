from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


logger = logging.getLogger("mlproject")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)
