from __future__ import annotations

import torch

from mlproject.config import Config
from mlproject.model import build_model


def test_forward_pass() -> None:
    config = Config()
    model = build_model(4, config)
    x = torch.randn(2, 4)
    out = model(x)
    assert out.shape == (2, config.num_classes)
    y = torch.randint(0, 2, (2,)).float()
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(out.squeeze(), y)
    assert loss.item() > 0
