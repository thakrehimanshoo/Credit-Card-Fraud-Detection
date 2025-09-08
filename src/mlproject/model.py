from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .config import Config


class MLP(nn.Module):
    def __init__(self, input_dim: int, config: Config):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in config.hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, config.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def build_model(input_dim: int, config: Config) -> nn.Module:
    return MLP(input_dim, config)


def get_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def get_scheduler(optimizer: torch.optim.Optimizer, config: Config) -> torch.optim.lr_scheduler._LRScheduler | None:
    return None
