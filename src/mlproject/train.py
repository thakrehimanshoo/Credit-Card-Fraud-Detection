from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from sklearn.metrics import precision_score, recall_score
from torch import nn

from .config import Config
from .utils import ensure_dir, save_json, set_seed


class EarlyStopper:
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_loop(model: nn.Module, loaders: Dict[str, torch.utils.data.DataLoader], config: Config) -> Dict[str, List[float]]:
    set_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss() if config.num_classes == 1 or config.num_classes == 2 else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_precision": [], "val_precision": [], "train_recall": [], "val_recall": []}
    stopper = EarlyStopper()
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        all_preds, all_targets = [], []
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            if config.num_classes > 1:
                loss = criterion(logits, yb.long())
            else:
                loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
        train_loss = float(sum(train_losses) / len(train_losses))
        train_prec = precision_score(all_targets, all_preds, zero_division=0)
        train_rec = recall_score(all_targets, all_preds, zero_division=0)

        model.eval()
        val_losses = []
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(-1)
                if config.num_classes > 1:
                    loss = criterion(logits, yb.long())
                else:
                    loss = criterion(logits, yb.float())
                val_losses.append(loss.item())
                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).int()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())
        val_loss = float(sum(val_losses) / len(val_losses)) if val_losses else 0.0
        val_prec = precision_score(val_targets, val_preds, zero_division=0)
        val_rec = recall_score(val_targets, val_preds, zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_precision"].append(train_prec)
        history["val_precision"].append(val_prec)
        history["train_recall"].append(train_rec)
        history["val_recall"].append(val_rec)

        if val_loss < stopper.best_loss:
            best_state = model.state_dict()
        if stopper.step(val_loss):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_model(loaders: Dict[str, torch.utils.data.DataLoader], config: Config) -> Dict[str, List[float]]:
    input_dim = loaders["train"].dataset.x.shape[1]
    from .model import build_model

    model = build_model(input_dim, config)
    history = train_loop(model, loaders, config)
    ensure_dir(config.checkpoints_dir)
    torch.save(model.state_dict(), config.checkpoints_dir / "model.pt")
    save_json(history, config.output_dir / "metrics.json")
    return history
