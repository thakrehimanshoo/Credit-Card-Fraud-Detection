from __future__ import annotations


import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from .utils import save_json
from .model import build_model


def evaluate_model(loaders: Dict[str, torch.utils.data.DataLoader], config: Config) -> Dict[str, float]:
    device = torch.device(config.device)
    input_dim = loaders["train"].dataset.x.shape[1]
    model = build_model(input_dim, config)
    model.load_state_dict(torch.load(config.checkpoints_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loaders["test"]:
            xb = xb.to(device)
            logits = model(xb).squeeze(-1)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).int()
            preds.extend(pred.cpu().numpy().tolist())
            targets.extend(yb.numpy().tolist())
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    cm = confusion_matrix(targets, preds).tolist()
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "confusion_matrix": cm}
    save_json(metrics, config.output_dir / "eval_metrics.json")
    return metrics
