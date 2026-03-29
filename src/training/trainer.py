import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CKPT_KEYS = ["best_macro_f1", "best_f1_false", "best_pr_auc", "best_calib"]


class CheckpointManager:
    def __init__(self, run_id: str, base_dir: str = "checkpoints"):
        self.base   = base_dir
        self.run_id = run_id
        self.best   = {k: -1e9 for k in CKPT_KEYS}
        self.best["best_calib"] = 1e9   # lower is better

    def update(self, model, metrics: dict, epoch: int):
        mapping = {
            "best_macro_f1": ("macro_f1",     True),
            "best_f1_false": ("f1_false",     True),
            "best_pr_auc":   ("pr_auc_false", True),
            "best_calib":    ("brier",        False),  # lower is better
        }
        for key, (metric, higher_is_better) in mapping.items():
            val = metrics.get(metric)
            if val is None:
                continue
            improved = (val > self.best[key]) if higher_is_better else (val < self.best[key])
            if improved:
                self.best[key] = val
                path = os.path.join(self.base, key, self.run_id)
                os.makedirs(path, exist_ok=True)
                torch.save(model.state_dict(), f"{path}/epoch{epoch}.pt")
                print(f"  ✓ Checkpoint [{key}] epoch {epoch}: {metric}={val:.4f}")


def train_epoch(model, loader: DataLoader, optimizer, device,
                grad_accum_steps: int = 1, scheduler=None) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out  = model(**batch)
        loss = out["loss"] / grad_accum_steps
        loss.backward()
        total_loss += out["loss"].item()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader)


def eval_epoch(model, loader: DataLoader, device) -> tuple:
    """Returns (y_true, y_pred, y_prob) as numpy arrays."""
    import numpy as np

    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch.pop("labels")
            out    = model(**batch)
            logits = out["logits"]
            probs  = torch.softmax(logits, dim=-1)[:, 1]
            preds  = logits.argmax(dim=-1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
            all_prob.extend(probs.cpu().numpy())

    return (np.array(all_true), np.array(all_pred), np.array(all_prob))


def run_training(model, train_loader: DataLoader, val_loader: DataLoader,
                 optimizer, ckpt_manager: CheckpointManager,
                 device, n_epochs: int = 20, patience: int = 5,
                 metric_fn=None, scheduler=None):
    """Generic training loop with early stopping."""
    from src.evaluation.metrics import compute_all

    best_val = -1.0
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler=scheduler)
        y_true, y_pred, y_prob = eval_epoch(model, val_loader, device)
        metrics = compute_all(y_true, y_pred, y_prob)

        print(f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
              f"macro_f1={metrics['macro_f1']:.4f} | f1_false={metrics['f1_false']:.4f}")

        ckpt_manager.update(model, metrics, epoch)

        val_score = metrics["macro_f1"]
        if val_score > best_val:
            best_val   = val_score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    return best_val
