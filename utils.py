"""
Utility functions: metrics, checkpointing, plotting, early stopping.
"""

import os
import json
from typing import List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import config


# ─── Running average ──────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[List[float]] = None,
) -> dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : ground-truth labels
    y_pred : predicted labels
    y_prob : softmax probabilities  (N × num_classes)
    """
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred,
                              average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred,
                                     average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred,
                                  average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        probs_np = np.array(y_prob)
        try:
            auc = roc_auc_score(
                y_true, probs_np,
                multi_class="ovr", average="weighted"
            )
            metrics["auc_roc"] = float(auc)
        except ValueError:
            metrics["auc_roc"] = None

    return metrics


def print_classification_report(y_true, y_pred):
    print(classification_report(
        y_true, y_pred,
        target_names=config.CLASS_NAMES,
        zero_division=0,
    ))


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(model, epoch, val_metrics, hp, path):
    state = {
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_metrics": val_metrics,
        "fc_units":    hp.get("fc_units",     config.DEFAULT_FC_UNITS),
        "dropout":     hp.get("dropout",      config.DEFAULT_DROPOUT),
        "hyperparams": hp,
    }
    torch.save(state, path)
    print(f"[CKPT] Saved → {path}  (val_acc={val_metrics['accuracy']*100:.2f}%)")


def load_checkpoint_state(path: str, device: str = "cpu") -> dict:
    return torch.load(path, map_location=device)


# ─── Early stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ─── Plot helpers (optional, require matplotlib) ──────────────────────────────

def plot_training_history(history_path: str, save_dir: str = config.RESULTS_DIR):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not installed — skipping plots.")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs       = [h["epoch"]                 for h in history]
    train_loss   = [h["train"]["loss"]          for h in history]
    val_loss     = [h["val"]["loss"]            for h in history]
    train_acc    = [h["train"]["acc"]           for h in history]
    val_acc      = [h["val"]["accuracy"]        for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss,   label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Acc")
    axes[1].plot(epochs, val_acc,   label="Val Acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    out = os.path.join(save_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[PLOT] Saved → {out}")


def plot_confusion_matrix(cm: list, save_dir: str = config.RESULTS_DIR):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[PLOT] matplotlib/seaborn not installed — skipping.")
        return

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ax=ax,
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")

    out = os.path.join(save_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[PLOT] Saved → {out}")


def plot_osa_convergence(curve: list, save_dir: str = config.RESULTS_DIR):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(curve) + 1), curve, marker="o", linewidth=2)
    plt.title("OSA Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (Val Accuracy)")
    plt.grid(True)

    out = os.path.join(save_dir, "osa_convergence.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[PLOT] Saved → {out}")
