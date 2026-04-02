"""
Evaluation script — runs the best saved model on the test set
and prints a full classification report with metrics.
"""

import os
import json
import argparse

import torch

import config
from model   import load_checkpoint
from dataset import get_dataloader
from train   import validate
from utils   import (
    print_classification_report,
    plot_confusion_matrix,
    compute_metrics,
)


def evaluate(checkpoint_path: str = None, device: str = "cpu"):
    if checkpoint_path is None:
        # Auto-select the best checkpoint
        ckpts = [
            f for f in os.listdir(config.CHECKPOINT_DIR)
            if f.startswith("best_") and f.endswith(".pth")
        ]
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoints found in {config.CHECKPOINT_DIR}"
            )
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, sorted(ckpts)[-1])

    print(f"\n[EVAL] Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    test_loader, _ = get_dataloader("test", shuffle=False)

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    metrics   = validate(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy   : {metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score   : {metrics['f1']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    if metrics.get("auc_roc"):
        print(f"  AUC-ROC    : {metrics['auc_roc']:.4f}")
    print(f"{'='*60}\n")

    # Classification report
    all_preds, all_labels = _collect_predictions(model, test_loader, device)
    print_classification_report(all_labels, all_preds)

    # Confusion matrix plot
    plot_confusion_matrix(metrics["confusion_matrix"])

    # Save metrics
    out_path = os.path.join(config.RESULTS_DIR, "test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[EVAL] Metrics saved → {out_path}")

    return metrics


@torch.no_grad()
def _collect_predictions(model, loader, device):
    preds_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds  = logits.argmax(dim=1)
        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(labels.tolist())
    return preds_all, labels_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BoneCancerNet")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint (auto-selects if omitted)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    evaluate(checkpoint_path=args.checkpoint, device=args.device)
