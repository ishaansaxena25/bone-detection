"""
Training loop for the BoneCancerNet model.
Supports both standard training and OSA-optimised hyper-parameter sets.
"""

import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from model   import BoneCancerNet, build_model
from dataset import get_dataloader
from utils   import AverageMeter, compute_metrics, save_checkpoint, EarlyStopping


# ─── Single epoch ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    scheduler=None, scheduler_per_batch=False):
    model.train()
    loss_meter = AverageMeter()
    correct, total = 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # OneCycleLR must step every batch, not every epoch
        if scheduler is not None and scheduler_per_batch:
            scheduler.step()

        loss_meter.update(loss.item(), images.size(0))
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"loss={loss_meter.avg:.4f} "
                f"acc={100*correct/total:.2f}%"
            )

    return {"loss": loss_meter.avg, "acc": correct / total}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        loss_meter.update(loss.item(), images.size(0))

        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = loss_meter.avg
    return metrics


# ─── Full training run ────────────────────────────────────────────────────────

def train(
    hyperparams: dict = None,
    tag:         str  = "default",
    num_epochs:  int  = config.NUM_EPOCHS,
    device:      str  = "cpu",
) -> dict:
    """
    Run a complete training+validation loop.

    Parameters
    ----------
    hyperparams : dict returned by OSA (or None → use config defaults)
    tag         : label for checkpoint naming
    num_epochs  : override config.NUM_EPOCHS
    device      : 'cpu' | 'cuda' | 'mps'

    Returns
    -------
    best_val_metrics : dict
    """
    hp = hyperparams or {
        "lr":           config.DEFAULT_LR,
        "backbone_lr":  config.DEFAULT_BACKBONE_LR,
        "dropout":      config.DEFAULT_DROPOUT,
        "weight_decay": config.DEFAULT_WEIGHT_DECAY,
        "momentum":     config.DEFAULT_MOMENTUM,
        "fc_units":     config.DEFAULT_FC_UNITS,
    }
    print(f"\n{'='*60}")
    print(f"[TRAIN] tag={tag}")
    print(f"[TRAIN] hyper-params: {hp}")
    print(f"{'='*60}")

    # Data
    train_loader, train_ds = get_dataloader("train")
    val_loader,   _        = get_dataloader("val")

    # Class-weighted loss + label smoothing to handle imbalance & overconfidence
    class_weights = train_ds.class_weights().to(device)
    criterion     = nn.CrossEntropyLoss(
        weight           = class_weights,
        label_smoothing  = config.LABEL_SMOOTHING,
    )

    # Model
    model = build_model(
        fc_units = hp["fc_units"],
        dropout  = hp["dropout"],
        device   = device,
    )

    # Differential learning rates:
    #   backbone (pretrained) → 10× lower LR to avoid destroying ImageNet features
    #   classification head  → full LR for fast convergence on the new task
    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.classifier.parameters())
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": hp.get("backbone_lr", hp["lr"] * 0.1)},
            {"params": head_params,     "lr": hp["lr"]},
        ],
        weight_decay = hp["weight_decay"],
    )

    # OneCycleLR: ramps up then anneals — best single-cycle policy for fine-tuning
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr        = [hp.get("backbone_lr", hp["lr"] * 0.1), hp["lr"]],
        steps_per_epoch = len(train_loader),
        epochs        = num_epochs,
        pct_start     = 0.3,        # 30% of training spent warming up
        anneal_strategy = "cos",
        div_factor    = 10.0,
        final_div_factor = 1e3,
    )

    early_stop      = EarlyStopping(patience=config.EARLY_STOP_PATIENCE)
    best_val_acc    = 0.0
    best_val_metrics = {}
    history          = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scheduler=scheduler, scheduler_per_batch=True   # OneCycleLR steps per batch
        )
        val_metrics = validate(model, val_loader, criterion, device)
        # scheduler already stepped inside the batch loop — do NOT call here

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']*100:.2f}% | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']*100:.2f}% "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"{elapsed:.1f}s"
        )

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc     = val_metrics["accuracy"]
            best_val_metrics = val_metrics
            save_checkpoint(
                model       = model,
                epoch       = epoch,
                val_metrics = val_metrics,
                hp          = hp,
                path        = os.path.join(
                    config.CHECKPOINT_DIR, f"best_{tag}.pth"
                ),
            )

        if early_stop(val_metrics["loss"]):
            print(f"[EarlyStopping] Triggered at epoch {epoch}.")
            break

    # Save training history
    hist_path = os.path.join(config.RESULTS_DIR, f"history_{tag}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[TRAIN] History saved → {hist_path}")

    return best_val_metrics
