"""
Main entry point for:
  Bone Cancer Detection and Classification Using
  Owl Search Algorithm With Deep Learning on X-Ray Images

Workflow
--------
  1. (Optional) Run OSA to find optimal hyper-parameters
  2. Train BoneCancerNet with the discovered HPs
  3. Evaluate on the test set
  4. Generate plots

Usage examples
--------------
  # Full pipeline with OSA optimisation
  python main.py --mode full

  # Train only (default HPs)
  python main.py --mode train

  # Evaluate only (requires a saved checkpoint)
  python main.py --mode eval

  # OSA optimisation only
  python main.py --mode osa

  # Use GPU
  python main.py --mode full --device cuda
"""

import argparse
import json
import os

import torch

import config
from train      import train
from evaluate   import evaluate
from owl_search import OwlSearchAlgorithm
from utils      import (
    plot_training_history,
    plot_osa_convergence,
    plot_confusion_matrix,
)


# ─── OSA fitness function ─────────────────────────────────────────────────────

def make_fitness_fn(device: str):
    """
    Returns a fitness function that trains a model for a few warm-up epochs
    and returns the validation accuracy as a fitness score.
    OSA maximises this score.
    """
    def fitness(hyperparams: dict) -> float:
        print(f"\n  [fitness] evaluating hp={hyperparams}")
        metrics = train(
            hyperparams = hyperparams,
            tag         = "osa_probe",
            num_epochs  = 5,        # short warm-up for speed
            device      = device,
        )
        score = metrics.get("accuracy", 0.0)
        print(f"  [fitness] val_acc={score:.4f}")
        return score

    return fitness


# ─── Pipeline stages ──────────────────────────────────────────────────────────

def run_osa(device: str) -> dict:
    print("\n" + "="*60)
    print("STAGE 1 — Owl Search Algorithm Hyperparameter Optimisation")
    print("="*60)

    osa = OwlSearchAlgorithm(
        fitness_fn = make_fitness_fn(device),
        verbose    = True,
    )
    best_hp = osa.run()

    # Save results
    hp_path = os.path.join(config.RESULTS_DIR, "osa_best_hp.json")
    with open(hp_path, "w") as f:
        json.dump(best_hp, f, indent=2)
    print(f"\n[OSA] Best HPs saved → {hp_path}")

    plot_osa_convergence(osa.convergence_curve())
    return best_hp


def run_train(hyperparams: dict = None, device: str = "cpu"):
    print("\n" + "="*60)
    print("STAGE 2 — Model Training")
    print("="*60)

    tag = "osa" if hyperparams else "default"
    metrics = train(
        hyperparams = hyperparams,
        tag         = tag,
        device      = device,
    )

    hist_path = os.path.join(config.RESULTS_DIR, f"history_{tag}.json")
    if os.path.exists(hist_path):
        plot_training_history(hist_path)

    return metrics


def run_eval(device: str = "cpu"):
    print("\n" + "="*60)
    print("STAGE 3 — Test Set Evaluation")
    print("="*60)
    metrics = evaluate(device=device)
    plot_confusion_matrix(metrics["confusion_matrix"])
    return metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bone Cancer Detection — OSA + Deep Learning"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "eval", "osa"],
        default="full",
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cpu | cuda | mps",
    )
    parser.add_argument(
        "--hp_file",
        type=str,
        default=None,
        help="Path to JSON with pre-computed OSA hyper-parameters (skips OSA stage)",
    )
    args = parser.parse_args()

    print(f"\nDevice : {args.device}")
    print(f"Mode   : {args.mode}")

    hyperparams = None

    if args.hp_file and os.path.exists(args.hp_file):
        with open(args.hp_file) as f:
            hyperparams = json.load(f)
        print(f"Loaded HPs from {args.hp_file}: {hyperparams}")

    if args.mode == "osa":
        run_osa(args.device)

    elif args.mode == "train":
        run_train(hyperparams=hyperparams, device=args.device)

    elif args.mode == "eval":
        run_eval(device=args.device)

    elif args.mode == "full":
        if hyperparams is None:
            hyperparams = run_osa(args.device)
        run_train(hyperparams=hyperparams, device=args.device)
        run_eval(device=args.device)

    print("\n[DONE] All stages complete.")


if __name__ == "__main__":
    main()
