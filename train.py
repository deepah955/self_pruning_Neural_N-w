"""
train.py
--------
Training loop for the SelfPruningNet on CIFAR-10.

Usage
─────
    python train.py --lambdas 0.0001 0.001 0.01 --epochs 15

The script:
  1. Downloads CIFAR-10 (cached to ./data/).
  2. Trains one model per lambda value.
  3. Saves checkpoints to ./checkpoints/.
  4. Returns a summary dict used by main.py.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from model import SelfPruningNet


# ─────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128, data_dir: str = "./data"):
    """
    Download CIFAR-10 and return (train_loader, test_loader).

    Normalisation values are the standard CIFAR-10 channel means/stds.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────
# Core training function (one run for a single lambda)
# ─────────────────────────────────────────────────────────────────────

def train_one(
    lam: float,
    epochs: int,
    device: torch.device,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    checkpoint_dir: str = "./checkpoints",
    verbose: bool = True,
) -> dict:
    """
    Train SelfPruningNet with sparsity weight *lam* for *epochs* epochs.

    Returns
    -------
    dict with keys: lambda, test_accuracy, sparsity, epoch_logs
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = SelfPruningNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    epoch_logs = []          # list of {epoch, cls_loss, spar_loss, total_loss}

    for epoch in range(1, epochs + 1):
        model.train()
        running_cls   = 0.0
        running_spar  = 0.0
        running_total = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimiser.zero_grad()

            logits = model(images)

            # ── Total loss ────────────────────────────────────────────
            cls_loss  = criterion(logits, labels)
            spar_loss = model.sparsity_loss()
            loss      = cls_loss + lam * spar_loss

            loss.backward()
            optimiser.step()

            running_cls   += cls_loss.item()
            running_spar  += spar_loss.item()
            running_total += loss.item()
            n_batches += 1

        scheduler.step()

        avg_cls   = running_cls   / n_batches
        avg_spar  = running_spar  / n_batches
        avg_total = running_total / n_batches
        sparsity  = model.overall_sparsity()

        if verbose:
            print(
                f"  [Lambda={lam}] Epoch {epoch:>2}/{epochs} | "
                f"cls={avg_cls:.4f}  spar={avg_spar:.2f}  "
                f"total={avg_total:.4f}  sparse={sparsity:.1%}"
            )

        epoch_logs.append({
            "epoch":       epoch,
            "cls_loss":    round(avg_cls,   4),
            "spar_loss":   round(avg_spar,  2),
            "total_loss":  round(avg_total, 4),
            "sparsity":    round(sparsity,  4),
        })

    # ── Evaluate on test set ─────────────────────────────────────────
    acc = evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity()

    # ── Save checkpoint ──────────────────────────────────────────────
    ckpt_path = os.path.join(checkpoint_dir, f"model_lam{lam}.pt")
    torch.save({"state_dict": model.state_dict(), "lam": lam}, ckpt_path)

    # ── Also save the gate values for plotting ───────────────────────
    gates_path = os.path.join(checkpoint_dir, f"gates_lam{lam}.pt")
    torch.save(model.gate_values_flat(), gates_path)

    return {
        "lambda":        lam,
        "test_accuracy": round(acc * 100, 2),
        "sparsity":      round(final_sparsity * 100, 2),
        "epoch_logs":    epoch_logs,
        "ckpt_path":     ckpt_path,
        "gates_path":    gates_path,
    }


# ─────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return top-1 accuracy on *loader*."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def run_experiment(
    lambdas: list,
    epochs:  int,
    batch_size: int = 128,
    data_dir:   str = "./data",
    checkpoint_dir: str = "./checkpoints",
    verbose: bool = True,
) -> list:
    """
    Run a full experiment for each lambda in *lambdas*.
    Returns a list of result dicts (one per lambda).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"\nDevice: {device}")
        print(f"Lambdas: {lambdas}   Epochs: {epochs}\n")

    train_loader, test_loader = get_cifar10_loaders(batch_size, data_dir)

    results = []
    for lam in lambdas:
        if verbose:
            print(f"\n{'='*60}")
            print(f" Training with Lambda = {lam}")
            print(f"{'='*60}")
        result = train_one(
            lam, epochs, device,
            train_loader, test_loader,
            checkpoint_dir, verbose,
        )
        results.append(result)
        if verbose:
            print(
                f"\n  OK Lambda={lam} done | "
                f"Accuracy={result['test_accuracy']}%  "
                f"Sparsity={result['sparsity']}%"
            )

    # ── Persist summary JSON ──────────────────────────────────────────
    summary_path = os.path.join(checkpoint_dir, "results_summary.json")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # epoch_logs can be large; omit from summary for readability
    slim = [{k: v for k, v in r.items() if k != "epoch_logs"} for r in results]
    with open(summary_path, "w") as f:
        json.dump(slim, f, indent=2)
    if verbose:
        print(f"\nSummary saved -> {summary_path}")

    return results


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Self-Pruning Network on CIFAR-10")
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.0001, 0.001, 0.01],
                        help="Sparsity regularisation weights (space-separated)")
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--ckpt_dir",   type=str, default="./checkpoints")
    args = parser.parse_args()

    run_experiment(
        lambdas=args.lambdas,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        checkpoint_dir=args.ckpt_dir,
    )
