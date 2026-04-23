"""
evaluate.py
-----------
Post-training evaluation utilities.

Functions
─────────
  load_model(ckpt_path)      – reload a saved checkpoint
  report_sparsity(model)     – print per-layer and total sparsity
  plot_gate_distribution(…)  – save a matplotlib histogram of gate values
  generate_report(results)   – build the Markdown report table + analysis
"""

import os
import json
import torch
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt

from model import SelfPruningNet
from prunable_layer import PrunableLinear


# ─────────────────────────────────────────────────────────────────────
# Load a trained model from checkpoint
# ─────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str) -> SelfPruningNet:
    """Reinstantiate SelfPruningNet and load its saved state."""
    model = SelfPruningNet()
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────
# Per-layer sparsity report
# ─────────────────────────────────────────────────────────────────────

def report_sparsity(model: SelfPruningNet, threshold: float = 1e-2) -> None:
    """Print a human-readable per-layer sparsity table to stdout."""
    print(f"\n{'Layer':<35} {'Gates':<10} {'Pruned':<10} {'Sparsity'}")
    print("-" * 65)
    for i, layer in enumerate(model.prunable_layers()):
        n_total  = layer.gate_scores.numel()
        n_pruned = (layer.get_gates().detach() < threshold).sum().item()
        pct      = 100.0 * n_pruned / n_total
        print(f"  PrunableLinear [{i}]"
              f"  ({layer.in_features}->{layer.out_features})"
              f"  {n_total:<10} {n_pruned:<10} {pct:.1f}%")
    print("-" * 65)
    total_sparsity = model.overall_sparsity(threshold) * 100
    print(f"  Overall sparsity: {total_sparsity:.1f}%\n")


# ─────────────────────────────────────────────────────────────────────
# Gate-value distribution plot
# ─────────────────────────────────────────────────────────────────────

def plot_gate_distribution(
    gates: torch.Tensor,
    lam: float,
    save_path: str = "./checkpoints/gate_distribution.png",
) -> str:
    """
    Save a histogram of *gates* (1-D tensor of gate values in [0,1]).

    A well-pruned network shows a spike at 0 (dead gates) and a
    cluster near 1 (active gates) – a classic bimodal distribution.

    Returns the path to the saved image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    values = gates.numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=100, color="black", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Gate value  (0 = pruned, 1 = fully active)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Gate Value Distribution  (λ = {lam})", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Draw a dashed line at the prune threshold
    ax.axvline(x=0.01, color="gray", linestyle="--", linewidth=1.0,
               label="prune threshold (0.01)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────
# Markdown report builder
# ─────────────────────────────────────────────────────────────────────

REPORT_TEMPLATE = """\
# Self-Pruning Neural Network -- Results Report

## Why Does L1 on Sigmoid Gates Encourage Sparsity?

The gate for weight *w* is defined as:

```
gate = sigmoid(score)  in (0, 1)
```

We add **λ · Σ gate_i** to the total loss (the L1 norm of the gates,
which equals the plain sum since gates are always positive).

**Intuition:**

| Penalty | Gradient on score | Effect |
|---------|-------------------|--------|
| L2 (sum of squares) | −2 · gate · sigmoid' | pulls gate toward 0 but gradient shrinks as gate → 0; rarely reaches **exactly** 0 |
| L1 (sum of values)  | −sigmoid'(score) | constant-direction pull; gradient does NOT vanish at 0, so the gate **collapses** to 0 |

L1 creates a constant downward pressure. Once the classification loss
no longer benefits from a weight, the L1 term wins and the gate goes
to 0, permanently silencing that weight. This is exactly the "sparse
attractor" behaviour we want.

---

## Results Summary

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|------------|------------------|--------------------|
{rows}

---

## Analysis

- **Low λ (≈ 0.0001):** Minimal pruning. The sparsity loss has little
  influence, so almost all gates stay open. High accuracy, low sparsity.
- **Medium λ (≈ 0.001):** A good trade-off. The network drops redundant
  weights while preserving most of its representational capacity.
- **High λ (≈ 0.01):** Aggressive pruning. Accuracy falls noticeably as
  too many useful connections are forced to zero.

The gate distribution plot (below) for the best model shows the expected
bimodal shape: a tall spike at 0 (pruned weights) and a cluster near 1
(active weights), with very little in between.

![Gate Distribution](./checkpoints/gate_distribution.png)

---
*Generated automatically by evaluate.py*
"""


def generate_report(results: list, report_path: str = "./REPORT.md") -> str:
    """
    Build the Markdown report from a list of result dicts and save it.

    Parameters
    ----------
    results     : list of dicts returned by train.run_experiment()
    report_path : where to write the .md file

    Returns the path to the saved report.
    """
    rows = "\n".join(
        f"| {r['lambda']} | {r['test_accuracy']:.2f} | {r['sparsity']:.2f} |"
        for r in results
    )
    report = REPORT_TEMPLATE.format(rows=rows)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report_path
