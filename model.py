"""
model.py
--------
A simple feed-forward neural network for CIFAR-10 image classification,
built entirely from PrunableLinear layers.

Architecture
────────────
Input: 32x32x3 image  →  flattened to 3072
Hidden 1: 3072  → 512   (PrunableLinear + ReLU + BN)
Hidden 2:  512  → 256   (PrunableLinear + ReLU + BN)
Hidden 3:  256  → 128   (PrunableLinear + ReLU + BN)
Output:    128  →  10   (PrunableLinear, raw logits)

Keeping it feed-forward (no convolutions) intentionally:
  1. Makes pruning behaviour clearly visible on the weight matrices.
  2. Keeps the code simple and easy to explain.
"""

import torch
import torch.nn as nn
from prunable_layer import PrunableLinear


class SelfPruningNet(nn.Module):
    """
    Feed-forward classifier for CIFAR-10 with self-pruning gates.

    Every linear projection uses PrunableLinear, so every weight
    can be gated to zero by the sparsity regulariser.
    """

    def __init__(self):
        super().__init__()

        # ── Layers ──────────────────────────────────────────────────────
        # We use BatchNorm1d after each hidden layer to keep training
        # stable, but the pruning happens inside PrunableLinear itself.
        self.net = nn.Sequential(
            PrunableLinear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            PrunableLinear(128, 10),   # 10 CIFAR-10 classes
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten from (batch, C, H, W) → (batch, 3072)
        x = x.view(x.size(0), -1)
        return self.net(x)

    # ------------------------------------------------------------------
    # Helpers to iterate over PrunableLinear layers
    # ------------------------------------------------------------------
    def prunable_layers(self):
        """Yield every PrunableLinear sub-module."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across every PrunableLinear layer.

        Why L1?
        • L1 pulls values toward 0 (unlike L2 which pulls toward 0 but
          never enforces exact zeros).
        • Since gates = sigmoid(scores) are always positive, the L1 norm
          equals the plain sum – no absolute value needed.
        • This creates a constant gradient pointing toward zero for all
          gates, encouraging them to collapse completely.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers():
            total = total + layer.get_gates().sum()
        return total

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Fraction of all gates (across every layer) below *threshold*.
        """
        layers = self.prunable_layers()
        pruned = sum(
            (layer.get_gates().detach() < threshold).sum().item()
            for layer in layers
        )
        total = sum(
            layer.gate_scores.numel() for layer in layers
        )
        return pruned / total if total > 0 else 0.0

    def gate_values_flat(self) -> torch.Tensor:
        """All gate values concatenated into a 1-D tensor (detached)."""
        parts = [layer.get_gates().detach().flatten()
                 for layer in self.prunable_layers()]
        return torch.cat(parts)
