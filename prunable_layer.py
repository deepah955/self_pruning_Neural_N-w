"""
prunable_layer.py
-----------------
A custom PyTorch linear layer that learns to prune itself.

Each weight has a paired "gate_score". During the forward pass:
  gates = sigmoid(gate_scores)   -> values in (0, 1)
  pruned_weights = weight * gates
  output = pruned_weights @ input.T + bias

By penalising the sum of all gate values (L1 norm on the gates),
the optimiser is pushed to drive gates toward 0, effectively
removing the corresponding weight from the network.
"""

import torch
import torch.nn as nn


class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a learnable gate
    to every weight element.

    Parameters
    ----------
    in_features  : int  – number of input features
    out_features : int  – number of output features
    bias         : bool – whether to include a bias term (default True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # ── Standard weight & bias ──────────────────────────────────────
        # Initialised with kaiming_uniform (same as nn.Linear default).
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # ── Gate scores ─────────────────────────────────────────────────
        # Same shape as weight. Initialised to 0 so sigmoid(0)=0.5,
        # meaning all gates start at 50% activation – neutral ground.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_gates(self) -> torch.Tensor:
        """Return the current gate values in [0, 1]."""
        return torch.sigmoid(self.gate_scores)

    def sparsity(self, threshold: float = 1e-2) -> float:
        """
        Fraction of gates whose value is below *threshold*.
        A gate below the threshold is considered 'pruned'.
        """
        gates = self.get_gates().detach()
        return (gates < threshold).float().mean().item()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 – convert raw scores to gates that live in (0, 1)
        gates = self.get_gates()                    # shape: (out, in)

        # Step 2 – mask the weights element-wise with the gates
        pruned_weights = self.weight * gates        # same shape as weight

        # Step 3 – standard linear operation with the gated weights
        return torch.nn.functional.linear(x, pruned_weights, self.bias)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
