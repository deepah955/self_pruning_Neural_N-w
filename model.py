import torch
import torch.nn as nn
from prunable_layer import PrunableLinear

class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
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
            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)

    def prunable_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers():
            total = total + layer.get_gates().sum()
        return total

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        pruned = sum(
            (l.get_gates().detach() < threshold).sum().item()
            for l in self.prunable_layers()
        )
        total = sum(l.gate_scores.numel() for l in self.prunable_layers())
        return pruned / total if total > 0 else 0.0

    def gate_values_flat(self) -> torch.Tensor:
        return torch.cat([
            l.get_gates().detach().flatten()
            for l in self.prunable_layers()
        ])
