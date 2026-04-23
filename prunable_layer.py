import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def sparsity(self, threshold: float = 1e-2) -> float:
        gates = self.get_gates().detach()
        return (gates < threshold).float().mean().item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = self.get_gates()
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)
