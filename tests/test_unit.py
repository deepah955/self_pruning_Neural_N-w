"""
tests/test_unit.py
------------------
Unit tests for individual components.

Tests
─────
  1.  PrunableLinear output shape is correct.
  2.  PrunableLinear gates are strictly in (0, 1).
  3.  Gradients flow through both weight and gate_scores.
  4.  Gate initialised to 0.5  (sigmoid(0) == 0.5).
  5.  sparsity() returns a float in [0, 1].
  6.  SelfPruningNet output shape is (batch_size, 10).
  7.  sparsity_loss() is a positive scalar tensor.
  8.  overall_sparsity() is in [0, 1].
  9.  gate_values_flat() length equals total gate count.
  10. Pruning a gate to near-zero: manually set gate_score to −100,
      check that the effective gate < 1e-2.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
from prunable_layer import PrunableLinear
from model import SelfPruningNet


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def simple_layer():
    """A small PrunableLinear(4, 3) for quick tests."""
    return PrunableLinear(in_features=4, out_features=3)


@pytest.fixture
def net():
    """A fresh SelfPruningNet instance."""
    return SelfPruningNet()


# ─── 1. Output shape ─────────────────────────────────────────────────

def test_prunable_output_shape(simple_layer):
    x = torch.randn(5, 4)          # batch=5, in=4
    out = simple_layer(x)
    assert out.shape == (5, 3), f"Expected (5,3), got {out.shape}"


# ─── 2. Gates in (0, 1) ──────────────────────────────────────────────

def test_gates_in_0_1(simple_layer):
    gates = simple_layer.get_gates()
    assert (gates > 0).all(), "Some gate values are not > 0"
    assert (gates < 1).all(), "Some gate values are not < 1"


# ─── 3. Gradients flow ───────────────────────────────────────────────

def test_gradients_flow(simple_layer):
    x      = torch.randn(2, 4)
    target = torch.randn(2, 3)
    out    = simple_layer(x)
    loss   = (out - target).pow(2).sum()
    loss.backward()

    assert simple_layer.weight.grad is not None, "No gradient on weight"
    assert simple_layer.gate_scores.grad is not None, "No gradient on gate_scores"


# ─── 4. Default gate value ───────────────────────────────────────────

def test_default_gate_value(simple_layer):
    # gate_scores initialised to 0  →  sigmoid(0) = 0.5
    gates = simple_layer.get_gates()
    assert torch.allclose(gates, torch.full_like(gates, 0.5), atol=1e-5), \
        "Default gate values should all be 0.5 (sigmoid(0))"


# ─── 5. sparsity() in [0, 1] ─────────────────────────────────────────

def test_sparsity_range(simple_layer):
    s = simple_layer.sparsity()
    assert 0.0 <= s <= 1.0, f"sparsity() out of range: {s}"


# ─── 6. Network output shape ─────────────────────────────────────────

def test_net_output_shape(net):
    x = torch.randn(8, 3, 32, 32)    # batch=8, CIFAR-10 image size
    out = net(x)
    assert out.shape == (8, 10), f"Expected (8,10), got {out.shape}"


# ─── 7. sparsity_loss() is positive scalar ───────────────────────────

def test_sparsity_loss_positive(net):
    x    = torch.randn(4, 3, 32, 32)
    _    = net(x)                      # run forward to ensure graph exists
    loss = net.sparsity_loss()
    assert loss.ndim == 0,      "sparsity_loss() should be a scalar"
    assert loss.item() > 0,     "sparsity_loss() should be positive"


# ─── 8. overall_sparsity() in [0, 1] ────────────────────────────────

def test_overall_sparsity_range(net):
    s = net.overall_sparsity()
    assert 0.0 <= s <= 1.0, f"overall_sparsity() out of range: {s}"


# ─── 9. gate_values_flat() length ────────────────────────────────────

def test_gate_values_flat_length(net):
    total_gates = sum(
        l.gate_scores.numel() for l in net.prunable_layers()
    )
    flat = net.gate_values_flat()
    assert flat.shape[0] == total_gates, (
        f"Expected {total_gates} gate values, got {flat.shape[0]}"
    )


# ─── 10. Manual gate pruning ─────────────────────────────────────────

def test_manual_gate_forced_to_zero(simple_layer):
    """
    Set gate_scores to a large negative number  →  sigmoid → ≈ 0
    This simulates what happens when the L1 penalty drives scores down.
    """
    with torch.no_grad():
        simple_layer.gate_scores.fill_(-100.0)
    gates = simple_layer.get_gates()
    assert (gates < 1e-2).all(), \
        "All gates should be near 0 when scores are −100"


# ─── 11. Bias parameter present ──────────────────────────────────────

def test_bias_present():
    layer = PrunableLinear(8, 4, bias=True)
    assert layer.bias is not None

def test_bias_absent():
    layer = PrunableLinear(8, 4, bias=False)
    assert layer.bias is None


# ─── 12. No-bias layer still works ───────────────────────────────────

def test_no_bias_forward():
    layer = PrunableLinear(4, 2, bias=False)
    x   = torch.randn(3, 4)
    out = layer(x)
    assert out.shape == (3, 2)
